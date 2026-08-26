"""Pure Recording-owned admission, persistence, cancellation, and label ports."""

from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

import numpy as np

from base.acquisition_recording_defaults import normalize_record_only_detail
from base.audio_sample_rate import resolve_duplex_sample_rate, resolve_input_sample_rate
from base.file_ops import FileOps
from base.play_and_record import get_recorded_info
from base.recording_management import RecordingManager
from base.save_data import save_recorded_data_to_json
from consts import error_code
from consts.audio_consts import VALID_SAMPLE_RATES
from consts.running_consts import DEFAULT_DIR
from ui.sequence.sequence_event_bus import _continuation_message_identity
from ui.sequence.sequence_messages import (
    ConfigurationSnapshot,
    ManualLabelRequested,
    RecordingMarkActionRequested,
    ReplayRequested,
)
from ui.sequence.sequence_recording_model import (
    RecordingCounterStatistics,
    RecordingStatisticsSnapshot,
    immutable_recording_value,
    plain_message_value,
    thaw_recording_session_value,
)


_NUMPY_INTEGER_SCALAR_TYPES = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
)
_NUMPY_FLOAT_SCALAR_TYPES = (np.float16, np.float32, np.float64)


def recording_session_admission_snapshot(command: Any, configuration: Any) -> dict:
    """Create the shared plain admission payload for starts and replays."""
    snapshot = {
        "command_id": command.command_id,
        "source": command.source,
        "record_id": getattr(command, "record_id", command.command_id),
        "label": getattr(command, "label", "not_labeled"),
        "skip_sn_regex_validation": bool(
            getattr(command, "skip_sn_regex_validation", False)
        ),
        "configuration": configuration,
    }
    configuration_generation = getattr(command, "configuration_generation", None)
    if type(configuration_generation) is int:
        snapshot["configuration_generation"] = configuration_generation
    return snapshot


@dataclass(frozen=True, slots=True)
class RecordingReadinessRuntimeCapabilities:
    """Raw device availability sampled once for one readiness request."""

    audio_devices_available: bool
    audio_devices_unavailable_message: str = ""

    def __post_init__(self) -> None:
        if type(self.audio_devices_available) is not bool:
            raise TypeError("audio device availability must be an exact boolean")
        if type(self.audio_devices_unavailable_message) is not str:
            raise TypeError("audio device availability message must be plain text")


@dataclass(frozen=True, slots=True)
class RecordingReadinessSnapshot:
    """Immutable inputs used by the Recording-owned readiness decision."""

    command_id: str
    configuration_generation: int
    sequence_acquisition_config: Any
    recording_mode: str | None
    monitor_playback: bool
    speaker_required: bool
    audio_devices_available: bool
    audio_devices_unavailable_message: str
    input_sample_rate_source: Any
    output_sample_rate_source: Any


@dataclass(frozen=True, slots=True)
class RecordingReadinessResult:
    """One immutable, terminal readiness result and its UI projection."""

    ready: bool
    reason: str
    warning_title: str
    warning_text: str
    runtime_sample_rate: int | float | None
    snapshot: RecordingReadinessSnapshot | None


class SequenceRecordingReadinessService:
    """Own frozen recording-start preflight decisions and warning admission."""

    _MISSING_CONFIGURATION = (
        "未找到可用配置。\n"
        "请先在上方【使用配置】下拉框中选择配置；\n"
        "如无可选项，请到【功能-测试队列】中保存或导入配置。"
    )
    _DEFAULT_DEVICE_FAILURE = (
        "音频设备不可用，请检查设备连接或在【硬件-硬件选择】中重新选择设备。"
    )
    _MISSING_SPEAKER = "未选择扬声器，请在【硬件-硬件选择】中选择扬声器。"
    _MISSING_MICROPHONE = "未找到麦克风，请在硬件中设置"
    _MISSING_LEGACY_SPEAKER = "未找到扬声器，请在硬件中设置"
    _SNAPSHOT_FAILURE = "录音启动检查失败，请重试。"
    _REENTRY_FAILURE = "录音启动检查正在进行，请重试。"
    _CAPACITY_FAILURE = "录音启动检查请求过多，请稍后重试。"
    _MAX_WARNING_LENGTH = 2_048

    def __init__(
        self,
        *,
        runtime_capabilities_provider: Callable[
            [], RecordingReadinessRuntimeCapabilities
        ],
        view: Any = None,
        input_sample_rate_resolver: Callable[[Any], Any] = resolve_input_sample_rate,
        duplex_sample_rate_resolver: Callable[[Any, Any], Any] = (
            resolve_duplex_sample_rate
        ),
        completed_request_limit: int = 128,
        presenting_limit: int | None = None,
        logger: Any = None,
    ) -> None:
        if type(completed_request_limit) is not int or completed_request_limit < 1:
            raise ValueError("completed request limit must be a positive integer")
        if presenting_limit is None:
            presenting_limit = completed_request_limit
        if type(presenting_limit) is not int or presenting_limit < 1:
            raise ValueError("presenting limit must be a positive integer")
        self._runtime_capabilities_provider = runtime_capabilities_provider
        self._view = view
        self._input_sample_rate_resolver = input_sample_rate_resolver
        self._duplex_sample_rate_resolver = duplex_sample_rate_resolver
        self._completed_request_limit = completed_request_limit
        self._presenting_limit = presenting_limit
        # Presenting identities are a protected subset of completed identities;
        # completed plus in-flight identities never exceed this hard bound.
        self._tracked_request_limit = completed_request_limit + presenting_limit
        self._logger = logger
        self._completed: OrderedDict[tuple[str, int], RecordingReadinessResult] = (
            OrderedDict()
        )
        self._inflight: set[tuple[str, int]] = set()
        self._presenting: set[tuple[str, int]] = set()
        self._presented: set[tuple[str, int]] = set()
        self._lock = RLock()

    @property
    def completed_request_count(self) -> int:
        with self._lock:
            return len(self._completed)

    @staticmethod
    def _request_key(command: Any) -> tuple[str, int]:
        command_id = getattr(command, "command_id", None)
        generation = getattr(command, "configuration_generation", None)
        if type(command_id) is not str or not command_id:
            raise TypeError("recording readiness command identifier is invalid")
        if type(generation) is not int or generation < 0:
            raise TypeError("recording readiness configuration generation is invalid")
        return command_id, generation

    def _log_failure(self, context: str, error: BaseException) -> None:
        try:
            log = getattr(self._logger, "warning", None)
            if callable(log):
                log(f"recording readiness {context} failed: {error}")
        except BaseException:
            return

    @staticmethod
    def _acquisition(
        configuration: ConfigurationSnapshot,
    ) -> tuple[Any, str | None, Any]:
        sequence_config = configuration.sequence_config
        if not sequence_config:
            return None, None, {}
        try:
            acquisition = sequence_config[0]["seq1"]["acq"]
            mode = acquisition.get("mode")
            detail = acquisition.get("detail", {}) or {}
        except (AttributeError, IndexError, KeyError, TypeError):
            return immutable_recording_value({}), None, {}
        return immutable_recording_value(plain_message_value(acquisition)), mode, detail

    @classmethod
    def _bounded_plain_text(cls, value: Any) -> str:
        return value[: cls._MAX_WARNING_LENGTH] if type(value) is str else ""

    @classmethod
    def _bounded_warning_text(cls, value: Any, fallback: str) -> str:
        bounded = cls._bounded_plain_text(value)
        if bounded.strip():
            return bounded
        bounded_fallback = cls._bounded_plain_text(fallback)
        if bounded_fallback.strip():
            return bounded_fallback
        return cls._SNAPSHOT_FAILURE[: cls._MAX_WARNING_LENGTH]

    def _snapshot(
        self,
        command: Any,
        configuration: ConfigurationSnapshot,
    ) -> RecordingReadinessSnapshot:
        if type(configuration) is not ConfigurationSnapshot:
            raise TypeError("recording readiness requires ConfigurationSnapshot")
        runtime = self._runtime_capabilities_provider()
        if type(runtime) is not RecordingReadinessRuntimeCapabilities:
            raise TypeError("recording readiness runtime provider returned an invalid value")
        if type(runtime.audio_devices_available) is not bool:
            raise TypeError("recording readiness device availability is invalid")
        device_message = (
            self._bounded_plain_text(runtime.audio_devices_unavailable_message)
            if runtime.audio_devices_available
            else self._bounded_warning_text(
                runtime.audio_devices_unavailable_message,
                self._DEFAULT_DEVICE_FAILURE,
            )
        )
        acquisition, mode, detail = self._acquisition(configuration)
        normalized_detail = (
            normalize_record_only_detail(plain_message_value(detail))
            if mode == "RECORD_ONLY"
            else {}
        )
        monitor_playback = bool(normalized_detail.get("monitor_playback", False))
        speaker_required = mode == "PLAY_AND_RECORD" or (
            mode == "RECORD_ONLY" and monitor_playback
        )
        output_source = (
            configuration.speaker
            if speaker_required or mode not in {"RECORD_ONLY", "PLAY_AND_RECORD"}
            else None
        )
        return RecordingReadinessSnapshot(
            command_id=command.command_id,
            configuration_generation=command.configuration_generation,
            sequence_acquisition_config=acquisition,
            recording_mode=mode if type(mode) is str else None,
            monitor_playback=monitor_playback,
            speaker_required=speaker_required,
            audio_devices_available=runtime.audio_devices_available,
            audio_devices_unavailable_message=device_message,
            input_sample_rate_source=immutable_recording_value(configuration.mic),
            output_sample_rate_source=immutable_recording_value(output_source),
        )

    @staticmethod
    def _accepted(
        snapshot: RecordingReadinessSnapshot,
        sample_rate: int | float | None = None,
    ) -> RecordingReadinessResult:
        return RecordingReadinessResult(True, "", "", "", sample_rate, snapshot)

    @classmethod
    def _rejected(
        cls,
        snapshot: RecordingReadinessSnapshot | None,
        warning_text: str,
        *,
        reason: str = "recording preflight rejected",
    ) -> RecordingReadinessResult:
        return RecordingReadinessResult(
            False,
            reason,
            "提示",
            cls._bounded_warning_text(warning_text, cls._SNAPSHOT_FAILURE),
            None,
            snapshot,
        )

    @staticmethod
    def _normalize_supported_sample_rate(value: Any) -> int:
        if type(value) is int:
            normalized = value
        elif type(value) is float:
            if not math.isfinite(value) or not value.is_integer():
                raise ValueError("sample-rate resolution returned an invalid value")
            normalized = int(value)
        elif type(value) in _NUMPY_INTEGER_SCALAR_TYPES:
            item = value.item()
            if type(item) is not int:
                raise TypeError("sample-rate resolution returned an invalid scalar")
            normalized = item
        elif type(value) in _NUMPY_FLOAT_SCALAR_TYPES:
            item = value.item()
            if (
                type(item) is not float
                or not math.isfinite(item)
                or not item.is_integer()
            ):
                raise ValueError("sample-rate resolution returned an invalid value")
            normalized = int(item)
        else:
            raise TypeError("sample-rate resolution returned an invalid scalar")
        if normalized <= 0 or normalized not in VALID_SAMPLE_RATES:
            raise ValueError("sample-rate resolution returned an unsupported value")
        return normalized

    @classmethod
    def _normalized_rate_result(cls, result: Any) -> tuple[bool, int | None, str]:
        ok = getattr(result, "ok")
        if type(ok) is not bool:
            raise TypeError("sample-rate resolution status must be an exact boolean")
        if ok:
            sample_rate = cls._normalize_supported_sample_rate(
                getattr(result, "sample_rate")
            )
            return True, sample_rate, ""
        message = getattr(result, "message", "")
        return False, None, cls._bounded_warning_text(message, cls._SNAPSHOT_FAILURE)

    def _evaluate(
        self, snapshot: RecordingReadinessSnapshot
    ) -> RecordingReadinessResult:
        if snapshot.sequence_acquisition_config is None:
            return self._rejected(snapshot, self._MISSING_CONFIGURATION)
        if not snapshot.audio_devices_available:
            return self._rejected(
                snapshot,
                snapshot.audio_devices_unavailable_message,
            )
        if snapshot.speaker_required and not snapshot.output_sample_rate_source:
            return self._rejected(snapshot, self._MISSING_SPEAKER)
        if snapshot.recording_mode == "RECORD_ONLY":
            if snapshot.monitor_playback:
                raw_result = self._duplex_sample_rate_resolver(
                    plain_message_value(snapshot.input_sample_rate_source),
                    plain_message_value(snapshot.output_sample_rate_source),
                )
            else:
                raw_result = self._input_sample_rate_resolver(
                    plain_message_value(snapshot.input_sample_rate_source)
                )
            ok, sample_rate, message = self._normalized_rate_result(raw_result)
            return (
                self._accepted(snapshot, sample_rate)
                if ok
                else self._rejected(snapshot, message)
            )
        if snapshot.recording_mode == "PLAY_AND_RECORD":
            raw_result = self._duplex_sample_rate_resolver(
                plain_message_value(snapshot.input_sample_rate_source),
                plain_message_value(snapshot.output_sample_rate_source),
            )
            ok, sample_rate, message = self._normalized_rate_result(raw_result)
            return (
                self._accepted(snapshot, sample_rate)
                if ok
                else self._rejected(snapshot, message)
            )
        if not snapshot.input_sample_rate_source:
            return self._rejected(snapshot, self._MISSING_MICROPHONE)
        if not snapshot.output_sample_rate_source:
            return self._rejected(snapshot, self._MISSING_LEGACY_SPEAKER)
        return self._accepted(snapshot)

    def _reserve_presentation_locked(
        self,
        key: tuple[str, int],
        result: RecordingReadinessResult,
    ) -> bool:
        if (
            result.ready
            or self._view is None
            or len(self._presenting) >= self._presenting_limit
            or key in self._presented
            or key in self._presenting
        ):
            return False
        self._presenting.add(key)
        return True

    def _trim_completed_locked(
        self,
        *,
        protected_key: tuple[str, int] | None = None,
    ) -> None:
        while len(self._completed) > self._completed_request_limit:
            evicted = next(
                (
                    key
                    for key in self._completed
                    if key != protected_key
                    and key not in self._inflight
                    and key not in self._presenting
                ),
                None,
            )
            if evicted is None:
                return
            self._completed.pop(evicted)
            self._presented.discard(evicted)

    def _commit_locked(
        self,
        key: tuple[str, int],
        result: RecordingReadinessResult,
    ) -> bool:
        self._completed[key] = result
        reserved = self._reserve_presentation_locked(key, result)
        self._trim_completed_locked(protected_key=key)
        return reserved

    def _present_reserved(
        self,
        key: tuple[str, int],
        result: RecordingReadinessResult,
    ) -> None:
        succeeded = False
        failure = None
        try:
            callback = getattr(self._view, "present_readiness_warning", None)
            if callable(callback):
                succeeded = callback(result.warning_title, result.warning_text) is not False
        except BaseException as error:
            failure = error
        if failure is not None:
            self._log_failure("presentation", failure)
        with self._lock:
            self._presenting.discard(key)
            if succeeded and self._completed.get(key) is result:
                self._presented.add(key)
            self._trim_completed_locked()

    def _present_reserved_safely(
        self,
        key: tuple[str, int],
        result: RecordingReadinessResult,
    ) -> None:
        failure = None
        try:
            self._present_reserved(key, result)
        except BaseException as error:
            failure = error
            with self._lock:
                self._presenting.discard(key)
                self._trim_completed_locked()
        if failure is not None:
            try:
                self._log_failure("presentation boundary", failure)
            except BaseException:
                return

    def assess(
        self, command: Any, configuration: ConfigurationSnapshot
    ) -> RecordingReadinessResult:
        key = self._request_key(command)
        with self._lock:
            completed = self._completed.get(key)
            if completed is not None:
                present = self._reserve_presentation_locked(key, completed)
            else:
                present = False
            if key in self._inflight:
                return self._rejected(
                    None,
                    self._REENTRY_FAILURE,
                    reason="recording readiness re-entry rejected",
                )
            if completed is not None:
                owner = False
            else:
                if (
                    len(self._completed) + len(self._inflight)
                    >= self._tracked_request_limit
                ):
                    return self._rejected(
                        None,
                        self._CAPACITY_FAILURE,
                        reason="recording readiness capacity unavailable",
                    )
                self._inflight.add(key)
                owner = True
        if not owner:
            if present:
                self._present_reserved_safely(key, completed)
            return completed

        snapshot = None
        result = self._rejected(
            None,
            self._SNAPSHOT_FAILURE,
            reason="recording readiness snapshot is unavailable",
        )
        present = False
        try:
            try:
                snapshot = self._snapshot(command, configuration)
                result = self._evaluate(snapshot)
            except BaseException as error:
                result = self._rejected(
                    snapshot,
                    self._SNAPSHOT_FAILURE,
                    reason="recording readiness snapshot is unavailable",
                )
                try:
                    self._log_failure("snapshot or decision", error)
                except BaseException:
                    pass
        finally:
            with self._lock:
                self._inflight.discard(key)
                present = self._commit_locked(key, result)
        if present:
            self._present_reserved_safely(key, result)
        return result

    def __call__(
        self, command: Any, configuration: ConfigurationSnapshot
    ) -> tuple[bool, str]:
        result = self.assess(command, configuration)
        return result.ready, result.reason


@dataclass(slots=True)
class RecordingAdmissionInputs:
    """Raw View/runtime values sampled by the Recording admission owner."""

    configuration_generation: int
    product_model: str
    serial_number: str
    scanner_enabled: bool
    current_recorded_count: int
    last_play_count: int | None
    recorded_path: str | None
    recorded_signal_info: Mapping[str, Any] | None
    stimulus_data: Any
    stimulus_info: Any
    alignment_sample_count: Any = None


class RecordingAdmissionService:
    """Own path/count/replay decisions for one SequenceWindow instance."""

    def __init__(
        self,
        *,
        raw_inputs: Callable[[], RecordingAdmissionInputs],
        recorded_info_factory: Callable[..., tuple[str, Mapping[str, Any]]] = (
            get_recorded_info
        ),
        command_id_factory: Callable[[], str] = lambda: uuid4().hex,
        pending_replay_limit: int = 64,
    ) -> None:
        if type(pending_replay_limit) is not int or pending_replay_limit < 1:
            raise ValueError("pending replay limit must be a positive integer")
        self._raw_inputs = raw_inputs
        self._recorded_info_factory = recorded_info_factory
        self._command_id_factory = command_id_factory
        self._pending_replay_limit = pending_replay_limit
        self._pending_replays: OrderedDict[str, Mapping[str, Any]] = OrderedDict()

    @property
    def pending_replay_count(self) -> int:
        return len(self._pending_replays)

    def _inputs(self) -> RecordingAdmissionInputs:
        values = self._raw_inputs()
        if type(values) is not RecordingAdmissionInputs:
            raise TypeError("recording raw-input provider returned an invalid value")
        if type(values.configuration_generation) is not int:
            raise TypeError("recording configuration generation must be an integer")
        return values

    def create_replay_request(
        self, *, source: str = "replay-button"
    ) -> ReplayRequested | None:
        values = self._inputs()
        info = values.recorded_signal_info
        record_id = info.get("file_path") if isinstance(info, Mapping) else None
        if type(record_id) is not str or not record_id:
            record_id = values.recorded_path
        if (
            values.last_play_count is None
            or type(record_id) is not str
            or not record_id
            or type(values.recorded_path) is not str
            or not values.recorded_path
        ):
            return None
        command_id = self._command_id_factory()
        if type(command_id) is not str or not command_id:
            raise TypeError("replay command identifier must be a non-empty string")
        target = immutable_recording_value(
            {
                "record_id": record_id,
                "recorded_path": values.recorded_path,
                "recorded_count": values.last_play_count,
                "configuration_generation": values.configuration_generation,
                "product_model": values.product_model,
                "serial_number": values.serial_number,
                "scanner_enabled": bool(values.scanner_enabled),
                "recorded_signal_info": dict(values.recorded_signal_info or {}),
                "stimulus_data": values.stimulus_data,
                "stimulus_info": values.stimulus_info,
                "alignment_sample_count": values.alignment_sample_count,
            }
        )
        self._pending_replays[command_id] = target
        while len(self._pending_replays) > self._pending_replay_limit:
            self._pending_replays.popitem(last=False)
        return ReplayRequested(command_id, source, record_id)

    def replay_readiness(self, command: Any, _configuration: Any) -> tuple[bool, str]:
        target = self._pending_replays.get(getattr(command, "command_id", None))
        if target is None or target.get("record_id") != getattr(command, "record_id", None):
            return False, "replay target is unavailable"
        values = self._inputs()
        if target.get("configuration_generation") != values.configuration_generation:
            return False, "replay configuration is stale"
        if target.get("recorded_count") is None or target.get("recorded_path") is None:
            return False, "replay target is incomplete"
        return True, ""

    def discard_rejected(self, event: Any) -> bool:
        command_id = getattr(event, "command_id", None)
        if type(command_id) is not str:
            return False
        return self._pending_replays.pop(command_id, None) is not None

    def session_snapshot(self, command: Any, configuration: Any) -> dict[str, Any]:
        snapshot = recording_session_admission_snapshot(command, configuration)
        values = self._inputs()
        replay = type(command) is ReplayRequested
        target = None
        target_values = None
        if replay:
            target = self._pending_replays.pop(command.command_id, None)
            if target is None:
                raise ValueError("replay target is unavailable")
            target_values = plain_message_value(target)
            snapshot.update(target_values)
        admitted_values = target_values if target_values is not None else {
            "configuration_generation": getattr(
                command,
                "configuration_generation",
                values.configuration_generation,
            ),
            "product_model": values.product_model,
            "serial_number": values.serial_number,
            "scanner_enabled": bool(values.scanner_enabled),
            "stimulus_data": values.stimulus_data,
            "stimulus_info": values.stimulus_info,
            "alignment_sample_count": values.alignment_sample_count,
        }
        snapshot.update(
            {
                "configuration_generation": admitted_values[
                    "configuration_generation"
                ],
                "product_model": admitted_values["product_model"],
                "serial_number": admitted_values["serial_number"],
                "scanner_enabled": admitted_values["scanner_enabled"],
                "stimulus_data": admitted_values["stimulus_data"],
                "stimulus_info": admitted_values["stimulus_info"],
            }
        )
        if admitted_values["alignment_sample_count"] is not None:
            snapshot["alignment_sample_count"] = admitted_values[
                "alignment_sample_count"
            ]
        if replay:
            recorded_path = snapshot["recorded_path"]
            signal_info = dict(target_values["recorded_signal_info"])
            signal_info["file_path"] = recorded_path
            snapshot["recorded_signal_info"] = signal_info
            snapshot["pending_count"] = None
            return snapshot
        pending_count = int(values.current_recorded_count) + 1
        recorded_path, signal_info = self._recorded_info_factory(
            values.product_model,
            pending_count,
            values.serial_number,
            snapshot.get("label", "not_labeled"),
        )
        snapshot.update(
            {
                "recorded_path": recorded_path,
                "recorded_signal_info": dict(signal_info),
                "record_id": recorded_path,
                "pending_count": pending_count,
            }
        )
        return snapshot


class RecordingPersistenceService:
    """Own the concrete recording database and count persistence boundaries."""

    def __init__(
        self,
        *,
        recording_manager_factory: Callable[[], Any] = RecordingManager,
        count_saver: Callable[..., Any] = save_recorded_data_to_json,
    ) -> None:
        self._recording_manager_factory = recording_manager_factory
        self._count_saver = count_saver

    def save_recording_database(
        self, signal_info: Mapping[str, Any], stimulus_info: Any
    ) -> Any:
        signal_payload = thaw_recording_session_value(signal_info)
        stimulus_payload = (
            None
            if stimulus_info is None
            else thaw_recording_session_value(stimulus_info)
        )
        if type(signal_payload) is not dict:
            raise TypeError("recording database signal info must be a plain dictionary")
        if stimulus_payload is not None and type(stimulus_payload) is not dict:
            raise TypeError(
                "recording database stimulus info must be a plain dictionary"
            )
        manager = self._recording_manager_factory()
        return manager.save_signal_info_to_db(signal_payload, stimulus_payload)

    def persist_count(
        self, pending_count: int | None, persistence: Mapping[str, Any]
    ) -> None:
        if pending_count is None:
            return
        self.persist_view_count(
            persistence.get("product_model") or "",
            pending_count,
            persistence.get("serial_number") or "",
            bool(persistence.get("scanner_enabled", False)),
        )

    def persist_view_count(
        self,
        product_model: Any,
        count: Any,
        serial_number: Any,
        scanner_enabled: Any,
    ) -> None:
        self._count_saver(
            product_model,
            count,
            serial_number,
            bool(scanner_enabled),
        )


class RecordingCancellationService:
    """Provide one idempotent cleanup boundary for blocking and streaming sessions."""

    _HISTORY_LIMIT = 256

    def __init__(self, blocking_adapter: Any) -> None:
        self._blocking_adapter = blocking_adapter
        self._completed: OrderedDict[str, Mapping[str, Any]] = OrderedDict()
        self._lock = RLock()

    def cancel(self, prepared: Any, reason: str) -> Mapping[str, Any]:
        session_id = prepared.snapshot.session_id
        with self._lock:
            completed = self._completed.get(session_id)
            if completed is not None:
                return dict(completed)
            outcome = self._blocking_adapter.cancel(prepared, reason)
            normalized = (
                dict(outcome)
                if isinstance(outcome, Mapping)
                else {"restored": True, "errors": ()}
            )
            frozen = immutable_recording_value(normalized)
            self._completed[session_id] = frozen
            while len(self._completed) > self._HISTORY_LIMIT:
                self._completed.popitem(last=False)
            return dict(frozen)


class RecordingManualLabelRequestService:
    """Own count-board OK/NG admission and immutable Workflow command creation."""

    _NO_RECORDING_WARNING = ("警告", "请先录制声音！")
    _IMPORT_AUDIO_WARNING = (
        "警告",
        "当前为导入音频模式，无需点击 OK/NG 按钮。",
    )
    _IMPORT_STIMULUS_WARNING = (
        "警告",
        "当前为导入激励信号与音频模式，无需点击 OK/NG 按钮。",
    )

    def __init__(
        self,
        *,
        data_provider: Callable[[], Any],
        sequence_config_provider: Callable[[], Any],
        retained_record_id_provider: Callable[[], Any],
        recorded_signal_info_provider: Callable[[], Any],
        recorded_path_provider: Callable[[], Any],
        ok_button: Any,
        ng_button: Any,
        publish: Callable[[ManualLabelRequested], Any],
        present_warning: Callable[[str, str], Any],
        command_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._data_provider = data_provider
        self._sequence_config_provider = sequence_config_provider
        self._retained_record_id_provider = retained_record_id_provider
        self._recorded_signal_info_provider = recorded_signal_info_provider
        self._recorded_path_provider = recorded_path_provider
        self._ok_button = ok_button
        self._ng_button = ng_button
        self._publish = publish
        self._present_warning = present_warning
        self._command_id_factory = command_id_factory or (
            lambda: f"manual-label-{uuid4().hex}"
        )
        self._operation_lock = RLock()
        self._operation_active = False

    @staticmethod
    def _has_samples(value: Any) -> bool:
        if value is None:
            return False
        try:
            return len(value) > 0
        except (TypeError, ValueError, OverflowError):
            return False

    @staticmethod
    def _acquisition_mode(sequence_config: Any) -> str | None:
        try:
            mode = sequence_config[0]["seq1"]["acq"]["mode"]
        except (IndexError, KeyError, TypeError, AttributeError):
            return None
        return mode if type(mode) is str else None

    def _warn(self, warning: tuple[str, str]) -> bool:
        try:
            self._present_warning(*warning)
        except Exception:
            return False
        return False

    def _record_id(self) -> str:
        retained = self._retained_record_id_provider()
        if retained:
            return str(retained)
        information = self._recorded_signal_info_provider()
        if isinstance(information, Mapping):
            try:
                selected = information.get("file_path")
            except Exception:
                selected = None
            if selected:
                return str(selected)
        selected = self._recorded_path_provider()
        return str(selected or "")

    @staticmethod
    def _button(value: Any) -> Any:
        return value() if callable(value) else value

    def request(self, sender: Any, *, manual: Any = True) -> bool:
        """Publish exactly one label command after all legacy guards pass."""
        del manual  # retained only as a compatibility call shape
        with self._operation_lock:
            if self._operation_active:
                return False
            self._operation_active = True
        try:
            try:
                data = self._data_provider()
                sequence_config = self._sequence_config_provider()
            except Exception:
                return self._warn(self._NO_RECORDING_WARNING)
            if not self._has_samples(data):
                return self._warn(self._NO_RECORDING_WARNING)
            mode = self._acquisition_mode(sequence_config)
            if mode == "IMPORT_AUDIO":
                return self._warn(self._IMPORT_AUDIO_WARNING)
            if mode == "IMPORT_STIMULUS_AUDIO":
                return self._warn(self._IMPORT_STIMULUS_WARNING)
            if sender is self._button(self._ok_button):
                label = "OK"
            elif sender is self._button(self._ng_button):
                label = "NG"
            else:
                return False
            try:
                record_id = self._record_id()
                command_id = self._command_id_factory()
                command = ManualLabelRequested(command_id, record_id, label)
            except Exception:
                return False
            published = self._publish(command)
            return published is not False
        finally:
            with self._operation_lock:
                self._operation_active = False


@dataclass(frozen=True, slots=True)
class RecordingLabelContext:
    recorded_path: str
    recorded_signal_info: MutableMapping[str, Any]


@dataclass(frozen=True, slots=True)
class RecordingLabelCommitReceipt:
    old_path: str
    new_path: str
    old_signal_info: Mapping[str, Any]
    new_signal_info: Mapping[str, Any]
    statistics_snapshot: RecordingStatisticsSnapshot | None = None


@dataclass(frozen=True, slots=True)
class RecordingLabelFileCheckpoint:
    old_path: Path
    old_existed: bool
    old_data: bytes
    target_path: Path
    target_existed: bool
    target_data: bytes


@dataclass(frozen=True, slots=True)
class RecordingLabelRollbackOutcome:
    converged: bool
    restored: tuple[str, ...]
    pending: tuple[str, ...]
    errors: tuple[str, ...]


class RecordingLabelRecoveryPending(RuntimeError):
    """A label command whose rollback must converge before any terminal event."""

    recording_label_recovery_pending = True

    def __init__(
        self,
        command_id: str,
        command_identity: tuple[Any, ...],
        outcome: RecordingLabelRollbackOutcome,
    ) -> None:
        super().__init__("recording label rollback recovery is pending")
        self.command_id = command_id
        self.command_identity = command_identity
        self.rollback_outcome = outcome


@dataclass(slots=True)
class _RecordingLabelRecoveryState:
    command: Any
    command_identity: tuple[Any, ...]
    context: RecordingLabelContext
    receipt: RecordingLabelCommitReceipt
    file_checkpoint: RecordingLabelFileCheckpoint
    count_checkpoint: RecordingCountFileCheckpoint | None
    projection: Any
    projection_checkpoint: Any
    original_error: BaseException | None = None
    move_attempted: bool = False
    database_attempted: bool = False
    mapping_attempted: bool = False
    count_attempted: bool = False
    projection_attempted: bool = False
    actual_new_path: str | None = None
    recovery_path: str | None = None
    database_current_path: str | None = None
    database_recovery_attempted: bool = False
    confirmed_layers: set[str] = field(default_factory=set)


_NO_STATISTICS_CHECKPOINT = object()


@dataclass(frozen=True, slots=True)
class RecordingStatisticsOperationPaths:
    test: Path
    mark: Path
    datatime: str

    def __post_init__(self) -> None:
        if not isinstance(self.test, Path) or not isinstance(self.mark, Path):
            raise TypeError("recording statistics paths must be Path values")
        if type(self.datatime) is not str or not self.datatime:
            raise ValueError("recording statistics date must be non-empty text")


@dataclass(frozen=True, slots=True)
class RecordingCountFileCheckpoint:
    path: Path
    existed: bool
    data: bytes
    statistics: Any = _NO_STATISTICS_CHECKPOINT
    operation_paths: RecordingStatisticsOperationPaths | None = None
    statistics_section: str | None = None


@dataclass(frozen=True, slots=True)
class RecordingStatisticsFilesCheckpoint:
    operation_paths: RecordingStatisticsOperationPaths
    test: RecordingCountFileCheckpoint
    mark: RecordingCountFileCheckpoint


class RecordingCountBoardPersistence:
    """Own count-board initialization and atomic label/reset persistence."""

    def __init__(
        self,
        *,
        test_path_provider: Callable[[], Any] | None = None,
        mark_path_provider: Callable[[], Any] | None = None,
        replace_file: Callable[[str, str], Any] = os.replace,
        date_provider: Callable[[], str] | None = None,
        statistics_model: Any = None,
    ) -> None:
        self._test_path_provider = test_path_provider
        self._mark_path_provider = mark_path_provider or (
            lambda: Path(DEFAULT_DIR) / "ui" / "ui_config" / "mark_result.json"
        )
        self._replace_file = replace_file
        self._date_provider = date_provider or (lambda: f"{datetime.now():%Y-%m-%d}")
        self.statistics_model = statistics_model
        self._operation_lock = RLock()
        self._active_operation_identity: Any = None

    def try_begin_operation(self, identity: Any) -> bool:
        if identity is None:
            raise ValueError("recording statistics operation identity is required")
        with self._operation_lock:
            if self._active_operation_identity is None:
                self._active_operation_identity = immutable_recording_value(identity)
                return True
            return self._active_operation_identity == identity

    def end_operation(self, identity: Any) -> bool:
        with self._operation_lock:
            if self._active_operation_identity != identity:
                return False
            self._active_operation_identity = None
            return True

    def current_statistics(self) -> RecordingStatisticsSnapshot:
        if self.statistics_model is None:
            raise RuntimeError("recording statistics model is unavailable")
        snapshot = self.statistics_model.statistics
        if type(snapshot) is not RecordingStatisticsSnapshot:
            raise RuntimeError("recording statistics model is not initialized")
        return snapshot

    @staticmethod
    def _canonical_target(value: Any, name: str) -> Path:
        try:
            raw = os.fspath(value)
        except TypeError as error:
            raise TypeError(
                f"recording statistics {name} path must be path-like"
            ) from error
        if type(raw) is not str:
            raise TypeError(f"recording statistics {name} path must be text")
        if not raw or raw.strip() != raw or "\x00" in raw:
            raise ValueError(f"recording statistics {name} path is invalid")
        raw_path = Path(raw)
        reserved = {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *(f"COM{index}" for index in range(1, 10)),
            *(f"LPT{index}" for index in range(1, 10)),
        }
        for part in raw_path.parts:
            if part == raw_path.anchor or part in {".", ".."}:
                continue
            if (
                part.rstrip(" .") != part
                or any(character in part for character in '<>:"|?*')
                or part.split(".", 1)[0].upper() in reserved
            ):
                raise ValueError(
                    f"recording statistics {name} path is invalid"
                )
        try:
            target = Path(os.path.normcase(str(raw_path.resolve(strict=False))))
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError(
                f"recording statistics {name} path is invalid"
            ) from error
        try:
            if target.exists() and not target.is_file():
                raise ValueError(
                    f"recording statistics {name} target must be a regular file"
                )
            if target.parent.exists() and not target.parent.is_dir():
                raise ValueError(
                    f"recording statistics {name} parent must be a directory"
                )
            ancestor = target.parent
            while not ancestor.exists() and ancestor != ancestor.parent:
                ancestor = ancestor.parent
            if ancestor.exists() and not ancestor.is_dir():
                raise ValueError(
                    f"recording statistics {name} ancestor must be a directory"
                )
        except OSError as error:
            raise ValueError(
                f"recording statistics {name} target cannot be validated"
            ) from error
        return target

    def resolve_operation_paths(self) -> RecordingStatisticsOperationPaths:
        datatime = self._date_provider()
        if type(datatime) is not str or len(datatime) != 10:
            raise ValueError("recording statistics date must be exact YYYY-MM-DD")
        try:
            parsed_date = datetime.strptime(datatime, "%Y-%m-%d")
        except ValueError as error:
            raise ValueError(
                "recording statistics date must be exact YYYY-MM-DD"
            ) from error
        if parsed_date.strftime("%Y-%m-%d") != datatime:
            raise ValueError("recording statistics date must be exact YYYY-MM-DD")
        test_value = (
            Path(DEFAULT_DIR) / "log" / "test_result_log" / f"{datatime}.dat"
            if self._test_path_provider is None
            else self._test_path_provider()
        )
        test_path = self._canonical_target(test_value, "test")
        mark_path = self._canonical_target(self._mark_path_provider(), "mark")
        if test_path == mark_path:
            raise ValueError("recording statistics targets must be distinct")
        if test_path.exists() and mark_path.exists():
            try:
                aliases = os.path.samefile(test_path, mark_path)
            except OSError as error:
                raise ValueError(
                    "recording statistics target identity cannot be validated"
                ) from error
            if aliases:
                raise ValueError("recording statistics targets must be distinct")
        return RecordingStatisticsOperationPaths(test_path, mark_path, datatime)

    @staticmethod
    def _test_payload(datatime: str) -> bytes:
        text = (
            "total: 0\n"
            "ok: 0\n"
            "ng: 0\n"
            "ok_percent: 0%\n"
            f"datatime: {datatime}\n"
        )
        return text.replace("\n", os.linesep).encode("utf-8")

    @staticmethod
    def _mark_payload(datatime: str) -> bytes:
        text = json.dumps(
            {
                "total": 0,
                "ok": 0,
                "ng": 0,
                "not_labels": 0,
                "datatime": datatime,
            },
            indent=4,
        )
        return text.replace("\n", os.linesep).encode("utf-8")

    @staticmethod
    def _checkpoint_path(path: Path) -> RecordingCountFileCheckpoint:
        existed = path.exists()
        return RecordingCountFileCheckpoint(
            path=path,
            existed=existed,
            data=path.read_bytes() if existed else b"",
        )

    def checkpoint_statistics(
        self,
        operation_paths: RecordingStatisticsOperationPaths | None = None,
    ) -> RecordingStatisticsFilesCheckpoint:
        paths = operation_paths or self.resolve_operation_paths()
        return RecordingStatisticsFilesCheckpoint(
            operation_paths=paths,
            test=self._checkpoint_path(paths.test),
            mark=self._checkpoint_path(paths.mark),
        )

    def _atomic_write(self, path: Path, payload: bytes, suffix: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.{suffix}")
        try:
            temporary.write_bytes(payload)
            replaced = self._replace_file(str(temporary), str(path))
            if replaced is False:
                raise RuntimeError("recording statistics atomic replace was rejected")
            if path.read_bytes() != payload:
                raise RuntimeError(
                    "recording statistics atomic replace readback is inconsistent"
                )
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _parse_test_payload(
        payload: bytes, fallback_date: str
    ) -> RecordingCounterStatistics:
        values = {}
        for line in payload.decode("utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            values[key.strip().lower()] = value.strip()
        ok_percent = values.get("ok_percent", "0%")
        if ok_percent and not ok_percent.endswith("%"):
            ok_percent = f"{ok_percent}%"
        return RecordingCounterStatistics(
            total=int(values.get("total", "0")),
            ok=int(values.get("ok", "0")),
            ng=int(values.get("ng", "0")),
            ok_percent=ok_percent,
            datatime=values.get("datatime", fallback_date),
        )

    @staticmethod
    def _parse_mark_payload(payload: bytes) -> RecordingCounterStatistics:
        values = json.loads(payload.decode("utf-8"))
        return RecordingCounterStatistics(
            total=int(values["total"]),
            ok=int(values["ok"]),
            ng=int(values["ng"]),
            not_labels=int(values.get("not_labels", 0)),
            datatime=str(values["datatime"]),
        )

    @classmethod
    def _parse_test(cls, path: Path) -> RecordingCounterStatistics:
        return cls._parse_test_payload(path.read_bytes(), path.stem)

    @classmethod
    def _parse_mark(cls, path: Path) -> RecordingCounterStatistics:
        return cls._parse_mark_payload(path.read_bytes())

    def read_statistics(
        self,
        operation_paths: RecordingStatisticsOperationPaths | None = None,
    ) -> RecordingStatisticsSnapshot:
        paths = operation_paths or self.resolve_operation_paths()
        return RecordingStatisticsSnapshot(
            test=self._parse_test(paths.test),
            mark=self._parse_mark(paths.mark),
        )

    def _statistics_checkpoint(
        self,
        checkpoint_or_paths: RecordingStatisticsFilesCheckpoint
        | RecordingStatisticsOperationPaths
        | None,
    ) -> RecordingStatisticsFilesCheckpoint:
        if type(checkpoint_or_paths) is RecordingStatisticsFilesCheckpoint:
            return checkpoint_or_paths
        return self.checkpoint_statistics(checkpoint_or_paths)

    @classmethod
    def _snapshot_from_payloads(
        cls,
        paths: RecordingStatisticsOperationPaths,
        test_payload: bytes,
        mark_payload: bytes,
    ) -> RecordingStatisticsSnapshot:
        return RecordingStatisticsSnapshot(
            test=cls._parse_test_payload(test_payload, paths.datatime),
            mark=cls._parse_mark_payload(mark_payload),
        )

    def initialize_statistics(
        self,
        checkpoint_or_paths: RecordingStatisticsFilesCheckpoint
        | RecordingStatisticsOperationPaths
        | None = None,
    ) -> RecordingStatisticsSnapshot:
        files = self._statistics_checkpoint(checkpoint_or_paths)
        paths = files.operation_paths
        today = paths.datatime
        if files.test.existed:
            test_payload = files.test.data
        else:
            test_payload = self._test_payload(today)
            self._atomic_write(paths.test, test_payload, "initialize")
        if files.mark.existed:
            mark_payload = files.mark.data
            mark_statistics = self._parse_mark_payload(mark_payload)
            if mark_statistics.datatime != today:
                mark_payload = self._mark_payload(today)
                self._atomic_write(paths.mark, mark_payload, "initialize")
        else:
            mark_payload = self._mark_payload(today)
            self._atomic_write(paths.mark, mark_payload, "initialize")
        return self._snapshot_from_payloads(paths, test_payload, mark_payload)

    def reset_statistics(
        self,
        checkpoint_or_paths: RecordingStatisticsFilesCheckpoint
        | RecordingStatisticsOperationPaths
        | None = None,
    ) -> RecordingStatisticsSnapshot:
        files = self._statistics_checkpoint(checkpoint_or_paths)
        paths = files.operation_paths
        test_payload = self._test_payload(paths.datatime)
        self._atomic_write(
            paths.test,
            test_payload,
            "reset",
        )
        return self._snapshot_from_payloads(
            paths,
            test_payload,
            files.mark.data,
        )

    @staticmethod
    def _post_analysis(command: Any) -> bool:
        outcome = command.export_outcome
        return bool(
            isinstance(outcome, Mapping)
            and outcome.get("continuation") == "post-analysis"
        )

    def checkpoint(self, command: Any) -> RecordingCountFileCheckpoint:
        operation_paths = self.resolve_operation_paths()
        statistics_section = "test" if self._post_analysis(command) else "mark"
        path = getattr(operation_paths, statistics_section)
        checkpoint = self._checkpoint_path(path)
        return RecordingCountFileCheckpoint(
            checkpoint.path,
            checkpoint.existed,
            checkpoint.data,
            self.current_statistics(),
            operation_paths,
            statistics_section,
        )

    @staticmethod
    def _empty_section(
        statistics_section: str, datatime: str
    ) -> RecordingCounterStatistics:
        if statistics_section == "test":
            return RecordingCounterStatistics(
                total=0,
                ok=0,
                ng=0,
                datatime=datatime,
                ok_percent="0%",
            )
        return RecordingCounterStatistics(
            total=0,
            ok=0,
            ng=0,
            datatime=datatime,
            not_labels=0,
        )

    def _reconciled_statistics(
        self, checkpoint: RecordingCountFileCheckpoint
    ) -> RecordingStatisticsSnapshot:
        operation_paths = checkpoint.operation_paths
        if type(operation_paths) is not RecordingStatisticsOperationPaths:
            raise RuntimeError("recording statistics operation paths are unavailable")
        section = checkpoint.statistics_section
        if section not in {"test", "mark"}:
            raise RuntimeError("recording statistics section is unavailable")
        if not checkpoint.existed:
            selected = self._empty_section(section, operation_paths.datatime)
        else:
            try:
                if section == "test":
                    field_names = {
                        line.split(":", 1)[0].strip().lower()
                        for line in checkpoint.data.decode("utf-8").splitlines()
                        if ":" in line
                    }
                    if not {
                        "total",
                        "ok",
                        "ng",
                        "ok_percent",
                        "datatime",
                    }.issubset(field_names):
                        raise ValueError("persisted test fields are incomplete")
                selected = (
                    self._parse_test_payload(
                        checkpoint.data, operation_paths.datatime
                    )
                    if section == "test"
                    else self._parse_mark_payload(checkpoint.data)
                )
            except (
                UnicodeError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                raise ValueError(
                    "recording persisted statistics payload is invalid"
                ) from error
            if selected.datatime != operation_paths.datatime:
                raise ValueError(
                    "recording persisted statistics date does not match operation"
                )
            if selected.total != selected.ok + selected.ng:
                raise ValueError(
                    "recording persisted statistics counts are inconsistent"
                )
        original = checkpoint.statistics
        if type(original) is not RecordingStatisticsSnapshot:
            raise RuntimeError("recording statistics checkpoint is unavailable")
        return RecordingStatisticsSnapshot(
            test=selected if section == "test" else original.test,
            mark=selected if section == "mark" else original.mark,
        )

    def reconcile(
        self, checkpoint: RecordingCountFileCheckpoint
    ) -> RecordingStatisticsSnapshot:
        reconciled = self._reconciled_statistics(checkpoint)
        self.statistics_model.statistics = reconciled
        if self.current_statistics() != reconciled:
            raise RuntimeError("recording statistics reconciliation was rejected")
        return reconciled

    def commit(
        self,
        command: Any,
        checkpoint: RecordingCountFileCheckpoint,
    ) -> Mapping[str, Any]:
        snapshot = self.current_statistics()
        operation_paths = checkpoint.operation_paths
        if type(operation_paths) is not RecordingStatisticsOperationPaths:
            raise RuntimeError("recording statistics operation paths are unavailable")
        statistics_section = checkpoint.statistics_section
        if statistics_section not in {"test", "mark"}:
            raise RuntimeError("recording statistics section is unavailable")
        reconciled = self._reconciled_statistics(checkpoint)
        if snapshot == checkpoint.statistics:
            snapshot = self.reconcile(checkpoint)
        if snapshot != reconciled:
            raise RuntimeError(
                "recording statistics changed after label checkpoint reconciliation"
            )
        if statistics_section == "test":
            total = snapshot.test.total + 1
            ok = snapshot.test.ok
            ng = snapshot.test.ng
            if command.label == "OK":
                ok += 1
            else:
                ng += 1
            ok_percent = round(ok / total * 100, 2) if total else 0
            committed = RecordingStatisticsSnapshot(
                test=RecordingCounterStatistics(
                    total=total,
                    ok=ok,
                    ng=ng,
                    ok_percent=f"{ok_percent}%",
                    datatime=operation_paths.datatime,
                ),
                mark=snapshot.mark,
            )
            payload = (
                f"total: {total}\n"
                f"ok: {ok}\n"
                f"ng: {ng}\n"
                f"ok_percent: {ok_percent}%\n"
                f"datatime: {operation_paths.datatime}\n"
            ).encode("utf-8")
        else:
            total = snapshot.mark.total + 1
            ok = snapshot.mark.ok
            ng = snapshot.mark.ng
            datatime = operation_paths.datatime
            if command.label == "OK":
                ok += 1
            else:
                ng += 1
            committed = RecordingStatisticsSnapshot(
                test=snapshot.test,
                mark=RecordingCounterStatistics(
                    total=total,
                    ok=ok,
                    ng=ng,
                    not_labels=0,
                    datatime=datatime,
                ),
            )
            payload = json.dumps(
                {
                    "total": total,
                    "ng": ng,
                    "ok": ok,
                    "datatime": datatime,
                },
                indent=4,
            ).encode("utf-8")
        checkpoint.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.path.with_name(
            f".{checkpoint.path.name}.{uuid4().hex}.commit"
        )
        try:
            temporary.write_bytes(payload)
            replaced = self._replace_file(str(temporary), str(checkpoint.path))
            if replaced is False:
                raise RuntimeError("recording count-board atomic commit was rejected")
            if checkpoint.path.read_bytes() != payload:
                raise RuntimeError(
                    "recording count-board atomic commit readback is inconsistent"
                )
        finally:
            temporary.unlink(missing_ok=True)
        self.statistics_model.statistics = committed
        return {
            "count_file": str(checkpoint.path),
            "count_file_updated": True,
        }

    def restore(
        self,
        checkpoint: RecordingCountFileCheckpoint,
        _original_error: BaseException,
    ) -> bool:
        if type(checkpoint) is not RecordingCountFileCheckpoint:
            raise TypeError("recording count-board checkpoint is invalid")
        path = checkpoint.path
        if not checkpoint.existed:
            path.unlink(missing_ok=True)
            if checkpoint.statistics is not _NO_STATISTICS_CHECKPOINT:
                self.statistics_model.statistics = checkpoint.statistics
            return not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.rollback")
        try:
            temporary.write_bytes(checkpoint.data)
            replaced = self._replace_file(str(temporary), str(path))
            if replaced is False:
                raise RuntimeError("recording count-board atomic restore was rejected")
        finally:
            temporary.unlink(missing_ok=True)
        restored = path.exists() and path.read_bytes() == checkpoint.data
        if restored and checkpoint.statistics is not _NO_STATISTICS_CHECKPOINT:
            self.statistics_model.statistics = checkpoint.statistics
        return restored

    def restore_statistics(
        self,
        checkpoint: RecordingStatisticsFilesCheckpoint,
        original_error: BaseException,
    ) -> bool:
        if type(checkpoint) is not RecordingStatisticsFilesCheckpoint:
            raise TypeError("recording statistics checkpoint is invalid")
        failures = []
        for item in (checkpoint.mark, checkpoint.test):
            try:
                if self.restore(item, original_error) is False:
                    raise RuntimeError("restore was rejected")
            except BaseException as error:
                failures.append(str(error) or type(error).__name__)
        if failures:
            try:
                original_error.add_note(
                    "recording statistics file rollback failed: "
                    + "; ".join(failures)
                )
            except (AttributeError, TypeError):
                pass
        return not failures


@dataclass(slots=True)
class _RecordingStatisticsRecovery:
    files: RecordingStatisticsFilesCheckpoint
    model_statistics: RecordingStatisticsSnapshot | None
    last_play_count: int | None
    projection_checkpoint: Any
    original_error: BaseException


class RecordingStatisticsService:
    """Own initialization/reset as one recoverable Recording transaction."""

    def __init__(
        self,
        model: Any,
        persistence: RecordingCountBoardPersistence,
        projection: Any,
        *,
        logger: Any = None,
    ) -> None:
        self.model = model
        self.persistence = persistence
        self.projection = projection
        self.logger = logger
        if persistence.statistics_model is None:
            persistence.statistics_model = model
        elif persistence.statistics_model is not model:
            raise ValueError(
                "recording statistics service and persistence must share one model"
            )
        self._lock = RLock()
        self._operation_active = False
        self._recovery: _RecordingStatisticsRecovery | None = None
        self._operation_generation = 0
        self._gate_identity: Any = None

    @property
    def operation_active(self) -> bool:
        with self._lock:
            return self._operation_active

    def _log(self, message: str) -> None:
        callback = getattr(self.logger, "error", None)
        if callable(callback):
            callback(message)

    def _restore(self, recovery: _RecordingStatisticsRecovery) -> bool:
        failures = []
        try:
            if (
                self.projection.restore_statistics_projection(
                    recovery.projection_checkpoint, recovery.original_error
                )
                is False
            ):
                raise RuntimeError("projection restore was rejected")
        except BaseException as error:
            failures.append(f"projection: {error}")
        try:
            self.model.statistics = recovery.model_statistics
            self.model.last_play_count = recovery.last_play_count
        except BaseException as error:
            failures.append(f"model: {error}")
        try:
            if (
                self.persistence.restore_statistics(
                    recovery.files, recovery.original_error
                )
                is False
            ):
                raise RuntimeError("file restore was rejected")
        except BaseException as error:
            failures.append(f"files: {error}")
        if failures:
            try:
                recovery.original_error.add_note(
                    "recording statistics rollback incomplete: "
                    + "; ".join(failures)
                )
            except (AttributeError, TypeError):
                pass
        return not failures

    def _run(self, *, reset_runtime: bool) -> bool:
        with self._lock:
            if self._operation_active:
                return False
            self._operation_active = True
            if self._gate_identity is None:
                self._operation_generation += 1
                self._gate_identity = (
                    "statistics",
                    id(self),
                    self._operation_generation,
                )
            gate_identity = self._gate_identity
        try:
            if not self.persistence.try_begin_operation(gate_identity):
                return False
            if self._recovery is not None:
                if not self._restore(self._recovery):
                    return False
                self._recovery = None
            files = self.persistence.checkpoint_statistics()
            model_statistics = self.model.statistics
            last_play_count = self.model.last_play_count
            projection_checkpoint = (
                self.projection.capture_statistics_projection()
            )
            try:
                snapshot = (
                    self.persistence.reset_statistics(files)
                    if reset_runtime
                    else self.persistence.initialize_statistics(files)
                )
                self.model.statistics = snapshot
                if reset_runtime:
                    self.model.last_play_count = None
                applied = self.projection.apply_statistics_projection(
                    snapshot, reset_runtime=reset_runtime
                )
                if applied is False:
                    raise RuntimeError(
                        "recording statistics projection was rejected"
                    )
                return True
            except BaseException as error:
                recovery = _RecordingStatisticsRecovery(
                    files,
                    model_statistics,
                    last_play_count,
                    projection_checkpoint,
                    error,
                )
                self._recovery = recovery
                if self._restore(recovery):
                    self._recovery = None
                if isinstance(error, Exception):
                    self._log(f"recording statistics transaction failed: {error}")
                    return False
                raise
        finally:
            with self._lock:
                self._operation_active = False
                if self._recovery is None and self._gate_identity is not None:
                    self.persistence.end_operation(self._gate_identity)
                    self._gate_identity = None

    def initialize_statistics(self) -> bool:
        return self._run(reset_runtime=False)

    def reset_statistics(self) -> bool:
        return self._run(reset_runtime=True)


@dataclass(slots=True)
class _RecordingMarkActionRecovery:
    command: RecordingMarkActionRequested
    model_recorded_path: Any
    model_recorded_signal_info: Any
    model_forward_recorded_path: Any
    model_forward_recorded_signal_info: Any
    projection_checkpoint: Any
    original_error: BaseException
    phase: str
    model_path_restored: bool = False
    model_signal_info_restored: bool = False


@dataclass(frozen=True, slots=True)
class RecordingMarkActionRecoveryPending:
    """Immutable diagnostic snapshot of privately owned recovery work."""

    command_id: str
    workflow_generation: int
    phase: str


class RecordingMarkActionService:
    """Own one transactional reset when the operator selects mark mode."""

    def __init__(
        self,
        model: Any,
        projection: Any,
        *,
        workflow_generation_provider: Callable[[], int] | None = None,
        logger: Any = None,
    ) -> None:
        self.model = model
        self.projection = projection
        self.workflow_generation_provider = workflow_generation_provider
        self.logger = logger
        self._lock = RLock()
        self._operation_active = False
        self._recovery: _RecordingMarkActionRecovery | None = None
        self._pending_recovery: RecordingMarkActionRecoveryPending | None = None

    @property
    def pending_recovery(self) -> RecordingMarkActionRecoveryPending | None:
        with self._lock:
            return self._pending_recovery

    def _log(self, message: str) -> None:
        callback = getattr(self.logger, "error", None)
        if callable(callback):
            try:
                callback(message)
            except BaseException:
                return

    def _generation_matches(self, command: RecordingMarkActionRequested) -> bool:
        provider = self.workflow_generation_provider
        if provider is None:
            return True
        generation = provider()
        return (
            type(generation) is int
            and generation == command.workflow_generation
        )

    def _publish_recovery(
        self, recovery: _RecordingMarkActionRecovery, phase: str
    ) -> None:
        with self._lock:
            recovery.phase = phase
            self._recovery = recovery
            self._pending_recovery = RecordingMarkActionRecoveryPending(
                command_id=recovery.command.command_id,
                workflow_generation=recovery.command.workflow_generation,
                phase=phase,
            )

    def _clear_recovery(self) -> None:
        with self._lock:
            self._recovery = None
            self._pending_recovery = None

    def _restore(self, recovery: _RecordingMarkActionRecovery) -> bool:
        failures = []
        try:
            restored = self.projection.restore_mark_action_projection(
                recovery.projection_checkpoint,
                recovery.original_error,
            )
            if restored is False:
                raise RuntimeError("mark-action projection restore was rejected")
        except BaseException as error:
            failures.append(f"projection: {error}")
        if (
            not recovery.model_path_restored
            and not recovery.model_signal_info_restored
            and (
                self.model.recorded_path is not recovery.model_forward_recorded_path
                or self.model.recorded_signal_info
                is not recovery.model_forward_recorded_signal_info
            )
        ):
            recovery.model_path_restored = True
            recovery.model_signal_info_restored = True
        for name, value, forward_value, restored_name in (
            (
                "recorded_path",
                recovery.model_recorded_path,
                recovery.model_forward_recorded_path,
                "model_path_restored",
            ),
            (
                "recorded_signal_info",
                recovery.model_recorded_signal_info,
                recovery.model_forward_recorded_signal_info,
                "model_signal_info_restored",
            ),
        ):
            if getattr(recovery, restored_name):
                continue
            try:
                if getattr(self.model, name) is not forward_value:
                    setattr(recovery, restored_name, True)
                    continue
                setattr(self.model, name, value)
                if getattr(self.model, name) is not value:
                    raise RuntimeError(f"{name} identity restore was rejected")
                setattr(recovery, restored_name, True)
            except BaseException as error:
                failures.append(f"model {name}: {error}")
        if failures:
            try:
                recovery.original_error.add_note(
                    "recording mark-action rollback incomplete: "
                    + "; ".join(failures)
                )
            except (AttributeError, TypeError):
                pass
        return not failures

    def _fail_closed(self, recovery: _RecordingMarkActionRecovery) -> None:
        callback = getattr(
            self.projection, "fail_closed_mark_action_projection", None
        )
        if not callable(callback):
            return
        try:
            if callback(
                recovery.projection_checkpoint,
                recovery.original_error,
            ) is False:
                raise RuntimeError("mark-action fail-closed projection was rejected")
        except BaseException as error:
            try:
                recovery.original_error.add_note(
                    f"recording mark-action fail-closed failed: {error}"
                )
            except (AttributeError, TypeError):
                pass

    def apply(
        self,
        command: RecordingMarkActionRequested,
        *,
        admission_current: Callable[[RecordingMarkActionRequested], bool]
        | None = None,
    ) -> bool:
        if type(command) is not RecordingMarkActionRequested:
            return False
        with self._lock:
            if self._operation_active:
                return False
            if self._recovery is not None:
                return False
            self._operation_active = True
        try:
            if not self._generation_matches(command) or (
                admission_current is not None and not admission_current(command)
            ):
                return False
            forward_signal_info: dict[str, Any] = {}
            recovery = _RecordingMarkActionRecovery(
                command=command,
                model_recorded_path=self.model.recorded_path,
                model_recorded_signal_info=self.model.recorded_signal_info,
                model_forward_recorded_path=None,
                model_forward_recorded_signal_info=forward_signal_info,
                projection_checkpoint=(
                    self.projection.capture_mark_action_projection(command)
                ),
                original_error=RuntimeError(
                    "recording mark-action transaction did not complete"
                ),
                phase="forward",
            )
            try:
                self.model.recorded_path = None
                self.model.recorded_signal_info = forward_signal_info
                applied = self.projection.apply_mark_action_projection(
                    command, recovery.projection_checkpoint
                )
                if applied is False:
                    raise RuntimeError("recording mark-action projection was rejected")
                if not self._generation_matches(command) or (
                    admission_current is not None
                    and not admission_current(command)
                ):
                    raise RuntimeError("recording mark-action admission became stale")
            except BaseException as error:
                recovery.original_error = error
                if self._restore(recovery):
                    self._clear_recovery()
                else:
                    self._publish_recovery(recovery, "rollback")
                    self._fail_closed(recovery)
                if isinstance(error, Exception):
                    self._log(f"recording mark-action transaction failed: {error}")
                    return False
                raise
            try:
                finalized = self.projection.finalize_mark_action_projection(
                    command, recovery.projection_checkpoint
                )
                if finalized is False:
                    raise RuntimeError(
                        "recording mark-action finalization was rejected"
                    )
            except BaseException as error:
                recovery.original_error = error
                self._publish_recovery(recovery, "finalize")
                if isinstance(error, Exception):
                    self._log(
                        f"recording mark-action finalization failed: {error}"
                    )
                    return False
                raise
            return True
        finally:
            with self._lock:
                self._operation_active = False

    def retry_pending_recovery(self) -> bool:
        """Retry only the private canonical recovery, independent of forward gates."""
        with self._lock:
            if self._operation_active or self._recovery is None:
                return False
            recovery = self._recovery
            self._operation_active = True
        try:
            if recovery.phase == "rollback":
                if not self._restore(recovery):
                    self._fail_closed(recovery)
                    return False
                self._clear_recovery()
                return False
            if recovery.phase != "finalize":
                return False
            try:
                finalized = self.projection.finalize_mark_action_projection(
                    recovery.command,
                    recovery.projection_checkpoint,
                )
                if finalized is False:
                    raise RuntimeError(
                        "recording mark-action finalization was rejected"
                    )
            except BaseException as error:
                recovery.original_error = error
                self._publish_recovery(recovery, "finalize")
                if isinstance(error, Exception):
                    self._log(
                        f"recording mark-action recovery finalization failed: {error}"
                    )
                    return False
                raise
            self._clear_recovery()
            return True
        finally:
            with self._lock:
                self._operation_active = False


class RecordingLabelService:
    """Commit WAV/DB label state and roll it back if View projection fails."""

    def __init__(
        self,
        *,
        context_provider: Callable[[], RecordingLabelContext],
        move_wav: Callable[[str, str], str] = FileOps.move_wav_to_dir,
        update_label: Callable[[Mapping[str, Any], str], Any] | None = None,
        count_board_persistence: RecordingCountBoardPersistence | None = None,
        move_target_resolver: Callable[[str, str], Any] | None = None,
        quarantine_move: Callable[[str, str], Any] = os.replace,
        root_dir: str = DEFAULT_DIR,
        replace_file: Callable[[str, str], Any] = os.replace,
    ) -> None:
        self._context_provider = context_provider
        self._move_wav = move_wav
        self._update_label = update_label or (
            lambda info, old: RecordingManager().update_audio_label(info, old)
        )
        self._count_board_persistence = count_board_persistence
        self._move_target_resolver = move_target_resolver or self._default_move_target
        self._quarantine_move = quarantine_move
        self._root_dir = root_dir
        self._replace_file = replace_file
        self._pending_recovery: _RecordingLabelRecoveryState | None = None
        self._operation_lock = RLock()
        self._operation_active = False

    def _stored_path(self, path: str) -> str:
        return path.replace(self._root_dir, "") if self._root_dir else path

    @staticmethod
    def _database_succeeded(result: Any) -> bool:
        if isinstance(result, tuple) and result:
            return result[0] == error_code.OK
        if type(result) is int:
            return result == error_code.OK
        return result is not False

    @staticmethod
    def _restore_mapping(
        target: MutableMapping[str, Any], values: Mapping[str, Any]
    ) -> bool:
        desired = dict(values)
        previous = dict(target)
        try:
            if target.update(desired) is False:
                raise RuntimeError("recording label state update was rejected")
            for key in tuple(target):
                if key not in desired:
                    del target[key]
            if dict(target) != desired:
                raise RuntimeError("recording label state restore is inconsistent")
        except BaseException as error:
            try:
                if target.update(previous) is False:
                    raise RuntimeError(
                        "recording label live state preservation was rejected"
                    )
                for key in tuple(target):
                    if key not in previous:
                        del target[key]
            except BaseException as preservation_error:
                try:
                    error.add_note(
                        "recording label live state preservation failed: "
                        f"{type(preservation_error).__name__}"
                    )
                except (AttributeError, TypeError):
                    pass
            raise
        return True

    @staticmethod
    def _default_move_target(old_path: str, label: str) -> Path:
        source = Path(old_path)
        return source.parent.parent / label / source.name

    @staticmethod
    def _checkpoint_path(path: Path) -> tuple[bool, bytes]:
        existed = path.exists()
        return existed, path.read_bytes() if existed else b""

    def _capture_file_checkpoint(
        self, old_path: str, label: str
    ) -> RecordingLabelFileCheckpoint:
        source = Path(old_path)
        target = Path(self._move_target_resolver(old_path, label))
        old_existed, old_data = self._checkpoint_path(source)
        target_existed, target_data = self._checkpoint_path(target)
        return RecordingLabelFileCheckpoint(
            source,
            old_existed,
            old_data,
            target,
            target_existed,
            target_data,
        )

    def _replace_bytes(self, path: Path, data: bytes, suffix: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.{suffix}")
        try:
            temporary.write_bytes(data)
            replaced = self._replace_file(str(temporary), str(path))
            if replaced is False:
                raise RuntimeError(f"recording label {suffix} was rejected")
            if path.read_bytes() != data:
                raise RuntimeError(
                    f"recording label {suffix} readback is inconsistent"
                )
        finally:
            temporary.unlink(missing_ok=True)

    def _restore_file(self, state: _RecordingLabelRecoveryState) -> bool:
        checkpoint = state.file_checkpoint
        if checkpoint.old_existed:
            self._replace_bytes(
                checkpoint.old_path, checkpoint.old_data, "file-rollback"
            )
        else:
            checkpoint.old_path.unlink(missing_ok=True)
        candidates = {checkpoint.target_path}
        if state.actual_new_path:
            candidates.add(Path(state.actual_new_path))
        if state.recovery_path:
            candidates.add(Path(state.recovery_path))
        for path in candidates:
            if path == checkpoint.old_path:
                continue
            if path == checkpoint.target_path and checkpoint.target_existed:
                self._replace_bytes(path, checkpoint.target_data, "target-rollback")
            else:
                path.unlink(missing_ok=True)
        old_matches = checkpoint.old_path.exists() == checkpoint.old_existed
        if checkpoint.old_existed and old_matches:
            old_matches = checkpoint.old_path.read_bytes() == checkpoint.old_data
        target_matches = checkpoint.target_path.exists() == checkpoint.target_existed
        if checkpoint.target_existed and target_matches:
            target_matches = (
                checkpoint.target_path.read_bytes() == checkpoint.target_data
            )
        restored = old_matches and target_matches
        if restored:
            state.recovery_path = None
        return restored

    def _preserve_file_for_recovery(
        self, state: _RecordingLabelRecoveryState
    ) -> Path:
        checkpoint = state.file_checkpoint
        if not checkpoint.old_existed:
            raise RuntimeError("recording label recovery has no original audio")
        if state.recovery_path:
            recovery = Path(state.recovery_path)
            if recovery.exists() and recovery.read_bytes() == checkpoint.old_data:
                return recovery
        candidates = [checkpoint.old_path]
        if state.actual_new_path:
            candidates.append(Path(state.actual_new_path))
        candidates.append(checkpoint.target_path)
        source = None
        for path in candidates:
            try:
                if path.exists() and path.read_bytes() == checkpoint.old_data:
                    source = path
                    break
            except BaseException:
                continue
        if source is None:
            raise RuntimeError("recording label recovery audio cannot be located")
        recovery = (
            checkpoint.old_path.parent.parent
            / "RECOVERY_PENDING"
            / f"{checkpoint.old_path.name}.{uuid4().hex}.recovery"
        )
        recovery.parent.mkdir(parents=True, exist_ok=True)
        try:
            moved = self._quarantine_move(str(source), str(recovery))
            if moved is False:
                raise RuntimeError("recording label quarantine move was rejected")
        except BaseException:
            if not recovery.exists() or recovery.read_bytes() != checkpoint.old_data:
                raise
        if recovery.read_bytes() != checkpoint.old_data:
            raise RuntimeError("recording label quarantine audio is inconsistent")
        state.recovery_path = str(recovery)
        return recovery

    @staticmethod
    def _error_text(error: BaseException) -> str:
        try:
            return str(error) or type(error).__name__
        except BaseException:
            return type(error).__name__

    @staticmethod
    def _attach_rollback_outcome(
        error: BaseException, outcome: RecordingLabelRollbackOutcome
    ) -> None:
        try:
            setattr(error, "recording_label_rollback_outcome", outcome)
        except (AttributeError, TypeError):
            return

    def _rollback(
        self,
        state: _RecordingLabelRecoveryState,
    ) -> RecordingLabelRollbackOutcome:
        original_error = state.original_error or RuntimeError(
            "recording label rollback requested"
        )
        restored: list[str] = []
        pending: list[str] = []
        errors: list[str] = []

        def settle(
            name: str,
            restore: Callable[[], Any],
        ) -> None:
            if name in state.confirmed_layers:
                return
            try:
                if restore() is False:
                    raise RuntimeError(f"{name} restore was rejected")
            except BaseException as restore_error:
                errors.append(f"{name} restore: {self._error_text(restore_error)}")
                pending.append(name)
                return
            state.confirmed_layers.add(name)
            restored.append(name)

        if state.projection_attempted:
            settle(
                "view",
                lambda: state.projection.restore_label_projection(
                    state.projection_checkpoint, original_error
                ),
            )
        if (
            state.count_attempted
            and self._count_board_persistence is not None
            and state.count_checkpoint is not None
        ):
            settle(
                "count-board",
                lambda: self._count_board_persistence.restore(
                    state.count_checkpoint, original_error
                ),
            )
        if state.database_attempted:
            settle(
                "database",
                lambda: self._database_compensation(state),
            )
        if state.move_attempted:
            settle(
                "file",
                lambda: self._restore_file(state),
            )
        if state.mapping_attempted:
            settle(
                "state",
                lambda: self._restore_mapping(
                    state.context.recorded_signal_info,
                    state.receipt.old_signal_info,
                ),
            )
        if "file" in pending:
            try:
                self._preserve_file_for_recovery(state)
            except BaseException as recovery_error:
                errors.append(
                    f"file recovery: {self._error_text(recovery_error)}"
                )
        for message in errors:
            try:
                original_error.add_note(message)
            except (AttributeError, TypeError):
                break
        return RecordingLabelRollbackOutcome(
            not pending,
            tuple(restored),
            tuple(pending),
            tuple(errors),
        )

    def _database_compensation(self, state: _RecordingLabelRecoveryState) -> bool:
        old_path = state.receipt.old_signal_info["file_path"]
        conditions = [
            state.database_current_path
            or state.receipt.new_signal_info["file_path"]
        ]
        if state.recovery_path or state.database_recovery_attempted:
            conditions.extend(
                (
                    None
                    if state.recovery_path is None
                    else self._stored_path(state.recovery_path),
                    state.receipt.new_signal_info["file_path"],
                    old_path,
                )
            )
        attempted = []
        succeeded = True
        compensation_errors = []
        for condition in conditions:
            if type(condition) is not str or condition in attempted:
                continue
            attempted.append(condition)
            try:
                result = self._update_label(
                    dict(state.receipt.old_signal_info), condition
                )
                if not self._database_succeeded(result):
                    succeeded = False
            except BaseException as error:
                succeeded = False
                compensation_errors.append(error)
        if compensation_errors:
            raise RuntimeError(
                "recording label database compensation was incomplete"
            ) from compensation_errors[0]
        if succeeded:
            state.database_current_path = old_path
            state.database_recovery_attempted = False
        return succeeded

    @staticmethod
    def _command_identity(command: Any) -> tuple[Any, ...]:
        identity = _continuation_message_identity("label-commit", command)
        if identity is None:
            raise ValueError("recording label command identity is unsupported")
        return identity

    def _resume_pending_recovery(
        self, command: Any, command_identity: tuple[Any, ...]
    ) -> None:
        state = self._pending_recovery
        if state is None:
            return
        if command_identity != state.command_identity:
            raise RecordingLabelRecoveryPending(
                state.command.command_id,
                state.command_identity,
                RecordingLabelRollbackOutcome(
                    False, (), ("command",), ("another label recovery is pending",)
                ),
            )
        outcome = self._rollback(state)
        if state.original_error is not None:
            self._attach_rollback_outcome(state.original_error, outcome)
        if not outcome.converged:
            raise RecordingLabelRecoveryPending(
                command.command_id, command_identity, outcome
            )
        self._pending_recovery = None

    def commit(
        self,
        command: Any,
        projection: Any,
    ) -> Mapping[str, Any]:
        with self._operation_lock:
            if self._operation_active:
                raise RuntimeError("recording label operation is already active")
            self._operation_active = True
        command_identity = self._command_identity(command)
        try:
            self._resume_pending_recovery(command, command_identity)
            gate_identity = ("label", id(self), command_identity)
            persistence = self._count_board_persistence
            begin_operation = getattr(persistence, "try_begin_operation", None)
            end_operation = getattr(persistence, "end_operation", None)
            if callable(begin_operation) and not begin_operation(gate_identity):
                raise RuntimeError("recording statistics operation is already active")
            try:
                return self._commit_admitted(command, projection)
            finally:
                if callable(end_operation) and self._pending_recovery is None:
                    end_operation(gate_identity)
        finally:
            with self._operation_lock:
                self._operation_active = False

    def _commit_admitted(
        self,
        command: Any,
        projection: Any,
    ) -> Mapping[str, Any]:
        command_identity = self._command_identity(command)
        if command.label not in {"OK", "NG"}:
            raise ValueError("recording label must be OK or NG")
        context = self._context_provider()
        if type(context) is not RecordingLabelContext:
            raise TypeError("recording label context is invalid")
        if not isinstance(context.recorded_signal_info, MutableMapping):
            raise TypeError("recording label signal information must be mutable")
        old_info = dict(context.recorded_signal_info)
        old_record_id = old_info.get("file_path")
        if command.record_id not in {context.recorded_path, old_record_id}:
            raise ValueError("recording label command is stale")
        if old_info.get("labels") not in {None, "not_labeled", command.label}:
            raise ValueError("recording already has a different label")
        projection_checkpoint = projection.capture_label_projection(command)
        count_checkpoint = (
            None
            if self._count_board_persistence is None
            else self._count_board_persistence.checkpoint(command)
        )
        file_checkpoint = self._capture_file_checkpoint(
            context.recorded_path, command.label
        )
        expected_path = str(file_checkpoint.target_path)
        new_info = dict(old_info)
        new_info["labels"] = command.label
        new_info["file_path"] = self._stored_path(expected_path)
        receipt = RecordingLabelCommitReceipt(
            context.recorded_path,
            expected_path,
            immutable_recording_value(old_info),
            immutable_recording_value(new_info),
        )
        state = _RecordingLabelRecoveryState(
            command,
            command_identity,
            context,
            receipt,
            file_checkpoint,
            count_checkpoint,
            projection,
            projection_checkpoint,
        )
        count_outcome: Mapping[str, Any] = {}
        try:
            if (
                self._count_board_persistence is not None
                and count_checkpoint is not None
            ):
                state.count_attempted = True
                self._count_board_persistence.reconcile(count_checkpoint)
            state.move_attempted = True
            new_path = self._move_wav(context.recorded_path, command.label)
            if type(new_path) is not str or not new_path:
                raise RuntimeError("recording label file move failed")
            state.actual_new_path = new_path
            if Path(new_path) != file_checkpoint.target_path:
                raise RuntimeError("recording label file moved to an unexpected path")
            new_info["file_path"] = self._stored_path(new_path)
            receipt = RecordingLabelCommitReceipt(
                context.recorded_path,
                new_path,
                immutable_recording_value(old_info),
                immutable_recording_value(new_info),
            )
            state.receipt = receipt
            state.database_attempted = True
            state.database_current_path = new_info["file_path"]
            result = self._update_label(new_info, old_record_id)
            if not self._database_succeeded(result):
                raise RuntimeError("recording label database update failed")
            state.mapping_attempted = True
            self._restore_mapping(context.recorded_signal_info, new_info)
            if self._count_board_persistence is not None:
                count_outcome = self._count_board_persistence.commit(
                    command, count_checkpoint
                )
                if not isinstance(count_outcome, Mapping):
                    raise RuntimeError(
                        "recording label count-board commit was rejected"
                    )
                current_statistics = getattr(
                    self._count_board_persistence, "current_statistics", None
                )
                if callable(current_statistics):
                    receipt = RecordingLabelCommitReceipt(
                        receipt.old_path,
                        receipt.new_path,
                        receipt.old_signal_info,
                        receipt.new_signal_info,
                        current_statistics(),
                    )
                    state.receipt = receipt
            state.projection_attempted = True
            if projection.apply_label_projection(
                command, receipt, projection_checkpoint
            ) is False:
                raise RuntimeError("recording label View projection was rejected")
        except BaseException as error:
            state.original_error = error
            outcome = self._rollback(state)
            self._attach_rollback_outcome(error, outcome)
            if not outcome.converged:
                self._pending_recovery = state
                raise RecordingLabelRecoveryPending(
                    command.command_id, command_identity, outcome
                ) from error
            raise
        return {
            "file_path": new_info["file_path"],
            "database_updated": True,
            "view_projected": True,
            **dict(count_outcome),
            "export_outcome": command.export_outcome,
        }
