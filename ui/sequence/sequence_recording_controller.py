"""Recording domain orchestration for admitted sequence sessions."""

from __future__ import annotations

import os
import math
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path, PurePath
from threading import Event, Lock, RLock, Thread, get_ident
from types import MappingProxyType, SimpleNamespace
from typing import Any
from uuid import uuid4
from weakref import ref

from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot

import numpy as np

from base.acquisition_recording_defaults import (
    normalize_play_record_detail,
    normalize_record_only_detail,
)
from base import soundcard_audio_processor as soundcard_audio_module
from base.audio_sample_rate import resolve_duplex_sample_rate, resolve_input_sample_rate
from base.load_config import LoadUiConfig
from base.pre_processing.split_repeat_signal import SplitRepeatSignal
from base.recording_calibration_snapshot import build_recording_wav_calibration_metadata
from base.soundcard_audio_processor import (
    SoundcardAudioProcessor,
    alignment_reference_from_stimulus,
)
from base.save_data import save_audio_simple
from base.wav_calibration_metadata import append_wav_calibration_metadata
from consts import error_code
from consts.audio_consts import normalize_float_bit_depth

from ui.sequence.sequence_event_bus import (
    SequenceEventBus,
    _continuation_message_identity,
)
from ui.sequence.sequence_messages import (
    BeginRecordingRequested,
    CancelImportedAudioRequested,
    CancelRecordingRequested,
    CommitRecordingLabelRequested,
    RecordingCancelled,
    RecordingCompleted,
    RecordingFailed,
    RecordingLabelCommitted,
    RecordingLabelCommitFailed,
    RecordingStarted,
    RecordingMarkActionRequested,
    LoadImportedAudioRequested,
)
from ui.sequence.sequence_recording_import_owner import (
    SequenceRecordingImportController,
)
from ui.sequence.sequence_recording_model import (
    RecordingModel,
    RecordingSessionSnapshot,
    RecordingState,
    StagedRecording,
    freeze_recording_session_value,
    immutable_recording_value,
    plain_message_value,
    thaw_recording_session_value,
)
from ui.sequence.sequence_recording_transaction import (
    RecordingCancellationRequested,
    RecordingCommitResult,
    RecordingDurableResult,
    RecordingTransaction,
)
from ui.sequence.sequence_recording_view import SequenceRecordingView


def _recording_sample_counts(
    mode: str,
    recorded_dict: Mapping[str, Any],
    stimulus_dict: Mapping[str, Any],
) -> tuple[int, int]:
    """Return physical acquisition frames and the post-processing result size."""

    def frames(name: str, default: int = 0) -> int:
        value = recorded_dict.get(name, default)
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a non-negative integer")
        try:
            normalized = int(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be a non-negative integer") from error
        if normalized < 0 or normalized != value:
            raise ValueError(f"{name} must be a non-negative integer")
        return normalized

    delay = frames("recording_start_delay_frames")
    prolong = frames("prolong_frames")
    if mode == "PLAY_AND_RECORD":
        prepare = frames("prepare_frames", 1000)
        stimulus = np.asarray(stimulus_dict.get("data", []))
        if stimulus.ndim == 0:
            raise ValueError("play-and-record stimulus must contain samples")
        full_stimulus_count = int(stimulus.shape[0])
        result_count = int(
            np.asarray(alignment_reference_from_stimulus(stimulus_dict)).shape[0]
        )
        acquisition_count = delay + prepare + full_stimulus_count + prolong
    else:
        recorded_frames = frames("num_frames")
        acquisition_count = delay + recorded_frames
        result_count = recorded_frames - prolong
    if acquisition_count <= 0:
        raise ValueError(
            "recording acquisition target sample count must be positive"
        )
    if result_count <= 0:
        raise ValueError("recording target sample count must be positive")
    return acquisition_count, result_count


_RECORDING_ACQUISITION_FIELDS = (
    "detail",
    "stimulus_dict",
    "recorded_dict",
    "recorded_signal_info",
    "stimulus_info",
    "stimulus_data",
    "alignment_sample_count",
    "count_persistence",
    "use_streaming",
    "wav_calibration_metadata",
)


@dataclass(frozen=True, slots=True, init=False)
class RecordingAcquisitionContext(Mapping[str, Any]):
    """Typed, deeply immutable values consumed by one acquisition worker."""

    detail: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    stimulus_dict: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    recorded_dict: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    recorded_signal_info: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    stimulus_info: Any = None
    stimulus_data: Any = None
    alignment_sample_count: Any = None
    count_persistence: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    use_streaming: bool = False
    wav_calibration_metadata: Any = None

    def __init__(self, values: Any = None, /) -> None:
        if values is None:
            values = {}
        if type(values) is not dict:
            raise TypeError("recording acquisition context must be an exact dict")
        supplied_names = []
        unknown = []
        for name in dict.__iter__(values):
            if type(name) is not str:
                raise TypeError(
                    "recording acquisition field names must use exact text"
                )
            supplied_names.append(name)
            if name not in _RECORDING_ACQUISITION_FIELDS:
                unknown.append(name)
        if unknown:
            raise ValueError(
                "unsupported recording acquisition values: "
                + ", ".join(sorted(unknown))
            )
        supplied_values = {
            name: dict.__getitem__(values, name)
            for name in supplied_names
        }
        frozen_context = freeze_recording_session_value(supplied_values)
        frozen = {name: frozen_context[name] for name in supplied_names}
        self._install_frozen_values(frozen)

    def _install_frozen_values(self, frozen: dict[str, Any]) -> None:
        empty_mapping = freeze_recording_session_value({})
        for name in (
            "detail",
            "stimulus_dict",
            "recorded_dict",
            "recorded_signal_info",
            "count_persistence",
        ):
            candidate = frozen.get(name, empty_mapping)
            if not isinstance(candidate, Mapping):
                raise TypeError(f"{name} must be an exact data-only mapping")
            frozen[name] = candidate
        use_streaming = frozen.get("use_streaming", False)
        if type(use_streaming) is not bool:
            raise TypeError("use_streaming must be an exact boolean")
        frozen["use_streaming"] = use_streaming
        for name in _RECORDING_ACQUISITION_FIELDS:
            if name in frozen:
                value = frozen[name]
            elif name in (
                "detail",
                "stimulus_dict",
                "recorded_dict",
                "recorded_signal_info",
                "count_persistence",
            ):
                value = empty_mapping
            elif name == "use_streaming":
                value = False
            else:
                value = None
            object.__setattr__(self, name, value)

    @classmethod
    def create(cls, values: Any) -> "RecordingAcquisitionContext":
        return cls(values)

    @classmethod
    def canonicalize(cls, value: Any) -> "RecordingAcquisitionContext":
        if type(value) is not cls:
            raise TypeError(
                "recording acquisition context must be RecordingAcquisitionContext"
            )
        supplied_values = {
            name: object.__getattribute__(value, name)
            for name in _RECORDING_ACQUISITION_FIELDS
        }
        return cls(supplied_values)

    def __getitem__(self, key: str) -> Any:
        if type(key) is not str or key not in _RECORDING_ACQUISITION_FIELDS:
            raise KeyError(key)
        return object.__getattribute__(self, key)

    def __iter__(self):
        return iter(_RECORDING_ACQUISITION_FIELDS)

    def __len__(self) -> int:
        return len(_RECORDING_ACQUISITION_FIELDS)


@dataclass(frozen=True, slots=True)
class PreparedRecordingSession:
    snapshot: RecordingSessionSnapshot
    acquisition_context: RecordingAcquisitionContext

    def __post_init__(self) -> None:
        if type(self.snapshot) is not RecordingSessionSnapshot:
            raise TypeError("prepared session requires RecordingSessionSnapshot")
        canonical_snapshot = RecordingSessionSnapshot.canonicalize(self.snapshot)
        context = self.acquisition_context
        if context is None:
            canonical_context = RecordingAcquisitionContext.create({})
        elif type(context) is RecordingAcquisitionContext:
            canonical_context = RecordingAcquisitionContext.canonicalize(context)
        else:
            canonical_context = RecordingAcquisitionContext.create(context)
        context_values = {
            name: object.__getattribute__(canonical_context, name)
            for name in _RECORDING_ACQUISITION_FIELDS
        }
        supplied_count_persistence = context_values["count_persistence"]
        canonical_count_persistence = canonical_snapshot.count_persistence
        if supplied_count_persistence and (
            supplied_count_persistence != canonical_count_persistence
        ):
            raise ValueError(
                "count_persistence must match the prepared recording snapshot"
            )
        context_values["count_persistence"] = canonical_count_persistence
        object.__setattr__(self, "snapshot", canonical_snapshot)
        object.__setattr__(
            self,
            "acquisition_context",
            RecordingAcquisitionContext.create(context_values),
        )

    @classmethod
    def canonicalize(cls, value: Any) -> "PreparedRecordingSession":
        if type(value) is not cls:
            raise TypeError("prepared session must be PreparedRecordingSession")
        return cls(
            snapshot=object.__getattribute__(value, "snapshot"),
            acquisition_context=object.__getattribute__(
                value, "acquisition_context"
            ),
        )


def _mutable_recording_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            _mutable_recording_value(key): _mutable_recording_value(item)
            for key, item in value.items()
        }
    if type(value) is tuple:
        return [_mutable_recording_value(item) for item in value]
    if type(value) is frozenset:
        return {_mutable_recording_value(item) for item in value}
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, PurePath):
        return type(value)(value)
    return value


class BlockingRecordingAdapter:
    """Concrete blocking acquisition service fed only frozen admission values."""

    def __init__(
        self,
        *,
        data_struct: Any,
        save_database: Callable[[Mapping[str, Any], Any], Any],
        commit_count: Callable[[int | None], None],
        persist_count: Callable[[int | None, Mapping[str, Any]], None],
        audio_processor_factory: Callable[[], Any] = SoundcardAudioProcessor,
        logger: Any = None,
    ) -> None:
        self.data_struct = data_struct
        self.save_database_port = save_database
        self.commit_count_port = commit_count
        self.persist_count_port = persist_count
        self.audio_processor_factory = audio_processor_factory
        self.logger = logger
        self._acquisition_lock = RLock()
        self._active_processors: dict[str, Any] = {}
        self._cancelled_sessions: set[str] = set()

    @staticmethod
    def _normalized_acquisition(configuration: Any) -> tuple[str, dict[str, Any]]:
        sequence_config = _mutable_recording_value(configuration.sequence_config)
        acq = sequence_config[0]["seq1"]["acq"]
        mode = acq["mode"]
        detail = dict(acq.get("detail") or {})
        if mode == "RECORD_ONLY":
            detail = normalize_record_only_detail(detail)
        elif mode == "PLAY_AND_RECORD":
            detail = normalize_play_record_detail(detail)
        return mode, detail

    @staticmethod
    def _resolved_sample_rate(
        mode: str, detail: Mapping[str, Any], mic: Any, speaker: Any
    ) -> int:
        if mode == "RECORD_ONLY":
            if detail.get("monitor_playback", False):
                result = resolve_duplex_sample_rate(mic, speaker)
            else:
                result = resolve_input_sample_rate(mic)
        elif mode == "PLAY_AND_RECORD":
            result = resolve_duplex_sample_rate(mic, speaker)
        else:
            raise ValueError(f"unsupported blocking recording mode: {mode}")
        if not result.ok:
            raise RuntimeError(result.message)
        return result.sample_rate

    @staticmethod
    def _normalize_recorded_data(
        recorded_data: Any, input_channels: tuple[int, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        recorded_array = np.array(recorded_data, copy=True)
        if recorded_array.ndim not in (1, 2):
            raise ValueError(
                f"recorded data dimension is unsupported: {recorded_array.ndim}D"
            )
        if not (
            np.issubdtype(recorded_array.dtype, np.floating)
            or np.issubdtype(recorded_array.dtype, np.signedinteger)
            or np.issubdtype(recorded_array.dtype, np.unsignedinteger)
        ):
            raise ValueError("recorded data must contain real numeric samples")
        if not np.all(np.isfinite(recorded_array)):
            raise ValueError("recorded data contains non-finite samples")
        if not np.issubdtype(recorded_array.dtype, np.floating):
            recorded_array = recorded_array.astype(np.float32)
        if recorded_array.size == 0:
            raise ValueError("empty recorded data")
        expected_channels = len(input_channels)
        if expected_channels < 1:
            raise ValueError("recorded data requires at least one frozen channel")
        if recorded_array.ndim == 1:
            if expected_channels != 1:
                raise ValueError(
                    "recorded data channel layout does not match frozen channels"
                )
            mono = recorded_array.reshape(-1)
            return mono, mono.reshape(-1, 1)
        if (
            recorded_array.shape[0] == expected_channels
            and recorded_array.shape[1] != expected_channels
        ):
            recorded_array = recorded_array.T
        if recorded_array.shape[1] != expected_channels:
            raise ValueError(
                "recorded data channel layout does not match frozen channels: "
                f"expected {expected_channels}, got {recorded_array.shape[1]}"
            )
        multi = np.asarray(recorded_array)
        mono = (
            multi[:, 0]
            if multi.shape[1] == 1
            else multi.mean(axis=1).astype(multi.dtype, copy=False)
        )
        return mono.reshape(-1), multi

    def prepare(
        self, command: BeginRecordingRequested
    ) -> PreparedRecordingSession:
        admission = command.session_snapshot
        if not isinstance(admission, Mapping):
            raise TypeError("recording admission snapshot must be a mapping")
        configuration = admission.get("configuration")
        if configuration is None:
            raise ValueError("recording admission has no configuration snapshot")
        mode, detail = self._normalized_acquisition(configuration)
        mic = _mutable_recording_value(configuration.mic)
        speaker = _mutable_recording_value(configuration.speaker)
        input_channels = tuple(int(item) for item in (configuration.mic_channels or (0,)))
        if not input_channels:
            input_channels = (0,)
        speaker_required = mode == "PLAY_AND_RECORD" or (
            mode == "RECORD_ONLY" and detail.get("monitor_playback", False)
        )
        if (
            speaker_required
            and isinstance(mic, Mapping)
            and isinstance(speaker, Mapping)
            and mic.get("hostapi") != speaker.get("hostapi")
        ):
            raise RuntimeError(
                "播放+录制需要选择同一驱动类型（Host API）的输入/输出设备。\n"
                f"当前输入: {mic.get('name')} (hostapi={mic.get('hostapi')})\n"
                f"当前输出: {speaker.get('name')} (hostapi={speaker.get('hostapi')})"
            )
        sample_rate = self._resolved_sample_rate(mode, detail, mic, speaker)
        stimulus_data = _mutable_recording_value(admission.get("stimulus_data"))
        stimulus_info = _mutable_recording_value(admission.get("stimulus_info"))
        alignment_sample_count = admission.get("alignment_sample_count")
        staged_struct = SimpleNamespace(
            sample_rate=sample_rate,
            stimulus_data=stimulus_data,
            stimulus_info=stimulus_info,
        )
        if alignment_sample_count is not None:
            staged_struct.alignment_sample_count = alignment_sample_count
        try:
            total_time = float(detail.get("total_time", 5.0))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("recording duration must be positive and finite") from error
        if not math.isfinite(total_time) or total_time <= 0:
            raise ValueError("recording duration must be positive and finite")
        delay_ms = detail.get("recording_start_delay_ms")
        stimulus_dict, recorded_dict = (
            LoadUiConfig.get_rec_and_play_dict_base_sequence_dict(
                staged_struct,
                total_time,
                recording_start_delay_ms=delay_ms,
            )
        )
        recorded_dict["sr"] = sample_rate
        recorded_dict["sample_rate"] = sample_rate
        if stimulus_dict:
            stimulus_dict["sr"] = sample_rate
        bit_depth = normalize_float_bit_depth(
            mic.get("bit_depth", 32) if isinstance(mic, dict) else 32
        )
        recorded_dict.update(
            {
                "device": mic,
                "input_device": mic,
                "bit_depth": bit_depth,
                "input_channels": list(input_channels),
                "channels": max(1, len(input_channels)),
            }
        )
        if mode == "PLAY_AND_RECORD":
            recorded_dict["output_device"] = speaker
            metadata_channels = (input_channels[0],)
        elif detail.get("monitor_playback", False):
            monitor_channel = int(detail.get("monitor_input_channel", input_channels[0]))
            if monitor_channel not in input_channels:
                monitor_channel = input_channels[0]
            recorded_dict.update(
                {
                    "monitor_playback": True,
                    "monitor_input_channel": monitor_channel,
                    "monitor_gain_db": float(detail.get("monitor_gain_db", 0.0)),
                    "output_device": speaker,
                }
            )
            metadata_channels = input_channels
        else:
            recorded_dict["output_device"] = None
            metadata_channels = input_channels
        hardware_id = mic.get("hardware_id") if isinstance(mic, dict) else None
        metadata = build_recording_wav_calibration_metadata(
            metadata_channels,
            hardware_id=hardware_id,
            logger=self.logger,
        )
        recorded_dict["wav_calibration_metadata"] = metadata
        output_path = Path(admission["recorded_path"])
        temp_path = output_path.with_name(
            f"{output_path.stem}.{command.session_id}.tmp{output_path.suffix or '.wav'}"
        )
        backup_path = (
            output_path.with_name(f"{output_path.stem}_replay_backup{output_path.suffix or '.wav'}")
            if command.replay
            else None
        )
        pending_count = admission.get("pending_count")
        count_persistence = {
            "product_model": admission.get("product_model") or "",
            "serial_number": admission.get("serial_number") or "",
            "scanner_enabled": bool(admission.get("scanner_enabled", False)),
        }
        acquisition_sample_count, target_samples = _recording_sample_counts(
            mode, recorded_dict, stimulus_dict
        )
        snapshot = RecordingSessionSnapshot.create(
            session_id=command.session_id,
            workflow_generation=admission["workflow_generation"],
            configuration_generation=admission.get("configuration_generation", 0),
            mode=mode,
            sample_rate=sample_rate,
            bit_depth=bit_depth,
            input_channels=input_channels,
            input_device=mic,
            output_device=recorded_dict.get("output_device"),
            stimulus_snapshot={
                "data": stimulus_data,
                "info": stimulus_info,
                "alignment_sample_count": alignment_sample_count,
            },
            target_samples=target_samples,
            acquisition_sample_count=acquisition_sample_count,
            output_path=output_path,
            temp_path=temp_path,
            backup_path=backup_path,
            record_id=str(admission.get("record_id") or output_path),
            label=str(admission.get("label") or "not_labeled"),
            analysis_config=_mutable_recording_value(configuration.analysis_config),
            configuration_snapshot={
                "sequence_config": _mutable_recording_value(configuration.sequence_config),
                "analysis_config": _mutable_recording_value(configuration.analysis_config),
            },
            product_model=count_persistence["product_model"],
            serial_number=count_persistence["serial_number"],
            scanner_enabled=count_persistence["scanner_enabled"],
            count_persistence=count_persistence,
            pending_count=pending_count,
            replay=command.replay,
        )
        context = {
            "detail": detail,
            "stimulus_dict": stimulus_dict,
            "recorded_dict": recorded_dict,
            "recorded_signal_info": _mutable_recording_value(
                admission["recorded_signal_info"]
            ),
            "stimulus_info": stimulus_info,
            "stimulus_data": stimulus_data,
            "alignment_sample_count": alignment_sample_count,
            "count_persistence": count_persistence,
            "use_streaming": bool(detail.get("use_streaming_recording", False)),
            "wav_calibration_metadata": metadata,
        }
        return PreparedRecordingSession(snapshot, context)

    def acquire(self, prepared: PreparedRecordingSession) -> StagedRecording:
        snapshot = prepared.snapshot
        context = prepared.acquisition_context
        recorded_dict = thaw_recording_session_value(context["recorded_dict"])
        processor = self.audio_processor_factory()
        with self._acquisition_lock:
            if snapshot.session_id in self._cancelled_sessions:
                raise RuntimeError("recording acquisition was cancelled")
            self._active_processors[snapshot.session_id] = processor
        try:
            if snapshot.mode == "PLAY_AND_RECORD":
                acquisition_dict = dict(recorded_dict)
                acquisition_dict["wav_calibration_metadata"] = None
                code, recorded_data = processor.sd_play_rec(
                    acquisition_dict,
                    thaw_recording_session_value(context["stimulus_dict"]),
                    str(snapshot.temp_path),
                    calibration_metadata=None,
                )
                normalized_channels = snapshot.input_channels[:1]
            else:
                code, recorded_data = processor.sd_rec(recorded_dict)
                normalized_channels = snapshot.input_channels
            self._raise_if_cancelled(snapshot.session_id)
            if code != error_code.OK or recorded_data is None:
                raise RuntimeError(
                    recorded_data if recorded_data is not None else code
                )
            mono, multi = self._normalize_recorded_data(
                recorded_data, normalized_channels
            )
            self._raise_if_cancelled(snapshot.session_id)
            if snapshot.mode == "RECORD_ONLY":
                save_audio_simple(
                    str(snapshot.temp_path),
                    multi if multi.shape[1] > 1 else mono,
                    snapshot.sample_rate,
                    bit_depth=snapshot.bit_depth,
                )
            self._raise_if_cancelled(snapshot.session_id)
            signal_info = thaw_recording_session_value(
                context["recorded_signal_info"]
            )
            signal_info["sample_rate"] = snapshot.sample_rate
            fields = {
                "store_wave_data": mono,
                "store_wave_data_multi": multi,
                "sample_rate": snapshot.sample_rate,
                "audio_lenth": int(multi.shape[0]),
                "fft_result": None,
                "stft_result": None,
                "split_repeat_data": None,
                "wav_calibration_metadata": None,
                "wav_calibration_metadata_authoritative": False,
                "wav_calibration_warning_shown": False,
                "stimulus_data": thaw_recording_session_value(
                    context.get("stimulus_data")
                ),
                "stimulus_info": thaw_recording_session_value(
                    context.get("stimulus_info")
                ),
            }
            if context.get("alignment_sample_count") is not None:
                fields["alignment_sample_count"] = context[
                    "alignment_sample_count"
                ]
            return StagedRecording.create(
                snapshot=snapshot,
                sample_count=int(multi.shape[0]),
                data_struct_fields=fields,
                recorded_signal_info=signal_info,
                stimulus_info=thaw_recording_session_value(
                    context.get("stimulus_info")
                ),
            )
        finally:
            with self._acquisition_lock:
                if self._active_processors.get(snapshot.session_id) is processor:
                    self._active_processors.pop(snapshot.session_id, None)

    def _raise_if_cancelled(self, session_id: str) -> None:
        with self._acquisition_lock:
            cancelled = session_id in self._cancelled_sessions
        if cancelled:
            raise RuntimeError("recording acquisition was cancelled")

    def request_cancel(
        self, prepared: PreparedRecordingSession, _reason: str
    ) -> None:
        session_id = prepared.snapshot.session_id
        with self._acquisition_lock:
            self._cancelled_sessions.add(session_id)
            processor = self._active_processors.get(session_id)
        for method_name in ("request_cancel", "cancel", "stop", "stop_streaming"):
            callback = getattr(processor, method_name, None)
            if callable(callback):
                callback()
                return
        soundcard_audio_module.sd.stop()

    def transaction(self, prepared: PreparedRecordingSession) -> RecordingTransaction:
        def alignment_handoff(staged: StagedRecording) -> Mapping[str, Any]:
            stimulus_info = staged.stimulus_info
            if not isinstance(stimulus_info, Mapping):
                return {}
            repeat_times = int(stimulus_info.get("repeat_times", 1))
            if repeat_times <= 1:
                return {}
            split = SplitRepeatSignal().split_repeat_signal(
                staged.data_struct_fields["store_wave_data"],
                staged.snapshot.sample_rate,
                repeat_times=repeat_times,
            )
            return {"split_repeat_data": split}

        def save_database(info: Mapping[str, Any], stimulus: Any) -> None:
            result = self.save_database_port(info, stimulus)
            if isinstance(result, tuple) and result and result[0] != error_code.OK:
                raise RuntimeError(str(result[1]) if len(result) > 1 else "database save failed")

        def finalize_metadata(staged: StagedRecording) -> None:
            metadata = prepared.acquisition_context.get(
                "wav_calibration_metadata"
            )
            if not metadata:
                return
            appended = append_wav_calibration_metadata(
                staged.snapshot.temp_path,
                thaw_recording_session_value(metadata),
                logger=self.logger,
            )
            if not appended:
                raise RuntimeError("WAV calibration metadata append failed")

        return RecordingTransaction(
            data_struct=self.data_struct,
            alignment_handoff=alignment_handoff,
            finalize_metadata=finalize_metadata,
            save_database=save_database,
            commit_count=self.commit_count_port,
            persist_count=lambda count: self.persist_count_port(
                count,
                thaw_recording_session_value(
                    prepared.snapshot.count_persistence
                ),
            ),
            logger=self.logger,
        )

    def cancel(
        self, prepared: PreparedRecordingSession, _reason: str
    ) -> Mapping[str, Any]:
        snapshot = prepared.snapshot
        errors: list[str] = []
        try:
            if snapshot.temp_path.exists():
                snapshot.temp_path.unlink()
        except OSError as error:
            errors.append(f"temp cleanup: {error}")
        try:
            if (
                snapshot.replay
                and snapshot.backup_path is not None
                and snapshot.backup_path.exists()
                and not snapshot.output_path.exists()
            ):
                os.replace(snapshot.backup_path, snapshot.output_path)
        except OSError as error:
            errors.append(f"replay restore: {error}")
        with self._acquisition_lock:
            self._cancelled_sessions.discard(snapshot.session_id)
        return {"restored": not errors, "errors": tuple(errors)}


@dataclass(frozen=True, slots=True)
class _RecordingToken:
    token_id: str
    session_id: str
    workflow_generation: int
    controller_generation: int


@dataclass(frozen=True, slots=True)
class _AcquisitionOutcome:
    staged: StagedRecording | None = None
    error: BaseException | "_WorkerErrorDescriptor" | None = None


@dataclass(frozen=True, slots=True)
class _WorkerErrorDescriptor:
    """Frame-free error data safe to retain in a cross-thread Qt event."""

    category: str
    module: str
    name: str
    message: str
    diagnostic: str

    @staticmethod
    def capture(error: BaseException) -> "_WorkerErrorDescriptor":
        fallback = _WorkerErrorDescriptor(
            category="base-exception",
            module="builtins",
            name="BaseException",
            message="worker interrupted",
            diagnostic="worker error: worker interrupted",
        )
        try:
            is_exception = isinstance(error, Exception)
        except BaseException:
            return fallback
        category = "exception" if is_exception else "base-exception"
        name = "Exception" if is_exception else "BaseException"
        default_message = "worker failed" if is_exception else "worker interrupted"
        try:
            raw_message = str(error)
            if not isinstance(raw_message, str):
                return fallback
            message = str.__getitem__(raw_message, slice(0, 512))
            if type(message) is not str:
                return fallback
            if not message:
                message = default_message
            diagnostic = str.__add__("worker error: ", message)
            if type(diagnostic) is not str:
                return fallback
            return _WorkerErrorDescriptor(
                category=category,
                module="builtins",
                name=name,
                message=message,
                diagnostic=diagnostic,
            )
        except BaseException:
            return fallback


class _QueuedWorkerInterruption(Exception):
    """Main-thread normalization for a worker BaseException payload."""


class _BlockingResultRelay(QObject):
    finished = pyqtSignal(object)

    def __init__(
        self,
        deliver: Callable[[_AcquisitionOutcome], Any],
        parent: QObject | None,
        late_delivery: Callable[[], None] = lambda: None,
    ) -> None:
        super().__init__(parent)
        self._deliver: Callable[[_AcquisitionOutcome], Any] | None = deliver
        self._late_delivery = late_delivery
        self.queued_delivery_error: str | None = None
        self.finished.connect(self._deliver_queued, Qt.QueuedConnection)

    @pyqtSlot(object)
    def _deliver_queued(self, outcome: _AcquisitionOutcome) -> None:
        deliver = self._deliver
        if deliver is None or (
            hasattr(deliver, "is_open") and not deliver.is_open()
        ):
            self._deliver = None
            self._late_delivery()
            return
        self._deliver = None
        try:
            deliver(outcome)
        except BaseException as error:
            # A production relay is a Qt slot boundary: its delivery capability
            # owns normalization/retirement and must never unwind through Qt.
            # A deliberately injected synchronous boundary keeps the original
            # rethrow contract for direct callers.
            if getattr(deliver, "consumes_queued_base_exceptions", False):
                self.queued_delivery_error = _WorkerErrorDescriptor.capture(
                    error
                ).message
                try:
                    deliver.recover_queued_delivery_escape(error)
                except BaseException as recovery_error:
                    self.queued_delivery_error = (
                        _WorkerErrorDescriptor.capture(recovery_error).message
                    )
                return
            raise


class InlineBlockingAcquisitionJob:
    """Deterministic executor used by synchronous/controller unit tests."""

    def __init__(
        self,
        work: Callable[[], StagedRecording],
        deliver: Callable[[_AcquisitionOutcome], Any],
        request_cancel: Callable[[str], None],
    ) -> None:
        self._work = work
        self._deliver = deliver
        self._request_cancel = request_cancel

    def start(self) -> Any:
        try:
            outcome = _AcquisitionOutcome(staged=self._work())
        except BaseException as error:
            outcome = _AcquisitionOutcome(error=error)
        return self._deliver(outcome)

    def cancel(self, reason: str) -> None:
        self._request_cancel(reason)

    def is_alive(self) -> bool:
        return False


class InlineBlockingAcquisitionExecutor:
    queued_delivery = False

    def create(
        self,
        work: Callable[[], StagedRecording],
        deliver: Callable[[_AcquisitionOutcome], Any],
        request_cancel: Callable[[str], None],
    ) -> InlineBlockingAcquisitionJob:
        return InlineBlockingAcquisitionJob(work, deliver, request_cancel)


class ThreadedBlockingAcquisitionJob:
    """Run blocking audio I/O off the Qt thread and relay one queued outcome."""

    def __init__(
        self,
        work: Callable[[], StagedRecording],
        deliver: Callable[[_AcquisitionOutcome], Any],
        request_cancel: Callable[[str], None],
        parent: QObject,
    ) -> None:
        self._work = work
        self._request_cancel = request_cancel
        self.late_delivery_count = 0
        self._relay = _BlockingResultRelay(
            deliver, parent, self._record_late_delivery
        )
        self._thread = Thread(target=self._run, daemon=True)
        self.completion_delivery_error: str | None = None

    def _record_late_delivery(self) -> None:
        self.late_delivery_count += 1

    def _run(self) -> None:
        try:
            outcome = _AcquisitionOutcome(staged=self._work())
        except BaseException as error:
            # Never retain an exception/traceback/frame graph in a queued event.
            outcome = _AcquisitionOutcome(
                error=_WorkerErrorDescriptor.capture(error)
            )
        finally:
            # A completed worker no longer needs to retain external acquisition
            # or cancellation services while its result waits in the Qt queue.
            self._work = lambda: None
            self._request_cancel = lambda _reason: None
        try:
            self._relay.finished.emit(outcome)
        except RuntimeError as error:
            # Qt may destroy a relay while an owning window is being torn down.
            # The worker is already quiescent at this point, so a bounded late
            # delivery is harmless but remains inspectable for diagnostics.
            self.completion_delivery_error = str(error) or type(error).__name__

    def start(self) -> bool:
        self._thread.start()
        return True

    def cancel(self, reason: str) -> None:
        self._request_cancel(reason)

    def is_alive(self) -> bool:
        return self._thread.is_alive()


class ThreadedBlockingAcquisitionExecutor:
    queued_delivery = True

    def __init__(self, parent: QObject | None = None) -> None:
        # A parentless relay remains alive until its worker emits the one bounded
        # completion, even if the originating widget has begun Qt destruction.
        self._parent = None

    def create(
        self,
        work: Callable[[], StagedRecording],
        deliver: Callable[[_AcquisitionOutcome], Any],
        request_cancel: Callable[[str], None],
    ) -> ThreadedBlockingAcquisitionJob:
        return ThreadedBlockingAcquisitionJob(
            work,
            deliver,
            request_cancel,
            self._parent,
        )


class _RecordingSessionPhase(Enum):
    ACQUIRING = auto()
    STREAMING = auto()
    CANCELLING = auto()
    STREAMING_QUIESCING = auto()
    STREAMING_STOP_FAILED = auto()
    STREAMING_CLEANUP = auto()
    FINALIZING_PRECOMMIT = auto()
    PROMOTION_IN_PROGRESS = auto()
    PROMOTION_SUCCEEDED = auto()
    DATABASE_WORKER = auto()
    COUNT_PROJECTION = auto()
    PERSISTENCE_WORKER = auto()
    TERMINAL = auto()


class _ProducerStartState(Enum):
    NOT_STARTED = auto()
    STARTING = auto()
    STARTED = auto()
    CLOSED = auto()


@dataclass(frozen=True, slots=True)
class _DeferredStartOutcome:
    kind: str
    payload: Any


class _StartAttemptGate:
    """Session-local truth for a producer start and its first early terminal."""

    __slots__ = (
        "_accepted",
        "_deferred",
        "_lock",
        "_operation",
        "_replay_claimed",
    )

    def __init__(self) -> None:
        self._lock = Lock()
        self._deferred: _DeferredStartOutcome | None = None
        self._replay_claimed = False
        self._operation = ""
        self._accepted = False

    def reserve(self, outcome: _DeferredStartOutcome) -> bool:
        with self._lock:
            if self._deferred is not None or self._replay_claimed:
                return False
            self._deferred = outcome
            return True

    def claim_deferred(self) -> _DeferredStartOutcome | None:
        with self._lock:
            if self._replay_claimed or self._deferred is None:
                return None
            self._replay_claimed = True
            return self._deferred

    def has_unresolved_deferred(self) -> bool:
        with self._lock:
            return self._deferred is not None and not self._operation

    def resolve(self, operation: str, accepted: bool) -> None:
        with self._lock:
            self._operation = operation
            self._accepted = bool(accepted)

    def truthful_result(self) -> bool | None:
        with self._lock:
            if not self._operation:
                return None
            return bool(
                self._accepted and self._operation in {"completed", "staged"}
            )


@dataclass(slots=True)
class _ActiveRecordingSession:
    token: _RecordingToken
    prepared: PreparedRecordingSession
    terminal: "RecordingTerminalPort"
    phase: _RecordingSessionPhase
    worker_gate: "_TerminalLifecycleGate"
    blocking_job: Any = None
    finalization_job: Any = None
    transaction: RecordingTransaction | None = None
    staged: StagedRecording | None = None
    durable: RecordingDurableResult | None = None
    streaming_handle: Any = None
    streaming_admission_closed: bool = False
    streaming_stop_acknowledged: bool = False
    streaming_disconnect_retry_started: bool = False
    cancellation_requested: bool = False
    cancellation_reason: str = ""
    disconnect_requested: bool = False
    suppress_terminal: bool = False
    producer_start_state: _ProducerStartState = _ProducerStartState.NOT_STARTED
    producer_kind: str = ""
    cancellation_reports_failure: bool = False
    streaming_failure_rollback: Any = None
    deferred_start_failure_reason: str = ""
    start_attempt_gate: _StartAttemptGate = field(default_factory=_StartAttemptGate)


@dataclass(frozen=True, slots=True)
class _TerminalRequest:
    token: _RecordingToken
    operation: str
    payload: tuple[Any, ...]


class _TerminalRelay(QObject):
    requested = pyqtSignal(object)

    def __init__(
        self,
        owner: "SequenceRecordingController",
        controller_gate: "_TerminalLifecycleGate",
        session_gate: "_TerminalLifecycleGate",
    ) -> None:
        # This relay is deliberately parentless.  Producer ports retain it, so
        # emitting can never dereference a relay deleted with the controller.
        super().__init__(None)
        self._owner_ref = ref(owner)
        self._controller_gate = controller_gate
        self._session_gate = session_gate
        self.queued_delivery_error: str | None = None
        self.requested.connect(self._deliver, Qt.QueuedConnection)

    def submit(self, request: _TerminalRequest) -> bool:
        if not self.delivery_is_open():
            return False
        try:
            self.requested.emit(request)
        except (AttributeError, RuntimeError):
            return False
        return True

    def delivery_is_open(self) -> bool:
        return self._controller_gate.is_open() and self._session_gate.is_open()

    @pyqtSlot(object)
    def _deliver(self, request: _TerminalRequest) -> None:
        if not self.delivery_is_open():
            return
        owner = self._owner_ref()
        if owner is not None:
            try:
                owner._deliver_queued_terminal_request(request)
            except (AttributeError, RuntimeError) as error:
                # A concurrent native QObject teardown closes the lifecycle
                # gate, but a delivery already executing must still be benign.
                if not self.delivery_is_open():
                    return
                self.queued_delivery_error = _WorkerErrorDescriptor.capture(
                    error
                ).message
                try:
                    owner._recover_queued_terminal_escape(request, error)
                except BaseException as recovery_error:
                    self.queued_delivery_error = (
                        _WorkerErrorDescriptor.capture(recovery_error).message
                    )
            except BaseException as error:
                # Last-resort Qt slot containment. The controller route above
                # normally consumes this after one truthful retirement.
                self.queued_delivery_error = _WorkerErrorDescriptor.capture(
                    error
                ).message
                try:
                    owner._recover_queued_terminal_escape(request, error)
                except BaseException as recovery_error:
                    self.queued_delivery_error = (
                        _WorkerErrorDescriptor.capture(recovery_error).message
                    )


class _TerminalLifecycleGate:
    """Pure-Python one-request admission gate retained safely by producer ports."""

    __slots__ = ("_lock", "_open", "_reserved", "__weakref__")

    def __init__(self) -> None:
        self._lock = Lock()
        self._open = True
        self._reserved = False

    def reserve(self) -> bool:
        with self._lock:
            if not self._open or self._reserved:
                return False
            self._reserved = True
            return True

    def close(self) -> None:
        with self._lock:
            self._open = False

    def is_open(self) -> bool:
        with self._lock:
            return self._open


@dataclass(frozen=True, slots=True)
class _PendingRecordingAdmission:
    reservation_id: str
    command_identity: tuple[str, str]
    session_id: str
    workflow_generation: int | None
    controller_generation: int
    gate: _TerminalLifecycleGate


class _TerminalSubmissionBoundary:
    """Use direct delivery only on the admission thread; workers only emit."""

    __slots__ = ("_relay", "_owner_ref", "_owner_thread_id")

    def __init__(
        self,
        relay: _TerminalRelay,
        owner: "SequenceRecordingController",
        owner_thread_id: int,
    ) -> None:
        self._relay = relay
        self._owner_ref = ref(owner)
        self._owner_thread_id = owner_thread_id

    def submit(self, request: _TerminalRequest) -> bool:
        if not self._relay.delivery_is_open():
            return False
        if get_ident() == self._owner_thread_id:
            owner = self._owner_ref()
            if owner is None:
                return False
            try:
                return bool(owner._deliver_terminal_request(request))
            except (AttributeError, RuntimeError):
                return False
        return self._relay.submit(request)


class _WorkerResultBoundary:
    """Weak, generation-gated route from a worker relay to one controller slot."""

    __slots__ = (
        "_controller_gate",
        "_extra_args",
        "_handler_name",
        "_owner_ref",
        "_queued_delivery",
        "_session_gate",
        "_token",
    )

    def __init__(
        self,
        owner: "SequenceRecordingController",
        token: _RecordingToken,
        handler_name: str,
        controller_gate: _TerminalLifecycleGate,
        session_gate: _TerminalLifecycleGate,
        queued_delivery: bool,
        extra_args: tuple[Any, ...] = (),
    ) -> None:
        self._owner_ref = ref(owner)
        self._token = token
        self._handler_name = handler_name
        self._controller_gate = controller_gate
        self._session_gate = session_gate
        self._queued_delivery = queued_delivery
        self._extra_args = extra_args

    @property
    def consumes_queued_base_exceptions(self) -> bool:
        return self._queued_delivery

    def is_open(self) -> bool:
        return self._controller_gate.is_open() and self._session_gate.is_open()

    def recover_queued_delivery_escape(self, error: BaseException) -> bool:
        if not self._queued_delivery:
            return False
        owner = self._owner_ref()
        if owner is None:
            return False
        return owner._recover_queued_worker_delivery_escape(
            self._token, self._handler_name, error
        )

    def __call__(self, outcome: _AcquisitionOutcome) -> Any:
        if not self.is_open():
            return False
        owner = self._owner_ref()
        if owner is None or not self.is_open():
            return False
        if self._queued_delivery and isinstance(
            outcome.error, _WorkerErrorDescriptor
        ):
            outcome = _AcquisitionOutcome(
                staged=outcome.staged,
                error=_QueuedWorkerInterruption(outcome.error.message),
            )
        elif self._queued_delivery and outcome.error is not None and not isinstance(
            outcome.error, Exception
        ):
            # Custom queued executors may still supply a legacy raw outcome.
            # Normalize at the last boundary while built-in threaded executors
            # guarantee that no such object ever enters the Qt queue.
            outcome = _AcquisitionOutcome(
                staged=outcome.staged,
                error=_QueuedWorkerInterruption(
                    _WorkerErrorDescriptor.capture(outcome.error).message
                ),
            )
        try:
            return getattr(owner, self._handler_name)(
                self._token, outcome, *self._extra_args
            )
        except BaseException as error:
            if not self._queued_delivery:
                raise
            return owner._consume_queued_worker_delivery_escape(
                self._token, self._handler_name, error
            )


class _WorkerTransactionCheckpointPort:
    """Worker-safe transaction checkpoints retaining no controller graph."""

    __slots__ = (
        "_controller_gate",
        "_owner_ref",
        "_session_gate",
        "_token",
    )

    def __init__(
        self,
        owner: "SequenceRecordingController",
        token: _RecordingToken,
        controller_gate: _TerminalLifecycleGate,
        session_gate: _TerminalLifecycleGate,
    ) -> None:
        self._owner_ref = ref(owner)
        self._token = token
        self._controller_gate = controller_gate
        self._session_gate = session_gate

    def _owner(self) -> "SequenceRecordingController":
        if not self._controller_gate.is_open() or not self._session_gate.is_open():
            raise RecordingCancellationRequested(
                "recording controller lifecycle ended"
            )
        owner = self._owner_ref()
        if owner is None:
            raise RecordingCancellationRequested(
                "recording controller lifecycle ended"
            )
        return owner

    def cancellation_checkpoint(self) -> None:
        self._owner()._check_precommit_cancellation(self._token)

    def begin_durable_commit(self) -> None:
        self._owner()._begin_durable_commit(self._token)

    def promotion_succeeded(self) -> None:
        try:
            self._owner()._mark_promotion_succeeded(self._token)
        except RecordingCancellationRequested as error:
            raise RuntimeError(str(error)) from error


class RecordingTerminalPort:
    """Terminal capability bound to exactly one admitted session."""

    __slots__ = ("_boundary", "_gate", "_session_snapshot", "_token")

    def __init__(
        self,
        token: _RecordingToken,
        session_snapshot: Any,
        boundary: _TerminalSubmissionBoundary,
        gate: _TerminalLifecycleGate,
    ) -> None:
        self._token = token
        self._session_snapshot = immutable_recording_value(session_snapshot)
        self._boundary = boundary
        self._gate = gate

    @property
    def session_id(self) -> str:
        return self._token.session_id

    @property
    def workflow_generation(self) -> int:
        return self._token.workflow_generation

    @property
    def session_snapshot(self) -> Any:
        return self._session_snapshot

    def close(self) -> None:
        self._gate.close()

    def _submit(self, operation: str, *payload: Any) -> bool:
        if not self._gate.reserve():
            return False
        accepted = self._boundary.submit(
            _TerminalRequest(self._token, operation, tuple(payload))
        )
        if not accepted:
            self._gate.close()
        return accepted

    def staged_recording_ready(self, staged: StagedRecording) -> bool:
        return self._submit("staged", staged)

    def recording_completed(self, *, sample_count: int, result_snapshot: Any) -> bool:
        return self._submit(
            "completed",
            sample_count,
            immutable_recording_value(result_snapshot),
        )

    def recording_failed(
        self, reason: Any, rollback_outcome: Any = None
    ) -> bool:
        return self._submit(
            "failed",
            str(reason) or "recording failed",
            immutable_recording_value(rollback_outcome),
        )

    def streaming_consumer_failed(
        self,
        reason: Any,
        rollback_outcome: Any,
        producer_quiesced: bool,
        shutdown_diagnostic: Any = "",
    ) -> bool:
        return self._submit(
            "streaming-consumer-failed",
            str(reason) or "streaming consumer failed",
            immutable_recording_value(rollback_outcome),
            bool(producer_quiesced),
            str(shutdown_diagnostic or ""),
        )

    def recording_cancelled(self, reason: Any) -> bool:
        return self._submit("cancelled", str(reason) or "recording cancelled")


class _QueuedCommandGuard(QObject):
    def __init__(
        self, owner: "SequenceRecordingController", handler_name: str
    ) -> None:
        super().__init__(owner)
        self._owner_ref = ref(owner)
        self._handler_name = handler_name
        self.queued_delivery_error: str | None = None

    @pyqtSlot(object)
    def deliver(self, command: Any) -> None:
        owner = self._owner_ref()
        if owner is not None and owner._accept_queued_delivery:
            try:
                owner._deliver_queued_command(self._handler_name, command)
            except BaseException as error:
                # A queue entry slot is the final native boundary even if a
                # secondary diagnostic callback itself is defective.
                self.queued_delivery_error = _WorkerErrorDescriptor.capture(
                    error
                ).message
                try:
                    owner._recover_queued_command_escape(
                        self._handler_name, command, error
                    )
                except BaseException as recovery_error:
                    self.queued_delivery_error = (
                        _WorkerErrorDescriptor.capture(recovery_error).message
                    )


class _MarkActionQtOwnerCallState(Enum):
    PENDING = auto()
    DELIVERING = auto()
    COMPLETED = auto()
    CANCELLED = auto()


@dataclass(slots=True, weakref_slot=True)
class _MarkActionQtOwnerCall:
    token: str
    operation: str
    command: Any = None
    completion: Event = field(default_factory=Event)
    result: bool = False
    error: BaseException | None = None
    state: _MarkActionQtOwnerCallState = _MarkActionQtOwnerCallState.PENDING


@dataclass(frozen=True, slots=True)
class RecordingMarkActionDispatchTerminal:
    """Exact terminal outcome of one asynchronously accepted Qt-owner call."""

    token: str
    result: bool
    error: BaseException | None = field(default=None, repr=False, compare=False)

    def unwrap(self) -> bool:
        if self.error is not None:
            raise self.error
        return self.result


@dataclass(frozen=True, slots=True)
class RecordingMarkActionDispatchPending:
    """Bounded handoff result for a call already claimed by the Qt owner."""

    _call: _MarkActionQtOwnerCall = field(repr=False, compare=False)

    @property
    def token(self) -> str:
        return self._call.token

    @property
    def done(self) -> bool:
        return self._call.completion.is_set()

    def terminal(
        self,
        timeout: float = 0.0,
    ) -> RecordingMarkActionDispatchTerminal | None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout < 0
        ):
            raise ValueError("timeout must be a finite non-negative number")
        if not self._call.completion.wait(float(timeout)):
            return None
        return RecordingMarkActionDispatchTerminal(
            self._call.token,
            self._call.result,
            self._call.error,
        )

    def __bool__(self) -> bool:
        raise TypeError(
            "pending mark-action dispatch has no boolean result; "
            "inspect terminal()"
        )


class _MarkActionQtOwnerDispatchRegistry:
    """Python-owned exact-token handoff state surviving QObject deletion."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._calls: dict[str, _MarkActionQtOwnerCall] = {}

    def snapshot(self) -> tuple[_MarkActionQtOwnerCall, ...]:
        with self._lock:
            return tuple(self._calls.values())

    def register(self, call: _MarkActionQtOwnerCall) -> bool:
        with self._lock:
            if self._calls:
                return False
            self._calls[call.token] = call
            return True

    def claim(self, token: str) -> _MarkActionQtOwnerCall | None:
        with self._lock:
            call = self._calls.get(token)
            if (
                call is None
                or call.state is not _MarkActionQtOwnerCallState.PENDING
            ):
                return None
            call.state = _MarkActionQtOwnerCallState.DELIVERING
            return call

    def complete(
        self,
        call: _MarkActionQtOwnerCall,
        *,
        result: bool,
        error: BaseException | None,
    ) -> bool:
        with self._lock:
            if (
                self._calls.get(call.token) is not call
                or call.state is not _MarkActionQtOwnerCallState.DELIVERING
            ):
                return False
            call.result = bool(result)
            call.error = error
            call.state = _MarkActionQtOwnerCallState.COMPLETED
            del self._calls[call.token]
            call.completion.set()
            return True

    def cancel(
        self,
        call: _MarkActionQtOwnerCall,
    ) -> bool:
        with self._lock:
            if self._calls.get(call.token) is not call:
                return False
            if call.state is not _MarkActionQtOwnerCallState.PENDING:
                return False
            call.result = False
            call.error = None
            call.state = _MarkActionQtOwnerCallState.CANCELLED
            del self._calls[call.token]
            call.completion.set()
            return True

    def cancel_all(self, _lifecycle_event: Any = None) -> None:
        with self._lock:
            for token, call in tuple(self._calls.items()):
                if call.state is _MarkActionQtOwnerCallState.PENDING:
                    call.result = False
                    call.error = None
                    call.state = _MarkActionQtOwnerCallState.CANCELLED
                    del self._calls[token]
                    call.completion.set()

    def settle_owner_thread_finished(self, _lifecycle_event: Any = None) -> None:
        with self._lock:
            for token, call in tuple(self._calls.items()):
                if call.state is _MarkActionQtOwnerCallState.PENDING:
                    call.result = False
                    call.error = None
                elif call.state is _MarkActionQtOwnerCallState.DELIVERING:
                    call.result = False
                    call.error = RuntimeError(
                        "mark-action Qt owner thread ended during delivery"
                    )
                else:
                    continue
                call.state = _MarkActionQtOwnerCallState.CANCELLED
                del self._calls[token]
                call.completion.set()


class SequenceRecordingController(QObject):
    """Consume admitted starts and publish recording terminal events only."""

    disconnect_quiesced = pyqtSignal(str)
    streaming_quiescence_failed = pyqtSignal(str, str)
    _mark_action_qt_owner_requested = pyqtSignal(object)

    def __init__(
        self,
        model: RecordingModel,
        bus: SequenceEventBus,
        *,
        view: SequenceRecordingView | None = None,
        label_service: Any = None,
        mark_action_service: Any = None,
        manual_label_request_service: Any = None,
        import_view: Any = None,
        import_runtime: Any = None,
        import_service: Any = None,
        import_workflow_identity_provider: Callable[[], Mapping[str, Any]] | None = None,
        prepare_session: Callable[
            [BeginRecordingRequested], PreparedRecordingSession
        ] | None = None,
        blocking_acquirer: Callable[
            [PreparedRecordingSession], StagedRecording
        ] | None = None,
        transaction_factory: Callable[
            [PreparedRecordingSession], RecordingTransaction
        ] | None = None,
        use_streaming: Callable[[PreparedRecordingSession], bool] = lambda _p: False,
        streaming_adapter: Callable[
            [PreparedRecordingSession, RecordingTerminalPort], bool
        ] | None = None,
        request_blocking_cancel: Callable[
            [PreparedRecordingSession, str], None
        ] | None = None,
        close_streaming_admission: Callable[[PreparedRecordingSession], Any]
        | None = None,
        quiesce_streaming: Callable[
            [PreparedRecordingSession, str, Any], Any
        ]
        | None = None,
        cancel_adapter: Callable[[PreparedRecordingSession, str], Any] | None = None,
        blocking_executor: Any = None,
        finalization_executor: Any = None,
        workflow_generation_provider: Callable[[], int] | None = None,
        logger: Any = None,
        recent_identity_limit: int = 256,
        parent: QObject | None = None,
        connect_queued: bool = True,
    ) -> None:
        super().__init__(parent)
        if type(recent_identity_limit) is not int or recent_identity_limit < 1:
            raise ValueError("recent_identity_limit must be a positive integer")
        self.model = model
        self.bus = bus
        bind_recording_admission_consumer = getattr(
            self.bus,
            "_bind_canonical_recording_consumer",
            None,
        )
        self._canonical_recording_admission_capability = (
            bind_recording_admission_consumer(self)
            if callable(bind_recording_admission_consumer)
            else None
        )
        self.view = view or SequenceRecordingView(logger=logger, parent=self)
        self.label_service = label_service
        self.mark_action_service = mark_action_service
        self.manual_label_request_service = manual_label_request_service
        self.import_controller = SequenceRecordingImportController(
            model,
            import_view,
            bus=bus,
            runtime=import_runtime,
            workflow_identity_provider=import_workflow_identity_provider,
            import_service=import_service,
            logger=logger,
            parent=self,
        )
        self.prepare_session = prepare_session
        self.blocking_acquirer = blocking_acquirer
        self.transaction_factory = transaction_factory
        self.use_streaming = use_streaming
        self.streaming_adapter = streaming_adapter
        self.request_blocking_cancel = request_blocking_cancel or (
            lambda _prepared, _reason: None
        )
        self.close_streaming_admission = close_streaming_admission or (
            lambda _prepared: None
        )
        self.quiesce_streaming = quiesce_streaming or (
            lambda _prepared, _reason, _handle: {"quiesced": True}
        )
        self.cancel_adapter = cancel_adapter or (
            lambda _prepared, _reason: {"restored": True, "errors": ()}
        )
        self.workflow_generation_provider = workflow_generation_provider
        self.logger = logger
        self._active = True
        self._disconnecting = False
        self._controller_generation = 0
        self._lock = RLock()
        self._session: _ActiveRecordingSession | None = None
        self.blocking_executor = blocking_executor or (
            ThreadedBlockingAcquisitionExecutor()
            if connect_queued
            else InlineBlockingAcquisitionExecutor()
        )
        self.finalization_executor = finalization_executor or (
            ThreadedBlockingAcquisitionExecutor()
            if connect_queued
            else InlineBlockingAcquisitionExecutor()
        )
        self._recent_identity_limit = recent_identity_limit
        self._recent_identities: set[tuple[str, str]] = set()
        self._recent_identity_order: deque[tuple[str, str]] = deque()
        self._published_terminal_identities: set[tuple[str, str]] = set()
        self._queued_command_failure_identities: set[tuple[str, str]] = set()
        self._label_terminal_commands: OrderedDict[str, None] = OrderedDict()
        self._label_recovery_command_id: str | None = None
        self._label_recovery_command_identity: tuple[Any, ...] | None = None
        self._label_inflight_command_id: str | None = None
        self._label_inflight_command_identity: tuple[Any, ...] | None = None
        self._mark_action_inflight_identity: tuple[str, int] | None = None
        self._mark_action_terminal_identities: OrderedDict[
            tuple[str, int], None
        ] = OrderedDict()
        self._mark_blocked_admission_terminal_identities: OrderedDict[
            tuple[str, int], None
        ] = OrderedDict()
        self._mark_disconnect_quiescence_pending = False
        self._mark_action_qt_owner_dispatch = (
            _MarkActionQtOwnerDispatchRegistry()
        )
        self._mark_action_qt_owner_requested.connect(
            self._run_mark_action_qt_owner_call,
            Qt.QueuedConnection,
        )
        self._active_command_identity: tuple[str, str] | None = None
        self._pending_admission: _PendingRecordingAdmission | None = None
        self._validated_workflow_generation: int | None = None
        self._accept_queued_delivery = bool(connect_queued)
        self._queued_command_depth = 0
        self._controller_terminal_gate = _TerminalLifecycleGate()
        self._controller_worker_gate = _TerminalLifecycleGate()
        self.destroyed.connect(self._controller_terminal_gate.close)
        self.destroyed.connect(self._controller_worker_gate.close)
        self.destroyed.connect(
            self._mark_action_qt_owner_dispatch.cancel_all,
            Qt.DirectConnection,
        )
        owner_thread = self.thread()
        if owner_thread is not None:
            owner_thread.finished.connect(
                self._mark_action_qt_owner_dispatch.settle_owner_thread_finished,
                Qt.DirectConnection,
            )
        self._queued_worker_delivery = bool(connect_queued)
        self._begin_guard = _QueuedCommandGuard(
            self, "handle_begin_recording"
        )
        self._cancel_guard = _QueuedCommandGuard(
            self, "handle_cancel_recording"
        )
        self._guard = self._begin_guard
        if connect_queued:
            bus.commands.begin_recording_requested.connect(
                self._begin_guard.deliver, Qt.QueuedConnection
            )
            bus.commands.cancel_recording_requested.connect(
                self._cancel_guard.deliver, Qt.QueuedConnection
            )
            if self.label_service is not None:
                self._label_recipient_name = f"recording-label:{id(self)}"
                bus.register_workflow_continuation_recipient(
                    "label-commit",
                    self._label_recipient_name,
                    self.handle_commit_recording_label,
                    owner=self,
                )
            else:
                self._label_recipient_name = None
        else:
            self._label_recipient_name = None
        if connect_queued:
            self._replay_recording_admission()

    @property
    def recent_identity_count(self) -> int:
        return len(self._recent_identities)

    @property
    def streaming_cancellation_pending(self) -> bool:
        with self._lock:
            session = self._session
            return bool(
                session is not None
                and session.cancellation_requested
                and session.phase
                in {
                    _RecordingSessionPhase.STREAMING_QUIESCING,
                    _RecordingSessionPhase.STREAMING_STOP_FAILED,
                    _RecordingSessionPhase.STREAMING_CLEANUP,
                }
            )

    def _log(self, level: str, message: str) -> None:
        try:
            callback = getattr(self.logger, level, None)
            if callable(callback):
                callback(message)
        except BaseException:
            # Diagnostics are best-effort and may never interrupt recording
            # cleanup, retirement, or a caller's original exception path.
            return

    @pyqtSlot(object)
    def handle_load_imported_audio_requested(
        self, command: LoadImportedAudioRequested
    ) -> bool:
        return self.import_controller.handle_load_imported_audio_requested(command)

    @pyqtSlot(object)
    def handle_cancel_imported_audio_requested(
        self, command: CancelImportedAudioRequested
    ) -> bool:
        return self.import_controller.handle_cancel_imported_audio_requested(command)

    def retry_pending_import_terminal(self, import_id: str) -> bool:
        return self.import_controller.retry_pending_import_terminal(import_id)

    def handle_import_terminal_retry_timeout(self) -> bool:
        return self.import_controller.handle_import_terminal_retry_timeout()

    @property
    def pending_import_terminal_identity(self) -> str | None:
        return self.import_controller.pending_import_terminal_identity

    @property
    def import_terminal_retry_timer(self) -> Any:
        return self.import_controller.import_terminal_retry_timer

    @property
    def import_terminal_retry_delay_ms(self) -> int:
        return self.import_controller.import_terminal_retry_delay_ms

    @property
    def import_terminal_retry_max_delay_ms(self) -> int:
        return self.import_controller.import_terminal_retry_max_delay_ms

    def disconnect(self, _lifecycle_request=None) -> None:
        self.import_controller.disconnect(_lifecycle_request)
        self._mark_action_qt_owner_dispatch.cancel_all()
        with self._lock:
            self._accept_queued_delivery = False
        self._release_recording_admission_capability()
        if self._label_recipient_name is not None:
            self.bus.unregister_workflow_continuation_recipient(
                "label-commit",
                self._label_recipient_name,
                self.handle_commit_recording_label,
            )
            self._label_recipient_name = None
        job = None
        streaming_token: _RecordingToken | None = None
        immediately_quiesced = False
        fatal_cancel_error: BaseException | None = None
        repeated_disconnect = False
        with self._lock:
            if not self._active:
                return
            if self._disconnecting:
                session = self._session
                if (
                    session is None
                    or session.phase is not _RecordingSessionPhase.STREAMING_STOP_FAILED
                ):
                    return
                streaming_token = session.token
                repeated_disconnect = True
            else:
                self._disconnecting = True
                self._accept_queued_delivery = False
                self._controller_terminal_gate.close()
                pending = self._pending_admission
                if pending is not None:
                    self._clear_pending_admission_locked(pending)
                session = self._session
                if session is None:
                    self._active = False
                    self._controller_generation += 1
                    mark_recovery_pending = (
                        getattr(self.mark_action_service, "pending_recovery", None)
                        is not None
                    )
                    if (
                        self._mark_action_inflight_identity is None
                        and not mark_recovery_pending
                    ):
                        immediately_quiesced = True
                    else:
                        self._mark_disconnect_quiescence_pending = True
                else:
                    session.disconnect_requested = True
                    session.suppress_terminal = True
                    session.terminal.close()
                terminal_already_won = bool(
                    session is not None
                    and session.producer_start_state is _ProducerStartState.STARTING
                    and session.start_attempt_gate.has_unresolved_deferred()
                )
                if (
                    session is not None
                    and not session.cancellation_requested
                    and not terminal_already_won
                ):
                    session.cancellation_requested = True
                    session.cancellation_reason = "disconnect"
                    if (
                        session.producer_start_state
                        is _ProducerStartState.NOT_STARTED
                    ):
                        session.producer_start_state = _ProducerStartState.CLOSED
                    if session.phase is _RecordingSessionPhase.ACQUIRING:
                        session.phase = _RecordingSessionPhase.CANCELLING
                        if (
                            session.producer_start_state
                            is not _ProducerStartState.STARTING
                        ):
                            job = session.blocking_job
                    elif session.phase is _RecordingSessionPhase.STREAMING:
                        streaming_token = session.token
                elif session is not None and session.phase is _RecordingSessionPhase.CANCELLING:
                    job = session.blocking_job
                elif (
                    session is not None
                    and session.phase is _RecordingSessionPhase.STREAMING_STOP_FAILED
                ):
                    session.streaming_disconnect_retry_started = True
                    streaming_token = session.token
        if repeated_disconnect:
            self._start_streaming_quiescence(streaming_token)
            return
        if job is not None:
            try:
                job.cancel("disconnect")
            except Exception as error:
                self._log("warning", f"recording disconnect producer stop failed: {error}")
            except BaseException as error:
                # State remains cancellation-pending and cleanup is still owned by
                # the producer's eventual completion continuation.
                fatal_cancel_error = error
        if streaming_token is not None:
            self._start_streaming_quiescence(streaming_token)
        try:
            self.bus.commands.begin_recording_requested.disconnect(
                self._begin_guard.deliver
            )
        except (RuntimeError, TypeError):
            pass
        try:
            self.bus.commands.cancel_recording_requested.disconnect(
                self._cancel_guard.deliver
            )
        except (RuntimeError, TypeError):
            pass
        if immediately_quiesced:
            self.disconnect_quiesced.emit("")
        if fatal_cancel_error is not None:
            raise fatal_cancel_error

    @staticmethod
    def _admission_generation(command: BeginRecordingRequested) -> int | None:
        snapshot = command.session_snapshot
        if not isinstance(snapshot, Mapping):
            return None
        generation = snapshot.get("workflow_generation")
        return generation if type(generation) is int else None

    def _current_generation(self) -> int | None:
        if self.workflow_generation_provider is None:
            return None
        generation = self.workflow_generation_provider()
        return generation if type(generation) is int else None

    def _token_matches_frozen_session_locked(self, token: _RecordingToken) -> bool:
        session = self._session
        if session is None or token != session.token:
            return False
        if not self._active:
            return False
        if token.controller_generation != self._controller_generation:
            return False
        return True

    def _token_is_current_locked(self, token: _RecordingToken) -> bool:
        if not self._token_matches_frozen_session_locked(token):
            return False
        return token.workflow_generation == self._validated_workflow_generation

    def _deliver_queued_command(self, handler_name: str, command: Any) -> bool:
        self._queued_command_depth += 1
        try:
            return bool(getattr(self, handler_name)(command))
        except BaseException as error:
            return self._consume_queued_command_escape(
                handler_name, command, error
            )
        finally:
            self._queued_command_depth -= 1

    def _consume_queued_command_escape(
        self, handler_name: str, command: Any, error: BaseException
    ) -> bool:
        reason = _WorkerErrorDescriptor.capture(error).message
        self._log(
            "error",
            f"recording queued {handler_name} command interrupted: {reason}",
        )
        return self._recover_queued_command_escape(
            handler_name, command, error
        )

    def _recover_queued_command_escape(
        self, handler_name: str, command: Any, error: BaseException
    ) -> bool:
        reason = _WorkerErrorDescriptor.capture(error).message
        if handler_name != "handle_begin_recording":
            # Cancellation remains pending until its producer result or explicit
            # streaming acknowledgement establishes quiescence.
            return False
        session_id = getattr(command, "session_id", "")
        identity = (getattr(command, "command_id", ""), session_id)
        defer_until_producer_ack = False
        with self._lock:
            session = self._session
            token = None if session is None else session.token
            if token is not None and token.session_id != session_id:
                token = None
            already_published = any(
                published_identity[1] == session_id
                for published_identity in self._published_terminal_identities
            ) or identity in self._queued_command_failure_identities
            if not already_published:
                self._remember_identity(identity)
                self._queued_command_failure_identities.add(identity)
            if (
                token is not None
                and session is not None
                and session.producer_start_state
                in {
                    _ProducerStartState.STARTING,
                    _ProducerStartState.STARTED,
                }
                and (
                    session.cancellation_requested
                    or session.disconnect_requested
                )
            ):
                session.deferred_start_failure_reason = reason
                defer_until_producer_ack = True
        if defer_until_producer_ack:
            return False
        if token is not None:
            self._interrupt_active_session(token, error, cleanup=True)
        if already_published:
            return False
        if self.model.active_session_id == session_id:
            self.model.fail()
        self.bus.events.recording_failed.emit(
            RecordingFailed(session_id, reason or "recording command interrupted")
        )
        return False

    def _in_queued_command_delivery(self) -> bool:
        return self._queued_command_depth > 0

    def _deliver_queued_terminal_request(
        self, request: _TerminalRequest
    ) -> bool:
        try:
            return self._deliver_terminal_request(request)
        except BaseException as error:
            return self._consume_queued_terminal_escape(request, error)

    def _consume_queued_terminal_escape(
        self, request: _TerminalRequest, error: BaseException
    ) -> bool:
        reason = _WorkerErrorDescriptor.capture(error).message
        self._log(
            "error",
            f"recording queued terminal {request.operation} interrupted: {reason}",
        )
        return self._recover_queued_terminal_escape(request, error)

    def _recover_queued_terminal_escape(
        self, request: _TerminalRequest, error: BaseException
    ) -> bool:
        token = request.token
        reason = _WorkerErrorDescriptor.capture(error).message
        identity = (token.token_id, token.session_id)
        with self._lock:
            already_published = identity in self._published_terminal_identities
            current = self._token_matches_frozen_session_locked(token)
        if current:
            self._interrupt_active_session(token, error, cleanup=True)
        if already_published:
            return False
        self._mark_terminal_event_published(token)
        if self.model.active_session_id == token.session_id:
            self.model.fail()
        self.bus.events.recording_failed.emit(
            RecordingFailed(token.session_id, reason)
        )
        return False

    def _mark_terminal_event_published(self, token: _RecordingToken) -> None:
        with self._lock:
            self._published_terminal_identities.add(
                (token.token_id, token.session_id)
            )

    def _deliver_terminal_request(self, request: _TerminalRequest) -> bool:
        with self._lock:
            if not self._token_is_current_locked(request.token):
                self._log(
                    "debug",
                    f"ignored late recording terminal request: {request.operation}",
                )
                return False
            session = self._session
            if (
                session is not None
                and session.producer_start_state is _ProducerStartState.STARTING
            ):
                return session.start_attempt_gate.reserve(
                    _DeferredStartOutcome("terminal", request)
                )
        return self._dispatch_terminal_request(request)

    def _dispatch_terminal_request(self, request: _TerminalRequest) -> bool:
        if request.operation == "staged":
            return self._commit_terminal(request.token, request.payload[0])
        elif request.operation == "completed":
            return self._compatibility_completed(
                request.token,
                sample_count=request.payload[0],
                result_snapshot=request.payload[1],
            )
        elif request.operation == "failed":
            return self._fail_terminal(
                request.token,
                request.payload[0],
                request.payload[1] if len(request.payload) > 1 else None,
            )
        elif request.operation == "streaming-consumer-failed":
            return self._request_streaming_failure(
                request.token,
                request.payload[0],
                request.payload[1],
                request.payload[2],
                request.payload[3],
            )
        elif request.operation == "cancelled":
            return self._cancel_terminal(request.token, request.payload[0])
        return False

    def _remember_identity(
        self, identity: tuple[str, str], *, refresh: bool = False
    ) -> None:
        if identity in self._recent_identities:
            if not refresh:
                return
            self._recent_identity_order.remove(identity)
        else:
            self._recent_identities.add(identity)
        self._recent_identity_order.append(identity)
        while len(self._recent_identity_order) > self._recent_identity_limit:
            retired = self._recent_identity_order.popleft()
            self._recent_identities.discard(retired)
            self._published_terminal_identities.discard(retired)
            self._queued_command_failure_identities.discard(retired)

    def _forget_identity_locked(self, identity: tuple[str, str]) -> None:
        self._recent_identities.discard(identity)
        try:
            self._recent_identity_order.remove(identity)
        except ValueError:
            pass
        self._published_terminal_identities.discard(identity)
        self._queued_command_failure_identities.discard(identity)

    def _has_recent_identity_locked(self, identity: tuple[str, str]) -> bool:
        return bool(
            identity in self._recent_identities
            or identity == self._active_command_identity
        )

    def _clear_pending_admission_locked(
        self, reservation: _PendingRecordingAdmission | None
    ) -> bool:
        if reservation is None or self._pending_admission is not reservation:
            return False
        self._remember_identity(reservation.command_identity, refresh=True)
        self._pending_admission = None
        reservation.gate.close()
        return True

    def _pending_admission_is_current_locked(
        self, reservation: _PendingRecordingAdmission
    ) -> bool:
        return bool(
            self._pending_admission is reservation
            and reservation.gate.is_open()
            and self._controller_terminal_gate.is_open()
            and self._controller_worker_gate.is_open()
            and self._active
            and not self._disconnecting
            and reservation.controller_generation == self._controller_generation
            and self._session is None
            and self._active_command_identity is None
        )

    def _abort_invalidated_preparation(
        self,
        prepared: PreparedRecordingSession,
        reason: str = "recording admission invalidated",
    ) -> None:
        try:
            self.cancel_adapter(prepared, reason)
        except BaseException as error:
            # The reservation is already closed and no model/session state was
            # installed. Resource cleanup is best-effort at this adapter boundary.
            reason = _WorkerErrorDescriptor.capture(error).message
            self._log(
                "warning",
                f"recording invalidated preparation cleanup failed: {reason}",
            )

    def _restore_model_checkpoint(
        self,
        checkpoint: tuple[Any, Any, Any, Any],
        original_error: BaseException,
    ) -> None:
        try:
            self.model.fail()
        except BaseException as recovery_error:
            original_error.add_note(
                f"recording model rollback failed: {recovery_error}"
            )
        finally:
            (
                self.model.state,
                self.model.active_session_id,
                self.model.active_snapshot,
                self.model.final_result,
            ) = checkpoint

    @staticmethod
    def _close_admission_resource(
        resource: Any,
        original_error: BaseException,
    ) -> None:
        if resource is None:
            return
        try:
            resource.close()
        except BaseException as close_error:
            original_error.add_note(
                f"recording admission resource close failed: {close_error}"
            )

    def _continue_unstarted_session(
        self,
        token: _RecordingToken,
        session: _ActiveRecordingSession,
    ) -> bool:
        cancellation_reason = ""
        cancellation_pending = False
        with self._lock:
            if self._session is not session or session.token != token:
                return False
            if (
                self._token_is_current_locked(token)
                and session.phase is _RecordingSessionPhase.ACQUIRING
                and session.producer_start_state
                is _ProducerStartState.NOT_STARTED
                and not session.cancellation_requested
                and not session.disconnect_requested
                and session.terminal._gate.is_open()
                and session.worker_gate.is_open()
                and self._controller_terminal_gate.is_open()
                and self._controller_worker_gate.is_open()
            ):
                return True
            cancellation_pending = bool(
                session.cancellation_requested
                or session.disconnect_requested
                or session.phase is _RecordingSessionPhase.CANCELLING
            )
            cancellation_reason = (
                session.cancellation_reason
                or ("disconnect" if session.disconnect_requested else "recording cancelled")
            )
        if cancellation_pending:
            self._start_cancellation_cleanup(
                token,
                cancellation_reason,
                cancelled=True,
            )
        return False

    def _claim_producer_start(
        self,
        token: _RecordingToken,
        session: _ActiveRecordingSession,
        kind: str,
    ) -> bool:
        with self._lock:
            if (
                self._session is not session
                or not self._token_is_current_locked(token)
                or session.phase is not _RecordingSessionPhase.ACQUIRING
                or session.producer_start_state
                is not _ProducerStartState.NOT_STARTED
                or session.cancellation_requested
                or session.disconnect_requested
                or not session.terminal._gate.is_open()
                or not session.worker_gate.is_open()
                or not self._controller_terminal_gate.is_open()
                or not self._controller_worker_gate.is_open()
            ):
                return False
            session.producer_kind = kind
            session.producer_start_state = _ProducerStartState.STARTING
            return True

    def _reconcile_producer_start(
        self,
        token: _RecordingToken,
        session: _ActiveRecordingSession,
        *,
        started: bool,
    ) -> tuple[bool, bool, str, _DeferredStartOutcome | None]:
        with self._lock:
            if (
                self._session is not session
                or session.token != token
                or not self._token_is_current_locked(token)
                or session.producer_start_state
                is not _ProducerStartState.STARTING
            ):
                return False, False, "", None
            deferred = session.start_attempt_gate.claim_deferred()
            reconciled_started = bool(started or deferred is not None)
            session.producer_start_state = (
                _ProducerStartState.STARTED
                if reconciled_started
                else _ProducerStartState.CLOSED
            )
            cancellation_pending = bool(
                session.cancellation_requested or session.disconnect_requested
            )
            cancellation_reason = (
                session.cancellation_reason
                or ("disconnect" if session.disconnect_requested else "recording cancelled")
            )
            if (
                reconciled_started
                and session.producer_kind == "streaming"
                and session.phase is _RecordingSessionPhase.ACQUIRING
                and (not cancellation_pending or deferred is not None)
            ):
                session.phase = _RecordingSessionPhase.STREAMING
        if deferred is not None:
            self._replay_deferred_start_outcome(session, deferred)
        return True, cancellation_pending, cancellation_reason, deferred

    @staticmethod
    def _start_outcome_operation(outcome: _AcquisitionOutcome) -> str:
        if outcome.error is not None or outcome.staged is None:
            return "failed"
        return "staged"

    def _replay_deferred_start_outcome(
        self,
        session: _ActiveRecordingSession,
        deferred: _DeferredStartOutcome,
    ) -> bool:
        if deferred.kind == "terminal":
            request = deferred.payload
            operation = request.operation
            callback = lambda: self._dispatch_terminal_request(request)
        else:
            outcome = deferred.payload
            if session.cancellation_requested or session.disconnect_requested:
                operation = (
                    "failed"
                    if session.cancellation_reports_failure
                    else "cancelled"
                )
            else:
                operation = self._start_outcome_operation(outcome)
            callback = lambda: self._deliver_blocking_outcome(
                session.token, outcome
            )
        try:
            accepted = bool(callback())
        except BaseException:
            session.start_attempt_gate.resolve(operation, False)
            raise
        session.start_attempt_gate.resolve(operation, accepted)
        return accepted

    def _truthful_start_result(
        self,
        session: _ActiveRecordingSession,
        adapter_started: bool,
    ) -> bool:
        terminal_result = session.start_attempt_gate.truthful_result()
        if terminal_result is not None:
            return terminal_result
        with self._lock:
            return bool(
                adapter_started
                and self._session is session
                and self._token_is_current_locked(session.token)
                and not session.cancellation_requested
                and not session.disconnect_requested
                and session.phase is not _RecordingSessionPhase.TERMINAL
            )

    def _settle_unstarted_callback_interruption(
        self,
        token: _RecordingToken,
        session: _ActiveRecordingSession,
        error: BaseException,
    ) -> bool:
        with self._lock:
            if self._session is not session or session.token != token:
                return False
            cancellation_pending = bool(
                session.cancellation_requested
                or session.disconnect_requested
                or session.phase is _RecordingSessionPhase.CANCELLING
            )
            cancellation_reason = (
                session.cancellation_reason
                or ("disconnect" if session.disconnect_requested else "recording cancelled")
            )
        if cancellation_pending:
            try:
                self._start_cancellation_cleanup(
                    token,
                    cancellation_reason,
                    cancelled=True,
                )
            except BaseException as cleanup_error:
                error.add_note(
                    f"recording unstarted cancellation cleanup failed: {cleanup_error}"
                )
                self._interrupt_active_session(token, error, cleanup=False)
            return True
        self._interrupt_active_session(token, error, cleanup=True)
        return False

    def _clear_active_locked(self) -> None:
        self._session = None

    def _finish_failed(
        self,
        token: _RecordingToken,
        reason: str,
        result: RecordingCommitResult | None = None,
    ) -> bool:
        event = RecordingFailed(
            token.session_id,
            reason or "recording failed",
            None
            if result is None
            else plain_message_value(result.rollback_outcome),
            False if result is None else result.audio_committed,
            None
            if result is None or result.recovery_path is None
            else str(result.recovery_path),
        )
        self.model.fail()
        self._mark_terminal_event_published(token)
        self.bus.events.recording_failed.emit(event)
        if self._terminal_presentation_allowed(token):
            self._present("failure", self.view.recording_failed, event.reason)
        return False

    def _finish_completed(
        self,
        token: _RecordingToken,
        *,
        sample_count: int,
        result_snapshot: Any,
        staged: StagedRecording | None = None,
    ) -> bool:
        if isinstance(result_snapshot, Mapping):
            normalized = dict(result_snapshot)
        else:
            normalized = {"result": result_snapshot}
        normalized.setdefault("record_id", token.session_id)
        normalized.setdefault("warnings", ())
        immutable_result = immutable_recording_value(normalized)
        event = RecordingCompleted(
            token.session_id, sample_count, plain_message_value(immutable_result)
        )
        self.model.finish(immutable_result)
        self._mark_terminal_event_published(token)
        self.bus.events.recording_completed.emit(event)
        if staged is not None and self._terminal_presentation_allowed(token):
            self._present("completion", self.view.recording_completed, staged)
        return True

    def _terminal_presentation_allowed(self, token: _RecordingToken) -> bool:
        with self._lock:
            session = self._session
            return bool(
                self._token_is_current_locked(token)
                and session is not None
                and not session.disconnect_requested
            )

    def _present(self, name: str, callback: Callable[..., Any], *args: Any) -> bool:
        try:
            callback(*args)
            return True
        except Exception as error:
            self._log("warning", f"recording {name} presentation failed: {error}")
            return False

    def _retire_locked(self, token: _RecordingToken) -> None:
        session = self._session
        if session is not None:
            session.terminal.close()
            session.worker_gate.close()
            if session.disconnect_requested:
                self._controller_worker_gate.close()
        self._remember_identity((token.token_id, token.session_id))
        if self._active_command_identity is not None:
            self._remember_identity(
                self._active_command_identity,
                refresh=True,
            )
            self._active_command_identity = None
        self._clear_active_locked()
        self._validated_workflow_generation = None
        if session is not None and session.disconnect_requested:
            self._active = False
            self._controller_generation += 1
            self.disconnect_quiesced.emit(token.session_id)

    def _fail_model_for_terminal_claim(
        self,
        token: _RecordingToken,
        session: _ActiveRecordingSession,
    ) -> BaseException | None:
        if self.model.active_session_id != token.session_id:
            return None
        try:
            self.model.fail()
        except BaseException as error:
            # A Model extension may re-enter controller cancellation/disconnect
            # or interrupt. Normalize only this still-matching model claim; the
            # controller claim is revalidated separately before retirement.
            if self.model.active_session_id == token.session_id:
                self.model.state = RecordingState.IDLE
                self.model.active_session_id = None
                self.model.active_snapshot = None
                self.model.final_result = None
            return error
        return None

    def _retire_suppressed_terminal(
        self,
        token: _RecordingToken,
        session: _ActiveRecordingSession,
    ) -> bool:
        session.worker_gate.close()
        model_error = self._fail_model_for_terminal_claim(token, session)
        self._mark_terminal_event_published(token)
        with self._lock:
            if self._session is session and session.token == token:
                self._retire_locked(token)
        if model_error is not None:
            raise model_error
        return False

    def _publish_terminal(
        self, token: _RecordingToken, callback: Callable[[], bool]
    ) -> bool:
        result = False
        callback_error: BaseException | None = None
        try:
            result = bool(callback())
        except BaseException as error:
            callback_error = error
        with self._lock:
            session = self._session
            claim = session if session is not None and session.token == token else None
        model_error = (
            None
            if claim is None
            else self._fail_model_for_terminal_claim(token, claim)
        )
        with self._lock:
            if claim is not None and self._session is claim and claim.token == token:
                self._retire_locked(token)
        if callback_error is not None:
            if model_error is not None:
                callback_error.add_note(
                    f"recording model terminal cleanup failed: {model_error}"
                )
            raise callback_error
        if model_error is not None:
            raise model_error
        return result

    def _retire_terminal_failure(
        self,
        token: _RecordingToken,
        reason: str,
        result: RecordingCommitResult | None = None,
    ) -> bool:
        """Resolve every worker-result failure through one retirement path."""
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            session = self._session
            if session is None:
                return False
            prior_phase = session.phase
            durable = session.durable
            if result is None and durable is not None:
                rollback_outcome = dict(durable.rollback_outcome)
                rollback_outcome["data_struct_committed"] = prior_phase in {
                    _RecordingSessionPhase.DATABASE_WORKER,
                    _RecordingSessionPhase.COUNT_PROJECTION,
                    _RecordingSessionPhase.PERSISTENCE_WORKER,
                }
                rollback_outcome["count_committed"] = (
                    prior_phase is _RecordingSessionPhase.PERSISTENCE_WORKER
                )
                result = RecordingCommitResult(
                    audio_committed=durable.audio_committed,
                    completed=False,
                    recovery_path=durable.recovery_path,
                    analysis_snapshot=None,
                    warnings=durable.warnings,
                    rollback_outcome=rollback_outcome,
                    reason=reason,
                    sample_count=durable.staged.sample_count,
                )
            session.phase = _RecordingSessionPhase.TERMINAL
            suppressed = session.suppress_terminal
        if suppressed:
            return self._retire_suppressed_terminal(token, session)
        return self._publish_terminal(
            token,
            lambda: self._finish_failed(token, reason, result),
        )

    @pyqtSlot(object)
    def handle_begin_recording(self, command: BeginRecordingRequested) -> bool:
        if type(command) is not BeginRecordingRequested:
            return False
        capability = self._canonical_recording_admission_capability
        claim_begin = getattr(
            self.bus,
            "_claim_canonical_recording_begin",
            None,
        )
        if capability is None or not callable(claim_begin):
            return False
        canonical_claim = claim_begin(capability, command)
        if canonical_claim is None:
            return False
        cancellation = canonical_claim.cancellation
        if cancellation is not None:
            self.bus.events.recording_cancelled.emit(
                RecordingCancelled(command.session_id, cancellation.reason)
            )
            return False
        identity = (command.command_id, command.session_id)
        admission_generation = self._admission_generation(command)
        mark_action_blocks_admission = False
        with self._lock:
            mark_action_blocks_admission = bool(
                self._mark_action_inflight_identity is not None
                or getattr(self.mark_action_service, "pending_recovery", None)
                is not None
            )
            if (
                not self._active
                or self._disconnecting
                or self._has_recent_identity_locked(identity)
                or self._pending_admission is not None
                or self._session is not None
                or mark_action_blocks_admission
            ):
                reservation = None
            else:
                self._remember_identity(identity)
                reservation = _PendingRecordingAdmission(
                    uuid4().hex,
                    identity,
                    command.session_id,
                    admission_generation,
                    self._controller_generation,
                    _TerminalLifecycleGate(),
                )
                self._pending_admission = reservation
        if reservation is None:
            if not canonical_claim.workflow_admitted:
                self._retire_recording_admission(command)
                return False
            if mark_action_blocks_admission and self._in_queued_command_delivery():
                return self._publish_mark_blocked_admission_terminal(command)
            return self._publish_claimed_admission_terminal(
                command,
                "recording admission is unavailable",
            )
        if not self._retire_recording_admission(command):
            with self._lock:
                self._clear_pending_admission_locked(reservation)
                self._forget_identity_locked(identity)
            return False

        try:
            initial_generation = (
                admission_generation
                if self.workflow_generation_provider is None
                else self._current_generation()
            )
        except BaseException:
            with self._lock:
                self._clear_pending_admission_locked(reservation)
            raise
        with self._lock:
            reservation_current = self._pending_admission_is_current_locked(
                reservation
            )
            generation_current = bool(
                reservation_current
                and type(admission_generation) is int
                and initial_generation == admission_generation
            )
            if reservation_current and not generation_current:
                self._clear_pending_admission_locked(reservation)
        if not reservation_current:
            return False
        if not generation_current:
            self.bus.events.recording_failed.emit(
                RecordingFailed(command.session_id, "stale workflow generation")
            )
            return False

        try:
            if self.prepare_session is None:
                raise RuntimeError("recording session preparation is unavailable")
            adapter_prepared = self.prepare_session(command)
            if type(adapter_prepared) is not PreparedRecordingSession:
                raise TypeError("session preparation returned an invalid value")
            prepared = PreparedRecordingSession.canonicalize(adapter_prepared)
            snapshot = prepared.snapshot
            if (
                snapshot.session_id != command.session_id
                or snapshot.workflow_generation != admission_generation
                or snapshot.replay is not command.replay
            ):
                raise ValueError("prepared recording snapshot does not match admission")
        except Exception as error:
            with self._lock:
                still_owned = self._clear_pending_admission_locked(reservation)
            if not still_owned:
                self._log(
                    "debug",
                    f"ignored invalidated recording preparation failure: {error}",
                )
                return False
            self._log("error", f"recording preparation failed: {error}")
            event = RecordingFailed(
                command.session_id,
                str(error) or "recording preparation failed",
            )
            self.bus.events.recording_failed.emit(event)
            self._present("preparation failure", self.view.recording_failed, event.reason)
            return False
        except BaseException:
            with self._lock:
                self._clear_pending_admission_locked(reservation)
            raise

        with self._lock:
            reservation_current = self._pending_admission_is_current_locked(
                reservation
            )
        if not reservation_current:
            self._abort_invalidated_preparation(prepared)
            return False

        try:
            commit_generation = (
                admission_generation
                if self.workflow_generation_provider is None
                else self._current_generation()
            )
        except BaseException:
            with self._lock:
                self._clear_pending_admission_locked(reservation)
            self._abort_invalidated_preparation(prepared)
            raise
        with self._lock:
            reservation_current = self._pending_admission_is_current_locked(
                reservation
            )
            generation_current = bool(
                reservation_current
                and commit_generation == reservation.workflow_generation
            )
            if reservation_current and not generation_current:
                self._clear_pending_admission_locked(reservation)
        if not reservation_current:
            self._abort_invalidated_preparation(prepared)
            return False
        if not generation_current:
            self._abort_invalidated_preparation(prepared)
            self.bus.events.recording_failed.emit(
                RecordingFailed(command.session_id, "stale workflow generation")
            )
            return False

        token = None
        terminal_gate = None
        worker_gate = None
        terminal = None
        session = None
        installed = False
        model_transition_started = False
        model_checkpoint = (
            self.model.state,
            self.model.active_session_id,
            self.model.active_snapshot,
            self.model.final_result,
        )
        try:
            token = _RecordingToken(
                uuid4().hex,
                command.session_id,
                admission_generation,
                reservation.controller_generation,
            )
            terminal_gate = _TerminalLifecycleGate()
            worker_gate = _TerminalLifecycleGate()
            terminal_relay = _TerminalRelay(
                self,
                self._controller_terminal_gate,
                terminal_gate,
            )
            terminal = RecordingTerminalPort(
                token,
                snapshot.as_message_payload(),
                _TerminalSubmissionBoundary(
                    terminal_relay,
                    self,
                    get_ident(),
                ),
                terminal_gate,
            )
            session = _ActiveRecordingSession(
                token,
                prepared,
                terminal,
                _RecordingSessionPhase.ACQUIRING,
                worker_gate,
            )
            with self._lock:
                reservation_current = self._pending_admission_is_current_locked(
                    reservation
                )
            if reservation_current:
                model_transition_started = True
                self.model.begin(snapshot)
                with self._lock:
                    reservation_current = (
                        self._pending_admission_is_current_locked(reservation)
                    )
            if reservation_current:
                self.model.mark_recording()
                with self._lock:
                    reservation_current = (
                        self._pending_admission_is_current_locked(reservation)
                    )
            if reservation_current:
                with self._lock:
                    if not self._pending_admission_is_current_locked(reservation):
                        reservation_current = False
                    else:
                        self._session = session
                        self._active_command_identity = identity
                        self._validated_workflow_generation = (
                            reservation.workflow_generation
                        )
                        self._clear_pending_admission_locked(reservation)
                        installed = True
            if not reservation_current:
                with self._lock:
                    self._clear_pending_admission_locked(reservation)
        except BaseException as error:
            with self._lock:
                if self._session is session:
                    self._session = None
                    self._active_command_identity = None
                    self._validated_workflow_generation = None
            if model_transition_started:
                self._restore_model_checkpoint(model_checkpoint, error)
            with self._lock:
                try:
                    self._clear_pending_admission_locked(reservation)
                except BaseException as close_error:
                    error.add_note(
                        f"recording reservation close failed: {close_error}"
                    )
                    if self._pending_admission is reservation:
                        self._pending_admission = None
            self._close_admission_resource(terminal, error)
            self._close_admission_resource(terminal_gate, error)
            self._close_admission_resource(worker_gate, error)
            self._close_admission_resource(reservation.gate, error)
            reason = str(error) or type(error).__name__
            self._abort_invalidated_preparation(prepared, reason)
            if not isinstance(error, Exception):
                raise
            self._log("error", f"recording preparation failed: {error}")
            event = RecordingFailed(command.session_id, reason)
            self.bus.events.recording_failed.emit(event)
            self._present("preparation failure", self.view.recording_failed, event.reason)
            return False
        if not installed:
            invalidated = RuntimeError("recording admission invalidated")
            if model_transition_started:
                self._restore_model_checkpoint(model_checkpoint, invalidated)
            with self._lock:
                self._clear_pending_admission_locked(reservation)
            self._close_admission_resource(terminal, invalidated)
            self._close_admission_resource(terminal_gate, invalidated)
            self._close_admission_resource(worker_gate, invalidated)
            self._abort_invalidated_preparation(prepared)
            return False
        try:
            self.bus.events.recording_started.emit(
                RecordingStarted(command.session_id, snapshot.as_message_payload())
            )
        except BaseException as error:
            self._settle_unstarted_callback_interruption(token, session, error)
            raise
        if not self._continue_unstarted_session(token, session):
            return False
        try:
            self._present("start", self.view.recording_started)
        except BaseException as error:
            self._settle_unstarted_callback_interruption(token, session, error)
            raise
        if not self._continue_unstarted_session(token, session):
            return False

        try:
            policy_prepared = PreparedRecordingSession.canonicalize(prepared)
            streaming = bool(self.use_streaming(policy_prepared))
        except Exception as error:
            with self._lock:
                cancellation_pending = bool(
                    self._session is session
                    and (
                        session.cancellation_requested
                        or session.disconnect_requested
                        or session.phase is _RecordingSessionPhase.CANCELLING
                    )
                )
            if cancellation_pending:
                self._settle_unstarted_callback_interruption(
                    token, session, error
                )
                return False
            self._log("error", f"recording acquisition policy failed: {error}")
            return terminal.recording_failed(error)
        except BaseException as error:
            self._settle_unstarted_callback_interruption(token, session, error)
            raise
        if not self._continue_unstarted_session(token, session):
            return False

        if streaming:
            if not self._claim_producer_start(token, session, "streaming"):
                self._continue_unstarted_session(token, session)
                return False
            try:
                if self.streaming_adapter is None:
                    raise RuntimeError("streaming recording adapter is unavailable")
                worker_prepared = PreparedRecordingSession.canonicalize(prepared)
                started = bool(self.streaming_adapter(worker_prepared, terminal))
            except Exception as error:
                self._log("error", f"recording acquisition failed: {error}")
                current, cancellation_pending, _reason, deferred = (
                    self._reconcile_producer_start(
                        token,
                        session,
                        started=True,
                    )
                )
                if deferred is not None:
                    return self._truthful_start_result(session, True)
                if not current:
                    return False
                if not cancellation_pending:
                    with self._lock:
                        if self._session is session:
                            session.cancellation_requested = True
                            session.cancellation_reason = (
                                str(error) or "streaming recording failed"
                            )
                            session.cancellation_reports_failure = True
                            session.phase = _RecordingSessionPhase.CANCELLING
                            session.terminal.close()
                self._start_streaming_quiescence(token)
                return False
            except BaseException as error:
                current, cancellation_pending, _reason, deferred = (
                    self._reconcile_producer_start(
                        token,
                        session,
                        started=True,
                    )
                )
                if current:
                    if deferred is not None:
                        raise
                    if not cancellation_pending:
                        with self._lock:
                            if self._session is session:
                                session.cancellation_requested = True
                                session.cancellation_reason = (
                                    str(error) or type(error).__name__
                                )
                                session.cancellation_reports_failure = True
                                session.phase = _RecordingSessionPhase.CANCELLING
                                session.terminal.close()
                    try:
                        self._start_streaming_quiescence(token)
                    except BaseException as cleanup_error:
                        error.add_note(
                            f"recording streaming interruption cleanup failed: {cleanup_error}"
                        )
                raise
            current, cancellation_pending, cancellation_reason, deferred = (
                self._reconcile_producer_start(
                    token,
                    session,
                    started=started,
                )
            )
            if deferred is not None:
                return self._truthful_start_result(session, started)
            if not current:
                return False
            if not started:
                if cancellation_pending:
                    self._start_cancellation_cleanup(
                        token,
                        cancellation_reason,
                        cancelled=True,
                    )
                    return False
                return terminal.recording_failed(
                    "streaming recording did not start"
                )
            if cancellation_pending:
                self._start_streaming_quiescence(token)
                return False
            return True
        if self.blocking_acquirer is None:
            return self._fail_terminal(
                token, "blocking recording adapter is unavailable"
            )
        blocking_acquirer = self.blocking_acquirer
        request_blocking_cancel = self.request_blocking_cancel
        try:
            worker_prepared = PreparedRecordingSession.canonicalize(prepared)
            job = self.blocking_executor.create(
                lambda: StagedRecording.canonicalize(
                    blocking_acquirer(worker_prepared)
                ),
                _WorkerResultBoundary(
                    self,
                    token,
                    "_deliver_blocking_outcome",
                    self._controller_worker_gate,
                    worker_gate,
                    self._worker_delivery_is_queued(self.blocking_executor),
                ),
                lambda reason: request_blocking_cancel(worker_prepared, reason),
            )
        except Exception as error:
            self._log("error", f"recording acquisition setup failed: {error}")
            rollback = self._cancel_outcome(
                prepared, str(error) or "recording acquisition setup failed"
            )
            with self._lock:
                if not self._token_is_current_locked(token):
                    return False
                session.phase = _RecordingSessionPhase.TERMINAL
            return self._publish_terminal(
                token,
                lambda: self._finish_failed_with_rollback(
                    token,
                    str(error) or "recording acquisition setup failed",
                    rollback,
                ),
            )
        except BaseException as error:
            self._interrupt_active_session(token, error, cleanup=True)
            raise
        cancellation_reason = ""
        job_is_current = False
        with self._lock:
            if self._token_is_current_locked(token):
                job_is_current = True
                session.blocking_job = job
                if session.cancellation_requested:
                    cancellation_reason = session.cancellation_reason
        if not job_is_current:
            try:
                job.cancel("recording admission invalidated")
            except BaseException as error:
                self._log(
                    "warning",
                    f"recording invalidated producer cleanup failed: {error}",
                )
            return False
        if cancellation_reason:
            try:
                job.cancel(cancellation_reason)
            except Exception as error:
                self._log(
                    "warning",
                    f"recording producer cancellation failed: {error}",
                )
            except BaseException as error:
                self._interrupt_active_session(token, error, cleanup=True)
                raise
            with self._lock:
                if not self._token_is_current_locked(token):
                    return False
                session.blocking_job = None
            return self._start_cancellation_cleanup(
                token,
                cancellation_reason or "recording cancelled",
                cancelled=True,
            )
        if not self._claim_producer_start(token, session, "blocking"):
            try:
                job.cancel("recording admission invalidated")
            except BaseException as error:
                self._log(
                    "warning",
                    f"recording invalidated producer cleanup failed: {error}",
                )
            self._continue_unstarted_session(token, session)
            return False
        try:
            start_result = bool(job.start())
        except Exception as error:
            self._log("error", f"recording acquisition start failed: {error}")
            reason = str(error) or "recording start failed"
            try:
                producer_alive = bool(job.is_alive())
            except AttributeError:
                producer_alive = False
            except BaseException as probe_error:
                self._log(
                    "warning",
                    f"recording producer liveness probe failed: {probe_error}",
                )
                producer_alive = True
            current, cancellation_won, cancellation_reason, deferred = (
                self._reconcile_producer_start(
                    token,
                    session,
                    started=producer_alive,
                )
            )
            if deferred is not None:
                return self._truthful_start_result(session, producer_alive)
            if not current:
                return False
            if producer_alive:
                with self._lock:
                    if self._session is session and not cancellation_won:
                        session.cancellation_requested = True
                        session.cancellation_reason = reason
                        session.cancellation_reports_failure = True
                        session.phase = _RecordingSessionPhase.CANCELLING
                        session.terminal.close()
                try:
                    job.cancel(cancellation_reason or reason)
                except BaseException as cancel_error:
                    self._log(
                        "warning",
                        f"recording producer cancellation failed: {cancel_error}",
                    )
                return False
            with self._lock:
                if not self._token_is_current_locked(token):
                    return False
                session.blocking_job = None
            if cancellation_won:
                return self._start_cancellation_cleanup(
                    token,
                    cancellation_reason,
                    cancelled=True,
                )
            return self._start_cancellation_cleanup(
                token,
                reason,
                cancelled=False,
            )
        except BaseException as error:
            try:
                producer_alive = bool(job.is_alive())
            except AttributeError:
                producer_alive = False
            except BaseException as probe_error:
                error.add_note(
                    f"recording producer liveness probe failed: {probe_error}"
                )
                producer_alive = True
            current, cancellation_won, cancellation_reason, deferred = (
                self._reconcile_producer_start(
                    token,
                    session,
                    started=producer_alive,
                )
            )
            if deferred is not None:
                raise
            if current and producer_alive:
                with self._lock:
                    if self._session is session and not cancellation_won:
                        session.cancellation_requested = True
                        session.cancellation_reason = (
                            str(error) or type(error).__name__
                        )
                        session.cancellation_reports_failure = True
                        session.phase = _RecordingSessionPhase.CANCELLING
                        session.terminal.close()
                try:
                    job.cancel(
                        cancellation_reason
                        or str(error)
                        or type(error).__name__
                    )
                except BaseException as cancel_error:
                    error.add_note(
                        f"recording producer interruption cancel failed: {cancel_error}"
                    )
            elif current:
                self._interrupt_active_session(token, error, cleanup=True)
            raise
        current, cancellation_won, cancellation_reason, deferred = (
            self._reconcile_producer_start(
                token,
                session,
                started=start_result,
            )
        )
        if deferred is not None:
            return self._truthful_start_result(session, start_result)
        if not current:
            return self._truthful_start_result(session, start_result)
        if cancellation_won:
            if not start_result:
                with self._lock:
                    if self._session is session:
                        session.blocking_job = None
                self._start_cancellation_cleanup(
                    token,
                    cancellation_reason,
                    cancelled=True,
                )
            else:
                try:
                    job.cancel(cancellation_reason)
                except BaseException as cancel_error:
                    self._log(
                        "warning",
                        f"recording producer cancellation failed: {cancel_error}",
                    )
            return False
        if not start_result:
            with self._lock:
                if self._session is session:
                    session.blocking_job = None
            return self._start_cancellation_cleanup(
                token,
                "recording start failed",
                cancelled=False,
            )
        return self._truthful_start_result(session, start_result)

    def _commit_terminal(
        self, token: _RecordingToken, staged: StagedRecording
    ) -> bool:
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            session = self._session
            if session is None or session.phase not in {
                _RecordingSessionPhase.ACQUIRING,
                _RecordingSessionPhase.STREAMING,
            }:
                return False
        try:
            staged = StagedRecording.canonicalize(staged)
        except Exception as error:
            return self._start_cancellation_cleanup(
                token,
                str(error) or "recording worker returned invalid data",
                cancelled=False,
            )
        except BaseException as error:
            self._interrupt_active_session(token, error, cleanup=True)
            raise
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            session = self._session
            if session is None or session.phase not in {
                _RecordingSessionPhase.ACQUIRING,
                _RecordingSessionPhase.STREAMING,
            }:
                return False
            if staged.snapshot.session_id != token.session_id:
                return False
            session.phase = _RecordingSessionPhase.FINALIZING_PRECOMMIT
            prepared = session.prepared
        self.model.mark_finalizing()
        try:
            if self.transaction_factory is None:
                raise RuntimeError("recording transaction is unavailable")
            transaction = self.transaction_factory(prepared)
            if not isinstance(transaction, RecordingTransaction):
                raise TypeError("recording transaction factory returned invalid value")
            checkpoint_port = _WorkerTransactionCheckpointPort(
                self,
                token,
                self._controller_worker_gate,
                session.worker_gate,
            )
            transaction.bind_cancellation_checkpoint(
                checkpoint_port.cancellation_checkpoint
            )
            transaction.bind_commit_barrier(
                checkpoint_port.begin_durable_commit
            )
            transaction.bind_promotion_succeeded(
                checkpoint_port.promotion_succeeded
            )
        except Exception as error:
            reason = str(error) or "recording transaction failed"
            return self._start_cancellation_cleanup(
                token, reason, cancelled=False
            )
        except BaseException as error:
            self._interrupt_active_session(token, error, cleanup=True)
            raise
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            session = self._session
            session.transaction = transaction
            session.staged = staged
        return self._launch_finalization_worker(
            token,
            lambda: transaction.prepare_durable(staged),
            self._deliver_durable_result,
            self._handle_precommit_worker_launch_failure,
            cleanup_on_base_exception=True,
        )

    def _worker_delivery_is_queued(self, executor: Any) -> bool:
        """Resolve the actual executor boundary, preserving injected inline APIs."""
        return bool(
            getattr(executor, "queued_delivery", self._queued_worker_delivery)
        )

    def _consume_queued_worker_delivery_escape(
        self,
        token: _RecordingToken,
        handler_name: str,
        error: BaseException,
    ) -> bool:
        """Last-resort retirement for a BaseException raised inside a Qt slot."""
        reason = _WorkerErrorDescriptor.capture(error).message
        self._log(
            "error",
            f"recording queued {handler_name} delivery interrupted: {reason}",
        )
        return self._recover_queued_worker_delivery_escape(
            token, handler_name, error
        )

    def _recover_queued_worker_delivery_escape(
        self,
        token: _RecordingToken,
        handler_name: str,
        error: BaseException,
    ) -> bool:
        reason = _WorkerErrorDescriptor.capture(error).message
        if handler_name == "_deliver_streaming_quiescence":
            self._record_streaming_quiescence_interruption(token, error)
            return False
        if handler_name == "_deliver_blocking_outcome":
            return self._start_cancellation_cleanup(
                token, reason, cancelled=False
            )
        if handler_name == "_deliver_durable_result":
            return self._retire_durable_protocol_failure(token, reason)
        return self._retire_terminal_failure(token, reason)

    def _launch_finalization_worker(
        self,
        token: _RecordingToken,
        work: Callable[[], Any],
        deliver: Callable[[_RecordingToken, _AcquisitionOutcome], Any],
        launch_failure: Callable[[_RecordingToken, Exception], bool],
        *,
        cleanup_on_base_exception: bool = False,
        base_exception_handler: Callable[[_RecordingToken, BaseException], None]
        | None = None,
        suppress_base_exception: bool = False,
    ) -> bool:
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            session = self._session
            if session is None:
                return False
            delivery_boundary = _WorkerResultBoundary(
                self,
                token,
                deliver.__name__,
                self._controller_worker_gate,
                session.worker_gate,
                self._worker_delivery_is_queued(self.finalization_executor),
            )
        try:
            job = self.finalization_executor.create(
                work,
                delivery_boundary,
                lambda _reason: None,
            )
        except Exception as error:
            return launch_failure(token, error)
        except BaseException as error:
            if base_exception_handler is None:
                self._interrupt_active_session(
                    token, error, cleanup=cleanup_on_base_exception
                )
            else:
                with self._lock:
                    session = self._session
                    already_handled = bool(
                        session is not None
                        and session.token == token
                        and session.phase
                        is _RecordingSessionPhase.STREAMING_STOP_FAILED
                    )
                if not already_handled:
                    base_exception_handler(token, error)
            if suppress_base_exception:
                return False
            raise
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            self._session.finalization_job = job
        try:
            return bool(job.start())
        except Exception as error:
            return launch_failure(token, error)
        except BaseException as error:
            if base_exception_handler is None:
                self._interrupt_active_session(
                    token, error, cleanup=cleanup_on_base_exception
                )
            else:
                with self._lock:
                    session = self._session
                    already_handled = bool(
                        session is not None
                        and session.token == token
                        and session.phase
                        is _RecordingSessionPhase.STREAMING_STOP_FAILED
                    )
                if not already_handled:
                    base_exception_handler(token, error)
            if suppress_base_exception:
                return False
            raise

    def _handle_precommit_worker_launch_failure(
        self, token: _RecordingToken, error: Exception
    ) -> bool:
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            session = self._session
            if session.phase is not _RecordingSessionPhase.FINALIZING_PRECOMMIT:
                return False
            transaction = session.transaction
            staged = session.staged
            session.finalization_job = None
        if transaction is None or staged is None:
            return self._start_cancellation_cleanup(
                token,
                str(error) or type(error).__name__,
                cancelled=False,
            )
        result = transaction.precommit_launch_failure(staged, error)
        return self._publish_commit_result(token, result, staged)

    def _deliver_durable_result(
        self, token: _RecordingToken, outcome: _AcquisitionOutcome
    ) -> bool:
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            session = self._session
            if session.phase not in {
                _RecordingSessionPhase.FINALIZING_PRECOMMIT,
                _RecordingSessionPhase.PROMOTION_IN_PROGRESS,
                _RecordingSessionPhase.PROMOTION_SUCCEEDED,
            }:
                return False
            session.finalization_job = None
            transaction = session.transaction
        if outcome.error is not None:
            if not isinstance(outcome.error, Exception):
                self._interrupt_active_session(token, outcome.error, cleanup=False)
                raise outcome.error
            return self._retire_durable_protocol_failure(
                token,
                str(outcome.error) or "recording durable finalization failed",
            )
        durable = outcome.staged
        if not isinstance(durable, RecordingDurableResult) or transaction is None:
            return self._retire_durable_protocol_failure(
                token, "recording durable result is invalid"
            )
        with self._lock:
            if self._token_is_current_locked(token):
                self._session.durable = durable
        if not durable.audio_committed:
            result = transaction.apply_data_struct_projection(durable)
            if result is None:
                raise RuntimeError("non-durable result unexpectedly projected")
            if result.cancelled:
                return self._start_cancellation_cleanup(
                    token,
                    session.cancellation_reason or result.reason,
                    cancelled=True,
                    base_result=result,
                )
            return self._publish_commit_result(token, result, durable.staged)
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            session = self._session
            if session.disconnect_requested:
                session.phase = _RecordingSessionPhase.TERMINAL
                suppressed_session = session
            else:
                suppressed_session = None
        if suppressed_session is not None:
            return self._retire_suppressed_terminal(token, suppressed_session)
        projection_failure = transaction.apply_data_struct_projection(durable)
        if projection_failure is not None:
            return self._publish_commit_result(
                token, projection_failure, durable.staged
            )
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            self._session.phase = _RecordingSessionPhase.DATABASE_WORKER
        return self._launch_finalization_worker(
            token,
            lambda: transaction.save_database_worker(durable),
            self._deliver_database_result,
            lambda current_token, error: self._handle_database_worker_launch_failure(
                current_token, durable, error
            ),
        )

    def _retire_durable_protocol_failure(
        self, token: _RecordingToken, reason: str
    ) -> bool:
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            session = self._session
            if session is None:
                return False
            promoted = session.phase is _RecordingSessionPhase.PROMOTION_SUCCEEDED
            staged = session.staged
        if promoted and staged is not None:
            result = RecordingCommitResult(
                audio_committed=True,
                completed=False,
                recovery_path=staged.snapshot.output_path,
                analysis_snapshot=None,
                warnings=(),
                rollback_outcome={
                    "audio_committed": True,
                    "data_struct_committed": False,
                    "count_committed": False,
                },
                reason=reason,
                sample_count=staged.sample_count,
            )
            return self._retire_terminal_failure(token, reason, result)
        return self._start_cancellation_cleanup(
            token, reason, cancelled=False
        )

    def _handle_database_worker_launch_failure(
        self,
        token: _RecordingToken,
        durable: RecordingDurableResult,
        error: Exception,
    ) -> bool:
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            session = self._session
            if session.phase is not _RecordingSessionPhase.DATABASE_WORKER:
                return False
            transaction = session.transaction
            session.finalization_job = None
        if transaction is None:
            return self._retire_terminal_failure(
                token, "recording transaction is unavailable"
            )
        warned = transaction.with_worker_warning(durable, "database", error)
        return self._continue_after_database(token, warned)

    def _deliver_database_result(
        self, token: _RecordingToken, outcome: _AcquisitionOutcome
    ) -> bool:
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            session = self._session
            if session.phase is not _RecordingSessionPhase.DATABASE_WORKER:
                return False
            session.finalization_job = None
            transaction = session.transaction
        if outcome.error is not None:
            if not isinstance(outcome.error, Exception):
                self._interrupt_active_session(token, outcome.error, cleanup=False)
                raise outcome.error
            if transaction is None or session.durable is None:
                return self._retire_terminal_failure(
                    token, str(outcome.error) or "database worker failed"
                )
            return self._continue_after_database(
                token,
                transaction.with_worker_warning(
                    session.durable, "database", outcome.error
                ),
            )
        durable = outcome.staged
        if not isinstance(durable, RecordingDurableResult) or transaction is None:
            return self._retire_terminal_failure(
                token, "recording database result is invalid"
            )
        return self._continue_after_database(token, durable)

    def _continue_after_database(
        self, token: _RecordingToken, durable: RecordingDurableResult
    ) -> bool:
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            session = self._session
            transaction = session.transaction
            session.durable = durable
        if transaction is None:
            return self._retire_terminal_failure(
                token, "recording transaction is unavailable"
            )
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            self._session.phase = _RecordingSessionPhase.COUNT_PROJECTION
        count_failure = transaction.apply_count_projection(durable)
        if count_failure is not None:
            return self._publish_commit_result(
                token, count_failure, durable.staged
            )
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            self._session.phase = _RecordingSessionPhase.PERSISTENCE_WORKER
        return self._launch_finalization_worker(
            token,
            lambda: transaction.persist_and_cleanup_worker(durable),
            self._deliver_persistence_result,
            lambda current_token, error: self._handle_persistence_worker_launch_failure(
                current_token, durable, error
            ),
        )

    def _handle_persistence_worker_launch_failure(
        self,
        token: _RecordingToken,
        durable: RecordingDurableResult,
        error: Exception,
    ) -> bool:
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            session = self._session
            if session.phase is not _RecordingSessionPhase.PERSISTENCE_WORKER:
                return False
            transaction = session.transaction
            session.finalization_job = None
        if transaction is None:
            return self._retire_terminal_failure(
                token, "recording transaction is unavailable"
            )
        result = transaction.persistence_launch_failure(durable, error)
        return self._publish_commit_result(token, result, durable.staged)

    def _deliver_persistence_result(
        self, token: _RecordingToken, outcome: _AcquisitionOutcome
    ) -> bool:
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            session = self._session
            if session.phase is not _RecordingSessionPhase.PERSISTENCE_WORKER:
                return False
            session.finalization_job = None
        if outcome.error is not None:
            if not isinstance(outcome.error, Exception):
                self._interrupt_active_session(token, outcome.error, cleanup=False)
                raise outcome.error
            transaction = session.transaction
            if transaction is None or session.durable is None:
                return self._retire_terminal_failure(
                    token,
                    str(outcome.error) or "recording persistence worker failed",
                )
            result = transaction.persistence_launch_failure(
                session.durable, outcome.error
            )
        else:
            result = outcome.staged
        if not isinstance(result, RecordingCommitResult):
            return self._retire_terminal_failure(
                token, "recording persistence result is invalid"
            )
        return self._publish_commit_result(token, result, session.staged)

    def _publish_commit_result(
        self,
        token: _RecordingToken,
        result: RecordingCommitResult,
        staged: StagedRecording | None,
    ) -> bool:
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            session = self._session
            session.phase = _RecordingSessionPhase.TERMINAL
            suppressed = session.suppress_terminal
        if suppressed:
            return self._retire_suppressed_terminal(token, session)
        if result.cancelled:
            return self._publish_terminal(
                token,
                lambda: self._finish_cancelled(
                    token,
                    session.cancellation_reason or result.reason,
                    result.rollback_outcome,
                ),
            )
        if not result.completed:
            return self._publish_terminal(
                token, lambda: self._finish_failed(token, result.reason, result)
            )
        return self._publish_terminal(
            token,
            lambda: self._finish_completed(
                token,
                sample_count=result.sample_count,
                result_snapshot=result.analysis_snapshot,
                staged=staged,
            ),
        )

    def _compatibility_completed(
        self,
        token: _RecordingToken,
        *,
        sample_count: int,
        result_snapshot: Any,
    ) -> bool:
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            session = self._session
            if session is None or session.phase is not _RecordingSessionPhase.STREAMING:
                return False
            session.phase = _RecordingSessionPhase.TERMINAL
        return self._publish_terminal(
            token,
            lambda: self._finish_completed(
                token,
                sample_count=sample_count,
                result_snapshot=result_snapshot,
            ),
        )

    def _fail_terminal(
        self,
        token: _RecordingToken,
        reason: str,
        rollback_outcome: Any = None,
    ) -> bool:
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            session = self._session
            if session is None or session.cancellation_requested:
                return False
            session.phase = _RecordingSessionPhase.TERMINAL
        if isinstance(rollback_outcome, Mapping):
            return self._publish_terminal(
                token,
                lambda: self._finish_failed_with_rollback(
                    token, reason, rollback_outcome
                ),
            )
        return self._publish_terminal(
            token, lambda: self._finish_failed(token, reason)
        )

    def _check_precommit_cancellation(self, token: _RecordingToken) -> None:
        with self._lock:
            # Transaction checkpoints run on the recording worker.  They may
            # consult only the frozen token and controller-owned protected
            # session state; the live workflow provider is Qt-thread-only.
            if not self._token_matches_frozen_session_locked(token):
                raise RecordingCancellationRequested(
                    "recording session is no longer active"
                )
            session = self._session
            if (
                session is None
                or session.phase is not _RecordingSessionPhase.FINALIZING_PRECOMMIT
            ):
                raise RuntimeError("recording cancellation checkpoint is out of order")
            if session.cancellation_requested:
                raise RecordingCancellationRequested(
                    session.cancellation_reason or "recording cancelled"
                )

    def _begin_durable_commit(self, token: _RecordingToken) -> None:
        with self._lock:
            if not self._token_matches_frozen_session_locked(token):
                raise RecordingCancellationRequested("recording session is no longer active")
            session = self._session
            if (
                session is None
                or session.phase is not _RecordingSessionPhase.FINALIZING_PRECOMMIT
            ):
                raise RuntimeError("recording durable commit barrier is out of order")
            if session.cancellation_requested:
                raise RecordingCancellationRequested(
                    session.cancellation_reason or "recording cancelled"
                )
            session.phase = _RecordingSessionPhase.PROMOTION_IN_PROGRESS

    def _mark_promotion_succeeded(self, token: _RecordingToken) -> None:
        with self._lock:
            if not self._token_matches_frozen_session_locked(token):
                raise RuntimeError("recording session ended during promotion")
            session = self._session
            if (
                session is None
                or session.phase is not _RecordingSessionPhase.PROMOTION_IN_PROGRESS
            ):
                raise RuntimeError("recording promotion completion is out of order")
            session.phase = _RecordingSessionPhase.PROMOTION_SUCCEEDED

    def _phase_before_durable_commit(self, token: _RecordingToken) -> bool:
        with self._lock:
            return bool(
                self._token_matches_frozen_session_locked(token)
                and self._session is not None
                and self._session.phase
                is _RecordingSessionPhase.FINALIZING_PRECOMMIT
            )

    def _deliver_blocking_outcome(
        self, token: _RecordingToken, outcome: _AcquisitionOutcome
    ) -> bool:
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            session = self._session
            if session is None:
                return False
            if session.producer_start_state is _ProducerStartState.STARTING:
                return session.start_attempt_gate.reserve(
                    _DeferredStartOutcome("blocking", outcome)
                )
            session.producer_start_state = _ProducerStartState.CLOSED
            session.blocking_job = None
            cancelling = session.phase is _RecordingSessionPhase.CANCELLING
            if session.cancellation_reports_failure:
                reason = (
                    session.deferred_start_failure_reason
                    or session.cancellation_reason
                )
            else:
                reason = (
                    session.cancellation_reason
                    or session.deferred_start_failure_reason
                )
            cancelled = not session.cancellation_reports_failure
        if outcome.error is not None and not isinstance(outcome.error, Exception):
            self._interrupt_active_session(token, outcome.error, cleanup=True)
            raise outcome.error
        if cancelling:
            return self._start_cancellation_cleanup(
                token,
                reason or "recording cancelled",
                cancelled=cancelled,
            )
        if outcome.error is not None:
            self._log("error", f"recording acquisition failed: {outcome.error}")
            return self._start_cancellation_cleanup(
                token,
                str(outcome.error) or "recording acquisition failed",
                cancelled=False,
            )
        if outcome.staged is None:
            return self._fail_terminal(token, "recording acquisition returned no data")
        return self._commit_terminal(token, outcome.staged)

    def _start_cancellation_cleanup(
        self,
        token: _RecordingToken,
        reason: str,
        *,
        cancelled: bool,
        base_result: RecordingCommitResult | None = None,
    ) -> bool:
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            session = self._session
            if (
                session.producer_start_state
                is _ProducerStartState.STARTING
            ):
                return False
            prepared = session.prepared
            delivery_boundary = _WorkerResultBoundary(
                self,
                token,
                "_deliver_cancellation_cleanup",
                self._controller_worker_gate,
                session.worker_gate,
                self._worker_delivery_is_queued(self.finalization_executor),
                (reason, cancelled, base_result),
            )
        cancel_adapter = self.cancel_adapter
        try:
            job = self.finalization_executor.create(
                lambda: cancel_adapter(prepared, reason),
                delivery_boundary,
                lambda _reason: None,
            )
        except Exception as error:
            return self._deliver_cancellation_cleanup(
                token,
                _AcquisitionOutcome(error=error),
                reason,
                cancelled,
                base_result,
            )
        except BaseException as error:
            self._interrupt_active_session(token, error, cleanup=False)
            raise
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            self._session.finalization_job = job
        try:
            return bool(job.start())
        except Exception as error:
            return self._deliver_cancellation_cleanup(
                token,
                _AcquisitionOutcome(error=error),
                reason,
                cancelled,
                base_result,
            )
        except BaseException as error:
            self._interrupt_active_session(token, error, cleanup=False)
            raise

    def _deliver_cancellation_cleanup(
        self,
        token: _RecordingToken,
        outcome: _AcquisitionOutcome,
        reason: str,
        cancelled: bool,
        base_result: RecordingCommitResult | None,
    ) -> bool:
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            session = self._session
            session.finalization_job = None
            suppressed = session.suppress_terminal
            streaming_failure_rollback = session.streaming_failure_rollback
        if outcome.error is not None:
            if not isinstance(outcome.error, Exception):
                self._interrupt_active_session(token, outcome.error, cleanup=False)
                raise outcome.error
            cleanup_outcome = {
                "restored": False,
                "errors": (str(outcome.error) or type(outcome.error).__name__,),
            }
        elif isinstance(outcome.staged, Mapping):
            cleanup_outcome = dict(outcome.staged)
        else:
            cleanup_outcome = {"restored": True, "errors": ()}
        if isinstance(streaming_failure_rollback, Mapping):
            cleanup_outcome = {
                "restored": bool(
                    streaming_failure_rollback.get("restored", False)
                )
                and bool(cleanup_outcome.get("restored", False)),
                "errors": tuple(
                    streaming_failure_rollback.get("errors", ())
                )
                + tuple(cleanup_outcome.get("errors", ())),
            }
        if base_result is not None:
            cleanup_outcome = {
                "restored": bool(
                    base_result.rollback_outcome.get("restored", False)
                )
                and bool(cleanup_outcome.get("restored", False)),
                "errors": tuple(base_result.rollback_outcome.get("errors", ()))
                + tuple(cleanup_outcome.get("errors", ())),
            }
        if suppressed:
            with self._lock:
                if self._session is session and session.token == token:
                    session.phase = _RecordingSessionPhase.TERMINAL
            return self._retire_suppressed_terminal(token, session)
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            session.phase = _RecordingSessionPhase.TERMINAL
        if cancelled:
            return self._publish_terminal(
                token,
                lambda: self._finish_cancelled(
                    token, reason or "recording cancelled", cleanup_outcome
                ),
            )
        return self._publish_terminal(
            token,
            lambda: self._finish_failed_with_rollback(
                token, reason or "recording failed", cleanup_outcome
            ),
        )

    def _cancel_outcome(
        self, prepared: PreparedRecordingSession, reason: str
    ) -> Mapping[str, Any]:
        try:
            outcome = self.cancel_adapter(prepared, reason)
        except Exception as error:
            self._log("warning", f"recording cancellation cleanup failed: {error}")
            return {
                "restored": False,
                "errors": (str(error) or type(error).__name__,),
            }
        if isinstance(outcome, Mapping):
            return dict(outcome)
        return {"restored": True, "errors": ()}

    def _cancel_terminal(self, token: _RecordingToken, reason: str) -> bool:
        return self._request_cancellation(
            token, reason, deferred_terminal_claim=True
        )

    def _request_streaming_failure(
        self,
        token: _RecordingToken,
        reason: str,
        rollback_outcome: Any,
        producer_quiesced: bool,
        shutdown_diagnostic: str,
    ) -> bool:
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            session = self._session
            if (
                session is None
                or session.phase is not _RecordingSessionPhase.STREAMING
                or session.cancellation_requested
            ):
                return False
            session.cancellation_requested = True
            session.cancellation_reports_failure = True
            session.cancellation_reason = str(reason) or "streaming consumer failed"
            session.streaming_failure_rollback = immutable_recording_value(
                rollback_outcome
            )
            session.terminal.close()
        if shutdown_diagnostic:
            self._log(
                "warning",
                "streaming consumer requested producer shutdown: "
                f"{shutdown_diagnostic}",
            )
        elif not producer_quiesced:
            self._log(
                "warning",
                "streaming consumer requested producer shutdown without an acknowledgement",
            )
        return self._start_streaming_quiescence(token)

    def _request_cancellation(
        self,
        token: _RecordingToken,
        reason: str,
        *,
        deferred_terminal_claim: bool = False,
    ) -> bool:
        job = None
        start_streaming_quiescence = False
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            session = self._session
            if session is None:
                return False
            if (
                session.producer_start_state is _ProducerStartState.STARTING
                and session.start_attempt_gate.has_unresolved_deferred()
                and not deferred_terminal_claim
            ):
                return False
            if session.cancellation_requested:
                if session.phase is _RecordingSessionPhase.STREAMING_STOP_FAILED:
                    start_streaming_quiescence = True
                else:
                    return False
            if session.phase is _RecordingSessionPhase.TERMINAL:
                return False
            if not session.cancellation_requested:
                session.cancellation_requested = True
                session.cancellation_reason = reason
                session.terminal.close()
                if (
                    session.producer_start_state
                    is _ProducerStartState.NOT_STARTED
                ):
                    session.producer_start_state = _ProducerStartState.CLOSED
                if session.phase is _RecordingSessionPhase.ACQUIRING:
                    session.phase = _RecordingSessionPhase.CANCELLING
                    if (
                        session.producer_start_state
                        is not _ProducerStartState.STARTING
                    ):
                        job = session.blocking_job
                elif session.phase is _RecordingSessionPhase.STREAMING:
                    start_streaming_quiescence = True
                elif session.phase in {
                    _RecordingSessionPhase.FINALIZING_PRECOMMIT,
                    _RecordingSessionPhase.PROMOTION_IN_PROGRESS,
                    _RecordingSessionPhase.PROMOTION_SUCCEEDED,
                    _RecordingSessionPhase.DATABASE_WORKER,
                    _RecordingSessionPhase.COUNT_PROJECTION,
                    _RecordingSessionPhase.PERSISTENCE_WORKER,
                    _RecordingSessionPhase.STREAMING_QUIESCING,
                    _RecordingSessionPhase.STREAMING_CLEANUP,
                }:
                    return True
                elif session.phase is _RecordingSessionPhase.CANCELLING:
                    return False
        if job is not None:
            try:
                job.cancel(reason)
            except Exception as error:
                self._log("warning", f"recording producer cancellation failed: {error}")
            return True
        if start_streaming_quiescence:
            return self._start_streaming_quiescence(token)
        return True

    def _start_streaming_quiescence(self, token: _RecordingToken) -> bool:
        close_admission = False
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            session = self._session
            valid_starting_phase = bool(
                session.phase is _RecordingSessionPhase.CANCELLING
                and session.producer_kind == "streaming"
                and session.producer_start_state
                is _ProducerStartState.STARTED
            )
            if (
                session.phase
                not in {
                    _RecordingSessionPhase.STREAMING,
                    _RecordingSessionPhase.STREAMING_STOP_FAILED,
                }
                and not valid_starting_phase
            ):
                return False
            prepared = session.prepared
            reason = session.cancellation_reason or "recording cancelled"
            session.phase = _RecordingSessionPhase.STREAMING_QUIESCING
            session.terminal.close()
            close_admission = not session.streaming_admission_closed
            handle = session.streaming_handle
        if close_admission:
            try:
                handle = self.close_streaming_admission(prepared)
            except Exception as error:
                return self._mark_streaming_quiescence_failed(token, error)
            except BaseException as error:
                self._record_streaming_quiescence_interruption(token, error)
                raise
            with self._lock:
                if not self._token_is_current_locked(token):
                    return False
                self._session.streaming_handle = handle
                self._session.streaming_admission_closed = True
        quiesce_streaming = self.quiesce_streaming
        return self._launch_finalization_worker(
            token,
            lambda: quiesce_streaming(prepared, reason, handle),
            self._deliver_streaming_quiescence,
            self._mark_streaming_quiescence_failed,
            base_exception_handler=self._record_streaming_quiescence_interruption,
            suppress_base_exception=self._in_queued_command_delivery(),
        )

    def _deliver_streaming_quiescence(
        self, token: _RecordingToken, outcome: _AcquisitionOutcome
    ) -> bool:
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            session = self._session
            if session.phase is not _RecordingSessionPhase.STREAMING_QUIESCING:
                return False
            session.finalization_job = None
            reason = session.cancellation_reason or "recording cancelled"
            producer_stop_acknowledged = session.streaming_stop_acknowledged
            cancelled = not session.cancellation_reports_failure
        if outcome.error is not None:
            if producer_stop_acknowledged:
                with self._lock:
                    if not self._token_is_current_locked(token):
                        return False
                    self._session.phase = _RecordingSessionPhase.STREAMING_CLEANUP
                return self._start_cancellation_cleanup(
                    token, reason, cancelled=cancelled
                )
            if not isinstance(outcome.error, Exception):
                self._record_streaming_quiescence_interruption(
                    token, outcome.error
                )
                raise outcome.error
            return self._mark_streaming_quiescence_failed(token, outcome.error)
        acknowledgement = outcome.staged
        acknowledged = bool(
            acknowledgement is True
            or (
                isinstance(acknowledgement, Mapping)
                and acknowledgement.get("quiesced") is True
            )
        )
        if not acknowledged:
            return self._mark_streaming_quiescence_failed(
                token,
                RuntimeError("streaming producer did not acknowledge quiescence"),
            )
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            self._session.phase = _RecordingSessionPhase.STREAMING_CLEANUP
        return self._start_cancellation_cleanup(
            token, reason, cancelled=cancelled
        )

    @pyqtSlot(str)
    def acknowledge_streaming_quiesced(self, session_id: str) -> bool:
        """Continue cancellation after a producer reports a delayed stop ack."""
        with self._lock:
            session = self._session
            if (
                session is None
                or session.token.session_id != session_id
                or session.phase
                not in {
                    _RecordingSessionPhase.STREAMING_QUIESCING,
                    _RecordingSessionPhase.STREAMING_STOP_FAILED,
                }
                or session.streaming_stop_acknowledged
                or not self._token_is_current_locked(session.token)
            ):
                return False
            token = session.token
            reason = session.cancellation_reason or "recording cancelled"
            cancelled = not session.cancellation_reports_failure
            session.streaming_stop_acknowledged = True
            if session.phase is _RecordingSessionPhase.STREAMING_QUIESCING:
                return True
            session.phase = _RecordingSessionPhase.STREAMING_CLEANUP
        return self._start_cancellation_cleanup(
            token, reason, cancelled=cancelled
        )

    def _mark_streaming_quiescence_failed(
        self, token: _RecordingToken, error: Exception
    ) -> bool:
        reason = str(error) or type(error).__name__
        return self._set_streaming_quiescence_failed(token, reason)

    def _record_streaming_quiescence_interruption(
        self, token: _RecordingToken, error: BaseException
    ) -> None:
        reason = str(error) or type(error).__name__
        self._set_streaming_quiescence_failed(token, reason)

    def _set_streaming_quiescence_failed(
        self, token: _RecordingToken, reason: str
    ) -> bool:
        retry_disconnect = False
        with self._lock:
            if not self._token_is_current_locked(token):
                return False
            session = self._session
            if session.streaming_stop_acknowledged:
                return False
            session.finalization_job = None
            session.phase = _RecordingSessionPhase.STREAMING_STOP_FAILED
            if (
                session.disconnect_requested
                and not session.streaming_disconnect_retry_started
            ):
                session.streaming_disconnect_retry_started = True
                retry_disconnect = True
        self._log(
            "error", f"recording streaming producer quiescence failed: {reason}"
        )
        self.streaming_quiescence_failed.emit(token.session_id, reason)
        if retry_disconnect:
            return self._start_streaming_quiescence(token)
        return False

    def _finish_cancelled(
        self,
        token: _RecordingToken,
        reason: str,
        outcome: Mapping[str, Any],
    ) -> bool:
        event = RecordingCancelled(
            token.session_id, reason, plain_message_value(outcome)
        )
        self.model.fail()
        self._mark_terminal_event_published(token)
        self.bus.events.recording_cancelled.emit(event)
        cancelled_view = getattr(self.view, "recording_cancelled", None)
        if callable(cancelled_view) and self._terminal_presentation_allowed(token):
            self._present("cancellation", cancelled_view, reason)
        return True

    def _finish_failed_with_rollback(
        self,
        token: _RecordingToken,
        reason: str,
        rollback_outcome: Mapping[str, Any],
    ) -> bool:
        event = RecordingFailed(
            token.session_id,
            reason,
            plain_message_value(rollback_outcome),
            False,
            None,
        )
        self.model.fail()
        self._mark_terminal_event_published(token)
        self.bus.events.recording_failed.emit(event)
        if self._terminal_presentation_allowed(token):
            self._present("failure", self.view.recording_failed, reason)
        return False

    def _interrupt_active_session(
        self,
        token: _RecordingToken,
        error: BaseException,
        *,
        cleanup: bool,
    ) -> None:
        with self._lock:
            if not self._token_is_current_locked(token):
                return
            session = self._session
            prepared = None if session is None else session.prepared
            if session is not None:
                session.phase = _RecordingSessionPhase.TERMINAL
        if cleanup and prepared is not None:
            try:
                self.cancel_adapter(prepared, str(error) or type(error).__name__)
            except BaseException as cleanup_error:
                error.add_note(f"recording interruption cleanup failed: {cleanup_error}")
        try:
            self.model.fail()
        except BaseException as model_error:
            error.add_note(f"recording model interruption cleanup failed: {model_error}")
        finally:
            self.model.state = RecordingState.IDLE
            self.model.active_session_id = None
            self.model.active_snapshot = None
            self.model.final_result = None
        with self._lock:
            session = self._session
            if session is not None and session.token == token:
                self._retire_locked(token)
        cancelled_view = getattr(self.view, "recording_cancelled", None)
        if callable(cancelled_view):
            try:
                self._present(
                    "interruption",
                    cancelled_view,
                    str(error) or type(error).__name__,
                )
            except BaseException as presentation_error:
                error.add_note(
                    f"recording interruption presentation failed: {presentation_error}"
                )

    @pyqtSlot(object)
    def handle_cancel_recording(self, command: CancelRecordingRequested) -> bool:
        if type(command) is not CancelRecordingRequested:
            return False
        with self._lock:
            pending = self._pending_admission
            if (
                pending is not None
                and command.session_id == pending.session_id
                and command.workflow_generation == pending.workflow_generation
            ):
                return self._clear_pending_admission_locked(pending)
            session = self._session
            token = None if session is None else session.token
            if (
                token is None
                or command.session_id != token.session_id
                or command.workflow_generation != token.workflow_generation
            ):
                return False
        return self._request_cancellation(token, command.reason)

    def request_mark_action(
        self,
    ) -> bool | RecordingMarkActionDispatchPending:
        """Translate one raw Qt click into a generation-bound Recording command."""
        if self._mark_action_requires_qt_owner_dispatch():
            return self._invoke_mark_action_qt_owner("request")
        pending = getattr(self.mark_action_service, "pending_recovery", None)
        if pending is not None:
            return self._retry_mark_action_recovery(pending)
        try:
            generation = self._current_generation()
        except BaseException as error:
            if isinstance(error, Exception):
                self._log("error", f"recording mark-action generation failed: {error}")
                return False
            raise
        if generation is None:
            generation = 0
        return self.handle_mark_action(
            RecordingMarkActionRequested(uuid4().hex, generation)
        )

    def _mark_action_requires_qt_owner_dispatch(self) -> bool:
        projection = getattr(self.mark_action_service, "projection", None)
        return bool(
            getattr(projection, "requires_qt_owner_thread", False)
            and QThread.currentThread() is not self.thread()
        )

    def _invoke_mark_action_qt_owner(
        self,
        operation: str,
        command: RecordingMarkActionRequested | None = None,
    ) -> bool | RecordingMarkActionDispatchPending:
        owner_thread = self.thread()
        if owner_thread is None or not owner_thread.isRunning():
            return False
        call = _MarkActionQtOwnerCall(
            token=uuid4().hex,
            operation=operation,
            command=command,
        )
        if not self._mark_action_qt_owner_dispatch.register(call):
            return False
        try:
            self._mark_action_qt_owner_requested.emit(call.token)
        except BaseException:
            self._mark_action_qt_owner_dispatch.cancel(call)
            raise
        if not call.completion.wait(1.0):
            if self._mark_action_qt_owner_dispatch.cancel(call):
                return False
            # Delivery already won the exact-token race. Expose the accepted
            # asynchronous operation rather than reporting False while its
            # Qt-owner mutation remains live or blocking this caller forever.
            return RecordingMarkActionDispatchPending(call)
        if call.error is not None:
            error = call.error
            del call
            raise error
        return call.result

    @pyqtSlot(object)
    def _run_mark_action_qt_owner_call(self, token: Any) -> None:
        if type(token) is not str:
            return
        call = self._mark_action_qt_owner_dispatch.claim(token)
        if call is None:
            return
        result, error = self._execute_mark_action_qt_owner_call(
            call.operation,
            call.command,
        )
        self._mark_action_qt_owner_dispatch.complete(
            call,
            result=result,
            error=error,
        )
        # PyQt may retain the most recently invoked slot frame until another
        # event is dispatched. Drop the relay local explicitly while keeping
        # the exact exception and its originating traceback untouched.
        del call

    def _execute_mark_action_qt_owner_call(
        self,
        operation: str,
        command: RecordingMarkActionRequested | None,
    ) -> tuple[bool, BaseException | None]:
        result = False
        try:
            if operation == "request":
                owner_result = self.request_mark_action()
            elif operation == "handle":
                owner_result = self.handle_mark_action(command)
            else:
                owner_result = False
            if type(owner_result) is not bool:
                raise RuntimeError(
                    "Qt-owner mark-action delivery returned a pending handoff"
                )
            result = owner_result
        except BaseException as caught:
            return False, caught
        return result, None

    def _cancel_mark_action_qt_owner_calls(self, _destroyed: Any = None) -> None:
        self._mark_action_qt_owner_dispatch.cancel_all(_destroyed)

    def _record_mark_action_terminal_locked(
        self, identity: tuple[str, int]
    ) -> None:
        self._mark_action_terminal_identities[identity] = None
        while len(self._mark_action_terminal_identities) > self._recent_identity_limit:
            self._mark_action_terminal_identities.popitem(last=False)

    def _publish_mark_blocked_admission_terminal(
        self,
        command: BeginRecordingRequested,
    ) -> bool:
        return self._publish_claimed_admission_terminal(
            command,
            "recording admission blocked by mark cleanup",
        )

    def _publish_claimed_admission_terminal(
        self,
        command: BeginRecordingRequested,
        reason: str,
    ) -> bool:
        admission_generation = self._admission_generation(command)
        if type(admission_generation) is not int:
            return False
        current_generation = self._current_generation()
        if (
            current_generation is not None
            and current_generation != admission_generation
        ):
            return False
        capability = self._canonical_recording_admission_capability
        claim = getattr(
            self.bus,
            "_claim_canonical_recording_terminal",
            None,
        )
        if capability is None or not callable(claim):
            return False
        identity = claim(capability, command)
        if identity != (command.session_id, admission_generation):
            return False
        with self._lock:
            if identity in self._mark_blocked_admission_terminal_identities:
                return False
            self._mark_blocked_admission_terminal_identities[identity] = None
            while (
                len(self._mark_blocked_admission_terminal_identities)
                > self._recent_identity_limit
            ):
                self._mark_blocked_admission_terminal_identities.popitem(
                    last=False
                )
        self.bus.events.recording_failed.emit(
            RecordingFailed(
                command.session_id,
                reason,
            )
        )
        return False

    def _retire_recording_admission(
        self,
        command: BeginRecordingRequested,
    ) -> bool:
        capability = self._canonical_recording_admission_capability
        retire = getattr(
            self.bus,
            "_retire_canonical_recording_admission",
            None,
        )
        if capability is None or not callable(retire):
            return False
        return bool(retire(capability, command))

    def _replay_recording_admission(self) -> bool:
        capability = self._canonical_recording_admission_capability
        replay = getattr(
            self.bus,
            "_replay_canonical_recording_admission",
            None,
        )
        if capability is None or not callable(replay):
            return False
        command = replay(capability)
        if type(command) is not BeginRecordingRequested:
            return False
        self.bus.commands.begin_recording_requested.emit(command)
        return True

    def _release_recording_admission_capability(self) -> bool:
        capability = self._canonical_recording_admission_capability
        if capability is None:
            return False
        self._canonical_recording_admission_capability = None
        release = getattr(
            self.bus,
            "_release_canonical_recording_consumer",
            None,
        )
        if not callable(release):
            return False
        return bool(release(capability))

    def _finish_mark_action(self, identity: tuple[str, int]) -> None:
        pending = getattr(self.mark_action_service, "pending_recovery", None)
        mark_disconnect_quiesced = False
        with self._lock:
            self._mark_action_inflight_identity = None
            if pending is None:
                self._record_mark_action_terminal_locked(identity)
                if self._mark_disconnect_quiescence_pending:
                    self._mark_disconnect_quiescence_pending = False
                    mark_disconnect_quiesced = True
        if mark_disconnect_quiesced:
            self.disconnect_quiesced.emit("")

    def _retry_mark_action_recovery(self, pending: Any) -> bool:
        command_id = getattr(pending, "command_id", None)
        generation = getattr(pending, "workflow_generation", None)
        if (
            type(command_id) is not str
            or not command_id
            or type(generation) is not int
        ):
            return False
        identity = (command_id, generation)
        with self._lock:
            if self._mark_action_inflight_identity is not None:
                return False
            self._mark_action_inflight_identity = identity
        try:
            completed = bool(self.mark_action_service.retry_pending_recovery())
        except BaseException:
            self._finish_mark_action(identity)
            raise
        self._finish_mark_action(identity)
        return completed

    def _mark_action_command_is_current(
        self, command: RecordingMarkActionRequested
    ) -> bool:
        current_generation = self._current_generation()
        identity = (command.command_id, command.workflow_generation)
        with self._lock:
            return bool(
                self._active
                and not self._disconnecting
                and self._session is None
                and self._pending_admission is None
                and self._mark_action_inflight_identity == identity
                and (
                    current_generation is None
                    or current_generation == command.workflow_generation
                )
            )

    @pyqtSlot(object)
    def handle_mark_action(
        self, command: RecordingMarkActionRequested
    ) -> bool | RecordingMarkActionDispatchPending:
        if self._mark_action_requires_qt_owner_dispatch():
            return self._invoke_mark_action_qt_owner("handle", command)
        if (
            type(command) is not RecordingMarkActionRequested
            or self.mark_action_service is None
        ):
            return False
        if getattr(self.mark_action_service, "pending_recovery", None) is not None:
            return False
        identity = (command.command_id, command.workflow_generation)
        current_generation = self._current_generation()
        with self._lock:
            if (
                not self._active
                or self._disconnecting
                or self._session is not None
                or self._pending_admission is not None
                or self._mark_action_inflight_identity is not None
                or identity in self._mark_action_terminal_identities
                or (
                    current_generation is not None
                    and current_generation != command.workflow_generation
                )
            ):
                return False
            self._mark_action_inflight_identity = identity
        try:
            completed = bool(
                self.mark_action_service.apply(
                    command,
                    admission_current=self._mark_action_command_is_current,
                )
            )
        except BaseException:
            self._finish_mark_action(identity)
            raise
        self._finish_mark_action(identity)
        return completed

    @pyqtSlot(object)
    def handle_commit_recording_label(
        self, command: CommitRecordingLabelRequested
    ) -> bool:
        """Consume one Workflow-authorized label transaction terminally."""
        if (
            type(command) is not CommitRecordingLabelRequested
            or self.label_service is None
        ):
            return False
        command_identity = _continuation_message_identity("label-commit", command)
        if command_identity is None:
            return False
        with self._lock:
            if command.command_id in self._label_terminal_commands:
                return False
            if self._label_inflight_command_id is not None:
                return False
            if (
                self._label_recovery_command_identity is not None
                and command_identity != self._label_recovery_command_identity
            ):
                return False
            self._label_recovery_command_id = command.command_id
            self._label_recovery_command_identity = command_identity
            self._label_inflight_command_id = command.command_id
            self._label_inflight_command_identity = command_identity
        try:
            outcome = self.label_service.commit(command, self.view)
        except BaseException as error:
            if getattr(error, "recording_label_recovery_pending", False) is True:
                with self._lock:
                    self._label_inflight_command_id = None
                    self._label_inflight_command_identity = None
                    self._label_recovery_command_id = getattr(
                        error, "command_id", command.command_id
                    )
                    self._label_recovery_command_identity = getattr(
                        error, "command_identity", command_identity
                    )
                # Recovery has no terminal outcome yet.  Keep the exact
                # Workflow continuation unacknowledged so its bounded
                # single-shot outbox retry redelivers the immutable command.
                return False
            with self._lock:
                self._label_recovery_command_id = None
                self._label_recovery_command_identity = None
                self._label_inflight_command_id = None
                self._label_inflight_command_identity = None
                self._label_terminal_commands[command.command_id] = None
                while len(self._label_terminal_commands) > self._recent_identity_limit:
                    self._label_terminal_commands.popitem(last=False)
            try:
                reason = (str(error) or type(error).__name__)[:512]
            except BaseException:
                reason = "recording label commit failed"
            self.bus.events.recording_label_commit_failed.emit(
                RecordingLabelCommitFailed(
                    command.command_id,
                    command.record_id,
                    command.label,
                    reason,
                    command.export_outcome,
                )
            )
            return True
        with self._lock:
            self._label_recovery_command_id = None
            self._label_recovery_command_identity = None
            self._label_inflight_command_id = None
            self._label_inflight_command_identity = None
            self._label_terminal_commands[command.command_id] = None
            while len(self._label_terminal_commands) > self._recent_identity_limit:
                self._label_terminal_commands.popitem(last=False)
        self.bus.events.recording_label_committed.emit(
            RecordingLabelCommitted(
                command.command_id,
                command.record_id,
                command.label,
                outcome,
            )
        )
        return True

    def request_manual_label(self, sender: Any, *, manual: Any = True) -> bool:
        service = self.manual_label_request_service
        if service is None:
            return False
        return service.request(sender, manual=manual)

    # Deliberately refuse unbound callbacks: late work must retain its session port.
    def recording_completed(self, *, sample_count: int, result_snapshot: Any) -> bool:
        return False

    def recording_failed(self, reason: Any) -> bool:
        return False
