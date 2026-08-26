"""Pure staging service for Workflow-admitted imported-audio recording loads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import PurePath
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable

import numpy as np

from base.load_audio import load_audio_preserve_rate
from base.stimulus_resolver import set_data_struct_analysis_reference_signal
from base.wav_calibration_metadata import read_wav_calibration_metadata
from ui.sequence.sequence_messages import LoadImportedAudioRequested


@dataclass(frozen=True, slots=True)
class ImportedAudioStage:
    """Detached import result that is safe to commit as one transaction."""

    file_path: str
    mode: str
    sample_rate: Any
    audio_multi: np.ndarray
    audio_mono: np.ndarray
    sample_count: int
    calibration_metadata: Any
    reference: Any = None


@dataclass(frozen=True, slots=True)
class ImportedAudioReference:
    """Whitelisted, detached reference state prepared by the legacy builder."""

    sample_rate: Any
    stimulus_data: Any = None
    stimulus_info: Any = None
    alignment_sample_count: Any = None
    alignment_sample_count_present: bool = False
    total_time: Any = None


class AudioImportFailure(RuntimeError):
    """Normalized service failure with stable user feedback."""

    def __init__(
        self,
        reason: str,
        title: str,
        user_message: str,
        *,
        clear_runtime: bool = True,
    ) -> None:
        super().__init__(reason)
        self.reason = str(reason or "audio import failed")
        self.title = str(title or "提示")
        self.user_message = str(user_message or self.reason)
        self.clear_runtime = bool(clear_runtime)


class AudioImportLifecycleClosed(RuntimeError):
    """Abort pure staging when its owning Recording transaction closes."""


def _mutable_snapshot(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            _mutable_snapshot(key): _mutable_snapshot(item)
            for key, item in value.items()
        }
    if type(value) in {list, tuple}:
        return [_mutable_snapshot(item) for item in value]
    if type(value) is frozenset:
        return {_mutable_snapshot(item) for item in value}
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, PurePath):
        return type(value)(value)
    if value is None or type(value) in {bool, int, float, str, bytes}:
        return value
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("import configuration contains an unsupported value")


def mutable_import_snapshot(value: Any) -> Any:
    """Return a detached legacy-compatible projection of a frozen stage value."""
    return _mutable_snapshot(value)


class _SnapshotBudget:
    __slots__ = ("remaining",)

    def __init__(self, limit: int = 4096) -> None:
        self.remaining = limit

    def consume(self) -> None:
        self.remaining -= 1
        if self.remaining < 0:
            raise ValueError("import snapshot exceeds the supported size")


def _freeze_detached(
    value: Any,
    *,
    budget: _SnapshotBudget,
    depth: int = 0,
    array_byte_limit: int = 8 * 1024 * 1024,
) -> Any:
    budget.consume()
    if depth > 16:
        raise ValueError("import snapshot exceeds the supported depth")
    if value is None or type(value) in {bool, int, str, bytes}:
        if type(value) in {str, bytes} and len(value) > 8192:
            raise ValueError("import snapshot text is oversized")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("import snapshot numbers must be finite")
        return value
    if isinstance(value, np.generic):
        return _freeze_detached(
            value.item(), budget=budget, depth=depth, array_byte_limit=array_byte_limit
        )
    if type(value) is np.ndarray:
        if value.dtype.hasobject or value.nbytes > array_byte_limit:
            raise ValueError("import snapshot array is unsupported or oversized")
        detached = np.array(value, copy=True, order="C")
        if detached.dtype.kind in "fc" and not np.isfinite(detached).all():
            raise ValueError("import snapshot arrays must be finite")
        detached.setflags(write=False)
        return detached
    if type(value) is dict:
        return MappingProxyType(
            {
                _freeze_detached(
                    key, budget=budget, depth=depth + 1, array_byte_limit=array_byte_limit
                ): _freeze_detached(
                    item, budget=budget, depth=depth + 1, array_byte_limit=array_byte_limit
                )
                for key, item in dict.items(value)
            }
        )
    if type(value) in {list, tuple}:
        iterator = list.__iter__(value) if type(value) is list else tuple.__iter__(value)
        return tuple(
            _freeze_detached(
                item, budget=budget, depth=depth + 1, array_byte_limit=array_byte_limit
            )
            for item in iterator
        )
    if type(value) in {set, frozenset}:
        iterator = set.__iter__(value) if type(value) is set else frozenset.__iter__(value)
        return frozenset(
            _freeze_detached(
                item, budget=budget, depth=depth + 1, array_byte_limit=array_byte_limit
            )
            for item in iterator
        )
    if isinstance(value, PurePath):
        return str(value)
    raise TypeError("import snapshot contains an unsupported value")


def _positive_finite_sample_rate(value: Any) -> int | float:
    if type(value) is bool or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("sample rate must be numeric")
    normalized = value.item() if isinstance(value, np.generic) else value
    if not math.isfinite(float(normalized)) or normalized <= 0:
        raise ValueError("sample rate must be positive and finite")
    return normalized


def _owned_audio(decoded: Any) -> tuple[np.ndarray, np.ndarray, int]:
    source = np.array(decoded, dtype=np.float32, copy=True, order="C")
    if source.ndim == 1:
        if source.size == 0:
            raise ValueError("decoded audio is empty")
        multi = source.reshape(-1, 1).copy(order="C")
    elif source.ndim == 2:
        if 0 in source.shape:
            raise ValueError("decoded audio is empty")
        # load_audio_preserve_rate returns channel-major decoded audio.
        multi = source.T.copy(order="C")
    else:
        raise ValueError("decoded audio must be one- or two-dimensional")
    if not np.isfinite(multi).all():
        raise ValueError("decoded audio must contain only finite samples")
    mono = np.asarray(multi.mean(axis=1), dtype=np.float32).copy(order="C")
    if mono.shape != (multi.shape[0],):
        raise ValueError("decoded mono projection is inconsistent")
    multi.setflags(write=False)
    mono.setflags(write=False)
    return multi, mono, int(multi.shape[0])


def _freeze_reference(reference: Any, sample_rate: Any) -> ImportedAudioReference:
    stimulus_data = _freeze_detached(
        getattr(reference, "stimulus_data", None),
        budget=_SnapshotBudget(),
        array_byte_limit=256 * 1024 * 1024,
    )
    stimulus_info = _freeze_detached(
        getattr(reference, "stimulus_info", None), budget=_SnapshotBudget()
    )
    alignment_present = hasattr(reference, "alignment_sample_count")
    alignment = (
        getattr(reference, "alignment_sample_count")
        if alignment_present
        else None
    )
    if alignment is not None:
        if type(alignment) is bool or not isinstance(alignment, (int, np.integer)):
            raise ValueError("reference alignment sample count is invalid")
        alignment = int(alignment)
        if alignment < 0:
            raise ValueError("reference alignment sample count is invalid")
    total_time = getattr(reference, "total_time", None)
    if total_time is not None:
        if type(total_time) is bool or not isinstance(
            total_time, (int, float, np.integer, np.floating)
        ):
            raise ValueError("reference total time is invalid")
        total_time = float(total_time)
        if not math.isfinite(total_time) or total_time < 0:
            raise ValueError("reference total time is invalid")
    return ImportedAudioReference(
        sample_rate=sample_rate,
        stimulus_data=stimulus_data,
        stimulus_info=stimulus_info,
        alignment_sample_count=alignment,
        alignment_sample_count_present=alignment_present,
        total_time=total_time,
    )


class SequenceImportedAudioService:
    """Decode and prepare an import without mutating main-thread runtime state."""

    def __init__(
        self,
        *,
        audio_loader: Callable[..., Any] = load_audio_preserve_rate,
        metadata_reader: Callable[..., Any] = read_wav_calibration_metadata,
        reference_builder: Callable[..., Any] = (
            set_data_struct_analysis_reference_signal
        ),
        reference_logger: Any = None,
        logger: Any = None,
    ) -> None:
        self.audio_loader = audio_loader
        self.metadata_reader = metadata_reader
        self.reference_builder = reference_builder
        self.reference_logger = reference_logger
        self.logger = logger

    def _log(self, level: str, message: str) -> None:
        callback = getattr(self.logger, level, None)
        if callable(callback):
            callback(message)

    @staticmethod
    def _require_owner_live(
        boundary_check: Callable[[], bool] | None,
    ) -> None:
        if boundary_check is not None and boundary_check() is not True:
            raise AudioImportLifecycleClosed("audio import owner closed")

    @staticmethod
    def _acquisition_detail(command: LoadImportedAudioRequested) -> Any:
        snapshot = command.configuration_snapshot
        try:
            acquisition = snapshot.sequence_config[0]["seq1"]["acq"]
            detail = acquisition["detail"]
        except (AttributeError, IndexError, KeyError, TypeError) as error:
            raise AudioImportFailure(
                "import configuration is incomplete",
                "提示",
                "导入配置不完整，请重新选择配置。",
            ) from error
        return _mutable_snapshot(detail)

    def load(
        self,
        command: LoadImportedAudioRequested,
        selected_path: str,
        *,
        boundary_check: Callable[[], bool] | None = None,
    ) -> ImportedAudioStage:
        if type(command) is not LoadImportedAudioRequested:
            raise TypeError("command must be LoadImportedAudioRequested")
        file_path = str(selected_path)
        try:
            decoded, sample_rate = self.audio_loader(file_path, mono=False)
        except Exception as error:
            raise AudioImportFailure(
                "audio import failed",
                "提示",
                "导入音频失败，请重新选择音频文件。",
            ) from error
        self._require_owner_live(boundary_check)
        if decoded is None or sample_rate is None:
            raise AudioImportFailure(
                "audio import failed",
                "提示",
                "导入音频失败，请重新选择音频文件。",
            )

        try:
            calibration_metadata_source = self.metadata_reader(
                file_path, logger=self.logger
            )
        except Exception as error:
            self._log("warning", f"读取WAV校准数据失败: {error}")
            calibration_metadata_source = None
        self._require_owner_live(boundary_check)

        try:
            sample_rate = _positive_finite_sample_rate(sample_rate)
            audio_multi, audio_mono, sample_count = _owned_audio(decoded)
            calibration_metadata = (
                None
                if calibration_metadata_source is None
                else _freeze_detached(
                    calibration_metadata_source, budget=_SnapshotBudget()
                )
            )
        except (IndexError, TypeError, ValueError, OverflowError) as error:
            raise AudioImportFailure(
                "audio import failed",
                "提示",
                "导入音频失败，请重新选择音频文件。",
            ) from error
        self._require_owner_live(boundary_check)

        reference = None
        if command.mode == "IMPORT_STIMULUS_AUDIO":
            detail = self._acquisition_detail(command)
            reference = SimpleNamespace(sample_rate=sample_rate)
            try:
                ready = self.reference_builder(
                    reference,
                    detail,
                    using_config_path=(
                        command.configuration_snapshot.using_config_path
                    ),
                    runtime_sample_rate=sample_rate,
                    logger=self.reference_logger,
                )
            except Exception as error:
                detail_text = str(error)[:200]
                raise AudioImportFailure(
                    f"analysis reference load failed: {detail_text}",
                    "提示",
                    f"加载分析参考激励失败: {detail_text}",
                ) from error
            self._require_owner_live(boundary_check)
            if not ready:
                raise AudioImportFailure(
                    "analysis reference load failed",
                    "提示",
                    "加载分析参考激励失败，请检查激励配置。",
                )
            try:
                reference = _freeze_reference(reference, sample_rate)
            except (AttributeError, TypeError, ValueError, OverflowError) as error:
                raise AudioImportFailure(
                    "analysis reference snapshot failed",
                    "提示",
                    "加载分析参考激励失败，请检查激励配置。",
                ) from error
            self._require_owner_live(boundary_check)

        self._require_owner_live(boundary_check)
        return ImportedAudioStage(
            file_path=file_path,
            mode=command.mode,
            sample_rate=sample_rate,
            audio_multi=audio_multi,
            audio_mono=audio_mono,
            sample_count=sample_count,
            calibration_metadata=calibration_metadata,
            reference=reference,
        )
