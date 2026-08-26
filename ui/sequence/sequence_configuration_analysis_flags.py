"""Configuration-owned projection of analysis FFT/STFT runtime flags."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Protocol

from ui.sequence.sequence_messages import _FrozenMapping


_FFT_ANALYSIS_TYPES = frozenset(
    {
        "频响 (FR) ",
        "谐波失真 (HD) ",
        "快速傅里叶变换 (FFT) ",
        "FR",
        "HD",
        "FFT",
    }
)
_STFT_ANALYSIS_TYPES = frozenset({"频谱分析 (Spec) ", "Spec"})


@dataclass(frozen=True, slots=True)
class AnalysisFlagProjection:
    """Immutable analysis-runtime flag values derived from one config."""

    fft_flag: int
    stft_flag: int


@dataclass(frozen=True, slots=True)
class AnalysisFlagRuntimeCheckpoint:
    """Exact existence/value checkpoint for the two projected fields."""

    fft_existed: bool
    fft_value: Any
    stft_existed: bool
    stft_value: Any


class AnalysisFlagProjectionRejected(RuntimeError):
    """The runtime port rejected or violated the projection contract."""


class AnalysisFlagProjectionRollbackError(RuntimeError):
    """The runtime port could not confirm restoration of a checkpoint."""


class AnalysisFlagProjectionPort(Protocol):
    def capture(self) -> AnalysisFlagRuntimeCheckpoint: ...

    def apply(self, projection: AnalysisFlagProjection) -> bool | None: ...

    def restore(
        self, checkpoint: AnalysisFlagRuntimeCheckpoint
    ) -> bool | None: ...


class DataStructAnalysisFlagProjectionPort:
    """Narrow adapter for the stable, main-thread-owned DataDealStruct."""

    def __init__(self, data_struct: Any) -> None:
        self._data_struct = data_struct

    def capture(self) -> AnalysisFlagRuntimeCheckpoint:
        state = vars(self._data_struct)
        return AnalysisFlagRuntimeCheckpoint(
            fft_existed="fft_flag" in state,
            fft_value=state.get("fft_flag"),
            stft_existed="stft_flag" in state,
            stft_value=state.get("stft_flag"),
        )

    def apply(self, projection: AnalysisFlagProjection) -> None:
        state = vars(self._data_struct)
        state["fft_flag"] = projection.fft_flag
        state["stft_flag"] = projection.stft_flag

    def restore(self, checkpoint: AnalysisFlagRuntimeCheckpoint) -> None:
        if type(checkpoint) is not AnalysisFlagRuntimeCheckpoint:
            raise TypeError("checkpoint must be an AnalysisFlagRuntimeCheckpoint")
        state = vars(self._data_struct)
        if checkpoint.fft_existed:
            state["fft_flag"] = checkpoint.fft_value
        else:
            state.pop("fft_flag", None)
        if checkpoint.stft_existed:
            state["stft_flag"] = checkpoint.stft_value
        else:
            state.pop("stft_flag", None)


def _mapping_get(mapping: Any, key: str, default: Any = None) -> Any:
    if type(mapping) is dict:
        return dict.get(mapping, key, default)
    if type(mapping) is _FrozenMapping:
        try:
            return mapping[key]
        except KeyError:
            return default
    raise TypeError("analysis configuration must be a plain or frozen mapping")


class SequenceAnalysisFlagProjectionService:
    """Derive and transactionally apply analysis flags for a config snapshot."""

    def __init__(self, port: AnalysisFlagProjectionPort) -> None:
        self._port = port
        self._lock = RLock()
        self._active = False

    @staticmethod
    def _require_port_success(
        outcome: Any,
        *,
        operation: str,
        error_type: type[RuntimeError],
    ) -> None:
        if outcome is None or outcome is True:
            return
        raise error_type(
            f"analysis flag {operation} rejected completion "
            f"with {type(outcome).__name__}"
        )

    def _restore_port_checkpoint(
        self, checkpoint: AnalysisFlagRuntimeCheckpoint
    ) -> None:
        try:
            outcome = self._port.restore(checkpoint)
        except BaseException as restore_error:
            if not isinstance(restore_error, Exception):
                raise
            rollback_error = AnalysisFlagProjectionRollbackError(
                "analysis flag rollback failed with "
                f"{type(restore_error).__name__}"
            )
            raise rollback_error from restore_error
        self._require_port_success(
            outcome,
            operation="rollback",
            error_type=AnalysisFlagProjectionRollbackError,
        )

    @staticmethod
    def derive(analysis_config: Any) -> AnalysisFlagProjection:
        display_sequence = _mapping_get(
            analysis_config,
            "display_sequence",
            (),
        )
        if type(display_sequence) not in {list, tuple}:
            raise TypeError("display_sequence must be a plain or frozen sequence")

        fft_flag = 0
        stft_flag = 0
        for item_name in display_sequence:
            if type(item_name) is not str:
                raise TypeError("analysis item names must be strings")
            missing = object()
            item = _mapping_get(analysis_config, item_name, missing)
            if item is missing:
                raise ValueError(f"analysis item is missing: {item_name}")
            item_type = _mapping_get(item, "type", missing)
            if item_type is missing:
                raise ValueError(f"analysis item type is missing: {item_name}")
            if type(item_type) is not str:
                raise TypeError("analysis item type must be a string")
            if item_type in _FFT_ANALYSIS_TYPES:
                fft_flag += 1
            elif item_type in _STFT_ANALYSIS_TYPES:
                stft_flag += 1
        return AnalysisFlagProjection(fft_flag=fft_flag, stft_flag=stft_flag)

    def capture_runtime_state(self) -> AnalysisFlagRuntimeCheckpoint:
        with self._lock:
            if self._active:
                raise RuntimeError("analysis flag projection is already active")
            return self._port.capture()

    def restore_runtime_state(
        self, checkpoint: AnalysisFlagRuntimeCheckpoint
    ) -> None:
        with self._lock:
            if self._active:
                raise RuntimeError("analysis flag projection is already active")
            self._active = True
            try:
                self._restore_port_checkpoint(checkpoint)
            finally:
                self._active = False

    def project(self, analysis_config: Any) -> AnalysisFlagProjection:
        with self._lock:
            if self._active:
                raise RuntimeError("analysis flag projection is already active")
            self._active = True
            checkpoint: AnalysisFlagRuntimeCheckpoint | None = None
            try:
                projection = self.derive(analysis_config)
                checkpoint = self._port.capture()
                outcome = self._port.apply(projection)
                self._require_port_success(
                    outcome,
                    operation="apply",
                    error_type=AnalysisFlagProjectionRejected,
                )
                return projection
            except BaseException as primary_error:
                primary_traceback = primary_error.__traceback__
                if checkpoint is not None:
                    try:
                        self._restore_port_checkpoint(checkpoint)
                    except BaseException as restore_error:
                        if not isinstance(primary_error, Exception):
                            try:
                                BaseException.add_note(
                                    primary_error,
                                    "analysis flag rollback also failed with "
                                    f"{type(restore_error).__name__}",
                                )
                            except BaseException:
                                pass
                            raise primary_error.with_traceback(
                                primary_traceback
                            ) from restore_error
                        if not isinstance(restore_error, Exception):
                            try:
                                BaseException.add_note(
                                    restore_error,
                                    "analysis flag projection also failed with "
                                    f"{type(primary_error).__name__}",
                                )
                            except BaseException:
                                pass
                            raise restore_error from primary_error
                        if isinstance(
                            restore_error,
                            AnalysisFlagProjectionRollbackError,
                        ):
                            try:
                                BaseException.add_note(
                                    restore_error,
                                    "analysis flag projection also failed with "
                                    f"{type(primary_error).__name__}",
                                )
                            except BaseException:
                                pass
                            raise
                        try:
                            BaseException.add_note(
                                primary_error,
                                "analysis flag projection rollback also failed: "
                                f"{type(restore_error).__name__}: {restore_error}",
                            )
                        except BaseException:
                            pass
                raise
            finally:
                self._active = False
