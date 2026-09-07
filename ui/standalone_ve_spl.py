"""Standalone VE WAV SPL configuration, preflight, worker, and result UI."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import json
from numbers import Integral
import os
from pathlib import Path
import tempfile
import threading
from types import MappingProxyType
from typing import Any, Callable

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
from scipy.io import wavfile
from scipy.signal import lfilter

from base.core_algorithm.harmonic_distortion.weighted import (
    A_weighting_filter,
    B_weighting_filter,
    C_weighting_filter,
    D_weighting_filter,
)

from base.pre_processing.spl_runtime_config import (
    resolve_spl_limit_data,
    resolve_spl_overall_limit_values,
)
from base.pre_processing.standalone_spl_analysis import (
    StandaloneSplResult,
    analyze_standalone_spl,
)
from base.ve3668n_wav_metadata import (
    resolve_ve_wav_channel_v2pa_factor,
    validate_ve_wav_metadata,
)
from base.wav_calibration_metadata import (
    WavCalibrationMetadataReadStatus,
    inspect_wav_calibration_metadata,
)
from ui.plot_view import apply_plot_view_range
from ui.ui_analysis_config.spl_config_dialog import SplConfigWindow


DEFAULT_STANDALONE_SPL_CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "ui_config"
    / "record_vkinging_spl_config.json"
)
DEFAULT_MAX_PLOT_POINTS = 12000
STANDALONE_SAMPLE_RATE = 51200


class AnalysisAdmissionError(ValueError):
    """The current WAV cannot safely start a standalone SPL batch."""


def _immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a recursively owned, immutable configuration snapshot."""
    def freeze(item):
        if isinstance(item, Mapping):
            return MappingProxyType({str(key): freeze(child) for key, child in item.items()})
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        if isinstance(item, tuple):
            return tuple(freeze(child) for child in item)
        return deepcopy(item)

    return freeze(dict(value))


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    def thaw(item):
        if isinstance(item, Mapping):
            return {str(key): thaw(child) for key, child in item.items()}
        if isinstance(item, tuple):
            return [thaw(child) for child in item]
        return deepcopy(item)

    return thaw(value)


def _finite_nonnegative(value, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a finite nonnegative number")
    try:
        resolved = float(value or 0.0)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite nonnegative number") from exc
    if not np.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return resolved


def validate_standalone_spl_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate worker-relevant SPL semantics and return an immutable snapshot."""
    if not isinstance(config, Mapping):
        raise ValueError("SPL configuration must be a mapping")
    resolved = _plain_mapping(config)
    weighting = str(resolved.get("weighting", "Z") or "Z").strip().upper()
    if weighting not in {"NONE", "Z", "A", "B", "C", "D"}:
        raise ValueError(f"unsupported SPL weighting: {weighting}")
    resolved["weighting"] = weighting

    if bool(resolved.get("analysis_time_range_enabled", False)):
        start = _finite_nonnegative(
            resolved.get("analysis_start_time_sec", 0.0), "analysis start time"
        )
        end = _finite_nonnegative(
            resolved.get("analysis_end_time_sec", 0.0), "analysis end time"
        )
        if end != 0.0 and end <= start:
            raise ValueError("analysis end must be greater than analysis start")

    if bool(resolved.get("limit_checked", False)):
        metric = str(resolved.get("limit_metric", "overall_spl") or "overall_spl").lower()
        try:
            if metric == "overall_spl":
                resolve_spl_overall_limit_values(resolved)
            elif metric == "curve_y":
                resolve_spl_limit_data(resolved, np.asarray([0.0], dtype=float))
            else:
                raise ValueError(f"unsupported SPL limit metric: {metric}")
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{metric} SPL threshold configuration is invalid: {exc}"
            ) from exc
        resolved["limit_metric"] = metric
    return _immutable_mapping(resolved)


class StandaloneSplConfigOwner:
    """Own the standalone SPL file and its last read-back-verified snapshot."""

    def __init__(self, config_path=DEFAULT_STANDALONE_SPL_CONFIG_PATH, *, dialog_factory=None):
        self.path = Path(config_path)
        self._dialog_factory = dialog_factory or SplConfigWindow
        self._active_config: Mapping[str, Any] | None = None
        self._proposed_config: Mapping[str, Any] | None = None
        self.diagnostic: str | None = None
        self._load_initial()

    @property
    def active_config(self) -> Mapping[str, Any] | None:
        return self._active_config

    @property
    def analyze_available(self) -> bool:
        return self._active_config is not None

    def _load_initial(self) -> None:
        if not self.path.exists():
            self._active_config = validate_standalone_spl_config(
                SplConfigWindow.new_item_default_config()
            )
            return
        try:
            self._active_config = self._readback()
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            self.diagnostic = f"Standalone SPL configuration requires recovery: {exc}"

    def _readback(self) -> Mapping[str, Any]:
        return self._readback_path(self.path)

    def _readback_path(self, path: Path) -> Mapping[str, Any]:
        with Path(path).open("r", encoding="utf-8") as source:
            payload = json.load(source)
        if not isinstance(payload, dict) or set(payload) != {"SPL"}:
            raise ValueError("configuration must contain only the SPL key")
        return validate_standalone_spl_config(payload["SPL"])

    def _atomic_write(self, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as target:
                json.dump(payload, target, ensure_ascii=False, indent=2)
                target.write("\n")
                target.flush()
                os.fsync(target.fileno())
            self._readback_path(Path(temporary_name))
            os.replace(temporary_name, self.path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    def _restore_persisted_state(
        self,
        *,
        existed: bool,
        content: bytes | None,
    ) -> None:
        if not existed:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            return
        if content is None:
            raise RuntimeError("prior configuration bytes are unavailable")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.restore.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        try:
            with os.fdopen(descriptor, "wb") as target:
                target.write(content)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary_name, self.path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    # SplConfigWindow manager surface. Default-button edits remain provisional
    # until the outer dialog itself is explicitly accepted.
    def load_config(self) -> dict[str, Any]:
        proposed = self._proposed_config or self._active_config
        if proposed is None:
            proposed = validate_standalone_spl_config(
                SplConfigWindow.new_item_default_config()
            )
        return {"SPL": _plain_mapping(proposed)}

    def save_default_config(self, model_type, config) -> bool:
        if model_type != "SPL":
            return False
        try:
            self._proposed_config = validate_standalone_spl_config(config)
        except ValueError:
            return False
        return True

    def accept_config(self, config: Mapping[str, Any]) -> bool:
        try:
            prior_existed = self.path.exists()
            prior_content = self.path.read_bytes() if prior_existed else None
        except OSError as exc:
            self.diagnostic = (
                "Standalone SPL configuration was not activated: "
                f"could not snapshot prior persisted state: {exc}"
            )
            self._proposed_config = None
            return False
        committed = False
        try:
            candidate = validate_standalone_spl_config(config)
            self._atomic_write({"SPL": _plain_mapping(candidate)})
            committed = True
            readback = self._readback()
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            primary = str(exc)
            rollback_error = None
            if committed:
                try:
                    self._restore_persisted_state(
                        existed=prior_existed,
                        content=prior_content,
                    )
                except (OSError, RuntimeError) as restore_exc:
                    rollback_error = str(restore_exc)
            self.diagnostic = f"Standalone SPL configuration was not activated: {primary}"
            if rollback_error:
                self.diagnostic += f"; rollback failed: {rollback_error}"
            self._proposed_config = None
            return False
        self._active_config = readback
        self._proposed_config = None
        self.diagnostic = None
        return True

    def cancel_config(self) -> None:
        self._proposed_config = None

    def open_dialog(self, parent=None) -> bool:
        application = QtWidgets.QApplication.instance()
        if (
            application is not None
            and QtCore.QThread.currentThread() is not application.thread()
        ):
            raise RuntimeError("SPL parameters must be opened on the GUI thread")
        # The existing constructor intentionally receives no available_channels.
        dialog = self._dialog_factory(self, "SPL")
        if parent is not None and hasattr(dialog, "setParent"):
            dialog.setParent(parent, dialog.windowFlags() | QtCore.Qt.Dialog)
        if dialog.exec_() != getattr(dialog, "Accepted", QtWidgets.QDialog.Accepted):
            self.cancel_config()
            return False
        return self.accept_config(dialog.get_default_config())


@dataclass(frozen=True)
class StandaloneSplChannel:
    wav_channel_index: int
    physical_input_channel: int
    label: str
    v2pa_factor: float


@dataclass(frozen=True)
class StandaloneSplAdmission:
    path: Path
    sample_rate: int
    frame_count: int
    duration_seconds: float
    range_start_sample: int
    range_stop_sample: int
    channels: tuple[StandaloneSplChannel, ...]
    config: Mapping[str, Any]


def _close_memmap(array) -> None:
    current = array
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, np.memmap):
            mapping = getattr(current, "_mmap", None)
            if mapping is not None:
                mapping.close()
            return
        current = getattr(current, "base", None)


def _wav_shape(array: np.ndarray) -> tuple[int, int]:
    if array.ndim == 1:
        return int(array.shape[0]), 1
    if array.ndim == 2:
        return int(array.shape[0]), int(array.shape[1])
    raise AnalysisAdmissionError("current WAV must contain frame-major audio")


def _resolve_admission_range(frame_count: int, config: Mapping[str, Any]) -> tuple[int, int]:
    if not bool(config.get("analysis_time_range_enabled", False)):
        return 0, frame_count
    duration = frame_count / STANDALONE_SAMPLE_RATE
    start_seconds = _finite_nonnegative(
        config.get("analysis_start_time_sec", 0.0), "analysis start time"
    )
    end_seconds = _finite_nonnegative(
        config.get("analysis_end_time_sec", 0.0), "analysis end time"
    )
    if start_seconds >= duration:
        raise AnalysisAdmissionError(
            "analysis range rejected: actual duration "
            f"{duration:.6f}s, configured interval [{start_seconds}, {end_seconds}]s"
        )
    try:
        start = int(np.floor(start_seconds * STANDALONE_SAMPLE_RATE))
    except (OverflowError, TypeError, ValueError) as exc:
        raise AnalysisAdmissionError(
            "analysis range rejected: actual duration "
            f"{duration:.6f}s, configured interval [{start_seconds}, {end_seconds}]s"
        ) from exc
    stop = (
        frame_count
        if end_seconds == 0.0 or end_seconds >= duration
        else int(np.ceil(end_seconds * STANDALONE_SAMPLE_RATE))
    )
    if start >= frame_count or stop <= start:
        raise AnalysisAdmissionError(
            "analysis range rejected: actual duration "
            f"{duration:.6f}s, configured interval [{start_seconds}, {end_seconds}]s"
        )
    return start, stop


def admit_standalone_spl_analysis(
    current_path,
    expected_physical_channels,
    active_config,
) -> StandaloneSplAdmission:
    """Perform complete file-local batch admission before a worker is created."""
    if current_path is None:
        raise AnalysisAdmissionError("there is no current recorded WAV")
    path = Path(current_path)
    if not path.is_file():
        raise AnalysisAdmissionError("the current recorded WAV is not a readable file")
    try:
        raw_expected = tuple(expected_physical_channels)
    except TypeError as exc:
        raise AnalysisAdmissionError("expected physical channels are invalid") from exc
    if any(
        isinstance(channel, (bool, np.bool_)) or not isinstance(channel, Integral)
        for channel in raw_expected
    ):
        raise AnalysisAdmissionError("expected physical channels are invalid")
    expected = tuple(int(channel) for channel in raw_expected)
    if not expected or len(set(expected)) != len(expected) or any(channel < 0 for channel in expected):
        raise AnalysisAdmissionError("expected physical channels are invalid")
    try:
        config = validate_standalone_spl_config(active_config)
    except ValueError as exc:
        raise AnalysisAdmissionError(f"active SPL configuration is invalid: {exc}") from exc

    mapped = None
    try:
        rate, mapped = wavfile.read(path, mmap=True)
        frame_count, channel_count = _wav_shape(mapped)
        if rate != STANDALONE_SAMPLE_RATE:
            raise AnalysisAdmissionError("current WAV sample rate must be exactly 51200 Hz")
        if mapped.dtype != np.dtype("float32"):
            raise AnalysisAdmissionError("current WAV format must be FLOAT/float32")
        if frame_count <= 0:
            raise AnalysisAdmissionError("current WAV must not be empty")
        if channel_count != len(expected):
            raise AnalysisAdmissionError("current WAV channel count does not match the recording")
    except AnalysisAdmissionError:
        raise
    except Exception as exc:
        # scipy's WAV parser can surface several ordinary implementation-level
        # exceptions for malformed RIFF input; admission exposes one stable
        # diagnostic while deliberately leaving BaseException controls alone.
        detail = str(exc) or type(exc).__name__
        raise AnalysisAdmissionError(f"current WAV cannot be read: {detail}") from exc
    finally:
        if mapped is not None:
            _close_memmap(mapped)
            del mapped

    diagnostic = inspect_wav_calibration_metadata(path)
    if (
        diagnostic.status is not WavCalibrationMetadataReadStatus.VALID
        or diagnostic.metadata is None
        or not diagnostic.handles_released
    ):
        detail = diagnostic.primary_error or "; ".join(diagnostic.close_errors)
        labels = ", ".join(f"In{channel + 1}" for channel in expected)
        raise AnalysisAdmissionError(
            f"current WAV has invalid embedded VE metadata (channels: {labels})"
            + (f": {detail}" if detail else "")
        )
    try:
        metadata = validate_ve_wav_metadata(diagnostic.metadata)
    except ValueError as exc:
        raise AnalysisAdmissionError(f"current WAV has invalid embedded VE metadata: {exc}") from exc
    metadata_rate = metadata["acquisition"]["sample_rate"]
    if metadata_rate != rate or metadata_rate != STANDALONE_SAMPLE_RATE:
        raise AnalysisAdmissionError(
            "embedded VE metadata acquisition sample rate "
            f"{metadata_rate} Hz does not match the actual WAV rate {rate} Hz "
            f"and required fixed rate {STANDALONE_SAMPLE_RATE} Hz"
        )
    entries = sorted(metadata["recorded_channels"], key=lambda item: item["wav_channel_index"])
    physical = tuple(item["physical_input_channel"] for item in entries)
    if physical != expected:
        raise AnalysisAdmissionError("embedded VE channel order does not match the current recording")

    channels = []
    unavailable = []
    for index, entry in enumerate(entries):
        label = f"In{entry['physical_input_channel'] + 1}"
        resolution = resolve_ve_wav_channel_v2pa_factor(metadata, index)
        if resolution.factor is None or resolution.state != "measured":
            unavailable.append(label)
            continue
        channels.append(
            StandaloneSplChannel(
                wav_channel_index=index,
                physical_input_channel=entry["physical_input_channel"],
                label=label,
                v2pa_factor=float(resolution.factor),
            )
        )
    if unavailable:
        raise AnalysisAdmissionError(
            "uncalibrated or invalid channels: "
            + ", ".join(unavailable)
            + "; calibrate them and make a new recording"
        )
    start, stop = _resolve_admission_range(frame_count, config)
    return StandaloneSplAdmission(
        path=path,
        sample_rate=STANDALONE_SAMPLE_RATE,
        frame_count=frame_count,
        duration_seconds=frame_count / STANDALONE_SAMPLE_RATE,
        range_start_sample=start,
        range_stop_sample=stop,
        channels=tuple(channels),
        config=config,
    )


@contextmanager
def mapped_wav_channel(
    path: Path,
    wav_channel_index: int,
    start_sample: int,
    stop_sample: int,
    *,
    weighting: str = "Z",
    chunk_frames: int = 65536,
):
    """Map one WAV and expose one bounded, causally weighted channel range."""
    if (
        isinstance(chunk_frames, (bool, np.bool_))
        or not isinstance(chunk_frames, Integral)
        or int(chunk_frames) <= 0
    ):
        raise ValueError("chunk_frames must be a positive integer")
    chunk_frames = int(chunk_frames)
    normalized_weighting = str(weighting or "Z").strip().upper()
    mapped = None
    channel = None
    try:
        rate, mapped = wavfile.read(path, mmap=True)
        if rate != STANDALONE_SAMPLE_RATE:
            raise ValueError("WAV rate changed after admission")
        if normalized_weighting in {"NONE", "Z"}:
            channel = (
                mapped[start_sample:stop_sample]
                if mapped.ndim == 1
                else mapped[start_sample:stop_sample, wav_channel_index]
            )
        else:
            coefficient_factory = {
                "A": A_weighting_filter,
                "B": B_weighting_filter,
                "C": C_weighting_filter,
                "D": D_weighting_filter,
            }.get(normalized_weighting)
            if coefficient_factory is None:
                raise ValueError(f"unsupported SPL weighting: {normalized_weighting}")
            b, a = coefficient_factory(rate)
            state = np.zeros(max(len(a), len(b)) - 1, dtype=np.float64)
            channel = np.empty(stop_sample - start_sample, dtype=np.float64)
            written = 0
            for cursor in range(0, stop_sample, chunk_frames):
                chunk_stop = min(cursor + chunk_frames, stop_sample)
                raw_chunk = (
                    mapped[cursor:chunk_stop]
                    if mapped.ndim == 1
                    else mapped[cursor:chunk_stop, wav_channel_index]
                )
                input_chunk = np.asarray(raw_chunk, dtype=np.float64)
                filtered_chunk, state = lfilter(
                    b,
                    a,
                    input_chunk,
                    zi=state,
                )
                retain_start = max(start_sample, cursor) - cursor
                if retain_start < filtered_chunk.size:
                    retained = filtered_chunk[retain_start:]
                    channel[written : written + retained.size] = retained
                    written += retained.size
                    del retained
                del raw_chunk, input_chunk, filtered_chunk
            if written != channel.size:
                raise RuntimeError("weighted WAV range did not produce the admitted frame count")
            del b, a, state
        yield channel
    finally:
        if channel is not None:
            del channel
        if mapped is not None:
            _close_memmap(mapped)
            del mapped


@dataclass(frozen=True)
class StandaloneSplChannelResult:
    channel: StandaloneSplChannel
    result: StandaloneSplResult

    @property
    def label(self) -> str:
        return self.channel.label


@dataclass(frozen=True)
class StandaloneSplBatchSummary:
    succeeded_labels: tuple[str, ...]
    failures: tuple[tuple[str, str], ...]
    cancelled: bool = False


class StandaloneSplAnalysisWorker(QtCore.QObject):
    """Sequential worker that owns at most one channel mapping at a time."""

    channel_succeeded = QtCore.pyqtSignal(object)
    channel_failed = QtCore.pyqtSignal(str, str)
    finished = QtCore.pyqtSignal(object)

    def __init__(
        self,
        admission: StandaloneSplAdmission,
        *,
        analyzer: Callable = analyze_standalone_spl,
        channel_reader: Callable = mapped_wav_channel,
        max_plot_points: int = DEFAULT_MAX_PLOT_POINTS,
        filter_chunk_frames: int = 65536,
    ):
        super().__init__()
        self._admission = admission
        self._analyzer = analyzer
        self._channel_reader = channel_reader
        self._max_plot_points = int(max_plot_points)
        if (
            isinstance(filter_chunk_frames, (bool, np.bool_))
            or not isinstance(filter_chunk_frames, Integral)
            or int(filter_chunk_frames) <= 0
        ):
            raise ValueError("filter_chunk_frames must be a positive integer")
        self._filter_chunk_frames = int(filter_chunk_frames)
        self._cancel_requested = threading.Event()
        worker_config = _plain_mapping(admission.config)
        worker_config["analysis_time_range_enabled"] = False
        self._worker_config = validate_standalone_spl_config(worker_config)

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    @QtCore.pyqtSlot()
    def run(self) -> None:
        successes = []
        failures = []
        for channel in self._admission.channels:
            if self._cancel_requested.is_set():
                break
            try:
                with self._channel_reader(
                    self._admission.path,
                    channel.wav_channel_index,
                    self._admission.range_start_sample,
                    self._admission.range_stop_sample,
                    weighting=self._worker_config.get("weighting", "Z"),
                    chunk_frames=self._filter_chunk_frames,
                ) as voltage:
                    preweighted = str(
                        self._worker_config.get("weighting", "Z") or "Z"
                    ).upper() not in {"NONE", "Z"}
                    result = self._analyzer(
                        voltage,
                        sample_rate=self._admission.sample_rate,
                        v2pa_factor=channel.v2pa_factor,
                        config=self._worker_config,
                        max_plot_points=self._max_plot_points,
                        source_start_sample=self._admission.range_start_sample,
                        preweighted=preweighted,
                    )
                if not isinstance(result, StandaloneSplResult):
                    raise TypeError("standalone analyzer returned an invalid result")
                bounded = StandaloneSplChannelResult(channel, result)
                self.channel_succeeded.emit(bounded)
                successes.append(channel.label)
                del bounded, result
            except Exception as exc:
                message = str(exc) or type(exc).__name__
                failures.append((channel.label, message))
                self.channel_failed.emit(channel.label, message)
        self.finished.emit(
            StandaloneSplBatchSummary(
                tuple(successes),
                tuple(failures),
                cancelled=self._cancel_requested.is_set(),
            )
        )


class StandaloneSplResultWindow(QtWidgets.QWidget):
    """GUI-thread-only view over one already bounded SPL result."""

    def __init__(self, channel_result: StandaloneSplChannelResult, config, parent=None):
        if QtCore.QThread.currentThread() is not QtWidgets.QApplication.instance().thread():
            raise RuntimeError("SPL result windows must be created on the GUI thread")
        super().__init__(parent, QtCore.Qt.Window)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.setWindowTitle(f"SPL - {channel_result.label}")
        layout = QtWidgets.QVBoxLayout(self)
        result = channel_result.result
        self.judgment_label = None
        if result.judged_ok is not None:
            verdict = "OK" if result.judged_ok else "NG"
            deviation = (
                "unavailable"
                if result.deviation_db is None
                else f"{result.deviation_db:.2f} {result.unit}"
            )
            self.judgment_label = QtWidgets.QLabel(
                f"{verdict} | Deviation: {deviation}", self
            )
            self.judgment_label.setObjectName("splJudgmentLabel")
            self.judgment_label.setProperty(
                "judgment", "ok" if result.judged_ok else "ng"
            )
            self.judgment_label.setStyleSheet(
                "font-weight: bold; color: #087f23;"
                if result.judged_ok
                else "font-weight: bold; color: #b00020;"
            )
            layout.addWidget(self.judgment_label)
        self.analysis_plot = pg.PlotWidget(self)
        layout.addWidget(self.analysis_plot)
        display_config = _plain_mapping(config)
        self.analysis_plot.setBackground("white")
        self.analysis_plot.plot(result.time_seconds, result.spl_db, pen=pg.mkPen("b", width=2))
        self.analysis_plot.setLabel("bottom", "Time", units="s")
        self.analysis_plot.setLabel("left", "SPL", units=result.unit)
        self.analysis_plot.showGrid(x=True, y=True, alpha=0.25)
        if bool(display_config.get("show_overall_spl", False)) and result.overall_spl is not None:
            self.analysis_plot.setTitle(f"Overall SPL: {result.overall_spl:.2f} {result.unit}")
        apply_plot_view_range(self.analysis_plot, display_config)


class StandaloneSplController(QtCore.QObject):
    """Narrow GUI coordinator intended for the Task 7 window state machine."""

    availability_changed = QtCore.pyqtSignal(bool)
    analysis_started = QtCore.pyqtSignal()
    analysis_finished = QtCore.pyqtSignal(object)
    analysis_failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        *,
        config_path=DEFAULT_STANDALONE_SPL_CONFIG_PATH,
        parent=None,
        config_owner=None,
        worker_factory=StandaloneSplAnalysisWorker,
        result_window_factory=StandaloneSplResultWindow,
    ):
        super().__init__(parent)
        self.config_owner = config_owner or StandaloneSplConfigOwner(config_path)
        self._worker_factory = worker_factory
        self._result_window_factory = result_window_factory
        self._current_path: Path | None = None
        self._physical_channels: tuple[int, ...] = ()
        self._thread = None
        self._worker = None
        self._pending_summary = None
        self._presentation_failures: dict[str, str] = {}
        self.result_windows = []

    @property
    def analyzing(self) -> bool:
        return self._thread is not None

    @property
    def analyze_available(self) -> bool:
        return (
            not self.analyzing
            and self._current_path is not None
            and bool(self._physical_channels)
            and self.config_owner.analyze_available
        )

    def set_current_run(self, path, physical_channels) -> None:
        if self.analyzing:
            raise RuntimeError("cannot replace the current run while analyzing")
        self._current_path = None if path is None else Path(path)
        self._physical_channels = tuple(physical_channels or ())
        self.availability_changed.emit(self.analyze_available)

    def open_parameters(self) -> bool:
        if self.analyzing:
            raise RuntimeError("cannot change SPL parameters while analyzing")
        accepted = self.config_owner.open_dialog(parent=self.parent())
        self.availability_changed.emit(self.analyze_available)
        return accepted

    def close_result_windows(self) -> None:
        windows = tuple(self.result_windows)
        self.result_windows.clear()
        for window in windows:
            try:
                window.close()
                if hasattr(window, "deleteLater"):
                    window.deleteLater()
            except RuntimeError:
                # A user may already have closed a WA_DeleteOnClose result.
                continue

    def start_analysis(self) -> bool:
        if self.analyzing:
            raise RuntimeError("SPL analysis is already active")
        try:
            admission = admit_standalone_spl_analysis(
                self._current_path,
                self._physical_channels,
                self.config_owner.active_config,
            )
        except AnalysisAdmissionError as exc:
            self.analysis_failed.emit(str(exc))
            return False
        self.close_result_windows()
        self._presentation_failures.clear()
        thread = QtCore.QThread(self)
        worker = self._worker_factory(admission)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.channel_succeeded.connect(self._show_result)
        worker.channel_failed.connect(self._report_channel_failure)
        worker.finished.connect(self._retain_summary)
        worker.finished.connect(thread.quit, type=QtCore.Qt.DirectConnection)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._thread_finished)
        self._thread = thread
        self._worker = worker
        self.analysis_started.emit()
        self.availability_changed.emit(False)
        thread.start()
        return True

    @QtCore.pyqtSlot(object)
    def _show_result(self, channel_result) -> None:
        window = None
        try:
            window = self._result_window_factory(
                channel_result,
                self.config_owner.active_config,
                parent=self.parent(),
            )
            window.show()
        except Exception as exc:
            cleanup_errors = []
            if window is not None:
                try:
                    window.close()
                except Exception as cleanup_exc:
                    cleanup_errors.append(f"close: {cleanup_exc}")
                try:
                    if hasattr(window, "deleteLater"):
                        window.deleteLater()
                except Exception as cleanup_exc:
                    cleanup_errors.append(f"delete: {cleanup_exc}")
            label = channel_result.label
            message = str(exc) or type(exc).__name__
            if cleanup_errors:
                message += "; presentation cleanup failed: " + "; ".join(
                    cleanup_errors
                )
            if label not in self._presentation_failures:
                self._presentation_failures[label] = message
                self.analysis_failed.emit(f"{label}: {message}")
            return
        self.result_windows.append(window)

    @QtCore.pyqtSlot(str, str)
    def _report_channel_failure(self, label: str, message: str) -> None:
        self.analysis_failed.emit(f"{label}: {message}")

    @QtCore.pyqtSlot(object)
    def _retain_summary(self, summary) -> None:
        self._pending_summary = summary

    @QtCore.pyqtSlot()
    def _thread_finished(self) -> None:
        thread = self._thread
        summary = self._pending_summary or StandaloneSplBatchSummary((), ())
        if self._presentation_failures:
            presentation_labels = set(self._presentation_failures)
            succeeded = tuple(
                label
                for label in summary.succeeded_labels
                if label not in presentation_labels
            )
            failures = list(summary.failures)
            failed_labels = {label for label, _message in failures}
            for label, message in self._presentation_failures.items():
                if label not in failed_labels:
                    failures.append((label, message))
                    failed_labels.add(label)
            summary = StandaloneSplBatchSummary(
                succeeded,
                tuple(failures),
                cancelled=summary.cancelled,
            )
        self._thread = None
        self._worker = None
        self._pending_summary = None
        self._presentation_failures.clear()
        if thread is not None:
            thread.deleteLater()
        self.analysis_finished.emit(summary)
        self.availability_changed.emit(self.analyze_available)

    def request_cancel(self) -> None:
        if self._worker is not None:
            self._worker.request_cancel()
