"""Qt-facing presentation adapter for the Recording MVC domain."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from PyQt5 import sip
from PyQt5.QtCore import QObject, QTimer, pyqtSlot
from PyQt5.QtWidgets import QFileDialog

from consts.running_consts import DEFAULT_DIR

from ui.sequence.sequence_recording_model import RecordingStatisticsSnapshot


def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


class SequenceRecordingImportView:
    """Raw picker and projection ports for the Recording-owned import transaction."""

    def __init__(
        self,
        *,
        parent: Any = None,
        warning_presenter: Callable[[str, str], None] | None = None,
        clear_import_plot: Callable[[], Any] | None = None,
        plot_imported_audio: Callable[[Any, Any], Any] | None = None,
        import_data_enabled_setter: Callable[[bool], Any] | None = None,
        import_projection_capturer: Callable[[], Any] | None = None,
        import_projection_restorer: Callable[[Any], Any] | None = None,
        import_plot_projection_restorer: Callable[[Any], Any] | None = None,
    ) -> None:
        self.parent = parent
        self.warning_presenter = warning_presenter
        self.clear_import_plot = clear_import_plot
        self.plot_imported_audio = plot_imported_audio
        self.import_data_enabled_setter = import_data_enabled_setter
        self.import_projection_capturer = import_projection_capturer
        self.import_projection_restorer = import_projection_restorer
        self.import_plot_projection_restorer = import_plot_projection_restorer

    def choose_import_audio_path(self, selected_path: Any) -> str | None:
        if selected_path:
            return str(selected_path)
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self.parent,
            "选择音频文件",
            DEFAULT_DIR + "audio_data/stored_data",
            "WAV Files (*.wav)",
        )
        return str(file_path) if file_path else None

    def present_import_warning(self, title: str, text: str) -> None:
        if callable(self.warning_presenter):
            self.warning_presenter(title, text)

    def capture_import_projection(self) -> Any:
        if callable(self.import_projection_capturer):
            return self.import_projection_capturer()
        return None

    def restore_import_projection(self, checkpoint: Any) -> Any:
        if callable(self.import_projection_restorer):
            return self.import_projection_restorer(checkpoint)
        return None

    def restore_import_plot(self, checkpoint: Any) -> Any:
        if callable(self.import_plot_projection_restorer):
            return self.import_plot_projection_restorer(checkpoint)
        if callable(self.import_projection_restorer):
            return self.import_projection_restorer(checkpoint)
        return None

    def clear_import_projection(self) -> Any:
        if callable(self.clear_import_plot):
            return self.clear_import_plot()
        return None

    def show_imported_audio(self, audio: Any, sample_rate: Any) -> None:
        if callable(self.clear_import_plot):
            self.clear_import_plot()
        if callable(self.plot_imported_audio):
            self.plot_imported_audio(audio, sample_rate)

    def set_import_data_enabled(self, enabled: bool) -> Any:
        if callable(self.import_data_enabled_setter):
            return self.import_data_enabled_setter(bool(enabled))
        return None


class RecordingCountProjection:
    """Commit the count widget and Recording-owned replay count atomically."""

    def __init__(self, model: Any, line_edit: Any) -> None:
        self.model = model
        self.line_edit = line_edit

    def __call__(self, pending_count: int | None) -> None:
        if pending_count is None:
            return
        previous_count = self.model.current_recorded_count
        previous_last_play_count = self.model.last_play_count
        previous_text = self.line_edit.text()
        try:
            normalized = int(pending_count)
            self.model.current_recorded_count = normalized
            self.line_edit.setText(str(normalized))
            self.model.last_play_count = normalized
        except BaseException as error:
            restore_errors = []
            for name, value in (
                ("current_recorded_count", previous_count),
                ("last_play_count", previous_last_play_count),
            ):
                try:
                    setattr(self.model, name, value)
                except BaseException as restore_error:
                    restore_errors.append(f"{name}: {restore_error}")
            try:
                self.line_edit.setText(previous_text)
            except BaseException as restore_error:
                restore_errors.append(f"lineedit_count: {restore_error}")
            if restore_errors:
                if isinstance(error, Exception):
                    raise RuntimeError(
                        f"{error}; count projection restore failed: "
                        + "; ".join(restore_errors)
                    ) from error
                for restore_error in restore_errors:
                    try:
                        error.add_note(
                            f"count projection restore failed: {restore_error}"
                        )
                    except (AttributeError, TypeError):
                        break
            raise


class SequenceRecordingView(QObject):
    """Keep recording presentation callbacks out of controller decision logic."""

    def __init__(
        self,
        *,
        set_recording_locked: Callable[[bool], None] = _noop,
        set_started: Callable[[], None] = _noop,
        set_finished: Callable[[bool], None] = _noop,
        present_error: Callable[[str], None] = _noop,
        present_readiness_warning: Callable[[str, str], None] = _noop,
        plot_recording: Callable[[Any, float], None] = _noop,
        commit_identity: Callable[[Any], None] = _noop,
        commit_label_projection: Callable[[Any, Any], Any] = _noop,
        schedule_waveform_refresh: Callable[[Callable[[], None]], None]
        | None = None,
        logger: Any = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._set_recording_locked = set_recording_locked
        self._set_started = set_started
        self._set_finished = set_finished
        self._present_error = present_error
        self._present_readiness_warning = present_readiness_warning
        self._plot_recording = plot_recording
        self._commit_identity = commit_identity
        self._commit_label_projection = commit_label_projection
        self._schedule_waveform_refresh = (
            schedule_waveform_refresh
            if schedule_waveform_refresh is not None
            else lambda callback: QTimer.singleShot(0, callback)
        )
        self._logger = logger
        self._waveform_session_id: str | None = None
        self._waveform_sample_rate = 1.0
        self._waveform_buffer: np.ndarray | None = None
        self._waveform_accepted_cursor = 0
        self._waveform_display_cursor = 0
        self._pending_waveform_range: tuple[int, int] | None = None
        self._waveform_refresh_scheduled = False
        self._waveform_generation = 0

    def present_readiness_warning(self, title: str, text: str) -> bool:
        """Project one admitted preflight warning without leaking UI failures."""
        try:
            outcome = self._present_readiness_warning(title, text)
            return outcome is not False
        except BaseException as error:
            log = getattr(self._logger, "warning", None)
            if callable(log):
                try:
                    log(f"recording readiness warning presentation failed: {error}")
                except BaseException:
                    pass
            return False

    @property
    def waveform_display_cursor(self) -> int:
        return self._waveform_display_cursor

    @property
    def pending_waveform_range(self) -> tuple[int, int] | None:
        return self._pending_waveform_range

    def _present(self, name: str, callback: Callable[..., Any], *args: Any) -> bool:
        try:
            callback(*args)
            return True
        except Exception as error:
            log = getattr(self._logger, "warning", None)
            if callable(log):
                log(f"recording view {name} failed: {error}")
            return False

    def recording_started(self) -> None:
        self._present("lock start", self._set_recording_locked, True)
        self._present("started", self._set_started)

    def begin_streaming_session(self, session_id: str, sample_rate: float) -> None:
        self._waveform_generation += 1
        self._waveform_session_id = session_id
        self._waveform_sample_rate = float(sample_rate)
        self._waveform_buffer = None
        self._waveform_accepted_cursor = 0
        self._waveform_display_cursor = 0
        self._pending_waveform_range = None
        self._waveform_refresh_scheduled = False

    def end_streaming_session(self, session_id: str) -> bool:
        if session_id != self._waveform_session_id:
            return False
        self._waveform_generation += 1
        self._waveform_session_id = None
        self._waveform_buffer = None
        self._waveform_accepted_cursor = 0
        self._waveform_display_cursor = 0
        self._pending_waveform_range = None
        self._waveform_refresh_scheduled = False
        return True

    def _reserve_waveform_capacity(self, samples: np.ndarray, required: int) -> None:
        current = self._waveform_buffer
        if current is not None and current.shape[0] >= required:
            return
        capacity = 64 if current is None else max(64, int(current.shape[0]))
        while capacity < required:
            capacity *= 2
        shape = (capacity,) if samples.ndim == 1 else (capacity, samples.shape[1])
        replacement = np.empty(shape, dtype=samples.dtype)
        if current is not None and self._waveform_accepted_cursor:
            replacement[: self._waveform_accepted_cursor] = current[
                : self._waveform_accepted_cursor
            ]
        self._waveform_buffer = replacement

    @pyqtSlot(object)
    def queue_recording_batch(self, batch: Any) -> bool:
        if getattr(batch, "session_id", None) != self._waveform_session_id:
            return False
        sample_start = getattr(batch, "sample_start", None)
        sample_stop = getattr(batch, "sample_stop", None)
        if (
            type(sample_start) is not int
            or type(sample_stop) is not int
            or sample_start != self._waveform_accepted_cursor
            or sample_stop <= sample_start
        ):
            log = getattr(self._logger, "warning", None)
            if callable(log):
                log("ignored non-contiguous recording waveform batch")
            return False
        display = getattr(batch, "display", None)
        if display is None:
            display = getattr(batch, "mono", None)
        samples = np.array(display, copy=True)
        if (
            samples.ndim not in (1, 2)
            or samples.shape[0] != sample_stop - sample_start
        ):
            return False
        if self._waveform_buffer is not None and (
            samples.ndim != self._waveform_buffer.ndim
            or (
                samples.ndim == 2
                and samples.shape[1] != self._waveform_buffer.shape[1]
            )
        ):
            return False
        self._reserve_waveform_capacity(samples, sample_stop)
        self._waveform_buffer[sample_start:sample_stop] = samples
        self._waveform_accepted_cursor = sample_stop
        if self._pending_waveform_range is None:
            self._pending_waveform_range = (sample_start, sample_stop)
        else:
            self._pending_waveform_range = (
                self._pending_waveform_range[0],
                sample_stop,
            )
        if not self._waveform_refresh_scheduled:
            self._waveform_refresh_scheduled = True
            generation = self._waveform_generation
            self._schedule_waveform_refresh(
                lambda: self._flush_waveform_refresh(generation)
            )
        return True

    def _flush_waveform_refresh(self, generation: int | None = None) -> None:
        if generation is not None and generation != self._waveform_generation:
            return
        pending = self._pending_waveform_range
        self._pending_waveform_range = None
        self._waveform_refresh_scheduled = False
        if pending is None:
            return
        _sample_start, sample_stop = pending
        if self._waveform_buffer is None:
            return
        signal = self._waveform_buffer[:sample_stop].view()
        signal.setflags(write=False)
        self._waveform_display_cursor = sample_stop
        self._present(
            "streaming waveform",
            self._plot_recording,
            signal,
            self._waveform_sample_rate,
        )

    def recording_completed(self, staged: Any) -> None:
        self.end_streaming_session(staged.snapshot.session_id)
        self._present("identity", self._commit_identity, staged)
        fields = staged.data_struct_fields
        signal = fields.get("store_wave_data_multi")
        if signal is None:
            signal = fields.get("store_wave_data")
        if signal is not None:
            self._present(
                "waveform", self._plot_recording, signal, staged.snapshot.sample_rate
            )
        self._present("unlock success", self._set_recording_locked, False)
        self._present("finished success", self._set_finished, True)

    def recording_failed(self, reason: str) -> None:
        if self._waveform_session_id is not None:
            self.end_streaming_session(self._waveform_session_id)
        self._present("unlock failure", self._set_recording_locked, False)
        self._present("finished failure", self._set_finished, False)
        self._present("error", self._present_error, reason)

    def recording_cancelled(self, reason: str) -> None:
        if self._waveform_session_id is not None:
            self.end_streaming_session(self._waveform_session_id)
        self._present("unlock cancellation", self._set_recording_locked, False)
        self._present("finished cancellation", self._set_finished, False)

    def commit_label_projection(self, command: Any, receipt: Any) -> Any:
        """Apply the label UI projection as part of the owner's transaction."""
        checkpoint = self.capture_label_projection(command)
        return self.apply_label_projection(command, receipt, checkpoint)

    def capture_label_projection(self, command: Any) -> Any:
        callback = getattr(
            self._commit_label_projection, "capture_label_projection", None
        )
        return callback(command) if callable(callback) else None

    def apply_label_projection(
        self, command: Any, receipt: Any, checkpoint: Any
    ) -> Any:
        callback = getattr(
            self._commit_label_projection, "apply_label_projection", None
        )
        if callable(callback):
            return callback(command, receipt, checkpoint)
        return self._commit_label_projection(command, receipt)

    def restore_label_projection(
        self, checkpoint: Any, original_error: BaseException
    ) -> Any:
        callback = getattr(
            self._commit_label_projection, "restore_label_projection", None
        )
        return callback(checkpoint, original_error) if callable(callback) else True

class SequenceRecordingLabelProjection:
    """Transactional adapter for the legacy count-board and recording widgets."""

    _COUNT_TEXT_FIELDS = (
        "total_line_edit",
        "ok_line_edit",
        "ng_line_edit",
        "yield_line_edit",
        "datatime_line_edit",
        "mark_total_edit",
        "mark_ok_edit",
        "mark_ng_edit",
    )

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    @staticmethod
    def _button_enabled(button: Any) -> bool | None:
        callback = getattr(button, "isEnabled", None)
        return bool(callback()) if callable(callback) else None

    @staticmethod
    def _widget_value(widget: Any, name: str) -> Any:
        callback = getattr(widget, name, None)
        return callback() if callable(callback) else None

    @staticmethod
    def _required(callback: Callable[..., Any], name: str, *args: Any) -> Any:
        result = callback(*args)
        if result is False:
            raise RuntimeError(f"recording label {name} was rejected")
        return result

    @classmethod
    def _apply_statistics_snapshot(
        cls, count_board: Any, snapshot: RecordingStatisticsSnapshot
    ) -> None:
        if type(snapshot) is not RecordingStatisticsSnapshot:
            raise RuntimeError("recording statistics snapshot is unavailable")
        values = (
            ("total_line_edit", snapshot.test.total),
            ("ok_line_edit", snapshot.test.ok),
            ("ng_line_edit", snapshot.test.ng),
            ("yield_line_edit", snapshot.test.ok_percent),
            ("datatime_line_edit", snapshot.test.datatime),
            ("mark_total_edit", snapshot.mark.total),
            ("mark_ok_edit", snapshot.mark.ok),
            ("mark_ng_edit", snapshot.mark.ng),
        )
        for name, value in values:
            field = getattr(count_board, name, None)
            setter = getattr(field, "setText", None)
            if not callable(setter):
                raise RuntimeError(f"recording statistics field is unavailable: {name}")
            cls._required(setter, f"{name} projection", str(value))

    @staticmethod
    def _capture_analysis(runtime: Any) -> dict[str, Any] | None:
        view = getattr(runtime, "analysis_view", None)
        model = getattr(view, "model", None)
        if view is None or model is None:
            return None
        instances = list(getattr(model, "analysis_instances", ()) or ())
        registry = dict(getattr(model, "analysis_registry", {}) or {})
        summary = getattr(view, "summary_window", None)
        feedback = list(getattr(view, "feedback_dialogs", ()) or ())
        window_keys = dict(getattr(view, "window_keys", {}) or {})
        windows = []
        seen = set()
        for window in (*instances, summary, *feedback):
            if window is None or id(window) in seen:
                continue
            seen.add(id(window))
            visible = SequenceRecordingLabelProjection._widget_value(
                window, "isVisible"
            )
            windows.append((window, visible))
        return {
            "instances": instances,
            "registry": registry,
            "summary": summary,
            "feedback": feedback,
            "window_keys": window_keys,
            "windows": windows,
        }

    def _capture(self) -> dict[str, Any]:
        runtime = self.runtime
        data_struct = runtime.data_struct
        serial_input = runtime.lineedit_s_or_n
        count_texts = {}
        count_board = getattr(runtime, "count_board", None)
        for name in self._COUNT_TEXT_FIELDS:
            widget = getattr(count_board, name, None)
            if widget is not None:
                count_texts[name] = self._widget_value(widget, "text")
        capture_plot = getattr(runtime, "_capture_plot_projection_state", None)
        player_button = getattr(runtime, "player_btn", None)
        return {
            "recorded_path": runtime.recorded_path,
            "signal_info": dict(runtime.signal_info),
            "serial_number": serial_input.text(),
            "serial_enabled": self._button_enabled(serial_input),
            "serial_read_only": self._widget_value(serial_input, "isReadOnly"),
            "replay_enabled": self._button_enabled(runtime.replayer_btn),
            "data_enabled": self._button_enabled(runtime.data_btn),
            "player_enabled": self._button_enabled(player_button),
            "player_icon": self._widget_value(player_button, "icon"),
            "player_icon_size": self._widget_value(player_button, "iconSize"),
            "count_texts": count_texts,
            "plot": capture_plot() if callable(capture_plot) else None,
            "analysis": self._capture_analysis(runtime),
            "sn_clear_on_next_scan": getattr(
                runtime, "_sn_clear_on_next_scan", False
            ),
            "store_wave_data": getattr(data_struct, "store_wave_data", None),
            "store_wave_data_multi": getattr(
                data_struct, "store_wave_data_multi", None
            ),
            "wav_calibration_metadata": getattr(
                data_struct, "wav_calibration_metadata", None
            ),
            "wav_calibration_metadata_authoritative": getattr(
                data_struct, "wav_calibration_metadata_authoritative", False
            ),
            "wav_calibration_warning_shown": getattr(
                data_struct, "wav_calibration_warning_shown", False
            ),
        }

    @staticmethod
    def _restore_collection(target: Any, values: Any) -> None:
        target.clear()
        if isinstance(target, list):
            target.extend(values)
        else:
            target.update(values)

    def _restore_analysis(
        self, checkpoint: dict[str, Any] | None, failures: list[str]
    ) -> None:
        if checkpoint is None:
            return
        view = getattr(self.runtime, "analysis_view", None)
        model = getattr(view, "model", None)
        if view is None or model is None:
            failures.append("analysis: owner unavailable")
            return
        try:
            self._restore_collection(
                model.analysis_instances, checkpoint["instances"]
            )
            self._restore_collection(
                model.analysis_registry, checkpoint["registry"]
            )
            view.summary_window = checkpoint["summary"]
            self._restore_collection(view.feedback_dialogs, checkpoint["feedback"])
            self._restore_collection(view.window_keys, checkpoint["window_keys"])
            for window, visible in checkpoint["windows"]:
                callback = getattr(window, "show" if visible else "hide", None)
                if callable(callback):
                    self._required(callback, "analysis window restore")
        except BaseException as error:
            failures.append(f"analysis: {error}")

    def _restore(self, state: dict[str, Any], original: BaseException) -> bool:
        runtime = self.runtime
        failures = []
        restorers = (
            lambda: setattr(runtime, "recorded_path", state["recorded_path"]),
            lambda: (
                runtime.signal_info.clear(),
                runtime.signal_info.update(state["signal_info"]),
            ),
            lambda: runtime.lineedit_s_or_n.setText(state["serial_number"]),
            lambda: setattr(
                runtime,
                "_sn_clear_on_next_scan",
                state["sn_clear_on_next_scan"],
            ),
            lambda: setattr(
                runtime.data_struct,
                "store_wave_data",
                state["store_wave_data"],
            ),
            lambda: setattr(
                runtime.data_struct,
                "store_wave_data_multi",
                state["store_wave_data_multi"],
            ),
            lambda: setattr(
                runtime.data_struct,
                "wav_calibration_metadata",
                state["wav_calibration_metadata"],
            ),
            lambda: setattr(
                runtime.data_struct,
                "wav_calibration_metadata_authoritative",
                state["wav_calibration_metadata_authoritative"],
            ),
            lambda: setattr(
                runtime.data_struct,
                "wav_calibration_warning_shown",
                state["wav_calibration_warning_shown"],
            ),
        )
        for restore in restorers:
            try:
                if restore() is False:
                    raise RuntimeError("restore was rejected")
            except BaseException as error:
                failures.append(str(error) or type(error).__name__)
        count_board = getattr(runtime, "count_board", None)
        for name, text in state["count_texts"].items():
            try:
                if getattr(count_board, name).setText(text) is False:
                    raise RuntimeError("restore was rejected")
            except BaseException as error:
                failures.append(f"{name}: {error}")
        serial = runtime.lineedit_s_or_n
        for setter_name, value in (
            ("setEnabled", state["serial_enabled"]),
            ("setReadOnly", state["serial_read_only"]),
        ):
            if value is None:
                continue
            try:
                if getattr(serial, setter_name)(value) is False:
                    raise RuntimeError("restore was rejected")
            except BaseException as error:
                failures.append(f"serial {setter_name}: {error}")
        for button, enabled in (
            (runtime.replayer_btn, state["replay_enabled"]),
            (runtime.data_btn, state["data_enabled"]),
        ):
            if enabled is None:
                continue
            try:
                if button.setEnabled(enabled) is False:
                    raise RuntimeError("restore was rejected")
            except BaseException as error:
                failures.append(str(error) or type(error).__name__)
        player_button = getattr(runtime, "player_btn", None)
        for setter_name, value in (
            ("setIcon", state["player_icon"]),
            ("setIconSize", state["player_icon_size"]),
            ("setEnabled", state["player_enabled"]),
        ):
            if value is None:
                continue
            try:
                if getattr(player_button, setter_name)(value) is False:
                    raise RuntimeError("restore was rejected")
            except BaseException as error:
                failures.append(f"player {setter_name}: {error}")
        restore_plot = getattr(runtime, "_restore_plot_projection_state", None)
        if callable(restore_plot):
            try:
                if restore_plot(state["plot"]) is False:
                    raise RuntimeError("restore was rejected")
            except BaseException as error:
                failures.append(f"plot: {error}")
        self._restore_analysis(state["analysis"], failures)
        if failures:
            try:
                original.add_note(
                    "recording label View rollback failed: " + "; ".join(failures)
                )
            except (AttributeError, TypeError):
                return False
        return not failures

    @staticmethod
    def _clear_calibration(data_struct: Any) -> None:
        data_struct.wav_calibration_metadata = None
        data_struct.wav_calibration_metadata_authoritative = False
        data_struct.wav_calibration_warning_shown = False

    def capture_label_projection(self, _command: Any) -> dict[str, Any]:
        return self._capture()

    def apply_label_projection(
        self, command: Any, receipt: Any, checkpoint: Any
    ) -> bool:
        runtime = self.runtime
        post_analysis = bool(
            isinstance(command.export_outcome, Mapping)
            and command.export_outcome.get("continuation") == "post-analysis"
        )
        runtime.recorded_path = receipt.new_path
        self._apply_statistics_snapshot(
            runtime.count_board, receipt.statistics_snapshot
        )
        if not post_analysis:
            self._required(
                runtime._close_analysis_windows, "analysis window projection"
            )
            analysis = checkpoint.get("analysis")
            if analysis is not None:
                for window, _visible in analysis["windows"]:
                    visible = self._widget_value(window, "isVisible")
                    if visible:
                        hide = getattr(window, "hide", None)
                        if callable(hide):
                            self._required(hide, "analysis window hide")
            runtime.data_struct.store_wave_data = None
            runtime.data_struct.store_wave_data_multi = None
            self._clear_calibration(runtime.data_struct)
            self._required(runtime._clear_plot_area, "plot projection")
        runtime.signal_info.clear()
        self._required(runtime.lineedit_s_or_n.clear, "serial clear")
        self._required(
            runtime.replayer_btn.setEnabled, "replay button projection", False
        )
        self._required(
            runtime.data_btn.setEnabled, "data button projection", False
        )
        runtime._sn_clear_on_next_scan = False
        self._required(runtime.clicked_scanner, "scanner projection")
        self._required(
            runtime.update_player_btn_is_paused, "player projection"
        )
        return True

    def restore_label_projection(
        self, checkpoint: dict[str, Any], original: BaseException
    ) -> bool:
        return self._restore(checkpoint, original)

    def __call__(self, command: Any, receipt: Any) -> bool:
        checkpoint = self.capture_label_projection(command)
        try:
            return self.apply_label_projection(command, receipt, checkpoint)
        except BaseException as error:
            self.restore_label_projection(checkpoint, error)
            raise


@dataclass(slots=True)
class _AnalysisWindowsMarkCheckpoint:
    instances: list[Any]
    instance_items: tuple[Any, ...]
    registry: dict[Any, Any]
    registry_items: tuple[tuple[Any, Any], ...]
    summary: Any
    feedback_dialogs: list[Any]
    feedback_items: tuple[Any, ...]
    window_keys: Any
    window_key_items: tuple[tuple[Any, Any], ...]
    windows: tuple[Any, ...]
    visibility: tuple[tuple[Any, bool | None], ...]
    remaining: list[Any] = field(default_factory=list)
    prepared: bool = False
    finalize_started: bool = False


class SequenceRecordingAnalysisWindowsPort:
    """Two-phase mark cleanup for Analysis windows without controller coupling."""

    def __init__(self, analysis_view: Any) -> None:
        self.analysis_view = analysis_view

    @staticmethod
    def _native_alive(window: Any) -> bool:
        try:
            return not sip.isdeleted(window)
        except (AttributeError, TypeError):
            return True

    @staticmethod
    def _required(callback: Callable[..., Any], name: str, *args: Any) -> Any:
        result = callback(*args)
        if result is False:
            raise RuntimeError(f"analysis window {name} was rejected")
        return result

    @staticmethod
    def _unique_windows(*collections: Any) -> tuple[Any, ...]:
        windows = []
        identities = set()
        for collection in collections:
            for window in collection:
                if window is None or id(window) in identities:
                    continue
                identities.add(id(window))
                windows.append(window)
        return tuple(windows)

    def capture_mark_action_windows(self) -> _AnalysisWindowsMarkCheckpoint:
        view = self.analysis_view
        model = view.model
        instances = model.analysis_instances
        registry = model.analysis_registry
        feedback = view.feedback_dialogs
        window_keys = view.window_keys
        if not isinstance(instances, list) or not isinstance(registry, dict):
            raise TypeError("analysis window registries must be mutable wrappers")
        if not isinstance(feedback, list):
            raise TypeError("analysis feedback registry must be a list")
        instance_items = tuple(instances)
        registry_items = tuple(registry.items())
        feedback_items = tuple(feedback)
        window_key_items = tuple(window_keys.items())
        summary = view.summary_window
        captured_windows = self._unique_windows(
            instance_items,
            tuple(window for _key, window in registry_items),
            (summary,),
            feedback_items,
            tuple(key for key, _value in window_key_items),
        )
        visibility = []
        windows = []
        for window in captured_windows:
            if not self._native_alive(window):
                continue
            callback = getattr(window, "isVisible", None)
            visibility.append(
                (window, bool(callback()) if callable(callback) else None)
            )
            windows.append(window)
        return _AnalysisWindowsMarkCheckpoint(
            instances=instances,
            instance_items=instance_items,
            registry=registry,
            registry_items=registry_items,
            summary=summary,
            feedback_dialogs=feedback,
            feedback_items=feedback_items,
            window_keys=window_keys,
            window_key_items=window_key_items,
            windows=tuple(windows),
            visibility=tuple(visibility),
            remaining=list(windows),
        )

    def _require_checkpoint(self, checkpoint: Any) -> _AnalysisWindowsMarkCheckpoint:
        if type(checkpoint) is not _AnalysisWindowsMarkCheckpoint:
            raise TypeError("invalid analysis window mark checkpoint")
        return checkpoint

    def prepare_mark_action_windows(self, checkpoint: Any) -> bool:
        state = self._require_checkpoint(checkpoint)
        if state.finalize_started:
            return False
        if state.prepared:
            return True
        view = self.analysis_view
        model = view.model
        if (
            model.analysis_instances is not state.instances
            or model.analysis_registry is not state.registry
            or view.feedback_dialogs is not state.feedback_dialogs
            or view.window_keys is not state.window_keys
        ):
            raise RuntimeError("analysis window registry identity changed")
        for window, visible in state.visibility:
            if visible and self._native_alive(window):
                hide = getattr(window, "hide", None)
                if not callable(hide):
                    raise RuntimeError("analysis window hide is unavailable")
                self._required(hide, "hide")
        self._detach_checkpoint_entries(state)
        state.prepared = True
        return True

    @staticmethod
    def _remove_owned_list(container: list[Any], owned: tuple[Any, ...]) -> None:
        remaining_counts: dict[int, int] = {}
        for item in owned:
            remaining_counts[id(item)] = remaining_counts.get(id(item), 0) + 1
        kept = []
        for item in container:
            identity = id(item)
            count = remaining_counts.get(identity, 0)
            if count and any(item is candidate for candidate in owned):
                remaining_counts[identity] = count - 1
                continue
            kept.append(item)
        container[:] = kept

    @staticmethod
    def _remove_owned_mapping(
        mapping: Any, owned: tuple[tuple[Any, Any], ...]
    ) -> None:
        for key, value in owned:
            try:
                if key in mapping and mapping[key] is value:
                    del mapping[key]
            except (KeyError, RuntimeError, TypeError):
                continue

    def _detach_checkpoint_entries(
        self, state: _AnalysisWindowsMarkCheckpoint
    ) -> None:
        view = self.analysis_view
        model = view.model
        instance_wrappers = [state.instances]
        current_instances = getattr(model, "analysis_instances", None)
        if isinstance(current_instances, list) and current_instances is not state.instances:
            instance_wrappers.append(current_instances)
        for wrapper in instance_wrappers:
            self._remove_owned_list(wrapper, state.instance_items)

        registry_wrappers = [state.registry]
        current_registry = getattr(model, "analysis_registry", None)
        if isinstance(current_registry, dict) and current_registry is not state.registry:
            registry_wrappers.append(current_registry)
        for wrapper in registry_wrappers:
            self._remove_owned_mapping(wrapper, state.registry_items)

        if getattr(view, "summary_window", None) is state.summary:
            view.summary_window = None
        feedback_wrappers = [state.feedback_dialogs]
        current_feedback = getattr(view, "feedback_dialogs", None)
        if isinstance(current_feedback, list) and current_feedback is not state.feedback_dialogs:
            feedback_wrappers.append(current_feedback)
        for wrapper in feedback_wrappers:
            self._remove_owned_list(wrapper, state.feedback_items)

        key_wrappers = [state.window_keys]
        current_keys = getattr(view, "window_keys", None)
        if current_keys is not state.window_keys and hasattr(current_keys, "items"):
            key_wrappers.append(current_keys)
        for wrapper in key_wrappers:
            self._remove_owned_mapping(wrapper, state.window_key_items)

    @staticmethod
    def _append_missing_identity(container: list[Any], item: Any, count: int) -> None:
        present = sum(1 for candidate in container if candidate is item)
        if present < count:
            container.extend(item for _index in range(count - present))

    def _merge_checkpoint_entries(
        self,
        state: _AnalysisWindowsMarkCheckpoint,
        remaining: list[Any],
    ) -> None:
        live_remaining = [
            window for window in remaining if self._native_alive(window)
        ]
        remaining_ids = {id(window) for window in live_remaining}
        state.remaining[:] = live_remaining
        self._detach_checkpoint_entries(state)
        view = self.analysis_view
        model = view.model
        instances = getattr(model, "analysis_instances", state.instances)
        if not isinstance(instances, list):
            instances = state.instances
        for window in live_remaining:
            count = sum(
                1 for candidate in state.instance_items if candidate is window
            )
            if count:
                self._append_missing_identity(instances, window, count)

        registry = getattr(model, "analysis_registry", state.registry)
        if not isinstance(registry, dict):
            registry = state.registry
        for key, window in state.registry_items:
            if id(window) in remaining_ids and key not in registry:
                registry[key] = window
        if (
            state.summary is not None
            and id(state.summary) in remaining_ids
            and getattr(view, "summary_window", None) is None
        ):
            view.summary_window = state.summary
        feedback = getattr(view, "feedback_dialogs", state.feedback_dialogs)
        if not isinstance(feedback, list):
            feedback = state.feedback_dialogs
        for window in live_remaining:
            count = sum(
                1 for candidate in state.feedback_items if candidate is window
            )
            if count:
                self._append_missing_identity(feedback, window, count)
        window_keys = getattr(view, "window_keys", state.window_keys)
        for window, key in state.window_key_items:
            if id(window) not in remaining_ids:
                continue
            try:
                if window not in window_keys:
                    window_keys[window] = key
            except (RuntimeError, TypeError):
                continue

    def restore_mark_action_windows(
        self, checkpoint: Any, _original_error: BaseException
    ) -> bool:
        state = self._require_checkpoint(checkpoint)
        if state.finalize_started:
            return False
        failures = []
        self._merge_checkpoint_entries(state, list(state.windows))
        for window, visible in state.visibility:
            if visible is None or not self._native_alive(window):
                continue
            callback = getattr(window, "show" if visible else "hide", None)
            if not callable(callback):
                failures.append("window visibility callback unavailable")
                continue
            try:
                self._required(callback, "visibility restore")
            except BaseException as error:
                failures.append(f"window visibility: {error}")
        if not failures:
            state.prepared = False
        return not failures

    def finalize_mark_action_windows(self, checkpoint: Any) -> bool:
        state = self._require_checkpoint(checkpoint)
        if not state.prepared:
            return False
        state.finalize_started = True
        remaining = list(state.remaining)
        for index, window in enumerate(remaining):
            if not self._native_alive(window):
                continue
            close = getattr(window, "close", None)
            if not callable(close):
                self._merge_checkpoint_entries(state, remaining[index:])
                return False
            try:
                if close() is False:
                    self._merge_checkpoint_entries(state, remaining[index:])
                    return False
            except BaseException:
                self._merge_checkpoint_entries(state, remaining[index:])
                raise
        self._merge_checkpoint_entries(state, [])
        return True


class _NoopAnalysisWindowsPort:
    @staticmethod
    def capture_mark_action_windows() -> None:
        return None

    @staticmethod
    def prepare_mark_action_windows(_checkpoint: Any) -> bool:
        return True

    @staticmethod
    def restore_mark_action_windows(
        _checkpoint: Any, _original_error: BaseException
    ) -> bool:
        return True

    @staticmethod
    def finalize_mark_action_windows(_checkpoint: Any) -> bool:
        return True


class SequenceRecordingMarkActionProjection:
    """Transactional Qt projection for selecting mark mode and clearing replay."""

    requires_qt_owner_thread = True

    def __init__(
        self,
        runtime: Any,
        *,
        clear_plot: Callable[[], Any] = _noop,
        analysis_windows_port: Any = None,
        capture_plot: Callable[[], Any] | None = None,
        restore_plot: Callable[[Any], Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self.clear_plot = clear_plot
        self.analysis_windows_port = analysis_windows_port or _NoopAnalysisWindowsPort()
        self.capture_plot = capture_plot
        self.restore_plot = restore_plot

    @staticmethod
    def _enabled(widget: Any) -> bool | None:
        callback = getattr(widget, "isEnabled", None)
        return bool(callback()) if callable(callback) else None

    @staticmethod
    def _required(callback: Callable[..., Any], name: str, *args: Any) -> Any:
        result = callback(*args)
        if result is False:
            raise RuntimeError(f"recording mark-action {name} was rejected")
        return result

    def capture_mark_action_projection(self, _command: Any) -> dict[str, Any]:
        runtime = self.runtime
        board = runtime.count_board
        stack = getattr(board, "stacked_widget", None)
        current_index = getattr(stack, "currentIndex", None)
        signal_info = runtime.signal_info
        if not isinstance(signal_info, Mapping):
            raise TypeError("recording mark-action signal state must be a mapping")
        data_struct = runtime.data_struct
        return {
            "board_mode": getattr(board, "mode", None),
            "board_index": current_index() if callable(current_index) else None,
            "mark_enabled": self._enabled(getattr(board, "mark_btn", None)),
            "test_enabled": self._enabled(getattr(board, "test_btn", None)),
            "signal_info": signal_info,
            "signal_values": dict(signal_info),
            "store_wave_data": getattr(data_struct, "store_wave_data", None),
            "store_wave_data_multi": getattr(
                data_struct, "store_wave_data_multi", None
            ),
            "wav_calibration_metadata": getattr(
                data_struct, "wav_calibration_metadata", None
            ),
            "wav_calibration_metadata_authoritative": getattr(
                data_struct, "wav_calibration_metadata_authoritative", False
            ),
            "wav_calibration_warning_shown": getattr(
                data_struct, "wav_calibration_warning_shown", False
            ),
            "player_enabled": self._enabled(runtime.player_btn),
            "replay_enabled": self._enabled(runtime.replayer_btn),
            "data_enabled": self._enabled(runtime.data_btn),
            "plot": self.capture_plot() if callable(self.capture_plot) else None,
            "analysis_windows": (
                self.analysis_windows_port.capture_mark_action_windows()
            ),
        }

    def apply_mark_action_projection(
        self, _command: Any, _checkpoint: Any
    ) -> bool:
        runtime = self.runtime
        board = runtime.count_board
        select_mark = getattr(board, "on_mark_btn_clicked", None)
        if not callable(select_mark):
            raise RuntimeError("recording mark action is unavailable")
        self._required(select_mark, "mode selection")

        data_struct = runtime.data_struct
        data_struct.store_wave_data = None
        data_struct.store_wave_data_multi = None
        data_struct.wav_calibration_metadata = None
        data_struct.wav_calibration_metadata_authoritative = False
        data_struct.wav_calibration_warning_shown = False
        runtime.signal_info.clear()
        self._required(self.clear_plot, "plot cleanup")
        self._required(
            self.analysis_windows_port.prepare_mark_action_windows,
            "analysis window prepare",
            _checkpoint["analysis_windows"],
        )
        self._required(runtime.player_btn.setEnabled, "player enable", True)
        self._required(runtime.replayer_btn.setEnabled, "replay disable", False)
        self._required(runtime.data_btn.setEnabled, "analysis disable", False)
        return True

    def finalize_mark_action_projection(
        self, _command: Any, checkpoint: dict[str, Any]
    ) -> bool:
        self._required(
            self.analysis_windows_port.finalize_mark_action_windows,
            "analysis window finalize",
            checkpoint["analysis_windows"],
        )
        return True

    @staticmethod
    def _restore_enabled(
        widget: Any, enabled: bool | None, name: str, failures: list[str]
    ) -> None:
        if enabled is None:
            return
        try:
            result = widget.setEnabled(enabled)
            if result is False:
                raise RuntimeError("restore was rejected")
        except BaseException as error:
            failures.append(f"{name}: {error}")

    def restore_mark_action_projection(
        self, checkpoint: dict[str, Any], original_error: BaseException
    ) -> bool:
        runtime = self.runtime
        board = runtime.count_board
        data_struct = runtime.data_struct
        failures = []
        restorers = (
            ("board mode", lambda: setattr(board, "mode", checkpoint["board_mode"])),
            (
                "board stack",
                lambda: (
                    None
                    if checkpoint["board_index"] is None
                    else board.stacked_widget.setCurrentIndex(
                        checkpoint["board_index"]
                    )
                ),
            ),
            (
                "signal",
                lambda: (
                    checkpoint["signal_info"].clear(),
                    checkpoint["signal_info"].update(checkpoint["signal_values"]),
                    setattr(runtime, "signal_info", checkpoint["signal_info"]),
                ),
            ),
            (
                "recorded mono",
                lambda: setattr(
                    data_struct, "store_wave_data", checkpoint["store_wave_data"]
                ),
            ),
            (
                "recorded multi",
                lambda: setattr(
                    data_struct,
                    "store_wave_data_multi",
                    checkpoint["store_wave_data_multi"],
                ),
            ),
            (
                "calibration metadata",
                lambda: setattr(
                    data_struct,
                    "wav_calibration_metadata",
                    checkpoint["wav_calibration_metadata"],
                ),
            ),
            (
                "calibration authority",
                lambda: setattr(
                    data_struct,
                    "wav_calibration_metadata_authoritative",
                    checkpoint["wav_calibration_metadata_authoritative"],
                ),
            ),
            (
                "calibration warning",
                lambda: setattr(
                    data_struct,
                    "wav_calibration_warning_shown",
                    checkpoint["wav_calibration_warning_shown"],
                ),
            ),
        )
        for name, restore in restorers:
            try:
                if restore() is False:
                    raise RuntimeError("restore was rejected")
            except BaseException as error:
                failures.append(f"{name}: {error}")
        self._restore_enabled(
            getattr(board, "mark_btn", None),
            checkpoint["mark_enabled"],
            "mark button",
            failures,
        )
        self._restore_enabled(
            getattr(board, "test_btn", None),
            checkpoint["test_enabled"],
            "test button",
            failures,
        )
        for widget, enabled, name in (
            (runtime.player_btn, checkpoint["player_enabled"], "player button"),
            (runtime.replayer_btn, checkpoint["replay_enabled"], "replay button"),
            (runtime.data_btn, checkpoint["data_enabled"], "analysis button"),
        ):
            self._restore_enabled(widget, enabled, name, failures)
        if callable(self.restore_plot):
            try:
                if self.restore_plot(checkpoint["plot"]) is False:
                    raise RuntimeError("restore was rejected")
            except BaseException as error:
                failures.append(f"plot: {error}")
        try:
            if self.analysis_windows_port.restore_mark_action_windows(
                checkpoint["analysis_windows"], original_error
            ) is False:
                raise RuntimeError("restore was rejected")
        except BaseException as error:
            failures.append(f"analysis windows: {error}")
        if failures:
            try:
                original_error.add_note(
                    "recording mark-action View rollback failed: "
                    + "; ".join(failures)
                )
            except (AttributeError, TypeError):
                pass
        return not failures

    def fail_closed_mark_action_projection(
        self, _checkpoint: Any, _original_error: BaseException
    ) -> bool:
        failures = []
        runtime = self.runtime
        for widget, name in (
            (runtime.player_btn, "player"),
            (runtime.replayer_btn, "replay"),
            (runtime.data_btn, "analysis"),
        ):
            try:
                if widget.setEnabled(False) is False:
                    raise RuntimeError("disable was rejected")
            except BaseException as error:
                failures.append(f"{name}: {error}")
        return not failures


class SequenceRecordingStatisticsProjection:
    """Transactional count-board and reset-runtime projection owned by Recording."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self._runtime_projection = SequenceRecordingLabelProjection(runtime)

    def capture_statistics_projection(self) -> dict[str, Any]:
        return {
            "runtime": self._runtime_projection._capture(),
        }

    def apply_statistics_projection(
        self, _snapshot: Any, *, reset_runtime: bool
    ) -> bool:
        runtime = self.runtime
        count_board = getattr(runtime, "count_board", None)
        if count_board is not None:
            self._runtime_projection._apply_statistics_snapshot(
                count_board, _snapshot
            )
        if not reset_runtime:
            return True

        runtime._sn_clear_on_next_scan = False
        data_struct = runtime.data_struct
        data_struct.store_wave_data = None
        data_struct.store_wave_data_multi = None
        self._runtime_projection._clear_calibration(data_struct)
        self._runtime_projection._required(
            runtime.replayer_btn.setDisabled,
            "replay reset projection",
            True,
        )
        self._runtime_projection._required(
            runtime.data_btn.setDisabled,
            "analysis reset projection",
            True,
        )
        self._runtime_projection._required(
            runtime.update_player_btn_is_paused,
            "player reset projection",
        )
        return True

    def restore_statistics_projection(
        self, checkpoint: dict[str, Any], original_error: BaseException
    ) -> bool:
        failures = []
        if not self._runtime_projection._restore(
            checkpoint["runtime"], original_error
        ):
            failures.append("runtime projection restore was rejected")
        if failures:
            try:
                original_error.add_note(
                    "recording statistics View rollback failed: "
                    + "; ".join(failures)
                )
            except (AttributeError, TypeError):
                pass
        return not failures
