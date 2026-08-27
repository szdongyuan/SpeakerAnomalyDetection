import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QRadioButton

from base.soundcard_calibration_manager import (
    MicCalibrationFormatError,
    MicCalibrationIOError,
    get_mic_v2pa_factor,
    save_mic_channel_calibration,
)
from ui.calibration_window import CalibrationWindow, InputCalibration


DEVICE = {
    "index": 7,
    "name": "Test Microphone",
    "hostapi": 3,
    "max_input_channels": 2,
}


@pytest.fixture(scope="session")
def qapp(ui_qapp):
    return ui_qapp


@pytest.fixture(autouse=True)
def isolated_logger():
    logger = SimpleNamespace(
        info=mock.Mock(),
        warning=mock.Mock(),
        error=mock.Mock(),
    )
    with mock.patch(
        "ui.calibration_window.LogManager.set_log_handler",
        return_value=logger,
    ), mock.patch(
        "ui.calibration_window.load_mic_channel_v2pa_factors",
        return_value={},
    ):
        yield


class _FakeProcessor:
    def __init__(
        self,
        data=None,
        *,
        target_samples=4,
        error_message="",
        process_error=None,
    ):
        self.data = np.asarray(
            data if data is not None else [0.1, 0.1, 0.1, 0.1],
            dtype=np.float32,
        )
        self.target_samples = target_samples
        self.sample_rate = 2
        self.is_recording = False
        self.error_occurred = bool(error_message)
        self.error_message = error_message
        self.process_calls = []
        self.stop_calls = 0
        self.process_error = process_error

    def process_queue(self, emit_signal=True):
        self.process_calls.append(emit_signal)
        if self.process_error is not None:
            raise self.process_error

    def get_recorded_data(self):
        return self.data

    def stop_streaming(self):
        self.stop_calls += 1
        self.is_recording = False


@pytest.mark.parametrize(
    "input_device,input_channels",
    [
        (None, [0]),
        (DEVICE, []),
        (DEVICE, [-1]),
    ],
)
def test_input_calibration_requires_one_valid_channel(
    qapp,
    input_device,
    input_channels,
):
    widget = InputCalibration(input_device, input_channels)
    widget.calibration_popup = mock.Mock()

    with mock.patch(
        "ui.calibration_window.stream_record_without_play"
    ) as start_recording:
        started = widget.clicked_calibration()

    assert started is False
    start_recording.assert_not_called()
    widget.calibration_popup.assert_called_once()


def test_selector_lists_physical_channels_and_prefers_first_missing(qapp):
    with mock.patch(
        "ui.calibration_window.load_mic_channel_v2pa_factors",
        return_value={0: 1.25},
    ):
        widget = InputCalibration(DEVICE, [0, 2, 0, -1, "bad", True])

    assert widget.input_channels == [0, 2]
    assert [
        widget.channel_combo_box.itemText(index)
        for index in range(widget.channel_combo_box.count())
    ] == ["In1", "In3"]
    assert [
        widget.channel_combo_box.itemData(index)
        for index in range(widget.channel_combo_box.count())
    ] == [0, 2]
    assert widget.current_channel == 2
    assert widget.channel_status_label.text() == "状态: 未校准"
    assert widget.v2pa_factor_lineedit.isReadOnly() is True


def test_all_calibrated_starts_on_first_channel(qapp):
    with mock.patch(
        "ui.calibration_window.load_mic_channel_v2pa_factors",
        return_value={0: 1.25, 2: 2.5},
    ):
        widget = InputCalibration(DEVICE, [0, 2])

    assert widget.current_channel == 0
    assert widget.channel_status_label.text() == "状态: 已校准"
    assert widget.v2pa_factor_lineedit.text() == "1.25"


@pytest.mark.parametrize(
    "error",
    [MicCalibrationFormatError("bad"), MicCalibrationIOError("denied")],
)
def test_registry_error_disables_only_input_calibration(qapp, error):
    logger = SimpleNamespace(error=mock.Mock(), info=mock.Mock(), warning=mock.Mock())
    with mock.patch(
        "ui.calibration_window.load_mic_channel_v2pa_factors",
        side_effect=error,
    ), mock.patch(
        "ui.calibration_window.QMessageBox.critical"
    ) as critical, mock.patch(
        "ui.calibration_window.LogManager.set_log_handler",
        return_value=logger,
    ):
        dialog = CalibrationWindow(input_device=DEVICE, input_channels=[0, 1])

    assert dialog.input_cal_wnd.calibration_available is False
    assert dialog.input_cal_wnd.current_channel is None
    assert dialog.input_cal_wnd.channel_combo_box.isEnabled() is False
    assert dialog.input_cal_wnd.channel_status_label.text() == "状态: 输入校准文件错误"
    dialog.tabwidget.setCurrentIndex(1)
    dialog._sync_calibration_button_state()
    assert dialog.cal_btn.isEnabled() is False
    assert dialog.reset_btn.isEnabled() is False
    with mock.patch.object(
        dialog.input_cal_wnd, "clicked_calibration"
    ) as calibrate, mock.patch.object(
        dialog.input_cal_wnd, "reset_btn_clicked"
    ) as reset:
        dialog.clicked_calibration_button()
        dialog.clicked_reset_button()
    calibrate.assert_not_called()
    reset.assert_not_called()
    assert critical.call_count == 1
    assert critical.call_args.args[2] == "输入校准文件错误，无法进行输入校准"
    assert str(error) in logger.error.call_args.args[0]
    dialog.tabwidget.setCurrentIndex(0)
    assert dialog.output_cal_wnd.isEnabled() is True
    assert dialog.cal_btn.isEnabled() is True
    assert dialog.reset_btn.isEnabled() is True
    dialog.show()
    qapp.processEvents()
    assert dialog.isVisible() is True
    dialog.clicked_close_button()
    assert dialog.isVisible() is False


def test_input_actions_disable_while_recording(qapp):
    dialog = CalibrationWindow(input_device=DEVICE, input_channels=[0, 1])
    dialog.tabwidget.setCurrentIndex(1)
    processor = _FakeProcessor()

    def start_calibration():
        dialog.input_cal_wnd.streaming_processor = processor
        return True

    dialog.input_cal_wnd.clicked_calibration = mock.Mock(
        side_effect=start_calibration
    )
    dialog.clicked_calibration_button()

    assert dialog.cal_btn.isEnabled() is False
    assert dialog.reset_btn.isEnabled() is False
    dialog.reject()


def test_missing_device_does_not_load_or_start_recording(qapp):
    with mock.patch(
        "ui.calibration_window.load_mic_channel_v2pa_factors"
    ) as load, mock.patch(
        "ui.calibration_window.stream_record_without_play"
    ) as start_recording:
        widget = InputCalibration(None, [0, 1])
        widget.calibration_popup = mock.Mock()
        started = widget.clicked_calibration()

    load.assert_not_called()
    start_recording.assert_not_called()
    assert started is False
    assert widget.calibration_available is False
    assert widget.input_device_label.text() == "未选择输入设备"
    assert widget.channel_status_label.text() == "未选择输入设备"
    assert widget.channel_combo_box.isEnabled() is False


@pytest.mark.parametrize("input_channels", [[], [-1, "bad", True, 1.5]])
def test_invalid_channels_do_not_load_or_start_recording(qapp, input_channels):
    with mock.patch(
        "ui.calibration_window.load_mic_channel_v2pa_factors"
    ) as load, mock.patch(
        "ui.calibration_window.stream_record_without_play"
    ) as start_recording:
        widget = InputCalibration(DEVICE, input_channels)
        widget.calibration_popup = mock.Mock()
        started = widget.clicked_calibration()

    load.assert_not_called()
    start_recording.assert_not_called()
    assert started is False
    assert widget.input_channels == []
    assert widget.calibration_available is False
    assert widget.channel_status_label.text() == "未选择有效输入通道"
    assert widget.channel_combo_box.isEnabled() is False


def test_selector_change_refreshes_saved_factor_and_status(qapp):
    with mock.patch(
        "ui.calibration_window.load_mic_channel_v2pa_factors",
        return_value={0: 1.25},
    ):
        widget = InputCalibration(DEVICE, [0, 2])

    widget.channel_combo_box.setCurrentIndex(0)
    assert widget.current_channel == 0
    assert widget.channel_status_label.text() == "状态: 已校准"
    assert widget.v2pa_factor_lineedit.text() == "1.25"

    widget.channel_combo_box.setCurrentIndex(1)
    assert widget.current_channel == 2
    assert widget.channel_status_label.text() == "状态: 未校准"
    assert widget.v2pa_factor_lineedit.text() == ""


def test_factor_display_is_read_only_without_manual_mode_controls(qapp):
    widget = InputCalibration(DEVICE, [0])

    assert widget.v2pa_factor_lineedit.isReadOnly() is True
    assert widget.v2pa_factor_lineedit.isEnabled() is True
    assert [button.text() for button in widget.findChildren(QRadioButton)] == [
        "94  dB",
        "114 dB",
    ]
    assert not hasattr(widget, "manual_mode")
    assert not hasattr(widget, "save_manual_mic")


def test_calibration_uses_selected_device_and_physical_channel(qapp):
    widget = InputCalibration(DEVICE, [1])
    processor = _FakeProcessor()
    widget._streaming_completion_processor = object()

    with mock.patch(
        "ui.calibration_window.stream_record_without_play",
        return_value=(processor, 44100),
    ) as start_recording:
        started = widget.clicked_calibration()

    assert started is True
    recorded_dict = start_recording.call_args.args[0]
    assert recorded_dict["device"] is DEVICE
    assert recorded_dict["input_channels"] == [1]
    assert recorded_dict["num_frames"] == 441000
    assert widget._streaming_completion_processor is None
    widget.cancel_calibration()


def test_non_first_selector_channel_is_pinned_for_capture(qapp):
    widget = InputCalibration(DEVICE, [0, 2])
    widget.channel_combo_box.setCurrentIndex(1)
    processor = _FakeProcessor()

    with mock.patch(
        "ui.calibration_window.stream_record_without_play",
        return_value=(processor, 44100),
    ) as start_recording:
        started = widget.clicked_calibration()

    assert started is True
    assert start_recording.call_args.args[0]["input_channels"] == [2]
    assert widget.active_capture_channel == 2
    assert widget.current_channel == 2
    assert widget.channel_combo_box.currentData() == 2
    assert widget.channel_combo_box.isEnabled() is False
    widget.channel_combo_box.setCurrentIndex(0)
    assert widget.current_channel == 2
    assert widget.channel_combo_box.currentData() == 2
    widget.cancel_calibration()


def test_completion_saves_pinned_channel_when_current_is_disturbed(qapp):
    widget = InputCalibration(DEVICE, [0, 2])
    widget.channel_combo_box.setCurrentIndex(1)
    widget.streaming_processor = _FakeProcessor()
    widget.active_capture_channel = 2
    widget.current_channel = 0
    widget._calculate_spl_from_data = mock.Mock(return_value=90.0)
    widget.calibration_popup = mock.Mock()

    with mock.patch(
        "ui.calibration_window.save_mic_channel_calibration",
    ) as save:
        widget._on_streaming_complete()

    assert save.call_args.kwargs["input_channel"] == 2
    assert widget.active_capture_channel is None
    assert widget.current_channel == 0
    assert widget.channel_combo_box.currentData() == 0
    assert widget.channel_combo_box.isEnabled() is True


def test_startup_failure_clears_pinned_channel_and_restores_selector(qapp):
    widget = InputCalibration(DEVICE, [0, 2])
    widget.channel_combo_box.setCurrentIndex(1)
    widget.calibration_popup = mock.Mock()

    with mock.patch(
        "ui.calibration_window.stream_record_without_play",
        side_effect=RuntimeError("device unavailable"),
    ):
        started = widget.clicked_calibration()

    assert started is False
    assert widget.active_capture_channel is None
    assert widget.current_channel == 2
    assert widget.channel_combo_box.currentData() == 2
    assert widget.channel_combo_box.isEnabled() is True


@pytest.mark.parametrize(
    "cleanup,expected_channel",
    [("failure", 2), ("cancel", 2), ("reset", 0)],
)
def test_capture_cleanup_clears_pinned_channel_and_restores_selector(
    qapp,
    cleanup,
    expected_channel,
):
    widget = InputCalibration(DEVICE, [0, 2])
    widget.channel_combo_box.setCurrentIndex(1)
    widget.active_capture_channel = 2
    widget.channel_combo_box.setEnabled(False)
    widget.streaming_processor = _FakeProcessor()
    widget.calibration_popup = mock.Mock()

    if cleanup == "failure":
        widget._finish_failed_calibration("failed")
    elif cleanup == "cancel":
        widget.cancel_calibration()
    else:
        widget.reset_btn_clicked()

    assert widget.active_capture_channel is None
    assert widget.streaming_processor is None
    assert widget.current_channel == expected_channel
    assert widget.channel_combo_box.currentData() == expected_channel
    assert widget.channel_combo_box.isEnabled() is True


def test_queue_ready_accumulates_without_waveform_for_active_processor(qapp):
    widget = InputCalibration(DEVICE, [1])
    processor = _FakeProcessor()
    widget.streaming_processor = processor

    widget._on_streaming_queue_ready(processor)

    assert processor.process_calls == [False]
    assert widget.streaming_processor is processor


def test_queue_failure_clears_pinned_channel_and_restores_selector(qapp):
    widget = InputCalibration(DEVICE, [0, 2])
    widget.channel_combo_box.setCurrentIndex(1)
    processor = _FakeProcessor()
    processor.process_queue = mock.Mock(side_effect=RuntimeError("queue failed"))
    widget.streaming_processor = processor
    widget.active_capture_channel = 2
    widget.channel_combo_box.setEnabled(False)
    widget.calibration_popup = mock.Mock()

    widget._on_streaming_queue_ready(processor)

    assert widget.active_capture_channel is None
    assert widget.streaming_processor is None
    assert widget.current_channel == 2
    assert widget.channel_combo_box.currentData() == 2
    assert widget.channel_combo_box.isEnabled() is True


def test_queue_ready_ignores_stale_processor(qapp):
    widget = InputCalibration(DEVICE, [1])
    active = _FakeProcessor()
    stale = _FakeProcessor()
    widget.streaming_processor = active

    widget._on_streaming_queue_ready(stale)

    assert stale.process_calls == []
    assert active.process_calls == []


def test_recording_finished_drains_before_completion_and_ignores_stale(qapp):
    widget = InputCalibration(DEVICE, [1])
    active = _FakeProcessor()
    stale = _FakeProcessor()
    widget.streaming_processor = active
    events = []
    active.process_queue = mock.Mock(side_effect=lambda **_: events.append("drain"))
    widget._on_streaming_complete = mock.Mock(
        side_effect=lambda: events.append("complete")
    )

    widget._on_streaming_recording_finished(stale)
    widget._on_streaming_recording_finished(active)

    assert stale.process_calls == []
    assert events == ["drain", "complete"]
    active.process_queue.assert_called_once_with(emit_signal=False)


def test_duplicate_finish_and_delayed_queue_ready_are_ignored(qapp):
    widget = InputCalibration(DEVICE, [1])
    processor = _FakeProcessor()
    widget.streaming_processor = processor
    widget._on_streaming_complete = mock.Mock()

    widget._on_streaming_recording_finished(processor)
    widget._on_streaming_recording_finished(processor)
    widget._on_streaming_queue_ready(processor)

    assert processor.process_calls == [False]
    widget._on_streaming_complete.assert_called_once_with()


def test_finish_processing_failure_guard_prevents_duplicate_dispatch(qapp):
    widget = InputCalibration(DEVICE, [1])
    processor = _FakeProcessor(process_error=RuntimeError("final drain failed"))
    widget.streaming_processor = processor
    widget._finish_failed_calibration = mock.Mock()

    widget._on_streaming_recording_finished(processor)
    widget._on_streaming_recording_finished(processor)

    assert processor.process_calls == [False]
    widget._finish_failed_calibration.assert_called_once_with(
        "输入校准录音数据处理失败，请重试。"
    )


def test_calibration_connections_are_explicitly_queued():
    source = (Path(__file__).resolve().parents[2] / "ui" / "calibration_window.py").read_text(
        encoding="utf-8"
    )

    assert "sign.stream_audio_queue_ready_signal.connect(" in source
    assert "self._on_streaming_queue_ready,\n            Qt.QueuedConnection," in source
    assert "sign.stream_audio_recording_finished_signal.connect(" in source
    assert "self._on_streaming_recording_finished,\n            Qt.QueuedConnection," in source


def test_only_countdown_timer_remains_for_input_calibration(qapp):
    widget = InputCalibration(DEVICE, [1])

    assert not hasattr(widget, "streaming_poll_timer")
    widget.update_ui_timer.start()
    assert widget.update_ui_timer.isActive() is True

    widget.cancel_calibration()

    assert widget.update_ui_timer.isActive() is False


def test_completed_calibration_saves_before_emitting_success(qapp):
    widget = InputCalibration(DEVICE, [1])
    processor = _FakeProcessor()
    widget.streaming_processor = processor
    widget.active_capture_channel = 1
    widget._calculate_spl_from_data = mock.Mock(return_value=90.0)
    widget.calibration_popup = mock.Mock()
    emitted = []
    widget.calibration_finished.connect(emitted.append)
    changes = []
    widget.calibration_state_changed.connect(changes.append)
    events = []

    with mock.patch(
        "ui.calibration_window.save_mic_channel_calibration",
        side_effect=lambda **kwargs: events.append("saved"),
    ) as save:
        widget.calibration_state_changed.connect(
            lambda _changed: events.append("changed")
        )
        widget._on_streaming_complete()

    save.assert_called_once_with(
        v2pa_factor=pytest.approx(10 ** (4.0 / 20.0)),
        input_device=DEVICE,
        input_channel=1,
        standard_spl_db=94.0,
        sample_rate_hz=2,
        duration_seconds=2.0,
    )
    assert emitted == [True]
    assert changes == [True]
    assert events == ["saved", "changed"]
    assert widget.streaming_processor is None
    assert float(widget.v2pa_factor_lineedit.text()) > 0.0
    widget.calibration_popup.assert_called_once_with(success_flag=True)


def test_completed_calibration_round_trips_through_runtime_resolver(qapp, tmp_path):
    calibration_path = tmp_path / "mic_input_calibration.json"
    widget = InputCalibration(DEVICE, [1])
    widget.streaming_processor = _FakeProcessor()
    widget.active_capture_channel = 1
    widget._calculate_spl_from_data = mock.Mock(return_value=90.0)
    widget.calibration_popup = mock.Mock()

    def save_to_temporary_path(**kwargs):
        return save_mic_channel_calibration(
            **kwargs,
            calibration_path=str(calibration_path),
            calibrated_at="2026-08-12T10:00:00+08:00",
        )

    with mock.patch(
        "base.soundcard_calibration_manager.SoundDeviceManager.get_api_info",
        return_value={"name": "Test API"},
    ), mock.patch(
        "ui.calibration_window.save_mic_channel_calibration",
        side_effect=save_to_temporary_path,
    ):
        widget._on_streaming_complete()
        factor = get_mic_v2pa_factor(
            DEVICE,
            [1],
            str(calibration_path),
        )

    assert factor == pytest.approx(10 ** (4.0 / 20.0))
    assert calibration_path.exists()


@pytest.mark.parametrize("processor", [
    _FakeProcessor(data=[0.1, 0.1], target_samples=4),
    _FakeProcessor(error_message="device lost"),
])
def test_failure_never_emits_success_or_replaces_result(
    qapp,
    processor,
):
    widget = InputCalibration(DEVICE, [1])
    widget.streaming_processor = processor
    widget.active_capture_channel = 1
    widget._calculate_spl_from_data = mock.Mock(return_value=90.0)
    widget.calibration_popup = mock.Mock()
    emitted = []
    widget.calibration_finished.connect(emitted.append)
    changes = []
    widget.calibration_state_changed.connect(changes.append)

    with mock.patch(
        "ui.calibration_window.save_mic_channel_calibration",
    ) as save:
        widget._on_streaming_complete()

    save.assert_not_called()
    assert emitted == [False]
    assert changes == []
    assert widget.streaming_processor is None
    assert widget.active_capture_channel is None
    assert widget.channel_combo_box.isEnabled() is True
    assert widget.v2pa_factor_lineedit.text() == ""
    assert processor.stop_calls == 1
    assert widget.calibration_popup.call_args.kwargs["success_flag"] is False


def test_dialog_marks_change_only_after_persisted_state_signal(qapp):
    dialog = CalibrationWindow(input_device=DEVICE, input_channels=[1])

    assert dialog.input_calibration_flag is False
    dialog._on_input_calibration_finished(False)
    assert dialog.input_calibration_flag is False
    dialog.input_cal_wnd.calibration_state_changed.emit(True)
    assert dialog.input_calibration_flag is True
    dialog.reject()


def test_cancel_stops_active_recording_without_reporting_failure(qapp):
    widget = InputCalibration(DEVICE, [1])
    processor = _FakeProcessor()
    processor.is_recording = True
    widget.streaming_processor = processor
    emitted = []
    widget.calibration_finished.connect(emitted.append)

    widget.cancel_calibration()

    assert processor.stop_calls == 1
    assert widget.streaming_processor is None
    assert emitted == []


def test_success_saves_current_channel_then_advances_without_starting_next(qapp):
    widget = InputCalibration(DEVICE, [0, 2])
    widget.streaming_processor = _FakeProcessor()
    widget.active_capture_channel = 0
    widget._calculate_spl_from_data = mock.Mock(return_value=90.0)
    widget.calibration_popup = mock.Mock()
    changes = []
    widget.calibration_state_changed.connect(changes.append)

    with mock.patch(
        "ui.calibration_window.save_mic_channel_calibration"
    ) as save, mock.patch(
        "ui.calibration_window.stream_record_without_play"
    ) as start_next:
        widget._on_streaming_complete()

    assert save.call_args.kwargs["input_channel"] == 0
    assert widget.saved_v2pa_factors[0] == pytest.approx(10 ** (4.0 / 20.0))
    assert widget.current_channel == 2
    assert widget.channel_combo_box.currentData() == 2
    assert widget.streaming_processor is None
    start_next.assert_not_called()
    assert changes == [True]


def test_advancement_wraps_in_stable_order_and_excludes_current(qapp):
    with mock.patch(
        "ui.calibration_window.load_mic_channel_v2pa_factors",
        return_value={2: 2.0},
    ):
        widget = InputCalibration(DEVICE, [0, 2, 4])
    widget.channel_combo_box.setCurrentIndex(2)
    widget.streaming_processor = _FakeProcessor()
    widget.active_capture_channel = 4
    widget._calculate_spl_from_data = mock.Mock(return_value=90.0)
    widget.calibration_popup = mock.Mock()

    with mock.patch("ui.calibration_window.save_mic_channel_calibration"):
        widget._on_streaming_complete()

    assert widget.current_channel == 0


def test_final_channel_stays_selected(qapp):
    with mock.patch(
        "ui.calibration_window.load_mic_channel_v2pa_factors",
        return_value={0: 1.25},
    ):
        widget = InputCalibration(DEVICE, [0, 2])
    widget.streaming_processor = _FakeProcessor()
    widget.active_capture_channel = 2
    widget._calculate_spl_from_data = mock.Mock(return_value=90.0)
    widget.calibration_popup = mock.Mock()

    with mock.patch("ui.calibration_window.save_mic_channel_calibration"):
        widget._on_streaming_complete()

    assert widget.current_channel == 2
    assert widget.channel_combo_box.currentData() == 2


def test_114_db_selection_is_saved_exactly(qapp):
    widget = InputCalibration(DEVICE, [0])
    widget.standard_spl_ii.setChecked(True)
    widget.set_standard_spl()
    widget.streaming_processor = _FakeProcessor()
    widget.active_capture_channel = 0
    widget._calculate_spl_from_data = mock.Mock(return_value=110.0)
    widget.calibration_popup = mock.Mock()

    with mock.patch(
        "ui.calibration_window.save_mic_channel_calibration"
    ) as save:
        widget._on_streaming_complete()

    assert save.call_args.kwargs["standard_spl_db"] == 114.0


@pytest.mark.parametrize(
    "error",
    [
        ValueError("invalid"),
        MicCalibrationFormatError("bad"),
        MicCalibrationIOError("denied"),
    ],
)
def test_save_failure_preserves_current_display_and_emits_no_change(qapp, error):
    with mock.patch(
        "ui.calibration_window.load_mic_channel_v2pa_factors",
        return_value={0: 1.25},
    ):
        widget = InputCalibration(DEVICE, [0, 2])
    widget.channel_combo_box.setCurrentIndex(1)
    widget.streaming_processor = _FakeProcessor()
    widget.active_capture_channel = 2
    widget._calculate_spl_from_data = mock.Mock(return_value=90.0)
    widget.calibration_popup = mock.Mock()
    changes = []
    widget.calibration_state_changed.connect(changes.append)

    with mock.patch(
        "ui.calibration_window.save_mic_channel_calibration",
        side_effect=error,
    ):
        widget._on_streaming_complete()

    assert widget.saved_v2pa_factors == {0: 1.25}
    assert widget.current_channel == 2
    assert widget.channel_status_label.text() == "状态: 未校准"
    assert widget.v2pa_factor_lineedit.text() == ""
    assert widget.streaming_processor is None
    assert widget.channel_combo_box.isEnabled() is True
    assert changes == []
    assert widget.calibration_popup.call_args.kwargs["success_flag"] is False


def test_invalid_calculated_result_never_saves_or_emits_change(qapp):
    widget = InputCalibration(DEVICE, [0])
    widget.streaming_processor = _FakeProcessor()
    widget.active_capture_channel = 0
    widget._calculate_spl_from_data = mock.Mock(return_value=np.nan)
    widget.calibration_popup = mock.Mock()
    changes = []
    widget.calibration_state_changed.connect(changes.append)

    with mock.patch(
        "ui.calibration_window.save_mic_channel_calibration"
    ) as save:
        widget._on_streaming_complete()

    save.assert_not_called()
    assert changes == []
    assert widget.streaming_processor is None


def test_startup_failure_never_saves_or_emits_change(qapp):
    widget = InputCalibration(DEVICE, [0])
    widget.calibration_popup = mock.Mock()
    changes = []
    widget.calibration_state_changed.connect(changes.append)

    with mock.patch(
        "ui.calibration_window.stream_record_without_play",
        side_effect=RuntimeError("device unavailable"),
    ), mock.patch(
        "ui.calibration_window.save_mic_channel_calibration"
    ) as save:
        assert widget.clicked_calibration() is False

    save.assert_not_called()
    assert changes == []
    assert widget.streaming_processor is None
    assert widget.active_capture_channel is None


def test_cancel_does_not_save_or_emit_change(qapp):
    widget = InputCalibration(DEVICE, [0, 2])
    widget.streaming_processor = _FakeProcessor()
    widget.active_capture_channel = 0
    changes = []
    widget.calibration_state_changed.connect(changes.append)

    with mock.patch(
        "ui.calibration_window.save_mic_channel_calibration"
    ) as save:
        widget.cancel_calibration()

    save.assert_not_called()
    assert changes == []
    assert widget.streaming_processor is None
    assert widget.active_capture_channel is None
    assert widget.update_ui_timer.isActive() is False
    assert not hasattr(widget, "streaming_poll_timer")


def test_partial_completion_can_close_without_starting_or_saving_remaining(qapp):
    dialog = CalibrationWindow(input_device=DEVICE, input_channels=[0, 2])
    widget = dialog.input_cal_wnd
    widget.streaming_processor = _FakeProcessor()
    widget.active_capture_channel = 0
    widget._calculate_spl_from_data = mock.Mock(return_value=90.0)
    widget.calibration_popup = mock.Mock()

    with mock.patch(
        "ui.calibration_window.save_mic_channel_calibration"
    ) as save, mock.patch(
        "ui.calibration_window.stream_record_without_play"
    ) as start_next:
        widget._on_streaming_complete()
        dialog.reject()

    assert save.call_count == 1
    start_next.assert_not_called()
    assert widget.current_channel == 2


def test_reset_clears_only_selected_channels_and_emits_change(qapp):
    with mock.patch(
        "ui.calibration_window.load_mic_channel_v2pa_factors",
        side_effect=[{0: 1.25, 2: 2.5}, {}],
    ):
        widget = InputCalibration(DEVICE, [0, 2])
        widget.channel_combo_box.setCurrentIndex(1)
        changes = []
        widget.calibration_state_changed.connect(changes.append)
        with mock.patch(
            "ui.calibration_window.clear_mic_channel_calibrations",
            return_value=True,
        ) as clear:
            widget.reset_btn_clicked()

    clear.assert_called_once_with(DEVICE, [0, 2])
    assert changes == [True]
    assert widget.saved_v2pa_factors == {}
    assert widget.current_channel == 0
    assert widget.channel_combo_box.currentData() == 0
    assert widget.channel_status_label.text() == "状态: 未校准"


def test_reset_stops_capture_before_clearing(qapp):
    widget = InputCalibration(DEVICE, [0, 2])
    processor = _FakeProcessor()
    processor.is_recording = True
    widget.streaming_processor = processor
    widget.active_capture_channel = 2
    events = []
    processor.stop_streaming = mock.Mock(side_effect=lambda: events.append("stop"))

    with mock.patch(
        "ui.calibration_window.clear_mic_channel_calibrations",
        side_effect=lambda *_args: events.append("clear") or False,
    ), mock.patch(
        "ui.calibration_window.load_mic_channel_v2pa_factors",
        return_value={},
    ):
        widget.reset_btn_clicked()

    assert events == ["stop", "clear"]
    assert widget.streaming_processor is None
    assert widget.active_capture_channel is None


def test_idempotent_reset_reloads_but_does_not_emit_change(qapp):
    widget = InputCalibration(DEVICE, [0, 2])
    widget.channel_combo_box.setCurrentIndex(1)
    changes = []
    widget.calibration_state_changed.connect(changes.append)

    with mock.patch(
        "ui.calibration_window.clear_mic_channel_calibrations",
        return_value=False,
    ), mock.patch(
        "ui.calibration_window.load_mic_channel_v2pa_factors",
        return_value={},
    ) as load:
        widget.reset_btn_clicked()

    load.assert_called_once_with(DEVICE)
    assert changes == []
    assert widget.current_channel == 0


@pytest.mark.parametrize(
    "error",
    [
        ValueError("invalid"),
        MicCalibrationFormatError("bad"),
        MicCalibrationIOError("denied"),
    ],
)
def test_reset_failure_preserves_displayed_state_and_restores_controls(qapp, error):
    with mock.patch(
        "ui.calibration_window.load_mic_channel_v2pa_factors",
        return_value={0: 1.25},
    ):
        widget = InputCalibration(DEVICE, [0, 2])
    widget.channel_combo_box.setCurrentIndex(0)
    widget.calibration_popup = mock.Mock()
    changes = []
    widget.calibration_state_changed.connect(changes.append)

    with mock.patch(
        "ui.calibration_window.clear_mic_channel_calibrations",
        side_effect=error,
    ), mock.patch(
        "ui.calibration_window.load_mic_channel_v2pa_factors"
    ) as reload:
        widget.reset_btn_clicked()

    reload.assert_not_called()
    assert widget.saved_v2pa_factors == {0: 1.25}
    assert widget.current_channel == 0
    assert widget.channel_status_label.text() == "状态: 已校准"
    assert widget.v2pa_factor_lineedit.text() == "1.25"
    assert widget.channel_combo_box.isEnabled() is True
    assert changes == []
    assert widget.calibration_popup.call_args.kwargs["success_flag"] is False


def test_dialog_marks_persisted_change_after_save_or_effective_reset(qapp):
    dialog = CalibrationWindow(input_device=DEVICE, input_channels=[0, 2])

    dialog.input_cal_wnd.calibration_finished.emit(True)
    assert dialog.input_calibration_flag is False
    dialog.input_cal_wnd.calibration_state_changed.emit(True)
    assert dialog.input_calibration_flag is True
    dialog.input_cal_wnd.calibration_state_changed.emit(False)
    assert dialog.input_calibration_flag is True
    dialog.reject()
