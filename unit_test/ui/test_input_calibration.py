import os
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
from base.recording_process_protocol import FrozenConfig
from unit_test.ui.test_recording_process_integration import CapturingBridge


DEVICE = {
    "index": 7,
    "name": "Test Microphone",
    "hostapi": 3,
    "max_input_channels": 3,
}


@pytest.fixture(scope="session")
def qapp(ui_qapp):
    return ui_qapp


@pytest.fixture(autouse=True)
def isolated_logger(monkeypatch):
    def bridge_for(widget):
        if widget.recording_bridge is None:
            widget.recording_bridge = CapturingBridge()
            widget.recording_bridge.service.cancel = mock.Mock()
        return widget.recording_bridge
    monkeypatch.setattr(InputCalibration, "_get_recording_bridge", bridge_for)
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
    ):
        self.session = SimpleNamespace(state="completed", request=SimpleNamespace(device=FrozenConfig.snapshot(DEVICE)))
        self.data = np.asarray(
            data if data is not None else [0.1, 0.1, 0.1, 0.1],
            dtype=np.float32,
        )
        self.target_samples = target_samples
        self.sample_rate = 2
        self.is_recording = False
        self.error_occurred = bool(error_message)
        self.error_message = error_message
        self.stop_calls = 0

    def get_recorded_data(self):
        if self.error_message:
            raise RuntimeError(self.error_message)
        return self.data

    def stop_streaming(self):
        self.stop_calls += 1
        self.is_recording = False


def _deliver_accepted(widget):
    widget._capture_standard_spl_db = 94.0 if widget.standard_spl_flag else 114.0
    widget._on_streaming_complete()


def _provisional_capture(widget):
    from base.recording_process_protocol import RecordingResult
    from base.recording_result_reader import RecordingAudio
    assert widget.clicked_calibration()
    session = widget.streaming_processor.session
    session.state = "delivering"
    session.service.accept_result = mock.Mock()
    session.service.reject_result = mock.Mock()
    data = np.full((441000, 1), .125, dtype=np.float32)
    request = session.request
    descriptor = RecordingResult(request.request_id, "calibration", request.path,
        44100, request.channels, 441000, 441000, False)
    return session, RecordingAudio(descriptor, data, data[:, 0].copy())


def test_provisional_only_requests_acceptance_and_authoritative_event_saves_once(qapp):
    widget = InputCalibration(DEVICE, [1])
    widget.calibration_popup = mock.Mock()
    widget._calculate_spl_from_data = mock.Mock(return_value=90)
    session, audio = _provisional_capture(widget)
    with mock.patch("ui.calibration_window.save_mic_channel_calibration") as save:
        widget._on_calibration_result_ready(session, audio)
        session.service.accept_result.assert_called_once_with(session.request.request_id)
        save.assert_not_called()
        widget._on_streaming_complete()
        save.assert_not_called()
        session.state = "completed"
        widget._on_calibration_accepted(session, audio)
        widget._on_calibration_accepted(session, audio)
        save.assert_called_once()


@pytest.mark.parametrize("invalid", ["short", "channels", "nan"])
def test_invalid_provisional_result_rejected_without_json(qapp, invalid):
    from dataclasses import replace
    widget = InputCalibration(DEVICE, [1])
    session, audio = _provisional_capture(widget)
    if invalid == "short":
        audio = replace(audio, multi=audio.multi[:-1], mono=audio.mono[:-1])
    elif invalid == "channels":
        audio = replace(audio, descriptor=replace(audio.descriptor, channels=(0,)))
    else:
        audio.multi[0, 0] = np.nan
    with mock.patch("ui.calibration_window.save_mic_channel_calibration") as save:
        widget._on_calibration_result_ready(session, audio)
        session.service.reject_result.assert_called_once()
        session.service.accept_result.assert_not_called()
        save.assert_not_called()
    widget.cancel_calibration()


def test_service_failure_restores_controls_and_reports_once(qapp):
    widget = InputCalibration(DEVICE, [1])
    widget.calibration_popup = mock.Mock()
    session, _audio = _provisional_capture(widget)
    finished = []
    widget.calibration_finished.connect(finished.append)
    with mock.patch("ui.calibration_window.save_mic_channel_calibration") as save:
        session.state = "failed"
        failure = SimpleNamespace(message="device failed")
        widget._on_calibration_failed(session, failure)
        widget._on_calibration_failed(session, failure)
        assert finished == [False]
        assert widget.streaming_processor is None
        assert widget.channel_combo_box.isEnabled()
        widget.calibration_popup.assert_called_once()
        save.assert_not_called()


def test_cleanup_warning_does_not_claim_json_saved_after_save_failure(qapp):
    widget = InputCalibration(DEVICE, [1])
    widget.calibration_popup = mock.Mock()
    widget._calculate_spl_from_data = mock.Mock(return_value=90)
    session, audio = _provisional_capture(widget)
    finished = []
    widget.calibration_finished.connect(finished.append)
    with mock.patch("ui.calibration_window.save_mic_channel_calibration",
            side_effect=MicCalibrationIOError("JSON denied")), mock.patch(
            "ui.calibration_window.QMessageBox.warning") as warning:
        session.state = "completed"
        widget._on_calibration_accepted(session, audio)
        widget._on_calibration_release_failed(session, "temporary cleanup denied")
        assert finished == [False]
        warning.assert_called_once()
        assert "已保存" not in warning.call_args.args[2]


@pytest.mark.parametrize("action", ["cancel", "reset", "close"])
def test_late_calibration_events_cannot_save_or_change_new_ui(qapp, action):
    widget = InputCalibration(DEVICE, [1, 0])
    widget.calibration_popup = mock.Mock()
    session, audio = _provisional_capture(widget)
    with mock.patch("ui.calibration_window.save_mic_channel_calibration") as save, mock.patch(
            "ui.calibration_window.clear_mic_channel_calibrations", return_value=False), mock.patch(
            "ui.calibration_window.QMessageBox.warning") as resource_warning:
        getattr(widget, {"cancel": "cancel_calibration", "reset": "reset_btn_clicked", "close": "close"}[action])()
        if action != "close":
            assert widget.clicked_calibration()
        current = widget.streaming_processor
        channel = widget.current_channel
        widget.calibration_popup.reset_mock()
        session.state = "completed"
        widget._on_calibration_result_ready(session, audio)
        widget._on_calibration_accepted(session, audio)
        widget._on_calibration_failed(session, SimpleNamespace(message="late"))
        session.release_error = "old cleanup failure"
        widget._on_calibration_release_failed(session, session.release_error)
        widget._on_calibration_released(session)
        assert widget.streaming_processor is current
        assert widget.current_channel == channel
        widget.calibration_popup.assert_not_called()
        resource_warning.assert_not_called()
        save.assert_not_called()
    widget.close()


def test_calibration_uses_capture_snapshot_for_device_and_reference_level(qapp):
    device = dict(DEVICE)
    widget = InputCalibration(device, [1])
    widget.calibration_popup = mock.Mock()
    widget._calculate_spl_from_data = mock.Mock(return_value=90)
    session, audio = _provisional_capture(widget)
    widget.standard_spl_flag = False
    device["index"] = 99
    with mock.patch("ui.calibration_window.save_mic_channel_calibration") as save:
        session.state = "completed"
        widget._on_calibration_accepted(session, audio)
        assert save.call_args.kwargs["input_device"] == DEVICE
        assert save.call_args.kwargs["standard_spl_db"] == 94
        assert save.call_args.kwargs["v2pa_factor"] == pytest.approx(10 ** (4 / 20))


def test_start_does_not_probe_temp_files_on_gui_thread(qapp):
    widget = InputCalibration(DEVICE, [1])
    with mock.patch("tempfile.gettempdir", side_effect=AssertionError("GUI temp-directory probe")):
        assert widget.clicked_calibration()
    widget.cancel_calibration()


def test_factor_helper_respects_new_selection_after_capture(qapp):
    widget = InputCalibration(DEVICE, [1])
    widget._capture_standard_spl_db = 94
    widget.standard_spl_flag = False
    assert widget.calculate_v2pa_factor(90) == pytest.approx(10 ** (24 / 20))


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
        "unit_test.ui.test_recording_process_integration.CapturingBridge.start"
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
        "unit_test.ui.test_recording_process_integration.CapturingBridge.start"
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
        "unit_test.ui.test_recording_process_integration.CapturingBridge.start"
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

    with mock.patch(
        "unit_test.ui.test_recording_process_integration.CapturingBridge.start",
        autospec=True, side_effect=CapturingBridge.start,
    ) as start_recording:
        started = widget.clicked_calibration()

    assert started is True
    request = start_recording.call_args.args[1]
    assert request.device.to_dict() == DEVICE
    assert request.channels == (1,)
    assert request.target_samples == 441000
    assert request.sample_rate == 44100
    widget.cancel_calibration()


def test_non_first_selector_channel_is_pinned_for_capture(qapp):
    widget = InputCalibration(DEVICE, [0, 2])
    widget.channel_combo_box.setCurrentIndex(1)

    with mock.patch(
        "unit_test.ui.test_recording_process_integration.CapturingBridge.start",
        autospec=True, side_effect=CapturingBridge.start,
    ) as start_recording:
        started = widget.clicked_calibration()

    assert started is True
    assert start_recording.call_args.args[1].channels == (2,)
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
        _deliver_accepted(widget)

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
        "unit_test.ui.test_recording_process_integration.CapturingBridge.start",
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
        _deliver_accepted(widget)

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
        _deliver_accepted(widget)
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
        _deliver_accepted(widget)

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
        "unit_test.ui.test_recording_process_integration.CapturingBridge.start"
    ) as start_next:
        _deliver_accepted(widget)

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
        _deliver_accepted(widget)

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
        _deliver_accepted(widget)

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
        _deliver_accepted(widget)

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
        _deliver_accepted(widget)

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
        _deliver_accepted(widget)

    save.assert_not_called()
    assert changes == []
    assert widget.streaming_processor is None


def test_startup_failure_never_saves_or_emits_change(qapp):
    widget = InputCalibration(DEVICE, [0])
    widget.calibration_popup = mock.Mock()
    changes = []
    widget.calibration_state_changed.connect(changes.append)

    with mock.patch(
        "unit_test.ui.test_recording_process_integration.CapturingBridge.start",
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
        "unit_test.ui.test_recording_process_integration.CapturingBridge.start"
    ) as start_next:
        _deliver_accepted(widget)
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
