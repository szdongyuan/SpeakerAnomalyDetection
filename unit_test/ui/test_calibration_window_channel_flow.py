import os
import queue

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

import ui.calibration_window as calibration_window


class _DummyStreamingProcessor:
    def __init__(self):
        self.stop_calls = 0

    def stop_streaming(self):
        self.stop_calls += 1


class _ContractStreamingProcessor:
    def __init__(self, payloads=None, *, is_recording=False):
        self.audio_queue = queue.Queue()
        for payload in payloads or []:
            self.audio_queue.put_nowait(payload)
        self.accumulated_chunks = []
        self.accumulated_multi_chunks = []
        self.is_recording = is_recording
        self.process_queue_calls = 0

    def process_queue(self):
        self.process_queue_calls += 1
        while True:
            try:
                payload = self.audio_queue.get_nowait()
            except queue.Empty:
                return
            mono = np.asarray(payload["mono"], dtype=np.float32).reshape(-1)
            multi = np.asarray(payload["multi"], dtype=np.float32)
            if multi.ndim == 1:
                multi = multi.reshape(-1, 1)
            self.accumulated_chunks.append(mono)
            self.accumulated_multi_chunks.append(multi)

    def get_recorded_data(self):
        return np.concatenate(self.accumulated_chunks).astype(np.float32)

    def stop_streaming(self):
        self.is_recording = False


class _FakeCloseEvent:
    def __init__(self):
        self.accepted = False
        self.ignored = False

    def ignore(self):
        self.ignored = True

    def accept(self):
        self.accepted = True


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _build_widget(
    monkeypatch,
    *,
    saved_channels=None,
    startup_channels=None,
    startup_device=None,
    saved_factors=None,
):
    saved_channels = list(saved_channels or [])
    startup_channels = list(startup_channels if startup_channels is not None else saved_channels)
    startup_device = startup_device if startup_device is not None else {
        "name": "Demo Mic",
        "index": 7,
        "max_input_channels": 8,
    }
    saved_factors = dict(saved_factors or {})

    monkeypatch.setattr(
        calibration_window.SoundDeviceManager,
        "load_selected_devices",
        staticmethod(lambda: {"mic_channels": saved_channels}),
    )
    monkeypatch.setattr(
        calibration_window.SoundDeviceManager,
        "get_startup_devices",
        lambda self: {"mic": startup_device, "mic_channels": startup_channels},
    )
    monkeypatch.setattr(
        calibration_window.SoundDeviceManager,
        "restore_mic_channels",
        staticmethod(lambda device, channels: list(channels or [])),
    )
    monkeypatch.setattr(
        calibration_window,
        "load_mic_channel_v2pa_factors",
        lambda: dict(saved_factors),
    )

    return calibration_window.InputCalibration()


def test_format_channel_labels_are_user_visible():
    assert calibration_window.InputCalibration._format_channel_labels([0, 2, 4]) == "In1, In3, In5"


def test_next_uncalibrated_channel_uses_selected_order(qapp, monkeypatch):
    widget = _build_widget(
        monkeypatch,
        saved_channels=[0, 2, 4],
        startup_channels=[0, 2, 4],
        saved_factors={4: 1.4},
    )
    try:
        widget.calibrated_channels = {0}
        widget.current_channel = None

        assert widget._next_uncalibrated_channel() == 2
        assert widget.uncalibrated_selected_channels() == [2]
    finally:
        widget.close()


def test_clicked_calibration_returns_false_without_current_channel(qapp, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        calibration_window.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append((args, kwargs))),
    )
    widget = _build_widget(
        monkeypatch,
        saved_channels=[],
        startup_channels=[],
        startup_device={"name": "No Input Mic", "index": 3, "max_input_channels": 0},
    )
    try:
        assert widget.current_channel is None
        assert widget.clicked_calibration() is False
        assert warnings
        assert "未选择输入通道" in warnings[-1][0][2]
    finally:
        widget.close()


def test_clicked_calibration_returns_false_when_stream_start_fails(qapp, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        calibration_window.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append((args, kwargs))),
    )
    monkeypatch.setattr(
        calibration_window,
        "stream_record_without_play",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    widget = _build_widget(monkeypatch, saved_channels=[1], startup_channels=[1])
    try:
        assert widget.current_channel == 1
        assert widget.clicked_calibration() is False
        assert widget.streaming_processor is None
        assert not widget.update_ui_timer.isActive()
        assert not widget.streaming_poll_timer.isActive()
        assert warnings
        assert "录音启动失败" in warnings[-1][0][2]
    finally:
        widget.close()


def test_clicked_calibration_uses_selected_device_and_current_channel(qapp, monkeypatch):
    startup_device = {
        "name": "Selected Mic",
        "index": 11,
        "max_input_channels": 8,
    }
    calls = []
    processor = _DummyStreamingProcessor()
    monkeypatch.setattr(
        calibration_window,
        "stream_record_without_play",
        lambda recorded_dict, recorded_path, recorded_signal_info: (
            calls.append((recorded_dict, recorded_path, recorded_signal_info)) or (processor, 44100)
        ),
    )
    widget = _build_widget(
        monkeypatch,
        saved_channels=[1, 3],
        startup_channels=[1, 3],
        startup_device=startup_device,
    )
    try:
        widget.channel_combo_box.setCurrentIndex(1)
        qapp.processEvents()

        assert widget.current_channel == 3
        assert widget.clicked_calibration() is True
        assert len(calls) == 1
        recorded_dict, recorded_path, recorded_signal_info = calls[0]
        assert recorded_path is None
        assert recorded_signal_info is None
        assert recorded_dict["channels"] == 1
        assert recorded_dict["device"] == startup_device
        assert recorded_dict["input_channels"] == [3]
        assert widget.active_capture_channel == 3
        assert widget.channel_combo_box.isEnabled() is False
        assert widget.update_ui_timer.isActive() is True
        assert widget.streaming_poll_timer.isActive() is True
    finally:
        widget.reset_btn_clicked()
        widget.close()


def test_inflight_channel_switch_does_not_change_saved_channel(qapp, monkeypatch):
    saved_calls = []
    popup_calls = []
    button_states = []
    processor = _ContractStreamingProcessor(
        payloads=[
            {
                "mono": np.array([0.25, 0.5], dtype=np.float32),
                "multi": np.array([[0.25], [0.5]], dtype=np.float32),
            }
        ],
        is_recording=False,
    )
    monkeypatch.setattr(
        calibration_window,
        "save_mic_channel_v2pa_factor",
        lambda channel, factor, standard_spl=None: saved_calls.append((channel, factor, standard_spl)),
    )
    monkeypatch.setattr(
        calibration_window,
        "stream_record_without_play",
        lambda recorded_dict, recorded_path, recorded_signal_info: (processor, 44100),
    )
    widget = _build_widget(monkeypatch, saved_channels=[1, 3], startup_channels=[1, 3])
    try:
        assert widget.current_channel == 1
        monkeypatch.setattr(widget, "_calculate_spl_from_data", lambda data: 90.0 if len(data) == 2 else -1.0)
        monkeypatch.setattr(widget, "calculate_v2pa_factor", lambda average_value: 2.5)
        monkeypatch.setattr(widget, "calibration_popup", lambda success_flag=True: popup_calls.append(success_flag))
        monkeypatch.setattr(widget, "_set_parent_calibration_button_enabled", lambda enabled: button_states.append(enabled))

        assert widget.clicked_calibration() is True
        assert widget.active_capture_channel == 1
        widget.channel_combo_box.setCurrentIndex(1)
        qapp.processEvents()

        assert widget.current_channel == 1
        assert widget.channel_combo_box.currentData() == 1

        widget._poll_streaming_queue()
        qapp.processEvents()

        assert widget.streaming_processor is None
        assert saved_calls == [(1, 2.5, 94)]
        assert popup_calls == [True]
        assert button_states == [True]
        assert widget.active_capture_channel is None
        assert widget.session_channel_factors[1] == 2.5
        assert widget.calibrated_channels == {1}
        assert widget.channel_combo_box.isEnabled() is True
        assert widget.current_channel == 3
    finally:
        widget.close()


def test_close_event_blocks_window_button_when_input_channels_missing(qapp, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        calibration_window.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append((args, kwargs))),
    )
    window = calibration_window.CalibrationWindow()
    try:
        window.tabwidget.setCurrentIndex(1)
        window.input_cal_wnd._reload_selected_input_hardware = lambda preferred_channel=None: None
        window.input_cal_wnd.current_channel = 0
        window.input_cal_wnd.uncalibrated_selected_channels = lambda: [0, 2]

        event = _FakeCloseEvent()
        window.closeEvent(event)

        assert event.ignored is True
        assert event.accepted is False
        assert warnings
        assert "In1, In3" in warnings[-1][0][2]
    finally:
        window.close()


def test_close_event_allows_window_button_after_all_channels_calibrated(qapp):
    window = calibration_window.CalibrationWindow()
    try:
        window.tabwidget.setCurrentIndex(1)
        window.input_cal_wnd._reload_selected_input_hardware = lambda preferred_channel=None: None
        window.input_cal_wnd.uncalibrated_selected_channels = lambda: []
        window.input_cal_wnd.stop_timer = False

        event = _FakeCloseEvent()
        window.closeEvent(event)

        assert event.ignored is False
        assert window.input_cal_wnd.stop_timer is True
    finally:
        window.close()


def test_close_event_stops_active_streaming_when_close_is_allowed(qapp):
    window = calibration_window.CalibrationWindow()
    processor = _DummyStreamingProcessor()
    try:
        window.tabwidget.setCurrentIndex(1)
        window.input_cal_wnd._reload_selected_input_hardware = lambda preferred_channel=None: None
        window.input_cal_wnd.uncalibrated_selected_channels = lambda: []
        window.input_cal_wnd.stop_timer = False
        window.input_cal_wnd.streaming_processor = processor
        window.input_cal_wnd.update_ui_timer.start()
        window.input_cal_wnd.streaming_poll_timer.start(50)

        event = _FakeCloseEvent()
        window.closeEvent(event)

        assert event.ignored is False
        assert event.accepted is True
        assert window.input_cal_wnd.stop_timer is True
        assert processor.stop_calls == 1
        assert window.input_cal_wnd.streaming_processor is None
        assert window.input_cal_wnd.update_ui_timer.isActive() is False
        assert window.input_cal_wnd.streaming_poll_timer.isActive() is False
    finally:
        window.close()
