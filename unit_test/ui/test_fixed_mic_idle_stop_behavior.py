import logging
import os
import sys
import types
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


if "concurrent_log_handler" not in sys.modules:
    concurrent_log_handler = types.ModuleType("concurrent_log_handler")

    class _ConcurrentRotatingFileHandler(logging.Handler):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def emit(self, record):
            return

    concurrent_log_handler.ConcurrentRotatingFileHandler = _ConcurrentRotatingFileHandler
    sys.modules["concurrent_log_handler"] = concurrent_log_handler

if "keyboard" not in sys.modules:
    keyboard = types.ModuleType("keyboard")
    keyboard.add_hotkey = lambda *args, **kwargs: None
    keyboard.unhook_all_hotkeys = lambda *args, **kwargs: None
    sys.modules["keyboard"] = keyboard

if "pywinusb" not in sys.modules:
    hid_module = types.ModuleType("hid")
    hid_module.find_all_hid_devices = lambda: []
    pywinusb_module = types.ModuleType("pywinusb")
    pywinusb_module.hid = hid_module
    sys.modules["pywinusb"] = pywinusb_module
    sys.modules["pywinusb.hid"] = hid_module


from PyQt5.QtWidgets import QApplication, QStackedWidget
import pyqtgraph as pg

from consts import error_code
from ui.sequence.fixed_mic.runtime_bridge import handle_fixed_mic_manual_trigger, stop_fixed_mic_runtime
from ui.sequence.sequence_widget import SequenceWindow


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def build_fixed_mic_window_stub():
    window = SequenceWindow.__new__(SequenceWindow)
    window.fixed_mic_plot_items = []
    window.fixed_mic_plot_widgets = []
    window.fixed_mic_live_y_limits = []
    window.fixed_mic_page_size = 4
    window.fixed_mic_current_page = 0
    window.fixed_mic_display_total_channels = 0
    window.fixed_mic_display_audio = None
    window.fixed_mic_display_sample_rate = 0.0
    window.fixed_mic_display_live_mode = False
    window.fixed_mic_plot_window_sec = 15.0
    window.data_struct = SimpleNamespace(sample_rate=44100)
    window.sequence_config = [
        {
            "seq1": {
                "acq": {
                    "mode": "FIXED_MIC_MULTI_SESSION",
                    "detail": {
                        "channels": 6,
                        "fixed_mic_channels": [{"label": f"Mic{i + 1}"} for i in range(6)],
                    },
                }
            }
        }
    ]
    window.line_graph = pg.PlotWidget()
    window._configure_main_waveform_plot(window.line_graph)
    window.fixed_mic_waveform_widget = window._create_fixed_mic_waveform_widget()
    window.waveform_stack = QStackedWidget()
    window.waveform_stack.addWidget(window.line_graph)
    window.waveform_stack.addWidget(window.fixed_mic_waveform_widget)
    return window


class DummyController(object):
    def __init__(self):
        self.stop_calls = 0

    def process_audio_queue(self):
        return 0, []

    def stop_capture(self):
        self.stop_calls += 1


class DummyStartController(object):
    def __init__(self, acq_detail, input_device=None):
        self.sample_rate = int(acq_detail.get("sample_rate", 44100))
        self.channels = int(acq_detail.get("channels", 1))
        self.buffer_duration = float(acq_detail.get("buffer_duration", 15.0))
        self.window_duration = float(acq_detail.get("window_duration", 3.0))
        self.is_running = True

    def start_capture(self):
        return error_code.OK, "ok"

    def create_manual_session(self, barcode):
        session = SimpleNamespace(session_id="session_01")
        return error_code.OK, "ok", session


class TestFixedMicIdleStopBehavior(object):
    def test_render_fixed_mic_waveforms_without_audio_shows_empty_subplots(self, qapp):
        window = build_fixed_mic_window_stub()

        window._render_fixed_mic_waveforms(
            None,
            sample_rate=44100,
            total_channels=6,
            reset_page=True,
            live_mode=False,
        )

        visible_count = sum(not plot_widget.isHidden() for plot_widget in window.fixed_mic_plot_widgets)
        assert visible_count == 4
        assert window.fixed_mic_page_label.text() == "1 / 2"
        assert window.waveform_stack.currentWidget() is window.fixed_mic_waveform_widget
        assert window.fixed_mic_plot_widgets[0].plotItem.titleLabel.text == "Mic1"

    def test_stop_fixed_mic_runtime_keeps_last_waveform_display(self):
        clear_calls = []
        controller = DummyController()
        window = SimpleNamespace(
            fixed_mic_poll_timer=SimpleNamespace(isActive=lambda: False, stop=lambda: None),
            fixed_mic_controller=controller,
            fixed_mic_analysis_queue=[],
            fixed_mic_analysis_busy=False,
            fixed_mic_pending_review_sessions=[],
            fixed_mic_current_review_session=None,
            player_status_flag=True,
            count_board=SimpleNamespace(mode="test", set_review_session_text=lambda *_: None, set_review_session_visible=lambda *_: None),
            _clear_waveform_display=lambda: clear_calls.append("cleared"),
            _update_fixed_mic_toolbar_state=lambda: None,
            _update_fixed_mic_session_status=lambda *args, **kwargs: None,
            _process_next_fixed_mic_analysis=lambda: None,
            fixed_mic_plot_item=object(),
            fixed_mic_last_plot_update_ts=1.0,
            fixed_mic_live_y_limit=0.5,
            fixed_mic_stream_buffer=[1],
        )

        stop_fixed_mic_runtime(window)

        assert clear_calls == []
        assert controller.stop_calls == 1
        assert window.fixed_mic_controller is None

    def test_handle_fixed_mic_manual_trigger_resets_to_first_page_empty_layout(self):
        render_calls = []
        clear_calls = []
        timer_calls = []
        register_calls = []
        toolbar_calls = []
        reset_calls = []
        window = SimpleNamespace(
            checked_work_status_message=lambda: False,
            clicked_player_flag=True,
            fixed_mic_controller=None,
            sequence_config=[{"seq1": {"acq": {"detail": {"sample_rate": 48000, "channels": 6, "buffer_duration": 15.0, "window_duration": 3.0}}}}],
            mic={"name": "mic"},
            fixed_mic_poll_timer=SimpleNamespace(start=lambda interval: timer_calls.append(interval)),
            _clear_waveform_display=lambda: clear_calls.append("cleared"),
            _reset_fixed_mic_session_views=lambda: reset_calls.append("reset"),
            _configure_fixed_mic_live_plot_view=lambda: None,
            _render_fixed_mic_waveforms=lambda *args, **kwargs: render_calls.append((args, kwargs)),
            lineedit_s_or_n=SimpleNamespace(text=lambda: ""),
            data_btn=SimpleNamespace(setEnabled=lambda enabled: None),
            _update_fixed_mic_toolbar_state=lambda: toolbar_calls.append("updated"),
            _register_fixed_mic_session=lambda session, status_text="": register_calls.append((session.session_id, status_text)),
            default_logger=SimpleNamespace(info=lambda *args, **kwargs: None),
            fixed_mic_plot_item=None,
            fixed_mic_last_plot_update_ts=0.0,
            fixed_mic_live_y_limit=0.01,
            fixed_mic_stream_buffer=[],
            player_status_flag=False,
        )

        handle_fixed_mic_manual_trigger(window, DummyStartController)

        assert clear_calls == ["cleared"]
        assert reset_calls == ["reset"]
        assert timer_calls == [50]
        assert render_calls[0][0][0] is None
        assert render_calls[0][1]["total_channels"] == 6
        assert render_calls[0][1]["reset_page"] is True
        assert render_calls[0][1]["live_mode"] is True
        assert register_calls == [("session_01", "采集中")]
