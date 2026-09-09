"""Real Qt delivery and real spawn tests for the application recording boundary."""
from collections import UserDict
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import soundfile as sf
from PyQt5.QtCore import QThread

from base.recording_process_protocol import RecordingFailure, RecordingRequest
from base.recording_service import RecordingCallbacks, RecordingService
from consts.recording_preview_consts import (
    PREVIEW_TIME_MODE_CUMULATIVE,
    PREVIEW_TIME_MODE_RELATIVE_LATEST,
    RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY,
)
from unit_test.base.recording_process_fakes import device_info, known_audio


class CapturingBridge:
    """Request boundary spy for existing UI-only configuration unit tests."""
    def __init__(self):
        self.service = SimpleNamespace(busy=False)

    def start(self, request, callbacks):
        from base.recording_service import RecordingSession
        self.request, self.callbacks = request, callbacks
        return RecordingSession(self.service, request, callbacks)


def pump(app, predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while not predicate():
        app.processEvents()
        if time.monotonic() > deadline:
            pytest.fail("Qt recording callback timed out")
        threading.Event().wait(.005)
    app.processEvents()


@pytest.fixture
def service(tmp_path, ui_qapp, monkeypatch):
    from base.file_ops import FileOps
    from consts import model_consts
    # A real completion/relabel must never reach the operator's storage or DB.
    monkeypatch.setattr(model_consts, "DATABASE_PATH", str(tmp_path / "recordings.db"))
    move = FileOps.move_wav_to_dir
    escaped_moves = []
    def move_inside_test(recorded_path, label, recording_root=""):
        target = FileOps.resolve_wav_label_target(recorded_path, label, recording_root)
        if not Path(target).is_relative_to(tmp_path):
            escaped_moves.append(target)
            raise AssertionError("recording test escaped its temporary root")
        return move(recorded_path, label, recording_root)
    monkeypatch.setattr(FileOps, "move_wav_to_dir", move_inside_test)
    instance = RecordingService(
        backend_factory="unit_test.base.recording_process_fakes:process_dependencies",
        backend_options={"trace_dir": str(tmp_path)}, preview_interval=.01)
    yield instance
    instance.shutdown()
    pump(ui_qapp, instance.closed.is_set)
    assert all(not thread.is_alive() for thread in instance.threads)
    assert not escaped_moves, "a caught relabel error must not hide unsafe test paths"


@pytest.mark.parametrize("channel,standard", [(1, 94), (2, 114)])
def test_calibration_child_captures_ten_seconds_and_saves_unchanged_json(
        ui_qapp, service, tmp_path, monkeypatch, channel, standard):
    from ui import calibration_window as calibration
    from ui.recording_service_bridge import RecordingServiceBridge
    from base import soundcard_calibration_manager as manager
    from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis
    from unit_test.base.recording_process_fakes import generated_audio

    service._backend_options.update(frames=441017, chunk_frames=4096)
    registry = tmp_path / "calibration.json"
    monkeypatch.setattr(manager, "MIC_INPUT_CALIBRATION_PATH", str(registry))
    monkeypatch.setattr(manager.SoundDeviceManager, "get_api_info", lambda _index: {"name": "Fake API"})
    widget = calibration.InputCalibration(device_info(), [channel, 0],
        recording_bridge=RecordingServiceBridge(service))
    widget.standard_spl_flag = standard == 94
    widget.calibration_popup = mock.Mock()
    received = []
    calculate = widget._calculate_spl_from_data
    def inspect_audio(data):
        assert QThread.currentThread() is ui_qapp.thread()
        received.append(data.copy())
        return calculate(data)
    widget._calculate_spl_from_data = inspect_audio
    finished = []
    widget.calibration_finished.connect(finished.append)
    try:
        assert widget.clicked_calibration()
        session = widget.streaming_processor.session
        pump(ui_qapp, lambda: bool(finished))
        assert finished == [True]
        expected = generated_audio(0, 441000)[:, channel]
        np.testing.assert_array_equal(received[0], expected)
        smooth = AudioThdFrequencyResponseAnalysis().spl_calculation(expected, method="rms", window_size=1201)
        mid, step = len(smooth) // 2, len(expected) // 3
        factor = 10 ** (round(standard - np.mean(smooth[mid-step:mid+step]), 3) / 20)
        assert manager.get_mic_v2pa_factor(device_info(), [channel]) == pytest.approx(factor)
        saved = json.loads(registry.read_text(encoding="utf-8"))["devices"][0]["channels"][str(channel)]
        assert saved["sample_rate_hz"] == 44100
        assert saved["duration_seconds"] == 10.0
        assert saved["standard_spl_db"] == standard
        assert widget.current_channel == 0
        pump(ui_qapp, session.released.is_set)
        assert not Path(session.request.path).parent.exists()
        trace = json.loads((tmp_path / "trace.json").read_text())
        assert trace["capture_pid"] == trace["writer_pid"] != os.getpid()
        assert trace["written_frames"] == 441000
        assert not list(tmp_path.glob("*.wav"))
    finally:
        widget.close()


def calibration_widget(service, monkeypatch, channels=(1, 0)):
    from ui import calibration_window as calibration
    from ui.recording_service_bridge import RecordingServiceBridge
    monkeypatch.setattr(calibration, "load_mic_channel_v2pa_factors", lambda device: {})
    saved = mock.Mock()
    monkeypatch.setattr(calibration, "save_mic_channel_calibration", saved)
    widget = calibration.InputCalibration(device_info(), channels,
        recording_bridge=RecordingServiceBridge(service))
    widget.calibration_popup = mock.Mock()
    return widget, saved


@pytest.mark.parametrize("cleanup_denied,action", [
    (False, "none"), (True, "none"), (True, "close"), (True, "reset"), (True, "new"),
])
def test_calibration_cleanup_failure_warns_without_releasing_or_revoking_success(
        ui_qapp, service, tmp_path, monkeypatch, cleanup_denied, action):
    from base import recording_service as service_module
    from base import soundcard_calibration_manager as manager
    from ui import calibration_window as calibration
    from ui.recording_service_bridge import RecordingServiceBridge

    service._backend_options.update(frames=441017, chunk_frames=16000)
    registry = tmp_path / "calibration.json"
    monkeypatch.setattr(manager, "MIC_INPUT_CALIBRATION_PATH", str(registry))
    monkeypatch.setattr(manager.SoundDeviceManager, "get_api_info", lambda _index: {"name": "Fake API"})
    def on_resource_warning(*args):
        assert QThread.currentThread() is ui_qapp.thread()
    warnings = mock.Mock(side_effect=on_resource_warning)
    monkeypatch.setattr(calibration.QMessageBox, "warning", warnings)
    bridge = RecordingServiceBridge(service)
    widget = calibration.InputCalibration(device_info(), [1, 0], recording_bridge=bridge)
    widget.calibration_popup = mock.Mock()
    finished = []
    widget.calibration_finished.connect(finished.append)
    pending = []
    enqueue = bridge._enqueue
    def defer_resource_notice(kind, session, value):
        if kind == "release_failed":
            pending.append((kind, session, value))
        else:
            enqueue(kind, session, value)
    monkeypatch.setattr(bridge, "_enqueue", defer_resource_notice)
    original_rmtree = service_module.shutil.rmtree
    denied_path = []
    def deny_owned_cleanup(path, *args, **kwargs):
        if cleanup_denied and not denied_path:
            denied_path.append(str(path))
        if str(path) in denied_path:
            raise PermissionError("injected calibration temp cleanup denial")
        return original_rmtree(path, *args, **kwargs)
    monkeypatch.setattr(service_module.shutil, "rmtree", deny_owned_cleanup)
    session = None
    try:
        assert widget.clicked_calibration()
        session = widget.streaming_processor.session
        pump(ui_qapp, lambda: finished and (session.release_error is not None or session.released.is_set()))
        assert finished == [True]
        assert session.state == "completed"
        assert manager.get_mic_v2pa_factor(device_info(), [1]) > 0
        assert widget.saved_v2pa_factors[1] > 0
        widget.calibration_popup.assert_called_once_with(success_flag=True)
        path = Path(session.request.path)
        if not cleanup_denied:
            assert not pending
            assert session.released.is_set()
            assert not service.is_path_leased(str(path))
            assert not path.parent.exists()
            warnings.assert_not_called()
            return
        pump(ui_qapp, lambda: bool(pending), timeout=3)
        assert len(pending) == 1, "cleanup failure needs its own resource notification"
        assert not session.released.is_set()
        assert service.is_path_leased(str(path))
        assert path.exists()
        if action == "close":
            widget.close()
        elif action == "reset":
            widget.reset_btn_clicked()
        elif action == "new":
            assert widget.clicked_calibration()
        current = widget.streaming_processor
        persisted = registry.read_bytes()
        channel = widget.current_channel
        enqueue(*pending[0])
        enqueue(*pending[0])  # A duplicate must not create another warning.
        ui_qapp.processEvents()
        assert session.state == "completed"
        assert not session.released.is_set()
        assert service.is_path_leased(str(path))
        assert path.exists()
        assert finished == [True]
        assert registry.read_bytes() == persisted
        assert widget.streaming_processor is current
        assert widget.current_channel == channel
        if action == "none":
            warnings.assert_called_once()
            assert "不会改变本次校准结果" in warnings.call_args.args[2]
            assert str(path) in warnings.call_args.args[2]
        else:
            warnings.assert_not_called()
    finally:
        widget.close()
        monkeypatch.setattr(service_module.shutil, "rmtree", original_rmtree)
        if session is not None and session.release_error:
            # Test-only recovery of the injected denial; production must retain
            # the exact lease until an authorized cleanup actually succeeds.
            session._cleanup_failed = False
            session.release_error = None
            service._release(session)
            pump(ui_qapp, session.released.is_set)


@pytest.mark.parametrize("stage,action", [("starting", "cancel"), ("starting", "close"),
    ("recording", "cancel"), ("recording", "reset"), ("recording", "reject")])
def test_calibration_cancel_owns_only_its_session(ui_qapp, service, tmp_path, monkeypatch, stage, action):
    from ui.calibration_window import CalibrationWindow
    from ui.recording_service_bridge import RecordingServiceBridge
    service._cancel_timeout = .2
    service._terminate_timeout = .2
    service._backend_options.update(manual=True)
    if stage == "starting":
        service._backend_options["hang_start_round"] = 1
    widget, saved = calibration_widget(service, monkeypatch)
    monkeypatch.setattr("ui.calibration_window.clear_mic_channel_calibrations", lambda *args: False)
    dialog = None
    if action == "reject":
        widget.close()
        dialog = CalibrationWindow(device_info(), [1], recording_bridge=RecordingServiceBridge(service))
        widget = dialog.input_cal_wnd
        widget.calibration_popup = mock.Mock()
    try:
        assert widget.clicked_calibration()
        session = widget.streaming_processor.session
        if stage == "starting":
            pump(ui_qapp, lambda: (tmp_path / "trace.json").exists())
            assert session.state == "starting"
        else:
            pump(ui_qapp, lambda: session.state == "recording")
        if action == "cancel":
            widget.cancel_calibration()
        elif action == "close":
            widget.close()
        elif action == "reset":
            widget.reset_btn_clicked()
        else:
            dialog.reject()
        assert widget.streaming_processor is None
        pump(ui_qapp, session.released.is_set)
        assert session.state in ("cancelled", "failed")
        assert not Path(session.request.path).parent.exists()
        assert not service.closed.is_set()
        saved.assert_not_called()
        assert widget.channel_combo_box.isEnabled()
    finally:
        widget.close()
        if dialog is not None:
            dialog.close()


def test_calibration_busy_does_not_steal_main_session(ui_qapp, service, tmp_path, monkeypatch):
    host = main_host(service, tmp_path)
    service._backend_options["manual"] = True
    host.judge_play_and_record()
    active = host._recording_process_session
    widget, saved = calibration_widget(service, monkeypatch)
    try:
        assert not widget.clicked_calibration()
        assert "忙" in widget.calibration_popup.call_args.kwargs["message"]
        assert not active.cancel_requested
        widget.close()
        assert not active.cancel_requested
        assert not service.closed.is_set()
        saved.assert_not_called()
    finally:
        host._cancel_process_recording()


def test_calibrator_close_with_paused_reader_then_new_main(ui_qapp, service, tmp_path, monkeypatch):
    from unit_test.base.test_recording_service import PausedReader
    from base.recording_result_reader import ResultReader
    reader = PausedReader()
    service._reader_factory = reader
    service._cancel_timeout = .2
    service._terminate_timeout = .2
    service._backend_options.update(frames=441017, chunk_frames=16000)
    widget, saved = calibration_widget(service, monkeypatch)
    try:
        assert widget.clicked_calibration()
        session = widget.streaming_processor.session
        pump(ui_qapp, reader.entered.is_set)
        old_path = Path(session.request.path)
        old_generation = session.generation
        widget.close()
        assert service.is_path_leased(str(old_path))
        assert old_path.exists()
        assert not session.acknowledged
        pump(ui_qapp, lambda: service.worker_pid is None)
        assert not session.released.is_set()
        service._reader_factory = ResultReader
        service._backend_options.pop("frames")
        host = main_host(service, tmp_path)
        host._on_streaming_complete = mock.Mock(return_value=True)
        host.judge_play_and_record()
        fresh = host._recording_process_session
        fresh_context = host._recording_process_contexts[fresh.request.request_id]
        pump(ui_qapp, lambda: fresh.state == "completed")
        assert fresh.generation != old_generation
        reader.release.set()
        pump(ui_qapp, session.released.is_set)
        assert reader.closed.is_set()
        assert not old_path.parent.exists()
        assert Path(fresh.request.path).exists()
        saved.assert_not_called()
        pump(ui_qapp, lambda: fresh_context.publication_delivered)
        host._on_streaming_complete.assert_called_once()
    finally:
        reader.release.set()
        widget.close()


def test_main_close_invalidates_already_accepted_calibration_before_export_events(
        ui_qapp, service, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QMainWindow
    from main_window import MainWindow
    from ui.recording_service_bridge import RecordingServiceBridge
    widget, saved = calibration_widget(service, monkeypatch)
    service._backend_options.update(frames=441017, chunk_frames=16000)
    queued = threading.Event()
    enqueue = widget.recording_bridge._enqueue
    def pause_accepted(kind, session, value):
        enqueue(kind, session, value)
        if kind == "accepted":
            queued.set()
    monkeypatch.setattr(widget.recording_bridge, "_enqueue", pause_accepted)
    class Window(MainWindow):
        def __init__(self):
            QMainWindow.__init__(self)
            self.recording_bridge = widget.recording_bridge
            self.sequence_window = SimpleNamespace(flush_excel_spool_build=mock.Mock(return_value=[]),
                _shutdown_product_pdf_exporter=mock.Mock(), _cancel_process_recording=mock.Mock())
            self._close_all_subwindows = mock.Mock()
    window = Window()
    try:
        assert widget.clicked_calibration()
        session = widget.streaming_processor.session
        # Wait without Qt processing after provisional accept is requested, so
        # the authoritative event is queued but cannot yet publish the JSON.
        deadline = time.monotonic() + 10
        while not session._accept_requested:
            ui_qapp.processEvents()
            assert time.monotonic() < deadline
            threading.Event().wait(.005)
        assert queued.wait(3)
        window.show()
        window.close()
        pump(ui_qapp, lambda: not window.isVisible())
        saved.assert_not_called()
        window.sequence_window.flush_excel_spool_build.assert_called_once()
        assert widget.streaming_processor is None
    finally:
        widget.close()
        window.deleteLater()


def test_actual_about_to_quit_terminates_native_hang_and_invalidates_calibration(
        ui_qapp, service, tmp_path, monkeypatch):
    from PyQt5.QtCore import QTimer
    widget, saved = calibration_widget(service, monkeypatch)
    service._backend_options.update(manual=True, hang_close=True)
    service._shutdown_timeout = .2
    service._terminate_timeout = .2
    bridge = widget.recording_bridge
    ui_qapp.aboutToQuit.connect(bridge.shutdown)
    try:
        assert widget.clicked_calibration()
        session = widget.streaming_processor.session
        pump(ui_qapp, lambda: session.state == "recording")
        QTimer.singleShot(0, ui_qapp.quit)
        ui_qapp.exec_()
        assert widget.streaming_processor is None
        assert service.closed.wait(3)
        assert service.worker_pid is None
        assert not Path(session.request.path).parent.exists()
        bridge.shutdown()
        saved.assert_not_called()
    finally:
        ui_qapp.aboutToQuit.disconnect(bridge.shutdown)
        widget.close()


def test_standalone_calibration_lazily_owns_service_and_closes_it(ui_qapp, service, monkeypatch):
    from ui import calibration_window as calibration
    monkeypatch.setattr(calibration, "load_mic_channel_v2pa_factors", lambda device: {})
    factory = mock.Mock(return_value=service)
    monkeypatch.setattr(calibration, "RecordingService", factory)
    service._backend_options.update(manual=True, hang_close=True)
    service._shutdown_timeout = .2
    service._terminate_timeout = .2
    widget = calibration.InputCalibration(device_info(), [1])
    widget.calibration_popup = mock.Mock()
    try:
        factory.assert_not_called()
        assert widget.clicked_calibration()
        factory.assert_called_once_with()
        assert widget._owns_recording_bridge
        session = widget.streaming_processor.session
        pump(ui_qapp, lambda: session.state == "recording")
        widget.close()
        pump(ui_qapp, service.closed.is_set, timeout=3)
        assert service.worker_pid is None
    finally:
        widget.close()


def test_launcher_import_in_fresh_process_has_no_qt_or_business_import():
    result = subprocess.run([sys.executable, "-c",
        "import sys; import main_window_Launcher; "
        "assert not any(n.startswith(('PyQt5', 'ui.', 'main_window.')) for n in sys.modules); "
        "assert 'main_window' not in sys.modules"],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("streaming,monitor", [(False, False), (False, True), (True, False), (True, True)])
def test_bridge_real_spawn_delivers_on_gui_thread_and_child_writes(
        ui_qapp, service, tmp_path, streaming, monitor):
    from ui.recording_service_bridge import RecordingServiceBridge
    bridge = RecordingServiceBridge(service)
    events = []
    threads = []
    request = RecordingRequest("qt-round", "main", 100, 9, (0, 2), device_info(),
        str(tmp_path / "qt.wav"), streaming, 2,
        {"enabled": monitor, "device": device_info(), "channels": (0, 1)},
        None, {"enabled": False})
    owner = {}

    def receive(kind, session, value=None):
        assert owner["session"] is session  # registered before even rapid delivery
        threads.append(QThread.currentThread())
        events.append(kind)
        if kind == "offer":
            np.testing.assert_array_equal(value.multi, known_audio()[2:9, (0, 2)])
            session.accept_result()

    callbacks = RecordingCallbacks(
        started=lambda s: receive("started", s),
        preview=lambda s, p: receive("preview", s, p),
        result_ready=lambda s, a: receive("offer", s, a),
        accepted=lambda s, a: receive("accepted", s, a),
        failed=lambda s, f: pytest.fail(f.message),
        released=lambda s: receive("released", s))
    owner["session"] = bridge.start(request, callbacks)
    pump(ui_qapp, lambda: "released" in events)
    assert events.count("accepted") == 1
    assert events.index("offer") < events.index("accepted") < events.index("released")
    assert all(thread is ui_qapp.thread() for thread in threads)
    if not streaming and not monitor:
        assert "preview" not in events
    trace = json.loads((tmp_path / "trace.json").read_text())
    assert trace["capture_pid"] == trace["writer_pid"] != os.getpid()
    audio, rate = sf.read(request.path, dtype="float32", always_2d=True)
    np.testing.assert_array_equal(audio, known_audio()[2:9, (0, 2)])
    assert rate == 100


def main_host(service, tmp_path, *, streaming=True, monitor=False):
    from consts import model_consts
    from ui.recording_service_bridge import RecordingServiceBridge
    from ui.sequence.sequence_widget_recording_process_ops import SequenceWidgetRecordingProcessOpsMixin
    from unit_test.ui.test_streaming_event_dispatch import _WorkflowHost, _Workspace
    from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin

    class Host(_WorkflowHost):
        _on_streaming_complete = SequenceWidgetStreamingOpsMixin._on_streaming_complete

    device = device_info()
    recorded = dict(num_frames=9, input_channels=[0, 2], device=device,
        monitor_playback=monitor, output_device=device, output_channels=[0, 1],
        monitor_gain_db=-6, monitor_mute_leading_samples=2, monitor_fade_in_samples=1)
    host = Host(recorded, [])
    host.recording_bridge = RecordingServiceBridge(service)
    host._recording_input_channels = (0, 2)
    host._active_input_channels = [0, 2]
    host.channel_workspace = _Workspace((0, 2))
    host.recorded_path = str(tmp_path / "main.wav")
    host.recorded_signal_info[model_consts.RECORDING_ROOT_CONFIG_KEY] = str(tmp_path)
    host.mic = device
    host.data_struct = SimpleNamespace(sample_rate=100, store_wave_data=None, store_wave_data_multi=None)
    detail = {"use_streaming_recording": streaming, "monitor_playback": monitor,
        "startup_trim_ms": 20, "audio_validation": {"enabled": False}}
    host.sequence_config = [{"seq1": {"acq": {"detail": detail}}}]
    host._resolve_recording_acq_detail = lambda: detail
    host._should_use_streaming_recording = lambda: streaming or monitor
    host.reset_work_pram = lambda *a, **k: (recorded, 100)
    host._begin_recent_session_for_current_run = lambda: host.events.append("registered")
    host._should_run_silent_analysis_after_recording = lambda: False
    host._cache_condition_record = mock.Mock()
    host._update_current_recent_session_result = mock.Mock()
    host._is_manual_product_condition_cycle_active = lambda: False
    host._advance_manual_product_condition_cycle_after_recording = mock.Mock()
    host.update_player_btn_is_paused = mock.Mock()
    host.count_board = SimpleNamespace(mode="view")
    host.barcode_scanner_box = SimpleNamespace(isChecked=lambda: False)
    host.data_btn = mock.Mock()
    host.replayer_btn = mock.Mock()
    host._handle_invalid_recording = mock.Mock()
    host._finalize_recording_channel_selection = mock.Mock()
    host._cleanup_failed_recording_initialization = mock.Mock(return_value=True)
    return host
def _enable_live_workspace_double(host):
    """Give the legacy process-test plot doubles the Task 5 live contract."""
    for window in host.channel_workspace.all_subwindows():
        window.data = None
        window.is_live_preview = False

        def set_live_data(time_axis, amplitude, *, _window=window):
            _window.data = (
                np.array(time_axis, copy=True),
                np.array(amplitude, copy=True),
            )
            _window.is_live_preview = True
            _window.calls.append(SimpleNamespace(
                channel=_window.channel_index,
                time=_window.data[0].copy(),
                amplitude=_window.data[1].copy(),
                live=True,
            ))

        def set_cumulative_preview_data(time_axis, amplitude, *, _window=window):
            _window.data = (
                np.array(time_axis, copy=True),
                np.array(amplitude, copy=True),
            )
            _window.is_live_preview = True
            _window.presentation_mode = PREVIEW_TIME_MODE_CUMULATIVE
            _window.calls.append(SimpleNamespace(
                channel=_window.channel_index,
                time=_window.data[0].copy(),
                amplitude=_window.data[1].copy(),
                live=True,
            ))

        def snapshot_plot_state(*, _window=window):
            data = None if _window.data is None else tuple(
                value.copy() for value in _window.data
            )
            return data, _window.is_live_preview

        def restore_plot_state(state, *, _window=window):
            data, _window.is_live_preview = state
            _window.data = None if data is None else tuple(
                value.copy() for value in data
            )

        window.set_live_data = set_live_data
        window.set_cumulative_preview_data = set_cumulative_preview_data
        window.snapshot_plot_state = snapshot_plot_state
        window.restore_plot_state = restore_plot_state
    return host


@pytest.mark.parametrize(
    "mode",
    [PREVIEW_TIME_MODE_RELATIVE_LATEST, PREVIEW_TIME_MODE_CUMULATIVE],
)
def test_main_request_freezes_preview_mode_from_acquisition_detail(
    ui_qapp, service, tmp_path, mode
):
    host = main_host(service, tmp_path)
    detail = host.sequence_config[0]["seq1"]["acq"]["detail"]
    detail[RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY] = mode

    host.judge_play_and_record()

    session = host._recording_process_session
    assert session.request.preview_time_mode == mode
    detail[RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY] = (
        PREVIEW_TIME_MODE_CUMULATIVE
        if mode == PREVIEW_TIME_MODE_RELATIVE_LATEST
        else PREVIEW_TIME_MODE_RELATIVE_LATEST
    )
    assert session.request.preview_time_mode == mode
    session.cancel()
    pump(ui_qapp, session.released.is_set)


def test_process_request_preserves_mapping_acquisition_detail(
    service, tmp_path
):
    from ui.sequence.sequence_widget_streaming_ops import (
        SequenceWidgetStreamingOpsMixin,
    )

    host = main_host(service, tmp_path)
    bridge = CapturingBridge()
    host.recording_bridge = bridge
    host.sequence_config = [{
        "seq1": {
            "acq": {
                "detail": UserDict({
                    RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY:
                        PREVIEW_TIME_MODE_CUMULATIVE,
                }),
            },
        },
    }]
    host._resolve_recording_acq_detail = (
        SequenceWidgetStreamingOpsMixin._resolve_recording_acq_detail.__get__(host)
    )
    recorded, sample_rate = host.reset_work_pram("not_labeled")

    host._start_process_recording(recorded, sample_rate)

    assert bridge.request.preview_time_mode == PREVIEW_TIME_MODE_CUMULATIVE


@pytest.mark.parametrize("invalid_detail", [None, []])
def test_process_request_rejects_explicit_non_mapping_detail_before_start(
    service, tmp_path, invalid_detail
):
    from ui.sequence.sequence_widget_streaming_ops import (
        SequenceWidgetStreamingOpsMixin,
    )

    host = main_host(service, tmp_path)
    bridge = CapturingBridge()
    bridge.start = mock.Mock(wraps=bridge.start)
    host.recording_bridge = bridge
    host.sequence_config = [{
        "seq1": {"acq": {"detail": invalid_detail}},
    }]
    host._resolve_recording_acq_detail = (
        SequenceWidgetStreamingOpsMixin._resolve_recording_acq_detail.__get__(host)
    )
    recorded, sample_rate = host.reset_work_pram("not_labeled")

    with pytest.raises(ValueError, match="detail must be a mapping"):
        host._start_process_recording(recorded, sample_rate)

    bridge.start.assert_not_called()




def test_publication_capacity_is_reserved_before_capture_and_reopens_after_terminal(
        ui_qapp, service, tmp_path):
    host = main_host(service, tmp_path)
    recorded, sample_rate = host.reset_work_pram()
    publication_started = {
        name: threading.Event() for name in ("A", "B", "C")}
    publication_finish = {
        name: threading.Event() for name in ("A", "B", "C")}

    def slow_publication(**kwargs):
        name = Path(kwargs["recording_context"].request.path).stem
        publication_started[name].set()
        assert publication_finish[name].wait(10)
        return True

    host._on_streaming_complete = slow_publication
    original_start = host.recording_bridge.start
    host.recording_bridge.start = mock.Mock(wraps=original_start)

    host.recorded_path = str(tmp_path / "A.wav")
    host._start_process_recording(recorded, sample_rate)
    first = host._recording_process_session
    pump(ui_qapp, publication_started["A"].is_set)

    host.recorded_path = str(tmp_path / "B.wav")
    host._start_process_recording(recorded, sample_rate)
    second = host._recording_process_session
    pump(ui_qapp, lambda: second.released.is_set())
    executor = host._request_scoped_recording_executor
    assert executor.reserved_count == 2
    assert executor.can_reserve is False
    assert host._can_start_recording_workflow() is False
    assert host.recording_bridge.start.call_count == 2

    host.recorded_path = str(tmp_path / "C.wav")
    with pytest.raises(RuntimeError, match="CAPACITY_BACKPRESSURE"):
        host._start_process_recording(recorded, sample_rate)
    assert host.recording_bridge.start.call_count == 2
    assert first.request.request_id in host._recording_process_contexts
    assert second.request.request_id in host._recording_process_contexts

    publication_finish["A"].set()
    pump(ui_qapp, lambda: first.request.request_id not in (
        host._recording_process_contexts))
    pump(ui_qapp, publication_started["B"].is_set)
    assert executor.can_reserve is True
    assert host._can_start_recording_workflow() is True

    host._start_process_recording(recorded, sample_rate)
    third = host._recording_process_session
    assert host.recording_bridge.start.call_count == 3
    publication_finish["B"].set()
    pump(ui_qapp, publication_started["C"].is_set)
    publication_finish["C"].set()
    pump(ui_qapp, lambda: third.released.is_set()
         and not host._recording_process_contexts)
    assert executor.reserved_count == 0
    assert host._shutdown_request_scoped_recording_executor() == ()


@pytest.mark.parametrize("failure_point", ("ui_prepare", "service_start"))
def test_pre_capture_failure_rolls_back_publication_reservation(
        ui_qapp, service, tmp_path, failure_point):
    host = main_host(service, tmp_path)
    recorded, sample_rate = host.reset_work_pram()
    if failure_point == "ui_prepare":
        host._begin_recent_session_for_current_run = mock.Mock(
            side_effect=RuntimeError("UI prepare failed"))
    else:
        host.recording_bridge.start = mock.Mock(
            side_effect=RuntimeError("worker rejected start"))

    with pytest.raises(RuntimeError):
        host._start_process_recording(recorded, sample_rate)

    executor = host._request_scoped_recording_executor
    assert executor.reserved_count == 0
    assert executor.can_reserve is True
    assert host._recording_process_contexts == {}
    if failure_point == "ui_prepare":
        assert service.worker_pid is None
    assert host._shutdown_request_scoped_recording_executor() == ()


@pytest.mark.parametrize("streaming,monitor", [(False, False), (False, True), (True, False), (True, True)])
def test_main_workflow_uses_prefinalized_child_arrays_and_snapshots(
        ui_qapp, service, tmp_path, monkeypatch, streaming, monitor):
    host = main_host(service, tmp_path, streaming=streaming, monitor=monitor)
    from ui.sequence import sequence_widget_analysis_ops as analysis, sequence_widget_streaming_ops as stream
    save = mock.Mock(return_value=(0, "saved"))
    monkeypatch.setattr(stream, "RecordingManager", lambda: SimpleNamespace(save_signal_info_to_db=save))
    forbidden = mock.Mock(side_effect=AssertionError("parent recording write/capture is forbidden"))
    monkeypatch.setattr(analysis, "StreamingWavWriter", forbidden)
    monkeypatch.setattr(analysis, "stream_record_without_play", forbidden)
    monkeypatch.setattr(analysis.SoundcardAudioProcessor, "sd_rec", forbidden)
    host._rewrite_recorded_wav = forbidden
    host._append_recording_wav_calibration_metadata = forbidden
    host._resolve_active_recording_waveform_direction = lambda fallback="": "first"

    host.judge_play_and_record()
    session = host._recording_process_session
    assert host._record_workflow_busy
    assert host.events == ["registered"]
    assert session.request.channels == (0, 2)
    assert session.request.device["index"] == 7
    host.mic["index"] = 99
    host._resolve_active_recording_waveform_direction = lambda fallback="": "later"
    host._resolve_recording_acq_detail()["startup_trim_ms"] = 700
    pump(ui_qapp, lambda: session.released.is_set() and not host._record_workflow_busy)
    assert session.request.device["index"] == 7
    assert session.request.trim_samples == 2
    np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, known_audio()[2:9, (0, 2)])
    np.testing.assert_array_equal(host.data_struct.store_wave_data, known_audio()[2:9, (0, 2)].mean(axis=1))
    save.assert_called_once()
    forbidden.assert_not_called()
    host._handle_invalid_recording.assert_not_called()
    host._cache_condition_record.assert_called_once()
    condition_call = host._cache_condition_record.call_args
    assert condition_call.args == ("first",)
    assert condition_call.kwargs["recorded_path"] == session.request.path
    assert condition_call.kwargs["recorded_signal_info"]["sample_rate"] == session.request.sample_rate


def test_main_host_relabel_destination_stays_in_test_root(ui_qapp, service, tmp_path):
    from base.file_ops import FileOps
    from consts import model_consts
    host = main_host(service, tmp_path)
    target = FileOps.resolve_wav_label_target(host.recorded_path, "OK",
        host.recorded_signal_info.get(model_consts.RECORDING_ROOT_CONFIG_KEY, ""))
    assert Path(target).is_relative_to(tmp_path)


def test_workspace_contract_failure_rejects_before_any_publication(
        ui_qapp, service, tmp_path):
    host = main_host(service, tmp_path)
    host.channel_workspace._windows.reverse()
    host.judge_play_and_record()
    session = host._recording_process_session
    pump(ui_qapp, session.released.is_set)
    assert session.state == "failed"
    assert host.data_struct.store_wave_data_multi is None
    host._handle_invalid_recording.assert_called_once()
    assert not Path(session.request.path).exists()


def test_exact_path_cleanup_keeps_reader_lease_until_delete_and_db_cleanup(ui_qapp, service, tmp_path):
    from unit_test.base.test_recording_service import PausedReader
    reader = PausedReader()
    service._reader_factory = reader
    host = main_host(service, tmp_path)
    host.judge_play_and_record()
    session = host._recording_process_session
    try:
        pump(ui_qapp, reader.entered.is_set)
        cleanup = []

        def delete_exact_path(path):
            assert reader.closed.is_set()
            assert service.is_path_leased(path)
            Path(path).unlink()
            cleanup.append(path)

        assert service.defer_path_cleanup(session.request.path, delete_exact_path)
        session.cancel()
        pump(ui_qapp, lambda: session.cancel_requested)
        assert Path(session.request.path).exists()
        assert not cleanup
        with pytest.raises(RuntimeError, match="leased"):
            service.start(session.request)
        reader.release.set()
        pump(ui_qapp, session.released.is_set)
        assert cleanup == [session.request.path]
        assert not service.is_path_leased(session.request.path)
        assert host.data_struct.store_wave_data_multi is None
        # No delayed callback may retain deletion authority over a reused path.
        Path(session.request.path).write_bytes(b"new recording")
        ui_qapp.processEvents()
        assert Path(session.request.path).read_bytes() == b"new recording"
    finally:
        reader.release.set()


def test_relabel_denies_leased_path_before_file_move(ui_qapp, service, tmp_path):
    from unit_test.base.test_recording_service import PausedReader
    reader = PausedReader()
    service._reader_factory = reader
    host = main_host(service, tmp_path)
    host.judge_play_and_record()
    try:
        pump(ui_qapp, reader.entered.is_set)
        result = host._relabel_stored_audio_record(host.recorded_path, {"labels": "not_labeled"}, "OK")
        assert result[0] != 0
        assert "释放" in result[1]
        assert Path(host.recorded_path).exists()
    finally:
        host._recording_process_session.cancel()
        reader.release.set()


def test_main_close_waits_for_service_then_preserves_exports(ui_qapp, service, tmp_path):
    from PyQt5.QtWidgets import QMainWindow
    from main_window import MainWindow
    from ui.recording_service_bridge import RecordingServiceBridge

    class Window(MainWindow):
        def __init__(self):
            QMainWindow.__init__(self)
            self.recording_bridge = RecordingServiceBridge(service)
            self.sequence_window = SimpleNamespace(flush_excel_spool_build=mock.Mock(return_value=[]),
                _shutdown_product_pdf_exporter=mock.Mock())
            self._close_all_subwindows = mock.Mock()

    window = Window()
    window.show()
    window.close()
    assert window.isVisible()
    window.sequence_window.flush_excel_spool_build.assert_not_called()
    pump(ui_qapp, lambda: not window.isVisible())
    assert service.closed.is_set()
    window.sequence_window.flush_excel_spool_build.assert_called_once_with(on_close=False)
    window.sequence_window._shutdown_product_pdf_exporter.assert_called_once()
    window.deleteLater()


@pytest.mark.parametrize(
    "preview_mode",
    [PREVIEW_TIME_MODE_RELATIVE_LATEST, PREVIEW_TIME_MODE_CUMULATIVE],
)
def test_main_close_driven_cancellation_retains_delivered_live_preview(
        ui_qapp, service, tmp_path, monkeypatch, preview_mode):
    from base.recording_process_protocol import RecordingPreview
    from base.streaming_waveform_accumulator import StreamingWaveformSnapshot
    from main_window import MainWindow
    from PyQt5.QtWidgets import QMainWindow, QMessageBox
    from unit_test.base.test_recording_service import PausedReader

    reader = PausedReader()
    service._reader_factory = reader
    host = _enable_live_workspace_double(main_host(service, tmp_path))
    host.sequence_config[0]["seq1"]["acq"]["detail"][
        RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY
    ] = preview_mode
    host.flush_excel_spool_build = mock.Mock(return_value=[])
    host._shutdown_product_pdf_exporter = mock.Mock()
    host.judge_play_and_record()
    session = host._recording_process_session
    pump(ui_qapp, reader.entered.is_set)
    waveform = StreamingWaveformSnapshot(
        (
            np.array([-0.01, 0.0], dtype=np.float64)
            if preview_mode == PREVIEW_TIME_MODE_RELATIVE_LATEST
            else np.array([0.0, 0.01], dtype=np.float64)
        ),
        np.array([1.0, 2.0], dtype=np.float32),
        2,
    )
    preview = RecordingPreview(
        session.request.request_id,
        session.generation,
        1,
        2,
        session.request.channels,
        (waveform, waveform),
        preview_mode,
    )
    host._on_process_recording_preview(session, preview)
    prior_states = [
        plot.snapshot_plot_state()
        for plot in host.channel_workspace.all_subwindows()
    ]

    class Window(MainWindow):
        def __init__(self):
            QMainWindow.__init__(self)
            self.recording_bridge = host.recording_bridge
            self.sequence_window = host
            self._close_all_subwindows = mock.Mock()

    monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)
    window = Window()
    window.show()
    try:
        window.close()
        assert window.isVisible()
        for plot in host.channel_workspace.all_subwindows():
            assert plot.is_live_preview is True
        reader.release.set()
        pump(ui_qapp, lambda: session.released.is_set() and not window.isVisible())
        for plot, prior in zip(host.channel_workspace.all_subwindows(), prior_states):
            current = plot.snapshot_plot_state()
            np.testing.assert_array_equal(current[0][0], prior[0][0])
            np.testing.assert_array_equal(current[0][1], prior[0][1])
            assert current[1] is True
    finally:
        reader.release.set()
        window.deleteLater()


def test_cleanup_error_after_acceptance_does_not_leave_successful_ui_busy(ui_qapp, service, tmp_path, monkeypatch):
    from ui.sequence import sequence_widget_streaming_ops as stream
    from PyQt5.QtWidgets import QMessageBox
    warnings = mock.Mock()
    monkeypatch.setattr(QMessageBox, "warning", warnings)
    save = mock.Mock(return_value=(0, "saved"))
    monkeypatch.setattr(stream, "RecordingManager", lambda: SimpleNamespace(save_signal_info_to_db=save))
    host = main_host(service, tmp_path)
    host.judge_play_and_record()
    session = host._recording_process_session
    service.defer_path_cleanup(session.request.path, mock.Mock(side_effect=OSError("cleanup denied")))
    try:
        pump(ui_qapp, lambda: session.release_error is not None)
        pump(ui_qapp, lambda: not host._record_workflow_busy, timeout=2)
        assert session.state == "completed"
        assert not session.released.is_set()
        assert service.is_path_leased(session.request.path)
        save.assert_called_once()
        host._handle_invalid_recording.assert_not_called()
        warnings.assert_called_once()
        assert session.request.path in warnings.call_args.args[2]
        assert "cleanup denied" in warnings.call_args.args[2]
        np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, known_audio()[2:9, (0, 2)])
        audio, rate = sf.read(session.request.path, dtype="float32", always_2d=True)
        np.testing.assert_array_equal(audio, host.data_struct.store_wave_data_multi)
        assert rate == 100
        host._on_process_recording_release_failed(session, session.release_error)
        host._check_accepted_recording_release(session)
        warnings.assert_called_once()
        save.assert_called_once()
    finally:
        # Restore the deliberately isolated test lease after proving it stayed
        # retained; production never pretends a failed cleanup released a path.
        session._cleanup_failed = False
        session.release_error = None
        service._release(session)


@pytest.mark.parametrize("action", ["none", "serial_abort", "close", "new", "modal_close"])
def test_main_resource_notice_is_once_and_cannot_overwrite_closed_or_replaced_view(
        ui_qapp, service, tmp_path, monkeypatch, action):
    from PyQt5.QtWidgets import QMessageBox
    host = controls_host(service, tmp_path, serial=action == "serial_abort")
    # Hold the actual queued resource event, independently of the timer fallback.
    host._check_accepted_recording_release = mock.Mock()
    bridge = host.recording_bridge
    enqueue = bridge._enqueue
    pending, published = [], []
    def defer_notice(kind, session, value):
        if kind == "release_failed":
            pending.append((kind, session, value))
        else:
            enqueue(kind, session, value)
    monkeypatch.setattr(bridge, "_enqueue", defer_notice)
    host._on_streaming_complete = lambda **kwargs: published.append(host._recording_process_id)
    def warn(*args):
        assert QThread.currentThread() is ui_qapp.thread()
        if action == "modal_close":
            host._cancel_process_recording()
    warnings = mock.Mock(side_effect=warn)
    monkeypatch.setattr(QMessageBox, "warning", warnings)
    host.judge_play_and_record()
    session = host._recording_process_session
    service.defer_path_cleanup(session.request.path, mock.Mock(side_effect=OSError("denied resource cleanup")))
    try:
        pump(ui_qapp, lambda: bool(pending))
        assert session.state == "completed" and not session.released.is_set()
        assert host._recording_process_audio is not None and published == []
        if action == "serial_abort":
            host._serial_product_session_started = True
            host._abort_serial_product_round("operator abort", show_warning=False)
        elif action == "close":
            host._cancel_process_recording()
            host.data_btn.setDisabled(True)
        elif action == "new":
            host._cancel_process_recording()
            host.recorded_path = str(tmp_path / "new-main.wav")
            host._start_process_recording(host._recorded_dict, 100)
        current_id = host._recording_process_id
        history_calls = list(host._discard_current_recent_session.mock_calls)
        enqueue(*pending[0])
        enqueue(*pending[0])
        ui_qapp.processEvents()
        assert host._recording_process_id == current_id
        assert session.state == "completed" and not session.released.is_set()
        assert service.is_path_leased(session.request.path)
        assert Path(session.request.path).exists()
        assert host._discard_current_recent_session.mock_calls == history_calls
        host._handle_invalid_recording.assert_not_called()
        if action in ("none", "modal_close"):
            warnings.assert_called_once()
        else:
            warnings.assert_not_called()
        assert (session.request.request_id in published) is (action == "none")
        if action == "close":
            assert not host.data_btn.isEnabled()
    finally:
        host._cancel_process_recording()
        session._cleanup_failed = False
        session.release_error = None
        service._release(session)


def test_cancel_view_after_service_acceptance_ignores_queued_success(ui_qapp, service, tmp_path):
    host = main_host(service, tmp_path)
    original = host._on_process_recording_result

    def offer(session, audio):
        original(session, audio)
        deadline = time.monotonic() + 5
        while session.state != "completed":
            assert time.monotonic() < deadline
            threading.Event().wait(.002)
        host._cancel_process_recording()

    host._on_process_recording_result = offer
    host.judge_play_and_record()
    session = host._recording_process_session
    pump(ui_qapp, session.released.is_set)
    assert host.data_struct.store_wave_data_multi is None


def test_dead_worker_before_acceptance_never_publishes_success(ui_qapp, service, tmp_path):
    host = main_host(service, tmp_path)
    original = host._on_process_recording_result

    def kill_before_accept(session, audio):
        service._worker.process.terminate()
        service._worker.process.join(5)  # test-only fault injection
        original(session, audio)

    host._on_process_recording_result = kill_before_accept
    host.judge_play_and_record()
    session = host._recording_process_session
    pump(ui_qapp, session.released.is_set)
    assert session.state == "failed"
    assert host.data_struct.store_wave_data_multi is None
    host._handle_invalid_recording.assert_called_once()


def test_bounded_shutdown_reports_retained_paths_without_endless_wait(ui_qapp, monkeypatch):
    from main_window import MainWindow
    from PyQt5.QtWidgets import QMainWindow, QMessageBox
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[-1]))

    class Window(MainWindow):
        def __init__(self):
            QMainWindow.__init__(self)
            self.recording_bridge = SimpleNamespace(service=SimpleNamespace(
                closed=threading.Event(), worker_pid=None, diagnostics=["pending reader: isolated.wav"]))
            self.close = mock.Mock()

    window = Window()
    window._finish_recording_shutdown()
    window.close.assert_called_once()
    assert "isolated.wav" in warnings[0]
    assert not window.recording_bridge.service.closed.is_set()
    window.deleteLater()


def test_preview_replaces_relative_live_envelope_and_ignores_stale_final(ui_qapp, service, tmp_path):
    from base.recording_process_protocol import RecordingPreview
    from base.streaming_waveform_accumulator import StreamingWaveformSnapshot
    host = _enable_live_workspace_double(main_host(service, tmp_path))
    host._recording_process_id = "display"
    host._recording_process_final = False
    host._recording_process_preview_enabled = True
    host._recording_process_sequence = 0
    session = SimpleNamespace(request=SimpleNamespace(
        request_id="display", channels=(0, 2), device=device_info(),
        calibration_metadata=None,
        preview_time_mode=PREVIEW_TIME_MODE_RELATIVE_LATEST), generation=3)

    def preview(sequence, values):
        relative_time = np.arange(1 - len(values), 1, dtype=np.float64)
        waveforms = tuple(StreamingWaveformSnapshot(relative_time,
            np.asarray(values, dtype=np.float32), len(values)) for _ in range(2))
        return RecordingPreview(
            "display", 3, sequence, len(values), (0, 2), waveforms,
            PREVIEW_TIME_MODE_RELATIVE_LATEST,
        )

    host._on_process_recording_preview(session, preview(1, [1, 2]))
    host._on_process_recording_preview(session, preview(2, [1, 2, 9, 3]))
    np.testing.assert_array_equal(host.channel_workspace.calls[-1].amplitude, [1, 2, 9, 3])
    assert all(window.is_live_preview for window in host.channel_workspace.all_subwindows())
    assert len(host.channel_workspace.calls) == 4
    host._recording_process_final = True
    host._on_process_recording_preview(session, preview(3, [8]))
    host._recording_process_final = False
    host._recording_process_id = "next"
    host._on_process_recording_preview(session, preview(4, [7]))
    assert len(host.channel_workspace.calls) == 4


@pytest.mark.parametrize(
    "request_mode,payload_mode,accepted",
    [
        (PREVIEW_TIME_MODE_RELATIVE_LATEST, PREVIEW_TIME_MODE_RELATIVE_LATEST, True),
        (PREVIEW_TIME_MODE_CUMULATIVE, PREVIEW_TIME_MODE_CUMULATIVE, True),
        (PREVIEW_TIME_MODE_RELATIVE_LATEST, PREVIEW_TIME_MODE_CUMULATIVE, False),
        (PREVIEW_TIME_MODE_CUMULATIVE, PREVIEW_TIME_MODE_RELATIVE_LATEST, False),
    ],
)
def test_process_preview_requires_request_payload_mode_match(
        ui_qapp, service, tmp_path, request_mode, payload_mode, accepted):
    from base.recording_process_protocol import RecordingPreview
    from base.streaming_waveform_accumulator import StreamingWaveformSnapshot
    host = _enable_live_workspace_double(main_host(service, tmp_path))
    host.default_logger = mock.Mock()
    host._recording_process_id = "display"
    host._recording_process_final = False
    host._recording_process_preview_enabled = True
    host._recording_process_sequence = 0
    session = SimpleNamespace(request=SimpleNamespace(
        request_id="display", channels=(0, 2), device=device_info(),
        calibration_metadata=None, preview_time_mode=request_mode), generation=3)
    time_axis = (
        np.array([-1.0, 0.0], dtype=np.float64)
        if payload_mode == PREVIEW_TIME_MODE_RELATIVE_LATEST
        else np.array([0.0, 1.0], dtype=np.float64)
    )
    waveform = StreamingWaveformSnapshot(
        time_axis,
        np.array([1.0, 2.0], dtype=np.float32),
        2,
    )
    preview = RecordingPreview(
        "display", 3, 1, 2, (0, 2), (waveform, waveform), payload_mode
    )

    host._on_process_recording_preview(session, preview)

    if accepted:
        assert len(host.channel_workspace.calls) == 2
        assert host._recording_process_preview_enabled is True
        host.default_logger.warning.assert_not_called()
    else:
        assert host.channel_workspace.calls == []
        assert host._recording_process_preview_enabled is False
        host.default_logger.warning.assert_called_once()


def test_process_preview_second_channel_failure_rolls_back_before_disabling(
        ui_qapp, service, tmp_path):
    from base.recording_process_protocol import RecordingPreview
    from base.streaming_waveform_accumulator import StreamingWaveformSnapshot
    host = _enable_live_workspace_double(main_host(service, tmp_path))
    host.default_logger = mock.Mock()
    host._recording_process_id = "display"
    host._recording_process_final = False
    host._recording_process_preview_enabled = True
    host._recording_process_sequence = 0
    session = SimpleNamespace(request=SimpleNamespace(
        request_id="display", channels=(0, 2), device=device_info(),
        calibration_metadata=None,
        preview_time_mode=PREVIEW_TIME_MODE_RELATIVE_LATEST), generation=3)
    prior_time = np.array([-1.0, 0.0], dtype=np.float64)
    prior_amplitude = np.array([7.0, 8.0], dtype=np.float32)
    for window in host.channel_workspace.all_subwindows():
        window.set_live_data(prior_time, prior_amplitude)
    prior_states = [window.snapshot_plot_state()
                    for window in host.channel_workspace.all_subwindows()]
    host.channel_workspace.calls.clear()
    host.channel_workspace.all_subwindows()[1].set_live_data = mock.Mock(
        side_effect=RuntimeError("second process plot failed")
    )
    waveform = StreamingWaveformSnapshot(
        np.array([-0.5, 0.0], dtype=np.float64),
        np.array([1.0, 2.0], dtype=np.float32),
        2,
    )
    preview = RecordingPreview(
        "display", 3, 1, 2, (0, 2), (waveform, waveform),
        PREVIEW_TIME_MODE_RELATIVE_LATEST,
    )

    host._on_process_recording_preview(session, preview)

    for window, prior in zip(host.channel_workspace.all_subwindows(), prior_states):
        current = window.snapshot_plot_state()
        np.testing.assert_array_equal(current[0][0], prior[0][0])
        np.testing.assert_array_equal(current[0][1], prior[0][1])
        assert current[1] is True
    assert host._recording_process_preview_enabled is False
    host.default_logger.warning.assert_called_once()


@pytest.mark.parametrize("terminal", ["failed", "cancelled"])
@pytest.mark.parametrize(
    "preview_mode",
    [PREVIEW_TIME_MODE_RELATIVE_LATEST, PREVIEW_TIME_MODE_CUMULATIVE],
)
def test_process_terminal_retains_last_successful_live_preview(
        ui_qapp, service, tmp_path, monkeypatch, terminal, preview_mode):
    from base.recording_process_protocol import RecordingPreview
    from base.streaming_waveform_accumulator import StreamingWaveformSnapshot
    from ui.sequence.recording_process_context import RecordingProcessContext
    from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin
    from PyQt5.QtWidgets import QMessageBox
    host = _enable_live_workspace_double(main_host(service, tmp_path))
    host._recording_process_id = "display"
    host._recording_process_final = False
    host._recording_process_preview_enabled = True
    host._recording_process_sequence = 0
    host._recording_process_cleanup_owned = False
    host._recording_process_failed = False
    host._recording_process_cancelled = False
    host._recording_process_validated_audio = None
    host._recording_process_audio = None
    host._recording_process_session = None
    host._serial_product_condition_executing = False
    host.clear_all_direction_waveforms = mock.Mock()
    host._handle_invalid_recording = (
        SequenceWidgetStreamingOpsMixin._handle_invalid_recording.__get__(host)
    )
    host._cleanup_failed_recording_initialization = mock.Mock(return_value=False)
    host._discard_current_recent_session = mock.Mock()
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)
    session = SimpleNamespace(request=SimpleNamespace(
        request_id="display", channels=(0, 2), device=device_info(),
        path=str(tmp_path / "display.wav"), calibration_metadata=None,
        preview_time_mode=preview_mode), generation=3)
    context = RecordingProcessContext(
        session.request, "", ("display", None), True, session=session)
    host._recording_process_contexts = {"display": context}
    host._active_recording_process_id = "display"
    waveform = StreamingWaveformSnapshot(
        (
            np.array([-1.0, 0.0], dtype=np.float64)
            if preview_mode == PREVIEW_TIME_MODE_RELATIVE_LATEST
            else np.array([0.0, 1.0], dtype=np.float64)
        ),
        np.array([1.0, 2.0], dtype=np.float32),
        2,
    )
    preview = RecordingPreview(
        "display", 3, 1, 2, (0, 2), (waveform, waveform),
        preview_mode,
    )
    host._on_process_recording_preview(session, preview)
    prior_states = [window.snapshot_plot_state()
                    for window in host.channel_workspace.all_subwindows()]

    if terminal == "failed":
        host._on_process_recording_failed(
            session, RecordingFailure(
                "display", "capture", str(tmp_path / "display.wav"), "failed"
            )
        )
    else:
        host._on_process_recording_cancelled(session, SimpleNamespace())

    for window, prior in zip(host.channel_workspace.all_subwindows(), prior_states):
        current = window.snapshot_plot_state()
        np.testing.assert_array_equal(current[0][0], prior[0][0])
        np.testing.assert_array_equal(current[0][1], prior[0][1])
        assert current[1] is True
    host.clear_all_direction_waveforms.assert_not_called()


def test_process_failure_without_preview_keeps_existing_clear_behavior(
        ui_qapp, service, tmp_path, monkeypatch):
    from ui.sequence.recording_process_context import RecordingProcessContext
    from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin
    from PyQt5.QtWidgets import QMessageBox
    host = _enable_live_workspace_double(main_host(service, tmp_path))
    host._recording_process_id = "display"
    host._recording_process_cleanup_owned = False
    host._recording_process_failed = False
    host._recording_process_session = None
    host._serial_product_condition_executing = False
    host.clear_all_direction_waveforms = mock.Mock()
    host._handle_invalid_recording = (
        SequenceWidgetStreamingOpsMixin._handle_invalid_recording.__get__(host)
    )
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)
    session = SimpleNamespace(request=SimpleNamespace(
        request_id="display", channels=(0, 2), device=device_info(),
        path=str(tmp_path / "display.wav")), generation=3)
    context = RecordingProcessContext(
        session.request, "", ("display", None), False, session=session)
    host._recording_process_contexts = {"display": context}
    host._active_recording_process_id = "display"

    host._on_process_recording_failed(
        session, RecordingFailure(
            "display", "capture", str(tmp_path / "display.wav"), "failed"
        )
    )

    host.clear_all_direction_waveforms.assert_called_once_with()


def test_paused_qt_preview_has_one_wakeup_while_child_file_continues(ui_qapp, service, tmp_path):
    import ctypes
    from ctypes import wintypes

    def rss(pid):
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        handle = kernel.OpenProcess(0x0400 | 0x0010, False, pid)
        assert handle
        try:
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            assert query(handle, ctypes.byref(counters), counters.cb)
            return counters.WorkingSetSize
        finally:
            kernel.CloseHandle(handle)
    from ui.recording_service_bridge import RecordingServiceBridge
    service._backend_options["manual"] = True
    bridge = RecordingServiceBridge(service)
    offered = []
    request = RecordingRequest("paused-ui", "main", 100, 100, (0, 2), device_info(),
        str(tmp_path / "paused.wav"), True, 0, {}, None, {"enabled": False})
    session = bridge.start(request, RecordingCallbacks(
        result_ready=lambda s, a: (offered.append(a), s.accept_result())))
    (tmp_path / "feed-0").touch()

    def wait_without_qt(predicate):
        deadline = time.monotonic() + 10
        while not predicate():
            assert time.monotonic() < deadline
            threading.Event().wait(.005)

    wait_without_qt(lambda: bool(bridge._previews))
    parent_rss = rss(os.getpid())
    child_rss = rss(session.worker_pid)
    (tmp_path / "feed-1").touch()
    (tmp_path / "feed-2").touch()
    wait_without_qt(lambda: session.audio is not None)
    trace = json.loads((tmp_path / "trace.json").read_text())
    assert trace["written_frames"] == 100
    assert len(bridge._previews) == len(bridge._preview_wakeups) == 1
    assert rss(os.getpid()) - parent_rss < 32 * 1024 * 1024
    assert rss(session.worker_pid) - child_rss < 32 * 1024 * 1024
    pump(ui_qapp, session.released.is_set)
    assert len(offered) == 1
    assert offered[0].multi[5, 0] == np.float32(.95)


def test_hidden_sequence_close_does_not_shutdown_injected_service(ui_qapp, service):
    from PyQt5.QtWidgets import QWidget
    from ui.recording_service_bridge import RecordingServiceBridge
    from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin

    class Window(SequenceWidgetStreamingOpsMixin, QWidget):
        def __init__(self):
            QWidget.__init__(self)
            self.recording_bridge = RecordingServiceBridge(service)

    window = Window()
    window.close()
    assert not service.closed.is_set()
    assert not service._closing
    window.deleteLater()


def test_serial_round_cleanup_waits_for_real_reader_before_manager_delete(ui_qapp, service, tmp_path, monkeypatch):
    from unit_test.base.test_recording_service import PausedReader
    from ui.sequence import sequence_widget_serial_trigger_ops as serial
    reader = PausedReader()
    service._reader_factory = reader
    host = main_host(service, tmp_path)
    host.judge_play_and_record()
    session = host._recording_process_session
    host.recent_test_sessions = ["row"]
    host.recent_test_session_by_id = {"row": {"group_id": "round", "recorded_path": host.recorded_path}}
    host._current_recent_session_id = "row"
    deleted = []

    def delete_audio(path):
        assert reader.closed.is_set()
        assert service.is_path_leased(path)
        Path(path).unlink()
        deleted.append(path)
        return 0, "file and database removed"

    monkeypatch.setattr(serial, "RecordingManager", lambda: SimpleNamespace(delete_audio=delete_audio))
    try:
        pump(ui_qapp, reader.entered.is_set)
        session.cancel()
        assert serial.SequenceWidgetSerialTriggerOpsMixin._delete_serial_product_round_records(host, "round") == 1
        assert host.recent_test_sessions == []
        assert deleted == []
        assert Path(host.recorded_path).exists()
        reader.release.set()
        pump(ui_qapp, session.released.is_set)
        assert deleted == [session.request.path]
        assert not Path(host.recorded_path).exists()
    finally:
        reader.release.set()


@pytest.mark.parametrize(
    "preview_mode",
    [PREVIEW_TIME_MODE_RELATIVE_LATEST, PREVIEW_TIME_MODE_CUMULATIVE],
)
def test_final_plot_failure_is_warning_and_still_saves_once(
    ui_qapp, service, tmp_path, monkeypatch, preview_mode
):
    from ui.sequence import sequence_widget_streaming_ops as stream
    host = _enable_live_workspace_double(main_host(service, tmp_path))
    host.sequence_config[0]["seq1"]["acq"]["detail"][
        RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY
    ] = preview_mode
    live_time = (
        np.asarray([-1.0, 0.0], dtype=np.float64)
        if preview_mode == PREVIEW_TIME_MODE_RELATIVE_LATEST
        else np.asarray([0.0, 1.0], dtype=np.float64)
    )
    live_columns = (
        np.asarray([7.0, 8.0], dtype=np.float32),
        np.asarray([9.0, 10.0], dtype=np.float32),
    )
    windows = host.channel_workspace.all_subwindows()
    for window, live_column in zip(windows, live_columns):
        setter = (
            window.set_live_data
            if preview_mode == PREVIEW_TIME_MODE_RELATIVE_LATEST
            else window.set_cumulative_preview_data
        )
        setter(live_time, live_column)
    retained_at_final_projection = {}

    def fail_second_final_projection(*_args):
        retained_at_final_projection["state"] = tuple(
            value.copy() for value in windows[1].data
        )
        raise RuntimeError("Qt plot failed")

    windows[1].set_data = mock.Mock(side_effect=fail_second_final_projection)
    save = mock.Mock(return_value=(0, "saved"))
    warnings = []
    monkeypatch.setattr(stream, "RecordingManager", lambda: SimpleNamespace(save_signal_info_to_db=save))
    monkeypatch.setattr(stream.QMessageBox, "warning", lambda *args: warnings.append(args[-1]))
    host.judge_play_and_record()
    session = host._recording_process_session
    pump(ui_qapp, lambda: session.released.is_set() and not host._record_workflow_busy)
    assert session.state == "completed"
    assert warnings == ["录音已保存，但波形刷新失败。"]
    save.assert_called_once()
    host._handle_invalid_recording.assert_not_called()
    np.testing.assert_array_equal(
        host.data_struct.store_wave_data_multi,
        known_audio()[2:9, (0, 2)],
    )
    saved_audio, saved_rate = sf.read(
        session.request.path,
        dtype="float32",
        always_2d=True,
    )
    np.testing.assert_array_equal(saved_audio, host.data_struct.store_wave_data_multi)
    assert saved_rate == 100
    for window in windows:
        restored_time, restored_column = window.data
        if preview_mode == PREVIEW_TIME_MODE_RELATIVE_LATEST:
            assert restored_time[-1] == 0.0
            assert np.all(restored_time <= 0.0)
        else:
            assert restored_time[0] == 0.0
            assert np.all(restored_time >= 0.0)
        np.testing.assert_array_equal(
            restored_column,
            host.data_struct.store_wave_data_multi[:, window.channel_index // 2],
        )
        assert window.is_live_preview is True
    np.testing.assert_array_equal(
        windows[1].data[0],
        retained_at_final_projection["state"][0],
    )
    np.testing.assert_array_equal(
        windows[1].data[1],
        retained_at_final_projection["state"][1],
    )


@pytest.mark.parametrize("downstream_failure", ["database", "analysis"])
@pytest.mark.parametrize(
    "preview_mode",
    [PREVIEW_TIME_MODE_RELATIVE_LATEST, PREVIEW_TIME_MODE_CUMULATIVE],
)
def test_downstream_failure_preserves_live_preview_until_business_succeeds(
    ui_qapp,
    service,
    tmp_path,
    monkeypatch,
    downstream_failure,
    preview_mode,
):
    from ui.sequence import sequence_widget_streaming_ops as stream

    host = _enable_live_workspace_double(main_host(service, tmp_path))
    host.sequence_config[0]["seq1"]["acq"]["detail"][
        RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY
    ] = preview_mode
    live_time = (
        np.asarray([-1.0, 0.0], dtype=np.float64)
        if preview_mode == PREVIEW_TIME_MODE_RELATIVE_LATEST
        else np.asarray([0.0, 1.0], dtype=np.float64)
    )
    live_amplitude = np.asarray([7.0, 8.0], dtype=np.float32)
    for window in host.channel_workspace.all_subwindows():
        setter = (
            window.set_live_data
            if preview_mode == PREVIEW_TIME_MODE_RELATIVE_LATEST
            else window.set_cumulative_preview_data
        )
        setter(live_time, live_amplitude)
    save = mock.Mock(
        return_value=(
            (1, "database unavailable")
            if downstream_failure == "database"
            else (0, "saved")
        )
    )
    if downstream_failure == "analysis":
        host._should_run_silent_analysis_after_recording = lambda: True
        host._run_request_scoped_recording_analysis = mock.Mock(
            return_value=False)
    monkeypatch.setattr(
        stream,
        "RecordingManager",
        lambda: SimpleNamespace(save_signal_info_to_db=save),
    )

    host.judge_play_and_record()
    session = host._recording_process_session
    pump(ui_qapp, lambda: session.released.is_set() and not host._record_workflow_busy)

    assert session.state == "completed"
    assert save.call_count == (3 if downstream_failure == "database" else 1)
    if downstream_failure == "analysis":
        assert host._run_request_scoped_recording_analysis.call_count == 3
    for window in host.channel_workspace.all_subwindows():
        assert window.is_live_preview is True
        retained_time, retained_amplitude = window.data
        assert retained_amplitude.size > 0
        if preview_mode == PREVIEW_TIME_MODE_RELATIVE_LATEST:
            assert retained_time[-1] == 0.0
            assert np.all(retained_time <= 0.0)
        else:
            assert retained_time[0] == 0.0
            assert np.all(retained_time >= 0.0)
    assert host.player_status_flag is False
    assert host._record_workflow_busy is False


@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize(
    "preview_mode",
    [PREVIEW_TIME_MODE_RELATIVE_LATEST, PREVIEW_TIME_MODE_CUMULATIVE],
)
def test_later_recording_replaces_retained_live_mode_without_inheriting_fixed_range(
    ui_qapp,
    service,
    tmp_path,
    monkeypatch,
    streaming,
    preview_mode,
):
    from ui.sequence import sequence_widget_streaming_ops as stream
    from ui.sequence.channel_plot_workspace import ChannelPlotWorkspace

    host = main_host(service, tmp_path, streaming=streaming, monitor=False)
    workspace = ChannelPlotWorkspace()
    workspace.set_channels((0, 2))
    host.channel_workspace = workspace
    retained_time = (
        np.asarray([-1.0, 0.0])
        if preview_mode == PREVIEW_TIME_MODE_RELATIVE_LATEST
        else np.asarray([0.0, 1.0])
    )
    retained_columns = (
        np.asarray([7.0, 8.0], dtype=np.float32),
        np.asarray([9.0, 10.0], dtype=np.float32),
    )
    for window, column in zip(workspace.all_subwindows(), retained_columns):
        setter = (
            window.set_live_data
            if preview_mode == PREVIEW_TIME_MODE_RELATIVE_LATEST
            else window.set_cumulative_preview_data
        )
        setter(retained_time, column)
    save = mock.Mock(return_value=(0, "saved"))
    monkeypatch.setattr(
        stream,
        "RecordingManager",
        lambda: SimpleNamespace(save_signal_info_to_db=save),
    )
    original_projection = host._project_normalized_waveform_to_workspace
    host._project_normalized_waveform_to_workspace = mock.Mock(
        wraps=original_projection
    )

    try:
        host.judge_play_and_record()
        session = host._recording_process_session
        assert host._recording_process_preview_enabled is streaming
        pump(
            ui_qapp,
            lambda: session.released.is_set() and not host._record_workflow_busy,
        )

        assert session.state == "completed"
        host._project_normalized_waveform_to_workspace.assert_called_once()
        assert (
            host._project_normalized_waveform_to_workspace.call_args.kwargs[
                "max_points"
            ]
            == 48_000
        )
        for window in workspace.all_subwindows():
            time_axis, _amplitude = window.plot_item.getData()
            assert time_axis[0] == 0.0
            assert time_axis[-1] == pytest.approx(
                (host.data_struct.store_wave_data_multi.shape[0] - 1) / 100
            )
            assert window.is_live_preview is False
            assert bool(
                window.plot_widget.getViewBox().state["autoRange"][0]
            ) is True
    finally:
        workspace.close()
        workspace.deleteLater()
        ui_qapp.processEvents()


def test_service_start_rejection_discards_registered_placeholder(ui_qapp, service, tmp_path):
    host = main_host(service, tmp_path)
    placeholder = {"request_id": "attempt", "business_completion_state": "waiting"}

    def begin_placeholder():
        host.events.append("registered")
        host._current_recent_session_id = "attempt"
        host.recent_test_session_by_id = {"attempt": placeholder}

    host._begin_recent_session_for_current_run = begin_placeholder
    host.recording_bridge.start = mock.Mock(side_effect=RuntimeError("path still leased"))
    host._discard_current_recent_session = mock.Mock()
    host.judge_play_and_record()
    assert host.events == ["registered"]
    host._discard_current_recent_session.assert_called_once()
    assert getattr(host, "_recording_process_id", None) is None


@pytest.mark.parametrize("failure_point", ["end_streaming", "direction"])
def test_start_failure_before_recent_placeholder_preserves_completed_history(
        ui_qapp, service, tmp_path, failure_point):
    host = main_host(service, tmp_path)
    prior = {"request_id": "prior", "business_completion_state": "completed"}
    host.recent_test_sessions = ["prior"]
    host.recent_test_session_by_id = {"prior": prior}
    host._current_recent_session_id = "prior"
    host._discard_current_recent_session = mock.Mock()
    if failure_point == "end_streaming":
        host._end_streaming_waveform_session = mock.Mock(
            side_effect=RuntimeError("end streaming failed"))
    else:
        host._resolve_active_recording_waveform_direction = mock.Mock(
            side_effect=RuntimeError("direction failed"))

    recorded, sample_rate = host.reset_work_pram()
    with pytest.raises(RuntimeError):
        host._start_process_recording(recorded, sample_rate)

    assert host.recent_test_sessions == ["prior"]
    assert host.recent_test_session_by_id == {"prior": prior}
    assert host._current_recent_session_id == "prior"
    host._discard_current_recent_session.assert_not_called()
    assert host._get_request_scoped_recording_executor().reserved_count == 0
    host._shutdown_request_scoped_recording_executor()


def test_partial_recent_placeholder_failure_discards_only_attempt_placeholder(
        ui_qapp, service, tmp_path):
    host = main_host(service, tmp_path)
    prior = {"request_id": "prior", "business_completion_state": "completed"}
    attempt = {"request_id": "attempt", "business_completion_state": "waiting"}
    host.recent_test_sessions = ["prior"]
    host.recent_test_session_by_id = {"prior": prior}
    host._current_recent_session_id = "prior"

    def partial_begin():
        host._current_recent_session_id = "attempt"
        host.recent_test_sessions.append("attempt")
        host.recent_test_session_by_id["attempt"] = attempt
        raise RuntimeError("placeholder hook failed")

    def discard_current():
        session_id = host._current_recent_session_id
        host.recent_test_sessions.remove(session_id)
        host.recent_test_session_by_id.pop(session_id)
        host._current_recent_session_id = None

    host._begin_recent_session_for_current_run = partial_begin
    host._discard_current_recent_session = mock.Mock(side_effect=discard_current)
    recorded, sample_rate = host.reset_work_pram()
    with pytest.raises(RuntimeError):
        host._start_process_recording(recorded, sample_rate)

    assert host.recent_test_sessions == ["prior"]
    assert host.recent_test_session_by_id == {"prior": prior}
    assert host._current_recent_session_id is None
    host._discard_current_recent_session.assert_called_once()
    assert host._get_request_scoped_recording_executor().reserved_count == 0
    host._shutdown_request_scoped_recording_executor()


def test_old_publication_does_not_finalize_a_reentrant_new_channel_selection(ui_qapp, service, tmp_path):
    host = main_host(service, tmp_path)
    host._recording_process_id = "old"
    host._recording_process_windows = []
    host._recording_process_audio = SimpleNamespace(
        descriptor=SimpleNamespace(warnings=(), sample_rate=100), mono=np.ones(2), multi=np.ones((2, 1)))
    session = SimpleNamespace(request=SimpleNamespace(request_id="old"), state="completed")
    host._on_streaming_complete = lambda **kwargs: setattr(host, "_recording_process_id", "new")
    host._publish_process_recording(session)
    host._finalize_recording_channel_selection.assert_not_called()


def controls_host(service, tmp_path, *, serial=False):
    """Actual lifecycle handlers with real buttons and no product/hardware UI."""
    from PyQt5.QtWidgets import QPushButton
    from ui.sequence.sequence_widget_serial_trigger_ops import SequenceWidgetSerialTriggerOpsMixin
    from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin
    host = main_host(service, tmp_path)
    del host._cleanup_failed_recording_initialization
    host.data_btn, host.replayer_btn = QPushButton(), QPushButton()
    host._abort_recording_channel_selection = mock.Mock()
    host._unlock_sn_after_recording_if_needed = mock.Mock()
    host._drain_queued_directional_trigger = mock.Mock()
    host._discard_current_recent_session = mock.Mock()
    host._reset_manual_product_condition_cycle = mock.Mock()
    host._show_serial_product_error_once = mock.Mock()
    host._serial_product_condition_executing = serial
    host._serial_product_session_started = False
    host._on_serial_product_runtime_error = SequenceWidgetSerialTriggerOpsMixin._on_serial_product_runtime_error.__get__(host)
    host._abort_serial_product_round = SequenceWidgetSerialTriggerOpsMixin._abort_serial_product_round.__get__(host)
    host._cleanup_streaming_resources = SequenceWidgetStreamingOpsMixin._cleanup_streaming_resources.__get__(host)
    return host


def tcp_host(service, tmp_path, monkeypatch, *, manual=False):
    from ui.sequence import sequence_widget_analysis_ops as analysis, sequence_widget_streaming_ops as stream
    host = main_host(service, tmp_path, streaming=False)
    host.clicked_player_flag = manual
    host.tcp_flag = True
    host.__class__.tcp_server = SimpleNamespace(client_address=["127.0.0.1", 12000])
    host._close_analysis_windows = mock.Mock()
    host._reserve_recorded_count_for_run = mock.Mock(return_value="run-1")
    save = mock.Mock(return_value=(0, "saved"))
    sent = mock.Mock()
    monkeypatch.setattr(stream, "RecordingManager", lambda: SimpleNamespace(save_signal_info_to_db=save))
    monkeypatch.setattr(analysis, "TempTcpClient", sent)
    return host, save, sent


@pytest.mark.parametrize("manual", [False, True])
def test_tcp_finish_waits_for_business_completion_and_snapshots_recipient(
        ui_qapp, service, tmp_path, monkeypatch, manual):
    service._backend_options["manual"] = True
    host, save, sent = tcp_host(service, tmp_path, monkeypatch, manual=manual)
    completion = []
    host._drain_queued_directional_trigger = lambda: completion.append("business complete")

    def on_send(*args):
        assert save.call_count == 1
        assert completion == ["business complete"]
        assert not host._record_workflow_busy

    sent.side_effect = on_send
    host.start_this_play()
    session = host._recording_process_session
    assert host._record_workflow_busy
    assert host.clicked_player_flag is False
    sent.assert_not_called()
    host.__class__.tcp_server.client_address[:] = ["192.0.2.2", 13000]
    host.tcp_flag = False  # current settings cannot change an admitted request's intent
    for index in range(3):
        (tmp_path / f"feed-{index}").touch()
    pump(ui_qapp, lambda: session.released.is_set() and not host._record_workflow_busy)
    if manual:
        sent.assert_not_called()
    else:
        sent.assert_called_once_with("127.0.0.1", 12000, "finish")
    host._on_process_recording_released(session)  # duplicate cannot notify twice
    assert sent.call_count == (0 if manual else 1)


@pytest.mark.parametrize(
    "outcome",
    ["failed", "write_failed", "cancelled", "admission_rejected", "db_failed", "analysis_failed"],
)
def test_tcp_finish_is_not_sent_for_unsuccessful_workflows(ui_qapp, service, tmp_path, monkeypatch, outcome):
    host, save, sent = tcp_host(service, tmp_path, monkeypatch)
    if outcome == "failed":
        service._backend_options["fail_close"] = True
    elif outcome == "write_failed":
        service._backend_options["fail_write"] = True
        host.run = mock.Mock()
    elif outcome == "admission_rejected":
        # A closed service rejects actual admission synchronously; no later
        # callback can repair an eager finish from the submission handler.
        service.shutdown()
        pump(ui_qapp, service.closed.is_set)
    elif outcome == "db_failed":
        save.return_value = (1, "database rejected recording")
    elif outcome == "analysis_failed":
        host._should_run_silent_analysis_after_recording = lambda: True
        host.analysis_config = {
            "display_sequence": ["unsafe"],
            "unsafe": {"type": "UI_ONLY", "analysis_channel": 0},
        }
    host.start_this_play()
    if outcome == "admission_rejected":
        assert getattr(host, "_recording_process_id", None) is None
    else:
        session = host._recording_process_session
        if outcome == "cancelled":
            session.cancel()
        pump(ui_qapp, session.released.is_set)
        if outcome in ("db_failed", "analysis_failed"):
            pump(ui_qapp, lambda: not host._record_workflow_busy)
            assert save.call_count == (3 if outcome == "db_failed" else 1)
        else:
            save.assert_not_called()
        if outcome == "write_failed":
            assert session.state == "failed"
            assert not Path(session.request.path).exists()
            host._handle_invalid_recording.assert_called_once()
            assert "disk write failure" in host._handle_invalid_recording.call_args.args[0]
            host.run.assert_not_called()
    sent.assert_not_called()


def test_tcp_finish_routes_each_reentrant_request_to_its_frozen_recipient(
        ui_qapp, service, tmp_path, monkeypatch):
    host, save, sent = tcp_host(service, tmp_path, monkeypatch)
    sessions = []

    def start_next():
        host._drain_queued_directional_trigger = mock.Mock()
        host.__class__.tcp_server.client_address = ["192.0.2.5", 14000]
        host.start_this_play()
        sessions.append(host._recording_process_session)

    host._drain_queued_directional_trigger = start_next
    host.start_this_play()
    old = host._recording_process_session
    pump(ui_qapp, lambda: bool(sessions))
    assert sessions[0] is not old
    sent.assert_called_once_with("127.0.0.1", 12000, "finish")
    host._on_process_recording_released(old)
    pump(ui_qapp, lambda: sessions[0].released.is_set() and not host._record_workflow_busy)
    assert sent.call_args_list == [
        mock.call("127.0.0.1", 12000, "finish"),
        mock.call("192.0.2.5", 14000, "finish"),
    ]
    assert save.call_count == 2


def test_production_second_start_detaches_finalizing_first_without_cancelling_it(
        ui_qapp, service, tmp_path, monkeypatch):
    from unit_test.base.test_recording_service import PausedReader
    from ui.sequence import sequence_widget_streaming_ops as stream

    from unit_test.base.ve3668n_fakes import capture_request
    request = capture_request(tmp_path / "seed.wav")
    service._backend_factory = "unit_test.base.ve3668n_fakes:capture_dependencies"
    service._backend_options = {"trace_path": str(tmp_path / "ve-trace.jsonl")}
    reader = PausedReader()
    service._reader_factory = reader
    host = main_host(service, tmp_path)
    recorded, _ = host.reset_work_pram("not_labeled")
    recorded.update(device=request.device, input_channels=list(request.channels),
                    num_frames=request.target_samples, monitor_playback=False)
    host.mic = request.device
    host._recording_ve_device = request.device
    host._recording_input_channels = request.channels
    host._active_input_channels = list(request.channels)
    from ui.sequence.channel_plot_workspace import ChannelPlotWorkspace
    host.channel_workspace = ChannelPlotWorkspace()
    host.channel_workspace.set_channels(request.channels)
    host.data_struct.sample_rate = request.sample_rate
    host.reset_work_pram = lambda *a, **k: (recorded, request.sample_rate)
    host._resolve_recording_acq_detail = lambda: {"startup_trim_ms": 0, "audio_validation": {"enabled": False}}
    host._capture_recording_wav_calibration_metadata = lambda: setattr(
        host, "_recording_wav_calibration_metadata", request.calibration_metadata)
    save = mock.Mock(return_value=(0, "saved"))
    monkeypatch.setattr(stream, "RecordingManager", lambda: SimpleNamespace(
        save_signal_info_to_db=save))
    host._cleanup_streaming_resources = (
        stream.SequenceWidgetStreamingOpsMixin._cleanup_streaming_resources.__get__(host)
    )
    host._send_recording_tcp_finish = mock.Mock()
    dropped = []
    original_drop = host._drop_recording_context

    def observe_drop(context):
        dropped.append(context.request.request_id)
        return original_drop(context)

    host._drop_recording_context = observe_drop
    try:
        host.recorded_path = str(tmp_path / "A.wav")
        host.judge_play_and_record(tcp_completion_address=("127.0.0.1", 1001))
        session_a = host._recording_process_session
        pump(ui_qapp, reader.entered.is_set)
        assert service.can_start_recording and service.busy

        host.recorded_path = str(tmp_path / "B.wav")
        host.judge_play_and_record(tcp_completion_address=("127.0.0.1", 1002))
        session_b = host._recording_process_session

        assert session_b is not session_a
        assert not session_a.cancel_requested
        assert host._active_recording_process_id == session_b.request.request_id

        reader.release.set()
        pump(ui_qapp, lambda: session_a.released.is_set() and session_b.released.is_set()
             and not host._recording_process_contexts)
        assert session_a.state == session_b.state == "completed"
        assert dropped.count(session_a.request.request_id) == 1
        assert dropped.count(session_b.request.request_id) == 1
        assert save.call_count == 2
        assert host._send_recording_tcp_finish.call_count == 2
        host._send_recording_tcp_finish.assert_has_calls([
            mock.call(("127.0.0.1", 1001)),
            mock.call(("127.0.0.1", 1002)),
        ], any_order=True)
    finally:
        reader.release.set()


def _late_context(host, request_id, path, token, *, direction, session_id, tcp):
    from ui.sequence.recording_process_context import RecordingProcessContext

    request = RecordingRequest(
        request_id, "main", 100, 2, (0,), device_info(), str(path), False,
        0, {}, None, {"enabled": False})
    return RecordingProcessContext(
        request, direction, (request_id, tcp), False,
        recorded_signal_info={"barcode": request_id, "labels": "OK"},
        recent_session_id=session_id, workflow_token=token)


def test_actual_callback_failure_retains_context_and_retries_uncommitted_effects(
        ui_qapp, service, tmp_path, monkeypatch):
    from unit_test.ui.test_recording_result_overlap import _result_session
    from ui.sequence import sequence_widget_streaming_ops as stream

    host = main_host(service, tmp_path)
    context = _late_context(
        host, "retry-A", tmp_path / "retry-A.wav", object(),
        direction="forward", session_id="retry-session",
        tcp=("127.0.0.1", 1001))
    session, audio = _result_session("retry-A", str(tmp_path / "retry-A.wav"))
    context.session = session
    context.accepted_audio = audio
    host._recording_process_contexts = {"retry-A": context}
    host._active_recording_process_id = "new-owner"
    host.recent_test_session_by_id = {"retry-session": {}}
    host._send_recording_tcp_finish = mock.Mock()
    host._cache_condition_record = mock.Mock()
    scheduled = []
    host._schedule_request_scoped_recording_retry = (
        lambda delay, callback: scheduled.append((delay, callback)))
    saves = mock.Mock(side_effect=[(1, "disk busy"), (0, "saved")])
    monkeypatch.setattr(stream, "RecordingManager", lambda: SimpleNamespace(
        save_signal_info_to_db=saves))

    host._publish_recording_context(context)
    pump(ui_qapp, lambda: bool(scheduled))
    assert host._recording_process_contexts["retry-A"] is context
    assert context.publication_delivered is False
    assert scheduled[0][0] == 50
    host._send_recording_tcp_finish.assert_not_called()
    scheduled.pop(0)[1]()
    pump(ui_qapp, lambda: context.publication_delivered)

    assert saves.call_count == 2
    host._cache_condition_record.assert_called_once()
    assert context.business_effect_attempts["database:upsert"] == 2
    assert context.business_effect_attempts["condition:cache"] == 1
    host._send_recording_tcp_finish.assert_called_once_with(
        ("127.0.0.1", 1001))
    assert "retry-A" not in host._recording_process_contexts


def test_retry_queued_before_shutdown_cannot_recreate_executor_or_mutate_context(
        ui_qapp, service, tmp_path):
    from ui.sequence.request_scoped_recording_executor import (
        RequestScopedExecutionOutcome,
    )

    host = main_host(service, tmp_path)
    context = _late_context(
        host, "closing-A", tmp_path / "closing.wav", object(),
        direction="", session_id="closing-session", tcp=None)
    host._recording_process_contexts = {"closing-A": context}
    context.publication_attempts = 1
    context.publication_max_attempts = 3
    context.publication_retry_deadline = time.monotonic() + 5
    context.publication_executor_generation = (
        host._request_scoped_recording_generation())
    scheduled = []
    host._schedule_request_scoped_recording_retry = (
        lambda delay, callback: scheduled.append(callback))

    host._deliver_request_scoped_recording_publication(
        context, RequestScopedExecutionOutcome("closing-A", value=False),
        context.publication_executor_generation)
    assert len(scheduled) == 1
    snapshot = (context.publication_attempts, context.publication_retry_pending,
                context.publication_delivered)
    host._shutdown_request_scoped_recording_executor()
    scheduled[0]()

    assert host._request_scoped_recording_executor is None
    assert host._get_request_scoped_recording_executor() is None
    assert (context.publication_attempts, context.publication_retry_pending,
            context.publication_delivered) == snapshot
    assert host._recording_process_contexts == {"closing-A": context}


def test_capacity_rejection_auto_resubmits_after_executor_frees(
        ui_qapp, service, tmp_path):
    from unit_test.ui.test_recording_result_overlap import _result_session

    host = main_host(service, tmp_path)
    executor = host._get_request_scoped_recording_executor()
    release = threading.Event()
    assert executor.submit("block-1", lambda: release.wait(2), lambda _outcome: None)
    assert executor.submit("block-2", lambda: None, lambda _outcome: None)
    context = _late_context(
        host, "capacity-A", tmp_path / "capacity.wav", object(),
        direction="", session_id="capacity-session",
        tcp=("127.0.0.1", 1020))
    session, audio = _result_session("capacity-A", str(tmp_path / "capacity.wav"))
    context.session = session
    context.accepted_audio = audio
    context.final_windows = ()
    host._recording_process_contexts = {"capacity-A": context}
    host._active_recording_process_id = "B"
    host._send_recording_tcp_finish = mock.Mock()
    host._on_streaming_complete = mock.Mock(return_value=True)
    scheduled = []
    host._schedule_request_scoped_recording_retry = (
        lambda _delay, callback: scheduled.append(callback))

    host._publish_recording_context(context)
    assert context.publication_attempts == 0
    assert context.publication_submission_failures == 1
    assert len(scheduled) == 1
    release.set()
    pump(ui_qapp, lambda: executor.pending_count == 0)
    scheduled.pop(0)()
    pump(ui_qapp, lambda: context.publication_delivered)

    host._on_streaming_complete.assert_called_once()
    host._send_recording_tcp_finish.assert_called_once_with(
        ("127.0.0.1", 1020))
    assert "capacity-A" not in host._recording_process_contexts


def test_unreserved_internal_capacity_rejection_deadline_cleans_without_tcp(
        ui_qapp, service, tmp_path):
    from unit_test.ui.test_recording_result_overlap import _result_session

    host = main_host(service, tmp_path)
    executor = host._get_request_scoped_recording_executor()
    release = threading.Event()
    assert executor.submit("block-1", lambda: release.wait(2), lambda _outcome: None)
    assert executor.submit("block-2", lambda: None, lambda _outcome: None)
    context = _late_context(
        host, "full-A", tmp_path / "full.wav", object(),
        direction="", session_id="full-session",
        tcp=("127.0.0.1", 1030))
    session, audio = _result_session("full-A", str(tmp_path / "full.wav"))
    context.session = session
    context.accepted_audio = audio
    # This context bypasses normal capture admission and therefore owns no
    # reservation. The deadline is an abnormal internal-path safety net, not
    # the behavior of an accepted recording result.
    assert context.publication_reservation_active is False
    context.publication_max_submission_failures = 2
    host._recording_process_contexts = {"full-A": context}
    host._active_recording_process_id = "B"
    host._send_recording_tcp_finish = mock.Mock()
    scheduled = []
    host._schedule_request_scoped_recording_retry = (
        lambda _delay, callback: scheduled.append(callback))

    host._publish_recording_context(context)
    assert len(scheduled) == 1
    context.publication_retry_deadline = time.monotonic() - .001
    scheduled.pop(0)()

    assert context.publication_delivered is True
    assert context.business_completed is False
    assert "deadline" in context.business_failure.lower()
    assert "full-A" not in host._recording_process_contexts
    host._send_recording_tcp_finish.assert_not_called()
    release.set()


def test_capacity_resubmission_queued_before_close_cannot_recreate_executor(
        ui_qapp, service, tmp_path):
    from unit_test.ui.test_recording_result_overlap import _result_session

    host = main_host(service, tmp_path)
    executor = host._get_request_scoped_recording_executor()
    release = threading.Event()
    assert executor.submit("block-1", lambda: release.wait(2), lambda _outcome: None)
    assert executor.submit("block-2", lambda: None, lambda _outcome: None)
    context = _late_context(
        host, "close-full-A", tmp_path / "close-full.wav", object(),
        direction="", session_id="close-full-session", tcp=None)
    session, audio = _result_session(
        "close-full-A", str(tmp_path / "close-full.wav"))
    context.session = session
    context.accepted_audio = audio
    host._recording_process_contexts = {"close-full-A": context}
    scheduled = []
    host._schedule_request_scoped_recording_retry = (
        lambda _delay, callback: scheduled.append(callback))

    host._publish_recording_context(context)
    assert len(scheduled) == 1
    snapshot = (
        context.publication_submission_failures,
        context.publication_submission_retry_pending,
        context.publication_delivered,
    )
    host._shutdown_request_scoped_recording_executor()
    scheduled.pop(0)()

    assert host._request_scoped_recording_executor is None
    assert host._get_request_scoped_recording_executor() is None
    assert (
        context.publication_submission_failures,
        context.publication_submission_retry_pending,
        context.publication_delivered,
    ) == snapshot
    release.set()


@pytest.mark.parametrize("order", [("c1", "c2"), ("c2", "c1")])
def test_actual_callbacks_publish_one_final_manual_group_count(
        ui_qapp, service, tmp_path, monkeypatch, order):
    from unit_test.ui.test_recording_result_overlap import _result_session
    from ui.sequence import request_scoped_count_publisher as counts
    from ui.sequence import sequence_widget_streaming_ops as stream

    host = main_host(service, tmp_path)
    host._active_recording_process_id = "new-owner"
    host._send_recording_tcp_finish = mock.Mock()
    published = []
    monkeypatch.setattr(counts, "increment_mark_result", published.append)
    monkeypatch.setattr(counts, "increment_shared_result", lambda _label: None)
    monkeypatch.setattr(stream, "RecordingManager", lambda: SimpleNamespace(
        save_signal_info_to_db=mock.Mock(return_value=(0, "saved"))))
    host.recent_test_session_by_id = {}

    def make(condition_key, label):
        request_id = f"manual-{condition_key}"
        context = _late_context(
            host, request_id, tmp_path / f"{request_id}.wav", object(),
            direction=condition_key, session_id=f"session-{condition_key}",
            tcp=None)
        context.manual_product_cycle_active = True
        context.count_mode = "mark"
        context.product_group_id = "production-group"
        context.publication_group_id = "production-group"
        context.product_condition_key = condition_key
        context.product_condition_keys = ("c1", "c2")
        context.recorded_signal_info["labels"] = label
        session, audio = _result_session(request_id, str(context.request.path))
        context.session = session
        context.accepted_audio = audio
        host.recent_test_session_by_id[context.recent_session_id] = {
            "session_id": context.recent_session_id,
            "product_recording_state": "pending",
            "analysis_report_state": "pending",
            "business_completion_state": "pending",
        }
        return context

    contexts = {"c1": make("c1", "OK"), "c2": make("c2", "NG")}
    host._recording_process_contexts = dict(
        (context.request.request_id, context) for context in contexts.values())

    first, second = (contexts[key] for key in order)
    host._publish_recording_context(first)
    pump(ui_qapp, lambda: first.publication_delivered)
    assert published == []
    host._publish_recording_context(second)
    pump(ui_qapp, lambda: second.publication_delivered)

    assert published == ["NG"]
    for context in contexts.values():
        record = host.recent_test_session_by_id[context.recent_session_id]
        assert record["product_recording_state"] == "completed"
        assert record["analysis_report_state"] == "not_required"
        assert record["business_completion_state"] == "completed"


def test_active_mark_retry_splits_durable_counts_and_refreshes_only_on_gui(
        ui_qapp, service, tmp_path, monkeypatch):
    from unit_test.ui.test_recording_result_overlap import _result_session
    from ui.sequence import request_scoped_count_publisher as counts
    from ui.sequence import sequence_widget_streaming_ops as stream

    host = main_host(service, tmp_path)
    token = object()
    context = _late_context(
        host, "count-A", tmp_path / "count.wav", token,
        direction="", session_id="count-session",
        tcp=("127.0.0.1", 1010))
    context.count_mode = "mark"
    context.publication_group_id = "count-run"
    context.final_windows = ()
    session, audio = _result_session("count-A", str(tmp_path / "count.wav"))
    context.session = session
    context.accepted_audio = audio
    host._recording_process_contexts = {"count-A": context}
    host._active_recording_process_id = "count-A"
    host._recording_workflow_token = token
    host._send_recording_tcp_finish = mock.Mock()
    host._cache_condition_record = mock.Mock()
    monkeypatch.setattr(stream, "RecordingManager", lambda: SimpleNamespace(
        save_signal_info_to_db=mock.Mock(return_value=(0, "saved"))))
    main_thread = threading.get_ident()
    calls = {"mark": [], "shared": [], "visual": []}

    def mark(label):
        calls["mark"].append((label, threading.get_ident()))
        return label

    def shared(label):
        calls["shared"].append((label, threading.get_ident()))
        if len(calls["shared"]) == 1:
            raise OSError("shared count failed")
        return label

    monkeypatch.setattr(counts, "increment_mark_result", mark)
    monkeypatch.setattr(counts, "increment_shared_result", shared)
    scheduled = []
    host._schedule_request_scoped_recording_retry = (
        lambda _delay, callback: scheduled.append(callback))
    host.count_board = SimpleNamespace(
        mode="mark",
        append_mark_result_file=mock.Mock(
            side_effect=AssertionError("worker touched QWidget count method")),
        set_test_result_file=mock.Mock(
            side_effect=AssertionError("worker touched QWidget count method")),
        set_mark_text=lambda: calls["visual"].append(threading.get_ident()))

    host._publish_recording_context(context)
    pump(ui_qapp, lambda: bool(scheduled))
    assert len(calls["mark"]) == len(calls["shared"]) == 1
    assert calls["visual"] == []
    scheduled.pop(0)()
    pump(ui_qapp, lambda: context.publication_delivered)

    assert len(calls["mark"]) == 1
    assert len(calls["shared"]) == 2
    assert all(thread_id != main_thread for _label, thread_id in (
        calls["mark"] + calls["shared"]))
    assert calls["visual"] == [main_thread]
    host.count_board.append_mark_result_file.assert_not_called()
    host.count_board.set_test_result_file.assert_not_called()
    host._send_recording_tcp_finish.assert_called_once_with(
        ("127.0.0.1", 1010))
    assert "count-A" not in host._recording_process_contexts


def test_active_owner_applies_full_gui_business_transition_exactly_once(
        ui_qapp, service, tmp_path, monkeypatch):
    from unit_test.ui.test_recording_result_overlap import _result_session
    from ui.sequence import sequence_widget_streaming_ops as stream

    host = main_host(service, tmp_path)
    token = object()
    context = _late_context(
        host, "active-transition", tmp_path / "active.wav", token,
        direction="gear-1", session_id="active-session", tcp=None)
    context.manual_product_cycle_active = True
    context.serial_product_condition_executing = True
    context.product_condition_key = "gear-1"
    context.product_group_id = "round-1"
    context.product_condition_keys = ("gear-1",)
    context.final_windows = ()
    session, audio = _result_session(
        "active-transition", str(tmp_path / "active.wav"))
    context.session = session
    context.accepted_audio = audio
    host._recording_process_contexts = {"active-transition": context}
    host._active_recording_process_id = "active-transition"
    host._recording_workflow_token = token
    host._cache_condition_record = mock.Mock()
    monkeypatch.setattr(stream, "RecordingManager", lambda: SimpleNamespace(
        save_signal_info_to_db=mock.Mock(return_value=(0, "saved"))))
    host._mark_manual_product_condition_recording_completed = mock.Mock()
    host._finalize_serial_product_condition_after_analysis = mock.Mock(
        return_value=True)
    host._advance_manual_product_condition_cycle_after_recording = mock.Mock()
    host._on_serial_product_condition_completed = mock.Mock()
    host._reset_barcode_commit_dedup = mock.Mock()

    host._publish_recording_context(context)
    pump(ui_qapp, lambda: context.publication_delivered)
    host._deliver_request_scoped_recording_publication(
        context, SimpleNamespace(error=None, value=True),
        context.publication_executor_generation)

    host._mark_manual_product_condition_recording_completed.assert_called_once()
    host._finalize_serial_product_condition_after_analysis.assert_called_once()
    host._advance_manual_product_condition_cycle_after_recording.assert_called_once()
    host._on_serial_product_condition_completed.assert_called_once()
    host._reset_barcode_commit_dedup.assert_called_once()
    assert host.data_btn.setEnabled.call_args_list[-1] == mock.call(False)
    assert host._awaiting_ok_ng is True


def test_active_owner_blocking_result_work_is_async_and_delivery_rechecks_owner(
        ui_qapp, service, tmp_path):
    from unit_test.ui.test_recording_result_overlap import _result_session

    host = main_host(service, tmp_path)
    token = object()
    context = _late_context(
        host, "active-A", tmp_path / "active-A.wav", token,
        direction="", session_id="active-session", tcp=None)
    session, audio = _result_session("active-A", str(tmp_path / "active-A.wav"))
    context.session = session
    context.accepted_audio = audio
    host._recording_process_contexts = {"active-A": context}
    host._active_recording_process_id = "active-A"
    host._recording_workflow_token = token
    release = threading.Event()
    started = threading.Event()

    def blocking_completion(**_kwargs):
        started.set()
        release.wait(2)
        return True

    host._on_streaming_complete = blocking_completion
    previous = np.asarray([9.0], dtype=np.float32)
    host.data_struct.store_wave_data = previous
    before = time.monotonic()
    host._publish_recording_context(context)
    assert time.monotonic() - before < .1
    assert started.wait(1)
    host._active_recording_process_id = "B"
    host._recording_workflow_token = object()
    assert host._can_start_recording_workflow() is True
    release.set()
    pump(ui_qapp, lambda: context.publication_delivered)
    np.testing.assert_array_equal(host.data_struct.store_wave_data, previous)


def test_late_a_success_uses_frozen_persistence_without_mutating_active_b(
        ui_qapp, service, tmp_path, monkeypatch):
    from ui.sequence import sequence_widget_streaming_ops as stream

    host = main_host(service, tmp_path)
    token_a, token_b = object(), object()
    context_a = _late_context(
        host, "A", tmp_path / "A.wav", token_a, direction="forward",
        session_id="session-A", tcp=("127.0.0.1", 1001))
    context_b = _late_context(
        host, "B", tmp_path / "B.wav", token_b, direction="reverse",
        session_id="session-B", tcp=("127.0.0.1", 1002))
    host._recording_process_contexts = {"A": context_a, "B": context_b}
    host._active_recording_process_id = "B"
    host._recording_workflow_token = token_b
    b_processor = object()
    host.streaming_processor = b_processor
    host.streaming_stimulus_data = "B-stimulus"
    host.streaming_mode = "B-mode"
    host.player_status_flag = True
    host._awaiting_ok_ng = False
    host._sn_clear_on_next_scan = False
    host._pending_recent_session_append = False
    host._queued_directional_trigger = "B-trigger"
    host._last_committed_barcode = "B-barcode"
    host.recorded_signal_info = {"barcode": "B-barcode", "owner": "B"}
    host._active_product_condition_key = "B-condition"
    host._active_product_condition_config = {"owner": "B"}
    host._manual_product_condition_results = {"B-condition": "pending"}
    host._serial_product_condition_executing = True
    host.recent_test_session_by_id = {"session-A": {}}
    host.data_struct.store_wave_data = np.asarray([9], dtype=np.float32)
    host.data_struct.store_wave_data_multi = np.asarray([[9]], dtype=np.float32)
    host.data_btn = mock.Mock()
    host.replayer_btn = mock.Mock()
    host._reset_barcode_commit_dedup = mock.Mock()
    host._clear_active_recording_direction = mock.Mock()
    host._unlock_sn_after_recording_if_needed = mock.Mock()
    host._on_directional_recording_completed = mock.Mock()
    host._cache_condition_record = (
        stream.SequenceWidgetStreamingOpsMixin._cache_condition_record.__get__(host))
    save = mock.Mock(return_value=(0, "saved"))
    monkeypatch.setattr(stream, "RecordingManager", lambda: SimpleNamespace(
        save_signal_info_to_db=save))

    result = host._on_streaming_complete(
        recorded_mono=np.asarray([1, 2], dtype=np.float32),
        recorded_multi=np.asarray([[1], [2]], dtype=np.float32), sample_rate=100,
        completion_source="process", prefinalized=True,
        final_waveform_windows=(), recording_context=context_a)

    assert result is True
    assert host.streaming_processor is b_processor
    assert host.streaming_stimulus_data == "B-stimulus"
    assert host.streaming_mode == "B-mode"
    assert host.player_status_flag is True
    assert host._awaiting_ok_ng is False
    assert host._sn_clear_on_next_scan is False
    assert host._pending_recent_session_append is False
    assert host._queued_directional_trigger == "B-trigger"
    assert host._last_committed_barcode == "B-barcode"
    assert host.recorded_signal_info == {"barcode": "B-barcode", "owner": "B"}
    assert host._active_product_condition_key == "B-condition"
    assert host._active_product_condition_config == {"owner": "B"}
    assert host._manual_product_condition_results == {"B-condition": "pending"}
    assert host._serial_product_condition_executing is True
    np.testing.assert_array_equal(host.data_struct.store_wave_data, [9])
    np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, [[9]])
    host.data_btn.setEnabled.assert_not_called()
    host.replayer_btn.setEnabled.assert_not_called()
    host._reset_barcode_commit_dedup.assert_not_called()
    host._clear_active_recording_direction.assert_not_called()
    host._unlock_sn_after_recording_if_needed.assert_not_called()
    host._on_directional_recording_completed.assert_not_called()
    assert context_a.condition_record_cache == {
        "forward": {
            "recorded_path": str(tmp_path / "A.wav"),
            "recorded_signal_info": context_a.recorded_signal_info,
            "session_id": "session-A",
        }
    }
    assert context_a.business_completed is True
    assert context_a.awaiting_ok_ng is True
    assert context_a.sn_clear_on_next_scan is True
    assert context_a.pending_recent_session_append is True
    assert context_a.recorded_signal_info["condition_key"] == "forward"
    host._update_current_recent_session_result.assert_not_called()
    assert host.recent_test_session_by_id["session-A"][
        "business_completion_state"] == "completed"
    save.assert_called_once_with(context_a.recorded_signal_info, None)


def test_late_a_completion_exception_does_not_cleanup_active_b(
        ui_qapp, service, tmp_path, monkeypatch):
    from ui.sequence import sequence_widget_streaming_ops as stream

    host = main_host(service, tmp_path)
    token_a, token_b = object(), object()
    context_a = _late_context(
        host, "A", tmp_path / "A.wav", token_a, direction="forward",
        session_id="session-A", tcp=("127.0.0.1", 1001))
    context_b = _late_context(
        host, "B", tmp_path / "B.wav", token_b, direction="reverse",
        session_id="session-B", tcp=("127.0.0.1", 1002))
    host._recording_process_contexts = {"A": context_a, "B": context_b}
    host._active_recording_process_id = "B"
    host._recording_workflow_token = token_b
    b_processor = object()
    host.streaming_processor = b_processor
    host.streaming_stimulus_data = "B-stimulus"
    host.streaming_mode = "B-mode"
    host.player_status_flag = True
    host._awaiting_ok_ng = True
    host._sn_clear_on_next_scan = True
    host._pending_recent_session_append = True
    host._queued_directional_trigger = "B-trigger"
    host._last_committed_barcode = "B-barcode"
    host.recorded_signal_info = {"barcode": "B-barcode", "owner": "B"}
    host._active_product_condition_key = "B-condition"
    host._active_product_condition_config = {"owner": "B"}
    host._manual_product_condition_results = {"B-condition": "pending"}
    host._serial_product_condition_executing = True
    host.data_struct.store_wave_data = np.asarray([9], dtype=np.float32)
    host.data_struct.store_wave_data_multi = np.asarray([[9]], dtype=np.float32)
    host.data_btn = mock.Mock()
    host.replayer_btn = mock.Mock()
    host._reset_barcode_commit_dedup = mock.Mock()
    monkeypatch.setattr(stream, "RecordingManager", lambda: SimpleNamespace(
        save_signal_info_to_db=mock.Mock(side_effect=RuntimeError("A save failed"))))

    assert host._on_streaming_complete(
        recorded_mono=np.asarray([1, 2], dtype=np.float32),
        recorded_multi=np.asarray([[1], [2]], dtype=np.float32), sample_rate=100,
        completion_source="process", prefinalized=True,
        final_waveform_windows=(), recording_context=context_a) is False
    assert context_a.business_completed is False
    assert "A save failed" in context_a.business_failure

    assert host.streaming_processor is b_processor
    assert host.streaming_stimulus_data == "B-stimulus"
    assert host.streaming_mode == "B-mode"
    assert host.player_status_flag is True
    assert host._awaiting_ok_ng is True
    assert host._sn_clear_on_next_scan is True
    assert host._pending_recent_session_append is True
    assert host._queued_directional_trigger == "B-trigger"
    assert host._last_committed_barcode == "B-barcode"
    assert host.recorded_signal_info == {"barcode": "B-barcode", "owner": "B"}
    assert host._active_product_condition_key == "B-condition"
    assert host._active_product_condition_config == {"owner": "B"}
    assert host._manual_product_condition_results == {"B-condition": "pending"}
    assert host._serial_product_condition_executing is True
    np.testing.assert_array_equal(host.data_struct.store_wave_data, [9])
    np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, [[9]])
    host.data_btn.setEnabled.assert_not_called()
    host.replayer_btn.setEnabled.assert_not_called()
    host._reset_barcode_commit_dedup.assert_not_called()


def test_detached_a_required_business_failure_never_sends_tcp_finish_or_advances_b(
        ui_qapp, service, tmp_path, monkeypatch):
    from unit_test.ui.test_recording_result_overlap import _result_session
    from ui.sequence import sequence_widget_streaming_ops as stream

    host = main_host(service, tmp_path)
    token_a, token_b = object(), object()
    context_a = _late_context(
        host, "A", tmp_path / "A.wav", token_a, direction="forward",
        session_id="session-A", tcp=("127.0.0.1", 1001))
    context_b = _late_context(
        host, "B", tmp_path / "B.wav", token_b, direction="reverse",
        session_id="session-B", tcp=("127.0.0.1", 1002))
    context_a.analysis_required = True
    context_a.recorded_signal_info["labels"] = "not_labeled"
    session_a, audio_a = _result_session("A", str(tmp_path / "A.wav"))
    context_a.session = session_a
    context_a.accepted_audio = audio_a
    host._recording_process_contexts = {"A": context_a, "B": context_b}
    host._active_recording_process_id = "B"
    host._recording_workflow_token = token_b
    host._send_recording_tcp_finish = mock.Mock()
    host._run_request_scoped_recording_analysis = mock.Mock(return_value=False)
    host._cache_condition_record = (
        stream.SequenceWidgetStreamingOpsMixin._cache_condition_record.__get__(host))
    update_label = mock.Mock(return_value=(0, "updated"))
    monkeypatch.setattr(stream, "RecordingManager", lambda: SimpleNamespace(
        save_signal_info_to_db=mock.Mock(return_value=(0, "saved")),
        update_audio_label=update_label))
    b_state = {
        "processor": object(), "product": {"owner": "B"},
        "barcode": "B-barcode", "awaiting": True,
    }
    host.streaming_processor = b_state["processor"]
    host._active_product_condition_config = b_state["product"]
    host._last_committed_barcode = b_state["barcode"]
    host._awaiting_ok_ng = b_state["awaiting"]

    host._publish_recording_context(context_a)
    pump(ui_qapp, lambda: context_a.publication_delivered)

    assert context_a.business_completed is False
    assert "analysis" in context_a.business_failure.lower()
    host._send_recording_tcp_finish.assert_not_called()
    assert host.streaming_processor is b_state["processor"]
    assert host._active_product_condition_config is b_state["product"]
    assert host._last_committed_barcode == b_state["barcode"]
    assert host._awaiting_ok_ng is b_state["awaiting"]
    assert host._run_request_scoped_recording_analysis.call_count == 3


def test_detached_a_request_scoped_analysis_completes_before_frozen_tcp_finish(
        ui_qapp, service, tmp_path, monkeypatch):
    from unit_test.ui.test_recording_result_overlap import _result_session
    from ui.sequence import sequence_widget_streaming_ops as stream

    host = main_host(service, tmp_path)
    token_a, token_b = object(), object()
    context_a = _late_context(
        host, "A", tmp_path / "A.wav", token_a, direction="forward",
        session_id="session-A", tcp=("127.0.0.1", 1001))
    context_b = _late_context(
        host, "B", tmp_path / "B.wav", token_b, direction="reverse",
        session_id="session-B", tcp=("127.0.0.1", 1002))
    context_a.analysis_required = True
    context_a.recorded_signal_info["labels"] = "not_labeled"
    context_a.recent_session_config_snapshot = {
        "active_input_channels": [0],
        "analysis_config": {
            "display_sequence": ["spl"],
            "spl": {
                "type": "SPL", "analysis_channel": 0,
                "limit_checked": True, "limit_metric": "overall_spl",
                "scalar_upper_enabled": True, "scalar_upper_value": 200.0,
                "scalar_lower_enabled": True, "scalar_lower_value": -200.0,
            },
        },
    }
    session_a, audio_a = _result_session("A", str(tmp_path / "A.wav"))
    context_a.session = session_a
    context_a.accepted_audio = audio_a
    host._recording_process_contexts = {"A": context_a, "B": context_b}
    host._active_recording_process_id = "B"
    host._recording_workflow_token = token_b
    host.streaming_processor = b_processor = object()
    host.recorded_signal_info = b_signal = {"owner": "B"}
    host._active_product_condition_config = b_product = {"owner": "B"}
    host._last_committed_barcode = "B-barcode"
    host._awaiting_ok_ng = False
    host.recent_test_session_by_id = {"session-A": {}}
    host._send_recording_tcp_finish = mock.Mock()
    host._cache_condition_record = (
        stream.SequenceWidgetStreamingOpsMixin._cache_condition_record.__get__(host))
    update_label = mock.Mock(return_value=(0, "updated"))
    monkeypatch.setattr(stream, "RecordingManager", lambda: SimpleNamespace(
        save_signal_info_to_db=mock.Mock(return_value=(0, "saved")),
        update_audio_label=update_label))

    host._publish_recording_context(context_a)
    pump(ui_qapp, lambda: context_a.publication_delivered)

    assert context_a.business_completed is True
    assert context_a.business_failure == ""
    assert context_a.analysis_result_dict["spl"][0] is True
    assert context_a.analysis_label == "OK"
    assert context_a.analysis_diagnostics == ()
    update_label.assert_called_once_with(
        context_a.recorded_signal_info, context_a.request.path)
    assert context_a.condition_record_cache["forward"][
        "recorded_signal_info"]["labels"] == "OK"
    assert context_a.condition_record_cache["forward"]["recorded_path"] == str(
        tmp_path / "A.wav")
    host._send_recording_tcp_finish.assert_called_once_with(("127.0.0.1", 1001))
    host._update_current_recent_session_result.assert_not_called()
    recent_call = host.recent_test_session_by_id["session-A"]
    assert recent_call["business_completion_state"] == "completed"
    assert recent_call["business_completion_error"] == ""
    assert host.streaming_processor is b_processor
    assert host.recorded_signal_info is b_signal
    assert host._active_product_condition_config is b_product
    assert host._last_committed_barcode == "B-barcode"
    assert host._awaiting_ok_ng is False


def test_detached_product_a_uses_production_request_scoped_completion_without_mutating_b(
        ui_qapp, service, tmp_path, monkeypatch):
    from unit_test.ui.test_recording_result_overlap import _result_session
    from ui.sequence import sequence_widget_analysis_ops as analysis
    from ui.sequence import sequence_widget_streaming_ops as stream
    from ui.sequence import request_scoped_count_publisher as counts

    host = main_host(service, tmp_path)
    token_a, token_b = object(), object()
    context_a = _late_context(
        host, "A", tmp_path / "A.wav", token_a, direction="gear-A",
        session_id="session-A", tcp=("127.0.0.1", 1001))
    context_b = _late_context(
        host, "B", tmp_path / "B.wav", token_b, direction="gear-B",
        session_id="session-B", tcp=("127.0.0.1", 1002))
    context_a.manual_product_cycle_active = True
    context_a.serial_product_condition_executing = True
    context_a.analysis_required = True
    context_a.count_mode = "mark"
    context_a.recorded_signal_info["labels"] = "not_labeled"
    context_a.product_condition_key = "gear-A"
    context_a.product_group_id = "round-A"
    context_a.product_condition_keys = ("gear-A",)
    context_a.recent_session_config_snapshot = {
        "active_input_channels": [0],
        "analysis_config": {
            "display_sequence": ["spl"],
            "spl": {
                "type": "SPL", "analysis_channel": 0,
                "limit_checked": True, "limit_metric": "overall_spl",
                "scalar_upper_enabled": True, "scalar_upper_value": 200.0,
                "scalar_lower_enabled": True, "scalar_lower_value": -200.0,
            },
            "excel": {"type": "Excel", "fast_mode": True},
        },
    }
    context_a.product_report_config = {"enabled": True}
    session_a, audio_a = _result_session("A", str(tmp_path / "A.wav"))
    context_a.session = session_a
    context_a.accepted_audio = audio_a
    host._recording_process_contexts = {"A": context_a, "B": context_b}
    host._active_recording_process_id = "B"
    host._recording_workflow_token = token_b
    host.streaming_processor = b_processor = object()
    host._manual_product_condition_completed_keys = b_completed = {"gear-B"}
    host._manual_product_condition_results = b_results = {"gear-B": "pending"}
    host._active_product_condition_key = "gear-B"
    host._serial_product_condition_executing = True
    a_recent = {"session_id": "session-A", "owner": "A"}
    b_recent = {"session_id": "session-B", "owner": "B"}
    host.recent_test_session_by_id = {
        "session-A": a_recent, "session-B": b_recent,
    }
    host.recent_session_panel = None
    append_mark = mock.Mock()
    host.count_board = SimpleNamespace(
        mode="B-mode", append_mark_result_file=append_mark)
    monkeypatch.setattr(counts, "increment_mark_result", append_mark)
    monkeypatch.setattr(counts, "increment_shared_result", lambda _label: None)
    export_pdf = mock.Mock(return_value=SimpleNamespace(
        ok=True, message="published", file_path="A-report.pdf"))
    monkeypatch.setattr(analysis, "export_product_test_pdf", export_pdf)
    export_csv = mock.Mock(return_value=SimpleNamespace(ok=True, message="spooled"))
    monkeypatch.setattr(analysis, "export_analysis_to_csv_spool", export_csv)
    host._send_recording_tcp_finish = mock.Mock()
    host._cache_condition_record = (
        stream.SequenceWidgetStreamingOpsMixin._cache_condition_record.__get__(host))
    update_label = mock.Mock(return_value=(0, "updated"))
    monkeypatch.setattr(stream, "RecordingManager", lambda: SimpleNamespace(
        save_signal_info_to_db=mock.Mock(return_value=(0, "saved")),
        update_audio_label=update_label))

    host._publish_recording_context(context_a)
    host._publish_recording_context(context_a)
    pump(ui_qapp, lambda: context_a.publication_delivered)

    assert context_a.business_completed is True
    assert context_a.product_completion_result == {
        "request_id": "A",
        "session_id": "session-A",
        "group_id": "round-A",
        "condition_key": "gear-A",
        "condition_keys": ("gear-A",),
        "completed_condition_keys": ("gear-A",),
        "serial_driven": True,
        "analysis_label": "OK",
    }
    assert context_a.analysis_result_dict["spl"][0] is True
    assert context_a.analysis_label == "OK"
    update_label.assert_called_once_with(
        context_a.recorded_signal_info, context_a.request.path)
    host._send_recording_tcp_finish.assert_called_once_with(("127.0.0.1", 1001))
    append_mark.assert_called_once_with("OK")
    export_csv.assert_called_once()
    export_pdf.assert_called_once()
    assert export_pdf.call_args.args[0] == {"enabled": True}
    assert export_pdf.call_args.args[1]["group_id"] == "round-A"
    assert export_pdf.call_args.args[1]["conditions"][0]["key"] == "gear-A"
    assert context_a.business_effects_published is True
    assert context_a.business_effects_result["mark_count_label"] == "OK"
    assert context_a.business_effects_result["excel"] == ("excel",)
    assert context_a.business_effects_result["product_report"] == "A-report.pdf"
    assert host.streaming_processor is b_processor
    assert host._manual_product_condition_completed_keys is b_completed
    assert host._manual_product_condition_results is b_results
    assert host._active_product_condition_key == "gear-B"
    assert host._serial_product_condition_executing is True
    assert host.count_board.mode == "B-mode"
    assert {key: a_recent[key] for key in (
        "session_id", "owner", "product_group_id", "group_id",
        "product_condition_key", "condition_key", "product_recording_state",
        "product_completed_condition_keys", "product_analysis_label",
        "serial_product_condition_completed", "recording_cycle_advanced",
        "serial_condition_finalized", "analysis_report_state",
        "analysis_result_dict", "analysis_diagnostics")
    } == {
        "session_id": "session-A", "owner": "A",
        "product_group_id": "round-A",
        "group_id": "round-A",
        "product_condition_key": "gear-A",
        "condition_key": "gear-A",
        "product_recording_state": "completed",
        "product_completed_condition_keys": ("gear-A",),
        "product_analysis_label": "OK",
        "serial_product_condition_completed": True,
        "recording_cycle_advanced": True,
        "serial_condition_finalized": True,
        "analysis_report_state": "completed",
        "analysis_result_dict": {"spl": (True, 0.0)},
        "analysis_diagnostics": (),
    }
    assert a_recent["business_completion_state"] == "completed"
    assert a_recent["business_completion_error"] == ""
    assert a_recent["recorded_path"] == str(tmp_path / "A.wav")
    assert host.recent_test_session_by_id["session-B"] is b_recent
    assert b_recent == {"session_id": "session-B", "owner": "B"}


def test_production_request_scoped_analysis_refuses_unsafe_ui_bound_analysis(
        ui_qapp, service, tmp_path, monkeypatch):
    from unit_test.ui.test_recording_result_overlap import _result_session
    from ui.sequence import sequence_widget_streaming_ops as stream

    host = main_host(service, tmp_path)
    token_a, token_b = object(), object()
    context_a = _late_context(
        host, "A", tmp_path / "A.wav", token_a, direction="forward",
        session_id="session-A", tcp=("127.0.0.1", 1001))
    context_b = _late_context(
        host, "B", tmp_path / "B.wav", token_b, direction="reverse",
        session_id="session-B", tcp=("127.0.0.1", 1002))
    context_a.analysis_required = True
    context_a.recent_session_config_snapshot = {"analysis_config": {
        "display_sequence": ["unsafe"],
        "unsafe": {"type": "UI_ONLY", "analysis_channel": 0},
    }}
    session_a, audio_a = _result_session("A", str(tmp_path / "A.wav"))
    context_a.session = session_a
    context_a.accepted_audio = audio_a
    host._recording_process_contexts = {"A": context_a, "B": context_b}
    host._active_recording_process_id = "B"
    host._recording_workflow_token = token_b
    host.streaming_processor = b_processor = object()
    host._send_recording_tcp_finish = mock.Mock()
    host._cache_condition_record = (
        stream.SequenceWidgetStreamingOpsMixin._cache_condition_record.__get__(host))
    monkeypatch.setattr(stream, "RecordingManager", lambda: SimpleNamespace(
        save_signal_info_to_db=mock.Mock(return_value=(0, "saved"))))

    host._publish_recording_context(context_a)
    pump(ui_qapp, lambda: context_a.publication_delivered)

    assert context_a.business_completed is False
    assert "limited to SPL/FBA/SPEC" in context_a.business_failure
    host._send_recording_tcp_finish.assert_not_called()
    assert host.streaming_processor is b_processor


def _headless_request(tmp_path, channels=(0, 2)):
    return RecordingRequest(
        "headless-A", "main", 8000, 4096, tuple(channels),
        device_info(),
        str(tmp_path / "headless-A.wav"), False, 0, {}, None,
        {"enabled": False})


def _frozen_ve_analysis_request(tmp_path, factors, *, request_id="frozen-ve"):
    """Build the same validated FrozenConfig request used by production VE capture."""
    from unit_test.base.ve3668n_fakes import (
        device_info as ve_device_info,
        wav_metadata,
    )

    channels = (7, 1)[:len(factors)]
    device = ve_device_info()
    metadata = wav_metadata(
        tuple("measured" if factor is not None else "none" for factor in factors),
        sample_rate=51200,
    )
    metadata["acquisition"]["machine_id"] = device["machine_id"]
    metadata["recorded_channels"] = metadata["recorded_channels"][:len(factors)]
    for index, (physical, factor) in enumerate(zip(channels, factors)):
        entry = metadata["recorded_channels"][index]
        entry["physical_input_channel"] = physical
        if factor is not None:
            entry["v2pa_factor"] = factor
    return RecordingRequest(
        request_id, "main", 51200, 8192, channels, device,
        str(tmp_path / f"{request_id}.wav"), False, 0, {}, metadata,
        {"enabled": False},
    )


def test_frozen_recording_request_calibration_factor_uses_wav_column_order(tmp_path):
    from base.recording_process_protocol import FrozenConfig
    from ui.sequence.request_scoped_recording_analysis import _calibration_factor

    request = _frozen_ve_analysis_request(tmp_path, (2.5, 7.0))

    assert isinstance(request.calibration_metadata, FrozenConfig)
    assert isinstance(request.calibration_metadata["recorded_channels"][0], FrozenConfig)
    assert _calibration_factor(request, 0) == 2.5
    assert _calibration_factor(request, 1) == 7.0


def test_frozen_ve_recording_request_rejects_missing_calibration(tmp_path):
    from ui.sequence.request_scoped_recording_analysis import _calibration_factor

    request = _frozen_ve_analysis_request(tmp_path, (None,))

    with pytest.raises(ValueError, match="In8 has no calibration"):
        _calibration_factor(request, 0)


@pytest.mark.parametrize("factor", [0, -1])
def test_frozen_recording_request_rejects_invalid_calibration_factor(tmp_path, factor):
    from ui.sequence.request_scoped_recording_analysis import _calibration_factor

    request = RecordingRequest(
        "invalid-factor", "main", 8000, 8, (0,), device_info(),
        str(tmp_path / "invalid.wav"), False, 0, {},
        {"recorded_channels": [{"v2pa_factor": factor}]},
        {"enabled": False},
    )

    with pytest.raises(ValueError, match="finite and positive"):
        _calibration_factor(request, 0)


def test_frozen_non_ve_recording_request_keeps_missing_calibration_fallback(tmp_path):
    from ui.sequence.request_scoped_recording_analysis import _calibration_factor

    request = RecordingRequest(
        "missing-factor", "main", 8000, 8, (0,), device_info(),
        str(tmp_path / "missing.wav"), False, 0, {},
        {"recorded_channels": [{"v2pa_factor": None}]},
        {"enabled": False},
    )

    assert _calibration_factor(request, 0) == 1.0


@pytest.mark.parametrize("analysis_type", ["SPL", "FBA", "SPEC"])
def test_frozen_ve_calibration_changes_request_scoped_pressure_results(
        tmp_path, analysis_type):
    from ui.sequence.request_scoped_recording_analysis import analyze_recording_request

    t = np.arange(8192, dtype=np.float64) / 51200
    signal = (.01 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    configs = {
        "SPL": {"type": "SPL", "analysis_channel": 7, "limit_checked": False},
        "FBA": {"type": "FBA", "analysis_channel": 7,
                "band_strategy": "1/3 倍频程", "f_min": 100, "f_max": 5000,
                "limit_checked": False},
        "SPEC": {"type": "SPEC", "analysis_channel": 7,
                 "n_fft": 512, "hop_length": 128},
    }

    def execute(factor):
        request = _frozen_ve_analysis_request(
            tmp_path, (factor,), request_id=f"{analysis_type}-{factor}")
        return analyze_recording_request(
            request=request, recorded_mono=signal,
            recorded_multi=signal[:, None], sample_rate=51200,
            config_snapshot={"analysis_config": {
                "display_sequence": ["item"], "item": configs[analysis_type]}},
            recorded_signal_info={},
        ).analysis_items_data["item"]["result"]

    base = execute(1.0)
    calibrated = execute(2.0)
    if analysis_type == "SPL":
        difference = calibrated["overall_spl"] - base["overall_spl"]
    elif analysis_type == "FBA":
        difference = calibrated["overall_weighted_db"] - base["overall_weighted_db"]
    else:
        base_db = np.asarray(base["spectrogram_db"])
        calibrated_db = np.asarray(calibrated["spectrogram_db"])
        finite = np.isfinite(base_db) & np.isfinite(calibrated_db)
        difference = float(np.median(calibrated_db[finite] - base_db[finite]))
    assert difference == pytest.approx(20 * np.log10(2.0), abs=1e-5)


def test_request_scoped_analysis_dispatches_approved_overlap_modes_and_expands_channels(
        tmp_path, monkeypatch):
    from ui.sequence import request_scoped_recording_analysis as scoped

    modes = ["SPL", "SPEC", "FBA"]
    called = []

    def analyzer(mode):
        def run(*args, **kwargs):
            config = args[-1] if args and isinstance(args[-1], dict) else kwargs.get("config", {})
            called.append((mode, int(config.get("analysis_channel", -1))))
            return (True, 0.0), {"mode": mode}
        return run

    helper_by_mode = {
        "SPL": "_analyze_spl", "SPEC": "_analyze_spec", "FBA": "_analyze_fba",
    }
    for mode, helper in helper_by_mode.items():
        monkeypatch.setattr(scoped, helper, analyzer(mode))
    analysis_config = {"display_sequence": []}
    expanding = set(modes)
    for index, mode in enumerate(modes):
        key = f"item-{index}"
        config = {"type": mode, "analysis_channel": 0}
        if mode in expanding:
            config["analysis_channels"] = [0, 2]
        analysis_config["display_sequence"].append(key)
        analysis_config[key] = config
    analysis_config["display_sequence"].append("excel")
    analysis_config["excel"] = {"type": "Excel", "fast_mode": True}
    audio = np.ones((4096, 2), dtype=np.float32)

    outcome = scoped.analyze_recording_request(
        request=_headless_request(tmp_path), recorded_mono=audio.mean(axis=1),
        recorded_multi=audio, sample_rate=8000,
        config_snapshot={"analysis_config": analysis_config},
        recorded_signal_info={})

    expected_count = len(modes) + len(expanding)
    assert len(outcome.analysis_result_dict) == expected_count
    assert len(outcome.analysis_items_data) == expected_count
    assert not any(item["type"] == "Excel" for item in outcome.analysis_items_data.values())
    for index, mode in enumerate(modes):
        key = f"item-{index}"
        if mode in expanding:
            assert f"{key}--通道1" in outcome.analysis_result_dict
            assert f"{key}--通道3" in outcome.analysis_result_dict
        else:
            assert key in outcome.analysis_result_dict
    assert {mode for mode, _channel in called} == set(modes)


def test_request_scoped_spl_uses_manual_limit_schema_emitted_by_threshold_editor(tmp_path):
    from ui.sequence.request_scoped_recording_analysis import analyze_recording_request

    sample_rate = 8000
    signal = (.01 * np.sin(2 * np.pi * 440 * np.arange(4096) / sample_rate)).astype(np.float32)
    config = {
        "type": "SPL", "analysis_channel": 0, "limit_checked": True,
        "limit_metric": "curve_y", "limit_mode": "manual",
        "manual_input_mode": "segments", "manual_upper_enabled": True,
        "manual_lower_enabled": True,
        "manual_upper_segments": [
            {"start_x": 0, "start_y": 200, "end_x": 1, "end_y": 200}],
        "manual_lower_segments": [
            {"start_x": 0, "start_y": -200, "end_x": 1, "end_y": -200}],
    }
    outcome = analyze_recording_request(
        request=_headless_request(tmp_path, channels=(0,)),
        recorded_mono=signal, recorded_multi=signal[:, None], sample_rate=sample_rate,
        config_snapshot={"analysis_config": {
            "display_sequence": ["spl"], "spl": config}},
        recorded_signal_info={})
    assert outcome.analysis_result_dict == {"spl": (True, 0.0)}
    assert outcome.label == "OK"


def test_request_scoped_fba_uses_production_pure_analyzer(tmp_path):
    from ui.sequence.request_scoped_recording_analysis import analyze_recording_request

    sample_rate = 8000
    t = np.arange(4096) / sample_rate
    multi = np.column_stack((.01 * np.sin(2 * np.pi * 440 * t),
                             .02 * np.sin(2 * np.pi * 880 * t))).astype(np.float32)
    analysis_config = {
        "display_sequence": ["fba"],
        "fba": {"type": "FBA", "analysis_channels": [0, 2],
                "band_strategy": "1/3 倍频程", "f_min": 20, "f_max": 3000,
                "limit_checked": False},
    }
    outcome = analyze_recording_request(
        request=_headless_request(tmp_path), recorded_mono=multi.mean(axis=1),
        recorded_multi=multi, sample_rate=sample_rate,
        config_snapshot={"analysis_config": analysis_config},
        recorded_signal_info={})
    assert set(outcome.analysis_result_dict) == {
        "fba--通道1", "fba--通道3"}
    assert all(value == (None, 0.0) for value in outcome.analysis_result_dict.values())
    assert outcome.analysis_items_data["fba--通道1"]["result"]["band_levels_db"]


def test_request_scoped_ve_pressure_analysis_rejects_missing_frozen_calibration(tmp_path):
    from ui.sequence.request_scoped_recording_analysis import analyze_recording_request

    request = SimpleNamespace(
        path=str(tmp_path / "ve.wav"), channels=(0,),
        device={"backend": "vkinging"},
        calibration_metadata={"recorded_channels": [{"v2pa_factor": None}]})
    signal = np.ones((4096, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="has no calibration"):
        analyze_recording_request(
            request=request, recorded_mono=signal[:, 0], recorded_multi=signal,
            sample_rate=8000, config_snapshot={"analysis_config": {
                "display_sequence": ["spl"],
                "spl": {"type": "SPL", "analysis_channel": 0,
                        "limit_checked": False}}}, recorded_signal_info={})


def test_request_scoped_ve_spec_uses_frozen_pressure_calibration(tmp_path):
    from ui.sequence.request_scoped_recording_analysis import analyze_recording_request

    signal = np.sin(2 * np.pi * 440 * np.arange(2048) / 8000).astype(np.float32)
    config = {"display_sequence": ["spec"],
              "spec": {"type": "Spec", "analysis_channel": 0,
                       "n_fft": 512, "hop_length": 128}}

    def execute(factor):
        request = SimpleNamespace(
            path=str(tmp_path / f"ve-{factor}.wav"), channels=(0,),
            device={"backend": "vkinging"},
            calibration_metadata={"recorded_channels": [
                {"v2pa_factor": factor}]})
        return analyze_recording_request(
            request=request, recorded_mono=signal,
            recorded_multi=signal[:, None], sample_rate=8000,
            config_snapshot={"analysis_config": config},
            recorded_signal_info={})

    base_db = np.asarray(execute(1.0).analysis_items_data[
        "spec"]["result"]["spectrogram_db"])
    calibrated_db = np.asarray(execute(2.0).analysis_items_data[
        "spec"]["result"]["spectrogram_db"])
    finite = np.isfinite(base_db) & np.isfinite(calibrated_db)
    assert np.median(calibrated_db[finite] - base_db[finite]) == pytest.approx(
        20 * np.log10(2.0), abs=1e-5)


def test_request_scoped_directional_mark_count_is_cycle_keyed_and_exactly_once(
        service, tmp_path, monkeypatch):
    from ui.sequence import request_scoped_count_publisher as counts
    host = main_host(service, tmp_path)
    forward = _late_context(
        host, "forward-A", tmp_path / "forward.wav", object(),
        direction="forward", session_id="forward-session", tcp=None)
    reverse = _late_context(
        host, "reverse-A", tmp_path / "reverse.wav", object(),
        direction="reverse", session_id="reverse-session", tcp=None)
    for context, label in ((forward, "OK"), (reverse, "NG")):
        context.directional_cycle_active = True
        context.count_mode = "mark"
        context.barcode = "cycle-001"
        context.publication_group_id = "directional-run-001"
        context.analysis_label = label
        context.recorded_signal_info["labels"] = label
    host.recent_test_session_by_id = {
        "forward-session": {}, "reverse-session": {}}
    append_mark = mock.Mock()
    host.count_board = SimpleNamespace(
        mode="B-mode", append_mark_result_file=append_mark)
    monkeypatch.setattr(counts, "increment_mark_result", append_mark)
    monkeypatch.setattr(counts, "increment_shared_result", lambda _label: None)

    assert host._publish_request_scoped_recording_business(reverse) == {}
    assert host._publish_request_scoped_recording_business(forward) == {
        "mark_count_label": "NG"}
    assert host._publish_request_scoped_recording_business(reverse) == {}
    assert host._publish_request_scoped_recording_business(forward) == {
        "mark_count_label": "NG"}
    append_mark.assert_called_once_with("NG")
    assert host.count_board.mode == "B-mode"


def test_detached_recording_publication_runs_off_gui_and_delivers_once(
        ui_qapp, service, tmp_path):
    from unit_test.ui.test_recording_result_overlap import _result_session

    host = main_host(service, tmp_path)
    context_a = _late_context(
        host, "A", tmp_path / "A.wav", object(), direction="forward",
        session_id="session-A", tcp=None)
    context_b = _late_context(
        host, "B", tmp_path / "B.wav", object(), direction="reverse",
        session_id="session-B", tcp=None)
    session_a, audio_a = _result_session("A", str(tmp_path / "A.wav"))
    session_a.released.set()
    context_a.session = session_a
    context_a.accepted_audio = audio_a
    context_a.validated_audio = audio_a
    host._recording_process_contexts = {"A": context_a, "B": context_b}
    host._active_recording_process_id = "B"
    started = threading.Event()
    release = threading.Event()
    calls = []
    gui_thread = threading.get_ident()

    def complete(**kwargs):
        calls.append((threading.get_ident(), kwargs["recording_context"]))
        started.set()
        release.wait(2)
        return True

    host._on_streaming_complete = complete
    host._notify_process_recording_finished = mock.Mock()
    before = time.monotonic()
    host._publish_recording_context(context_a)
    elapsed = time.monotonic() - before

    assert elapsed < .1
    assert started.wait(2)
    assert calls == [(calls[0][0], context_a)]
    assert calls[0][0] != gui_thread
    assert host._recording_process_contexts["A"] is context_a
    release.set()
    pump(ui_qapp, lambda: "A" not in host._recording_process_contexts)
    host._notify_process_recording_finished.assert_called_once_with(
        session_a, context=context_a)
    assert context_a.publication_delivered is True



def test_tcp_notification_error_does_not_retry_or_invalidate_saved_recording(
        ui_qapp, service, tmp_path, monkeypatch, caplog):
    host, save, sent = tcp_host(service, tmp_path, monkeypatch)
    sent.side_effect = OSError("connection failed after possible delivery")
    host.start_this_play()
    session = host._recording_process_session
    pump(ui_qapp, lambda: session.released.is_set() and not host._record_workflow_busy)
    host._on_process_recording_released(session)
    host._notify_process_recording_finished(session)
    sent.assert_called_once_with("127.0.0.1", 12000, "finish")
    save.assert_called_once()
    assert "Recording TCP completion failed" in caplog.text
    assert session.state == "completed"
    host._handle_invalid_recording.assert_not_called()


@pytest.mark.parametrize("fault", ["fail_close", "hang_close"])
@pytest.mark.parametrize("owner", ["serial_abort", "close", "ordinary"])
def test_delayed_failure_preserves_caller_cleanup_but_recovers_ordinary_failure(
        ui_qapp, service, tmp_path, monkeypatch, caplog, fault, owner):
    from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin
    from PyQt5.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)
    service._backend_options.update(manual=True, **{fault: True})
    service._cancel_timeout = service._terminate_timeout = .2
    host = controls_host(service, tmp_path, serial=owner == "serial_abort")
    host._handle_invalid_recording = SequenceWidgetStreamingOpsMixin._handle_invalid_recording.__get__(host)
    host.left_panel = mock.Mock()
    delivered = []
    callback = host._on_process_recording_failed

    def failed(session, failure):
        callback(session, failure)
        delivered.append(failure)

    host._on_process_recording_failed = failed
    host.judge_play_and_record()
    session = host._recording_process_session
    pump(ui_qapp, lambda: session.state == "recording")
    if owner == "serial_abort":
        host._serial_product_session_started = True
        host._abort_serial_product_round("operator abort", show_warning=False)
    elif owner == "close":
        host._cancel_process_recording()
        host.data_btn.setDisabled(True)
        host.replayer_btn.setDisabled(True)
    elif fault == "fail_close":
        # The ordinary case also covers a genuine capture finalization error,
        # without a preceding UI cancel/abort owning the failure disposition.
        for index in range(3):
            (tmp_path / f"feed-{index}").touch()
    else:
        session.cancel()
    old_stage_calls = list(host.left_panel.mock_calls)
    old_history_calls = list(host._discard_current_recent_session.mock_calls)
    old_cycle_calls = list(host._reset_manual_product_condition_cycle.mock_calls)
    pump(ui_qapp, lambda: bool(delivered) and session.released.is_set())
    assert session.state == "failed"
    assert delivered[0].stage == ("close_wav" if fault == "fail_close" else "cancel_timeout")
    assert host.data_struct.store_wave_data_multi is None
    assert host.data_btn.isEnabled() is (owner == "ordinary")
    assert host.replayer_btn.isEnabled() is (owner == "ordinary")
    if owner != "ordinary":
        assert host.left_panel.mock_calls == old_stage_calls
        assert host._discard_current_recent_session.mock_calls == old_history_calls
        assert host._reset_manual_product_condition_cycle.mock_calls == old_cycle_calls
    else:
        assert not host._record_workflow_busy
        host._discard_current_recent_session.assert_called_once()
    assert delivered[0].message in caplog.text


@pytest.mark.parametrize("serial", [False, True])
def test_admission_rejection_restores_only_nonserial_controls(ui_qapp, service, tmp_path, monkeypatch, serial):
    from PyQt5.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)
    path = str(tmp_path / "main.wav")
    request = RecordingRequest("retained", "main", 100, 9, (0, 2), device_info(),
        path, False, 0, {}, None, {"enabled": False})
    retained = service.start(request, RecordingCallbacks(result_ready=lambda s, a: s.accept_result()))
    service.defer_path_cleanup(path, mock.Mock(side_effect=OSError("retain lease")))
    try:
        pump(ui_qapp, lambda: retained.release_error is not None)
        assert not service.busy
        host = controls_host(service, tmp_path, serial=serial)
        host.judge_play_and_record()  # real service denies its still-leased path
        assert not host._record_workflow_busy
        assert not host.player_status_flag
        assert host.data_btn.isEnabled() is (not serial)
        assert host.replayer_btn.isEnabled() is (not serial)
        # This harness does not create a recent-session placeholder; rejection
        # must not discard any unrelated completed history.
        host._discard_current_recent_session.assert_not_called()
        assert service.is_path_leased(path)
    finally:
        retained._cleanup_failed = False
        retained.release_error = None
        service._release(retained)


@pytest.mark.parametrize("race", ["capacity", "ineligible"])
def test_reentrant_admission_change_after_initial_guard_recovers_reserved_ui(
        ui_qapp, service, tmp_path, monkeypatch, race):
    """The last service start is atomic; any later UI policy denial is explicit."""
    from PyQt5.QtWidgets import QApplication, QMessageBox
    from ui.sequence import sequence_widget_analysis_ops as analysis_ops

    host = controls_host(service, tmp_path)
    warnings = mock.Mock()
    monkeypatch.setattr(QMessageBox, "warning", warnings)
    original_process_events = QApplication.processEvents
    blocker = None

    if race == "ineligible":
        host._recording_admission_config_snapshot = lambda: {
            "analysis_config": {
                "display_sequence": ["ed"],
                "ed": {"type": "ED", "analysis_channel": 0},
            }}

    def change_state_reentrantly():
        nonlocal blocker
        monkeypatch.setattr(analysis_ops.QApplication, "processEvents", original_process_events)
        blocker_request = RecordingRequest(
            "race-blocker", "main", 100, 9, (0, 2), device_info(),
            str(tmp_path / "blocker.wav"), False, 0, {}, None,
            {"enabled": False},
        )
        blocker = service.start(blocker_request)

    monkeypatch.setattr(analysis_ops.QApplication, "processEvents", change_state_reentrantly)
    try:
        host.judge_play_and_record()
        assert getattr(host, "_active_recording_process_id", None) is None
        assert getattr(host, "_recording_process_id", None) is None
        assert not host.player_status_flag
        assert host.data_btn.isEnabled()
        assert host.replayer_btn.isEnabled()
        # The isolated host's begin hook records only an event, not a recent
        # session placeholder, so neither rejection owns history to discard.
        host._discard_current_recent_session.assert_not_called()
        warnings.assert_called_once()
        assert set(service._sessions) == {"race-blocker"}
        assert not service.is_path_leased(host.recorded_path)
    finally:
        if blocker is not None:
            blocker.cancel()
            assert blocker.released.wait(5)


def test_atomic_start_success_keeps_reserved_ui_owned_by_new_session(
        ui_qapp, service, tmp_path):
    host = controls_host(service, tmp_path)

    host.judge_play_and_record()
    session = host._recording_process_session
    try:
        assert session.request.request_id in host._recording_process_contexts
        assert host._record_workflow_busy
        assert host.player_status_flag
        assert not host.data_btn.isEnabled()
        assert not host.replayer_btn.isEnabled()
        assert service._capture_session is session
    finally:
        session.cancel()
        assert session.released.wait(5)


def test_initialization_cleanup_cannot_abort_a_reentrant_new_serial_recording(ui_qapp, service, tmp_path):
    host = controls_host(service, tmp_path)
    host._record_workflow_busy = True

    def start_next():
        host._drain_queued_directional_trigger = mock.Mock()
        host._serial_product_condition_executing = True
        host.judge_play_and_record()

    host._drain_queued_directional_trigger = start_next
    assert host._cleanup_failed_recording_initialization("old start failed") is True
    assert host._record_workflow_busy
    assert host.player_status_flag
    assert host._serial_product_condition_executing
    assert not host.data_btn.isEnabled()
    assert not host.replayer_btn.isEnabled()
    host._show_serial_product_error_once.assert_not_called()


def test_control_enable_event_cannot_reenable_buttons_for_a_new_recording(ui_qapp, service, tmp_path):
    from PyQt5.QtCore import QEvent
    from PyQt5.QtWidgets import QPushButton

    class ReentrantButton(QPushButton):
        on_enabled = None

        def changeEvent(self, event):
            super().changeEvent(event)
            if event.type() == QEvent.EnabledChange and self.isEnabled() and self.on_enabled:
                callback, self.on_enabled = self.on_enabled, None
                callback()

    host = controls_host(service, tmp_path)
    host.data_btn = ReentrantButton()
    host.data_btn.setDisabled(True)
    host.replayer_btn.setDisabled(True)
    host._record_workflow_busy = True
    host.data_btn.on_enabled = host.judge_play_and_record
    assert host._cleanup_failed_recording_initialization("old admission failed") is True
    assert host._record_workflow_busy
    assert not host.data_btn.isEnabled()
    assert not host.replayer_btn.isEnabled()


@pytest.mark.parametrize("owner", ["serial_abort", "close", "serial_runtime", "ordinary"])
def test_delayed_cancellation_preserves_caller_control_disposition(ui_qapp, service, tmp_path, owner):
    from unit_test.base.test_recording_service import PausedReader
    reader = PausedReader()
    service._reader_factory = reader
    host = controls_host(service, tmp_path, serial=owner.startswith("serial"))
    delivered = []
    callback = host._on_process_recording_cancelled

    def cancelled(session, descriptor):
        callback(session, descriptor)
        delivered.append(session)

    host._on_process_recording_cancelled = cancelled
    host.judge_play_and_record()
    session = host._recording_process_session
    try:
        pump(ui_qapp, reader.entered.is_set)
        if owner.startswith("serial"):
            host._serial_product_session_started = True
        if owner == "serial_abort":
            host._abort_serial_product_round("serial stopped", show_warning=False)
        elif owner == "close":
            host._cancel_process_recording()
            host.data_btn.setDisabled(True)
            host.replayer_btn.setDisabled(True)
        else:
            session.cancel()
        assert not host.data_btn.isEnabled()
        assert not host.replayer_btn.isEnabled()
        reader.release.set()
        pump(ui_qapp, lambda: bool(delivered) and session.released.is_set())
        assert host.data_btn.isEnabled() is (owner == "ordinary")
        assert host.replayer_btn.isEnabled() is (owner == "ordinary")
        assert host.data_struct.store_wave_data_multi is None
        if owner != "close":
            host._discard_current_recent_session.assert_called_once()
    finally:
        reader.release.set()


@pytest.mark.parametrize("nested", [False, True])
def test_relabel_rejects_uncreated_leased_destination_then_succeeds_after_release(
        ui_qapp, service, tmp_path, monkeypatch, nested):
    from base.recording_management import RecordingManager
    from consts import model_consts
    root = tmp_path / "recordings"
    folder = root / "Model" / "batch" if nested else root
    source = folder / "OK" / "foo.wav"
    target = folder / "not_labeled" / source.name
    source.parent.mkdir(parents=True)
    source.write_bytes(b"saved original recording")
    service._backend_options["hang_ready"] = True
    service._ready_timeout = service._cancel_timeout = service._terminate_timeout = .2
    request = RecordingRequest("reserved-destination", "main", 100, 9, (0, 2), device_info(),
        str(target), False, 0, {}, None, {"enabled": False})
    session = service.start(request)
    host = main_host(service, tmp_path)
    update = mock.Mock(return_value=(0, "updated"))
    monkeypatch.setattr(RecordingManager, "update_audio_label", update)
    info = {"labels": "OK", "file_path": str(source), model_consts.RECORDING_ROOT_CONFIG_KEY: str(root)}
    assert service.is_path_leased(str(target))
    assert not target.exists()
    result = host._relabel_stored_audio_record(str(source), info, "not_labeled")
    assert result[0] != 0
    assert "释放" in result[1]
    assert source.read_bytes() == b"saved original recording"
    assert not target.exists()
    update.assert_not_called()
    assert not target.parent.exists()
    session.cancel()
    pump(ui_qapp, session.released.is_set)
    result = host._relabel_stored_audio_record(str(source), info, "not_labeled")
    assert result[0] == 0
    assert Path(result[2]) == target
    assert target.read_bytes() == b"saved original recording"
    assert not source.exists()
    update.assert_called_once()
