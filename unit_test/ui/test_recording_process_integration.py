"""Real Qt delivery and real spawn tests for the application recording boundary."""
import json
import logging
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

from base.recording_process_protocol import RecordingRequest
from base.recording_service import RecordingCallbacks, RecordingService
from unit_test.base.recording_process_fakes import device_info, known_audio
from unit_test.base.test_recording_capture_startup_timing import records


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


def assert_main_envelope(audio, request, expected):
    from base.recording_final_result import PreparedRecordingResult

    assert isinstance(audio, PreparedRecordingResult)
    assert audio.is_prepared_for(request)
    assert not hasattr(audio, "multi") and not hasattr(audio, "mono")
    assert audio.descriptor.final_frames == len(expected)
    assert audio.display_error is None
    assert len(audio.waveforms) == expected.shape[1] == len(request.channels)
    for column, (times, values) in enumerate(audio.waveforms):
        np.testing.assert_array_equal(times, np.arange(len(expected)) / request.sample_rate)
        np.testing.assert_array_equal(values, expected[:, column])


def assert_saved_main_envelope(host, session, expected):
    assert_main_envelope(session.audio, session.request, expected)
    assert session.reader is None and session._reader_released
    assert host.data_struct.store_wave_data_multi is None
    assert host.data_struct.store_wave_data is None
    assert host.data_struct.audio_lenth == len(expected)
    source = host.data_struct.recording_source
    assert source.request is session.request
    assert source.path == session.request.path
    assert source.sample_rate == session.request.sample_rate
    assert source.frame_count == len(expected)


def wait_for_main_finalization(app, service, session):
    """Wait for the real child's open writer, before completion or release."""
    trace_path = Path(service._backend_options["trace_dir"]) / "trace.json"
    pump(app, lambda: trace_path.exists() and json.loads(
        trace_path.read_text()).get("finalize_waiting_for_release", False))
    assert not session._child_released
    assert session.reader is None
    assert session.audio is None
    assert service.is_path_leased(session.request.path)
    assert Path(session.request.path).exists()


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
    # closed is published before the supervisor returns from shutdown callbacks.
    pump(ui_qapp, lambda: instance.closed.is_set()
         and all(not thread.is_alive() for thread in instance.threads))
    assert all(not thread.is_alive() for thread in instance.threads)
    assert not escaped_moves, "a caught relabel error must not hide unsafe test paths"


@pytest.mark.parametrize("channel,standard", [(1, 94), (2, 114)])
def test_calibration_child_captures_ten_seconds_and_saves_unchanged_json(
        ui_qapp, service, tmp_path, monkeypatch, channel, standard):
    from ui import calibration_window as calibration
    from ui.recording_service_bridge import RecordingServiceBridge
    from base import soundcard_calibration_manager as manager
    from base.core_algorithm.response.spl_analyzer import spl_calculation
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
        smooth = spl_calculation(expected, method="rms", window_size=1201)
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
        widget.calibration_popup.assert_called_once_with(
            success_flag=True,
            message=(
                "校准成功\n"
                f"本次校准结果：{widget.saved_v2pa_factors[1]:.6f} Pa/V\n"
                "下次校准通道：In1"
            ),
        )
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
    from unit_test.base.test_recording_capture import request
    service._backend_options["manual"] = True
    active = service.start(request(tmp_path), RecordingCallbacks())
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
        active.cancel()


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
        from unit_test.base.test_recording_capture import request
        accepted = mock.Mock()
        fresh = service.start(request(tmp_path), RecordingCallbacks(
            result_ready=lambda session, audio: session.accept_result(), accepted=accepted))
        pump(ui_qapp, lambda: fresh.state == "completed")
        assert fresh.generation != old_generation
        reader.release.set()
        pump(ui_qapp, session.released.is_set)
        assert reader.closed.is_set()
        assert not old_path.parent.exists()
        assert Path(fresh.request.path).exists()
        saved.assert_not_called()
        accepted.assert_called_once()
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
            self.sequence_window = SimpleNamespace(close_workbench_input=mock.Mock(), close=mock.Mock())
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
        window.sequence_window.close_workbench_input.assert_called_once()
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
            assert_main_envelope(value, session.request, known_audio()[2:9, (0, 2)])
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
    if not streaming:
        assert "preview" not in events
    trace = json.loads((tmp_path / "trace.json").read_text())
    assert trace["capture_pid"] == trace["writer_pid"] != os.getpid()
    audio, rate = sf.read(request.path, dtype="float32", always_2d=True)
    np.testing.assert_array_equal(audio, known_audio()[2:9, (0, 2)])
    assert rate == 100


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
    assert not hasattr(offered[0], "multi") and not hasattr(offered[0], "mono")
    assert offered[0].descriptor.final_frames == 100
    assert offered[0].waveforms[0][0][5] == .05
    assert offered[0].waveforms[0][1][5] == np.float32(7969177 / 8388608)


def test_qt_started_delivery_logs_once_at_existing_slot(ui_qapp, tmp_path, monkeypatch, caplog):
    from dataclasses import replace
    from base.recording_service import RecordingCallbacks, RecordingSession
    from ui.recording_service_bridge import RecordingServiceBridge
    from unit_test.base.test_recording_service import request
    caplog.set_level(logging.INFO)
    clock = [10.0]
    monkeypatch.setattr('base.recording_service.perf_counter', lambda: clock[0])
    session = RecordingSession(SimpleNamespace(), request(tmp_path), RecordingCallbacks())
    session._startup_timing = replace(session._startup_timing, accepted_seconds=9.0)
    session._observe_capture_started()
    bridge = RecordingServiceBridge(SimpleNamespace())
    called = []
    bridge._callbacks['one'] = RecordingCallbacks(started=called.append)
    bridge._enqueue('started', session, None)
    assert not [r for r in records(caplog) if r['stage'] == 'qt_started']
    clock[0] += 2.2
    ui_qapp.processEvents()
    bridge._enqueue('started', session, None)
    ui_qapp.processEvents()
    assert called == [session]
    row, = [r for r in records(caplog, 'one') if r['stage'] == 'qt_started']
    assert float(row['seconds']) == pytest.approx(2.2)
