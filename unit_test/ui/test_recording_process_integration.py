"""Real Qt delivery and real spawn tests for the application recording boundary."""
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

from base.recording_process_protocol import RecordingRequest
from base.recording_service import RecordingCallbacks, RecordingService
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
        host._on_streaming_complete = mock.Mock()
        host.judge_play_and_record()
        fresh = host._recording_process_session
        pump(ui_qapp, lambda: fresh.state == "completed")
        assert fresh.generation != old_generation
        reader.release.set()
        pump(ui_qapp, session.released.is_set)
        assert reader.closed.is_set()
        assert not old_path.parent.exists()
        assert Path(fresh.request.path).exists()
        saved.assert_not_called()
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
    host._cache_condition_record.assert_called_once_with("first")


def test_main_host_relabel_destination_stays_in_test_root(ui_qapp, service, tmp_path):
    from base.file_ops import FileOps
    from consts import model_consts
    host = main_host(service, tmp_path)
    target = FileOps.resolve_wav_label_target(host.recorded_path, "OK",
        host.recorded_signal_info.get(model_consts.RECORDING_ROOT_CONFIG_KEY, ""))
    assert Path(target).is_relative_to(tmp_path)


def test_workspace_contract_failure_rejects_before_any_publication(ui_qapp, service, tmp_path):
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


def test_preview_replaces_cumulative_envelope_and_ignores_stale_final(ui_qapp, service, tmp_path):
    from base.recording_process_protocol import RecordingPreview
    from base.streaming_waveform_accumulator import StreamingWaveformSnapshot
    host = main_host(service, tmp_path)
    host._recording_process_id = "display"
    host._recording_process_final = False
    host._recording_process_preview_enabled = True
    host._recording_process_sequence = 0
    session = SimpleNamespace(request=SimpleNamespace(request_id="display", channels=(0, 2)), generation=3)

    def preview(sequence, values):
        waveforms = tuple(StreamingWaveformSnapshot(np.arange(len(values), dtype=np.float64),
            np.asarray(values, dtype=np.float32), len(values)) for _ in range(2))
        return RecordingPreview("display", 3, sequence, len(values), (0, 2), waveforms)

    host._on_process_recording_preview(session, preview(1, [1, 2]))
    host._on_process_recording_preview(session, preview(2, [1, 2, 9, 3]))
    np.testing.assert_array_equal(host.channel_workspace.calls[-1].amplitude, [1, 2, 9, 3])
    assert len(host.channel_workspace.calls) == 4
    host._recording_process_final = True
    host._on_process_recording_preview(session, preview(3, [8]))
    host._recording_process_final = False
    host._recording_process_id = "next"
    host._on_process_recording_preview(session, preview(4, [7]))
    assert len(host.channel_workspace.calls) == 4


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


def test_final_plot_failure_is_warning_and_still_saves_once(ui_qapp, service, tmp_path, monkeypatch):
    from ui.sequence import sequence_widget_streaming_ops as stream
    host = main_host(service, tmp_path)
    host._project_normalized_waveform_to_workspace = mock.Mock(side_effect=RuntimeError("Qt plot failed"))
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


def test_service_start_rejection_discards_registered_placeholder(ui_qapp, service, tmp_path):
    host = main_host(service, tmp_path)
    host.recording_bridge.start = mock.Mock(side_effect=RuntimeError("path still leased"))
    host._discard_current_recent_session = mock.Mock()
    host.judge_play_and_record()
    assert host.events == ["registered"]
    host._discard_current_recent_session.assert_called_once()
    assert host._recording_process_id is None


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


@pytest.mark.parametrize("outcome", ["failed", "write_failed", "cancelled", "admission_rejected", "db_failed", "analysis_failed"])
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
        host.run = mock.Mock(return_value=False)
    host.start_this_play()
    if outcome == "admission_rejected":
        assert host._recording_process_id is None
    else:
        session = host._recording_process_session
        if outcome == "cancelled":
            session.cancel()
        pump(ui_qapp, session.released.is_set)
        if outcome in ("db_failed", "analysis_failed"):
            pump(ui_qapp, lambda: not host._record_workflow_busy)
            save.assert_called_once()
        else:
            save.assert_not_called()
        if outcome == "write_failed":
            assert session.state == "failed"
            assert not Path(session.request.path).exists()
            host._handle_invalid_recording.assert_called_once()
            assert "disk write failure" in host._handle_invalid_recording.call_args.args[0]
            host.run.assert_not_called()
    sent.assert_not_called()


def test_tcp_finish_cannot_use_a_reentrant_new_requests_intent(ui_qapp, service, tmp_path, monkeypatch):
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
    sent.assert_not_called()
    host._on_process_recording_released(old)
    pump(ui_qapp, lambda: sessions[0].released.is_set() and not host._record_workflow_busy)
    sent.assert_called_once_with("192.0.2.5", 14000, "finish")
    assert save.call_count == 2


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
        host._discard_current_recent_session.assert_called_once()
        assert service.is_path_leased(path)
    finally:
        retained._cleanup_failed = False
        retained.release_error = None
        service._release(retained)


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
