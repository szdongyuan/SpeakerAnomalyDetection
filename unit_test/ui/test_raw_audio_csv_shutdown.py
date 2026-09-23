"""Window shutdown must preserve CSV resource ownership without blocking Qt."""
import multiprocessing
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PyQt5 import sip
from PyQt5.QtCore import QTimer, QObject, pyqtSignal
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QMainWindow, QWidget, QMessageBox

from base.raw_audio_csv_service import RawAudioCsvService
from consts.product_test_project_consts import EXPORT_RAW_AUDIO_CSV_KEY
from main_window import MainWindow
from ui.raw_audio_csv_service_bridge import RawAudioCsvServiceBridge
from ui.sequence.sequence_widget_raw_csv_ops import SequenceWidgetRawCsvOpsMixin
from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin
from ui.sequence.sequence_widget_recording_process_ops import SequenceWidgetRecordingProcessOpsMixin
from unit_test.base.raw_audio_csv_fakes import blocked_worker
from unit_test.base.test_raw_audio_csv_worker import make_command


def pump(app, predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.002)
    assert predicate()


class Sequence(SequenceWidgetRawCsvOpsMixin, SequenceWidgetStreamingOpsMixin,
               SequenceWidgetRecordingProcessOpsMixin, QWidget):
    def __init__(self, bridge, *, owned=False):
        QWidget.__init__(self)
        self._initialize_raw_audio_csv_runtime(raw_audio_csv_bridge=bridge)
        self._owns_raw_audio_csv_service = owned
        self.product_test_project_context = {EXPORT_RAW_AUDIO_CSV_KEY: True}
        self.default_logger = Mock()
        self.hw_manager = SimpleNamespace(stop=Mock())
        self._shutdown_product_pdf_exporter = Mock()
        self._cleanup_streaming_resources = Mock()
        self._on_raw_audio_csv_export_failed = Mock()
        self._on_raw_audio_csv_export_succeeded = Mock()
        self._analysis_has_pending_tasks = Mock(return_value=False)


class Window(MainWindow):
    def __init__(self, sequence, bridge):
        QMainWindow.__init__(self)
        self.sequence_window = sequence
        self.raw_audio_csv_bridge = bridge
        self.raw_audio_csv_service = bridge.service
        self._owns_raw_audio_csv_service = False  # launcher-created application service
        self._close_all_subwindows = Mock()
        self.default_logger = Mock()


@pytest.fixture(params=['zip_write', 'zip_verify'])
def runtime(ui_qapp, request):
    from unit_test.base.raw_audio_csv_fakes import zip_phase_worker
    entered = multiprocessing.get_context('spawn').Event()
    gate = multiprocessing.get_context('spawn').Event()
    service = RawAudioCsvService(worker_target=zip_phase_worker, worker_args=(request.param, entered, gate))
    service.test_zip_entered = entered
    bridge = RawAudioCsvServiceBridge(service)
    yield service, bridge, gate
    gate.set()
    for token in tuple(service._ledger._reservations.values()):
        service.release_reservation(token)
    service.begin_shutdown()
    pump(ui_qapp, service.closed.is_set)
    bridge.close_delivery()


def submit(sequence, tmp_path, name='one'):
    request = make_command(tmp_path, name).request
    token = sequence.raw_audio_csv_service.reserve(request.recording_id).reservation
    sequence._owned_raw_audio_csv_tasks.add(name)
    assert sequence.raw_audio_csv_service.commit(token, request) == 'accepted'
    return request


@pytest.mark.parametrize('boundary', ['main', 'borrowed', 'owned'])
def test_blocked_export_close_keeps_qt_alive_and_drains(ui_qapp, runtime, tmp_path, boundary):
    service, bridge, gate = runtime
    sequence = Sequence(bridge, owned=boundary == 'owned')
    window = Window(sequence, bridge) if boundary == 'main' else sequence
    window.setWindowTitle('Capture')
    window.show()
    submit(sequence, tmp_path)
    pump(ui_qapp, service.test_zip_entered.is_set)
    pid = service._process.pid
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(5)
    try:
        for _ in range(3):
            event = QCloseEvent()
            window.closeEvent(event)
            assert not event.isAccepted()
        caption = window.statusBar().currentMessage() if boundary == 'main' else window.windowTitle()
        assert '保存/压缩原始 CSV' in caption
        pump(ui_qapp, lambda: len(ticks) >= 5)
        assert window.isVisible()
        assert service._process.pid == pid
        assert service.snapshot().outstanding == 1
        sequence._shutdown_product_pdf_exporter.assert_not_called()
        gate.set()
        pump(ui_qapp, lambda: not window.isVisible())
        assert not sequence._owned_raw_audio_csv_tasks
        sequence._shutdown_product_pdf_exporter.assert_called_once()
        if boundary == 'borrowed':
            assert service.snapshot().phase == 'open'
            sequence.show()
            assert sequence._can_prepare_recording_hardware()
            assert sequence.windowTitle() == 'Capture'
        else:
            pump(ui_qapp, service.closed.is_set)
            assert not service._thread.is_alive()
            assert service._control is None
            assert service._process is None
            assert pid not in [child.pid for child in multiprocessing.active_children()]
    finally:
        timer.stop()
        window.hide()
        sip.delete(window)
        if window is not sequence:
            sip.delete(sequence)


def test_hidden_initialization_close_preserves_pending_admission_and_hardware(ui_qapp, runtime):
    service, bridge, _ = runtime
    sequence = Sequence(bridge, owned=True)
    admission = sequence._reserve_raw_audio_csv_recording()
    sequence.close()
    assert service.snapshot().phase == 'open'
    assert sequence._pending_raw_audio_csv_recording is admission
    sequence._cleanup_streaming_resources.assert_not_called()
    sequence.hw_manager.stop.assert_not_called()
    assert sequence._can_prepare_recording_hardware()
    sequence._release_raw_audio_csv_recording(admission)
    sip.delete(sequence)


class Video(QObject):
    closed = pyqtSignal()
    is_shutdown_complete = False
    def __init__(self):
        super().__init__()
        self.shutdown = Mock()


@pytest.mark.parametrize('enabled', [True, False])
def test_main_closes_admission_before_video_wait_and_resolves_pending(ui_qapp, runtime, enabled):
    service, bridge, _ = runtime
    sequence = Sequence(bridge)
    sequence.product_test_project_context[EXPORT_RAW_AUDIO_CSV_KEY] = enabled
    sequence._reserve_raw_audio_csv_recording()
    window = Window(sequence, bridge)
    window.video_controller = Video()
    window.show()
    try:
        window.close()
        window.close()
        assert not sequence._can_prepare_recording_hardware()
        assert sequence._pending_raw_audio_csv_recording is None
        assert sequence._reserve_raw_audio_csv_recording() is None
        assert service.snapshot().reserved == 0
        window.video_controller.shutdown.assert_called_once()
        window.video_controller.is_shutdown_complete = True
        window.video_controller.closed.emit()
        pump(ui_qapp, lambda: not window.isVisible())
    finally:
        window.hide()
        sip.delete(window)
        sip.delete(sequence)


def test_analysis_block_does_not_begin_shutdown(ui_qapp, runtime, monkeypatch):
    service, bridge, _ = runtime
    sequence = Sequence(bridge)
    sequence._analysis_has_pending_tasks.return_value = True
    monkeypatch.setattr(QMessageBox, 'information', Mock())
    window = Window(sequence, bridge)
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted()
    assert service.snapshot().phase == 'open'
    assert sequence._can_prepare_recording_hardware()
    sip.delete(window)
    sip.delete(sequence)


def test_late_analysis_during_video_wait_still_blocks_exit(ui_qapp, runtime, monkeypatch):
    service, bridge, _ = runtime
    sequence = Sequence(bridge)
    window = Window(sequence, bridge)
    window.video_controller = Video()
    monkeypatch.setattr(QMessageBox, 'information', Mock())
    window.show()
    try:
        window.close()
        sequence._analysis_has_pending_tasks.return_value = True
        window.video_controller.is_shutdown_complete = True
        window.video_controller.closed.emit()
        pump(ui_qapp, service.closed.is_set)
        ui_qapp.processEvents()
        assert window.isVisible()
        assert window.isEnabled()  # operator can retry close after analysis ends
        sequence._shutdown_product_pdf_exporter.assert_not_called()
        sequence._analysis_has_pending_tasks.return_value = False
        window.close()
        pump(ui_qapp, lambda: not window.isVisible())
    finally:
        sip.delete(window)
        sip.delete(sequence)


def test_main_nested_sequence_cleanup_occurs_once_after_drain(ui_qapp, runtime, tmp_path):
    service, bridge, gate = runtime
    sequence = Sequence(bridge)
    sequence.show()
    window = Window(sequence, bridge)
    window._close_all_subwindows.side_effect = sequence.close
    window.show()
    submit(sequence, tmp_path)
    try:
        window.close()
        assert sequence.isVisible()
        sequence.close()
        assert sequence.isVisible()
        sequence.hw_manager.stop.assert_not_called()
        gate.set()
        pump(ui_qapp, lambda: not window.isVisible())
        assert not sequence.isVisible()
        sequence.hw_manager.stop.assert_called_once()
        sequence._cleanup_streaming_resources.assert_called_once()
        sequence._shutdown_product_pdf_exporter.assert_called_once()
    finally:
        sip.delete(window)
        sip.delete(sequence)


def test_main_exit_cleans_hidden_sequence_after_drain(ui_qapp, runtime):
    service, bridge, _ = runtime
    sequence = Sequence(bridge)
    window = Window(sequence, bridge)
    window.show()
    try:
        window.close()
        pump(ui_qapp, lambda: not window.isVisible())
        sequence.hw_manager.stop.assert_called_once()
        sequence._cleanup_streaming_resources.assert_called_once()
    finally:
        sip.delete(window)
        sip.delete(sequence)


def test_reopened_borrowed_sequence_is_cleaned_again_on_application_exit(ui_qapp, runtime):
    _, bridge, _ = runtime
    sequence = Sequence(bridge)
    sequence.show()
    sequence.close()
    sequence.hw_manager.stop.assert_called_once()
    sequence.show()
    window = Window(sequence, bridge)
    window.show()
    try:
        window.close()
        pump(ui_qapp, lambda: not window.isVisible())
        assert sequence.hw_manager.stop.call_count == 2
    finally:
        sip.delete(window)
        sip.delete(sequence)


def test_standalone_owned_window_shuts_recording_service_before_final_close(ui_qapp, runtime):
    service, bridge, _ = runtime
    sequence = Sequence(bridge, owned=True)
    callbacks = []
    sequence._owns_recording_bridge = True
    sequence.recording_bridge = SimpleNamespace(shutdown=Mock(side_effect=callbacks.append),
        service=SimpleNamespace(closed=SimpleNamespace(is_set=lambda: True)))
    sequence.show()
    try:
        sequence.close()
        assert len(callbacks) == 1
        pump(ui_qapp, service.closed.is_set)
        assert sequence.isVisible()
        sequence.close()
        assert len(callbacks) == 1
        callbacks[0]()
        pump(ui_qapp, lambda: not sequence.isVisible())
    finally:
        sip.delete(sequence)


@pytest.mark.parametrize('late_commit', [False, True])
def test_recording_reservation_resolves_during_main_drain(ui_qapp, runtime, tmp_path, late_commit):
    service, bridge, gate = runtime
    sequence = Sequence(bridge)
    request = make_command(tmp_path, 'late').request
    token = service.reserve(request.recording_id).reservation
    window = Window(sequence, bridge)
    window.show()
    try:
        window.close()
        assert window.isVisible()
        assert service.snapshot().phase == 'draining'
        assert service.reserve('new').status == 'closing'
        if late_commit:
            sequence._owned_raw_audio_csv_tasks.add(request.task_id)
            assert service.commit(token, request) == 'accepted'
        else:
            assert service.release_reservation(token)
        gate.set()
        pump(ui_qapp, lambda: not window.isVisible())
        assert not sequence._owned_raw_audio_csv_tasks
    finally:
        sip.delete(window)
        sip.delete(sequence)


def test_borrowed_close_waits_for_terminal_resource_release_not_other_work(ui_qapp, runtime, tmp_path, monkeypatch):
    service, bridge, gate = runtime
    sequence = Sequence(bridge)
    request = submit(sequence, tmp_path)
    # Hold the supervisor at the release boundary AFTER its terminal event.
    import threading
    release_gate = threading.Event()
    original_release = service._ledger.release_task
    def held_release(*args, **kwargs):
        release_gate.wait(10)
        return original_release(*args, **kwargs)
    monkeypatch.setattr(service._ledger, 'release_task', held_release)
    sequence.show()
    gate.set()
    try:
        pump(ui_qapp, lambda: sequence._on_raw_audio_csv_export_succeeded.called)
        assert request.task_id in sequence._owned_raw_audio_csv_tasks
        sequence.close()
        assert sequence.isVisible()
        unrelated = service.reserve('another-window').reservation
        release_gate.set()
        pump(ui_qapp, lambda: not sequence.isVisible())
        assert service.snapshot().reserved == 1
        service.release_reservation(unrelated)
    finally:
        release_gate.set()
        sip.delete(sequence)


def test_borrowed_close_waits_for_retained_recording_path(ui_qapp, runtime, tmp_path):
    service, bridge, _ = runtime
    sequence = Sequence(bridge)
    path = str(tmp_path / 'held.wav')
    permit = service.try_acquire_mutation((path,))
    sequence._raw_audio_csv_recording_paths.add(path)
    sequence.show()
    try:
        sequence.close()
        assert sequence.isVisible()
        service.release_mutation(permit)
        pump(ui_qapp, lambda: not sequence.isVisible())
        assert service.snapshot().phase == 'open'
    finally:
        service.release_mutation(permit)
        sip.delete(sequence)


def test_export_failure_still_closes_after_resource_release(ui_qapp, tmp_path):
    service = RawAudioCsvService()
    bridge = RawAudioCsvServiceBridge(service)
    sequence = Sequence(bridge, owned=True)
    request = make_command(tmp_path, 'missing').request
    from pathlib import Path
    Path(request.wav_path).unlink()
    token = service.reserve(request.recording_id).reservation
    sequence._owned_raw_audio_csv_tasks.add(request.task_id)
    service.commit(token, request)
    sequence.show()
    try:
        sequence.close()
        pump(ui_qapp, lambda: not sequence.isVisible())
        sequence._on_raw_audio_csv_export_failed.assert_called_once()
        assert not sequence._owned_raw_audio_csv_tasks
        assert bridge.service_closed
        assert not service._thread.is_alive()
        assert service._process is service._control is None
    finally:
        service.begin_shutdown()
        pump(ui_qapp, service.closed.is_set)
        sip.delete(sequence)
        bridge.close_delivery()


def test_incomplete_recording_shutdown_does_not_fabricate_csv_release(ui_qapp, runtime, tmp_path, monkeypatch):
    import threading
    service, bridge, _ = runtime
    sequence = Sequence(bridge)
    path = str(tmp_path / 'unreleased.wav')
    permit = service.try_acquire_mutation((path,))
    window = Window(sequence, bridge)
    callbacks = []
    recording_closed = threading.Event()
    window.recording_bridge = SimpleNamespace(
        service=SimpleNamespace(closed=recording_closed, diagnostics=['reader still owns WAV']),
        shutdown=callbacks.append)
    monkeypatch.setattr(QMessageBox, 'warning', Mock())
    window.show()
    try:
        window.close()
        callbacks[0]()
        assert window.isVisible()
        assert not recording_closed.is_set()
        assert not bridge.service_closed
        assert service.paths_busy((path,))
        sequence._shutdown_product_pdf_exporter.assert_not_called()
        service.release_mutation(permit)
        pump(ui_qapp, lambda: not window.isVisible())
    finally:
        service.release_mutation(permit)
        sip.delete(window)
        sip.delete(sequence)


@pytest.mark.parametrize('enabled', [True, False])
def test_reentrant_close_during_startup_prevents_hardware_start(ui_qapp, runtime, tmp_path, monkeypatch, enabled):
    from unit_test.ui.test_recording_process_integration import controls_host
    from base.recording_service import RecordingService
    service, bridge, _ = runtime
    recording = RecordingService()
    host = controls_host(recording, tmp_path)
    # Use the real startup adapter and admission scope on the established host.
    for name in ('_reserve_raw_audio_csv_recording', '_release_raw_audio_csv_recording',
                 '_begin_raw_audio_csv_close'):
        setattr(host, name, getattr(SequenceWidgetRawCsvOpsMixin, name).__get__(host))
    host.raw_audio_csv_service = service
    host.product_test_project_context = {EXPORT_RAW_AUDIO_CSV_KEY: enabled}
    host._start_process_recording = Mock()
    monkeypatch.setattr('PyQt5.QtWidgets.QApplication.processEvents',
                        lambda: host._begin_raw_audio_csv_close(application_exit=True))
    try:
        host.judge_play_and_record()
        host._start_process_recording.assert_not_called()
        assert host._pending_raw_audio_csv_recording is None
        assert service.snapshot().reserved == 0
        assert not host.player_status_flag
    finally:
        recording.shutdown()
        assert recording.closed.wait(5)


@pytest.mark.parametrize('enabled', [True, False])
@pytest.mark.parametrize('retained_lock', [None, 'product_round', 'cycle'])
def test_borrowed_startup_close_restores_controls_without_draining_triggers(
        ui_qapp, runtime, enabled, retained_lock):
    from PyQt5.QtWidgets import QLineEdit, QPushButton
    from ui.sequence.sequence_widget_barcode_ops import SequenceWidgetBarcodeOpsMixin
    from ui.sequence.sequence_widget_ui_ops import SequenceWidgetUiOpsMixin
    from unit_test.ui.test_streaming_event_dispatch import _WorkflowHost

    class StartupWindow(SequenceWidgetBarcodeOpsMixin, _WorkflowHost,
                        SequenceWidgetRawCsvOpsMixin, QWidget):
        def __init__(self, bridge):
            QWidget.__init__(self)
            _WorkflowHost.__init__(self, {}, [])
            self._initialize_raw_audio_csv_runtime(raw_audio_csv_bridge=bridge)
            self.product_test_project_context = {EXPORT_RAW_AUDIO_CSV_KEY: enabled}
            self.sequence_config = [{'seq1': {}}]
            self.count_board = SimpleNamespace(mode='mark')
            self.lineedit_s_or_n = QLineEdit('sample-1', self)
            self.lineedit_s_or_n.setToolTip('scan barcode')
            self.player_btn = QPushButton(self)
            self.replayer_btn = QPushButton(self)
            self.data_btn = QPushButton(self)
            self._start_process_recording = Mock()
            self._drain_queued_directional_trigger = Mock()
            self._on_serial_product_runtime_error = Mock()
            self._queued_directional_trigger = 'forward'
            self._cleanup_streaming_resources = (
                SequenceWidgetStreamingOpsMixin._cleanup_streaming_resources.__get__(self))
            self.update_player_btn_is_playing = (
                SequenceWidgetUiOpsMixin.update_player_btn_is_playing.__get__(self))
            self.update_player_btn_is_paused = Mock(wraps=(
                SequenceWidgetUiOpsMixin.update_player_btn_is_paused.__get__(self)))

    service, bridge, _ = runtime
    window = StartupWindow(bridge)
    window.show()
    closed = []

    def close_during_startup():
        assert window.lineedit_s_or_n.isReadOnly()
        assert not window.player_btn.isEnabled()
        assert not window.replayer_btn.isEnabled()
        assert not window.data_btn.isEnabled()
        # A longer-lived lock can be acquired during the admitted workflow.
        if retained_lock == 'product_round':
            window._lock_sn_for_product_round()
        elif retained_lock == 'cycle':
            window._lock_sn_for_cycle()
        window.close()
        closed.append(not window.isVisible())
        window.show()

    QTimer.singleShot(0, close_during_startup)
    try:
        window.judge_play_and_record()
        assert closed == [True]
        assert window.isVisible()
        assert not window.player_status_flag
        assert not window._record_workflow_busy
        assert not window._sn_locked_for_recording
        assert window.lineedit_s_or_n.isReadOnly() is (retained_lock is not None)
        if retained_lock is None:
            assert window.lineedit_s_or_n.toolTip() == 'scan barcode'
        else:
            assert getattr(window, '_sn_locked_for_' + retained_lock)
        assert window.player_btn.isEnabled()
        assert window.replayer_btn.isEnabled()
        assert window.data_btn.isEnabled()
        window.update_player_btn_is_paused.assert_called()
        window._start_process_recording.assert_not_called()
        window._drain_queued_directional_trigger.assert_not_called()
        window._on_serial_product_runtime_error.assert_not_called()
        assert window._queued_directional_trigger == 'forward'
        assert service.snapshot().reserved == 0
        assert service.snapshot().phase == 'open'
    finally:
        sip.delete(window)
