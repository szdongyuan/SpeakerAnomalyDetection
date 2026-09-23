import threading
import time
from dataclasses import replace
from unittest.mock import Mock

from PyQt5.QtCore import QCoreApplication, QEvent, QThread
from PyQt5.QtWidgets import QWidget

from base.raw_audio_csv_protocol import (
    CsvExportRequest, CsvFailure, CsvLedgerSnapshot, CsvServiceEvent, CsvTaskSnapshot,
)


class EventService:
    def __init__(self):
        self.callbacks = []
        self.closed = threading.Event()
        self.begin_shutdown = Mock()

    def subscribe(self, callback):
        self.callbacks.append(callback)
        return Mock(unsubscribe=lambda: self.callbacks.remove(callback))

    def emit(self, event):
        for callback in tuple(self.callbacks):
            callback(event)


def event(kind="state"):
    return CsvServiceEvent(kind, CsvLedgerSnapshot("open", 16, 0, 0, 0))


def terminal(generation=1):
    request = CsvExportRequest("task", "recording", "old.wav", "old.csv", (1,), "old-round", "old-record")
    return replace(event("terminal"),
        task=CsvTaskSnapshot(request, "failed", generation, "failed"),
        result=CsvFailure("task", generation, "export", "OSError", "disk full"))


def test_background_event_is_queued_to_application_thread(ui_qapp):
    from ui.raw_audio_csv_service_bridge import RawAudioCsvServiceBridge
    service = EventService()
    bridge = RawAudioCsvServiceBridge(service)
    received = []
    bridge.subscribe(lambda value: received.append((value, QThread.currentThread())))
    worker = threading.Thread(target=service.emit, args=(event(),))
    worker.start()
    worker.join()
    assert not received
    ui_qapp.processEvents()
    assert received == [(event(), ui_qapp.thread())]
    bridge.close_delivery()


def test_unsubscribe_and_destroy_drop_already_queued_events(ui_qapp):
    from ui.raw_audio_csv_service_bridge import RawAudioCsvServiceBridge
    service = EventService()
    bridge = RawAudioCsvServiceBridge(service)
    received = []
    subscription = bridge.subscribe(received.append)
    service.emit(event())
    subscription.unsubscribe()
    owner = QWidget()
    bridge.subscribe(received.append, owner=owner)
    service.emit(event())
    owner.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    ui_qapp.processEvents()
    assert received == []
    # A supervisor can already hold a copied callback when unsubscribe runs.
    callback = service.callbacks[0]
    bridge.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert not service.callbacks
    worker = threading.Thread(target=callback, args=(event(),))
    worker.start()
    worker.join()
    ui_qapp.processEvents()
    assert received == []


def test_terminal_deduplication_and_consumer_failure_are_isolated(ui_qapp, caplog):
    from ui.raw_audio_csv_service_bridge import RawAudioCsvServiceBridge
    service = EventService()
    bridge = RawAudioCsvServiceBridge(service)
    received = []
    def broken(value):
        raise ValueError("consumer broke")
    bridge.subscribe(broken)
    bridge.subscribe(received.append)
    for value in (terminal(2), terminal(2), terminal(1)):
        service.emit(value)
    ui_qapp.processEvents()
    assert received == [terminal(2)]
    assert "consumer broke" in caplog.text
    bridge.close_delivery()


def test_delivery_closed_does_not_shutdown_service_and_closed_event_is_authoritative(ui_qapp):
    from ui.raw_audio_csv_service_bridge import RawAudioCsvServiceBridge
    service = EventService()
    bridge = RawAudioCsvServiceBridge(service)
    closed = []
    bridge.closed.connect(lambda: closed.append(bridge.service_closed))
    service.emit(event("closed"))
    ui_qapp.processEvents()
    assert closed == [True]
    assert not service.closed.is_set()
    assert not bridge.delivery_closed
    bridge.close_delivery()
    assert bridge.delivery_closed
    service.begin_shutdown.assert_not_called()


def test_idle_real_service_starts_no_process_and_releases_without_gui_delivery(ui_qapp):
    from base.raw_audio_csv_service import RawAudioCsvService
    from ui.raw_audio_csv_service_bridge import RawAudioCsvServiceBridge
    service = RawAudioCsvService()
    bridge = RawAudioCsvServiceBridge(service)
    try:
        token = service.reserve("recording").reservation
        assert service._process is None
        bridge.close_delivery()
        assert service.release_reservation(token)
        assert service.snapshot().outstanding == 0
    finally:
        service.begin_shutdown()
        assert service.closed.wait(3)


def test_real_supervisor_state_is_delivered_on_gui_thread(ui_qapp):
    from base.raw_audio_csv_service import RawAudioCsvService
    from ui.raw_audio_csv_service_bridge import RawAudioCsvServiceBridge
    service = RawAudioCsvService()
    bridge = RawAudioCsvServiceBridge(service)
    received = []
    bridge.subscribe(lambda value: received.append((value, QThread.currentThread())))
    token = service.reserve("recording").reservation
    try:
        deadline = time.monotonic() + 3
        while not any(value.snapshot.reserved == 1 for value, _ in received):
            assert time.monotonic() < deadline
            ui_qapp.processEvents()
            threading.Event().wait(.005)
        assert all(thread is ui_qapp.thread() for _, thread in received)
    finally:
        bridge.close_delivery()
        service.release_reservation(token)
        service.begin_shutdown()
        assert service.closed.wait(3)


def test_result_presentation_uses_original_task_once(ui_qapp):
    from ui.raw_audio_csv_service_bridge import RawAudioCsvServiceBridge
    from ui.sequence.sequence_widget_raw_csv_ops import SequenceWidgetRawCsvOpsMixin
    class Window(SequenceWidgetRawCsvOpsMixin, QWidget):
        def __init__(self, bridge):
            super().__init__()
            self._on_raw_audio_csv_export_failed = Mock()
            self._initialize_raw_audio_csv_runtime(raw_audio_csv_bridge=bridge)
    service = EventService()
    bridge = RawAudioCsvServiceBridge(service)
    window = Window(bridge)
    window._owned_raw_audio_csv_tasks.add("task")
    service.emit(terminal())
    service.emit(terminal())
    ui_qapp.processEvents()
    window._on_raw_audio_csv_export_failed.assert_called_once_with("old.wav", "disk full")
    window.close()
    assert not window._owns_raw_audio_csv_service
    service.begin_shutdown.assert_not_called()
    bridge.close_delivery()


def test_correlated_parent_timing_survives_disabled_logging_and_delayed_qt(tmp_path, ui_qapp):
    import logging
    import os
    import numpy as np
    import soundfile as sf
    import pytest
    from base.raw_audio_csv_service import RawAudioCsvService
    from ui.raw_audio_csv_service_bridge import RawAudioCsvServiceBridge
    source = tmp_path / 'source.wav'
    sf.write(source, np.ones((32, 2), dtype=np.float32), 8000, subtype='FLOAT')
    service = RawAudioCsvService()
    bridge = RawAudioCsvServiceBridge(service)
    events, delivered, threads = [], [], []
    service.subscribe(lambda event: (events.append(event), threads.append(threading.current_thread().name)))
    bridge.subscribe(delivered.append)
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        request = CsvExportRequest('task', 'recording', str(source), str(tmp_path / 'out.csv'), (0, 2), 'g', 'r')
        token = service.reserve('recording').reservation
        assert service.commit(token, request) == 'accepted'
        service.begin_shutdown()
        assert service.closed.wait(15)

        time.sleep(.05)  # Deliberately leave Qt delivery queued after resources close.
        ui_qapp.processEvents()
        stages = {e.timing.stage: e for e in events if e.timing is not None}
        assert {'reserve', 'submit', 'dispatch', 'ready', 'terminal_received', 'released'} <= stages.keys()
        assert stages['reserve'].timing.recording_id == 'recording'
        assert stages['submit'].timing.token_id == token.token_id
        for stage in ('dispatch', 'terminal_received', 'released'):
            timing = stages[stage].timing
            assert timing.task_id == 'task'
            assert timing.generation == 1
            assert timing.worker_pid != os.getpid()
            assert timing.worker_pid > 0
        terminal = next(e for e in delivered if e.kind == 'terminal')
        assert terminal.qt_delivery_seconds >= .05
        assert terminal.result.elapsed_seconds == stages['terminal_received'].result.elapsed_seconds
        assert terminal.result.export_end_seconds - terminal.result.export_begin_seconds == pytest.approx(terminal.result.csv_export_seconds)
        phases = [terminal.result.csv_export_seconds, terminal.result.zip_write_seconds,
                  terminal.result.zip_verify_seconds, terminal.result.zip_publish_seconds,
                  terminal.result.csv_cleanup_seconds]
        assert all(value >= 0 for value in phases)
        assert terminal.result.elapsed_seconds >= sum(phases)
        assert service.snapshot().outstanding == 0
        assert all(name in ('raw-csv-supervisor', 'raw-csv-completion') for name in threads)
    finally:
        logging.disable(previous)
        bridge.close_delivery()
        service.begin_shutdown()
        assert service.closed.wait(15)
