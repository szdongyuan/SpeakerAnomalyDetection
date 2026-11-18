"""Exercise both application entry paths without a native recording device."""

import json
import logging
import sys
from dataclasses import replace
from datetime import date, datetime, timedelta
from threading import Event
from time import monotonic
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PyQt5 import QtWidgets, sip
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtTest import QTest

import main_window
from base import hardware_selection, recording_service
from base.db_manager import DataSave
from base.motor_daily_wav_records import DailyWavRecord
from base.record_analysis_reference import RecordAnalysisReference
from base.recording_process_protocol import FrozenConfig, SoundcardPrepareResult
from base.recording_service import RecordingWorkerStatus
from base.workbench_analysis_config import AnalysisTestQueue
from base.workbench_config_values import WorkbenchChannel, WorkbenchHardwareChannel
from base.workbench_recording_protocol import StationWavChannel
from base.workbench_summary_store import WorkbenchSummarySnapshot, WorkbenchSummaryStore
from base.workbench_wav_details_store import (
    WavDetailsRow, WorkbenchWavDetailsSnapshot, WorkbenchWavDetailsStore,
)
from consts import error_code
from main_window_Launcher import MainWindowLauncher
from ui import recording_service_bridge, vkinging_hardware_controls
from ui.sequence import motor_workbench_controller
from ui.sequence.motor_information_summary_panel import MotorInformationSummaryPanel
from ui.sequence.record_analysis_dialog import RecordAnalysisDialog
from ui.video_controller import VideoController
from unit_test.base.vkinging_fakes import device_info, input_config
from unit_test.ui.test_motor_workbench_controller import (
    config, configuration_with, paired_recording_workbench, recording_workbench, workbench,
)


class RecordingBridge(QObject):
    worker_status_changed = pyqtSignal(object)
    shutting_down = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.service = SimpleNamespace(closed=Event(), can_start_recording=True)
        self.worker_status = RecordingWorkerStatus(0, "idle")
        self.started = False
        self.hardware_busy = False
        self.probes = []
        self.prewarms = []
        self.retained_ve_signature = None
        self.shutdown_calls = 0

    def ensure_resident(self):
        self.started = True
        return "accepted"

    def prepare_soundcard(self, request, callback):
        assert self.started
        self.probes.append((request, callback))
        return "accepted"

    def prewarm_ve(self, request, callback):
        assert self.started
        self.prewarms.append((request, callback))
        return "accepted"

    def release_ve(self, signature, callback):
        if signature != self.retained_ve_signature:
            self.retained_ve_signature = None
        callback("released", ())
        return "released"

    def change_worker(self, generation, state="ready"):
        self.worker_status = RecordingWorkerStatus(generation, state)
        self.worker_status_changed.emit(self.worker_status)

    def answer_probe(self, index=-1, *, name="Default", generation=1):
        request, callback = self.probes[index]
        if name is None:
            callback(SoundcardPrepareResult(request.probe_id, generation, detail="input identity is missing"))
            return
        device = FrozenConfig.snapshot({"backend": "sounddevice", "index": 7, "name": name,
                                        "hostapi": 0, "hostapi_name": "WASAPI", "max_input_channels": 3})
        callback(SoundcardPrepareResult(request.probe_id, generation, device, (0, 1, 2), 48000))

    def finish_prewarm(self, *, success=True):
        request, callback = self.prewarms[-1]
        if success:
            self.retained_ve_signature = request.signature
        callback(SimpleNamespace(warmup_id=request.warmup_id, signature=request.signature,
                                 success=success, stage="completed" if success else "start_task",
                                 code=None, detail="", diagnostics=(), ownership_safe=True))

    def shutdown(self, callback=None):
        if not self.service.closed.is_set():
            self.shutdown_calls += 1
            self.shutting_down.emit()
            self.service.closed.set()
        if callback is not None:
            callback()


class Discovery:
    def __init__(self, *, on_result):
        self._on_result = on_result
        self.starts = 0
        self.generation = 0

    def start(self):
        self.starts += 1
        self.generation += 1
        return self.generation

    refresh = start

    def cancel(self):
        self.generation += 1

    def close(self):
        pass

    def answer(self, *devices, generation=None):
        self._on_result(SimpleNamespace(
            generation=self.generation if generation is None else generation,
            handles_released=True, result=SimpleNamespace(devices=devices, diagnostics=())))


class AnalysisServiceStub:
    """Keep hardware/startup tests independent of resident analysis processes."""

    def __init__(self):
        self.sync_calls = []
        self.shutdown_calls = 0
        self.poll_calls = 0
        self.closed = False
        self.allow_close = True
        self.events = []
        self.station_states = {}
        self.pending_task_count = 0
        self.submissions = []

    def submit_batch(self, payload):
        self.submissions.append(payload)
        return {"accepted": True, "accepted_task_ids": [t["task_id"] for t in payload["tasks"]], "issues": []}

    def sync_stations(self, station_ids):
        self.sync_calls.append(tuple(station_ids))
        self.station_states = {sid: {"state": "ready", "pid": index + 1}
                               for index, sid in enumerate(station_ids)}
        return {"accepted": True, "error": None}

    def shutdown(self):
        self.shutdown_calls += 1

    def poll_events(self):
        self.poll_calls += 1
        events, self.events = self.events, []
        if self.shutdown_calls and self.allow_close and not self.closed:
            self.closed = True
            events.append({"event_type": "service_closed", "payload": {}})
        return tuple(events)


@pytest.fixture
def startup(ui_qapp, monkeypatch, tmp_path):
    path = tmp_path / "hardware.json"
    database_path = tmp_path / "audio_data.db"
    database = DataSave(str(database_path))
    try:
        code, detail = database.create_table()
        assert code == error_code.OK, detail
    finally:
        database.close()
    monkeypatch.setattr(main_window, "DATABASE_PATH", str(database_path))
    bridge = RecordingBridge()
    mic = {"name": "Default", "index": 0, "hostapi": 0, "hostapi_name": "WASAPI", "max_input_channels": 3}
    manager = hardware_selection.SoundDeviceManager
    monkeypatch.setattr(ui_qapp, "_ve_prewarm_lifetime", None, raising=False)
    monkeypatch.setattr(ui_qapp, "_recording_bridge", None, raising=False)
    monkeypatch.setattr(hardware_selection, "_HARDWARE_SELECTION_PATH", str(path))
    monkeypatch.setattr(manager, "refresh_available_device", lambda: None)
    monkeypatch.setattr(manager, "get_device_info", lambda: {"WASAPI": {"input": [mic]}})
    monkeypatch.setattr(manager, "get_api_info", lambda *_: {"name": "WASAPI"})
    monkeypatch.setattr(manager, "change_default_input_device", lambda *_: None)
    monkeypatch.setattr(hardware_selection, "_os_default_device", lambda kind: mic if kind == "mic" else None)
    monkeypatch.setattr(vkinging_hardware_controls, "DiscoveryService", Discovery)
    monkeypatch.setattr(VideoController, "start_preview", lambda self: None)
    monkeypatch.setattr(main_window.MainWindow, "get_current_version", lambda self: "test")
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", Mock())
    from ui.sequence.workbench_input_ops import LoadUiConfig
    from base.unified_hid_device_manager import UnifiedHardwareManager
    monkeypatch.setattr(LoadUiConfig, "load_last_recorded_info", staticmethod(lambda _: {}))
    monkeypatch.setattr(UnifiedHardwareManager, "start_scanner_listeners", lambda _: True)
    monkeypatch.setattr(UnifiedHardwareManager, "stop_scanner_listeners", lambda _: None)
    windows = []
    state = SimpleNamespace(analysis_service=AnalysisServiceStub(), launch_options={})

    class Window(main_window.MainWindow):
        def __init__(self, **kwargs):
            windows.append(self)
            kwargs.setdefault("analysis_service", state.launch_options.get("analysis_service", state.analysis_service))
            super().__init__(**kwargs)

    monkeypatch.setattr(main_window, "MainWindow", Window)

    def login(execute):
        monkeypatch.setattr(main_window, "LoginWindow", lambda: SimpleNamespace(on_exec=lambda: execute(windows[-1])))

    def launch(execute, *, launcher=False, recording_bridge=bridge, **options):
        state.launch_options = options
        login(execute)
        if not launcher:
            return Window(recording_bridge=recording_bridge, **options)
        monkeypatch.setattr(QtWidgets, "QApplication", lambda _args: ui_qapp)
        monkeypatch.setattr(recording_service, "RecordingService", lambda: recording_bridge.service)
        monkeypatch.setattr(recording_service_bridge, "RecordingServiceBridge", lambda *_: recording_bridge)

        def fail_launch(_parent, _title, detail):
            pytest.fail(detail)

        monkeypatch.setattr(QtWidgets.QMessageBox, "critical", fail_launch)
        application = MainWindowLauncher()
        assert recording_bridge.started
        assert recording_bridge.probes == []
        application.on_loader_finished()
        assert application.window.recording_bridge is recording_bridge
        return application.window

    state.launch, state.login, state.path, state.bridge, state.mic = launch, login, path, bridge, mic
    state.database_path = database_path
    yield state
    for window in windows:
        if not sip.isdeleted(window):
            analysis = getattr(window, "analysis_service", None)
            if isinstance(analysis, AnalysisServiceStub):
                analysis.allow_close = True
            window.close()
            deadline = monotonic() + 40
            while not getattr(window, "_exit_cleanup_complete", False) and monotonic() < deadline:
                QTest.qWait(10)
            labels = getattr(window, "audio_label_sync_bridge", None)
            assert getattr(window, "_exit_cleanup_complete", False), (
                "startup window did not finish asynchronous shutdown",
                {"analysis_closed": getattr(analysis, "closed", None),
                 "analysis_events_drained": getattr(window, "_analysis_events_drained", None),
                 "archive_state": getattr(window, "_workbench_archive_close_state", None),
                 "label_pending": labels.pending if labels is not None else None,
                 "label_closed": labels.is_closed if labels is not None else None},
            )
            if labels is not None:
                assert labels.pending == 0 and labels.is_closed and labels._service.closed
            sip.delete(window)


@pytest.mark.parametrize("launcher", [False, True])
def test_soundcard_preparation_can_finish_inside_first_login(startup, launcher):
    bridge = startup.bridge

    def login(window):
        assert bridge.started
        assert len(bridge.probes) == 1
        request, _ = bridge.probes[0]
        assert request.name is request.hostapi_name is None
        device = window.workbench_recording_device
        assert device.snapshot is None
        assert not device.login_ready
        bridge.change_worker(1)
        bridge.answer_probe()
        assert device.snapshot.channels == (0, 1, 2)
        assert device.snapshot.sample_rate == 48000
        assert not device.login_ready
        return "Engineer", "test"

    window = startup.launch(login, launcher=launcher)
    device = window.workbench_recording_device
    assert device.login_ready
    assert device.snapshot.device["index"] == 7
    assert window.mic_channels == [0]


def workbench_summary_panel(window):
    return window.sequence_window.left_panel.dashboard_panel.summary_panel


def summary_labels(panel):
    return tuple(getattr(panel, f"{name}_label").text()
                 for name in ("date", "total", "ok", "ng", "unmarked", "rate"))


def home_summary_labels(window):
    return summary_labels(workbench_summary_panel(window))


def open_wav_details(window):
    summary = workbench_summary_panel(window)
    summary.findChild(QtWidgets.QPushButton, "summaryDetails").click()
    assert summary._details_dialog.isVisible()
    return summary._details_dialog


def displayed_wav_rows(details):
    return tuple(details.table.item(index, 5).data(Qt.UserRole)
                 for index in range(details.table.rowCount()))


@pytest.mark.parametrize("launcher", [False, True], ids=["direct", "launcher"])
@pytest.mark.parametrize("state", ["current", "missing", "stale", "damaged"])
def test_wav_details_restart_loads_once_before_first_publish(
        startup, launcher, state, isolate_workbench_wav_details_state, monkeypatch):
    instant = datetime(2026, 11, 18, 9)
    monkeypatch.setattr(main_window, "datetime", SimpleNamespace(now=lambda: instant))
    path = isolate_workbench_wav_details_state
    if state in {"current", "stale"}:
        saved_at = instant if state == "current" else instant - timedelta(days=1)
        row = DailyWavRecord(saved_at, 2, "原始录制.wav", "正转", "NG")
        WorkbenchWavDetailsStore().save(WorkbenchWavDetailsSnapshot(saved_at.date(), (
            WavDetailsRow("saved-request", "left", row),
        )))
    elif state == "damaged":
        path.write_text("{", encoding="utf-8")
    expected = (row,) if state == "current" else ()
    operations = []
    load = WorkbenchWavDetailsStore.load
    publish = MotorInformationSummaryPanel.set_daily_wav_records

    def track_load(store):
        operations.append(("read", store.path))
        return load(store)

    def track_publish(panel, records, record_date):
        operations.append(("publish", record_date, len(records)))
        return publish(panel, records, record_date)

    monkeypatch.setattr(WorkbenchWavDetailsStore, "load", track_load)
    monkeypatch.setattr(MotorInformationSummaryPanel, "set_daily_wav_records", track_publish)
    logger = logging.getLogger("core")
    warning = Mock(wraps=logger.warning)
    monkeypatch.setattr(logger, "warning", warning)

    def login(window):
        assert operations == [("read", path), ("publish", instant.date(), len(expected))]
        details = open_wav_details(window)
        assert details.date_label.text() == instant.date().isoformat()
        assert displayed_wav_rows(details) == expected
        details.close()
        return "Engineer", "test"

    window = startup.launch(login, launcher=launcher)
    window._poll_analysis_events()
    window._poll_analysis_events()
    assert operations == [("read", path), ("publish", instant.date(), len(expected))]
    assert displayed_wav_rows(open_wav_details(window)) == expected
    messages = [fmt % tuple(args) for fmt, *args in
                (call.args for call in warning.call_args_list)
                if "Workbench WAV details" in fmt]
    if state == "damaged":
        assert len(messages) == 1
        assert str(path) in messages[0] and "Expecting property name" in messages[0]
    else:
        assert messages == []


def test_wav_details_restart_displays_two_pages_filters_and_opens_record_analysis(
        startup, monkeypatch, config, tmp_path):
    from unit_test.ui.test_workbench_result_state import instance
    from unit_test.ui.test_workbench_result_sync import result_event

    instant = datetime(2026, 11, 18, 9)
    monkeypatch.setattr(main_window, "datetime", SimpleNamespace(now=lambda: instant))
    channels = tuple(
        WorkbenchChannel(channel_id, f"当前 {channel_id}", "sound_pressure",
                         WorkbenchHardwareChannel("current-device", physical_index))
        for channel_id, physical_index in (("near", 2), ("far", 0))
    )
    stations = tuple(replace(station, channels=channels)
                     for station in config.analysis.context.stations)
    queue = AnalysisTestQueue("wav", {
        "display_sequence": ["谱图"],
        "谱图": {"type": "Spec", "analysis_channels": [0, 1]},
    })
    configured = configuration_with(
        config, stations=stations, test_queues={**config.analysis.test_queues, "queue-a": queue},
    )
    repository = Mock(source_identity="history-product.json")
    repository.load_product_catalog.return_value = ()
    repository.load_active.return_value = configured
    monkeypatch.setattr(main_window, "MotorWorkbenchConfigRepository", lambda: repository)

    history_path = tmp_path / "上一轮" / "机组-20.wav"
    historical_channels = (
        StationWavChannel("near", "录制时近端", "sound_pressure", "recorded-device", 3, 0),
        StationWavChannel("far", "录制时远端", "sound_pressure", "recorded-device", 1, 1),
    )
    reference = RecordAnalysisReference(
        "record-20", "left", "warm", str(history_path), historical_channels,
    )
    rows = tuple(
        WavDetailsRow(
            f"record-{index}", "left" if index % 2 == 0 else "right",
            DailyWavRecord(
                instant + timedelta(seconds=index), 2 if index % 2 == 0 else 1,
                f"机组-{index:02}.wav", ("正转", "反转")[index % 2],
                ("NG", "未标记", "OK")[index % 3],
                reference if index == 20 else None,
            ),
        )
        for index in range(21)
    )
    WorkbenchWavDetailsStore().save(WorkbenchWavDetailsSnapshot(instant.date(), rows))
    WorkbenchSummaryStore().save(WorkbenchSummarySnapshot(instant.date(), 1, 3, 0))
    window = startup.launch(lambda _: ("Engineer", "test"))
    details = open_wav_details(window)

    def assert_page(expected_rows, label):
        table = details.table
        assert details.page_label.text() == label
        assert displayed_wav_rows(details) == expected_rows
        assert [
            tuple(table.item(index, column).text() for column in range(6))
            for index in range(table.rowCount())
        ] == [
            (row.completed_at.strftime("%H:%M:%S"), str(row.station_id),
             row.file_name, row.condition, row.result, "查看")
            for row in expected_rows
        ]

    newest_first = tuple(row.record for row in reversed(rows))
    assert_page(newest_first[:20], "1 / 2")
    details.next_button.click()
    assert_page(newest_first[20:], "2 / 2")
    details.station_combo.setCurrentIndex(details.station_combo.findData(2))
    station_rows = tuple(row for row in newest_first if row.station_id == 2)
    assert_page(station_rows, "1 / 1")
    details.result_combo.setCurrentIndex(details.result_combo.findData("OK"))
    assert_page(tuple(row for row in station_rows if row.result == "OK"), "1 / 1")
    details.search_edit.setText("机组-2")
    assert_page(newest_first[:1], "1 / 1")
    assert details.record_count_label.text() == "共 1 条 · 每页 20 条"
    assert summary_labels(details) == ("2026-11-18", "21", "7", "7", "7", "33.3%")
    assert home_summary_labels(window) == ("2026-11-18", "4", "1", "3", "0", "25.0%")

    operation = details.table.item(0, 5)
    details.table.scrollToItem(operation)
    QtWidgets.QApplication.processEvents()
    QTest.mouseClick(details.table.viewport(), Qt.LeftButton,
                     pos=details.table.visualItemRect(operation).center())
    view = details.findChild(RecordAnalysisDialog)
    assert view is not None and view.isVisible()
    (payload,) = startup.analysis_service.submissions
    (task,) = payload["tasks"]
    assert (task["wav_path"], task["station_id"], task["condition_id"]) == (
        str(history_path), "left", "warm",
    )
    assert task["persist_artifacts"] is False and task["analysis_result_directory"] is None
    startup.analysis_service.events.append(result_event(task, [
        {**instance("谱图", "Spec", "NG"), "channel_index": index, "judgement_status": "completed"}
        for index in (1, 0)
    ]))
    window._poll_analysis_events()
    assert tuple(view.channel_selector.itemText(index) for index in range(view.channel_selector.count())) == (
        "CH1 · 录制时近端", "CH2 · 录制时远端",
    )
    assert "OK" in view.original_result.text()
    assert view.current_result.text().count("NG") == 2
    assert_page(newest_first[:1], "1 / 1")
    assert summary_labels(details) == ("2026-11-18", "21", "7", "7", "7", "33.3%")
    assert home_summary_labels(window) == ("2026-11-18", "4", "1", "3", "0", "25.0%")


def test_wav_details_restart_poll_rolls_open_dialog_over_and_keeps_recording_day(
        startup, paired_recording_workbench, monkeypatch):
    from unit_test.ui.test_workbench_result_state import instance
    from unit_test.ui.test_workbench_result_sync import result_event

    clock = Mock(return_value=datetime(2026, 11, 18, 23, 59))
    clock_source = SimpleNamespace(now=clock)
    monkeypatch.setattr(main_window, "datetime", clock_source)
    monkeypatch.setattr(motor_workbench_controller, "datetime", clock_source)
    row = DailyWavRecord(clock.return_value, 1, "昨天.wav", "正转", "OK")
    WorkbenchWavDetailsStore().save(WorkbenchWavDetailsSnapshot(row.completed_at.date(), (
        WavDetailsRow("previous-request", "right", row),
    )))
    recording = paired_recording_workbench
    repository = Mock(source_identity="rollover-product.json")
    repository.load_product_catalog.return_value = ()
    repository.load_active.return_value = recording.config
    monkeypatch.setattr(main_window, "MotorWorkbenchConfigRepository", lambda: repository)
    window = startup.launch(lambda _: (None, None))
    controller = window.workbench_controller
    controller.bind_recording(recording.controller._recording)
    window._activate_analysis_runtime()
    details = open_wav_details(window)
    controller.start_recording()
    recording.complete()
    assert row in displayed_wav_rows(details) and details.table.rowCount() == 4
    yesterday, = startup.analysis_service.submissions

    clock.return_value = datetime(2026, 11, 19)
    clock.reset_mock()
    assert startup.analysis_service.events == []
    window._poll_analysis_events()
    clock.assert_called_once_with()
    assert details.isVisible()
    assert summary_labels(details) == ("2026-11-19", "0", "0", "0", "0", "--")
    assert displayed_wav_rows(details) == ()
    startup.analysis_service.events.append(result_event(yesterday["tasks"][0], [instance()]))
    window._poll_analysis_events()
    assert displayed_wav_rows(details) == ()

    controller.start_recording()
    recording.complete()
    new_rows = displayed_wav_rows(details)
    assert len(new_rows) == 3
    assert all(row.completed_at == clock.return_value and row.result == "未标记" for row in new_rows)
    assert len({row.analysis_reference.request_id for row in new_rows}) == 1
    assert new_rows[0].analysis_reference.request_id != recording.requests[-2].request_id


@pytest.mark.parametrize("launcher", [False, True], ids=["direct", "launcher"])
@pytest.mark.parametrize("state", ["current", "missing", "stale", "damaged"])
def test_summary_restart_loads_once_before_login(
        startup, launcher, state, isolate_workbench_summary_state, monkeypatch):
    day = date(2026, 11, 18)
    clock = Mock(return_value=datetime(2026, 11, 18, 9))
    monkeypatch.setattr(main_window, "datetime", SimpleNamespace(now=clock), raising=False)
    path = isolate_workbench_summary_state
    if state in {"current", "stale"}:
        saved_day = day if state == "current" else date(2026, 11, 17)
        WorkbenchSummaryStore().save(WorkbenchSummarySnapshot(saved_day, 2, 1, 1))
    elif state == "damaged":
        path.write_text("{", encoding="utf-8")

    read_paths = []
    load = WorkbenchSummaryStore.load

    def track_load(store):
        read_paths.append(store.path)
        return load(store)

    monkeypatch.setattr(WorkbenchSummaryStore, "load", track_load)
    logger = logging.getLogger("core")
    warning = Mock(wraps=logger.warning)
    monkeypatch.setattr(logger, "warning", warning)
    expected = ("4", "2", "1", "1", "50.0%") if state == "current" else ("0", "0", "0", "0", "--")

    def login(window):
        assert home_summary_labels(window) == (day.isoformat(), *expected)
        assert read_paths == [path]
        clock.assert_called_once_with()
        return "Engineer", "test"

    window = startup.launch(login, launcher=launcher)
    window._poll_analysis_events()
    window._poll_analysis_events()
    assert read_paths == [path]
    assert home_summary_labels(window) == (day.isoformat(), *expected)
    summary_messages = [fmt % tuple(args) for fmt, *args in
                        (call.args for call in warning.call_args_list)
                        if "Workbench summary" in fmt]
    if state == "damaged":
        assert len(summary_messages) == 1
        assert str(path) in summary_messages[0]
        assert "Expecting property name" in summary_messages[0]
    else:
        assert summary_messages == []


def test_summary_restart_displays_total_beyond_integer_string_limit(startup, monkeypatch):
    instant = datetime(2026, 11, 18, 9)
    monkeypatch.setattr(main_window, "datetime", SimpleNamespace(now=lambda: instant))
    digit_limit = sys.get_int_max_str_digits() or sys.int_info.default_max_str_digits
    count_text = "9" * digit_limit
    count = int(count_text)
    WorkbenchSummaryStore().save(WorkbenchSummarySnapshot(instant.date(), count, count, 0))
    total_text = "1" + "9" * (digit_limit - 1) + "8"

    def login(window):
        assert home_summary_labels(window) == (
            instant.date().isoformat(), total_text, count_text, count_text, "0", "50.0%",
        )
        return "Engineer", "test"

    startup.launch(login)


def test_summary_restart_poll_rolls_over_without_analysis_events(startup, monkeypatch):
    clock = Mock(return_value=datetime(2026, 11, 18, 23, 59))
    monkeypatch.setattr(main_window, "datetime", SimpleNamespace(now=clock), raising=False)
    WorkbenchSummaryStore().save(WorkbenchSummarySnapshot(date(2026, 11, 18), 2, 1, 1))
    window = startup.launch(lambda _: ("Engineer", "test"))
    assert home_summary_labels(window) == ("2026-11-18", "4", "2", "1", "1", "50.0%")
    clock.return_value = datetime(2026, 11, 19, 0, 0)
    clock.reset_mock()
    assert startup.analysis_service.events == []
    window._poll_analysis_events()
    clock.assert_called_once_with()
    assert home_summary_labels(window) == ("2026-11-19", "0", "0", "0", "0", "--")
    assert WorkbenchSummaryStore().load() == WorkbenchSummarySnapshot(date(2026, 11, 18), 2, 1, 1)


@pytest.mark.parametrize("saved_day, expected", [(date(2026, 11, 12), "10000"), (date(2026, 11, 11), "1")])
def test_daily_round_is_loaded_before_login_without_migrating_sample(startup, monkeypatch, saved_day, expected):
    from base.load_config import LoadUiConfig
    from base.workbench_round_store import RoundSnapshot, WorkbenchRoundStore
    from ui.sequence import motor_workbench_controller as controller_module

    monkeypatch.setattr(controller_module, "datetime", SimpleNamespace(now=lambda: datetime(2026, 11, 12, 9)))
    monkeypatch.setattr(LoadUiConfig, "load_last_recorded_info", staticmethod(
        lambda _logger: {"sample_number": "007", "current_test_round": 777},
    ))
    WorkbenchRoundStore().save(RoundSnapshot(saved_day, 10000))

    def login(window):
        sequence = window.sequence_window
        assert sequence.toolsbar.sample_number_lineedit.text() == expected
        assert sequence.toolsbar.sample_number_lineedit.isReadOnly()
        assert not sequence.player_btn.isEnabled()
        assert sequence.toolsbar.sample_number_lineedit.text() == expected
        return "Engineer", "test"

    startup.launch(login)
    assert WorkbenchRoundStore().load() == RoundSnapshot(date(2026, 11, 12), int(expected))


def test_saved_missing_soundcard_never_uses_legacy_fallback_and_can_be_reselected(startup, monkeypatch):
    startup.path.write_text(json.dumps({"api_name": "WASAPI", "mic_name": "Missing", "mic_channels": [2]}))
    bridge = startup.bridge
    window = startup.launch(lambda _: ("Engineer", "test"))
    bridge.change_worker(1)
    request, _ = bridge.probes[0]
    assert (request.name, request.hostapi_name) == ("Missing", "WASAPI")
    assert window.mic == startup.mic and window.mic_channels == [0]
    bridge.answer_probe(name=None)
    device = window.workbench_recording_device
    assert device.snapshot is None and "重新选择" in device.detail

    monkeypatch.setattr(main_window, "open_hardware_selection_window", lambda **_: (False, None, [], None, []))
    window.on_hardware_window_init()
    assert len(bridge.probes) == 1
    assert "重新选择" in device.detail

    def select(**_):
        hardware_selection.save_if_changed(startup.mic, None, [1], [], strict=True)
        startup.path.write_text(json.dumps({"api_name": "WASAPI", "mic_name": "External", "mic_channels": [2]}))
        return True, None, [], startup.mic, [1]

    monkeypatch.setattr(main_window, "open_hardware_selection_window", select)
    window.on_hardware_window_init()
    assert device.snapshot is None
    assert bridge.probes[-1][0].name == "Default"
    bridge.answer_probe(0, name="Missing")
    assert device.snapshot is None
    bridge.answer_probe()
    assert device.snapshot.channels == (0, 1, 2)
    assert window.mic_channels == [1]

    refresh = window.workbench_recording_device.invalidate_selection

    def refresh_after_save():
        refresh()
        assert device.snapshot is None

    monkeypatch.setattr(window.workbench_recording_device, "invalidate_selection", refresh_after_save)
    window.on_hardware_window_init()
    assert len(bridge.probes) == 3
    bridge.answer_probe()
    assert device.snapshot.channels == (0, 1, 2)


def test_startup_probe_uses_same_saved_identity_as_legacy_restore(startup, monkeypatch):
    startup.path.write_text(json.dumps({"api_name": "WASAPI", "mic_name": "Default", "mic_channels": [2]}))
    enumerate_devices = hardware_selection._enumerate_devices

    def replace_saved_selection():
        startup.path.write_text(json.dumps({"api_name": "WASAPI", "mic_name": "External", "mic_channels": [0]}))
        return enumerate_devices()

    monkeypatch.setattr(hardware_selection, "_enumerate_devices", replace_saved_selection)
    window = startup.launch(lambda _: ("Engineer", "test"))
    request, _ = startup.bridge.probes[0]
    assert window.mic == startup.mic and window.mic_channels == [2]
    assert (request.name, request.hostapi_name) == (window.mic["name"], window.mic["hostapi_name"])


def test_injected_bridge_shuts_down_once_when_application_quits(startup, ui_qapp, monkeypatch):
    shutdown = Mock(wraps=startup.bridge.shutdown)
    monkeypatch.setattr(startup.bridge, "shutdown", shutdown)
    startup.launch(lambda _: (None, None))
    startup.launch(lambda _: (None, None), analysis_service=AnalysisServiceStub())

    try:
        ui_qapp.aboutToQuit.emit()
        shutdown.assert_called_once_with()
    finally:
        ui_qapp.aboutToQuit.disconnect(shutdown)


def test_login_cancel_and_account_switch_preserve_resident_device(startup):
    bridge = startup.bridge
    window = startup.launch(lambda _: (None, None))
    bridge.change_worker(1)
    bridge.answer_probe()
    device = window.workbench_recording_device
    prepared = device.snapshot
    assert prepared is not None and not device.login_ready

    def switch(window):
        assert not device.login_ready
        assert device.snapshot is prepared
        return "Engineer", "different account"

    startup.login(switch)
    window.on_login_window_init()
    assert device.login_ready and window.user_name == "different account"
    startup.login(lambda _: (None, None) if not device.login_ready else pytest.fail("still admitted"))
    window.on_login_window_init()
    assert device.login_ready and window.user_name == "different account"
    assert device.snapshot is prepared and len(bridge.probes) == 1


def test_retired_worker_and_shutdown_reject_late_soundcard_results(startup, ui_qapp):
    bridge = startup.bridge
    window = startup.launch(lambda _: ("Engineer", "test"))
    device = window.workbench_recording_device
    bridge.change_worker(1)
    bridge.answer_probe()
    assert device.snapshot.generation == 1
    bridge.change_worker(1, "unavailable")
    assert device.snapshot is None
    bridge.change_worker(2, "starting")
    bridge.change_worker(2)
    assert len(bridge.probes) == 2
    bridge.answer_probe(0)
    assert device.snapshot is None
    bridge.answer_probe(generation=2)
    assert device.snapshot.generation == 2

    bridge.change_worker(3)
    assert len(bridge.probes) == 3
    window.close()
    ui_qapp.processEvents()
    assert bridge.shutdown_calls == 1
    bridge.answer_probe(generation=3)
    assert device.snapshot is None and not device.login_ready


def test_ve_prewarms_discovered_full_inventory_at_profile_rate_before_login(startup, ui_qapp):
    bridge = startup.bridge
    startup.path.write_text(json.dumps({"input_selection": {"schema_version": 1, "backend": "vkinging",
        "machine_id": "test-machine-1", "physical_channels": [1]}}))
    inventory = device_info(physical_channels=[7, 1, 4])
    profile = input_config(44100, range_min=-0.5, range_max=0.5)
    profiles = SimpleNamespace(load=lambda _device, _calibrations: profile)

    def login(window):
        device = window.workbench_recording_device
        assert bridge.started and window.ve_discovery.service.starts == 1
        assert not bridge.probes
        window.editor_context.queue_data = [{"seq1": {"acq": {"detail": {"sample_rate": 96000}}}}]
        bridge.change_worker(1)
        window.ve_discovery.service.answer(inventory)
        ui_qapp.processEvents()
        request, _ = bridge.prewarms[0]
        assert request.channels == (7, 1, 4)
        assert request.sample_rate == 44100
        assert request.signature[-2:] == (-0.5, 0.5)
        assert device.snapshot is None
        bridge.finish_prewarm()
        assert device.snapshot.ve_signature == request.signature
        assert device.snapshot.physical_to_column == {7: 0, 1: 1, 4: 2}
        assert not device.login_ready
        return "Engineer", "test"

    window = startup.launch(login, ve_profile_store=profiles)
    device = window.workbench_recording_device
    prepared = device.snapshot
    assert device.login_ready and window.mic_channels == [1]
    assert window._ve_signature(window.mic, window.mic_channels)[2:4] == ((1,), 96000)

    # Legacy calibration can leave a subset in the worker. The capability
    # snapshot still supplies the full signature to recording admission.
    bridge.retained_ve_signature = window._ve_signature(window.mic, window.mic_channels)
    assert device.snapshot is prepared and bridge.retained_ve_signature != prepared.ve_signature
    bridge.retained_ve_signature = prepared.ve_signature
    assert device.snapshot is prepared

    bridge.change_worker(1, "unavailable")
    assert device.snapshot is None
    bridge.change_worker(2)
    assert window.ve_discovery.service.starts == 2
    assert device.snapshot is None
    window.ve_discovery.service.answer(inventory)
    ui_qapp.processEvents()
    assert device.snapshot.generation == 2
    assert device.snapshot.ve_signature == prepared.ve_signature
    assert len(bridge.prewarms) == 1


def test_cancel_hardware_settings_resumes_only_pending_ve_discovery(startup, ui_qapp, monkeypatch):
    bridge = startup.bridge
    startup.path.write_text(json.dumps({"input_selection": {"schema_version": 1, "backend": "vkinging",
        "machine_id": "test-machine-1", "physical_channels": [1]}}))
    saved_selection = startup.path.read_bytes()
    inventory = device_info()
    profiles = SimpleNamespace(load=lambda device, _calibrations: device["input_config"])
    window = startup.launch(lambda _: ("Engineer", "test"), ve_profile_store=profiles)
    discovery = window.ve_discovery.service
    device = window.workbench_recording_device
    bridge.change_worker(1)
    discovery.answer(inventory)
    ui_qapp.processEvents()
    bridge.finish_prewarm()
    assert device.snapshot.generation == 1

    monkeypatch.setattr(main_window, "open_hardware_selection_window", lambda **_: (False, None, [], None, []))
    selected = window.mic
    window.on_hardware_window_init()
    assert discovery.starts == 1
    assert window.mic is selected

    bridge.change_worker(1, "unavailable")
    bridge.change_worker(2)
    assert discovery.starts == 2 and device.snapshot is None
    interrupted_generation = discovery.generation
    assert window.mic["available"]

    window.on_hardware_window_init()
    assert discovery.starts == 3
    assert window.mic is selected and startup.path.read_bytes() == saved_selection
    discovery.answer(inventory, generation=interrupted_generation)
    ui_qapp.processEvents()
    assert device.snapshot is None
    discovery.answer(inventory)
    ui_qapp.processEvents()
    assert device.snapshot.generation == 2
    assert len(bridge.prewarms) == 1


def test_accepting_new_ve_selection_ignores_scan_started_inside_hardware_dialog(startup, ui_qapp, monkeypatch):
    bridge = startup.bridge
    startup.path.write_text(json.dumps({"input_selection": {"schema_version": 1, "backend": "vkinging",
        "machine_id": "test-machine-1", "physical_channels": [1]}}))
    inventory = device_info()
    replacement = device_info(machine_id="new-machine", physical_channels=[6, 2])
    profiles = SimpleNamespace(load=lambda device, _calibrations: device["input_config"])
    window = startup.launch(lambda _: ("Engineer", "test"), ve_profile_store=profiles)
    discovery = window.ve_discovery.service
    device = window.workbench_recording_device
    bridge.change_worker(1)
    discovery.answer(inventory)
    ui_qapp.processEvents()
    bridge.finish_prewarm()
    assert device.snapshot.device["machine_id"] == inventory["machine_id"]

    def save_replacement(**options):
        assert options["mic_device"]["machine_id"] == inventory["machine_id"]
        bridge.change_worker(1, "unavailable")
        bridge.change_worker(2)
        assert discovery.starts == 2 and device.snapshot is None
        discovery.answer(inventory)
        return True, None, [], replacement, [2]

    monkeypatch.setattr(main_window, "open_hardware_selection_window", save_replacement)
    window.on_hardware_window_init()
    prepared = device.snapshot
    assert prepared.generation == 2 and prepared.channels == (6, 2)
    assert prepared.device["machine_id"] == replacement["machine_id"]

    ui_qapp.processEvents()
    assert device.snapshot is prepared
    assert window.mic is replacement
    assert discovery.starts == 2 and len(bridge.prewarms) == 1
