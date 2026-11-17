"""Native details intent through host runtime and shared-service routing."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from unittest.mock import Mock

from PyQt5 import sip
from PyQt5.QtCore import QCoreApplication, QEvent, Qt, QTimer
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

import main_window
from base.record_analysis_reference import RecordAnalysisReference
from base.workbench_analysis_config import AnalysisTestQueue
from base.workbench_config_values import WorkbenchChannel, WorkbenchHardwareChannel
from base.workbench_configuration_loader import WorkbenchConfigurationError
from base.workbench_recording_protocol import StationWavChannel
from ui.sequence.record_analysis_coordinator import RecordAnalysisCoordinator
from ui.sequence.record_analysis_dialog import RecordAnalysisDialog
from ui.sequence.record_analysis_presenter import RecordAnalysisPresenter
from unit_test.ui.test_motor_workbench_controller import (
    config, workbench, recording_workbench, paired_recording_workbench, configuration_with,
)
from unit_test.ui.test_workbench_result_sync import sync, result_event, labels
from unit_test.ui.test_workbench_result_state import instance
from unit_test.ui.test_workbench_recording_startup import startup
from unit_test.ui.test_workbench_analysis_lifecycle import wait_until
from unit_test.ui.test_workbench_wav_details_sync import wav_sync, rows


def create_view(record, parent):
    dialog = RecordAnalysisDialog(parent)
    dialog.show_record(record)
    return dialog, RecordAnalysisPresenter(dialog)


def click_operation(details, row=0):
    item = details.table.item(row, 5)
    details.table.scrollToItem(item)
    QApplication.processEvents()
    QTest.mouseClick(details.table.viewport(), Qt.LeftButton,
                     pos=details.table.visualItemRect(item).center())


def test_current_analysis_query_uses_applied_valid_snapshot(workbench, config):
    w = workbench
    assert w.controller.get_current_analysis_snapshot() == (None, None)
    w.repository.result = config
    w.controller.refresh_configuration()
    analysis, version = w.controller.get_current_analysis_snapshot()
    assert analysis is config.analysis and version
    w.repository.result = configuration_with(config, project_name="unapplied")
    assert w.controller.get_current_analysis_snapshot() == (analysis, version)
    w.repository.result = WorkbenchConfigurationError("invalid")
    w.controller.refresh_configuration()
    assert w.controller.get_analysis_config() is analysis
    assert w.controller.get_current_analysis_snapshot() == (None, None)
    assert w.controller.open_record_analysis(None, None) is None


def test_native_entry_routes_view_only_and_preserves_details_and_production(wav_sync):
    s = wav_sync
    jobs = []
    s.w.controller.bind_audio_label_sync(jobs.append)
    production = s.record()
    coordinator = RecordAnalysisCoordinator(s.service,
        s.w.controller.get_current_analysis_snapshot, create_view)
    s.w.controller.bind_record_analysis(coordinator)
    s.summary.record_analysis_requested.connect(s.w.controller.open_record_analysis)
    s.summary.show()
    s.summary._open_details()
    details = s.summary._details_dialog
    details.table.setFocus()
    details.table.setCurrentCell(0, 5)
    details.table.horizontalScrollBar().setValue(details.table.horizontalScrollBar().maximum())
    state = (details._station_id, details._result, details._query, details._page,
             details.table.horizontalScrollBar().value(), details.table.verticalScrollBar().value())
    focused = details.focusWidget()
    before = deepcopy(s.panel.get_station_data(1)), list(rows(s)), labels(s.summary)
    round_before = s.w.controller._current_round_id, s.w.controller._recording_round
    old_events = len(s.events)
    plot_calls = []
    original = s.w.controller._plot_presenter.analysis_event
    s.w.controller._plot_presenter.analysis_event = lambda event: (plot_calls.append(event), original(event))
    click_operation(details)
    view = details.findChild(RecordAnalysisDialog)
    assert view is not None and view.isVisible()
    assert view.parentWidget() is details and view.windowModality() == Qt.WindowModal
    assert QApplication.activeModalWidget() is view
    assert view.navigation.count() == 1 and view.navigation.currentRow() == 0
    assert len(s.service.submissions) == 2
    task = s.service.submissions[-1]["tasks"][0]
    assert task["persist_artifacts"] is False
    selected = details.table.item(0, 5).data(Qt.UserRole)
    assert task["wav_path"] == selected.analysis_reference.wav_path
    assert s.w.controller.open_record_analysis(selected, details) is view
    assert len(s.service.submissions) == 2
    s.deliver(result_event(task, [instance(judgement="NG")]))
    assert view.status.text() != "分析中"
    assert (s.panel.get_station_data(1), rows(s), labels(s.summary)) == before
    assert (s.w.controller._current_round_id, s.w.controller._recording_round) == round_before
    assert jobs == [] and len(s.events) == old_events and plot_calls == []
    view.close()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()
    assert (details._station_id, details._result, details._query, details._page,
            details.table.horizontalScrollBar().value(), details.table.verticalScrollBar().value()) == state
    assert details.focusWidget() is focused is details.table
    click_operation(details)
    reopened = details.findChild(RecordAnalysisDialog)
    assert reopened is not None and reopened is not view
    second_task = s.service.submissions[-1]["tasks"][0]
    assert second_task["task_id"] != task["task_id"]
    # The modal child keeps native dispatch alive for the existing production callback.
    QTimer.singleShot(0, lambda: s.deliver(result_event(production["left"], [instance()])))
    QTest.qWait(20)
    assert reopened.isVisible() and s.panel.get_station_data(2)["forward"] == "OK"
    assert jobs[0].task_id == production["left"]["task_id"]
    assert rows(s, 2)[0]["result"] == "OK"
    before_mixed = len(s.events)
    plot_calls.clear()
    unknown = result_event({**task, "task_id": "unregistered"}, [])
    closed = {"event_type": "service_closed", "payload": {}}
    s.deliver(closed, result_event(task, [instance()]), result_event(second_task, [instance()]),
              result_event(production["right"], [instance()]), unknown)
    assert [e["event_type"] for e in s.events[before_mixed:]] == ["service_closed", "task_result", "task_result"]
    assert s.events[-1] is unknown
    assert [job.task_id for job in jobs] == [production["left"]["task_id"], production["right"]["task_id"]]
    assert s.panel.get_station_data(1)["forward"] == "OK"
    assert rows(s, 1)[0]["result"] == "OK"
    assert labels(s.summary) == before[2]
    assert [e["event_type"] for e in plot_calls] == ["service_closed", "task_result", "task_result"]
    assert not coordinator._tasks and not coordinator._sessions
    details.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()
    assert sip.isdeleted(details)


def test_runtime_factory_summary_signal_and_parent_destroy(startup, monkeypatch, config, tmp_path):
    channel = WorkbenchChannel("input", "CH1", "sound_pressure", WorkbenchHardwareChannel("offline", 0))
    config = configuration_with(config,
        stations=tuple(replace(station, channels=(channel,)) for station in config.analysis.context.stations),
        test_queues={name: AnalysisTestQueue("wav", {"display_sequence": ["SPL", "FFT"],
            "SPL": {"type": "SPL", "analysis_channels": [0]},
            "FFT": {"type": "FFT", "analysis_channels": [0], "n_fft": 1024}}) for name in config.analysis.test_queues})
    repository = Mock(source_identity="isolated-product.json")
    repository.load_product_catalog.return_value = ()
    repository.load_active.return_value = config
    monkeypatch.setattr(main_window, "MotorWorkbenchConfigRepository", lambda: repository)
    window = startup.launch(lambda _: (None, None))
    wait_until(lambda: startup.analysis_service.poll_calls > 0)
    controller = window.workbench_controller
    summary = window.sequence_window.left_panel.dashboard_panel.summary_panel
    station = config.analysis.context.stations[0]
    condition = config.analysis.context.conditions[0]
    channels = tuple(StationWavChannel(ch.channel_id, ch.channel_name, ch.signal_type,
                     "recorded-device", i, i) for i, ch in enumerate(station.channels))
    reference = RecordAnalysisReference("recorded", station.station_id, condition.condition_id,
                                        str(tmp_path / "same.wav"), channels)
    now = datetime.now()
    summary.set_daily_wav_records([dict(completed_at=now, station_id=7, file_name="same.wav",
        condition="saved condition", result="NG", analysis_reference=replace(reference,
            request_id=f"recorded-{i}", wav_path=str(tmp_path / str(i) / "same.wav")))
        for i in range(45)], now.date())
    summary._open_details()
    details = summary._details_dialog
    details.station_combo.setCurrentIndex(details.station_combo.findData(7))
    details.result_combo.setCurrentIndex(details.result_combo.findData("NG"))
    details.search_edit.setText("same")
    details.next_button.click()
    summary._open_details()  # Cached dialog must not duplicate intent connections.
    forwarded = []
    summary.record_analysis_requested.connect(lambda *args: forwarded.append(args))
    click_operation(details)
    view = details.findChild(RecordAnalysisDialog)
    assert view is not None and view.isVisible()
    assert len(forwarded) == len(startup.analysis_service.submissions) == 1
    assert forwarded[0] == (details.table.item(0, 5).data(Qt.UserRole), details)
    assert view.original_result.text().endswith("NG")
    assert [view.navigation.item(i).data(Qt.UserRole) for i in range(2)] == ["SPL", "FFT"]
    assert view.navigation.currentRow() == 0
    task = startup.analysis_service.submissions[-1]["tasks"][0]
    selected = forwarded[0][0]
    assert task["wav_path"] == selected.analysis_reference.wav_path and task["persist_artifacts"] is False
    state = (details._station_id, details._result, details._query, details._page,
             details.table.horizontalScrollBar().value(), details.table.verticalScrollBar().value(),
             details.focusWidget())
    for row in (1, 0, 1, 0):
        view.navigation.setCurrentRow(row)
    assert len(startup.analysis_service.submissions) == 1
    view.close()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()
    assert (details._station_id, details._result, details._query, details._page,
            details.table.horizontalScrollBar().value(), details.table.verticalScrollBar().value(),
            details.focusWidget()) == state
    assert state[:4] == (7, "NG", "same", 2)
    click_operation(details)
    view = details.findChild(RecordAnalysisDialog)
    assert len(startup.analysis_service.submissions) == 2
    reopened_task = startup.analysis_service.submissions[-1]["tasks"][0]
    received = []
    window.workbench_analysis_event.connect(received.append)
    QTest.qWait(60)
    assert window._analysis_timer.isActive()
    details.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()
    assert sip.isdeleted(details) and sip.isdeleted(view)
    assert summary._details_dialog is None
    startup.analysis_service.events.extend(result_event(t, [instance()]) for t in (task, reopened_task))
    wait_until(lambda: not controller._record_analysis._tasks)
    assert received == []
    summary._open_details()
    click_operation(summary._details_dialog)
    assert len(startup.analysis_service.submissions) == 3
    window.close()
    wait_until(lambda: getattr(window, "_exit_cleanup_complete", False))
    assert startup.analysis_service.shutdown_calls == 1
    assert window.audio_label_sync_bridge.is_closed
    assert not window._analysis_timer.isActive()
    final_details = summary._details_dialog
    final_details.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()
    assert sip.isdeleted(final_details)
