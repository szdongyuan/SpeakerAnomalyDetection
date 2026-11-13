"""Window assembly and the analysis-to-label shutdown boundary."""

import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from threading import Event, enumerate as enumerate_threads
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PyQt5.QtCore import QTimer
from PyQt5.QtTest import QTest

import main_window
from base.audio_label_repository import (
    AudioLabelPersistenceError, AudioLabelRepository, AudioLabelUpdate,
)
from base.audio_label_sync import AudioLabelJob
from base.db_manager import DataSave
from base.recording_management import RecordingManager
from base.workbench_analysis_config import AnalysisTestQueue
from base.workbench_recording_archive import WorkbenchRecordingArchive
from base.workbench_recording_protocol import (
    StationWav, StationWavChannel, WorkbenchRecordingCompletion,
)
from ui.audio_label_sync_bridge import AudioLabelSyncBridge
from consts import error_code, model_consts
from unit_test.ui.test_motor_workbench_controller import (
    config, workbench, recording_workbench, paired_recording_workbench, configuration_with,
)
from unit_test.ui.test_workbench_analysis_lifecycle import wait_until
from unit_test.ui.test_workbench_recording_startup import startup, AnalysisServiceStub
from unit_test.ui.test_workbench_result_state import instance
from unit_test.ui.test_workbench_result_sync import result_event


def test_startup_fixture_drains_label_worker_before_deleting_window(request, monkeypatch):
    """Assert after startup teardown, recovering the test worker even on RED."""
    before = set(enumerate_threads())
    owned_services = []

    def assert_threads_drained():
        leaked = [thread for thread in enumerate_threads()
                  if thread not in before and thread.name == "audio-label-sync"]
        try:
            assert not leaked, f"startup deleted its window with live label workers: {leaked}"
        finally:
            # The assertion must not leave a non-daemon worker hanging on RED.
            for service in owned_services:
                service.begin_shutdown()
            wait_until(lambda: all(service.closed for service in owned_services))

    # Register before startup is acquired so this runs after its finalizer.
    request.addfinalizer(assert_threads_drained)
    startup = request.getfixturevalue("startup")
    configuration = Mock(source_identity="empty-product.json")
    configuration.load_product_catalog.return_value = ()
    configuration.load_active.return_value = None
    monkeypatch.setattr(main_window, "MotorWorkbenchConfigRepository", lambda: configuration)
    window = startup.launch(lambda _: (None, None))
    bridge = window.audio_label_sync_bridge
    owned_services.append(bridge._service)
    bridge.submit(AudioLabelJob("fixture-job", "missing-fixture.wav", "NG"))
    wait_until(lambda: bridge.pending == 0)
    assert bridge._service._thread.is_alive()


@pytest.fixture
def persisted_recording(paired_recording_workbench, workbench, monkeypatch, tmp_path):
    """Keep real archive/label I/O while controlling only acquisition and terminals."""
    from ui.sequence import motor_workbench_controller as controller_module

    w = paired_recording_workbench
    database_path = tmp_path / "label-chain.db"
    database = DataSave(str(database_path))
    try:
        code, detail = database.create_table()
        assert code == error_code.OK, detail
    finally:
        database.close()
    monkeypatch.setattr(model_consts, "DATABASE_PATH", str(database_path))
    clock = SimpleNamespace(value=datetime(2026, 11, 13, 12))
    monkeypatch.setattr(controller_module, "datetime", SimpleNamespace(now=lambda: clock.value))
    archive = WorkbenchRecordingArchive(str(database_path))

    def submit_archive(completion, product_name, recorded_on, callback):
        archive.save(completion, product_name, recorded_on)
        callback(completion.request_id, None)

    w.controller.bind_recording(replace(w.controller._recording, submit_archive=submit_archive))
    bridge = AudioLabelSyncBridge(AudioLabelRepository(str(database_path)))
    w.controller.bind_audio_label_sync(bridge.submit)
    changed, failed, events, published = [], [], [], []
    bridge.changed.connect(changed.append)
    bridge.failed.connect(lambda job, error: failed.append((job, error)))
    w.controller.bind_result_summary(lambda records, day: published.append((records, day)))

    def activate(service):
        w.controller.bind_analysis_service(service, events.append)
        w.controller.activate_analysis()

    def record(publish_wavs=None):
        clock.value += timedelta(seconds=1)
        before = set(w.controller._analysis_tasks)
        w.controller.start_recording()
        assert w.controller.recording_busy
        if publish_wavs is not None:
            publish_wavs(w.requests[-1])
        w.complete()
        assert not w.controller.recording_busy
        accepted = set(w.controller._analysis_tasks) - before
        assert len(accepted) == 3, (w.problems, events)
        return {context["station_id"]: {**context, "task_id": task_id}
                for task_id, context in w.controller._analysis_tasks.items() if task_id in accepted}

    def rows():
        code, result = RecordingManager().get_record_audio_data()
        assert code == error_code.OK, result
        return {row[1]: row for row in result}

    yield SimpleNamespace(w=w, controller=w.controller, panel=workbench.panel, bridge=bridge,
                          changed=changed, failed=failed, events=events, published=published,
                          activate=activate, record=record, rows=rows)
    bridge.begin_shutdown()
    wait_until(lambda: bridge.is_closed)
    assert bridge.pending == 0 and bridge._service.closed


def assert_committed_labels(chain, before, expected):
    """Check the public audio query, including every non-label metadata field."""
    after = chain.rows()
    assert set(after) == set(before)
    for path, original in before.items():
        assert after[path] == original[:5] + (expected.get(path, original[5]),) + original[6:]
    assert chain.failed == []


@pytest.mark.parametrize("forward,reverse,reverse_first,csv_failure", [
    (("OK", "NG", "OK"), ("NG", "OK", "NG"), False, False),
    (("NG", "OK", "NG"), ("OK", "NG", "OK"), True, False),
    (("OK", "NG", "OK"), ("NG", "OK", "NG"), True, True),
    ((None, None, None), (None, None, None), False, False),
], ids=["different-directions", "reverse-first", "csv-failure", "all-analysis-failed"])
def test_accepted_tasks_persist_each_wav_independently_of_station_summary(
        persisted_recording, forward, reverse, reverse_first, csv_failure):
    chain = persisted_recording
    service = AnalysisServiceStub()
    chain.activate(service)
    directions = [chain.record(), chain.record()]
    before = chain.rows()
    assert len(before) == 6 and all(row[5] == "not_labeled" for row in before.values())
    expected = {}
    ordered = [(directions[0], forward), (directions[1], reverse)]
    if reverse_first:
        ordered.reverse()
    for direction, judgements in ordered:
        for task, judgement in zip(direction.values(), judgements):
            items = [] if judgement is None else [instance(judgement=judgement)]
            if csv_failure and items:
                items[0]["artifacts"] = [{"status": "failed", "error": {
                    "stage": "saving", "message": "CSV access denied"}}]
            incoming = result_event(task, items, failed=judgement is None)
            service.events.append(incoming)
            chain.controller.poll_analysis_events()
            expected[RecordingManager.normalize_audio_path_for_db(task["wav_path"])] = (
                "not_labeled" if judgement is None else judgement)
            wait_until(lambda: len(chain.changed) == len(expected))
            assert_committed_labels(chain, before, expected)
            if len(expected) <= 3:
                assert chain.published == []  # A WAV label does not wait for its pair.
    assert len(chain.published[-1][0]) == 3
    # A duplicate terminal must not emit another committed update.
    service.events.append(incoming)
    chain.controller.poll_analysis_events()
    assert chain.bridge.pending == 0 and len(chain.changed) == 6


@pytest.mark.parametrize("late_direction", [0, 1], ids=["old-forward", "old-reverse"])
def test_late_old_round_persists_only_old_wavs_and_preserves_current_display(
        persisted_recording, late_direction):
    chain = persisted_recording
    service = AnalysisServiceStub()
    chain.activate(service)
    old = [chain.record(), chain.record()]
    current = chain.record()
    before = chain.rows()
    service.events.extend(result_event(task, [instance()]) for task in current.values())
    chain.controller.poll_analysis_events()
    wait_until(lambda: len(chain.changed) == 3)
    panel_before = [chain.panel.get_station_data(number) for number in (1, 2, 3)]
    service.events.extend(result_event(task, [instance(judgement="NG")])
                          for task in old[late_direction].values())
    chain.controller.poll_analysis_events()
    wait_until(lambda: len(chain.changed) == 6)
    assert all(event["is_current"] is False for event in chain.events[-3:])
    assert [chain.panel.get_station_data(number) for number in (1, 2, 3)] == panel_before
    expected = {RecordingManager.normalize_audio_path_for_db(task["wav_path"]): label
                for direction, label in ((current, "OK"), (old[late_direction], "NG"))
                for task in direction.values()}
    assert_committed_labels(chain, before, expected)


@pytest.mark.parametrize("bound", [False, True], ids=["sink-unbound-control", "sink-bound"])
def test_identical_accepted_chain_needs_the_label_sink_for_database_side_effect(
        persisted_recording, bound):
    chain = persisted_recording
    service = AnalysisServiceStub()
    chain.activate(service)
    chain.controller.bind_audio_label_sync(chain.bridge.submit if bound else None)
    tasks = chain.record()
    before = chain.rows()
    service.events.extend(result_event(task, [instance(judgement="NG")]) for task in tasks.values())
    chain.controller.poll_analysis_events()
    if bound:
        wait_until(lambda: len(chain.changed) == 3)
    assert len(chain.changed) == (3 if bound else 0)
    assert chain.panel.get_station_data(1)["forward"] == "NG"
    expected = {path: "NG" for path in before} if bound else {}
    assert_committed_labels(chain, before, expected)


def test_real_wav_analysis_workers_route_genuine_ok_ng_to_committed_archive_rows(
        persisted_recording):
    import multiprocessing
    import numpy as np
    import soundfile as sf
    from base.analysis.service import AnalysisService
    from base.wav_calibration_metadata import append_wav_calibration_metadata

    chain = persisted_recording
    w = chain.w
    stations = tuple(replace(station, channels=tuple(
        replace(channel, signal_type="sound_pressure") for channel in station.channels))
        for station in w.config.analysis.context.stations)
    queue = AnalysisTestQueue("wav", {"display_sequence": ["SPL"], "SPL": {
        "type": "SPL", "analysis_channels": [0], "weighting": "Z",
        "limit_checked": True, "limit_metric": "overall_spl",
        "scalar_upper_enabled": True, "scalar_upper_value": 60,
        "scalar_lower_enabled": False,
    }})
    w.repository.result = configuration_with(w.config, stations=stations,
                                             test_queues={"queue-a": queue})
    chain.controller.refresh_configuration()
    service = AnalysisService()
    chain.activate(service)
    timer = QTimer()
    timer.setInterval(20)
    timer.timeout.connect(chain.controller.poll_analysis_events)
    timer.start()
    expected = {}
    pids = set()

    def publish_wavs(request):
        group = request.station_group
        Path(group.final_path).mkdir(parents=True, exist_ok=True)
        time = np.arange(request.target_samples - request.trim_samples) / request.sample_rate
        for station in group.stations:
            label = "NG" if ((station.station_id == "left") == (group.condition_id == "warm")) else "OK"
            amplitude = .1 if label == "NG" else .01
            path = Path(group.final_path) / station.filename
            sf.write(path, amplitude * np.sin(2 * np.pi * 1000 * time),
                     request.sample_rate, subtype="PCM_24")
            assert append_wav_calibration_metadata(path, {"recorded_channels": [{
                "wav_channel_index": 0, "physical_input_channel": 0,
                "calibrated": True, "v2pa_factor": 1., "standard_spl": 94.,
            }]})
            expected[RecordingManager.normalize_audio_path_for_db(str(path))] = label

    try:
        wait_until(lambda: len(service.station_states) == 3 and all(
            state["state"] == "ready" for state in service.station_states.values()), timeout=60000)
        pids = {state["pid"] for state in service.station_states.values()}
        assert len(pids) == 3 and None not in pids
        directions = [chain.record(publish_wavs), chain.record(publish_wavs)]
        before = chain.rows()
        assert len(before) == 6 and all(row[5] == "not_labeled" for row in before.values())
        wait_until(lambda: len(chain.changed) == 6, timeout=60000)
        terminals = [event for event in chain.events if event["event_type"] == "task_result"]
        tasks = {task["task_id"]: task for direction in directions for task in direction.values()}
        assert len(terminals) == len(tasks) == 6
        assert {event["task_id"] for event in terminals} == set(tasks)
        for event in terminals:
            result = event["payload"]
            item, = result["instance_results"]
            path = RecordingManager.normalize_audio_path_for_db(tasks[event["task_id"]]["wav_path"])
            label = expected[path]
            assert result["execution_status"] == "分析完成"
            assert item["judgement_status"] == "completed" and item["judgement"] == label
            assert item["metrics"]["overall_spl"] == pytest.approx(
                70.9691 if label == "NG" else 50.9691, abs=.02)
            assert item["metrics"]["calibration_source"] == "wav"
            assert item["artifacts"] and all(artifact["status"] == "saved" for artifact in item["artifacts"])
            assert all(Path(artifact["path"]).is_file() for artifact in item["artifacts"])
        assert {update.label for update in chain.changed} == {"OK", "NG"}
        assert_committed_labels(chain, before, expected)
        assert all(chain.panel.get_station_data(number)["result"] == "NG" for number in (1, 2, 3))
    finally:
        chain.controller.begin_analysis_shutdown()
        try:
            wait_until(lambda: service.closed and any(
                event["event_type"] == "service_closed" for event in chain.events), timeout=40000)
        finally:
            timer.stop()
    assert not (pids & {child.pid for child in multiprocessing.active_children()})


@pytest.mark.parametrize("launcher", [False, True])
def test_entrypoints_bind_one_label_bridge_to_the_archive_database(
        startup, monkeypatch, launcher):
    repository_factory = Mock(wraps=AudioLabelRepository)
    bridge_factory = Mock(wraps=AudioLabelSyncBridge)
    monkeypatch.setattr(main_window, "AudioLabelRepository", repository_factory, raising=False)
    monkeypatch.setattr(main_window, "AudioLabelSyncBridge", bridge_factory, raising=False)
    window = startup.launch(lambda _: (None, None), launcher=launcher)
    try:
        window._init_analysis_runtime()
        repository_factory.assert_called_once_with(str(startup.database_path))
        bridge_factory.assert_called_once_with(
            window.audio_label_sync_bridge._service._repository, window)
        assert window._workbench_archive_bridge._archive._database_path == str(startup.database_path)
        assert window.workbench_controller._submit_audio_label == window.audio_label_sync_bridge.submit
    finally:
        window.close()
        wait_until(lambda: getattr(window, "_exit_cleanup_complete", False))


@pytest.fixture
def label_runtime(startup, paired_recording_workbench, monkeypatch):
    """Use real window/controller/label worker with deterministic recording and analysis."""
    recording = paired_recording_workbench
    configuration = Mock(source_identity="label-product.json")
    configuration.load_product_catalog.return_value = ()
    configuration.load_active.return_value = recording.config
    monkeypatch.setattr(main_window, "MotorWorkbenchConfigRepository", lambda: configuration)
    repository = Mock()
    repository.update.side_effect = lambda path, label: AudioLabelUpdate(path, path, label)
    monkeypatch.setattr(main_window, "AudioLabelRepository", lambda _: repository, raising=False)
    window = startup.launch(lambda _: (None, None))
    window.workbench_controller.bind_recording(recording.controller._recording)
    window._activate_analysis_runtime()

    def record():
        window.workbench_controller.start_recording()
        recording.complete()
        return startup.analysis_service.submissions[-1]["tasks"]

    yield window, startup.analysis_service, repository, record
    window.close()
    wait_until(lambda: getattr(window, "_exit_cleanup_complete", False))


def test_success_is_delivered_on_gui_and_failure_preserves_analysis(label_runtime):
    window, service, repository, record = label_runtime
    bridge = window.audio_label_sync_bridge
    changed, failed = [], []
    bridge.changed.connect(changed.append)
    bridge.failed.connect(lambda job, error: failed.append((job, error)))
    logger = Mock()
    window.default_logger = logger
    tasks = record()
    repository.update.side_effect = [
        AudioLabelUpdate("record-id", tasks[0]["wav_path"], "NG"),
        AudioLabelPersistenceError("injected label failure"),
    ]
    service.events.extend(result_event(task, [instance(judgement="NG")]) for task in tasks[:2])
    wait_until(lambda: len(changed) == len(failed) == 1)
    assert changed[0].audio_data_id == "record-id"
    assert failed[0][0].task_id == tasks[1]["task_id"]
    assert "injected label failure" in window.statusBar().currentMessage()
    message, *arguments = logger.warning.call_args.args
    diagnostic = message % tuple(arguments)
    assert tasks[1]["task_id"] in diagnostic
    assert tasks[1]["wav_path"] in diagnostic
    assert "injected label failure" in diagnostic
    assert window.sequence_window.left_panel.station_panel.get_station_data(2)["forward"] == "NG"


@pytest.mark.parametrize("failure", [False, True])
def test_closing_keeps_accepting_terminals_and_waits_without_blocking_gui(label_runtime, failure):
    window, service, repository, record = label_runtime
    bridge = window.audio_label_sync_bridge
    tasks = record()
    entered, release = Event(), Event()
    completed = []
    bridge.changed.connect(lambda update: completed.append("success"))
    bridge.failed.connect(lambda job, error: completed.append("failure"))

    def update(path, label):
        entered.set()
        assert release.wait(5), "test did not release label writer"
        if failure:
            raise AudioLabelPersistenceError("blocked write failed")
        return AudioLabelUpdate("last-record", path, label)

    repository.update.side_effect = update
    heartbeat = []
    timer = QTimer(window)
    timer.setInterval(5)
    timer.timeout.connect(lambda: heartbeat.append(True))
    timer.start()
    service.allow_close = False
    try:
        window.close()
        assert not bridge._closing
        service.events.append(result_event(tasks[0], [instance(judgement="NG")]))
        wait_until(entered.is_set)
        service.allow_close = True
        wait_until(lambda: service.closed)
        wait_until(lambda: len(heartbeat) >= 4)
        assert not getattr(window, "_exit_cleanup_complete", False)
        assert bridge.pending == 1
        assert completed == []
        release.set()
        wait_until(lambda: getattr(window, "_exit_cleanup_complete", False))
        assert completed == ["failure" if failure else "success"]
        assert bridge.is_closed
        assert bridge._service.closed
    finally:
        release.set()
        timer.stop()


@pytest.mark.parametrize("closed_first", [False, True])
def test_full_final_batch_is_consumed_before_label_shutdown_even_with_reentrant_close(
        label_runtime, startup, paired_recording_workbench, monkeypatch, closed_first):
    window, service, repository, record = label_runtime
    bridge = window.audio_label_sync_bridge
    tasks = record()
    WorkbenchRecordingArchive(str(startup.database_path)).save(
        paired_recording_workbench.completions[-1], "motor", date(2026, 11, 13))
    repository.update.side_effect = AudioLabelRepository(str(startup.database_path)).update
    service.allow_close = False
    window.close()
    # Drain the earlier recording/archive gates before injecting the last poll.
    wait_until(lambda: window._workbench_archive_close_state == "drained")
    service_closed = {"event_type": "service_closed", "payload": {}}
    expected_labels = ("OK", "NG", "OK")
    terminals = [result_event(task, [instance(judgement=label)])
                 for task, label in zip(tasks, expected_labels)]
    batch = ([service_closed] + terminals if closed_first
             else [terminals[0], service_closed, *terminals[1:]])

    def final_poll():
        service.closed = True  # Backend state becomes visible before consumer runs.
        return batch

    monkeypatch.setattr(service, "poll_events", final_poll)
    reentrant = []

    def close_during_batch(event):
        window.close()
        QTest.qWait(15)  # A nested Qt loop cannot establish a premature barrier.
        window._poll_analysis_events()
        reentrant.append((event["event_type"], bridge._closing,
                          getattr(window, "_exit_cleanup_complete", False)))

    window.workbench_analysis_event.connect(close_during_batch)
    wait_until(lambda: getattr(window, "_exit_cleanup_complete", False))
    assert len(reentrant) == len(tasks) + 1
    assert all(not label_closing and not exited for _, label_closing, exited in reentrant)
    assert [(call.args[0], call.args[1]) for call in repository.update.call_args_list] == [
        (task["wav_path"], label) for task, label in zip(tasks, expected_labels)]
    with closing(sqlite3.connect(startup.database_path)) as database:
        committed = dict(database.execute("SELECT file_path, labels FROM audio_data_table"))
    assert committed == {
        RecordingManager.normalize_audio_path_for_db(task["wav_path"]): label
        for task, label in zip(tasks, expected_labels)}
    assert bridge.pending == 0 and bridge.is_closed


def test_close_before_first_tick_drains_empty_label_worker(startup):
    window = startup.launch(lambda _: (None, None))
    bridge = window.audio_label_sync_bridge
    window.close()
    wait_until(lambda: getattr(window, "_exit_cleanup_complete", False))
    assert startup.analysis_service.sync_calls == []
    assert bridge.is_closed and bridge.pending == 0


def test_real_service_event_cap_keeps_polling_until_service_closed_is_consumed(
        startup, monkeypatch):
    from base.analysis.service import AnalysisService

    service = AnalysisService()
    configuration = Mock(source_identity="empty-product.json")
    configuration.load_product_catalog.return_value = ()
    configuration.load_active.return_value = None
    monkeypatch.setattr(main_window, "MotorWorkbenchConfigRepository", lambda: configuration)
    repository = Mock()
    repository.update.return_value = AudioLabelUpdate("cap-record", "cap.wav", "OK")
    monkeypatch.setattr(main_window, "AudioLabelRepository", lambda _: repository)
    window = startup.launch(lambda _: (None, None), analysis_service=service)
    bridge = window.audio_label_sync_bridge
    bridge.submit(AudioLabelJob("cap-job", "cap.wav", "OK"))
    received, closed_observations = [], []

    def receive(event):
        received.append(event)
        if event["event_type"] == "service_closed":
            closed_observations.append((getattr(window, "_exit_cleanup_complete", False), bridge._closing))

    window.workbench_analysis_event.connect(receive)
    # Reproduce the real default poll budget without starting any station workers.
    service._events.extend({"event_type": "station_status", "payload": {"state": "ready"}}
                           for _ in range(32))
    try:
        window.close()
        window._poll_analysis_events()
        assert service.closed  # The notification is queued for the next poll.
        assert len(received) == 32
        assert not window._analysis_events_drained
        assert window._analysis_timer.isActive()
        assert not bridge._closing
        assert not getattr(window, "_exit_cleanup_complete", False)
        wait_until(lambda: getattr(window, "_exit_cleanup_complete", False))
        assert [event["event_type"] for event in received].count("service_closed") == 1
        assert closed_observations == [(False, False)]
        assert bridge.is_closed and bridge.pending == 0
        window._poll_analysis_events()
        assert len(received) == 33
        repository.update.assert_called_once_with("cap.wav", "OK")
    finally:
        # Even a RED assertion must deliver the outstanding closure before teardown.
        window._poll_analysis_events()
        window.close()
        wait_until(lambda: getattr(window, "_exit_cleanup_complete", False))


def test_real_audio_manager_updates_while_open_and_reloads_committed_label(
        startup, monkeypatch, tmp_path):
    from consts import model_consts
    from ui import archive_audio_data_dialog

    monkeypatch.setattr(model_consts, "DATABASE_PATH", str(startup.database_path))
    playback = Mock()
    playback.is_audio_playing.return_value = False
    monkeypatch.setattr(archive_audio_data_dialog, "PlaybackController", lambda: playback)
    wav_path = str(tmp_path / "dialog-label.wav")
    completion = WorkbenchRecordingCompletion(
        "dialog-request", "product", "condition", 48000, 48000,
        (StationWav("station", wav_path,
                    (StationWavChannel("ch", "sound", "audio", "default", 0, 0),)),))
    WorkbenchRecordingArchive(str(startup.database_path)).save(completion, "motor", date(2026, 11, 13))
    window = startup.launch(lambda _: ("Engineer", "test"))
    bridge = window.audio_label_sync_bridge
    labels_at_open = []

    def execute(dialog):
        labels_at_open.append(dialog.all_audio_data[0][5])
        if len(labels_at_open) == 1:
            bridge.submit(AudioLabelJob("dialog-job", wav_path, "NG"))
            wait_until(lambda: dialog.model().item(0, 4).text() == "NG")
            assert dialog.all_audio_data[0][5] == "NG"
        dialog.close()
        dialog.deleteLater()

    monkeypatch.setattr(archive_audio_data_dialog.ArchiveAudioDataDialog, "exec", execute)
    try:
        window.on_audio_manager_init()
        assert bridge.receivers(bridge.changed) == 0
        window.on_audio_manager_init()
        assert labels_at_open == ["not_labeled", "NG"]
        assert bridge.receivers(bridge.changed) == 0
        assert RecordingManager().get_record_audio_data()[1][0][5] == "NG"
    finally:
        window.close()
        wait_until(lambda: getattr(window, "_exit_cleanup_complete", False))
