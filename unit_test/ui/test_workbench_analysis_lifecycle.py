"""Application wiring for the resident service; acquisition stays independent."""

from dataclasses import replace
from datetime import date, datetime
from time import monotonic
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtTest import QTest

import main_window
from base.workbench_configuration_loader import WorkbenchConfigurationError
from base.workbench_summary_store import (
    WorkbenchSummarySnapshot, WorkbenchSummaryStateError, WorkbenchSummaryStore,
)
from base.workbench_wav_details_store import (
    WorkbenchWavDetailsSnapshot, WorkbenchWavDetailsStateError, WorkbenchWavDetailsStore,
)
from unit_test.ui.test_motor_workbench_controller import (
    config, workbench, configuration_with,
)
from unit_test.ui.test_workbench_recording_startup import startup, AnalysisServiceStub


def wait_until(predicate, timeout=5000):
    end = monotonic() + timeout / 1000
    while not predicate() and monotonic() < end:
        QTest.qWait(10)
    assert predicate()


def test_controller_syncs_only_after_activation_and_valid_load(workbench, config):
    w = workbench
    service = AnalysisServiceStub()
    w.repository.result = config
    w.controller.refresh_configuration()
    w.controller.bind_analysis_service(service, Mock())
    assert service.sync_calls == []
    w.controller.activate_analysis()
    assert service.sync_calls == [("right", "left", "spare")]
    w.repository.result = WorkbenchConfigurationError("broken product")
    w.controller.refresh_configuration()
    assert len(service.sync_calls) == 1
    w.repository.result = config
    w.controller.refresh_configuration()
    assert service.sync_calls[-1] == ("right", "left", "spare")
    w.repository.result = configuration_with(config, stations=config.analysis.context.stations[:2])
    w.controller.refresh_configuration()
    assert service.sync_calls[-1] == ("right", "left")
    w.repository.result = None
    w.controller.refresh_configuration()
    assert service.sync_calls[-1] == ()


def test_shutdown_prevents_late_activation_and_config_sync(workbench, config):
    w = workbench
    service = AnalysisServiceStub()
    events = []
    w.controller.bind_analysis_service(service, events.append)
    w.controller.begin_analysis_shutdown()
    w.controller.begin_analysis_shutdown()
    w.controller.activate_analysis()
    w.repository.result = config
    w.controller.refresh_configuration()
    assert service.sync_calls == []
    assert service.shutdown_calls == 1
    w.controller.poll_analysis_events()
    assert [event["event_type"] for event in events] == ["service_closed"]


@pytest.mark.parametrize("launcher", [False, True])
def test_window_defers_activation_until_after_login_and_forwards_events(
        startup, monkeypatch, config, launcher):
    service = AnalysisServiceStub()
    repository = Mock(source_identity="isolated-product.json")
    repository.load_product_catalog.return_value = ()
    repository.load_active.return_value = config
    monkeypatch.setattr(main_window, "MotorWorkbenchConfigRepository", lambda: repository)
    # The shared startup fixture injects this instance into both entry paths.
    startup.analysis_service = service

    def login(window):
        QTest.qWait(30)  # Nested event loop must not start analysis during construction.
        assert service.sync_calls == []
        assert service.poll_calls == 0
        return None, None

    window = startup.launch(login, launcher=launcher)
    received = []
    window.workbench_analysis_event.connect(received.append)
    wait_until(lambda: service.poll_calls > 0)
    assert service.sync_calls == [("right", "left", "spare")]
    window._activate_analysis_runtime()
    assert len(service.sync_calls) == 1
    event = {"event_type": "task_result", "task_id": "old-task", "payload": {}}
    service.events.append(event)
    wait_until(lambda: event in received)
    window.close()
    wait_until(lambda: service.closed)
    wait_until(lambda: getattr(window, "_exit_cleanup_complete", False))
    assert window.audio_label_sync_bridge.is_closed
    assert not window._analysis_timer.isActive()
    assert service.shutdown_calls == 1


def test_close_waits_for_service_and_keeps_polling(startup, monkeypatch, config):
    service = startup.analysis_service
    service.allow_close = False
    repository = Mock(source_identity="isolated-product.json")
    repository.load_product_catalog.return_value = ()
    repository.load_active.return_value = config
    monkeypatch.setattr(main_window, "MotorWorkbenchConfigRepository", lambda: repository)
    window = startup.launch(lambda _: (None, None))
    wait_until(lambda: service.poll_calls > 0)
    window.close()
    window.close()
    previous = service.poll_calls
    wait_until(lambda: service.poll_calls > previous)
    assert service.shutdown_calls == 1
    assert not getattr(window, "_exit_cleanup_complete", False)
    repository.load_active.return_value = None
    window.on_workbench_configuration_committed()
    assert service.sync_calls == [("right", "left", "spare")]
    service.allow_close = True
    wait_until(lambda: getattr(window, "_exit_cleanup_complete", False))


@pytest.mark.parametrize("failures", [
    pytest.param((), id="both-saved"),
    pytest.param(("summary",), id="summary-failed"),
    pytest.param(("details",), id="details-failed"),
    pytest.param(("summary", "details"), id="both-failed"),
])
def test_wav_details_restart_final_close_saves_independently_and_ignores_nested_events(
        startup, monkeypatch, failures):
    day = date(2026, 11, 18)
    old_day = date(2026, 11, 17)
    monkeypatch.setattr(main_window, "datetime",
                        SimpleNamespace(now=lambda: datetime(2026, 11, 18, 9)), raising=False)
    stores = {"summary": WorkbenchSummaryStore(), "details": WorkbenchWavDetailsStore()}
    stores["summary"].save(WorkbenchSummarySnapshot(old_day, 2, 1, 1))
    stores["details"].save(WorkbenchWavDetailsSnapshot(old_day, ()))
    prior_bytes = {name: store.path.read_bytes() for name, store in stores.items()}
    window = startup.launch(lambda _: ("Engineer", "test"))
    logger = Mock()
    window.default_logger = logger
    nested_observations, attempts = [], []

    def dispatch_nested_close(phase):
        nested_event = QCloseEvent()
        QCoreApplication.sendEvent(window, nested_event)
        nested_observations.append((phase, nested_event.isAccepted(),
                                    getattr(window, "_exit_cleanup_complete", False)))

    def intercept_save(name, error_type):
        save = type(stores[name]).save

        def write(current_store, snapshot):
            attempts.append((name, current_store.path, snapshot))
            dispatch_nested_close(name)
            if name in failures:
                raise error_type(f"{current_store.path}: injected write failure")
            save(current_store, snapshot)

        return write

    attempted_before_warning = []

    def show_warning(_parent, title, _message):
        attempted_before_warning.append(tuple(name for name, _, _ in attempts))
        dispatch_nested_close(title)

    warning = Mock(side_effect=show_warning)
    monkeypatch.setattr(WorkbenchSummaryStore, "save", intercept_save("summary", WorkbenchSummaryStateError))
    monkeypatch.setattr(WorkbenchWavDetailsStore, "save",
                        intercept_save("details", WorkbenchWavDetailsStateError))
    monkeypatch.setattr(main_window.QMessageBox, "warning", warning)
    window.close()
    wait_until(lambda: getattr(window, "_exit_cleanup_complete", False))

    expected = {
        "summary": WorkbenchSummarySnapshot(day, 0, 0, 0),
        "details": WorkbenchWavDetailsSnapshot(day, ()),
    }
    assert attempts == [(name, store.path, expected[name]) for name, store in stores.items()]
    warning_titles = {"summary": "首页信息汇总保存失败", "details": "WAV 明细保存失败"}
    warning_messages = {"summary": "本次新增统计未保存", "details": "本次 WAV 明细未保存"}
    assert nested_observations == [
        (phase, False, False) for phase in (*stores, *(warning_titles[name] for name in failures))
    ]
    assert attempted_before_warning == [("summary", "details")] * len(failures)
    assert warning.call_count == logger.warning.call_count == len(failures)
    for name, notice, log in zip(failures, warning.call_args_list, logger.warning.call_args_list):
        path = stores[name].path
        parent, title, message = notice.args
        assert parent is window and title == warning_titles[name]
        assert warning_messages[name] in message and str(path) in message
        fmt, *args = log.args
        diagnostic = fmt % tuple(args)
        assert str(path) in diagnostic and "injected write failure" in diagnostic
    for name, store in stores.items():
        if name in failures:
            assert store.path.read_bytes() == prior_bytes[name]
        else:
            assert store.load() == expected[name]


def test_close_before_first_tick_does_not_start_stations(startup, monkeypatch, config):
    service = startup.analysis_service
    repository = Mock(source_identity="isolated-product.json")
    repository.load_product_catalog.return_value = ()
    repository.load_active.return_value = config
    monkeypatch.setattr(main_window, "MotorWorkbenchConfigRepository", lambda: repository)
    window = startup.launch(lambda _: (None, None))
    window.close()
    wait_until(lambda: service.closed)
    wait_until(lambda: getattr(window, "_exit_cleanup_complete", False))
    assert window.audio_label_sync_bridge.is_closed
    assert service.sync_calls == []


@pytest.mark.parametrize("launcher", [False, True])
def test_both_entrypoints_create_one_default_service(startup, monkeypatch, launcher):
    from base.analysis import service as service_module

    service = AnalysisServiceStub()
    factory = Mock(return_value=service)
    monkeypatch.setattr(service_module, "AnalysisService", factory)
    startup.analysis_service = None
    repository = Mock(source_identity="isolated-product.json")
    repository.load_product_catalog.return_value = ()
    repository.load_active.return_value = None
    monkeypatch.setattr(main_window, "MotorWorkbenchConfigRepository", lambda: repository)
    window = startup.launch(lambda _: (None, None), launcher=launcher)
    timer = window._analysis_timer
    window._init_analysis_runtime()
    wait_until(lambda: service.poll_calls > 0)
    factory.assert_called_once_with()
    assert window.analysis_service is service
    assert window._analysis_timer is timer
    assert service.sync_calls == [()]
    assert service.station_states == {}
    window.close()
    wait_until(lambda: service.closed)


def test_invalid_initial_config_recovers_without_rebinding(workbench, config):
    w = workbench
    service = AnalysisServiceStub()
    w.repository.result = WorkbenchConfigurationError("missing queue")
    w.controller.refresh_configuration()
    w.controller.bind_analysis_service(service, Mock())
    w.controller.activate_analysis()
    assert service.sync_calls == []
    w.repository.result = config
    w.controller.refresh_configuration()
    assert service.sync_calls == [("right", "left", "spare")]


def test_worker_failure_is_reported_and_does_not_stop_polling(startup, monkeypatch):
    service = startup.analysis_service
    repository = Mock(source_identity="isolated-product.json")
    repository.load_product_catalog.return_value = ()
    repository.load_active.return_value = None
    monkeypatch.setattr(main_window, "MotorWorkbenchConfigRepository", lambda: repository)
    window = startup.launch(lambda _: (None, None))
    event = {"event_type": "station_status", "station_id": "station-a", "payload": {
        "state": "failed", "error": {"message": "worker failed", "error_type": "WorkerUnavailable"}}}
    received = []
    window.workbench_analysis_event.connect(received.append)
    service.events.append(event)
    wait_until(lambda: event in received)
    assert "worker failed" in window.statusBar().currentMessage()
    previous = service.poll_calls
    wait_until(lambda: service.poll_calls > previous)
    assert window._analysis_timer.isActive()


def test_construction_failure_never_starts_workers(startup, monkeypatch):
    service = startup.analysis_service
    initialize = main_window.MainWindow._init_analysis_runtime

    def fail_after_setup(window):
        initialize(window)
        raise RuntimeError("injected initialization failure")

    monkeypatch.setattr(main_window.MainWindow, "_init_analysis_runtime", fail_after_setup)
    with pytest.raises(RuntimeError, match="injected initialization failure"):
        startup.launch(lambda _: (None, None))
    QTest.qWait(50)
    assert service.sync_calls == []
    assert service.poll_calls == 0


def test_real_workers_reused_resized_and_closed_by_window(startup, monkeypatch, config, tmp_path):
    import multiprocessing
    import numpy as np
    import soundfile as sf
    from base.analysis.service import AnalysisService

    service = AnalysisService()
    startup.analysis_service = service
    fourth = replace(config.analysis.context.stations[-1], station_id="fourth", station_name="工位4")
    layout = replace(config.ui.layout, columns=4,
                     cells=(config.ui.layout.cells[0] + (config.ui.layout.cells[0][0],),))
    original = configuration_with(config, stations=config.analysis.context.stations + (fourth,),
                                  plot_layout=layout)
    repository = Mock(source_identity="isolated-product.json")
    repository.load_product_catalog.return_value = ()
    repository.load_active.return_value = original
    monkeypatch.setattr(main_window, "MotorWorkbenchConfigRepository", lambda: repository)
    window = startup.launch(lambda _: (None, None))
    events = []
    window.workbench_analysis_event.connect(events.append)
    all_pids = set()
    try:
        wait_until(lambda: len(service.station_states) == 4 and all(
            s["state"] == "ready" for s in service.station_states.values()), timeout=60000)
        pids = {sid: state["pid"] for sid, state in service.station_states.items()}
        all_pids.update(pids.values())
        repository.load_active.return_value = configuration_with(original, project_name="renamed")
        window.on_workbench_configuration_committed()
        QTest.qWait(150)
        assert {sid: s["pid"] for sid, s in service.station_states.items()} == pids
        repository.load_active.return_value = configuration_with(
            original, stations=original.analysis.context.stations[:2])
        window.on_workbench_configuration_committed()
        wait_until(lambda: set(service.station_states) == {"right", "left"}, timeout=15000)
        assert {sid: s["pid"] for sid, s in service.station_states.items()} == {
            sid: pids[sid] for sid in ("right", "left")}
        repository.load_active.return_value = original
        window.on_workbench_configuration_committed()
        wait_until(lambda: len(service.station_states) == 4 and all(
            s["state"] == "ready" for s in service.station_states.values()), timeout=60000)
        all_pids.update(s["pid"] for s in service.station_states.values())
        assert service.station_states["right"]["pid"] == pids["right"]
        # Submit explicitly in the test: production recording handoff is the next phase.
        wav = tmp_path / "three-seconds.wav"
        rate = 44100
        sf.write(wav, 0.01 * np.sin(2 * np.pi * 1000 * np.arange(3 * rate) / rate),
                 rate, subtype="PCM_24")
        payload = {
            "stations": [{"station_id": "right", "channels": [
                {"channel_name": "CH1", "signal_type": "sound_pressure", "v2pa_factor": 1.0}]}],
            "conditions": [{"condition_id": "warm", "segmented_analysis": {"mode": "none"}}],
            "tasks": [{"task_id": "close-with-work", "station_id": "right", "condition_id": "warm",
                       "wav_path": str(wav), "analysis_result_directory": str(tmp_path / "results"),
                       "analysis_config": {"display_sequence": ["SPL"], "SPL": {
                           "type": "SPL", "analysis_channels": [0], "weighting": "Z"}}}],
        }
        assert service.submit_batch(payload)["accepted"]
        window.close()
        wait_until(lambda: service.closed, timeout=20000)
        wait_until(lambda: getattr(window, "_exit_cleanup_complete", False))
        assert not (all_pids & {child.pid for child in multiprocessing.active_children()})
        result, = [e["payload"] for e in events if e["event_type"] == "task_result"]
        assert result["execution_status"] == "分析完成"
        assert result["instance_results"][0]["metrics"]["overall_spl"] == pytest.approx(50.9691, abs=0.01)
        output = tmp_path / "results" / "csv" / wav.stem
        assert set((tmp_path / "results").rglob("*.csv")) == {
            output / "SPL_CH1_总体值.csv", output / "SPL_CH1_曲线.csv",
        }
    finally:
        window.close()
        wait_until(lambda: service.closed, timeout=40000)


def test_record_view_detaches_before_shared_service_shutdown_once(workbench):
    calls = []
    service = AnalysisServiceStub()
    service.shutdown = lambda: calls.append("service")
    coordinator = Mock()
    coordinator.begin_shutdown.side_effect = lambda: calls.append("views")
    workbench.controller.bind_analysis_service(service, Mock())
    workbench.controller.bind_record_analysis(coordinator)
    workbench.controller.begin_analysis_shutdown()
    workbench.controller.begin_analysis_shutdown()
    assert calls == ["views", "service"]
