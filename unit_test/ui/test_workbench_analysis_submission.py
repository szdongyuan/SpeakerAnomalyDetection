"""Published WAV handoff, rejection and real resident-worker execution."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from time import monotonic
from unittest.mock import Mock

import pytest
from PyQt5.QtCore import QTimer
from PyQt5.QtTest import QTest

from base.workbench_analysis_config import AnalysisTestQueue
from unit_test.ui.test_motor_workbench_controller import (
    config, workbench, recording_workbench, configuration_with,
)
from unit_test.ui.test_workbench_recording_startup import AnalysisServiceStub, startup


@pytest.fixture
def submission(recording_workbench):
    w = recording_workbench
    service = AnalysisServiceStub()
    events = []
    w.controller.bind_analysis_service(service, events.append)
    w.controller.activate_analysis()
    return SimpleNamespace(w=w, service=service, events=events)


def test_submit_only_after_success_and_use_published_paths(submission):
    w, service = submission.w, submission.service
    w.controller.start_recording()
    assert not service.submissions
    w.complete(outcome="failure")
    assert not service.submissions
    w.controller.start_recording()
    w.complete()
    payload, = service.submissions
    saved = {s.station_id: s.path for s in w.completions[-1].stations}
    assert {t["station_id"]: t["wav_path"] for t in payload["tasks"]} == saved
    for task in payload["tasks"]:
        wav = Path(task["wav_path"])
        assert wav.parent.name == "audio"
        assert task["analysis_result_directory"] == str(wav.parent.parent)
        assert "analysis_config" not in task
    w.complete()  # Duplicate recording completion must not submit again.
    assert len(service.submissions) == 1
    assert submission.events[-1]["payload"]["accepted"]


def test_empty_queue_records_without_submitting(submission):
    w = submission.w
    queues = {name: AnalysisTestQueue("wav", {"display_sequence": []})
              for name in w.config.analysis.test_queues}
    w.repository.result = configuration_with(w.config, test_queues=queues)
    w.controller.refresh_configuration()
    submission.service.station_states = {}
    w.controller.start_recording()
    w.complete()
    assert w.completions
    assert not submission.service.submissions
    assert submission.events[-1]["event_type"] == "analysis_skipped"


def test_not_ready_blocks_recording_and_poll_reenables_it(submission):
    w, service = submission.w, submission.service
    service.station_states["right"]["state"] = "starting"
    w.controller.start_recording()
    assert not w.requests
    assert "尚未就绪" in w.states[-1][1]
    service.station_states["right"]["state"] = "ready"
    service.events.append({"event_type": "station_status", "payload": {"state": "ready"}})
    w.controller.poll_analysis_events()
    assert w.states[-1][0]


@pytest.mark.parametrize("reason", ["QueueFull", "StationNotReady", "ServiceClosing"])
def test_rejection_is_delivered_without_retry_or_losing_recording(submission, reason):
    w, service = submission.w, submission.service
    calls = []

    def reject(payload):
        calls.append(payload)
        return {"accepted": False, "accepted_task_ids": [], "issues": [{
            "task_id": None, "field_path": "$input", "error_type": reason, "message": reason}]}

    service.submit_batch = reject
    w.controller.start_recording()
    w.complete()
    assert len(w.completions) == len(calls) == 1
    event = submission.events[-1]
    assert event["event_type"] == "submission_result"
    assert not event["payload"]["accepted"]
    assert event["request_id"] == w.completions[0].request_id
    assert not w.controller._analysis_tasks
    w.controller.poll_analysis_events()
    assert len(calls) == 1


def test_old_results_keep_identity_without_becoming_current(submission):
    w, service = submission.w, submission.service
    w.controller.start_recording()
    w.complete()
    old = service.submissions[0]["tasks"][0]
    request_id = w.completions[0].request_id
    # The recording handoff has ended; the next configuration may now be applied.
    w.repository.result = configuration_with(w.config, project_name="next product")
    w.controller.refresh_configuration()
    service.events.append({"event_type": "task_result", "task_id": old["task_id"],
                           "payload": {"execution_status": "分析完成"}})
    w.controller.poll_analysis_events()
    event = submission.events[-1]
    assert event["request_id"] == request_id
    assert not event["is_current"]
    assert old["task_id"] not in w.controller._analysis_tasks


def test_two_rounds_same_configuration_use_request_identity(submission):
    w, service = submission.w, submission.service
    for _ in range(3):  # warm, cold, warm again
        w.controller.start_recording()
        w.complete()
    first = service.submissions[0]["tasks"][0]
    latest = service.submissions[2]["tasks"][0]
    for task in (first, latest):
        service.events.append({"event_type": "task_result", "task_id": task["task_id"], "payload": {}})
    w.controller.poll_analysis_events()
    first_event, last_event = submission.events[-2:]
    assert not first_event["is_current"]
    assert last_event["is_current"]
    assert first_event["request_id"] != last_event["request_id"]


def test_submission_and_storage_failures_reach_main_window(startup, monkeypatch):
    import main_window

    repository = Mock(source_identity="isolated-product.json")
    repository.load_active.return_value = None
    monkeypatch.setattr(main_window, "MotorWorkbenchConfigRepository", lambda: repository)
    window = startup.launch(lambda _: (None, None))
    events = []
    window.workbench_analysis_event.connect(events.append)
    window._on_analysis_event({"event_type": "submission_result", "request_id": "req-1", "payload": {
        "accepted": False, "accepted_task_ids": [], "issues": [{"message": "queue full"}]}})
    assert "音频已保存" in window.statusBar().currentMessage()
    assert "queue full" in window.statusBar().currentMessage()
    window._on_analysis_event({"event_type": "task_result", "task_id": "task-1", "payload": {
        "execution_status": "分析完成", "error": None, "instance_results": [{"artifacts": [
            {"status": "failed", "error": {"stage": "saving", "message": "access denied"}}]}]}})
    assert "文件保存错误" in window.statusBar().currentMessage()
    assert len(events) == 2


def test_four_stations_two_conditions_read_saved_wavs(recording_workbench, tmp_path):
    import csv
    import hashlib
    import numpy as np
    import soundfile as sf
    from base.analysis.service import AnalysisService
    from base.wav_calibration_metadata import append_wav_calibration_metadata

    w = recording_workbench
    source = w.config.analysis.context.stations[0]
    stations = []
    for i in range(4):
        channels = tuple(replace(source.channels[0], channel_id=f"ch-{i}-{j}", channel_name=f"CH{j+1}",
                                 signal_type="sound_pressure", hardware_channel=replace(
                                     source.channels[0].hardware_channel, physical_channel_index=(i+j) % 2))
                         for j in range(2))
        stations.append(replace(source, station_id=f"s{i}", station_name=f"工位{i+1}", channels=channels))
    ids = tuple(s.station_id for s in stations)
    conditions = tuple(replace(c, applicable_station_ids=ids, segmented_analysis={"mode": "none"})
                       for c in w.config.analysis.context.conditions)
    queues = {}
    for c in conditions:
        items = {f"SPL {i}": {"type": "SPL", "analysis_channels": [i], "weighting": "Z"}
                 for i in (0, 1)}
        items["FFT"] = {"type": "FFT", "analysis_channels": [1], "n_fft": 2048, "weighting": "Z"}
        if c == conditions[-1]:
            items["Spec"] = {"type": "Spec", "analysis_channels": [0], "n_fft": 256, "hop_length": 128}
        queues[c.test_queue] = AnalysisTestQueue("wav", {"display_sequence": list(items), **items})
    w.config = configuration_with(w.config, stations=tuple(stations), conditions=conditions,
                                  test_queues=queues,
                                  plot_layout=replace(w.config.ui.layout, columns=4, cells=(
                                      w.config.ui.layout.cells[0] + (w.config.ui.layout.cells[0][0],),)))
    w.repository.result = w.config
    w.controller.refresh_configuration()
    service = AnalysisService()
    events, originals, expected = [], {}, {}
    w.controller.bind_analysis_service(service, events.append)
    w.controller.activate_analysis()
    timer = QTimer()
    timer.timeout.connect(w.controller.poll_analysis_events)
    timer.start(20)

    def wait(predicate, seconds=30):
        end = monotonic() + seconds
        while not predicate() and monotonic() < end:
            QTest.qWait(10)
        assert predicate(), service.station_states

    try:
        wait(lambda: all(s["state"] == "ready" for s in service.station_states.values()))
        pids = {s: state["pid"] for s, state in service.station_states.items()}
        for round_index in range(2):
            w.controller.start_recording()
            request = w.requests[-1]
            group = request.station_group
            Path(group.final_path).mkdir(parents=True, exist_ok=True)
            for station_index, plan in enumerate(group.stations):
                wav = Path(group.final_path) / plan.filename
                rate = 44100  # Deliberately differs from the fake recording completion's 48000 Hz.
                tone = np.sin(2 * np.pi * 1000 * np.arange(3 * rate) / rate)
                amplitude = 0.01 * (round_index + 1) * (station_index + 1)
                sf.write(wav, np.column_stack((amplitude * tone, amplitude * 2 * tone)), rate, subtype="PCM_24")
                assert append_wav_calibration_metadata(wav, {"recorded_channels": [
                    {"wav_channel_index": j, "physical_input_channel": plan.channels[j].physical_channel_index,
                     "calibrated": True, "v2pa_factor": 2.0, "standard_spl": 94.0} for j in (0, 1)]})
                originals[wav] = hashlib.sha256(wav.read_bytes()).hexdigest()
                expected[plan.station_id, group.condition_id] = 20 * np.log10(amplitude * 2 / np.sqrt(2) / 20e-6)
            # Only completed, closed files are announced; no audio samples cross the controller boundary.
            w.complete()
        wait(lambda: sum(e["event_type"] == "task_result" for e in events) == 8)
        results = [e for e in events if e["event_type"] == "task_result"]
        assert all(e["is_current"] for e in results)
        csv_paths = []
        for event in results:
            result = event["payload"]
            assert result["execution_status"] == "分析完成"
            for item in result["instance_results"]:
                assert item["metrics"]["sample_rate"] == 44100
                assert item["metrics"]["wav_duration_sec"] == 3
                assert item["metrics"]["calibration_source"] == "wav"
                if item["analysis_type"] == "Spec":
                    display = item["display_payload"]
                    assert display["reduction"] == "none"
                    assert np.asarray(display["values_db"]).shape == (
                        len(display["frequency_hz"]), len(display["time_s"]))
                if item["analysis_type"] == "SPL":
                    value = expected[result["station_id"], result["condition_id"]] + 20 * np.log10(item["channel_index"] + 1)
                    assert item["metrics"]["overall_spl"] == pytest.approx(value, abs=0.001)
                    artifact, curve_artifact = item["artifacts"]
                    assert curve_artifact["kind"] == "curve_csv"
                    assert curve_artifact["status"] == "saved"
                    assert artifact["status"] == "saved"
                    path = Path(artifact["path"])
                    with path.open(encoding="utf-8-sig", newline="") as stream:
                        row, = list(csv.DictReader(stream))
                    assert len(row) == 4
                    assert float(row["总体声压级dB"]) == pytest.approx(value, abs=0.001)
                    csv_paths.append(path)
        assert len(set(csv_paths)) == 16
        expected_directories = {wav.parent.parent / "csv" / wav.stem for wav in originals}
        assert {path.parent for path in csv_paths} == expected_directories
        for event in results:
            for item in event["payload"]["instance_results"]:
                for artifact in item["artifacts"]:
                    assert artifact["status"] == "saved"
                    assert Path(artifact["path"]).parent in expected_directories
                    assert Path(artifact["path"]).is_file()
        for audio_directory in {wav.parent for wav in originals}:
            assert not (audio_directory / "analysis").exists()
            assert not (audio_directory / "csv").exists()
        assert {s: state["pid"] for s, state in service.station_states.items()} == pids
        assert all(hashlib.sha256(path.read_bytes()).hexdigest() == value for path, value in originals.items())
        assert not w.controller._analysis_tasks
        # Files may be removed/damaged after publication; never fabricate successful results.
        w.controller.start_recording()
        group = w.requests[-1].station_group
        Path(group.final_path).mkdir(parents=True, exist_ok=True)
        paths = [Path(group.final_path) / plan.filename for plan in group.stations]
        paths[1].write_bytes(b"not a WAV")
        sf.write(paths[2], np.zeros(44100), 44100)  # Wrong number of WAV columns.
        sf.write(paths[3], np.ones((44100, 2)) * 0.01, 44100)  # Missing calibration.
        w.complete()
        wait(lambda: sum(e["event_type"] == "task_result" for e in events) == 12)
        failures = [e["payload"] for e in events if e["event_type"] == "task_result"][-4:]
        assert all(result["execution_status"] == "分析失败" for result in failures)
        assert not paths[0].exists()
        assert paths[1].read_bytes() == b"not a WAV"
        assert not w.controller._analysis_tasks
    finally:
        w.controller.begin_analysis_shutdown()
        wait(lambda: service.closed, seconds=40)
        timer.stop()
