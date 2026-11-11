"""WAV-to-result acceptance of the develop loudness algorithm integration."""

import csv
from copy import deepcopy
from dataclasses import replace
import json
import time

import numpy as np
import pytest
import soundfile as sf

from base.analysis import build_analysis_batch
from base.analysis.contracts import AnalysisChannel, AnalysisInstance, AnalysisTask
from base.analysis.executor import execute_task
from base.analysis.service import AnalysisService
from base.core_algorithm.sound_quality.service import run_sound_quality


@pytest.fixture
def loud_task(tmp_path):
    rate = 44100
    t = np.arange(int(0.7 * rate)) / rate
    tone = 0.02 * np.sin(2 * np.pi * 1000 * t)
    wav = tmp_path / "station.wav"
    sf.write(wav, np.column_stack((tone, 2 * tone)), rate, subtype="FLOAT")
    parameters = {
        "method": "per_segment", "analysis_channels": [0, 1],
        "advanced": {"curve_y_unit": "sone"},
        "display": {"curves": ["loudness_time", "specific_loudness_profile"]},
    }
    return AnalysisTask(
        "loud-task", "station-1", "工位1", "condition-1", "工况1",
        str(wav), str(tmp_path / "results"),
        (AnalysisChannel(0, "CH1", "sound_pressure", v2pa_factor=2.0),
         AnalysisChannel(1, "CH2", "sound_pressure", v2pa_factor=3.0)),
        tuple(AnalysisInstance("响度", "LOUD", ch, deepcopy(parameters)) for ch in (0, 1)),
        {"mode": "none"},
    )


@pytest.mark.parametrize("method", ["per_segment", "time_varying_iso532_1"])
@pytest.mark.parametrize("unit", ["sone", "phon"])
def test_result_matches_source_service_on_each_calibrated_wav_column(loud_task, method, unit):
    config = deepcopy(loud_task.instances[0].parameters)
    config.update(method=method)
    config["advanced"].update(
        curve_y_unit=unit, analysis_time_range_enabled=True,
        analysis_start_time_sec=0.1, analysis_end_time_sec=0.6,
    )
    task = replace(loud_task, instances=tuple(replace(i, parameters=deepcopy(config)) for i in loud_task.instances))
    before = deepcopy(config)
    result = execute_task(task)
    assert result["execution_status"] == "分析完成", result
    signal, rate = sf.read(task.wav_path, dtype="float32", always_2d=True)
    for item in result["instance_results"]:
        ch = item["channel_index"]
        reference_config = deepcopy(config)
        reference_config["enabled"] = True
        reference_config["required_summary_metrics"] = [
            "steady_state_average_sone", "steady_state_average_phon",
            "max_transient_sone", "max_transient_phon",
        ]
        reference = run_sound_quality(signal[:, ch], rate, task.channels[ch].v2pa_factor,
            {"enabled": True, "items": {"LOUD": reference_config}}).loudness
        values = (reference.raw_result.loudness_sone if unit == "sone"
                  else reference.raw_result.loudness_level_phon)
        curve = item["display_payload"]
        np.testing.assert_allclose(curve["y"], values, rtol=1e-10)
        np.testing.assert_allclose(curve["x"], reference.raw_result.time_s + 0.1)
        for key in reference_config["required_summary_metrics"]:
            assert item["metrics"][key] == pytest.approx(reference.summary[key], rel=1e-10)
        assert curve["y_unit"] == unit and curve["x_unit"] == "s"
        assert item["metrics"]["analysis_start_time_sec"] == pytest.approx(0.1)
        assert item["judgement_status"] == "not_enabled"
        assert all(a["status"] == "saved" for a in item["artifacts"])
        curve_file = next(a["path"] for a in item["artifacts"] if a["kind"] == "curve_csv")
        with open(curve_file, encoding="utf-8-sig", newline="") as stream:
            header, *rows = csv.reader(stream)
        assert header == ["时间(s)", f"响度{unit}", f"下限{unit}", f"上限{unit}"]
        np.testing.assert_allclose([float(row[1]) for row in rows], values, rtol=1e-9)
        assert len(rows) == len(curve["x"])
    assert config == before
    assert result["instance_results"][1]["metrics"]["steady_state_average_sone"] > result["instance_results"][0]["metrics"]["steady_state_average_sone"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("metric", ["curve_y", "steady_state_average", "max_transient"])
@pytest.mark.parametrize("limit_unit,upper,expected", [("sone", 0.001, "NG"), ("phon", 200.0, "OK")])
def test_thresholds_use_their_own_unit_and_hidden_summary_metrics(loud_task, metric, limit_unit, upper, expected):
    config = {
        "method": "per_segment", "display": {"curves": [], "summary_metrics": []},
        "advanced": {"curve_y_unit": "sone"}, "curve_limit_unit": limit_unit,
        "limit_checked": True, "limit_metric": metric,
        "limit_mode": "manual", "manual_input_mode": "constant",
        "constant_upper_enabled": True, "constant_upper_value": upper,
        "constant_lower_enabled": False,
        "scalar_upper_enabled": True, "scalar_upper_value": upper,
        "scalar_lower_enabled": False,
    }
    task = replace(loud_task, instances=(replace(loud_task.instances[0], parameters=config),))
    item, = execute_task(task)["instance_results"]
    assert item["execution_status"] == "分析完成", item
    assert item["judgement_status"] == "completed" and item["judgement"] == expected
    assert item["metrics"]["limit_unit"] == limit_unit
    if expected == "NG":
        assert item["violations"][0]["deviation_unit"] == limit_unit
    if metric == "curve_y" and limit_unit == "phon":
        assert item["display_payload"]["upper"] == []  # Never draw phon bounds on sone.
    with open(item["artifacts"][0]["path"], encoding="utf-8-sig", newline="") as stream:
        row, = csv.DictReader(stream)
    assert row["判定结果"] == expected


@pytest.mark.parametrize("problem,stage", [
    ("calibration", "calibration"), ("signal_type", "validation"),
    ("range", "validation"), ("method", "calculation"), ("limits", "judgement"),
])
def test_failure_preserves_other_channel_result(loud_task, problem, stage):
    task = loud_task
    config = deepcopy(task.instances[0].parameters)
    if problem == "calibration":
        task = replace(task, channels=(replace(task.channels[0], v2pa_factor=None), task.channels[1]))
    elif problem == "signal_type":
        task = replace(task, channels=(replace(task.channels[0], signal_type="acceleration"), task.channels[1]))
    elif problem == "range":
        config["advanced"].update(analysis_time_range_enabled=True, analysis_start_time_sec=0, analysis_end_time_sec=1.0)
    elif problem == "method":
        config["method"] = "unknown"
    else:
        config.update(limit_checked=True, limit_metric="max_transient", scalar_upper_enabled=True,
                      scalar_upper_value=1, scalar_lower_enabled=True, scalar_lower_value=100)
    task = replace(task, instances=(replace(task.instances[0], parameters=config), task.instances[1]))
    result = execute_task(task)
    failed, good = result["instance_results"]
    assert result["execution_status"] == "结果不完整"
    assert failed["error"]["stage"] == stage
    assert good["execution_status"] == "分析完成" and good["error"] is None


def test_silence_is_valid_zero_loudness_and_nonfinite_samples_fail(loud_task):
    samples, rate = sf.read(loud_task.wav_path, dtype="float32", always_2d=True)
    samples[:, 0] = 0
    samples[10, 1] = np.nan
    sf.write(loud_task.wav_path, samples, rate, subtype="FLOAT")
    silence, invalid = execute_task(loud_task)["instance_results"]
    assert silence["execution_status"] == "分析完成"
    assert silence["metrics"]["steady_state_average_sone"] == 0
    assert silence["metrics"]["max_transient_sone"] == 0
    assert invalid["execution_status"] == "分析失败"
    assert invalid["error"]["stage"] == "validation"


def _payload(task):
    return {
        "stations": [{"station_id": task.station_id, "channels": [
            {"channel_name": c.channel_name, "signal_type": c.signal_type, "v2pa_factor": c.v2pa_factor}
            for c in task.channels]}],
        "conditions": [{"condition_id": task.condition_id, "segmented_analysis": {"mode": "none"}}],
        "tasks": [{"task_id": task.task_id, "station_id": task.station_id, "condition_id": task.condition_id,
                   "wav_path": task.wav_path, "analysis_result_directory": task.analysis_result_directory,
                   "analysis_config": {"display_sequence": ["响度"], "响度": {
                       **task.instances[0].parameters, "type": "LOUD"}}}],
    }


def test_batch_admits_loudness_and_resident_worker_returns_real_results(loud_task):
    payload = _payload(loud_task)
    batch = build_analysis_batch(payload)
    assert not batch.preparation_issues
    assert [i.channel_index for i in batch.tasks[0].instances] == [0, 1]
    service = AnalysisService()

    def pump(predicate):
        deadline = time.monotonic() + 40
        events = []
        while time.monotonic() < deadline:
            events.extend(service.poll_events())
            if predicate(events):
                return events
            time.sleep(0.01)
        pytest.fail(f"Worker timed out: {service.station_states}")

    try:
        assert service.sync_stations([loud_task.station_id])["accepted"]
        pump(lambda _: service.station_states[loud_task.station_id]["state"] == "ready")
        assert service.submit_batch(payload)["accepted"]
        events = pump(lambda events: any(e["event_type"] == "task_result" for e in events))
        result, = [e["payload"] for e in events if e["event_type"] == "task_result"]
        assert result["execution_status"] == "分析完成", result
        assert len(result["instance_results"]) == 2
    finally:
        service.shutdown()
        pump(lambda events: any(e["event_type"] == "service_closed" for e in events))
