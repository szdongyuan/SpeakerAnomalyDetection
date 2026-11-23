"""SHRP/ROUGH integration through the real station executor and core algorithms."""

from copy import deepcopy
from dataclasses import replace
import json
import time

import numpy as np
import pytest
import soundfile as sf

from base.analysis import build_analysis_batch
from base.analysis.contracts import AnalysisChannel, AnalysisInputError, AnalysisInstance, AnalysisTask
from base.analysis.executor import execute_task
from base.analysis_algorithm_adapters import calculate_analysis_instance
from base.core_algorithm.sound_quality.loudness import LoudnessAnalyzer, LoudnessMethod
from base.core_algorithm.sound_quality.roughness import compute_roughness_daniel_weber
from base.core_algorithm.sound_quality.sharpness import compute_sharpness_din45692


@pytest.fixture
def quality_task(tmp_path):
    rate = 48000
    t = np.arange(rate) / rate
    signal = 0.03 * (1 + 0.7 * np.sin(2 * np.pi * 70 * t)) * np.sin(2 * np.pi * 1000 * t)
    path = tmp_path / "sound.wav"
    sf.write(path, np.column_stack((signal, signal * 0.6)), rate, subtype="FLOAT")
    return AnalysisTask(
        "quality", "station", "station", "condition", "condition", str(path), str(tmp_path / "out"),
        (AnalysisChannel(0, "CH1", "sound_pressure", v2pa_factor=2.0),
         AnalysisChannel(1, "CH2", "sound_pressure", v2pa_factor=3.0)),
        (), {"mode": "none"},
    )


@pytest.mark.parametrize("kind,method", [("SHRP", "time_varying_iso532_1"), ("ROUGH", "daniel_weber")])
@pytest.mark.parametrize("channel", [0, 1])
@pytest.mark.parametrize("rate", [48000, 44100])
def test_executor_matches_core_with_calibration_and_crop(quality_task, kind, method, channel, rate):
    time = np.arange(rate) / rate
    wave = 0.03 * (1 + 0.7 * np.sin(2 * np.pi * 70 * time)) * np.sin(2 * np.pi * 1000 * time)
    sf.write(quality_task.wav_path, np.column_stack((wave, wave * 0.6)), rate, subtype="FLOAT")
    config = {"method": method, "advanced": {"analysis_time_range_enabled": True,
                           "analysis_start_time_sec": 0.1, "analysis_end_time_sec": 0.8}}
    instance = AnalysisInstance("quality", kind, channel, config)
    result = execute_task(replace(quality_task, instances=(instance,)))
    item, = result["instance_results"]
    assert item["execution_status"] == "分析完成", item
    signal, rate = sf.read(quality_task.wav_path, dtype="float32", always_2d=True)
    wave = signal[int(0.1 * rate):int(0.8 * rate), channel].astype(np.float64)
    factor = quality_task.channels[channel].v2pa_factor
    if kind == "SHRP":
        raw = LoudnessAnalyzer(rate).compute(wave, v2pa_factor=factor, method=LoudnessMethod(method))
        expected = compute_sharpness_din45692(raw.specific_loudness, raw.bark_axis, raw.time_s, method=LoudnessMethod(method))
        values, unit = expected.sharpness_acum, "acum"
        summary = {"s_max_acum": expected.summary.s_max, "s_mean_acum": expected.summary.s_mean,
                   "s_stationary_acum": expected.summary.s_stationary}
    else:
        expected = compute_roughness_daniel_weber(wave * factor, rate)
        values, unit = expected.roughness_asper, "asper"
        summary = {"r_mean_asper": expected.summary.r_mean_asper, "r_max_asper": expected.summary.r_max_asper}
    curve = item["display_payload"]
    np.testing.assert_allclose(np.asarray(curve["y"], dtype=float), values, rtol=1e-10, equal_nan=True)
    np.testing.assert_allclose(curve["x"], expected.time_s + 0.1, rtol=1e-10)
    assert curve["x_unit"] == "s" and curve["y_unit"] == unit
    assert curve["valid_mask"] == np.isfinite(values).tolist()
    for key, value in summary.items():
        assert item["metrics"][key] == pytest.approx(value, rel=1e-10)
    assert len(item["artifacts"]) == 2
    assert all(artifact["status"] == "saved" for artifact in item["artifacts"])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("order", [("LOUD", "SHRP"), ("SHRP", "LOUD"), ("SHRP",)])
@pytest.mark.parametrize("method", ["time_varying_iso532_1"])
def test_sharpness_frontend_shared_independent_of_order(quality_task, monkeypatch, order, method):
    original = LoudnessAnalyzer.compute
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LoudnessAnalyzer, "compute", counted)
    config = {"method": method, "advanced": {"analysis_time_range_enabled": True,
                                             "analysis_start_time_sec": 0.05,
                                             "analysis_end_time_sec": 0.85}}
    task = replace(quality_task, instances=tuple(
        AnalysisInstance(kind, kind, 0, deepcopy(config)) for kind in order))
    shared = execute_task(task)
    assert shared["execution_status"] == "分析完成", shared
    assert len(calls) == 1
    assert execute_task(task, reuse_computations=False) == shared
    assert len(calls) == 1 + len(order)


def test_sharpness_frontend_uses_own_parameters(quality_task, monkeypatch):
    original = LoudnessAnalyzer.compute
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(kwargs["output_time_resolution_s"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LoudnessAnalyzer, "compute", counted)
    task = replace(quality_task, instances=(
        AnalysisInstance("LOUD", "LOUD", 0, {"advanced": {"output_time_resolution_ms": 4}}),
        AnalysisInstance("SHRP", "SHRP", 0, {}),
    ))
    result = execute_task(task)
    assert result["execution_status"] == "分析完成"
    assert calls == [0.004, 0.002]
    assert len(result["instance_results"][0]["display_payload"]["x"]) < len(result["instance_results"][1]["display_payload"]["x"])


@pytest.mark.parametrize("settings", [
    {"method": "per_segment"}, {"advanced": {"output_time_resolution_ms": 4}},
])
def test_legacy_sharpness_requires_explicit_config_update(quality_task, monkeypatch, settings):
    from base.analysis_config_validation import sharpness_execution_issue

    def unexpected(*args, **kwargs):
        pytest.fail("Old configurations must not start a silently changed computation")

    monkeypatch.setattr(LoudnessAnalyzer, "compute", unexpected)
    assert sharpness_execution_issue(settings)[0] == "SharpnessConfigurationMigrationRequired"
    result = execute_task(replace(quality_task, instances=(AnalysisInstance("SHRP", "SHRP", 0, settings),)))
    item, = result["instance_results"]
    assert item["execution_status"] == "分析失败"
    assert "复核阈值" in item["error"]["message"]


@pytest.mark.parametrize("order", [("LOUD", "SHRP"), ("SHRP", "LOUD")])
def test_fixed_sharpness_does_not_follow_segmented_loudness(quality_task, monkeypatch, order):
    original = LoudnessAnalyzer.compute
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(kwargs["method"].value)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LoudnessAnalyzer, "compute", counted)
    task = replace(quality_task, instances=tuple(
        AnalysisInstance(kind, kind, 0, {"method": "per_segment"} if kind == "LOUD" else {}) for kind in order
    ))
    result = execute_task(task)
    assert result["execution_status"] == "分析完成"
    assert sorted(calls) == ["per_segment", "time_varying_iso532_1"]
    assert result["instance_results"][order.index("SHRP")]["metrics"]["method"] == "time_varying_iso532_1"


def test_sharpness_ignores_inactive_legacy_segment_parameters_for_reuse(quality_task, monkeypatch):
    original = LoudnessAnalyzer.compute
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LoudnessAnalyzer, "compute", counted)
    task = replace(quality_task, instances=(
        AnalysisInstance("LOUD", "LOUD", 0, {}),
        AnalysisInstance("SHRP", "SHRP", 0, {"advanced": {"stationary_frame_duration_s": .5}}),
    ))
    assert execute_task(task)["execution_status"] == "分析完成"
    assert len(calls) == 1


def test_sharpness_postprocessing_failure_preserves_loudness_and_roughness(quality_task, monkeypatch):
    original = LoudnessAnalyzer.compute
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LoudnessAnalyzer, "compute", counted)
    task = replace(quality_task, instances=(
        AnalysisInstance("bad sharpness", "SHRP", 0, {"sharpness": {"weighting": "unsupported"}}),
        AnalysisInstance("loudness", "LOUD", 0, {}),
        AnalysisInstance("roughness", "ROUGH", 0, {}),
    ))
    result = execute_task(task)
    assert [i["execution_status"] for i in result["instance_results"]] == ["分析失败", "分析完成", "分析完成"]
    assert len(calls) == 1


def test_shared_frontend_failure_does_not_stop_roughness(quality_task, monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise ValueError("frontend unavailable")

    monkeypatch.setattr(LoudnessAnalyzer, "compute", fail)
    task = replace(quality_task, instances=tuple(AnalysisInstance(k, k, 0, {}) for k in ("SHRP", "LOUD", "ROUGH")))
    result = execute_task(task)
    assert [i["execution_status"] for i in result["instance_results"]] == ["分析失败", "分析失败", "分析完成"]
    assert len(calls) == 1


@pytest.mark.parametrize("overlap,expected_hop", [(0, 9600), (0.25, 2400), (0.75, 7200)])
def test_roughness_independent_frontend_and_exact_hop(quality_task, monkeypatch, overlap, expected_hop):
    def forbidden(*args, **kwargs):
        pytest.fail("ROUGH must not run the loudness frontend")

    monkeypatch.setattr(LoudnessAnalyzer, "compute", forbidden)
    item, = execute_task(replace(quality_task, instances=(
        AnalysisInstance("roughness", "ROUGH", 0, {"overlap": overlap}),
    )))["instance_results"]
    assert item["execution_status"] == "分析完成"
    assert item["display_payload"]["algorithm_metadata"]["hop_samples"] == expected_hop


@pytest.mark.parametrize("kind,metric", [
    ("SHRP", "curve_y"), ("SHRP", "s_max_acum"), ("SHRP", "s_mean_acum"),
    ("SHRP", "s_stationary_acum"), ("ROUGH", "curve_y"),
    ("ROUGH", "r_mean_asper"), ("ROUGH", "r_max_asper"),
])
@pytest.mark.parametrize("upper,verdict", [(100.0, "OK"), (-1.0, "NG")])
def test_curve_and_scalar_thresholds_use_correct_units(quality_task, kind, metric, upper, verdict):
    config = {
        "limit_checked": True, "limit_metric": metric,
        "limit_mode": "manual", "manual_input_mode": "constant",
        "constant_upper_enabled": True, "constant_upper_value": upper,
        "scalar_upper_enabled": True, "scalar_upper_value": upper,
    }
    item, = execute_task(replace(quality_task, instances=(AnalysisInstance("quality", kind, 0, config),)))["instance_results"]
    unit = "acum" if kind == "SHRP" else "asper"
    assert item["execution_status"] == "分析完成", item
    assert item["judgement_status"] == "completed" and item["judgement"] == verdict
    assert item["metrics"]["limit_unit"] == unit
    if verdict == "NG":
        assert item["violations"][0]["deviation_unit"] == unit
        assert item["violations"][0]["x_unit"] == ("s" if metric == "curve_y" else None)


@pytest.mark.parametrize("metric", ["curve_y", "s_mean_acum", "s_stationary_acum"])
def test_silent_sharpness_is_undefined_and_cannot_pass(quality_task, metric):
    sf.write(quality_task.wav_path, np.zeros((48000, 2)), 48000, subtype="FLOAT")
    config = {"limit_checked": True, "limit_metric": metric,
              "limit_mode": "manual", "manual_input_mode": "constant",
              "constant_upper_enabled": True, "constant_upper_value": 10,
              "scalar_upper_enabled": True, "scalar_upper_value": 10}
    item, = execute_task(replace(quality_task, instances=(AnalysisInstance("sharpness", "SHRP", 0, config),)))["instance_results"]
    assert item["execution_status"] == "分析完成"
    assert item["judgement_status"] == "failed" and item["judgement"] is None
    assert item["metrics"]["valid_frame_count"] == 0
    assert item["metrics"]["valid_summary_metrics"] == []
    assert all(value is None for value in item["display_payload"]["y"])
    assert item["metrics"]["s_mean_acum"] is None


@pytest.mark.parametrize("kind", ["SHRP", "ROUGH"])
def test_threshold_without_overlap_or_unknown_metric_cannot_pass(quality_task, kind):
    for config in ({"limit_mode": "csv", "limit_data": [[4, 5], [100, 100], [None, None]]},
                   {"limit_metric": "unknown", "scalar_upper_enabled": True, "scalar_upper_value": 100}):
        item, = execute_task(replace(quality_task, instances=(
            AnalysisInstance("quality", kind, 0, {"limit_checked": True, **config}),
        )))["instance_results"]
        assert item["judgement_status"] == "failed" and item["judgement"] is None


@pytest.mark.parametrize("duration,success", [(0.199, False), (0.2, True)])
def test_roughness_minimum_duration(quality_task, duration, success):
    config = {"advanced": {"analysis_time_range_enabled": True,
                           "analysis_start_time_sec": 0, "analysis_end_time_sec": duration}}
    item, = execute_task(replace(quality_task, instances=(AnalysisInstance("roughness", "ROUGH", 0, config),)))["instance_results"]
    assert (item["execution_status"] == "分析完成") == success
    if not success:
        assert "200 ms" in item["error"]["message"]


@pytest.mark.parametrize("kind", ["SHRP", "ROUGH"])
def test_direct_adapter_crop_and_legacy_judgement_match_station(quality_task, kind):
    signal, rate = sf.read(quality_task.wav_path, dtype="float32", always_2d=True)
    config = {"advanced": {"analysis_time_range_enabled": True,
                           "analysis_start_time_sec": 0.1, "analysis_end_time_sec": 0.8},
              "limit_checked": True, "limit_mode": "csv",
              "limit_data": [[0.1, 0.8], [100, 100], [None, None]]}
    direct = calculate_analysis_instance(kind, signal[:, 0], rate, config, 2.0,
                                         source="test", sequence_snapshot={})
    item, = execute_task(replace(quality_task, instances=(AnalysisInstance("quality", kind, 0, config),)))["instance_results"]
    assert direct["judgement"] == item["judgement"] == "OK"
    np.testing.assert_allclose(direct["curve"]["x"], item["display_payload"]["x"])
    np.testing.assert_allclose(direct["curve"]["y"], np.asarray(item["display_payload"]["y"], dtype=float), equal_nan=True)


@pytest.fixture
def quality_payload(quality_task):
    return {
        "stations": [{"station_id": "station", "channels": [
            {"channel_name": c.channel_name, "signal_type": c.signal_type, "v2pa_factor": c.v2pa_factor}
            for c in quality_task.channels]}],
        "conditions": [{"condition_id": "condition", "segmented_analysis": {"mode": "none"}}],
        "tasks": [{"task_id": "quality", "station_id": "station", "condition_id": "condition",
                   "wav_path": quality_task.wav_path, "analysis_result_directory": quality_task.analysis_result_directory,
                   "analysis_config": {"display_sequence": ["sharpness", "roughness"],
                                       "sharpness": {"type": "SHRP", "analysis_channels": [0, 1]},
                                       "roughness": {"type": "ROUGH", "analysis_channels": [0]}}}],
    }




def test_batch_accepts_new_types_and_validates_advanced_range(quality_payload):
    batch = build_analysis_batch(quality_payload)
    assert not batch.preparation_issues
    assert len(batch.tasks[0].instances) == 3
    quality_payload["tasks"][0]["analysis_config"]["roughness"]["advanced"] = {
        "analysis_time_range_enabled": True, "analysis_start_time_sec": 1, "analysis_end_time_sec": 0.5,
    }
    with pytest.raises(AnalysisInputError):
        build_analysis_batch(quality_payload)


def test_new_types_execute_in_resident_station_process(quality_payload):
    from base.analysis.service import AnalysisService

    service = AnalysisService()

    def pump(predicate):
        events = []
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            events.extend(service.poll_events())
            if predicate(events):
                return events
            time.sleep(0.01)
        pytest.fail(f"Worker timed out: {service.station_states}")

    try:
        assert service.sync_stations(["station"])["accepted"]
        pump(lambda _: service.station_states["station"]["state"] == "ready")
        assert service.submit_batch(quality_payload)["accepted"]
        events = pump(lambda events: any(e["event_type"] == "task_result" for e in events))
        result, = [e["payload"] for e in events if e["event_type"] == "task_result"]
        assert result["execution_status"] == "分析完成", result
        assert [i["analysis_type"] for i in result["instance_results"]] == ["SHRP", "SHRP", "ROUGH"]
        json.dumps(result, allow_nan=False)
    finally:
        service.shutdown()
        pump(lambda events: any(e["event_type"] == "service_closed" for e in events))


def test_sharpness_options_change_only_postprocessing(quality_task, monkeypatch):
    original = LoudnessAnalyzer.compute
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LoudnessAnalyzer, "compute", counted)
    task = replace(quality_task, instances=tuple(
        AnalysisInstance(str(k), "SHRP", 0, {"sharpness": {"k": k}}) for k in (0.11, 0.22)))
    first, second = execute_task(task)["instance_results"]
    assert len(calls) == 1
    np.testing.assert_allclose(np.asarray(first["display_payload"]["y"], dtype=float) * 2,
                               np.asarray(second["display_payload"]["y"], dtype=float), equal_nan=True)


@pytest.mark.parametrize("kind", ["SHRP", "ROUGH"])
def test_nonfinite_input_and_missing_calibration_fail_explicitly(quality_task, kind):
    signal, rate = sf.read(quality_task.wav_path, dtype="float32", always_2d=True)
    signal[100, 0] = np.nan
    sf.write(quality_task.wav_path, signal, rate, subtype="FLOAT")
    task = replace(quality_task, instances=tuple(AnalysisInstance(str(ch), kind, ch, {}) for ch in (0, 1)))
    bad, good = execute_task(task)["instance_results"]
    assert bad["error"]["stage"] == "validation"
    assert good["execution_status"] == "分析完成"
    task = replace(task, channels=(task.channels[0], replace(task.channels[1], v2pa_factor=None)))
    assert execute_task(task)["instance_results"][1]["error"]["stage"] == "calibration"


def test_short_sharpness_keeps_curve_but_unavailable_summary_cannot_pass(quality_task):
    config = {"advanced": {"analysis_time_range_enabled": True,
                           "analysis_start_time_sec": 0, "analysis_end_time_sec": 0.05},
              "limit_checked": True, "limit_metric": "s_mean_acum",
              "scalar_upper_enabled": True, "scalar_upper_value": 100}
    item, = execute_task(replace(quality_task, instances=(AnalysisInstance("sharpness", "SHRP", 0, config),)))["instance_results"]
    assert item["execution_status"] == "分析完成"
    assert item["metrics"]["valid_frame_count"] > 0
    assert item["metrics"]["s_mean_acum"] is None
    assert item["judgement_status"] == "failed"
