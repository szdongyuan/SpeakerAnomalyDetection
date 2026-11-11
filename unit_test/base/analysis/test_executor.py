"""Real WAV acceptance for the configuration-driven, process-free executor."""

import csv
import hashlib
import json
import pickle
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from base.analysis.contracts import AnalysisChannel, AnalysisInstance, AnalysisTask
from base.analysis.executor import execute_task


def assert_curve_csv(item):
    artifact, = [a for a in item["artifacts"] if a["kind"] == "curve_csv"]
    assert artifact["status"] == "saved", artifact
    with open(artifact["path"], encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = list(reader)
    curve = item["display_payload"]
    unit = {"dBA": "dB(A)", "dBC": "dB(C)"}.get(curve["y_unit"], curve["y_unit"])
    x_name, y_name = {
        "SPL": ("时间(s)", "声压级"),
        "FFT": ("频率(Hz)", "幅值"),
        "FBA": ("中心频率(Hz)", "频段声压级"),
    }[item["analysis_type"]]
    assert header == [x_name, f"{y_name}{unit}", f"下限{unit}", f"上限{unit}"]
    assert len(rows) == len(curve["x"])
    np.testing.assert_allclose([float(r[0]) for r in rows], curve["x"], rtol=1e-9)
    for column, key in ((1, "y"), (2, "lower"), (3, "upper")):
        expected = curve[key] or [None] * len(rows)
        for row, value in zip(rows, expected):
            if value is None:
                assert row[column] == ""
            else:
                assert float(row[column]) == pytest.approx(value, rel=1e-9)


def test_overall_save_failure_still_saves_complete_curve(task, monkeypatch):
    from base.analysis import executor

    def fail(*args):
        raise PermissionError("overall CSV is locked")

    monkeypatch.setattr(executor, "save_overall_spl", fail)
    item = execute_task(task)["instance_results"][0]
    assert item["execution_status"] == "分析完成"
    assert item["artifacts"][0]["status"] == "failed"
    assert_curve_csv(item)


@pytest.fixture
def task(tmp_path):
    rate = 44100
    time = np.arange(rate) / rate
    tone = 0.1 * np.sin(2 * np.pi * 1000 * time)
    path = tmp_path / "input.wav"
    sf.write(path, np.column_stack((tone, 2 * tone)), rate, subtype="FLOAT")
    return AnalysisTask(
        "task-1", "station-1", "工位1", "condition-1", "工况1",
        str(path), str(tmp_path / "results"),
        (AnalysisChannel(0, "CH1", "sound_pressure", v2pa_factor=2.0),
         AnalysisChannel(1, "CH2", "sound_pressure", v2pa_factor=2.0)),
        (AnalysisInstance("声压级", "SPL", 0, {"weighting": "Z"}),),
        {"mode": "none"},
    )


def test_special_item_names_save_independently_and_keep_stable_paths(task):
    fallback_name = "分析项_" + hashlib.sha256(b"/").hexdigest()[:12]
    names = ("SPL 1", "/", "\\", "...", "SPL/a", "SPL\\a", "spl_a", fallback_name)
    instances = tuple(replace(task.instances[0], config_key=name) for name in names)
    current = replace(task, instances=instances)
    result = execute_task(current)
    assert [item["config_key"] for item in result["instance_results"]] == list(names)
    paths = {}
    for item in result["instance_results"]:
        assert item["execution_status"] == "分析完成"
        assert_curve_csv(item)
        for artifact in item["artifacts"]:
            assert artifact["status"] == "saved", artifact
            path = Path(artifact["path"])
            assert path.is_file()
            assert path.parent == Path(task.analysis_result_directory) / "csv" / "input"
            paths[item["config_key"], artifact["kind"]] = str(path)
    assert len({path.casefold() for path in paths.values()}) == 2 * len(names)
    assert Path(paths["SPL 1", "overall_spl_csv"]).name == "SPL_1_CH1_总体值.csv"
    assert Path(paths["/", "overall_spl_csv"]).name.startswith("分析项_")

    repeated = execute_task(replace(current, task_id="task-2", instances=instances[::-1]))
    for item in repeated["instance_results"]:
        for artifact in item["artifacts"]:
            assert artifact["status"] == "saved", artifact
            assert artifact["path"] == paths[item["config_key"], artifact["kind"]]
    assert tuple(instance.config_key for instance in current.instances) == names
    assert len(list(Path(task.analysis_result_directory).rglob("*.csv"))) == len(paths)


def test_each_item_channel_has_own_csv_and_wav_is_read_once(task, monkeypatch):
    original = sf.read
    calls = []

    def read(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(sf, "read", read)
    items = tuple(AnalysisInstance(name, "SPL", channel, {"weighting": "Z"})
                  for name in ("SPL/a", "SPL\\a") for channel in (0, 1))
    result = execute_task(replace(task, instances=items))
    assert calls == [task.wav_path]
    assert result["execution_status"] == "分析完成"
    assert len(result["instance_results"]) == 4
    paths = []
    expected = 20 * np.log10(0.2 / np.sqrt(2) / 20e-6)
    for item, configured in zip(result["instance_results"], items):
        assert item["judgement_status"] == "not_enabled"
        assert item["judgement"] is None and item["violations"] is None
        assert item["metrics"]["overall_spl"] == pytest.approx(
            expected + configured.channel_index * 20 * np.log10(2), abs=1e-5)
        artifact = item["artifacts"][0]
        assert artifact["status"] == "saved"
        path = Path(artifact["path"])
        paths.extend(Path(a["path"]) for a in item["artifacts"])
        assert [a["kind"] for a in item["artifacts"]] == ["overall_spl_csv", "curve_csv"]
        assert_curve_csv(item)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1
        assert list(rows[0]) == ["总体声压级dB", "下限dB", "上限dB", "判定结果"]
        assert f"_CH{configured.channel_index + 1}_" in path.name
        assert rows[0]["判定结果"] == "未启用"
        assert rows[0]["下限dB"] == rows[0]["上限dB"] == ""
        assert float(rows[0]["总体声压级dB"]) == pytest.approx(item["metrics"]["overall_spl"])
    assert len(set(paths)) == 8
    assert set(Path(task.analysis_result_directory).rglob("*.csv")) == set(paths)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("weighting,unit", [("A", "dB(A)"), ("C", "dB(C)"), ("Z", "dB")])
@pytest.mark.parametrize("upper,expected_judgement", [(100, "OK"), (60, "NG")])
def test_compact_csv_preserves_weighting_scalar_limits_and_judgement(
    task, weighting, unit, upper, expected_judgement,
):
    config = {
        "weighting": weighting, "limit_checked": True, "limit_metric": "overall_spl",
        "scalar_upper_enabled": True, "scalar_upper_value": upper,
        "scalar_lower_enabled": False, "scalar_lower_value": 90,
    }
    item = execute_task(replace(task, instances=(replace(task.instances[0], parameters=config),)))["instance_results"][0]
    with open(item["artifacts"][0]["path"], encoding="utf-8-sig", newline="") as stream:
        header, row = list(csv.reader(stream))
    assert header == [f"总体声压级{unit}", f"下限{unit}", f"上限{unit}", "判定结果"]
    assert float(row[0]) == pytest.approx(item["metrics"]["overall_spl"])
    assert row[1] == ""
    assert float(row[2]) == upper
    assert row[3] == expected_judgement == item["judgement"]
    assert item["display_payload"]["upper"] == []
    assert_curve_csv(item)


def test_overall_csv_distinguishes_failed_judgement(task):
    config = {"weighting": "A", "limit_checked": True, "limit_metric": "overall_spl",
              "scalar_lower_enabled": True, "scalar_lower_value": 100,
              "scalar_upper_enabled": True, "scalar_upper_value": 50}
    item = execute_task(replace(task, instances=(replace(task.instances[0], parameters=config),)))["instance_results"][0]
    assert item["judgement_status"] == "failed"
    with open(item["artifacts"][0]["path"], encoding="utf-8-sig", newline="") as stream:
        row, = csv.DictReader(stream)
    assert row["判定结果"] == "判定失败"


def test_result_root_and_wav_stem_control_csv_layout(task, tmp_path):
    wav_directory = tmp_path / "recordings" / "WAV"
    wav_directory.mkdir(parents=True)
    result_root = tmp_path / "separate-results"
    originals = {}
    for name in ("工位1.正转", "工位2.正转"):
        wav = wav_directory / f"{name}.wav"
        wav.write_bytes(Path(task.wav_path).read_bytes())
        originals[wav] = wav.read_bytes()
        current = replace(task, task_id=name, wav_path=str(wav),
                          analysis_result_directory=str(result_root))
        result = execute_task(current)
        item, = result["instance_results"]
        assert result["execution_status"] == "分析完成"
        paths = {Path(a["path"]) for a in item["artifacts"]}
        expected = result_root / "csv" / name
        assert paths == {
            expected / "声压级_CH1_总体值.csv", expected / "声压级_CH1_曲线.csv",
        }
        assert all(path.is_file() for path in paths)
        assert_curve_csv(item)
    assert len(list(result_root.rglob("*.csv"))) == 4
    assert not list(wav_directory.rglob("*.csv"))
    assert all(wav.read_bytes() == content for wav, content in originals.items())


def test_config_dispatches_fft_fba_and_full_calibrated_spectrogram(task):
    instances = (
        AnalysisInstance("FFT", "FFT", 0, {"n_fft": 4096, "weighting": "Z"}),
        AnalysisInstance("FBA", "FBA", 0, {"weighting": "Z"}),
        AnalysisInstance("Spec", "Spec", 0, {"n_fft": 256, "hop_length": 16}),
    )
    result = execute_task(replace(task, instances=instances))
    assert result["execution_status"] == "分析完成"
    fft, fba, spec = result["instance_results"]
    assert fft["metrics"]["peak_frequency_hz"] == pytest.approx(1000, abs=12)
    assert len(fba["display_payload"]["band_labels"]) > 0
    display = spec["display_payload"]
    assert display["matrix_axes"] == ["frequency", "time"]
    assert display["reduction"] == "none"
    assert len(display["time_s"]) > 2000
    assert np.shape(display["values_db"]) == (129, len(display["time_s"]))
    assert_curve_csv(fft)
    assert_curve_csv(fba)
    assert spec["artifacts"] == []
    assert len(list(Path(task.analysis_result_directory).rglob("*.csv"))) == 2
    other = execute_task(replace(task, instances=(instances[2],), channels=(
        replace(task.channels[0], v2pa_factor=1.0), task.channels[1])))
    difference = np.asarray(display["values_db"]) - np.asarray(
        other["instance_results"][0]["display_payload"]["values_db"])
    # Compare the tone bin; the existing dB floor is intentionally retained.
    np.testing.assert_allclose(difference[6], 20 * np.log10(2), atol=2e-5)
    json.dumps(result, allow_nan=False)


def test_failures_are_isolated_and_judgement_failure_retains_calculation(task):
    items = (
        AnalysisInstance("unknown", "AI", 0, {}),
        AnalysisInstance("bad threshold", "SPL", 0, {"limit_checked": True}),
        AnalysisInstance("good", "SPL", 1, {"weighting": "Z"}),
    )
    result = execute_task(replace(task, instances=items))
    assert result["execution_status"] == "结果不完整"
    unknown, bad, good = result["instance_results"]
    assert unknown["error"]["type"] == "UnsupportedAnalysisType"
    assert bad["execution_status"] == "分析完成"
    assert bad["judgement_status"] == "failed"
    assert bad["judgement"] is None and bad["violations"] is None
    assert bad["metrics"]["overall_spl"] is not None
    assert bad["artifacts"][0]["status"] == "saved"
    assert good["error"] is None


def test_ng_is_completed_and_scalar_violation_has_no_curve_coordinates(task):
    params = {"weighting": "Z", "limit_checked": True, "limit_metric": "overall_spl",
              "scalar_upper_enabled": True, "scalar_upper_value": 70.0,
              "scalar_lower_enabled": False}
    result = execute_task(replace(task, instances=(replace(task.instances[0], parameters=params),)))
    assert result["execution_status"] == "分析完成"
    item = result["instance_results"][0]
    assert item["judgement"] == "NG"
    violation, = item["violations"]
    assert violation["x_start"] is None and violation["max_deviation_at"] is None
    assert violation["side"] == "above_upper"
    assert violation["max_deviation"] == pytest.approx(item["metrics"]["overall_spl"] - 70)


@pytest.mark.parametrize("directory", [None, "blocked"])
def test_save_failure_does_not_destroy_result(task, directory, tmp_path):
    if directory:
        path = tmp_path / directory
        path.write_text("not a directory")
        directory = str(path)
    result = execute_task(replace(task, analysis_result_directory=directory))
    assert result["execution_status"] == "分析完成"
    item = result["instance_results"][0]
    assert item["error"] is None
    assert item["artifacts"][0]["status"] == "failed"
    assert item["artifacts"][0]["error"]["stage"] == "saving"


def test_missing_calibration_does_not_assume_one(task):
    task = replace(task, channels=(replace(task.channels[0], v2pa_factor=None), task.channels[1]))
    result = execute_task(task)
    assert result["execution_status"] == "分析失败"
    assert result["instance_results"][0]["error"]["stage"] == "calibration"


def test_unsupported_signal_is_not_acoustic_fft(task):
    task = replace(task, channels=(replace(task.channels[0], signal_type="acceleration"), task.channels[1]))
    result = execute_task(task)
    assert result["instance_results"][0]["error"]["type"] == "UnsupportedSignalType"


def test_empty_and_broken_wav_task_errors(task):
    missing = replace(task, wav_path=task.wav_path + ".missing")
    assert execute_task(missing)["error"]["stage"] == "wav_read"
    empty = execute_task(replace(missing, instances=()))
    assert empty["execution_status"] == "分析完成"
    assert empty["instance_results"] == []
    assert empty["error"] is None


def test_time_window_uses_wav_and_does_not_fallback_to_whole_recording(task):
    params = {"weighting": "Z", "analysis_time_range_enabled": True,
              "analysis_start_time_sec": 0.25, "analysis_end_time_sec": 0.5}
    instance = replace(task.instances[0], parameters=params)
    result = execute_task(replace(task, instances=(instance,)))["instance_results"][0]
    assert result["metrics"]["analysis_start_time_sec"] == pytest.approx(0.25, abs=1/44100)
    assert result["metrics"]["analysis_end_time_sec"] == pytest.approx(0.5)
    assert min(result["display_payload"]["x"]) >= 0.25
    assert max(result["display_payload"]["x"]) <= 0.5
    assert_curve_csv(result)
    params = dict(params, analysis_end_time_sec=2.0)
    bad = execute_task(replace(task, instances=(replace(instance, parameters=params),)))
    assert bad["execution_status"] == "分析失败"
    assert bad["instance_results"][0]["error"]["stage"] == "validation"


@pytest.mark.parametrize("supplied, expected_source", [(None, "wav"), (2.0, "task_and_wav")])
def test_pcm24_file_calibration_is_used_once(task, supplied, expected_source):
    from base.wav_calibration_metadata import append_wav_calibration_metadata

    samples, rate = sf.read(task.wav_path, always_2d=True)
    sf.write(task.wav_path, samples, rate, subtype="PCM_24")
    assert append_wav_calibration_metadata(task.wav_path, {"recorded_channels": [
        {"wav_channel_index": i, "physical_input_channel": 7 - i,
         "calibrated": True, "v2pa_factor": 2.0, "standard_spl": 94.0}
        for i in (0, 1)
    ]})
    task = replace(task, channels=(replace(task.channels[0], v2pa_factor=supplied), task.channels[1]))
    result = execute_task(task)["instance_results"][0]
    assert result["execution_status"] == "分析完成"
    assert result["metrics"]["calibration_source"] == expected_source
    assert result["metrics"]["overall_spl"] == pytest.approx(
        20 * np.log10(0.2 / np.sqrt(2) / 20e-6), abs=1e-4)


def test_conflicting_calibration_is_not_silently_overridden(task):
    from base.wav_calibration_metadata import append_wav_calibration_metadata

    assert append_wav_calibration_metadata(task.wav_path, {"recorded_channels": [
        {"wav_channel_index": 0, "calibrated": True, "v2pa_factor": 10.0,
         "standard_spl": 94.0}
    ]})
    result = execute_task(task)["instance_results"][0]
    assert result["error"]["stage"] == "calibration"
    assert "不一致" in result["error"]["message"]
    assert result["artifacts"] == []


def test_ve_metadata_uses_wav_column_not_physical_channel(task):
    from base.wav_calibration_metadata import append_wav_calibration_metadata
    from unit_test.base.vkinging_fakes import wav_metadata

    metadata = wav_metadata(("measured", "measured"), sample_rate=44100)
    # The VE fake uses physical indexes 7 and 1, whereas WAV indexes are 0/1.
    assert append_wav_calibration_metadata(task.wav_path, metadata)
    task = replace(task, channels=tuple(replace(c, v2pa_factor=None) for c in task.channels))
    result = execute_task(task)["instance_results"][0]
    assert result["execution_status"] == "分析完成", result
    assert result["metrics"]["v2pa_factor"] == 10.0


def test_malformed_metadata_does_not_fallback_to_task_factor(task):
    from base.wav_calibration_metadata import append_wav_calibration_metadata

    assert append_wav_calibration_metadata(task.wav_path, {"recorded_channels": [
        {"wav_channel_index": 0, "calibrated": True, "v2pa_factor": 2.0,
         "standard_spl": 94.0}
    ]})
    path = Path(task.wav_path)
    raw = path.read_bytes()
    assert b'mic_calibration={' in raw
    path.write_bytes(raw.replace(b'mic_calibration={', b'mic_calibration=!', 1))
    result = execute_task(task)["instance_results"][0]
    assert result["error"]["stage"] == "calibration"
    assert "损坏" in result["error"]["message"]


def test_curve_judgement_is_not_written_as_overall_judgement(task):
    config = {"limit_checked": True, "limit_metric": "curve_y", "limit_mode": "manual",
              "manual_input_mode": "constant", "constant_upper_enabled": True,
              "constant_upper_value": 200, "constant_lower_enabled": False}
    result = execute_task(replace(task, instances=(replace(task.instances[0], parameters=config),)))
    item = result["instance_results"][0]
    assert item["judgement"] == "OK" and item["violations"] == []
    with open(item["artifacts"][0]["path"], encoding="utf-8-sig", newline="") as stream:
        row, = csv.DictReader(stream)
    assert row["判定结果"] == "未启用"
    assert row["下限dB"] == row["上限dB"] == ""
    assert_curve_csv(item)


def test_channel_count_and_selected_nonfinite_samples(task):
    samples, rate = sf.read(task.wav_path, always_2d=True)
    sf.write(task.wav_path, samples[:, 0], rate, subtype="FLOAT")
    mismatch = execute_task(task)
    assert mismatch["error"]["stage"] == "wav_read"
    samples[0, 0] = np.nan
    sf.write(task.wav_path, samples, rate, subtype="FLOAT")
    task = replace(task, instances=(task.instances[0], replace(task.instances[0], channel_index=1)))
    result = execute_task(task)
    assert result["execution_status"] == "结果不完整"
    assert result["instance_results"][0]["error"]["stage"] == "validation"
    assert result["instance_results"][1]["execution_status"] == "分析完成"


def test_same_wav_reanalysis_updates_same_files_and_atomic_failure_preserves_output(task, monkeypatch):
    from base import csv_writer

    first = execute_task(task)["instance_results"][0]["artifacts"][0]["path"]
    previous = Path(first).read_bytes()
    changed = replace(task.instances[0], channel_index=0, parameters={"weighting": "A"})
    item = execute_task(replace(task, task_id="task-2", instances=(changed,)))["instance_results"][0]
    second = item["artifacts"][0]["path"]
    assert first == second
    assert Path(second).name == "声压级_CH1_总体值.csv"
    with open(second, encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
        assert float(row["总体声压级dB(A)"]) == pytest.approx(item["metrics"]["overall_spl"])
    assert Path(second).read_bytes() != previous
    previous = Path(second).read_bytes()

    def cannot_replace(*args):
        raise PermissionError("blocked final rename")

    monkeypatch.setattr(csv_writer.os, "replace", cannot_replace)
    result = execute_task(task)["instance_results"][0]
    assert result["artifacts"][0]["status"] == "failed"
    assert Path(first).read_bytes() == previous
    assert len(list(Path(first).parent.iterdir())) == 2


def test_short_wav_and_segmentation_boundary(task):
    sf.write(task.wav_path, np.zeros((10, 2)), 44100, subtype="FLOAT")
    task = replace(task, instances=(task.instances[0], AnalysisInstance("FFT", "FFT", 1, {})))
    result = execute_task(task)
    assert result["instance_results"][0]["execution_status"] == "分析完成"
    assert result["instance_results"][1]["error"]["stage"] == "calculation"
    segmented = execute_task(replace(task, segmented_analysis={"mode": "time"}))
    assert segmented["error"]["type"] == "UnsupportedSegmentation"


def test_all_algorithms_execute_without_ui_or_plotting_imports(task, tmp_path):
    instances = tuple(AnalysisInstance(kind, kind, 0, {"weighting": "A"})
                      for kind in ("SPL", "FFT", "FBA", "Spec"))
    instances += (AnalysisInstance("CQT", "Spec", 0, {"freq_scale_type": "log"}),)
    serialized = tmp_path / "task.pickle"
    serialized.write_bytes(pickle.dumps(replace(task, instances=instances)))
    script = """
import pickle, sys
from base.analysis.executor import execute_task
with open(sys.argv[1], 'rb') as stream:
    task = pickle.load(stream)
result = execute_task(task)
assert result['execution_status'] == '分析完成', result
for prefix in ('ui', 'PyQt5', 'PySide6', 'matplotlib.pyplot', 'base.plot_manager',
               'base.pre_processing.audio_thd_frequency_response_analysis'):
    assert not any(k == prefix or k.startswith(prefix + '.') for k in sys.modules), prefix
"""
    process = subprocess.run([sys.executable, "-c", script, str(serialized)],
                             capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stderr


def test_controller_batch_executes_without_profiles_or_config_lookup(batch_payload, tmp_path):
    from base.analysis import build_analysis_batch

    rate = 44100
    tone = 0.1 * np.sin(2 * np.pi * 1000 * np.arange(rate) / rate)
    wav = tmp_path / "four.wav"
    sf.write(wav, np.column_stack([tone] * 4), rate, subtype="FLOAT")
    for station in batch_payload["stations"]:
        station["channels"][0]["v2pa_factor"] = 2.0
    for entry in batch_payload["tasks"]:
        task_wav = wav.with_name(entry["task_id"] + ".wav")
        task_wav.write_bytes(wav.read_bytes())
        entry["wav_path"] = str(task_wav)
        entry["analysis_result_directory"] = str(tmp_path / "results")
    batch = build_analysis_batch(batch_payload)
    results = [execute_task(task) for task in batch.tasks]
    assert len(results) == 8
    assert all(result["execution_status"] == "结果不完整" for result in results)
    for result in results:
        assert len(result["instance_results"]) == 5
        assert [r["execution_status"] for r in result["instance_results"]] == [
            "分析完成", "分析完成", "分析失败", "分析失败", "分析失败"]
    assert len(list((tmp_path / "results").rglob("*.csv"))) == 24


def test_spl_weights_before_time_crop_and_zero_end_means_wav_end(task):
    from base.core_algorithm.harmonic_distortion.weighted import apply_weighting_filter

    samples, rate = sf.read(task.wav_path, dtype="float32", always_2d=True)
    params = {"weighting": "A", "analysis_time_range_enabled": True,
              "analysis_start_time_sec": 0.2, "analysis_end_time_sec": 0.0}
    task = replace(task, instances=(replace(task.instances[0], parameters=params),))
    item = execute_task(task)["instance_results"][0]
    weighted = apply_weighting_filter(samples[:, 0], rate, weighting="A", zero_phase=False)
    expected = 20 * np.log10(np.sqrt(np.mean((weighted[int(0.2 * rate):] * 2)**2)) / 20e-6)
    assert item["metrics"]["overall_spl"] == pytest.approx(expected)
    assert item["metrics"]["analysis_end_time_sec"] == 1.0
    assert task.instances[0].parameters == params


def test_spec_time_axis_has_wav_offset_and_color_limits_are_not_judgement(task):
    config = {"analysis_time_range_enabled": True, "analysis_start_time_sec": 0.25,
              "analysis_end_time_sec": 0.5, "custom_limit": True,
              "bottom_limit": 20.0, "top_limit": 30.0}
    spec = AnalysisInstance("Spec", "Spec", 0, config)
    item = execute_task(replace(task, instances=(spec,)))["instance_results"][0]
    assert item["judgement_status"] == "not_enabled"
    assert item["display_payload"]["time_s"][0] == 0.25
    assert item["metrics"]["analysis_end_time_sec"] == 0.5
    spec = replace(spec, parameters=dict(config, limit_checked=True))
    item = execute_task(replace(task, instances=(spec,)))["instance_results"][0]
    assert item["execution_status"] == "分析完成"
    assert item["judgement_status"] == "failed"
    assert item["display_payload"]["values_db"]
