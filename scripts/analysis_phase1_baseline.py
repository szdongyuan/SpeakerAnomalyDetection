"""Capture/replay deterministic analysis baselines without changing production code.

Always write to a new directory. Use --reference to compare a replay with a
previous capture; never regenerate expected results in place. Inputs are
synthetic software fixtures, not measurement or standards reference recordings.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf

from base.analysis.contracts import AnalysisChannel, AnalysisInstance, AnalysisTask
from base.analysis.executor import execute_task
from base.analysis.wav_input import sample_window
from base.analysis_algorithm_adapters import calculate_analysis_instance
from base.audio_label_policy import audio_label_for_results
from base.core_algorithm.sound_quality.loudness import LoudnessMethod
from base.core_algorithm.sound_quality.roughness import compute_roughness_daniel_weber
from base.core_algorithm.sound_quality.service import run_sound_quality
from base.core_algorithm.sound_quality.sharpness import compute_sharpness_din45692


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def source_inventory():
    paths = set()
    for pattern in ("base/analysis/*.py", "base/core_algorithm/response/*.py",
                    "base/core_algorithm/sound_quality/*.py", "base/pre_processing/*.py",
                    "consts/acoustic_analysis/**/*.py"):
        paths.update(ROOT.glob(pattern))
    paths.update(ROOT / name for name in (
        "base/analysis_algorithm_adapters.py", "base/loudness_analysis_adapter.py",
        "base/audio_label_policy.py", "base/analysis_limit_evaluation.py",
        "base/analysis_plot_geometry.py", "base/analysis_display_payload.py",
        "base/analysis_csv_exporter.py", "base/wav_calibration_metadata.py",
        "base/core_algorithm/harmonic_distortion/weighted.py", "base/utils/smooth.py",
    ))
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(paths)}


class Capture:
    def __init__(self, output):
        self.output = output
        self.arrays = {}
        self.cases = {}
        self.timings = {}

    def encode(self, value):
        if is_dataclass(value):
            return self.encode({f.name: getattr(value, f.name) for f in fields(value)})
        if isinstance(value, dict):
            return {str(k): self.encode(v) for k, v in sorted(value.items())}
        if isinstance(value, (tuple, list)):
            try:
                candidate = np.asarray(value)
            except ValueError:
                candidate = None
            if candidate is not None and candidate.size and candidate.dtype.kind in "biuf":
                return self.encode(candidate)
            return [self.encode(v) for v in value]
        if isinstance(value, np.ndarray):
            key = f"array_{len(self.arrays):05d}"
            self.arrays[key] = np.array(value, copy=True)
            return {"array": key, "shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, np.generic):
            return self.encode(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return {"nonfinite": str(value)}
        if isinstance(value, str):
            return value.replace(str(self.output), "$OUTPUT").replace(
                self.output.as_posix(), "$OUTPUT")
        return value

    def run(self, case_id, function, *, expected_exception=None, config=None):
        start = perf_counter()
        try:
            value = function()
        except Exception as error:
            if expected_exception is None or not isinstance(error, expected_exception):
                raise
            value = {"exception": type(error).__name__, "message": str(error)}
        else:
            if expected_exception is not None:
                raise AssertionError(f"{case_id}: expected {expected_exception.__name__}")
        self.timings[case_id] = round((perf_counter() - start) * 1000, 3)
        self.cases[case_id] = self.encode({"config": config or {}, "result": value})
        print(f"captured {case_id}", flush=True)
        return value


def make_inputs(output, reference=None):
    inputs = output / "inputs"
    inputs.mkdir()
    if reference is not None:
        # FLOAT WAV writers add PEAK timestamps. Replay the frozen bytes rather
        # than regenerating nominally identical waveforms with different headers.
        for path in sorted((reference / "inputs").iterdir()):
            if path.is_file():
                shutil.copyfile(path, inputs / path.name)
        return
    for rate in (48000, 44100):
        t = np.arange(3 * rate, dtype=np.float64) / rate
        ramp = np.minimum(t / 0.025, 1.0)
        left = ramp * (0.03 * (1 + 0.6 * np.sin(2 * np.pi * 70 * t))
                       * np.sin(2 * np.pi * 1000 * t) + 0.005 * np.sin(2 * np.pi * 4200 * t))
        right = ramp * (0.018 * (1 + 0.3 * np.sin(2 * np.pi * 35 * t))
                        * np.sin(2 * np.pi * 700 * t) + 0.01 * np.sin(2 * np.pi * 2100 * t))
        sf.write(inputs / f"am_stereo_{rate}.wav", np.column_stack((left, right)), rate,
                 subtype="FLOAT")
    sf.write(inputs / "silence.wav", np.zeros(48000 * 3), 48000, subtype="FLOAT")
    sf.write(inputs / "short.wav", np.ones(4800) * 0.01, 48000, subtype="FLOAT")
    bad = np.zeros(48000)
    bad[123] = np.nan
    sf.write(inputs / "nonfinite.wav", bad, 48000, subtype="FLOAT")
    (inputs / "broken.wav").write_bytes(b"not a WAV")


def task_for(output, case_id, rate, instances, *, filename=None, factors=(2.0, 3.0)):
    return AnalysisTask(
        task_id=case_id, station_id="baseline_station", station_name="Baseline",
        condition_id="baseline_condition", condition_name="Synthetic",
        wav_path=str(output / "inputs" / (filename or f"am_stereo_{rate}.wav")),
        analysis_result_directory=str(output / "exports" / case_id),
        channels=tuple(AnalysisChannel(i, f"CH{i + 1}", "sound_pressure", v2pa_factor=f)
                       for i, f in enumerate(factors)),
        instances=tuple(instances), segmented_analysis={"mode": "none"},
    )


def config_for(kind, window, method):
    threshold = {"limit_checked": True, "limit_mode": "manual",
                 "manual_input_mode": "constant", "constant_upper_enabled": True,
                 "constant_upper_value": 150.0, "constant_lower_enabled": False}
    if kind == "LOUD":
        return {**threshold, "method": method, "advanced": {**window, "curve_y_unit": "sone"}}
    return {**threshold, **window} if kind != "Spec" else {**window}


def loud_service(signal, rate, factor, config):
    settings = deepcopy(config)
    settings["enabled"] = True
    settings["required_summary_metrics"] = [
        "steady_state_average_sone", "steady_state_average_phon",
        "max_transient_sone", "max_transient_phon",
    ]
    return run_sound_quality(signal, rate, factor,
                             {"enabled": True, "items": {"LOUD": settings}}).loudness


def capture_regular_cases(capture):
    output = capture.output
    tv = LoudnessMethod.TIME_VARYING_ISO532_1.value
    profiles = [(48000, False, tv), (48000, True, tv), (44100, True, tv),
                (44100, True, LoudnessMethod.PER_SEGMENT.value)]
    for rate, cropped, method in profiles:
        data, _ = sf.read(output / "inputs" / f"am_stereo_{rate}.wav", dtype="float32",
                          always_2d=True)
        window = {"analysis_time_range_enabled": cropped, "analysis_start_time_sec": 0.251,
                  "analysis_end_time_sec": 2.731}
        first, stop = sample_window(window, len(data), rate)
        for channel in (0, 1):
            key = f"{rate}_{'crop' if cropped else 'full'}_{method}_ch{channel}"
            kinds = ("SPL", "FFT", "FBA", "Spec", "LOUD")
            configs = {k: config_for(k, window, method) for k in kinds}
            # A-weighting exercises filter-before-crop on SPL; other defaults remain explicit in code.
            configs["SPL"]["weighting"] = "A"
            if method == LoudnessMethod.PER_SEGMENT.value:
                configs["LOUD"]["advanced"].update(
                    stationary_frame_duration_s=0.2, stationary_overlap_percent=25.0,
                    curve_y_unit="phon")
            instances = [AnalysisInstance(k, k, channel, configs[k]) for k in kinds]
            task = task_for(output, key, rate, instances)
            result = capture.run("station_" + key, lambda: execute_task(task), config=configs)
            assert all(i["execution_status"] == "分析完成" for i in result["instance_results"])
            assert all(a["status"] == "saved" for i in result["instance_results"] for a in i["artifacts"])
            factor = task.channels[channel].v2pa_factor
            loud = capture.run("service_" + key,
                               lambda: loud_service(data[:, channel], rate, factor, configs["LOUD"]),
                               config=configs["LOUD"])
            assert loud.enabled and loud.raw_result is not None
            raw = loud.raw_result
            expected_y = raw.loudness_level_phon if method == "per_segment" else raw.loudness_sone
            payload = result["instance_results"][-1]["display_payload"]
            np.testing.assert_allclose(payload["y"], expected_y, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(payload["x"], raw.time_s + first / rate, rtol=1e-10)
            capture.run("sharpness_" + key, lambda: compute_sharpness_din45692(
                raw.specific_loudness, raw.bark_axis, raw.time_s, method=LoudnessMethod(method)),
                config={"frontend_case": "service_" + key, "method": method})
            capture.run("roughness_" + key, lambda: compute_roughness_daniel_weber(
                data[first:stop, channel].astype(np.float64) * factor, rate),
                config={"channel": channel, "factor": factor, "first": first, "stop": stop,
                        "overlap": 0.5})


def capture_boundary_cases(capture):
    out = capture.output
    data, rate = sf.read(out / "inputs/am_stereo_48000.wav", dtype="float32", always_2d=True)
    signal = data[:, 0]

    def direct(kind, config, **kwargs):
        return calculate_analysis_instance(kind, signal, rate, config, 2.0,
                                           source="phase1_baseline", sequence_snapshot={}, **kwargs)

    def station(case_id, kind, config, **kwargs):
        task = task_for(out, case_id, rate, [AnalysisInstance(kind, kind, 0, config)], **kwargs)
        return execute_task(task)

    no_overlap = {"limit_checked": True, "limit_mode": "csv",
                  "limit_data": [[100000.0, 200000.0], [0.0, 0.0], [None, None]]}
    for kind in ("FFT", "FBA"):
        legacy_config = deepcopy(no_overlap)
        legacy_config["limit_data"][2] = [np.nan, np.nan]
        legacy = capture.run(f"legacy_no_overlap_{kind}", lambda: direct(kind, legacy_config),
                             config=legacy_config)
        runtime = capture.run(f"station_no_overlap_{kind}",
                              lambda: station("no_overlap_" + kind, kind, no_overlap), config=no_overlap)
        assert legacy["judgement"] == "OK"
        assert runtime["instance_results"][0]["judgement_status"] == "failed"
    for scale in ("linear", "log"):
        config = {"freq_scale_type": scale, "n_fft": 1024, "hop_length": 256}
        capture.run("legacy_spec_" + scale, lambda: direct("Spec", config), config=config)
        capture.run("station_spec_" + scale,
                    lambda: station("spec_" + scale, "Spec", config), config=config)
    for mode in ("overlay", "delta"):
        config = {"baseline_file_path": str(out / "inputs/am_stereo_48000.wav"),
                  "baseline_display_mode": mode, "baseline_smooth_third_octave": True}
        capture.run("fft_reference_" + mode, lambda: direct("FFT", config), config=config)
    capture.run("fft_missing_reference", lambda: direct("FFT", {
        "baseline_display_mode": "delta"}), expected_exception=ValueError)
    base_loud = capture.run("loud_default", lambda: loud_service(signal, rate, 2.0, {}))
    explicit = capture.run("loud_explicit_default", lambda: loud_service(signal, rate, 2.0, {
        "method": LoudnessMethod.TIME_VARYING_ISO532_1.value,
        "advanced": {"output_time_resolution_ms": 2.0}}))
    np.testing.assert_array_equal(base_loud.raw_result.specific_loudness,
                                  explicit.raw_result.specific_loudness)
    presentation_config = {
        "display": {"curves": [], "summary_metrics": []},
        "advanced": {"curve_y_unit": "phon", "specific_loudness_profile_mode": "max_loudness",
                     "specific_loudness_time_skip_s": 0.2},
    }
    presentation = capture.run("loud_presentation_only",
                               lambda: loud_service(signal, rate, 2.0, presentation_config),
                               config=presentation_config)
    np.testing.assert_array_equal(base_loud.raw_result.specific_loudness,
                                  presentation.raw_result.specific_loudness)
    resolution_config = {"advanced": {"output_time_resolution_ms": 4.0}}
    resolution = capture.run("loud_resolution_4ms",
                             lambda: loud_service(signal, rate, 2.0, resolution_config),
                             config=resolution_config)
    np.testing.assert_array_equal(base_loud.raw_result.specific_loudness[:, ::2],
                                  resolution.raw_result.specific_loudness)
    for name, advanced in (
        ("missing", {"background_noise_file_path": str(out / "inputs/missing.wav")}),
        ("broken", {"background_noise_file_path": str(out / "inputs/broken.wav")}),
        ("valid", {"background_noise_file_path": str(out / "inputs/am_stereo_44100.wav")}),
    ):
        config = {"advanced": {"background_noise_processing_enabled": True, **advanced}}
        result = capture.run("loud_noise_" + name, lambda: loud_service(signal, rate, 2.0, config),
                             config=config)
        assert result.enabled
        if name != "valid":
            np.testing.assert_array_equal(result.raw_result.specific_loudness,
                                          base_loud.raw_result.specific_loudness)
        else:
            assert not np.array_equal(result.raw_result.specific_loudness,
                                      base_loud.raw_result.specific_loudness)
    capture.run("loud_invalid_method_service", lambda: loud_service(signal, rate, 2.0,
                                                                   {"method": "invalid"}))
    capture.run("loud_invalid_method_adapter", lambda: direct("LOUD", {"method": "invalid"}),
                expected_exception=ValueError)
    range_config = {"advanced": {"analysis_time_range_enabled": True,
                                "analysis_start_time_sec": 0.1, "analysis_end_time_sec": 4.0}}
    capture.run("loud_overrun_service", lambda: loud_service(signal, rate, 2.0, range_config),
                config=range_config)
    capture.run("loud_overrun_station", lambda: station("overrun", "LOUD", range_config),
                config=range_config)
    for filename in ("silence.wav", "short.wav", "nonfinite.wav"):
        items = [AnalysisInstance(k, k, 0, {}) for k in ("SPL", "FFT", "FBA", "Spec", "LOUD")]
        task = task_for(out, "boundary_" + filename, rate, items, filename=filename, factors=(2.0,))
        capture.run("station_" + filename, lambda: execute_task(task))
        wave, _ = sf.read(task.wav_path, dtype="float32")
        capture.run("roughness_" + filename,
                    lambda: compute_roughness_daniel_weber(wave.astype(np.float64) * 2.0, rate),
                    expected_exception=ValueError if filename != "silence.wav" else None)
        if filename == "silence.wav":
            raw = loud_service(wave, rate, 2.0, {}).raw_result
            sharp = capture.run("sharpness_silence", lambda: compute_sharpness_din45692(
                raw.specific_loudness, raw.bark_axis, raw.time_s))
            assert np.isnan(sharp.sharpness_acum).all()
    capture.run("station_no_calibration", lambda: station("no_calibration", "LOUD", {},
                                                          factors=(None, 3.0)))
    for overlap in (0.0, 0.25, 0.75):
        capture.run(f"roughness_hop_{overlap}", lambda: compute_roughness_daniel_weber(
            signal.astype(np.float64) * 2.0, rate, overlap=overlap), config={"overlap": overlap})
    for kind in ("SHRP", "ROUGH"):
        capture.run("station_unsupported_" + kind,
                    lambda: station("unsupported_" + kind, kind, {}))
    good = {"limit_checked": True, "limit_metric": "overall_spl",
            "scalar_upper_enabled": True, "scalar_upper_value": 150.0}
    task = task_for(out, "partial_failure", rate, (
        AnalysisInstance("good", "SPL", 0, good),
        AnalysisInstance("bad", "LOUD", 0, {"method": "invalid", "limit_checked": True}),
    ))
    result = capture.run("partial_failure", lambda: execute_task(task))
    label = capture.run("partial_failure_audio_label",
                        lambda: audio_label_for_results(result["instance_results"]))
    assert result["execution_status"] == "结果不完整" and label == "not_labeled"


def compare(reference, output, rtol, atol):
    expected = json.loads((reference / "snapshots.json").read_text(encoding="utf-8"))
    actual = json.loads((output / "snapshots.json").read_text(encoding="utf-8"))
    checked = 0
    with np.load(reference / "arrays.npz", allow_pickle=False) as old, np.load(
            output / "arrays.npz", allow_pickle=False) as new:
        def walk(a, b, path):
            nonlocal checked
            if isinstance(a, dict) and set(a) == {"array", "shape", "dtype"}:
                assert a["shape"] == b["shape"] and a["dtype"] == b["dtype"], path
                np.testing.assert_allclose(new[b["array"]], old[a["array"]],
                                           rtol=rtol, atol=atol, equal_nan=True, err_msg=path)
                checked += 1
            elif isinstance(a, dict):
                assert a.keys() == b.keys(), path
                for key in a:
                    walk(a[key], b[key], path + "/" + key)
            elif isinstance(a, list):
                assert len(a) == len(b), path
                for i, (x, y) in enumerate(zip(a, b)):
                    walk(x, y, path + f"/{i}")
            elif isinstance(a, float):
                assert math.isclose(a, b, rel_tol=rtol, abs_tol=atol), (path, a, b)
            else:
                assert a == b, (path, a, b)
        walk(expected, actual, "cases")
    return {"cases": len(expected), "arrays_compared": checked, "rtol": rtol, "atol": atol,
            "status": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(filename=output / "execution.log", encoding="utf-8", level=logging.WARNING)
    before = source_inventory()
    make_inputs(output, args.reference.resolve() if args.reference else None)
    capture = Capture(output)
    capture_regular_cases(capture)
    capture_boundary_cases(capture)
    after = source_inventory()
    assert before == after, "Analysis sources changed during capture; use a stable source tree."
    np.savez_compressed(output / "arrays.npz", **capture.arrays)
    write_json(output / "snapshots.json", capture.cases)
    write_json(output / "timings_ms.json", capture.timings)
    manifest = {
        "scope": "Synthetic software regression baseline; no real recordings or standards certification",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "scipy", "soundfile", "librosa", "numba", "pytest")},
        "numba_cache_dir": os.environ.get("NUMBA_CACHE_DIR"),
        "script_sha256": sha256(Path(__file__)), "source_sha256": before,
        "input_sha256": {p.name: sha256(p) for p in sorted((output / "inputs").iterdir())},
        "artifact_sha256": {name: sha256(output / name) for name in ("snapshots.json", "arrays.npz")},
        "export_sha256": {p.relative_to(output).as_posix(): sha256(p)
                          for p in sorted((output / "exports").rglob("*.csv"))},
        "case_count": len(capture.cases), "array_count": len(capture.arrays),
        "source_stable_during_capture": True,
    }
    write_json(output / "manifest.json", manifest)
    if args.reference:
        result = compare(args.reference.resolve(), output, args.rtol, args.atol)
        old = json.loads((args.reference / "manifest.json").read_text(encoding="utf-8"))
        result["inputs_identical"] = old["input_sha256"] == manifest["input_sha256"]
        result["sources_identical"] = old["source_sha256"] == before
        result["exports_identical"] = old["export_sha256"] == manifest["export_sha256"]
        assert result["inputs_identical"], "Input fixtures changed; do not accept the comparison."
        assert result["exports_identical"], "CSV content changed; inspect independently of numerical tolerance."
        write_json(output / "comparison.json", result)
        print(json.dumps(result), flush=True)
    print(f"Saved {len(capture.cases)} cases and {len(capture.arrays)} arrays to {output}", flush=True)


if __name__ == "__main__":
    main()
