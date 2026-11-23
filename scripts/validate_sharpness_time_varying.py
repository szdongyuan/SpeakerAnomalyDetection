"""Offline SHRP cross-check; MoSQITo is a test-only, externally supplied dependency.

Compares project SHRP against independent MoSQITo specific loudness using the
SQAT integral-normalized definition, AND separately records the different
native MoSQITo time-varying definition. Neither comparison is certification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import butter, resample, sosfilt
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def signals():
    fs = 48000
    t = np.arange(3 * fs) / fs
    rng = np.random.default_rng(20261119)
    for hz in (250, 1000, 4000, 8000):
        yield f"tone_{hz}Hz_60dB", np.sqrt(2) * .02 * np.sin(2 * np.pi * hz * t), fs, {}
    tone = np.sqrt(2) * .02 * np.sin(2 * np.pi * 1000 * t)
    yield "am_10Hz_depth90", tone * (1 + .9 * np.sin(2 * np.pi * 10 * t)), fs, {}
    yield "burst_0.5_to_1.5s", tone * ((t >= .5) & (t < 1.5)), fs, {}
    yield "frequency_step_1k_to_4k", np.sqrt(2) * .02 * np.sin(2 * np.pi * np.where(t < 1.5, 1000, 4000) * t), fs, {}
    noise = rng.standard_normal(t.size)
    noise *= .02 / np.sqrt(np.mean(noise ** 2))
    yield "white_noise_60dB", noise, fs, {}
    band = sosfilt(butter(4, [900, 1100], btype="bandpass", fs=fs, output="sos"), noise)
    band *= .02 / np.sqrt(np.mean(band ** 2))
    yield "band_noise_900_1100Hz_60dB", band, fs, {}
    yield "low_level_1k_0dB", tone * .001, fs, {}
    yield "silence", np.zeros_like(t), fs, {}
    yield "short_50ms", tone[:2400], fs, {}
    for fs in (44100, 51200):
        t = np.arange(3 * fs) / fs
        yield f"tone_1k_{fs}Hz_sampling", np.sqrt(2) * .02 * np.sin(2 * np.pi * 1000 * t), fs, {}


def recorded_signals(root):
    from base.wav_calibration_metadata import inspect_wav_calibration_metadata, resolve_wav_channel_v2pa_factor

    count = 0
    seen = set()
    for path in sorted(root.rglob("*.wav")):
        metadata = inspect_wav_calibration_metadata(path)
        if metadata.metadata is None:
            continue
        info = sf.info(path)
        if info.frames < 3 * info.samplerate:
            continue
        factor = resolve_wav_channel_v2pa_factor(metadata.metadata, 0)
        if not factor.used_file_metadata:
            continue
        data, fs = sf.read(path, frames=3 * info.samplerate, dtype="float64", always_2d=True)
        digest = hashlib.sha256(data[:, 0].tobytes() + str((fs, factor.factor)).encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        count += 1
        yield f"stored_{count}", data[:, 0] * factor.factor, fs, {
            "source": str(path.relative_to(ROOT)), "channel": 1, "factor": factor.factor,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if count == 2:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stored-root", type=Path)
    parser.add_argument("--case", action="append", help="Run only these named cases")
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference_path.resolve()))
    import mosqito
    from mosqito.sq_metrics import loudness_zwtv, sharpness_din_from_loudness
    from base.core_algorithm.sound_quality.loudness import LoudnessAnalyzer
    from base.core_algorithm.sound_quality.sharpness import compute_sharpness_din45692
    from base.sound_quality_analysis_adapter import calculate_sharpness

    args.output.mkdir(parents=True, exist_ok=True)
    cases = list(signals())
    if args.stored_root:
        cases.extend(recorded_signals(args.stored_root.resolve()))
    if args.case:
        cases = [case for case in cases if case[0] in args.case]
        if set(args.case) != {case[0] for case in cases}:
            parser.error("A requested case was not found")
    rows = []
    for name, wave, fs, source in cases:
        raw = LoudnessAnalyzer(fs).compute(wave, v2pa_factor=1., method="time_varying_iso532_1")
        # MoSQITo's TV filter bank only accepts 48 kHz. Make resampling explicit.
        ref_wave = wave if fs == 48000 else resample(wave, round(len(wave) * 48000 / fs))
        n, ns, z, t = loudness_zwtv(ref_wave, 48000)
        ours = compute_sharpness_din45692(raw.specific_loudness, raw.bark_axis, raw.time_s)
        product = calculate_sharpness(wave, fs, {}, 1.)
        np.testing.assert_allclose(product["curve"]["y"], ours.sharpness_acum, equal_nan=True)
        # MoSQITo 1.2.1 uses inclusive linspace timestamps. Align by filter
        # output index; the project uses exact 2 ms sample timestamps.
        assert raw.time_s.shape == t.shape
        time_axis_difference = float(np.max(np.abs(raw.time_s - t)))
        t = raw.time_s
        integral = .1 * np.sum(ns, axis=0)
        own_integral = .1 * np.sum(raw.specific_loudness, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            same_definition = sharpness_din_from_loudness(integral, ns)
            same_input = sharpness_din_from_loudness(own_integral, raw.specific_loudness)
            native_tv = sharpness_din_from_loudness(n, ns)
        valid = np.isfinite(ours.sharpness_acum) & (integral >= .1) & (t >= .1)
        core_valid = np.isfinite(ours.sharpness_acum)
        same_input_error = float(np.max(np.abs(ours.sharpness_acum[core_valid] - same_input[core_valid]))) if core_valid.any() else 0.
        delta = np.abs(ours.sharpness_acum[valid] - same_definition[valid])
        passed = same_input_error < 1e-12 and bool(np.all(delta <= np.maximum(.01, .01 * np.abs(same_definition[valid]))))
        expected_empty = name in {"low_level_1k_0dB", "silence", "short_50ms"}
        passed = passed and (not valid.any() and not np.isfinite(ours.summary.s_mean)
                             if expected_empty else valid.any() and np.isfinite(ours.summary.s_mean))
        row = {
            "name": name, "sample_rate": fs, "duration_s": len(wave) / fs, **source,
            "frames": len(t), "valid_summary_frames": int(valid.sum()),
            "reference_time_axis_max_difference_s": time_axis_difference,
            "validity_disagreements": int(np.count_nonzero((own_integral >= .1) != (integral >= .1))),
            "same_input_max_abs_error": same_input_error,
            "reference_definition_pass": bool(passed),
            "project_mean": float(ours.summary.s_mean) if np.isfinite(ours.summary.s_mean) else None,
            "project_max": float(ours.summary.s_max) if np.isfinite(ours.summary.s_max) else None,
            "matched_mean": float(np.mean(ours.sharpness_acum[valid])) if valid.any() else None,
            "reference_integral_mean": float(np.mean(same_definition[valid])) if valid.any() else None,
            "reference_integral_max_abs_error": float(delta.max()) if valid.any() else None,
            "reference_integral_p99_abs_error": float(np.quantile(delta, .99)) if valid.any() else None,
        }
        native_valid = valid & (n >= .1)
        row["native_tv_max_abs_difference"] = (float(np.max(np.abs(ours.sharpness_acum[native_valid] - native_tv[native_valid])))
                                               if native_valid.any() else None)
        np.savez_compressed(args.output / f"{name}.npz", time=t, project=ours.sharpness_acum,
                            reference_integral=same_definition, reference_native_tv=native_tv,
                            comparison_mask=valid, reference_total_loudness=n, reference_integrated_loudness=integral)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)
    result = {"reference_version": mosqito.__version__,
              "criterion": "valid frames t>=0.1s: abs_error <= max(0.01 acum, 1% reference); kernel <1e-12",
              "note": "Engineering cross-check using SQAT's integral denominator; NOT ISO/DIN certification. Low-loudness NaN intentionally differs from MoSQITo zero.",
              "cases": rows}
    (args.output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if any(not row["reference_definition_pass"] for row in rows):
        raise SystemExit("Cross-check tolerance exceeded; inspect results.json")


if __name__ == "__main__":
    main()
