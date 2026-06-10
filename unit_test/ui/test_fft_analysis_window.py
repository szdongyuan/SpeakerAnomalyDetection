import ast
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base.core_algorithm.response.dominant_tone_analyzer import (
    FrequencyInterval,
    find_dominant_fft_peaks,
    parse_frequency_intervals,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SIGNAL_ANALYSIS_PATH = REPO_ROOT / "ui" / "signal_analysis_window.py"


def _load_class_method(class_name: str, method_name: str):
    source = SIGNAL_ANALYSIS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method_node = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    module = ast.Module(body=[method_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "np": np,
        "FrequencyInterval": FrequencyInterval,
        "find_dominant_fft_peaks": find_dominant_fft_peaks,
        "parse_frequency_intervals": parse_frequency_intervals,
    }
    exec(compile(module, str(SIGNAL_ANALYSIS_PATH), "exec"), namespace)
    return namespace[method_name]


def test_fft_delta_display_uses_main_minus_baseline():
    build_curves = _load_class_method("FftAnalysis", "_build_display_curves")
    freq = np.array([100.0, 1000.0])
    spectrum = np.array([30.0, 50.0])
    baseline = np.array([20.0, 45.0])

    result = build_curves(freq, spectrum, baseline, "delta")

    assert np.allclose(result["plot_y"], np.array([10.0, 5.0]))
    assert np.allclose(result["delta_db"], np.array([10.0, 5.0]))
    assert result["baseline_db"] is baseline


def test_fft_smooth_baseline_third_octave_averages_in_fractional_band():
    smooth = _load_class_method("FftAnalysis", "_smooth_baseline_third_octave")
    freq = np.array([100.0, 105.0, 110.0, 130.0, 200.0])
    baseline = np.array([20.0, 20.0, 50.0, 20.0, 20.0])

    smoothed = smooth(freq, baseline)

    assert smoothed[2] < baseline[2]
    assert smoothed[2] > 20.0
    assert np.isclose(smoothed[-1], 20.0)


def test_fft_smooth_baseline_third_octave_matches_naive_reference():
    smooth = _load_class_method("FftAnalysis", "_smooth_baseline_third_octave")
    freq = np.array([0.0, 80.0, 100.0, 110.0, 125.0, 160.0, 250.0])
    baseline = np.array([30.0, 20.0, 25.0, 50.0, np.nan, 35.0, 40.0])
    factor = 2.0 ** (1.0 / 6.0)
    expected = np.full_like(baseline, np.nan, dtype=float)
    for index, center_hz in enumerate(freq):
        if not (np.isfinite(center_hz) and center_hz > 0):
            continue
        mask = (
            np.isfinite(freq)
            & np.isfinite(baseline)
            & (freq >= center_hz / factor)
            & (freq <= center_hz * factor)
        )
        if np.any(mask):
            linear_power = np.power(10.0, baseline[mask] / 10.0)
            expected[index] = 10.0 * np.log10(np.maximum(np.nanmean(linear_power), 1e-30))

    smoothed = smooth(freq, baseline)

    assert np.allclose(smoothed, expected, equal_nan=True)


def test_fft_focus_filters_frequency_range_and_log_zero():
    apply_focus = _load_class_method("FftAnalysis", "_apply_frequency_focus")
    freq = np.array([0.0, 50.0, 100.0, 1000.0, 30000.0])
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    out_freq, out_values = apply_focus(freq, values, True, 100.0, 20000.0, "log")

    assert np.allclose(out_freq, np.array([100.0, 1000.0]))
    assert np.allclose(out_values, np.array([3.0, 4.0]))


def test_fft_dominant_annotation_x_uses_log_coordinate_for_log_axis():
    to_plot_x = _load_class_method("FftAnalysis", "_dominant_annotation_x")

    assert np.isclose(to_plot_x(199.2, "log"), np.log10(199.2))
    assert to_plot_x(199.2, "linear") == 199.2
    assert np.isnan(to_plot_x(0.0, "log"))


def test_fft_detect_dominant_tones_uses_configured_intervals():
    detect = _load_class_method("FftAnalysis", "_detect_dominant_tones")
    freq = np.arange(0, 2001, 10, dtype=float)
    values = np.zeros_like(freq)
    values[np.argmin(np.abs(freq - 300))] = 20.0
    values[np.argmin(np.abs(freq - 1200))] = 30.0
    config = {
        "dominant_tone_enabled": True,
        "dominant_tone_intervals_text": "100, 500, Low\n500, 1800, Mid",
        "dominant_tone_min_prominence_db": 3.0,
    }

    tones = detect(freq, values, config, fallback_low_hz=100, fallback_high_hz=1800)

    assert [tone["interval_label"] for tone in tones] == ["Low", "Mid"]
    assert tones[0]["frequency_hz"] == 300.0
    assert tones[1]["frequency_hz"] == 1200.0


def test_fft_detect_dominant_tones_uses_fallback_range_when_intervals_empty():
    detect = _load_class_method("FftAnalysis", "_detect_dominant_tones")
    freq = np.array([100.0, 300.0, 1000.0])
    values = np.array([1.0, 5.0, 2.0])
    config = {
        "dominant_tone_enabled": True,
        "dominant_tone_intervals_text": "",
        "dominant_tone_min_prominence_db": 3.0,
    }

    tones = detect(freq, values, config, fallback_low_hz=100, fallback_high_hz=1000)

    assert len(tones) == 1
    assert tones[0]["frequency_hz"] == 300.0
