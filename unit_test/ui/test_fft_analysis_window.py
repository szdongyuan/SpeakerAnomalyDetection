import ast
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SIGNAL_ANALYSIS_PATH = REPO_ROOT / "ui" / "signal_analysis_window.py"


def _load_class_method(class_name: str, method_name: str):
    source = SIGNAL_ANALYSIS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method_node = next(
        node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    module = ast.Module(body=[method_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, str(SIGNAL_ANALYSIS_PATH), "exec"), namespace)
    return namespace[method_name]


def test_fft_delta_display_uses_main_minus_baseline():
    build_curves = _load_class_method("FftAnalysis", "_build_display_curves")
    frequency = np.array([100.0, 1000.0])
    spectrum = np.array([30.0, 50.0])
    baseline = np.array([20.0, 45.0])

    result = build_curves(frequency, spectrum, baseline, "delta")

    assert np.allclose(result["plot_y"], np.array([10.0, 5.0]))
    assert np.allclose(result["delta_db"], np.array([10.0, 5.0]))
    assert result["baseline_db"] is baseline


def test_fft_smooth_baseline_third_octave_matches_naive_reference():
    smooth = _load_class_method("FftAnalysis", "_smooth_baseline_third_octave")
    frequency = np.array([0.0, 80.0, 100.0, 110.0, 125.0, 160.0, 250.0])
    baseline = np.array([30.0, 20.0, 25.0, 50.0, np.nan, 35.0, 40.0])
    factor = 2.0 ** (1.0 / 6.0)
    expected = np.full_like(baseline, np.nan, dtype=float)
    for index, center_hz in enumerate(frequency):
        if not (np.isfinite(center_hz) and center_hz > 0):
            continue
        mask = (
            np.isfinite(frequency)
            & np.isfinite(baseline)
            & (frequency >= center_hz / factor)
            & (frequency <= center_hz * factor)
        )
        if np.any(mask):
            linear_power = np.power(10.0, baseline[mask] / 10.0)
            expected[index] = 10.0 * np.log10(np.maximum(np.nanmean(linear_power), 1e-30))

    smoothed = smooth(frequency, baseline)

    assert np.allclose(smoothed, expected, equal_nan=True)


def test_fft_focus_mask_filters_range_and_log_zero():
    build_mask = _load_class_method("FftAnalysis", "_build_frequency_mask")
    frequency = np.array([0.0, 50.0, 100.0, 1000.0, 30000.0])

    mask = build_mask(frequency, True, 100.0, 20000.0, "log")

    assert np.array_equal(mask, np.array([False, False, True, True, False]))


def test_fft_class_has_no_dominant_tone_runtime_methods():
    tree = ast.parse(SIGNAL_ANALYSIS_PATH.read_text(encoding="utf-8"))
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FftAnalysis"
    )
    method_names = {node.name for node in class_node.body if isinstance(node, ast.FunctionDef)}

    assert not any("dominant" in name for name in method_names)
