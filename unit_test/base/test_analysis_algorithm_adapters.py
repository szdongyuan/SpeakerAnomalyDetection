import ast
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from base.analysis_algorithm_adapters import (
    _MAX_SPEC_TIME_BINS,
    calculate_analysis_instance,
)
from base.analysis_plot_renderer import _finish_axis, render_analysis_png


def _signal(sample_rate=48_000, seconds=0.25, amplitude=0.01):
    time_axis = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    return amplitude * np.sin(2.0 * np.pi * 1000.0 * time_axis)


def _calculate(analysis_type, config, signal=None):
    return calculate_analysis_instance(
        analysis_type,
        _signal() if signal is None else signal,
        48_000,
        config,
        1.0,
        source="自动分析",
        sequence_snapshot={"sequence_config": []},
    )


def test_spl_adapter_keeps_non_threshold_item_non_judging():
    result = _calculate(
        "SPL",
        {
            "weighting": "Z",
            "show_overall_spl": True,
            "limit_checked": False,
        },
    )

    assert result["judgement"] is None
    assert np.isfinite(result["metrics"]["overall_spl"])
    assert len(result["curve"]["x"]) == len(result["curve"]["y"])
    assert render_analysis_png(result["plot"]).startswith(b"\x89PNG")


def test_spl_adapter_uses_independent_overall_limits():
    result = _calculate(
        "SPL",
        {
            "weighting": "Z",
            "limit_checked": True,
            "limit_metric": "overall_spl",
            "scalar_upper_enabled": True,
            "scalar_upper_value": 50.0,
            "scalar_lower_enabled": False,
        },
    )

    assert result["judgement"] == "NG"
    assert result["metrics"]["overall_upper_limit"] == 50.0
    assert result["metrics"]["overall_lower_limit"] is None


def test_fba_and_fft_adapters_use_existing_headless_analyzers():
    fba = _calculate(
        "FBA",
        {
            "band_strategy": "1/3 倍频程",
            "weighting": "A",
            "f_min": 20,
            "f_max": 20_000,
            "limit_checked": False,
        },
    )
    fft = _calculate(
        "FFT",
        {
            "n_fft": 4096,
            "window": "hann",
            "overlap_ratio": 0.5,
            "weighting": "Z",
            "focus_range_enabled": True,
            "focus_min_hz": 100,
            "focus_max_hz": 20_000,
            "limit_checked": False,
        },
    )

    assert fba["judgement"] is None
    assert fba["plot"]["kind"] == "bar"
    assert len(fba["curve"]["x"]) == len(fba["curve"]["y"])
    assert fft["judgement"] is None
    assert fft["plot"]["kind"] == "curve"
    assert np.isfinite(fft["metrics"]["peak_value"])


def test_spec_adapter_returns_render_only_matrix_not_result_metrics():
    result = _calculate(
        "Spec",
        {
            "n_fft": 256,
            "hop_length": 64,
            "window_func": "hann",
            "freq_scale_type": "linear",
            "color_map": "viridis",
        },
    )

    assert result["judgement"] is None
    assert result["metrics"] == {}
    assert result["curve"] == {}
    assert result["plot"]["kind"] == "spectrogram"
    assert render_analysis_png(result["plot"]).startswith(b"\x89PNG")


def test_plot_renderer_does_not_repeat_item_title_inside_plot():
    axis = MagicMock()
    axis.get_legend_handles_labels.return_value = ([], [])

    _finish_axis(
        axis,
        {
            "x_label": "Time (s)",
            "y_label": "Frequency (Hz)",
            "title": "Spectrogram (Linear Scale)",
        },
    )

    axis.set_title.assert_not_called()


def test_spec_adapter_bounds_long_recording_time_columns():
    result = _calculate(
        "Spec",
        {
            "n_fft": 256,
            "hop_length": 64,
            "window_func": "hann",
            "freq_scale_type": "linear",
            "color_map": "viridis",
        },
        signal=_signal(seconds=4.0),
    )

    plot = result["plot"]
    assert plot["z"].shape[1] == _MAX_SPEC_TIME_BINS
    assert plot["z"].shape[1] == len(plot["x"])
    assert plot["x"][-1] > 3.9


def test_headless_analysis_adapter_does_not_import_ui_modules():
    project_root = Path(__file__).resolve().parents[2]
    runtime_paths = [
        project_root / "base" / "analysis_algorithm_adapters.py",
        project_root / "base" / "analysis_curve_style.py",
        project_root / "base" / "analysis_limit_evaluation.py",
    ]

    imported_ui_modules = []
    for path in runtime_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            imported_ui_modules.extend(
                module
                for module in modules
                if module == "ui" or module.startswith("ui.")
            )

    assert imported_ui_modules == []
