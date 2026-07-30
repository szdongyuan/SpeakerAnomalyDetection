import importlib.util
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication


def _stub_module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_signal_analysis_module_without_heavy_optional_imports():
    stub_modules = {
        "librosa": _stub_module(
            "librosa",
            load=lambda *args, **kwargs: (None, None),
        ),
        "librosa.core": _stub_module(
            "librosa.core",
            spectrum=types.SimpleNamespace(),
        ),
        "librosa.feature": _stub_module(
            "librosa.feature",
            spectral=types.SimpleNamespace(),
        ),
        "librosa.sequence": _stub_module(
            "librosa.sequence",
            dtw=lambda *args, **kwargs: None,
        ),
        "base.model_runtime_validation": _stub_module(
            "base.model_runtime_validation",
            build_blocked_ai_export_detail=lambda *args, **kwargs: {},
            should_validate_model_duration=lambda *args, **kwargs: False,
            validate_model_duration=lambda *args, **kwargs: None,
        ),
        "base.predict_model": _stub_module(
            "base.predict_model",
            predict_from_audio=lambda *args, **kwargs: None,
        ),
        "base.training_model_management": _stub_module(
            "base.training_model_management",
            TrainingModelManagement=type(
                "TrainingModelManagement",
                (),
                {},
            ),
        ),
    }
    previous_modules = {
        name: sys.modules.get(name)
        for name in stub_modules
    }
    module_name = "_fft_signal_analysis_under_test"
    try:
        sys.modules.update(stub_modules)
        spec = importlib.util.spec_from_file_location(
            module_name,
            PROJECT_ROOT / "ui" / "signal_analysis_window.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


@pytest.fixture(scope="module")
def signal_module():
    return _load_signal_analysis_module_without_heavy_optional_imports()


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _sine_wave(frequency_hz, sample_rate=48000, duration_s=1.0):
    time_s = np.arange(int(sample_rate * duration_s)) / sample_rate
    return 0.02 * np.sin(2.0 * np.pi * frequency_hz * time_s)


def _config(*, upper_db=None):
    config = {
        "n_fft": 4096,
        "window": "hann",
        "overlap_ratio": 0.5,
        "weighting": "Z",
        "analysis_channel": 0,
        "x_axis_scale": "log",
        "focus_range_enabled": True,
        "focus_min_hz": 100,
        "focus_max_hz": 2000,
        "baseline_file_path": "",
        "baseline_display_mode": "overlay",
        "baseline_smooth_third_octave": False,
        "limit_checked": upper_db is not None,
        "display": {
            "main_curve_color": "#112233",
            "upper_limit_color": "#445566",
            "lower_limit_color": "#778899",
            "plot_view": {
                "x_enabled": True,
                "x_min": 100,
                "x_max": 2000,
                "y_enabled": True,
                "y_min": -320,
                "y_max": 220,
            },
        },
    }
    if upper_db is not None:
        config.update(
            {
                "limit_mode": "manual",
                "manual_upper_enabled": True,
                "manual_lower_enabled": False,
                "manual_upper_segments": [
                    {
                        "start_x": 0,
                        "start_y": upper_db,
                        "end_x": 3000,
                        "end_y": upper_db,
                    }
                ],
                "manual_lower_segments": [],
            }
        )
    return config


def _widget(signal_module):
    widget = signal_module.FftAnalysis("快速傅里叶变换 (FFT) 1")
    widget.data_struct.store_wave_data = _sine_wave(1000.0)
    widget.data_struct.store_wave_data_multi = None
    widget.data_struct.sample_rate = 48000
    widget.data_struct.analysis_result_dict.clear()
    widget.v2pa_factor = 1.0
    return widget


def test_fft_runtime_detects_peak_and_applies_log_plot_range(
    signal_module,
    qapp,
):
    widget = _widget(signal_module)
    widget.analysis_config = _config()

    result = widget.calculate_fft()

    frequencies = np.asarray(result["frequency_bins"])
    levels = np.asarray(result["fft_db"])
    peak_hz = frequencies[int(np.nanargmax(levels))]
    assert abs(peak_hz - 1000.0) <= 48000 / 4096
    assert result["display_mode"] == "overlay"
    assert np.allclose(
        widget.analysis_plot.getViewBox().viewRange()[0],
        [2.0, np.log10(2000.0)],
    )
    assert np.allclose(
        widget.analysis_plot.getViewBox().viewRange()[1],
        [-320.0, 220.0],
    )
    widget.close()


def test_fft_runtime_manual_limits_update_pass_fail_judgment(
    signal_module,
    qapp,
):
    widget = _widget(signal_module)
    widget.analysis_config = _config(upper_db=200.0)

    passed_result = widget.calculate_fft()

    assert passed_result
    assert widget.data_struct.analysis_result_dict[widget.title_name][0] is True

    widget.analysis_config = _config(upper_db=-300.0)
    failed_result = widget.calculate_fft()

    assert failed_result
    assert widget.data_struct.analysis_result_dict[widget.title_name][0] is False
    widget.close()


def test_fft_runtime_delta_mode_uses_loaded_baseline(
    signal_module,
    qapp,
    monkeypatch,
):
    widget = _widget(signal_module)
    monkeypatch.setattr(
        signal_module.librosa,
        "load",
        lambda *args, **kwargs: (_sine_wave(1000.0), 48000),
    )
    config = _config()
    config["baseline_file_path"] = "baseline.wav"
    config["baseline_display_mode"] = "delta"
    widget.analysis_config = config

    result = widget.calculate_fft()

    assert result["display_mode"] == "delta"
    assert result["baseline_db"]
    assert np.nanmax(np.abs(result["plot_db"])) < 1e-6
    widget.close()


def test_fft_runtime_rejects_delta_mode_without_baseline(
    signal_module,
    qapp,
    monkeypatch,
):
    widget = _widget(signal_module)
    config = _config()
    config["baseline_display_mode"] = "delta"
    widget.analysis_config = config
    warnings = []
    monkeypatch.setattr(
        signal_module.QMessageBox,
        "warning",
        lambda *args: warnings.append(args[-1]),
    )

    assert widget.calculate_fft() is False
    assert any("背景音频基线" in message for message in warnings)
    widget.close()


def test_fft_csv_limits_only_apply_inside_defined_frequency_range(
    signal_module,
):
    target_x = np.asarray([50.0, 100.0, 150.0, 200.0, 250.0])
    upper, lower = signal_module.FftAnalysis._resolve_limits(
        {
            "limit_mode": "csv",
            "limit_data": (
                [100.0, 200.0],
                [10.0, 20.0],
                [0.0, 5.0],
            ),
        },
        target_x,
    )

    assert np.isnan(upper[0])
    assert np.allclose(upper[1:4], [10.0, 15.0, 20.0])
    assert np.isnan(upper[4])
    assert np.isnan(lower[0])
    assert np.allclose(lower[1:4], [0.0, 2.5, 5.0])
    assert np.isnan(lower[4])
    assert signal_module.get_class_mapping()["FFT"] is signal_module.FftAnalysis
