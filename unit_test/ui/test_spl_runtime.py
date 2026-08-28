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
        "librosa": _stub_module("librosa", load=lambda *args, **kwargs: (None, None)),
        "librosa.core": _stub_module("librosa.core", spectrum=types.SimpleNamespace()),
        "librosa.feature": _stub_module("librosa.feature", spectral=types.SimpleNamespace()),
        "librosa.sequence": _stub_module("librosa.sequence", dtw=lambda *args, **kwargs: None),
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
            TrainingModelManagement=type("TrainingModelManagement", (), {}),
        ),
    }
    previous_modules = {name: sys.modules.get(name) for name in stub_modules}
    module_name = "_spl_signal_analysis_under_test"
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


def _manual_upper_config(end_x=1.0, upper_db=8.0):
    return {
        "analysis_channel": 0,
        "weighting": "Z",
        "smooth_checked": False,
        "limit_checked": True,
        "limit_mode": "manual",
        "manual_upper_enabled": True,
        "manual_lower_enabled": False,
        "manual_upper_segments": [
            {
                "start_x": 0.0,
                "start_y": upper_db,
                "end_x": end_x,
                "end_y": upper_db,
            }
        ],
        "manual_lower_segments": [],
    }


def _constant_upper_config(upper_db=8.0):
    return {
        "analysis_channel": 0,
        "weighting": "Z",
        "smooth_checked": False,
        "limit_checked": True,
        "limit_mode": "manual",
        "manual_input_mode": "constant",
        "constant_upper_enabled": True,
        "constant_lower_enabled": False,
        "constant_upper_value": upper_db,
        "constant_lower_value": 0.0,
    }


def test_spl_constant_limit_covers_first_runtime_point(signal_module, qapp, monkeypatch):
    widget = signal_module.Spl("SPL")
    widget.data_struct.store_wave_data = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    widget.data_struct.store_wave_data_multi = None
    widget.data_struct.sample_rate = 10
    widget.data_struct.analysis_result_dict.clear()
    widget.v2pa_factor = 1.0
    widget.analysis_config = _constant_upper_config(upper_db=8.0)
    monkeypatch.setattr(
        signal_module.AudioThdFrequencyResponseAnalysis,
        "spl_calculation",
        lambda self, *args, **kwargs: np.asarray([12.0, 5.0, 4.0]),
    )

    result = widget.calculate_spl()

    assert result is not False
    assert widget.data_struct.analysis_result_dict["SPL"] == (False, 4.0)
    widget.close()


def test_spl_runtime_ignores_legacy_correction_config(
    signal_module,
    qapp,
    monkeypatch,
):
    widget = signal_module.Spl("SPL")
    widget.data_struct.store_wave_data = np.asarray(
        [0.1, 0.2, 0.3],
        dtype=np.float32,
    )
    widget.data_struct.store_wave_data_multi = None
    widget.data_struct.sample_rate = 10
    widget.data_struct.analysis_result_dict.clear()
    widget.v2pa_factor = 1.0
    widget.analysis_config = _constant_upper_config(upper_db=8.0)
    widget.analysis_config.update(
        {
            "free_field_distance_enabled": True,
            "measurement_distance_m": 0.0,
            "target_distance_m": 1.0,
            "directional_correction_enabled": True,
            "directional_additional_correction_db": float("inf"),
        }
    )
    monkeypatch.setattr(
        signal_module.AudioThdFrequencyResponseAnalysis,
        "spl_calculation",
        lambda self, *args, **kwargs: np.asarray([12.0, 5.0, 4.0]),
    )

    result = widget.calculate_spl()

    assert result["signal_spl"] == pytest.approx([12.0, 5.0, 4.0])
    assert "applied_correction_db" not in result
    assert widget.data_struct.analysis_result_dict["SPL"] == (False, 4.0)
    widget.close()


def test_spl_runtime_applies_analysis_time_range_and_preserves_source_time_axis(
    signal_module,
    qapp,
    monkeypatch,
):
    widget = signal_module.Spl("SPL")
    widget.data_struct.store_wave_data = np.arange(10, dtype=np.float32)
    widget.data_struct.store_wave_data_multi = None
    widget.data_struct.sample_rate = 10
    widget.v2pa_factor = 1.0
    widget.analysis_config = {
        "analysis_channel": 0,
        "weighting": "Z",
        "smooth_checked": False,
        "limit_checked": False,
        "analysis_time_range_enabled": True,
        "analysis_start_time_sec": 0.2,
        "analysis_end_time_sec": 0.6,
    }
    analyzed_signals = []
    plotted = []
    monkeypatch.setattr(
        signal_module.AudioThdFrequencyResponseAnalysis,
        "spl_calculation",
        lambda self, signal, *args, **kwargs: (
            analyzed_signals.append(np.asarray(signal).copy())
            or np.asarray([40.0, 41.0, 42.0, 43.0])
        ),
    )
    monkeypatch.setattr(
        widget,
        "plot_spl",
        lambda time_axis, spl: plotted.append(
            (np.asarray(time_axis), np.asarray(spl))
        ),
    )

    result = widget.calculate_spl()

    assert analyzed_signals[0].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert result["recorded_signal"] == [2.0, 3.0, 4.0, 5.0]
    assert result["signal_duration"] == pytest.approx([0.2, 0.3, 0.4, 0.5])
    assert plotted[0][0].tolist() == pytest.approx([0.2, 0.3, 0.4, 0.5])
    widget.close()


def test_splf_constant_limit_uses_sorted_valid_axis(signal_module):
    x_values = signal_module._sorted_finite_positive_x_for_limits(
        [300.0, 200.0, np.nan, 100.0],
        [1.0, 20.0, 5.0, 1.0],
    )

    limit_x, upper, lower = signal_module._resolve_spl_limit_data(
        _constant_upper_config(upper_db=10.0),
        x_values,
    )

    assert limit_x == [100.0, 200.0, 300.0]
    assert upper == [10.0, 10.0, 10.0]
    assert np.all(np.isnan(lower))


def test_splf_constant_limit_drives_runtime_judgment(
    signal_module,
    qapp,
    monkeypatch,
):
    widget = signal_module.SplFrequency("SPLF")
    widget.data_struct.store_wave_data = np.asarray(
        [0.1, 0.2, 0.3],
        dtype=np.float32,
    )
    widget.data_struct.store_wave_data_multi = None
    widget.data_struct.sample_rate = 48000
    widget.data_struct.stimulus_info = {
        "stimulus_method": "steps",
        "stimulus_type": "linear",
        "start_freq": 100.0,
        "stop_freq": 300.0,
        "num_steps": 3,
        "total_time": 0.3,
        "repeat_times": 1,
    }
    widget.data_struct.analysis_result_dict.clear()
    widget.v2pa_factor = 1.0
    widget.analysis_config = _constant_upper_config(upper_db=10.0)
    widget.analysis_config["splf_calc_mode"] = "fundamental"

    class FakeSplFrequencyAnalyzer:
        def __init__(self, sample_rate):
            self.sample_rate = sample_rate

        def compute(self, *args, **kwargs):
            return types.SimpleNamespace(
                frequencies_hz=np.asarray([300.0, 200.0, 100.0]),
                spl_db=np.asarray([1.0, 20.0, 1.0]),
            )

    monkeypatch.setattr(
        signal_module,
        "SplFrequencyAnalyzer",
        FakeSplFrequencyAnalyzer,
    )

    result = widget.calculate_spl()

    assert result is not False
    assert widget.data_struct.analysis_result_dict["SPLF"] == (False, 10.0)
    widget.close()


def test_spl_invalid_constant_limits_show_warning(signal_module, qapp, monkeypatch):
    widget = signal_module.Spl("SPL")
    widget.data_struct.store_wave_data = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    widget.data_struct.store_wave_data_multi = None
    widget.data_struct.sample_rate = 10
    widget.data_struct.analysis_result_dict.clear()
    widget.v2pa_factor = 1.0
    widget.analysis_config = {
        **_constant_upper_config(upper_db=5.0),
        "constant_lower_enabled": True,
        "constant_lower_value": 6.0,
    }
    warnings = []
    monkeypatch.setattr(
        signal_module.AudioThdFrequencyResponseAnalysis,
        "spl_calculation",
        lambda self, *args, **kwargs: np.asarray([4.0, 5.0, 6.0]),
    )
    monkeypatch.setattr(
        signal_module.QMessageBox,
        "warning",
        lambda *args: warnings.append(args[-1]),
    )

    assert widget.calculate_spl() is False
    assert any("下限不能大于上限" in message for message in warnings)
    assert "SPL" not in widget.data_struct.analysis_result_dict
    widget.close()


def test_spl_manual_limits_drive_runtime_judgment(signal_module, qapp, monkeypatch):
    widget = signal_module.Spl("SPL")
    widget.data_struct.store_wave_data = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    widget.data_struct.store_wave_data_multi = None
    widget.data_struct.sample_rate = 10
    widget.data_struct.analysis_result_dict.clear()
    widget.v2pa_factor = 1.0
    widget.analysis_config = _manual_upper_config(end_x=0.2, upper_db=8.0)
    monkeypatch.setattr(
        signal_module.AudioThdFrequencyResponseAnalysis,
        "spl_calculation",
        lambda self, *args, **kwargs: np.asarray([5.0, 12.0, 4.0]),
    )

    result = widget.calculate_spl()

    assert result is not False
    assert widget.data_struct.analysis_result_dict["SPL"] == (False, 4.0)
    widget.close()


def test_spl_legacy_csv_without_limit_mode_still_works(signal_module, qapp, monkeypatch):
    widget = signal_module.Spl("SPL")
    widget.data_struct.store_wave_data = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    widget.data_struct.store_wave_data_multi = None
    widget.data_struct.sample_rate = 10
    widget.data_struct.analysis_result_dict.clear()
    widget.v2pa_factor = 1.0
    widget.analysis_config = {
        "analysis_channel": 0,
        "weighting": "Z",
        "smooth_checked": False,
        "limit_checked": True,
        "limit_data": ([0.0, 0.1, 0.2], [8.0, 8.0, 8.0], [0.0, 0.0, 0.0]),
    }
    monkeypatch.setattr(
        signal_module.AudioThdFrequencyResponseAnalysis,
        "spl_calculation",
        lambda self, *args, **kwargs: np.asarray([5.0, 7.0, 4.0]),
    )

    result = widget.calculate_spl()

    assert result is not False
    assert widget.data_struct.analysis_result_dict["SPL"][0] is True
    widget.close()


def test_spl_sparse_csv_interpolates_between_points_for_runtime_judgment(
    signal_module,
    qapp,
    monkeypatch,
):
    widget = signal_module.Spl("SPL")
    widget.data_struct.analysis_result_dict.clear()
    widget.analysis_config = {}
    captured_out_masks = []
    monkeypatch.setattr(
        signal_module.LimitPlotUtils,
        "plot_out_segments",
        lambda _plot, _x, _y, out_mask: captured_out_masks.append(
            np.asarray(out_mask, dtype=bool).copy()
        ),
    )

    widget.plot_spl_with_limits(
        np.asarray([0.8, 0.86, 0.9]),
        np.asarray([80.0, 89.0, 80.0]),
        [0.8, 0.9],
        [85.0, 84.0],
        [np.nan, np.nan],
    )

    assert widget.data_struct.analysis_result_dict["SPL"] == (False, 4.6)
    assert captured_out_masks[0].tolist() == [False, True, False]
    widget.close()


def test_spl_sparse_lower_only_csv_interpolates_for_runtime_judgment(
    signal_module,
    qapp,
    monkeypatch,
):
    widget = signal_module.Spl("SPL")
    widget.data_struct.analysis_result_dict.clear()
    widget.analysis_config = {}
    captured_out_masks = []
    monkeypatch.setattr(
        signal_module.LimitPlotUtils,
        "plot_out_segments",
        lambda _plot, _x, _y, out_mask: captured_out_masks.append(
            np.asarray(out_mask, dtype=bool).copy()
        ),
    )

    widget.plot_spl_with_limits(
        np.asarray([0.8, 0.86, 0.9]),
        np.asarray([80.0, 70.0, 80.0]),
        [0.8, 0.9],
        [np.nan, np.nan],
        [75.0, 76.0],
    )

    assert widget.data_struct.analysis_result_dict["SPL"] == (False, 5.6)
    assert captured_out_masks[0].tolist() == [False, True, False]
    widget.close()


def test_spl_sparse_csv_renders_interpolated_out_of_limit_segment(
    signal_module,
    qapp,
):
    widget = signal_module.Spl("SPL")
    widget.data_struct.analysis_result_dict.clear()
    widget.analysis_config = {}

    widget.plot_spl_with_limits(
        np.asarray([0.8, 0.84, 0.85, 0.86, 0.9]),
        np.asarray([80.0, 89.0, 89.0, 89.0, 80.0]),
        [0.8, 0.9],
        [85.0, 84.0],
        [np.nan, np.nan],
    )

    red_x, red_y = widget.analysis_plot.listDataItems()[-1].getData()
    np.testing.assert_allclose(red_x, [0.84, 0.85, 0.86])
    np.testing.assert_allclose(red_y, [89.0, 89.0, 89.0])
    assert widget.data_struct.analysis_result_dict["SPL"] == (False, 4.6)
    widget.close()


def test_spl_csv_interpolation_keeps_missing_sides_and_outside_range_unjudged(
    signal_module,
):
    upper, lower = signal_module._interpolate_spl_limit_curves(
        [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
        [0.0, 1.0, 2.0],
        [10.0, np.nan, 20.0],
        [-1.0, 0.0, 1.0],
    )

    np.testing.assert_allclose(
        upper,
        [np.nan, 10.0, np.nan, np.nan, np.nan, 20.0, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        lower,
        [np.nan, -1.0, -0.5, 0.0, 0.5, 1.0, np.nan],
        equal_nan=True,
    )


def test_spl_csv_interpolation_preserves_duplicate_x_jump(signal_module):
    upper, lower = signal_module._interpolate_spl_limit_curves(
        [0.5, 1.0, 1.0001, 1.5],
        [0.0, 1.0, 1.0, 2.0],
        [10.0, 20.0, 30.0, 40.0],
        [np.nan, np.nan, np.nan, np.nan],
    )

    np.testing.assert_allclose(upper, [15.0, 20.0, 30.001, 35.0])
    assert np.isnan(lower).all()


def test_spl_invalid_manual_segments_show_warning(signal_module, qapp, monkeypatch):
    widget = signal_module.Spl("SPL")
    widget.data_struct.store_wave_data = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    widget.data_struct.store_wave_data_multi = None
    widget.data_struct.sample_rate = 10
    widget.data_struct.analysis_result_dict.clear()
    widget.v2pa_factor = 1.0
    widget.analysis_config = _manual_upper_config(end_x=0.0)
    warnings = []
    monkeypatch.setattr(
        signal_module.AudioThdFrequencyResponseAnalysis,
        "spl_calculation",
        lambda self, *args, **kwargs: np.asarray([5.0, 7.0, 4.0]),
    )
    monkeypatch.setattr(
        signal_module.QMessageBox,
        "warning",
        lambda *args: warnings.append(args[-1]),
    )

    assert widget.calculate_spl() is False
    assert any("上限" in message for message in warnings)
    assert "SPL" not in widget.data_struct.analysis_result_dict
    widget.close()


def test_splf_manual_limit_axis_is_sorted_and_keeps_gaps(signal_module):
    x_values = signal_module._sorted_finite_positive_x_for_limits(
        [300.0, 200.0, np.nan, 100.0],
        [1.0, 20.0, 5.0, 1.0],
    )
    config = _manual_upper_config(end_x=250.0, upper_db=10.0)
    config["manual_upper_segments"][0]["start_x"] = 100.0
    limit_x, upper, lower = signal_module._resolve_spl_limit_data(config, x_values)

    assert limit_x == [100.0, 200.0, 300.0]
    assert np.isnan(upper[0])
    assert upper[1] == pytest.approx(10.0)
    assert np.isnan(upper[2])
    assert np.all(np.isnan(lower))


def test_splf_manual_limits_handle_unsorted_analyzer_output(
    signal_module,
    qapp,
    monkeypatch,
):
    widget = signal_module.SplFrequency("SPLF")
    widget.data_struct.store_wave_data = np.asarray(
        [0.1, 0.2, 0.3],
        dtype=np.float32,
    )
    widget.data_struct.store_wave_data_multi = None
    widget.data_struct.sample_rate = 48000
    widget.data_struct.stimulus_info = {
        "stimulus_method": "steps",
        "stimulus_type": "linear",
        "start_freq": 100.0,
        "stop_freq": 300.0,
        "num_steps": 3,
        "total_time": 0.3,
        "repeat_times": 1,
    }
    widget.data_struct.analysis_result_dict.clear()
    widget.v2pa_factor = 1.0
    widget.analysis_config = _manual_upper_config(
        end_x=250.0,
        upper_db=10.0,
    )
    widget.analysis_config["manual_upper_segments"][0]["start_x"] = 100.0
    widget.analysis_config["splf_calc_mode"] = "fundamental"

    class FakeSplFrequencyAnalyzer:
        def __init__(self, sample_rate):
            self.sample_rate = sample_rate

        def compute(self, *args, **kwargs):
            return types.SimpleNamespace(
                frequencies_hz=np.asarray([300.0, 200.0, 100.0]),
                spl_db=np.asarray([1.0, 20.0, 1.0]),
            )

    monkeypatch.setattr(
        signal_module,
        "SplFrequencyAnalyzer",
        FakeSplFrequencyAnalyzer,
    )

    result = widget.calculate_spl()

    assert result is not False
    assert result["frequency_list"] == [300.0, 200.0, 100.0]
    assert widget.data_struct.analysis_result_dict["SPLF"] == (False, 10.0)
    widget.close()
