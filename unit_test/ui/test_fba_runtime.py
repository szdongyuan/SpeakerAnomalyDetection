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
            TrainingModelManagement=type("TrainingModelManagement", (), {}),
        ),
    }
    previous_modules = {
        name: sys.modules.get(name)
        for name in stub_modules
    }
    module_name = "_fba_signal_analysis_under_test"
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


def _sine_wave(frequency_hz, sample_rate=48000, duration_s=0.25):
    time_s = np.arange(int(sample_rate * duration_s)) / sample_rate
    return 0.02 * np.sin(2.0 * np.pi * frequency_hz * time_s)


def _manual_config(upper_db):
    return {
        "band_strategy": "自定义",
        "custom_bands_text": "900, 1100, 1 kHz\n2900, 3100, 3 kHz",
        "f_min": 20,
        "f_max": 5000,
        "weighting": "Z",
        "analysis_channel": 0,
        "limit_checked": True,
        "limit_mode": "manual",
        "manual_upper_enabled": True,
        "manual_lower_enabled": False,
        "manual_upper_segments": [
            {
                "start_x": 0,
                "start_y": upper_db,
                "end_x": 5000,
                "end_y": upper_db,
            }
        ],
        "manual_lower_segments": [],
        "display": {
            "main_curve_color": "#112233",
            "upper_limit_color": "#445566",
            "lower_limit_color": "#778899",
            "plot_view": {
                "y_enabled": True,
                "y_min": -320,
                "y_max": 220,
            },
        },
    }


def test_fba_runtime_analyzes_manual_limits_and_updates_judgment(
    signal_module,
    qapp,
):
    widget = signal_module.FrequencyBandAnalysis("频段能量 (FBA) 1")
    widget.data_struct.store_wave_data = _sine_wave(1000.0)
    widget.data_struct.store_wave_data_multi = None
    widget.data_struct.sample_rate = 48000
    widget.data_struct.analysis_result_dict.clear()
    widget.v2pa_factor = 1.0

    widget.analysis_config = _manual_config(200.0)
    passed_result = widget.calculate_fba()

    assert passed_result["bands"] == ["1 kHz", "3 kHz"]
    assert passed_result["exceeded_bands"] == []
    assert widget.data_struct.analysis_result_dict[widget.title_name][0] is True
    assert np.allclose(
        widget.analysis_plot.getViewBox().viewRange()[1],
        [-320.0, 220.0],
    )

    widget.analysis_config = _manual_config(-300.0)
    failed_result = widget.calculate_fba()

    assert failed_result["exceeded_bands"]
    assert widget.data_struct.analysis_result_dict[widget.title_name][0] is False
    widget.close()


def test_fba_runtime_interpolates_csv_limits_and_maps_class(signal_module):
    centers = np.asarray([100.0, 1000.0, 2000.0])
    upper, lower = signal_module.FrequencyBandAnalysis._resolve_limits(
        {
            "limit_mode": "csv",
            "limit_data": (
                [100.0, 2000.0],
                [80.0, 100.0],
                [20.0, 40.0],
            ),
        },
        centers,
    )

    assert np.allclose(upper, [80.0, 89.4736842105, 100.0])
    assert np.allclose(lower, [20.0, 29.4736842105, 40.0])
    assert (
        signal_module.get_class_mapping()["FBA"]
        is signal_module.FrequencyBandAnalysis
    )


def test_fba_csv_limits_without_band_overlap_are_ok(signal_module, qapp):
    widget = signal_module.FrequencyBandAnalysis("频段能量 (FBA) 无重叠")
    widget.data_struct.store_wave_data = _sine_wave(1000.0)
    widget.data_struct.store_wave_data_multi = None
    widget.data_struct.sample_rate = 48000
    widget.data_struct.analysis_result_dict.clear()
    widget.v2pa_factor = 1.0
    config = _manual_config(200.0)
    config.update(
        {
            "limit_mode": "csv",
            "limit_data": (
                [4000.0, 5000.0],
                [80.0, 80.0],
                [20.0, 20.0],
            ),
        }
    )
    widget.analysis_config = config

    result = widget.calculate_fba()

    assert result
    assert result["exceeded_bands"] == []
    assert widget.data_struct.analysis_result_dict[widget.title_name] == (
        True,
        0.0,
    )
    widget.close()


def test_fba_invalid_analysis_data_is_not_accepted_as_no_overlap(
    signal_module,
    qapp,
    monkeypatch,
):
    widget = signal_module.FrequencyBandAnalysis("频段能量 (FBA) 无效结果")
    widget.data_struct.store_wave_data = _sine_wave(1000.0)
    widget.data_struct.store_wave_data_multi = None
    widget.data_struct.sample_rate = 48000
    widget.data_struct.analysis_result_dict.clear()
    widget.v2pa_factor = 1.0
    config = _manual_config(200.0)
    config.update(
        {
            "limit_mode": "csv",
            "limit_data": (
                [4000.0, 5000.0],
                [80.0, 80.0],
                [20.0, 20.0],
            ),
        }
    )
    widget.analysis_config = config
    warnings = []
    monkeypatch.setattr(
        signal_module.FrequencyBandAnalyzer,
        "analyze",
        lambda *args, **kwargs: types.SimpleNamespace(
            bands=[
                types.SimpleNamespace(f_center=1000.0),
                types.SimpleNamespace(f_center=3000.0),
            ],
            band_levels_weighted_db=np.asarray([np.nan, np.nan]),
        ),
    )
    monkeypatch.setattr(
        signal_module.QMessageBox,
        "warning",
        lambda *args: warnings.append(args[-1]),
    )

    assert widget.calculate_fba() is False
    assert any("没有有效频段" in message for message in warnings)
    assert widget.title_name not in widget.data_struct.analysis_result_dict
    widget.close()


def test_fba_resolves_constant_manual_limits(signal_module):
    centers = np.asarray([100.0, 1000.0, 2000.0])

    upper, lower = signal_module.FrequencyBandAnalysis._resolve_limits(
        {
            "limit_mode": "manual",
            "manual_input_mode": "constant",
            "constant_upper_enabled": True,
            "constant_lower_enabled": True,
            "constant_upper_value": 85.0,
            "constant_lower_value": 25.0,
        },
        centers,
    )

    assert np.allclose(upper, [85.0, 85.0, 85.0])
    assert np.allclose(lower, [25.0, 25.0, 25.0])


def test_fba_tick_labels_adapt_to_available_width(signal_module):
    select_indices = signal_module.FrequencyBandAnalysis._select_fba_tick_indices

    assert select_indices([40] * 31, available_width=390) == [
        0,
        4,
        8,
        12,
        16,
        20,
        24,
        30,
    ]
    assert select_indices([40] * 5, available_width=600) == [0, 1, 2, 3, 4]


def test_fba_ticks_recompute_after_hidden_window_is_shown(signal_module, qapp):
    widget = signal_module.FrequencyBandAnalysis("FBA")
    widget._fba_tick_labels = [f"{index} Hz" for index in range(31)]
    bottom_axis = widget.analysis_plot.getAxis("bottom")

    widget._update_fba_axis_ticks()
    hidden_ticks = bottom_axis._tickLevels[0]

    widget.setGeometry(0, 0, 475, 320)
    widget.show()
    qapp.processEvents()
    shown_ticks = bottom_axis._tickLevels[0]

    widget.resize(1200, 500)
    qapp.processEvents()
    wide_ticks = bottom_axis._tickLevels[0]

    assert len(shown_ticks) < len(hidden_ticks)
    assert len(shown_ticks) < len(wide_ticks)
    assert shown_ticks[0] == (0, "0 Hz")
    assert shown_ticks[-1] == (30, "30 Hz")
    widget.close()
