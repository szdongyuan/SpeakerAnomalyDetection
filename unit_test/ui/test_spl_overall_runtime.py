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


@pytest.mark.parametrize(
    ("limits", "curve_value", "expected_ok", "expected_deviation"),
    [
        ({"scalar_upper_value": 101.0}, 130.0, True, 1.0),
        ({"scalar_upper_value": 99.0}, 20.0, False, 1.0),
        ({"scalar_upper_value": 100.0}, 130.0, True, 0.0),
        ({"scalar_upper_enabled": False, "scalar_lower_enabled": True,
          "scalar_lower_value": 101.0}, 130.0, False, 1.0),
        ({"scalar_upper_enabled": False, "scalar_lower_enabled": True,
          "scalar_lower_value": 100.0}, 130.0, True, 0.0),
        ({"scalar_upper_value": 101.0, "scalar_lower_enabled": True,
          "scalar_lower_value": 99.0}, 130.0, True, 1.0),
    ],
)
def test_overall_judgment_uses_rms_independently_of_curve_and_title(
    signal_module, qapp, monkeypatch, limits, curve_value, expected_ok,
    expected_deviation,
):
    widget = signal_module.Spl("overall")
    widget.data_struct.store_wave_data = np.tile([1.0, -1.0], 16)
    widget.data_struct.store_wave_data_multi = None
    widget.data_struct.sample_rate = 48000
    widget.data_struct.analysis_result_dict.clear()
    widget.v2pa_factor = 2.0
    widget.analysis_config = {
        "limit_checked": True, "limit_metric": "overall_spl",
        "show_overall_spl": False,
        **limits,
    }
    monkeypatch.setattr(
        signal_module.AudioThdFrequencyResponseAnalysis, "spl_calculation",
        lambda *_args, **_kwargs: np.array([curve_value, curve_value]),
    )
    titles = []
    monkeypatch.setattr(signal_module.QMessageBox, "warning", lambda *_args: None)
    monkeypatch.setattr(widget.analysis_plot, "setTitle", lambda title, **_kwargs: titles.append(title))

    result = widget.calculate_spl()

    assert result["overall_spl"] == pytest.approx(100.0)
    assert widget.data_struct.analysis_result_dict["overall"][0] == expected_ok
    assert widget.data_struct.analysis_result_dict["overall"][1] == pytest.approx(expected_deviation)
    assert titles == [""]
    widget.close()


@pytest.mark.parametrize("limits", [
    {"scalar_upper_enabled": False, "scalar_lower_enabled": False},
    {"scalar_upper_value": float("nan")},
    {"scalar_upper_value": 90.0, "scalar_lower_enabled": True, "scalar_lower_value": 95.0},
])
def test_invalid_overall_limits_do_not_produce_a_judgment(signal_module, qapp, monkeypatch, limits):
    widget = signal_module.Spl("invalid-overall")
    widget.data_struct.store_wave_data = np.tile([1.0, -1.0], 16)
    widget.data_struct.store_wave_data_multi = None
    widget.data_struct.sample_rate = 48000
    widget.data_struct.analysis_result_dict.clear()
    widget.v2pa_factor = 2.0
    widget.analysis_config = {"limit_checked": True, "limit_metric": "overall_spl", **limits}
    warnings = []
    monkeypatch.setattr(signal_module.QMessageBox, "warning", lambda *args: warnings.append(args))

    assert widget.calculate_spl() is False
    assert "invalid-overall" not in widget.data_struct.analysis_result_dict
    assert len(warnings) == 1
    widget.close()


def test_overall_judgment_uses_weighting_time_range_and_ignores_legacy_correction(signal_module, qapp, monkeypatch):
    widget = signal_module.Spl("weighted-overall")
    widget.data_struct.store_wave_data = np.array([0.1, 0.1, 1.0, -1.0, 0.1, 0.1])
    widget.data_struct.store_wave_data_multi = None
    widget.data_struct.sample_rate = 10
    widget.data_struct.analysis_result_dict.clear()
    widget.v2pa_factor = 2.0
    widget.analysis_config = {
        "weighting": "A",
        "analysis_time_range_enabled": True,
        "analysis_start_time_sec": 0.2,
        "analysis_end_time_sec": 0.4,
        "directional_correction_enabled": True,
        "directional_additional_correction_db": 3.0,
        "limit_checked": True, "limit_metric": "overall_spl",
        "scalar_upper_value": 96.0,
    }
    weighted = []

    def apply_weighting(signal, sample_rate, **kwargs):
        weighted.append((sample_rate, kwargs["weighting"]))
        return signal * 0.5

    monkeypatch.setattr(signal_module, "apply_weighting_filter", apply_weighting)
    monkeypatch.setattr(signal_module.QMessageBox, "warning", lambda *_args: None)
    result = widget.calculate_spl()

    assert weighted == [(10, "A")]
    assert result["recorded_signal"] == pytest.approx([0.5, -0.5])
    assert result["overall_spl"] == pytest.approx(20.0 * np.log10(1.0 / 20e-6))
    assert "applied_correction_db" not in result
    assert widget.data_struct.analysis_result_dict["weighted-overall"][0] is True
    widget.close()


def test_spl_runtime_optionally_displays_overall_level(
    signal_module,
    qapp,
    monkeypatch,
):
    widget = signal_module.Spl("声压级 (SPL) 1")
    widget.data_struct.store_wave_data = np.tile([1.0, -1.0], 16)
    widget.data_struct.store_wave_data_multi = None
    widget.data_struct.sample_rate = 48000
    widget.v2pa_factor = 2.0
    widget.analysis_config = {
        "analysis_channel": 0,
        "weighting": "C",
        "show_overall_spl": True,
        "smooth_checked": False,
        "limit_checked": False,
    }
    titles = []

    monkeypatch.setattr(
        signal_module,
        "apply_weighting_filter",
        lambda signal, *args, **kwargs: signal,
    )
    monkeypatch.setattr(
        signal_module.AudioThdFrequencyResponseAnalysis,
        "spl_calculation",
        lambda self, *args, **kwargs: np.array([42.0, 43.0]),
    )
    monkeypatch.setattr(widget, "plot_spl", lambda *args: None)
    monkeypatch.setattr(
        widget.analysis_plot,
        "setTitle",
        lambda title, **kwargs: titles.append(title),
    )

    result = widget.calculate_spl()

    assert result["overall_spl"] == pytest.approx(100.0)
    assert titles == ["总体声压级：100.00 dBC"]

    titles.clear()
    widget.analysis_config["show_overall_spl"] = False
    result = widget.calculate_spl()

    # The display switch only controls the plot title. Product CSV export
    # requires the overall value for every completed SPL analysis.
    assert result["overall_spl"] == pytest.approx(100.0)
    assert titles == [""]
    assert "distance_correction_db" not in result
    widget.close()


def test_spl_runtime_ignores_legacy_corrections_for_curve_and_overall_level(
    signal_module,
    qapp,
    monkeypatch,
):
    widget = signal_module.Spl("声压级 (SPL) 1")
    widget.data_struct.store_wave_data = np.tile([1.0, -1.0], 16)
    widget.data_struct.store_wave_data_multi = None
    widget.data_struct.sample_rate = 48000
    widget.v2pa_factor = 2.0
    widget.analysis_config = {
        "analysis_channel": 0,
        "weighting": "Z",
        "show_overall_spl": True,
        "smooth_checked": False,
        "limit_checked": False,
        "free_field_distance_enabled": True,
        "measurement_distance_m": 0.1,
        "target_distance_m": 1.0,
        "directional_correction_enabled": True,
        "directional_additional_correction_db": -5.0,
    }
    plotted = []

    monkeypatch.setattr(
        signal_module.AudioThdFrequencyResponseAnalysis,
        "spl_calculation",
        lambda self, *args, **kwargs: np.array([42.0, 43.0]),
    )
    monkeypatch.setattr(
        widget,
        "plot_spl",
        lambda time_axis, spl: plotted.append(np.asarray(spl)),
    )

    result = widget.calculate_spl()

    assert result["signal_spl"] == pytest.approx([42.0, 43.0])
    assert result["overall_spl"] == pytest.approx(100.0)
    assert "measurement_distance_m" not in result
    assert "target_distance_m" not in result
    assert "distance_correction_db" not in result
    assert "directional_additional_correction_db" not in result
    assert "applied_correction_db" not in result
    assert plotted[0].tolist() == pytest.approx([42.0, 43.0])
    widget.close()
