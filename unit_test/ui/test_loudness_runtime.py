import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

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
    previous_modules = {name: sys.modules.get(name) for name in stub_modules}
    module_name = "_loudness_signal_analysis_under_test"
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


def test_loudness_runtime_maps_class_and_builds_export_result(
    signal_module,
    qapp,
    monkeypatch,
):
    raw_result = SimpleNamespace(
        time_s=np.asarray([0.05, 0.15]),
        loudness_sone=np.asarray([1.0, 1.5]),
        loudness_level_phon=np.asarray([40.0, 45.85]),
        metadata={},
    )
    loudness_result = SimpleNamespace(
        enabled=True,
        raw_result=raw_result,
        skipped_reason=None,
        summary={"mean_sone": 1.25},
        display_payload={"summary_cards": [], "curves": [], "heatmaps": []},
    )
    monkeypatch.setattr(
        signal_module,
        "run_sound_quality",
        lambda *args, **kwargs: SimpleNamespace(
            loudness=loudness_result,
            skipped_reason=None,
        ),
    )

    widget = signal_module.LoudnessAnalysis("响度 (LOUD) 1")
    widget.data_struct.store_wave_data = np.ones(4800, dtype=np.float64)
    widget.data_struct.store_wave_data_multi = None
    widget.data_struct.sample_rate = 48000
    widget.v2pa_factor = 1.0
    widget.analysis_config = {
        "enabled": True,
        "analysis_channel": 0,
        "method": "per_segment",
        "display": {},
        "save": {},
        "advanced": {"curve_y_unit": "sone"},
        "limit_checked": False,
    }

    result = widget.calculate_loudness()

    assert signal_module.get_class_mapping()["LOUD"] is signal_module.LoudnessAnalysis
    assert result["time_s"] == [0.05, 0.15]
    assert result["loudness_sone"] == [1.0, 1.5]
    assert widget.export_detail["mean_sone"] == 1.25
    widget.close()


def test_loudness_curve_limit_reports_deviation(
    signal_module,
    qapp,
    monkeypatch,
):
    raw_result = SimpleNamespace(
        time_s=np.asarray([0.05, 0.15]),
        loudness_sone=np.asarray([1.0, 1.5]),
        loudness_level_phon=np.asarray([40.0, 45.85]),
        metadata={},
    )
    loudness_result = SimpleNamespace(
        enabled=True,
        raw_result=raw_result,
        skipped_reason=None,
        summary={},
        display_payload={"summary_cards": [], "curves": [], "heatmaps": []},
    )
    monkeypatch.setattr(
        signal_module,
        "run_sound_quality",
        lambda *args, **kwargs: SimpleNamespace(
            loudness=loudness_result,
            skipped_reason=None,
        ),
    )

    widget = signal_module.LoudnessAnalysis("响度 (LOUD) 1")
    widget.data_struct.store_wave_data = np.ones(4800, dtype=np.float64)
    widget.data_struct.store_wave_data_multi = None
    widget.data_struct.sample_rate = 48000
    widget.data_struct.analysis_result_dict.clear()
    widget.v2pa_factor = 1.0
    widget.analysis_config = {
        "enabled": True,
        "analysis_channel": 0,
        "method": "per_segment",
        "display": {},
        "save": {},
        "advanced": {"curve_y_unit": "sone"},
        "limit_checked": True,
        "limit_metric": "curve_y",
        "curve_upper_enabled": True,
        "curve_upper_value": 1.2,
        "curve_lower_enabled": False,
    }

    result = widget.calculate_loudness()

    assert result is not False
    assert widget.data_struct.analysis_result_dict["响度 (LOUD) 1"] == (
        False,
        pytest.approx(0.3),
    )
    widget.close()


@pytest.mark.parametrize(
    ("limit_metric", "limit_unit", "required_metric"),
    [
        ("steady_state_average", "sone", "steady_state_average_sone"),
        ("max_transient", "phon", "max_transient_phon"),
    ],
)
def test_loudness_runtime_requests_hidden_judgment_metric(
    signal_module,
    limit_metric,
    limit_unit,
    required_metric,
):
    sq_config = signal_module.LoudnessAnalysis._build_sq_config(
        {
            "enabled": True,
            "display": {"summary_metrics": []},
            "save": {},
            "advanced": {"curve_y_unit": limit_unit},
            "limit_checked": True,
            "limit_metric": limit_metric,
            "curve_limit_unit": limit_unit,
        }
    )

    loudness_config = sq_config["items"]["LOUD"]
    assert loudness_config["display"]["summary_metrics"] == []
    assert loudness_config["required_summary_metrics"] == [required_metric]


def test_loudness_scalar_limit_uses_configured_phon_unit(signal_module, qapp):
    widget = signal_module.LoudnessAnalysis("响度 phon 阈值")
    widget.data_struct.analysis_result_dict.clear()
    loudness_result = SimpleNamespace(
        summary={
            "steady_state_average_sone": 3.0,
            "steady_state_average_phon": 60.0,
        }
    )

    widget._apply_loudness_scalar_limit(
        loudness_result,
        {
            "advanced": {"curve_y_unit": "phon"},
            "curve_limit_unit": "phon",
            "curve_upper_enabled": True,
            "curve_upper_value": 50.0,
            "curve_lower_enabled": False,
        },
        metric="steady_state_average",
    )

    assert widget.data_struct.analysis_result_dict["响度 phon 阈值"] == (
        False,
        pytest.approx(10.0),
    )
    widget.close()


def test_loudness_curve_limit_uses_configured_phon_unit(signal_module, qapp):
    widget = signal_module.LoudnessAnalysis("响度曲线 phon 阈值")
    widget.data_struct.analysis_result_dict.clear()
    raw_result = SimpleNamespace(
        loudness_sone=np.asarray([3.0, 4.0]),
        loudness_level_phon=np.asarray([55.0, 60.0]),
    )

    widget._apply_loudness_curve_limit(
        raw_result,
        {
            "curve_limit_unit": "phon",
            "curve_upper_enabled": True,
            "curve_upper_value": 50.0,
            "curve_lower_enabled": False,
        },
        {"curve_y_unit": "sone"},
    )

    assert widget.data_struct.analysis_result_dict["响度曲线 phon 阈值"] == (
        False,
        pytest.approx(10.0),
    )
    widget.close()


def test_loudness_curve_limit_interpolates_csv_threshold(signal_module, qapp):
    widget = signal_module.LoudnessAnalysis("响度 CSV 曲线阈值")
    widget.data_struct.analysis_result_dict.clear()
    raw_result = SimpleNamespace(
        time_s=np.asarray([0.0, 1.0, 2.0]),
        loudness_sone=np.asarray([1.0, 2.0, 3.0]),
        loudness_level_phon=np.asarray([40.0, 50.0, 60.0]),
        metadata={},
    )

    widget._apply_loudness_curve_limit(
        raw_result,
        {
            "limit_mode": "csv",
            "limit_data": (
                [0.0, 2.0],
                [1.5, 2.5],
                [np.nan, np.nan],
            ),
            "curve_limit_unit": "sone",
        },
        {"curve_y_unit": "sone"},
    )

    assert widget.data_struct.analysis_result_dict["响度 CSV 曲线阈值"] == (
        False,
        pytest.approx(0.5),
    )
    widget.close()


def test_loudness_curve_limit_uses_edited_segments(signal_module, qapp):
    widget = signal_module.LoudnessAnalysis("响度编辑曲线阈值")
    widget.data_struct.analysis_result_dict.clear()
    raw_result = SimpleNamespace(
        time_s=np.asarray([0.0, 1.0, 2.0]),
        loudness_sone=np.asarray([1.0, 1.7, 2.5]),
        loudness_level_phon=np.asarray([40.0, 45.0, 50.0]),
        metadata={},
    )

    widget._apply_loudness_curve_limit(
        raw_result,
        {
            "limit_mode": "manual",
            "manual_input_mode": "segments",
            "manual_upper_enabled": True,
            "manual_upper_segments": [
                {
                    "start_x": 0.0,
                    "start_y": 1.2,
                    "end_x": 2.0,
                    "end_y": 2.2,
                }
            ],
            "manual_lower_enabled": False,
            "manual_lower_segments": [],
            "curve_limit_unit": "sone",
        },
        {"curve_y_unit": "sone"},
    )

    assert widget.data_struct.analysis_result_dict["响度编辑曲线阈值"] == (
        False,
        pytest.approx(0.3),
    )
    widget.close()


def test_loudness_scalar_limit_uses_scalar_threshold_fields(signal_module, qapp):
    widget = signal_module.LoudnessAnalysis("响度标量阈值")
    widget.data_struct.analysis_result_dict.clear()
    loudness_result = SimpleNamespace(
        summary={"steady_state_average_sone": 3.0}
    )

    widget._apply_loudness_scalar_limit(
        loudness_result,
        {
            "curve_limit_unit": "sone",
            "scalar_upper_enabled": True,
            "scalar_upper_value": 2.5,
            "scalar_lower_enabled": False,
            "curve_upper_enabled": True,
            "curve_upper_value": 100.0,
        },
        metric="steady_state_average",
    )

    assert widget.data_struct.analysis_result_dict["响度标量阈值"] == (
        False,
        pytest.approx(0.5),
    )
    widget.close()


def test_loudness_scalar_metric_does_not_draw_curve_limits(signal_module):
    class _PlotRecorder:
        def __init__(self):
            self.calls = []

        def plot(self, *args, **kwargs):
            self.calls.append((args, kwargs))

    plot = _PlotRecorder()
    signal_module.LoudnessAnalysis._draw_loudness_limit_lines(
        plot,
        {
            "limit_checked": True,
            "limit_metric": "steady_state_average",
            "scalar_upper_enabled": True,
            "scalar_upper_value": 2.0,
        },
        np.asarray([0.0, 1.0]),
    )

    assert plot.calls == []
