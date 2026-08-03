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

    assert "overall_spl" not in result
    assert titles == [""]
    widget.close()
