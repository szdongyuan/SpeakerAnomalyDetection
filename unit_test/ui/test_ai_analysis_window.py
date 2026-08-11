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
from PyQt5.QtWidgets import QApplication, QLabel

from consts import error_code


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
            build_blocked_ai_export_detail=lambda model_name, reason, message: {
                "label": None,
                "blocked_reason": reason,
                "blocked_message": message,
                "model_name": model_name,
            },
            should_validate_model_duration=lambda *args, **kwargs: True,
            validate_model_duration=lambda *args, **kwargs: (True, None),
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
    module_name = "_ai_signal_analysis_under_test"
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


def _prepare_ai_widget(signal_module, monkeypatch):
    widget = signal_module.AI("AI 分析 1")
    widget.analysis_config = {"analyse_model_name": "demo_model"}
    monkeypatch.setattr(
        widget,
        "get_model_info",
        lambda selected_model, logger: (
            error_code.OK,
            ("demo.keras", "demo.yml"),
        ),
    )
    monkeypatch.setattr(signal_module.os.path, "exists", lambda path: True)
    monkeypatch.setattr(
        signal_module,
        "resolve_analysis_channel_signal",
        lambda *args, **kwargs: np.zeros(16000, dtype=np.float32),
    )
    return widget


def _visible_ai_rows(widget):
    return [
        (name_label.text(), value_label.text())
        for row_widget, name_label, value_label in zip(
            widget.ai_row_widgets,
            widget.ai_row_name_labels,
            widget.ai_row_value_labels,
        )
        if not row_widget.isHidden()
    ]


def test_ai_window_uses_industrial_table_layout(signal_module, qapp):
    widget = signal_module.AI("AI 分析 1")
    title_label = widget.findChild(QLabel, "aiTitleLabel")
    title_label.ensurePolished()

    assert widget.objectName() == "aiAnalysisWindow"
    assert title_label.text() == "AI 分析"
    assert title_label.font().family() == "SimSun"
    assert not title_label.font().bold()
    assert _visible_ai_rows(widget) == [
        ("状态", "等待分析"),
        ("评分模型", "--"),
    ]
    assert len(widget.ai_row_widgets) == 5
    assert not hasattr(widget, "ai_analyse_score_textedit")
    stylesheet = widget.styleSheet()
    assert "background-color: #FFFFFF" in stylesheet
    assert "font-family: 'SimSun'" in stylesheet
    assert stylesheet.count("font-weight: 400") == 3
    assert "font-weight: 500" not in stylesheet
    assert "font-weight: 600" not in stylesheet
    assert "font-weight: 700" not in stylesheet
    assert "font-size: 30px" in stylesheet
    assert "font-size: 24px" in stylesheet
    assert "color: #000000" in stylesheet
    assert "background-color: #C7CDD6" in stylesheet
    assert "font-size:30px" in widget.ai_row_value_labels[0].styleSheet()
    assert "font-size:21px" in widget.ai_row_value_labels[1].styleSheet()
    assert "font-weight:400" in widget.ai_row_value_labels[0].styleSheet()
    assert "font-weight:400" in widget.ai_row_value_labels[1].styleSheet()
    assert widget.ai_row_name_labels[0].width() == 104
    assert widget.ai_row_widgets[0].layout().contentsMargins().top() == 11
    widget.close()


def test_duration_mismatch_stays_inside_ai_window_without_message_box(
    signal_module,
    qapp,
    monkeypatch,
):
    widget = _prepare_ai_widget(signal_module, monkeypatch)
    message = (
        "模型时长不匹配\n"
        "模型要求: 32000 点 (2.00s)\n"
        "当前音频: 16000 点 (1.00s)"
    )
    monkeypatch.setattr(
        signal_module,
        "validate_model_duration",
        lambda *args, **kwargs: (False, message),
    )
    popups = []
    monkeypatch.setattr(
        signal_module.QMessageBox,
        "information",
        lambda *args, **kwargs: popups.append((args, kwargs)),
    )

    widget.calculate_ai_scores("test", widget.analysis_config, "record")

    assert popups == []
    assert _visible_ai_rows(widget) == [
        ("状态", "模型时长不匹配"),
        ("评分模型", "demo_model"),
        ("模型要求", "32000 点 (2.00s)"),
        ("当前音频", "16000 点 (1.00s)"),
    ]
    assert widget.export_detail == {
        "label": None,
        "blocked_reason": "duration_mismatch",
        "blocked_message": message,
        "model_name": "demo_model",
    }
    widget.close()


def test_successful_ai_result_updates_industrial_table_rows(
    signal_module,
    qapp,
    monkeypatch,
):
    widget = _prepare_ai_widget(signal_module, monkeypatch)
    monkeypatch.setattr(
        signal_module,
        "validate_model_duration",
        lambda *args, **kwargs: (True, None),
    )

    def fake_model_predict(*args, **kwargs):
        widget.result = "OK"
        widget.export_detail = {
            "label": "OK",
            "ok_score": 87.25,
            "ng_score": 12.75,
            "model_name": "demo_model",
        }
        return (
            "评分结果: OK\n"
            "评分模型: demo_model\n"
            "OK Score: 87.25%\n"
            "NG Score: 12.75%"
        )

    monkeypatch.setattr(widget, "model_predict", fake_model_predict)

    widget.calculate_ai_scores("test", widget.analysis_config, "record")

    assert _visible_ai_rows(widget) == [
        ("最终判定", "OK"),
        ("评分模型", "demo_model"),
        ("OK Score", "87.25%"),
        ("NG Score", "12.75%"),
    ]
    assert "font-weight:700" in widget.ai_row_value_labels[0].styleSheet()
    assert "font-weight:400" in widget.ai_row_value_labels[1].styleSheet()
    assert "font-weight:400" in widget.ai_row_value_labels[2].styleSheet()
    assert "font-weight:400" in widget.ai_row_value_labels[3].styleSheet()
    widget.close()


def test_duration_mismatch_keeps_effective_preprocess_length_row(signal_module, qapp):
    widget = signal_module.AI("AI 分析 1")

    widget._show_ai_error(
        "模型时长不匹配\n"
        "模型要求: 144000 点 (3.00s)\n"
        "当前音频: 240000 点 (5.00s)\n"
        "按预处理后有效长度: 192000 点 (4.00s)",
        model_name="demo_model",
        mismatch=True,
    )

    assert _visible_ai_rows(widget) == [
        ("状态", "模型时长不匹配"),
        ("评分模型", "demo_model"),
        ("模型要求", "144000 点 (3.00s)"),
        ("当前音频", "240000 点 (5.00s)"),
        ("按预处理后有效长度", "192000 点 (4.00s)"),
    ]
    widget.close()
