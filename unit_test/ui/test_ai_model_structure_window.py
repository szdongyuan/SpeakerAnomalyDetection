import importlib.util
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from PyQt5.QtWidgets import QApplication, QTextEdit


def _stub_module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


class _LogManager:
    @staticmethod
    def set_log_handler(name):
        return object()


def _load_ai_window_without_model_runtime():
    placeholder_class = type("Placeholder", (), {})
    stub_modules = {
        "base.evaluate_model": _stub_module(
            "base.evaluate_model",
            evaluate=lambda *args, **kwargs: None,
            evaluate_with_data=lambda *args, **kwargs: None,
        ),
        "base.file_ops": _stub_module("base.file_ops", FileOps=placeholder_class),
        "base.log_manager": _stub_module("base.log_manager", LogManager=_LogManager),
        "base.model_config": _stub_module(
            "base.model_config",
            init_model_from_config=lambda *args, **kwargs: None,
        ),
        "base.training_model": _stub_module(
            "base.training_model",
            train_with_dir=lambda *args, **kwargs: None,
            train_with_data=lambda *args, **kwargs: None,
            save_trained_model=lambda *args, **kwargs: None,
        ),
        "base.training_model_management": _stub_module(
            "base.training_model_management",
            TrainingModelManagement=placeholder_class,
        ),
        "ui.model_manager_widget": _stub_module(
            "ui.model_manager_widget",
            ModelInfoList=placeholder_class,
        ),
        "ui.ai_select_audio_data": _stub_module(
            "ui.ai_select_audio_data",
            SelectAudioDataView=placeholder_class,
        ),
    }
    previous_modules = {name: sys.modules.get(name) for name in stub_modules}
    module_name = "_ai_window_structure_under_test"
    try:
        sys.modules.update(stub_modules)
        spec = importlib.util.spec_from_file_location(
            module_name,
            PROJECT_ROOT / "ui" / "ai_window.py",
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
def ai_window_module():
    return _load_ai_window_without_model_runtime()


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_model_structure_uses_compact_shared_font(ai_window_module, qapp):
    summary = "Model: CNN1d\n| Layer (type) | Output Shape |"
    dialog = ai_window_module.AiBrainModelStructure(model_structure=summary)
    editor = dialog.model_structure_texteditor
    editor.ensurePolished()

    assert editor.objectName() == "modelStructureTextEdit"
    assert editor.isReadOnly()
    assert editor.lineWrapMode() == QTextEdit.NoWrap
    assert editor.toPlainText() == summary
    assert "font-family: 'SimSun'" in editor.styleSheet()
    assert "font-size: 15px" in editor.styleSheet()
    assert editor.font().family() == "SimSun"
    assert editor.font().pixelSize() == 15
    dialog.close()


def test_model_structure_empty_message_uses_same_editor(ai_window_module, qapp):
    dialog = ai_window_module.AiBrainModelStructure(model_structure=None)

    assert dialog.model_structure_texteditor.toPlainText() == "模型结构为空"
    assert "font-size: 15px" in dialog.model_structure_texteditor.styleSheet()
    dialog.close()


def test_training_process_uses_compact_shared_font(ai_window_module, qapp):
    dialog = ai_window_module.Process_Widget()
    editor = dialog.model_structure_texteditor
    editor.ensurePolished()

    assert editor.objectName() == "trainingProcessTextEdit"
    assert editor.isReadOnly()
    assert editor.lineWrapMode() == QTextEdit.NoWrap
    assert "font-family: 'SimSun'" in editor.styleSheet()
    assert "font-size: 15px" in editor.styleSheet()
    assert editor.font().family() == "SimSun"
    assert editor.font().pixelSize() == 15
    dialog.close()
