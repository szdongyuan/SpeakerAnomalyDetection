import pytest
from unittest.mock import Mock, patch
from ui.operation_sequence import OptionList
from ui.ui_analysis_config.perceptual_rb_config_dialog import PerceptualRbConfigWindow
from PyQt5.QtWidgets import QApplication
import sys


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for GUI tests"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_create_config_dialog_prb(qapp):
    """Verify create_config_dialog returns PerceptualRbConfigWindow for type PRB"""
    option_list = OptionList(None)

    config_manager = Mock()
    config_manager.load_config.return_value = {"PRB": {"selected_labels": [10, 15, 20], "all_checked": False}}
    model_name = "PRB"

    dialog = option_list.create_config_dialog(None, config_manager, model_name, "PRB", 44100)

    assert isinstance(dialog, PerceptualRbConfigWindow)
    assert dialog.config_manager == config_manager
