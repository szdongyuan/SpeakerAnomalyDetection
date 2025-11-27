import pytest
import sys
from unittest.mock import Mock
from PyQt5.QtWidgets import QApplication
from ui.operation_sequence import OptionList
from ui.ui_analysis_config.rb_config_dialog import RbConfigWindow


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for GUI tests"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_create_config_dialog_rb(qapp):
    """Verify create_config_dialog returns RbConfigWindow for type RB"""
    option_list = OptionList(None)

    config_manager = Mock()
    model_name = "RB"  # Use the type name as model name
    config_manager.load_config.return_value = {"RB": {"selected_labels": [10, 15, 20], "all_checked": False}}

    dialog = option_list.create_config_dialog(None, config_manager, model_name, "RB", 44100)

    assert isinstance(dialog, RbConfigWindow)
    assert dialog.config_manager == config_manager
