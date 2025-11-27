import pytest
import sys
from unittest.mock import Mock
from PyQt5.QtWidgets import QApplication
from ui.ui_analysis_config.pipeline_pd_pm_config import PipelinePdPmConfigWindow
from ui.ui_analysis_config.rb_config_dialog import RbConfigWindow


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for GUI tests"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_create_child_dialog_rb(qapp):
    """Verify pipeline can create RB child dialog"""
    config_manager = Mock()
    config_manager.load_config.return_value = {}

    pipeline = PipelinePdPmConfigWindow(config_manager, "TEST")

    dialog = pipeline._create_child_dialog_by_type("RB", "TEST_RB")

    assert isinstance(dialog, RbConfigWindow)
