import pytest
from unittest.mock import Mock
from ui.ui_analysis_config.pipeline_pd_pm_config import PipelinePdPmConfigWindow
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


def test_create_child_dialog_prb(qapp):
    """Verify pipeline can create PRB child dialog"""
    config_manager = Mock()
    config_manager.load_config.return_value = {}

    pipeline = PipelinePdPmConfigWindow(config_manager, "TEST")

    dialog = pipeline._create_child_dialog_by_type("PRB", "TEST_PRB")

    assert isinstance(dialog, PerceptualRbConfigWindow)
