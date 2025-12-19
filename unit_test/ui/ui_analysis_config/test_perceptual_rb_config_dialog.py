import pytest
import sys
from unittest.mock import Mock
from PyQt5.QtWidgets import QApplication
from ui.ui_analysis_config.perceptual_rb_config_dialog import PerceptualRbConfigWindow


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for GUI tests"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_perceptual_rb_config_window_harmonic_range(qapp):
    """Verify PRB config exposes loudness method selection."""
    config_manager = Mock()
    config_manager.load_config.return_value = {"PRB": {"prb_method": "sc"}}

    window = PerceptualRbConfigWindow(config_manager, "PRB")

    assert window.method_combo.count() == 2
    assert window.method_combo.itemText(0) == "sc"
    assert window.method_combo.itemText(1) == "iso226 and iso 532"

    # Should select the saved method.
    assert window.method_combo.currentData() == "sc"


def test_perceptual_rb_config_window_title(qapp):
    """Verify window displays PRB title."""
    config_manager = Mock()
    config_manager.load_config.return_value = {"PRB": {}}

    window = PerceptualRbConfigWindow(config_manager, "PRB")

    group_box = window.findChild(object, name="prb_group_box")
    assert group_box is not None
    assert "PRB" in group_box.title()
