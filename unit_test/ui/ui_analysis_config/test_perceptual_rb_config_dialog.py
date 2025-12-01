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
    """Verify perceptual rub&buzz only allows harmonics 10-35"""
    config_manager = Mock()
    config_manager.load_config.return_value = {"PRB": {"selected_labels": [10, 15, 20], "all_checked": False}}

    window = PerceptualRbConfigWindow(config_manager, "PRB")

    # Should have 26 labels (harmonics 10-35 inclusive)
    assert window.box_layout.count() == 26

    # First label should be harmonic 10
    first_label = window.box_layout.itemAt(0).widget()
    assert "10" in first_label.text()

    # Last label should be harmonic 35
    last_label = window.box_layout.itemAt(25).widget()
    assert "35" in last_label.text()


def test_perceptual_rb_config_window_title(qapp):
    """Verify window displays perceptual rub&buzz title"""
    config_manager = Mock()
    config_manager.load_config.return_value = {"PRB": {}}

    window = PerceptualRbConfigWindow(config_manager, "PRB")

    # Group box should say "Perceptual Rub & Buzz"
    group_box = window.findChild(object, name="harmonic_group_box")
    assert "Perceptual" in group_box.title() or "感知" in group_box.title()
