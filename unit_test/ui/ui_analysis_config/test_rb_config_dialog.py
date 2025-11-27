import pytest
import sys
from unittest.mock import Mock
from PyQt5.QtWidgets import QApplication
from ui.ui_analysis_config.rb_config_dialog import RbConfigWindow


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for GUI tests"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_rb_config_window_harmonic_range(qapp):
    """Verify rub&buzz only allows harmonics 10-35"""
    config_manager = Mock()
    config_manager.load_config.return_value = {"RB": {"selected_labels": [10, 15, 20], "all_checked": False}}

    window = RbConfigWindow(config_manager, "RB")

    # Should have 26 labels (harmonics 10-35 inclusive)
    assert window.box_layout.count() == 26

    # First label should be harmonic 10
    first_label = window.box_layout.itemAt(0).widget()
    assert "10" in first_label.text()

    # Last label should be harmonic 35
    last_label = window.box_layout.itemAt(25).widget()
    assert "35" in last_label.text()


def test_rb_config_window_select_all(qapp):
    """Verify select all checkbox selects harmonics 10-35"""
    config_manager = Mock()
    config_manager.load_config.return_value = {"RB": {"selected_labels": [], "all_checked": False}}

    window = RbConfigWindow(config_manager, "RB")

    # Trigger select all
    from PyQt5.QtCore import Qt
    window.on_select_all_changed(Qt.Checked)

    # Should select harmonics 10-35
    assert window.selected_labels == list(range(10, 36))


def test_rb_config_window_title(qapp):
    """Verify window displays rub&buzz title"""
    config_manager = Mock()
    config_manager.load_config.return_value = {"RB": {}}

    window = RbConfigWindow(config_manager, "RB")

    # Group box should say "Rub & Buzz"
    assert "Rub" in window.findChild(object, name="harmonic_group_box").title()
