"""
PRB config dialog no longer exposes harmonic-order selection.

PRB is computed with a fixed harmonic range (2nd-35th), and users can only choose
which loudness method to run ("sc" vs "iso226 and iso 532").
"""

import sys
from unittest.mock import MagicMock

import pytest
from PyQt5.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_prb_config_dialog_defaults_to_iso(qapp):
    from ui.ui_analysis_config.perceptual_rb_config_dialog import PerceptualRbConfigWindow

    mock_config_manager = MagicMock()
    mock_config_manager.load_config.return_value = {"PRB": {}}

    dialog = PerceptualRbConfigWindow(mock_config_manager, "PRB")
    assert dialog.method_combo.currentData() == "iso"


def test_prb_config_dialog_accepts_legacy_selected_labels_without_crashing(qapp):
    from ui.ui_analysis_config.perceptual_rb_config_dialog import PerceptualRbConfigWindow

    # Old configs used to store selected_labels/all_checked; ensure they don't break the dialog.
    mock_config_manager = MagicMock()
    mock_config_manager.load_config.return_value = {
        "PRB": {
            "selected_labels": [1, 2, 10, 35, 36],
            "all_checked": False,
        }
    }

    dialog = PerceptualRbConfigWindow(mock_config_manager, "PRB")
    assert dialog.method_combo.currentData() == "iso"


def test_prb_config_dialog_selects_sc_when_saved(qapp):
    from ui.ui_analysis_config.perceptual_rb_config_dialog import PerceptualRbConfigWindow

    mock_config_manager = MagicMock()
    mock_config_manager.load_config.return_value = {"PRB": {"prb_method": "sc"}}

    dialog = PerceptualRbConfigWindow(mock_config_manager, "PRB")
    assert dialog.method_combo.currentData() == "sc"
    assert dialog.get_default_config() == {"prb_method": "sc"}

