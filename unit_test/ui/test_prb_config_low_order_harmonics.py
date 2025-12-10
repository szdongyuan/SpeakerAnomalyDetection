"""
Test PRB config dialog supports low-order harmonics (2-9).

Validates that UI configuration accepts harmonics in 2-35 range
after extending from original 10-35 range.
"""
import pytest
import sys
from unittest.mock import MagicMock
from PyQt5.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for GUI tests"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_prb_config_accepts_low_order_harmonics(qapp):
    """Test that PRB config accepts harmonics 2-9 in selected_labels."""
    # Import here to avoid PyQt dependencies in collection phase
    from ui.ui_analysis_config.perceptual_rb_config_dialog import PerceptualRbConfigWindow

    # Mock config manager with low-order harmonics
    mock_config_manager = MagicMock()
    mock_config_manager.load_config.return_value = {
        'PRB': {
            'selected_labels': [2, 3, 5, 7, 10, 15],  # Mix of low and high order
            'all_checked': False
        }
    }

    # Create dialog (should not reject low-order harmonics)
    dialog = PerceptualRbConfigWindow(mock_config_manager, 'PRB')

    # Verify all harmonics 2-35 are accepted
    expected_harmonics = [2, 3, 5, 7, 10, 15]
    assert dialog.selected_labels == expected_harmonics, \
        f"PRB should accept harmonics 2-35, got {dialog.selected_labels}"


def test_prb_config_select_all_includes_2_to_35(qapp):
    """Test that 'Select All' includes all harmonics from 2 to 35."""
    from ui.ui_analysis_config.perceptual_rb_config_dialog import PerceptualRbConfigWindow
    from PyQt5.QtCore import Qt

    # Mock config manager
    mock_config_manager = MagicMock()
    mock_config_manager.load_config.return_value = {
        'PRB': {
            'selected_labels': [],
            'all_checked': False
        }
    }

    # Create dialog
    dialog = PerceptualRbConfigWindow(mock_config_manager, 'PRB')

    # Trigger select all
    dialog.on_select_all_changed(Qt.Checked)

    # Verify all 34 harmonics (2-35) are selected
    expected = list(range(2, 36))  # 34 harmonics
    assert dialog.selected_labels == expected, \
        f"Select all should select harmonics 2-35 (34 total), got {len(dialog.selected_labels)}"
    assert len(dialog.selected_labels) == 34, \
        f"Should have 34 harmonics, got {len(dialog.selected_labels)}"


def test_prb_config_filters_out_of_range_harmonics(qapp):
    """Test that harmonics outside 2-35 range are filtered out."""
    from ui.ui_analysis_config.perceptual_rb_config_dialog import PerceptualRbConfigWindow

    # Mock config with out-of-range harmonics
    mock_config_manager = MagicMock()
    mock_config_manager.load_config.return_value = {
        'PRB': {
            'selected_labels': [1, 2, 10, 35, 36, 40],  # 1, 36, 40 are invalid
            'all_checked': False
        }
    }

    # Create dialog
    dialog = PerceptualRbConfigWindow(mock_config_manager, 'PRB')

    # Verify only 2-35 are kept
    expected = [2, 10, 35]
    assert dialog.selected_labels == expected, \
        f"Should filter to 2-35 range, got {dialog.selected_labels}"
