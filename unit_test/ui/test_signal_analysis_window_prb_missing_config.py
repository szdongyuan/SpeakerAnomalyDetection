"""
Test that PRB handles missing or incomplete analysis_config gracefully.
"""
import unittest
from unittest.mock import Mock, MagicMock
from PyQt5.QtWidgets import QApplication
import sys

# Ensure QApplication exists
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)


class TestPrbMissingConfig(unittest.TestCase):
    """Test PRB handles missing config gracefully."""

    def test_prb_handles_none_config(self):
        """Verify PRB doesn't crash when analysis_config is None."""
        from ui.signal_analysis_window import PerceptualRubAndBuzz

        # Create PRB instance
        prb = PerceptualRubAndBuzz("Test PRB")

        # Mock data_struct
        prb.data_struct = Mock()
        prb.data_struct.store_wave_data = None

        # analysis_config should be None by default
        assert prb.analysis_config is None

        # Call calculate_thd - should not crash
        result = prb.calculate_thd()

        # Should return empty result
        assert result == {"freq_value": [], "harmonic": [], "thd": []}

    def test_prb_handles_missing_selected_labels(self):
        """Verify PRB doesn't crash when selected_labels is missing from config."""
        from ui.signal_analysis_window import PerceptualRubAndBuzz

        # Create PRB instance
        prb = PerceptualRubAndBuzz("Test PRB")

        # Mock data_struct
        prb.data_struct = Mock()
        prb.data_struct.store_wave_data = None

        # Set config without selected_labels key
        prb.analysis_config = {"type": "PRB", "all_checked": False}

        # Call calculate_thd - should not crash
        result = prb.calculate_thd()

        # Should return empty result (no harmonics selected)
        assert result == {"freq_value": [], "harmonic": [], "thd": []}

    def test_prb_handles_empty_selected_labels(self):
        """Verify PRB handles empty selected_labels list."""
        from ui.signal_analysis_window import PerceptualRubAndBuzz

        # Create PRB instance
        prb = PerceptualRubAndBuzz("Test PRB")

        # Mock data_struct
        prb.data_struct = Mock()
        prb.data_struct.store_wave_data = None

        # Set config with empty selected_labels
        prb.analysis_config = {"type": "PRB", "selected_labels": [], "all_checked": False}

        # Call calculate_thd - should not crash
        result = prb.calculate_thd()

        # Should return empty result
        assert result == {"freq_value": [], "harmonic": [], "thd": []}


if __name__ == '__main__':
    unittest.main()
