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
        """Verify PRB raises when config exists but required data is missing."""
        from ui.signal_analysis_window import PerceptualRubAndBuzz

        # Create PRB instance
        prb = PerceptualRubAndBuzz("Test PRB")

        # Mock data_struct
        prb.data_struct = Mock()
        prb.data_struct.store_wave_data = None

        # Set config without selected_labels key (PRB no longer relies on harmonic selection).
        prb.analysis_config = {"type": "PRB", "prb_method": "sc"}

        with self.assertRaises(ValueError):
            prb.calculate_thd()

    def test_prb_handles_empty_selected_labels(self):
        """Verify legacy selected_labels key does not change PRB behaviour."""
        from ui.signal_analysis_window import PerceptualRubAndBuzz

        # Create PRB instance
        prb = PerceptualRubAndBuzz("Test PRB")

        # Mock data_struct
        prb.data_struct = Mock()
        prb.data_struct.store_wave_data = None

        # Set config with legacy selected_labels (should be ignored now).
        prb.analysis_config = {"type": "PRB", "selected_labels": [], "all_checked": False, "prb_method": "iso"}

        with self.assertRaises(ValueError):
            prb.calculate_thd()


if __name__ == '__main__':
    unittest.main()
