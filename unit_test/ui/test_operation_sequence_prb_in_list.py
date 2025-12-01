"""
Test that PRB appears in the analysis items list in operation_sequence.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
from PyQt5.QtWidgets import QApplication
import sys

# Ensure QApplication exists
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)


class TestPrbInAnalysisList(unittest.TestCase):
    """Test PRB is available in the UI analysis list."""

    @patch('ui.operation_sequence.LogManager')
    def test_prb_in_analysis_items_list(self, mock_log_manager):
        """Verify PRB appears in analysis items list."""
        from ui.operation_sequence import AnalysisModelSelect

        # Create the analysis model select dialog
        dialog = AnalysisModelSelect()

        # Get the model
        model = dialog.analysis_model

        # Find "音频分析" parent item (should be at index 1)
        audio_analysis_item = model.item(1, 0)
        assert audio_analysis_item is not None
        assert audio_analysis_item.text() == "音频分析"

        # Get all child items
        child_texts = []
        for row in range(audio_analysis_item.rowCount()):
            child_item = audio_analysis_item.child(row, 0)
            child_texts.append(child_item.text())

        # Verify PRB is in the list
        assert "感知高阶谐波失真 (PRB) " in child_texts, \
            f"PRB not found in analysis items. Available items: {child_texts}"

        # Verify RB is still there (regression check)
        assert "高阶谐波失真 (RB) " in child_texts


if __name__ == '__main__':
    unittest.main()
