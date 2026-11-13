import unittest
from unittest.mock import patch
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QSplitter, QVBoxLayout, QWidget
from ui.sequence.motor_left_panel import MotorDetectionLeftPanel
from ui.sequence.motor_panel_common import MotorSectionCard
from ui.sequence.station_plot_grid import StationPlotGrid
from ui.sequence.workbench_layout import build_workbench_layout

class _DummySequenceWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.left_panel = MotorDetectionLeftPanel()

    def create_waveform_layout(self):
        return build_workbench_layout(self)

_RealisticSequenceWidget = _DummySequenceWidget

class TestSequenceMainLayout(unittest.TestCase):


    def test_video_and_hidden_history_match_stashed_layout(self):
        widget = _RealisticSequenceWidget()
        widget.setLayout(widget.create_waveform_layout())
        self.addCleanup(widget.close)
        widget.show()
        self.app.processEvents()
        video = widget.left_panel.video_monitor_panel
        self.assertEqual(video.findChildren(QPushButton), [])
        self.assertIn("实时预览", [label.text() for label in video.findChildren(QLabel)])
        self.assertFalse(any(type(child).__name__ == "RecentSessionPanel" for child in widget.findChildren(QWidget)))


    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_waveform_layout_uses_left_sidebar_and_full_height_workspace(self):
        widget = _DummySequenceWidget()

        layout = widget.create_waveform_layout()

        self.assertIsInstance(layout, QVBoxLayout)
        self.assertEqual(layout.count(), 1)
        margins = layout.contentsMargins()
        self.assertEqual(
            (margins.left(), margins.top(), margins.right(), margins.bottom()),
            (8, 12, 8, 12),
        )

        main_splitter = layout.itemAt(0).widget()
        self.assertIsInstance(main_splitter, QSplitter)
        self.assertEqual(main_splitter.orientation(), Qt.Horizontal)
        self.assertEqual(main_splitter.count(), 2)

        left_sidebar_splitter = main_splitter.widget(0)
        self.assertIsInstance(left_sidebar_splitter, QSplitter)
        self.assertEqual(left_sidebar_splitter.orientation(), Qt.Vertical)
        self.assertEqual(left_sidebar_splitter.count(), 2)
        self.assertIs(left_sidebar_splitter.widget(0), widget.left_panel.dashboard_panel)
        self.assertIs(left_sidebar_splitter.widget(1), widget.left_panel.video_monitor_panel)
        self.assertIs(main_splitter.widget(1), widget.plot_grid)
        self.assertIsInstance(widget.plot_grid, StationPlotGrid)
        self.assertFalse(any(type(child).__name__ == "RecentSessionPanel" for child in widget.findChildren(QWidget)))

    def test_common_window_size_keeps_waveform_and_history_readable(self):
        widget = _DummySequenceWidget()
        widget.setLayout(widget.create_waveform_layout())
        widget.resize(1400, 900)
        widget.show()
        self.app.processEvents()

        self.assertGreaterEqual(widget.plot_grid.width(), 900)
        self.assertGreaterEqual(widget.plot_grid.height(), 700)
        self.assertGreater(widget.plot_grid.width(), widget.left_panel.dashboard_panel.width())
        self.assertFalse(any(type(child).__name__ == "RecentSessionPanel" for child in widget.findChildren(QWidget)))

    def test_real_operation_panel_keeps_minimum_height_within_window_budget(self):
        widget = _RealisticSequenceWidget()
        widget.setLayout(widget.create_waveform_layout())
        widget.resize(1400, 700)
        widget.show()
        self.app.processEvents()

        self.assertLessEqual(widget.minimumSizeHint().height(), 700)

    def test_left_cards_stretch_to_match_their_row_heights(self):
        widget = _RealisticSequenceWidget()
        widget.setLayout(widget.create_waveform_layout())
        widget.resize(1400, 900)
        widget.show()
        self.app.processEvents()

        dashboard = widget.left_panel.dashboard_panel
        video_card = widget.left_panel.video_monitor_panel.findChild(MotorSectionCard)
        self.assertIsNotNone(video_card)
        self.assertEqual(video_card.height(), widget.left_panel.video_monitor_panel.height())
        self.assertTrue(dashboard.summary_panel.isVisible())
        self.assertEqual(widget.left_panel.get_station_count(), 0)
        self.assertGreater(dashboard.station_panel.height(), 0)
        self.assertLess(dashboard.station_panel.geometry().bottom(), dashboard.summary_panel.y())

    def test_initial_station_height_transfers_actual_video_space_once(self):
        for height in (1100, 700, 480):
            with self.subTest(height=height):
                widget = _RealisticSequenceWidget()
                self.addCleanup(widget.close)
                with patch("ui.sequence.workbench_layout.resolve_workspace_screen_size", return_value=(1920, 1080)):
                    widget.setLayout(widget.create_waveform_layout())
                widget.resize(1600, height)
                widget.show()
                splitter = widget.layout().itemAt(0).widget().widget(0)
                dashboard = widget.left_panel.dashboard_panel
                video = widget.left_panel.video_monitor_panel
                initial_sizes = splitter.sizes()
                station_height = dashboard.station_panel.height()
                summary_height = dashboard.summary_panel.height()
                video_minimum = max(video.minimumHeight(), video.minimumSizeHint().height())
                transfer = min(100, max(0, initial_sizes[1] - video_minimum))

                self.app.processEvents()

                self.assertEqual(splitter.sizes(), [initial_sizes[0] + transfer, initial_sizes[1] - transfer])
                self.assertEqual(dashboard.station_panel.height(), station_height + transfer)
                self.assertEqual(dashboard.summary_panel.height(), summary_height)
                self.assertGreaterEqual(video.height(), video_minimum)
                # Move the real splitter handle as a user would, then hide/show.
                splitter.moveSplitter(initial_sizes[0], 1)
                manually_adjusted = splitter.sizes()
                self.app.processEvents()
                widget.hide()
                widget.show()
                self.app.processEvents()
                self.assertEqual(splitter.sizes(), manually_adjusted)
                widget.resize(1600, widget.height() + 80)
                resized = splitter.sizes()
                self.app.processEvents()
                self.assertEqual(splitter.sizes(), resized)

    def test_initial_station_height_waits_for_first_visible_layout(self):
        widget = _RealisticSequenceWidget()
        self.addCleanup(widget.close)
        widget.setLayout(widget.create_waveform_layout())
        # Startup can process events while the sequence is still hidden.
        self.app.processEvents()
        widget.resize(1600, 1100)
        widget.show()
        splitter = widget.layout().itemAt(0).widget().widget(0)
        initial_sizes = splitter.sizes()

        self.app.processEvents()

        self.assertEqual(splitter.sizes(), [initial_sizes[0] + 100, initial_sizes[1] - 100])

    def test_initial_sidebar_width_transfers_actual_pixels_once(self):
        for width in (2400, 1920, 1600, 1100):
            with self.subTest(width=width):
                widget = _RealisticSequenceWidget()
                self.addCleanup(widget.close)
                with patch("ui.sequence.workbench_layout.resolve_workspace_screen_size", return_value=(1920, 1080)):
                    widget.setLayout(widget.create_waveform_layout())
                # Processing events while hidden must not consume the allocation.
                self.app.processEvents()
                widget.resize(width, 1100)
                widget.show()
                main = widget.layout().itemAt(0).widget()
                sidebar = main.widget(0)
                initial = main.sizes()
                minimum = max(sidebar.minimumWidth(), sidebar.minimumSizeHint().width())
                transfer = min(200, max(0, initial[0] - minimum))
                self.app.processEvents()
                self.assertEqual(main.sizes(), [initial[0] - transfer, initial[1] + transfer])
                self.assertEqual(widget.left_panel.dashboard_panel.width(), sidebar.width())
                self.assertEqual(widget.left_panel.video_monitor_panel.width(), sidebar.width())
                main.moveSplitter(minimum + 60, 1)
                manual = main.sizes()
                widget.hide()
                widget.show()
                self.app.processEvents()
                self.assertEqual(main.sizes(), manual)
                widget.resize(width + 100, 1180)
                self.app.processEvents()
                resized = main.sizes()
                self.app.processEvents()
                self.assertEqual(main.sizes(), resized)
