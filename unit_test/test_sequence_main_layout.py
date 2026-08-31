import logging
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QComboBox, QLabel, QPushButton, QSplitter, QVBoxLayout, QWidget

if "concurrent_log_handler" not in sys.modules:
    concurrent_log_handler = types.ModuleType("concurrent_log_handler")

    class _ConcurrentRotatingFileHandler(logging.Handler):
        def emit(self, record):
            return None

    concurrent_log_handler.ConcurrentRotatingFileHandler = _ConcurrentRotatingFileHandler
    sys.modules["concurrent_log_handler"] = concurrent_log_handler

from ui.sequence.motor_left_panel import MotorDetectionLeftPanel
from ui.sequence.motor_panel_common import MotorSectionCard
from ui.sequence.channel_plot_workspace import ChannelPlotWorkspace
from ui.sequence.analysis_waveform_panel import AnalysisWaveformPanel
from ui.sequence.direction_waveform_panel import DirectionWaveformPanel
from ui.sequence.multichannel_waveform_session import MultichannelWaveformSession
from ui.sequence.sequencement_count_board import SequenceCountBoard
from ui.sequence.sequence_widget_config_ops import SequenceWidgetConfigOpsMixin
from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin
from consts import error_code
from base.product_test_project_config import ProductTestProjectConfigManager


class _DummyCountBoard(QWidget):
    def __init__(self):
        super().__init__()
        self.mode = "test"
        self.mode_change_callbacks = []

    def register_mode_change_callback(self, callback):
        self.mode_change_callbacks.append(callback)


class _DummyChannelWorkspace:
    def __init__(self, keys):
        self._keys = list(keys)
        self.results = []
        self.audio_paths = []
        self.cleared_directions = []
        self.clear_all_count = 0
        self.direction_data = []
        self.set_condition_calls = []

    def condition_keys(self):
        return list(self._keys)

    def set_conditions(self, condition_configs):
        self.set_condition_calls.append(list(condition_configs or []))
        self._keys = [
            str(item.get("key") or "")
            for item in DirectionWaveformPanel._normalize_conditions(condition_configs)
        ]

    def set_condition_result(self, key, label):
        self.results.append((key, label))

    def set_condition_audio_path(self, key, path):
        self.audio_paths.append((key, path))

    def clear_direction(self, key):
        self.cleared_directions.append(key)

    def clear_plots(self):
        self.clear_all_count += 1

    def set_direction_data(self, key, x, y):
        self.direction_data.append((key, list(x), list(y)))

    def set_live_channels(self, channels):
        workspace = self

        class _LiveWindow:
            def __init__(self, channel):
                self.channel_index = channel

            def set_data(self, x, y):
                workspace.direction_data.append(
                    (self.channel_index, list(x), list(y))
                )

        self._live_windows = [_LiveWindow(channel) for channel in channels]

    def all_subwindows(self):
        return list(self._live_windows)


class _SpyLeftPanel:
    def __init__(self):
        self.results = []

    def set_condition_result(self, key, label, tone=None):
        self.results.append((key, label, tone))


class _DummyProductProgramWidget(QWidget, SequenceWidgetConfigOpsMixin):
    def __init__(self, manager):
        super().__init__()
        self.product_program_manager = manager
        self.using_file_combobox = QComboBox()


class _DummySequenceWidget(QWidget, SequenceWidgetStreamingOpsMixin):
    def closeEvent(self, event):
        # This fixture does not own recording hardware or background exporters.
        event.accept()

    def __init__(self):
        super().__init__()
        self.count_board = _DummyCountBoard()
        self.left_panel = MotorDetectionLeftPanel(self.count_board)
        self.channel_workspace = None
        self.recent_session_panel = None
        self.recent_test_session_by_id = {}
        self._last_recent_session_mode = ""

    def _resolve_recent_session(self, session_id: str):
        return None

    def _show_recent_session_analysis_by_id(self, session_id: str):
        return None

    def _change_recent_session_result_by_id(self, session_id: str, label: str):
        return None

    def _configure_direction_waveform_workspace(self):
        return None

    def _on_recent_session_mode_changed(self, mode: str):
        self._last_recent_session_mode = mode

    @staticmethod
    def _format_recent_session_result_label(result_label):
        normalized = str(result_label or "").strip()
        return normalized.lower() if normalized.lower() in ("ok", "ng") else "not labeled"

    def _update_recent_session(self, session_id: str, **fields):
        self.recent_test_session_by_id[session_id].update(fields)


class _RealisticSequenceWidget(_DummySequenceWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.count_board = SequenceCountBoard({})
        self.left_panel = MotorDetectionLeftPanel(self.count_board)
        self.channel_workspace = None
        self.recent_session_panel = None
        self._last_recent_session_mode = ""


class _WaveformRefreshWidget(SequenceWidgetStreamingOpsMixin):
    def __init__(self):
        self.product_test_condition_configs = [
            {"condition_name": "6000", "trigger_state": "01"},
            {"condition_name": "7000", "trigger_state": "02"},
            {"condition_name": "8000", "trigger_state": "03"},
        ]
        self.channel_workspace = _DummyChannelWorkspace(["01", "02", "03"])
        self._direction_waveform_condition_signature = self._product_condition_signature(
            self.product_test_condition_configs
        )
        self._direction_waveform_cache = {}
        self._condition_record_cache = {}
        self._waveform_display_override_direction = ""
        self._current_trigger_direction = ""
        self._active_product_condition_key = ""
        self._active_input_channels = [0]
        self.recorded_path = None
        self.recorded_signal_info = {}
        self._current_recent_session_id = ""
        self.channel_workspace.set_live_channels(self._active_input_channels)

    def _apply_condition_mode_to_waveforms(self):
        return None

    def _get_active_product_condition_key(self):
        return self._active_product_condition_key


def _begin_deferred_streaming_waveform_session(
    widget, *, sample_rate, startup_trim_samples, direction
):
    widget._recording_input_channels = (0, 1)
    widget.channel_workspace.set_live_channels(widget._recording_input_channels)
    widget._streaming_waveform_session = MultichannelWaveformSession(
        max_points=widget._WAVEFORM_DISPLAY_MAX_POINTS
    )
    widget._streaming_waveform_generation = 0
    widget._streaming_waveform_refresh_scheduled = False
    widget._streaming_waveform_pending = False
    widget._streaming_waveform_live_enabled = False
    widget._streaming_waveform_failure_logged = False
    widget._streaming_chunk_contract_failed = False
    scheduled_callbacks = []
    widget._schedule_streaming_waveform_callback = scheduled_callbacks.append
    widget._begin_streaming_waveform_session(
        sample_rate, startup_trim_samples, direction
    )
    return scheduled_callbacks


class _SyncLeftPanel:
    def __init__(self):
        self.set_calls = []
        self.refresh_calls = []

    def set_condition_configs(self, condition_configs):
        self.set_calls.append(list(condition_configs or []))

    def refresh_condition_configs(self, condition_configs):
        self.refresh_calls.append(list(condition_configs or []))
        return True


class _SyncRecentPanel:
    def __init__(self):
        self.set_calls = []

    def set_conditions(self, condition_configs):
        self.set_calls.append(list(condition_configs or []))


class _ProductConditionSyncWidget(SequenceWidgetStreamingOpsMixin):
    def __init__(self, condition_configs):
        self.product_test_condition_configs = list(condition_configs)
        self.left_panel = _SyncLeftPanel()
        self.channel_workspace = _DummyChannelWorkspace(
            [
                str(item.get("key") or "")
                for item in DirectionWaveformPanel._normalize_conditions(condition_configs)
            ]
        )
        self.recent_session_panel = _SyncRecentPanel()
        self.cleared_history = 0
        self.reset_cycles = []
        self.reset_display_count = 0
        self.apply_mode_count = 0

    def _get_active_product_program_path(self):
        return "active_program.json"

    def _clear_recent_session_history(self, reset_panel=True):
        self.cleared_history += 1

    def _reset_manual_product_condition_cycle(self, clear_waveforms=False):
        self.reset_cycles.append(bool(clear_waveforms))

    def _reset_product_condition_display_state(self):
        self.reset_display_count += 1

    def _apply_condition_mode_to_waveforms(self):
        self.apply_mode_count += 1


class TestSequenceMainLayout(unittest.TestCase):
    def test_waveform_restores_display_only_channel_rows_and_grid(self):
        panel = AnalysisWaveformPanel(
            condition_configs=[{"key": "import", "condition_name": "导入工况"}],
        )
        self.addCleanup(panel.close)
        panel.set_channels([0, 2])
        panel.set_mode("mark")
        panel.set_condition_audio_path("import", "fixture.wav")
        self.assertEqual(panel.findChildren(QPushButton), [])
        first, second = panel.all_subwindows()
        self.assertEqual(first.channel_caption.text(), "CH1")
        self.assertEqual(second.channel_caption.text(), "CH3")
        for waveform_row in (first, second):
            left_axis = waveform_row.plot_widget.getAxis("left")
            bottom_axis = waveform_row.plot_widget.getAxis("bottom")
            self.assertTrue(left_axis.isVisible())
            self.assertTrue(bottom_axis.isVisible())
            self.assertEqual(left_axis.labelText, "Amplitude")
            self.assertEqual(bottom_axis.labelText, "Time")
            self.assertEqual(bottom_axis.labelUnits, "s")
            self.assertEqual(
                waveform_row.plot_widget.getViewBox().state["mouseEnabled"],
                [True, True],
            )
            self.assertIn(
                "Grid",
                [
                    action.text()
                    for action in waveform_row.plot_widget.getPlotItem().getMenu().actions()
                ],
            )
            self.assertNotIn(
                "同步辅助网格",
                [
                    action.text()
                    for action in waveform_row.plot_widget.getPlotItem().getMenu().actions()
                ],
            )
            plot_item = waveform_row.plot_widget.getPlotItem()
            self.assertTrue(plot_item.ctrl.xGridCheck.isChecked())
            self.assertTrue(plot_item.ctrl.yGridCheck.isChecked())
        first.set_data([0, 1], [0.5, -0.5])
        saved = first.snapshot_plot_state()
        panel.clear_plots()
        self.assertIsNone(first.plot_item)
        self.assertTrue(first.plot_widget.getPlotItem().ctrl.xGridCheck.isChecked())
        self.assertTrue(first.plot_widget.getPlotItem().ctrl.yGridCheck.isChecked())
        first.restore_plot_state(saved)
        np.testing.assert_array_equal(first.plot_item.getData()[1], saved[1])

    def test_video_and_hidden_history_match_stashed_layout(self):
        widget = _RealisticSequenceWidget()
        widget.setLayout(widget.create_waveform_layout())
        self.addCleanup(widget.close)
        widget.show()
        self.app.processEvents()
        video = widget.left_panel.video_monitor_panel
        self.assertEqual(video.findChildren(QPushButton), [])
        self.assertIn("2K预览", [label.text() for label in video.findChildren(QLabel)])
        self.assertFalse(widget.count_board.isVisible())
        self.assertFalse(widget.recent_session_panel.isVisible())

    def test_waveform_rows_support_dynamic_and_noncontiguous_channels(self):
        panel = AnalysisWaveformPanel()
        panel.resize(1000, 780)
        panel.show()
        self.addCleanup(panel.close)
        for channels in ([0], [2, 7], list(range(5)), list(range(8))):
            with self.subTest(channels=channels):
                panel.set_channels(channels)
                for _ in range(3):
                    self.app.processEvents()
                windows = panel.all_subwindows()
                self.assertEqual([window.channel_index for window in windows], channels)
                self.assertTrue(panel._layout_is_valid())
                for index, window in enumerate(windows):
                    values = np.array([index, index + 1.0, -index - 1.0])
                    window.set_data([0.0, 0.1, 0.2], values)
                    np.testing.assert_array_equal(window.snapshot_plot_state()[1], values)

    def test_waveform_selection_clears_previous_condition_data_and_preserves_runtime(self):
        widget = _DummySequenceWidget()
        widget.product_test_condition_configs = [
            {"key": "first", "condition_name": "第一档", "test_queue": "queue1"},
            {"key": "second", "condition_name": "第二档", "test_queue": "queue2"},
        ]
        widget._get_product_program_manager = lambda: types.SimpleNamespace(load_queue_catalog=lambda: {
            "queue1": {"duration": 10, "analysis_items": ["SPL"]},
            "queue2": {"duration": 600, "analysis_items": ["FFT"]},
        })
        widget.left_panel.set_condition_configs(widget.product_test_condition_configs)
        widget.setLayout(widget.create_waveform_layout())
        self.addCleanup(widget.close)
        widget._refresh_waveform_condition_metadata()
        panel = widget.channel_workspace
        panel.all_subwindows()[0].set_data([0, 1], [0.5, -0.5])
        widget.left_panel.ai_result_panel.select_condition("second")
        self.assertIn("第二档", panel.current_condition_label.text())
        self.assertIn("600秒", panel.duration_label.text())
        self.assertIn("FFT", panel.test_items_label.text())
        self.assertIsNone(panel.all_subwindows()[0].plot_item)
        widget._get_active_product_condition_key = lambda: "first"
        widget._sync_waveform_condition_from_left("second")
        self.assertIn("第一档", panel.current_condition_label.text())
        self.assertIn("10秒", panel.duration_label.text())

    def test_channel_alias_does_not_change_physical_plot_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "channel_layout.json")
            panel = AnalysisWaveformPanel(channel_layout_path=path)
            panel.set_channels([7])
            window = panel.all_subwindows()[0]
            window.direction_editor.setText("后排")
            window.direction_editor.editingFinished.emit()
            self.assertEqual(window.channel_index, 7)
            self.assertEqual(window.channel_caption.text(), "CH8")
            restored = AnalysisWaveformPanel(channel_layout_path=path)
            restored.set_channels([7])
            self.assertEqual(restored.all_subwindows()[0].direction_editor.text(), "后排")
            panel.close()
            restored.close()

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
        self.assertIs(left_sidebar_splitter.widget(0), widget.left_panel.ai_result_panel)
        self.assertIs(left_sidebar_splitter.widget(1), widget.left_panel.video_monitor_panel)
        self.assertIs(main_splitter.widget(1), widget.channel_workspace)
        self.assertIsInstance(widget.channel_workspace, AnalysisWaveformPanel)
        self.assertTrue(widget.recent_session_panel.isHidden())

    def test_common_window_size_keeps_waveform_and_history_readable(self):
        widget = _DummySequenceWidget()
        widget.setLayout(widget.create_waveform_layout())
        widget.resize(1400, 900)
        widget.show()
        self.app.processEvents()

        self.assertGreaterEqual(widget.channel_workspace.width(), 900)
        self.assertGreaterEqual(widget.channel_workspace.height(), 700)
        self.assertGreater(widget.channel_workspace.width(), widget.left_panel.ai_result_panel.width())
        self.assertTrue(widget.recent_session_panel.isHidden())

    def test_threshold_analyses_can_output_ok_ng(self):
        widget = _DummySequenceWidget()

        for analysis_type in ("LOUD", "FBA", "FFT"):
            widget.analysis_config = {
                "display_sequence": ["item"],
                "item": {"type": analysis_type, "limit_checked": True},
            }

            self.assertEqual(widget._can_output_ok_ng(), (True, ""))

    def test_channel_workspace_hides_old_subwindows_before_rebuilding(self):
        workspace = ChannelPlotWorkspace()
        workspace.resize(800, 400)
        workspace.show()
        workspace.set_channels([0, 1])
        self.app.processEvents()
        old_windows = workspace.subwindows()
        self.assertTrue(old_windows)

        workspace.set_channels([0])

        self.assertTrue(all(not window.isVisible() for window in old_windows))
        workspace.close()

    def test_direction_waveform_panel_hides_old_cards_before_rebuilding(self):
        panel = DirectionWaveformPanel(
            condition_configs=[
                {"condition_name": "6000", "trigger_state": "01"},
                {"condition_name": "7000", "trigger_state": "02"},
            ]
        )
        panel.resize(800, 400)
        panel.show()
        self.app.processEvents()
        old_cards = list(panel._cards.values())
        self.assertTrue(old_cards)

        panel.set_conditions([{"condition_name": "8000", "trigger_state": "03"}])

        self.assertTrue(all(not card.isVisible() for card in old_cards))
        panel.close()

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

        ai_card = widget.left_panel.ai_result_panel.findChild(MotorSectionCard)
        video_card = widget.left_panel.video_monitor_panel.findChild(MotorSectionCard)

        self.assertIsNotNone(ai_card)
        self.assertIsNotNone(video_card)
        self.assertEqual(ai_card.height(), widget.left_panel.ai_result_panel.height())
        self.assertEqual(video_card.height(), widget.left_panel.video_monitor_panel.height())

        task_header = widget.left_panel.ai_result_panel.findChild(QWidget, "testTaskHeader")
        self.assertIsNotNone(task_header)
        self.assertEqual(task_header.height(), widget.left_panel.video_monitor_panel.header.height())

    def test_default_main_splitter_ratio_balances_left_sidebar(self):
        widget = _RealisticSequenceWidget()
        with patch.object(
            widget,
            "_resolve_workspace_screen_size",
            return_value=(1920, 1080),
        ):
            layout = widget.create_waveform_layout()
        widget.setLayout(layout)
        widget.resize(1920, 1080)
        widget.show()
        self.app.processEvents()

        main_splitter = layout.itemAt(0).widget()
        left_width, right_width = main_splitter.sizes()
        total_width = left_width + right_width

        self.assertGreater(total_width, 0)
        self.assertAlmostEqual(left_width / total_width, 0.30, delta=0.03)
        self.assertAlmostEqual(right_width / total_width, 0.70, delta=0.03)

    def test_count_board_keeps_test_summary_visible_in_mark_mode(self):
        board = SequenceCountBoard({})

        board.on_mark_btn_clicked()

        self.assertEqual(board.mode, "mark")
        self.assertEqual(board.stacked_widget.currentIndex(), 0)

        board.on_test_btn_clicked()

        self.assertEqual(board.mode, "test")
        self.assertEqual(board.stacked_widget.currentIndex(), 0)

    def test_count_board_rejects_test_mode_with_condition_threshold_prompt(self):
        reason = (
            "以下工况未启用阈值判定，无法进入测试模式：\n"
            "- 7000 rpm 未启用可自动输出 OK/NG 的规则阈值：queue_7000\n"
            "请启用所有工况的阈值，或使用标记模式。"
        )
        with patch.object(SequenceCountBoard, "set_test_text"), patch.object(
            SequenceCountBoard,
            "set_mark_text",
        ):
            board = SequenceCountBoard({})
            board.on_mark_btn_clicked()
            board.set_test_available(False, reason)

            with patch(
                "ui.sequence.sequencement_count_board.QMessageBox.information"
            ) as information:
                board.on_test_btn_clicked()

        information.assert_called_once_with(board, "提示", reason)
        self.assertEqual(board.mode, "mark")
        self.assertFalse(board.test_btn.isEnabled())

    def test_sync_product_conditions_preserves_state_when_signature_unchanged(self):
        condition_configs = [
            {"key": "q6000", "condition_name": "6000", "test_queue": "queue_6000"},
            {"key": "q7000", "condition_name": "7000", "test_queue": "queue_7000"},
        ]
        widget = _ProductConditionSyncWidget(condition_configs)

        with patch(
            "ui.sequence.sequence_widget_streaming_ops.LoadUiConfig.load_product_test_program_condition_configs",
            return_value=[dict(item) for item in condition_configs],
        ):
            widget._sync_product_test_conditions(clear_recent_history=False)

        self.assertEqual(widget.left_panel.set_calls, [])
        self.assertEqual(len(widget.left_panel.refresh_calls), 1)
        self.assertEqual(widget.channel_workspace.set_condition_calls, [])
        self.assertEqual(widget.recent_session_panel.set_calls, [])
        self.assertEqual(widget.cleared_history, 0)
        self.assertEqual(widget.reset_display_count, 0)
        self.assertEqual(widget.reset_cycles, [])
        self.assertEqual(widget.apply_mode_count, 1)

    def test_sync_product_conditions_rebuilds_when_forced_by_config_switch(self):
        condition_configs = [
            {"key": "q6000", "condition_name": "6000", "test_queue": "queue_6000"},
            {"key": "q7000", "condition_name": "7000", "test_queue": "queue_7000"},
        ]
        widget = _ProductConditionSyncWidget(condition_configs)

        with patch(
            "ui.sequence.sequence_widget_streaming_ops.LoadUiConfig.load_product_test_program_condition_configs",
            return_value=[dict(item) for item in condition_configs],
        ):
            widget._sync_product_test_conditions(clear_recent_history=True)

        self.assertEqual(len(widget.left_panel.set_calls), 1)
        self.assertEqual(widget.left_panel.refresh_calls, [])
        self.assertEqual(len(widget.channel_workspace.set_condition_calls), 1)
        self.assertEqual(widget.cleared_history, 1)
        self.assertEqual(widget.reset_display_count, 1)
        self.assertEqual(widget.reset_cycles, [False])
        self.assertEqual(widget.apply_mode_count, 1)

    def test_using_config_combobox_reads_product_project_registry(self):
        with tempfile.TemporaryDirectory() as folder:
            manager = ProductTestProjectConfigManager(
                folder,
                os.path.join(folder, "program_registry.json"),
                os.path.join(folder, "sequence_config_registry.json"),
            )
            manager.save_registry({
                "active_file": "4条波形.json",
                "configs": [
                    {"file": "2条波形.json", "project_name": "2条波形"},
                    {"file": "4条波形.json", "project_name": "4条波形"},
                ],
            })
            widget = _DummyProductProgramWidget(manager)
            widget.add_file_to_using_file_combobox()
            combo = widget.using_file_combobox
            self.assertEqual([combo.itemText(i) for i in range(combo.count())], ["2条波形", "4条波形"])
            self.assertEqual(combo.currentText(), "4条波形")
            self.assertEqual(combo.currentData(), "4条波形.json")

    def test_initial_product_program_population_does_not_emit_switch_signal(self):
        manager = types.SimpleNamespace(
            load_registry=lambda: {
                "active_file": "b.json",
                "configs": [
                    {"file": "a.json", "project_name": "配置 A"},
                    {"file": "b.json", "project_name": "配置 B"},
                ],
            }
        )
        widget = _DummyProductProgramWidget(manager)
        emitted_texts = []
        widget.using_file_combobox.currentTextChanged.connect(
            emitted_texts.append
        )

        widget.add_file_to_using_file_combobox()

        self.assertEqual(widget.using_file_combobox.currentData(), "b.json")
        self.assertEqual(emitted_texts, [])

    def test_active_product_program_reports_partial_threshold_configuration(self):
        manager = types.SimpleNamespace(
            load_registry=lambda: {
                "active_file": "motor.json",
                "configs": [{"file": "motor.json", "name": "motor"}],
            },
            load_project=lambda _file_name: (
                error_code.OK,
                {
                    "name": "motor",
                    "sub_configs": [
                        {
                            "condition_name": "6000 rpm",
                            "trigger_state": "01",
                            "test_queue": "queue_6000",
                        },
                        {
                            "condition_name": "7000 rpm",
                            "trigger_state": "02",
                            "test_queue": "queue_7000",
                        },
                    ],
                },
            ),
            validate_project=lambda _program, _file_name: {
                "is_usable": True,
                "is_test_mode_usable": False,
                "use_errors": [],
                "test_mode_errors": [
                    "7000 rpm 未启用可自动输出 OK/NG 的规则阈值：queue_7000"
                ],
            },
        )
        widget = _DummyProductProgramWidget(manager)

        available, reason = widget._active_product_program_test_mode_availability()

        self.assertFalse(available)
        self.assertIn("7000 rpm", reason)
        self.assertIn("请启用所有工况的阈值，或使用标记模式", reason)

    def test_waveform_condition_actions_follow_mode(self):
        played = []
        marked = []
        panel = DirectionWaveformPanel(
            condition_configs=[{"condition_name": "6000 rpm", "trigger_state": "01"}],
            on_play_condition=played.append,
            on_mark_condition=lambda key, label: marked.append((key, label)),
        )
        card = panel._cards["01"]
        margins = panel.grid.contentsMargins()

        self.assertFalse(card.play_btn.isHidden())
        self.assertTrue(card.mark_panel.isHidden())
        self.assertEqual((margins.left(), margins.top(), margins.right(), margins.bottom()), (8, 0, 8, 8))

        panel.set_mode("mark")
        self.assertFalse(card.mark_panel.isHidden())

        card.play_btn.click()
        card.ok_btn.click()

        self.assertEqual(played, ["01"])
        self.assertEqual(marked, [("01", "OK")])

    def test_waveform_card_click_preview_is_removed(self):
        panel = DirectionWaveformPanel(
            condition_configs=[{"condition_name": "6000 rpm", "trigger_state": "01"}],
        )
        card = panel._cards["01"]
        panel.set_direction_data("01", [0, 1], [0.0, 0.5])

        self.assertNotIn("mousePressEvent", type(card).__dict__)

    def test_waveform_panel_keeps_cards_when_condition_keys_repeat(self):
        panel = DirectionWaveformPanel(
            condition_configs=[
                {"condition_name": "6000", "trigger_state": "", "test_queue": "默认配置"},
                {"condition_name": "7000", "trigger_state": "", "test_queue": "3"},
                {"condition_name": "8000", "trigger_state": "", "test_queue": "3"},
            ]
        )

        self.assertEqual(len(panel._cards), 3)
        self.assertEqual(len(set(panel.condition_keys())), 3)

    def test_waveform_panel_does_not_seed_fake_waveforms(self):
        panel = DirectionWaveformPanel(
            condition_configs=[
                {"condition_name": "6000", "trigger_state": "01"},
                {"condition_name": "7000", "trigger_state": "02"},
            ]
        )

        self.assertIsNone(panel._cards["01"].plot_item)
        self.assertIsNone(panel._cards["02"].plot_item)

        panel.set_direction_data("01", [0, 1], [0.0, 0.5])

        self.assertIsNotNone(panel._cards["01"].plot_item)
        self.assertIsNone(panel._cards["02"].plot_item)

    def test_final_projection_does_not_reconfigure_condition_workspace(self):
        widget = _WaveformRefreshWidget()

        widget._configure_direction_waveform_workspace()

        self.assertEqual(widget.channel_workspace.set_condition_calls, [])
        self.assertEqual(widget.channel_workspace.direction_data, [])

        widget.plot_waveform_to_workspace(
            [0.0, 0.4],
            1.0,
            channel_mapping=(0,),
        )

        channel, time_axis, amplitude = widget.channel_workspace.direction_data[-1]
        self.assertEqual(channel, 0)
        self.assertTrue(np.array_equal(time_axis, [0.0, 1.0]))
        self.assertTrue(np.allclose(amplitude, [0.0, 0.4]))
        self.assertEqual(widget._direction_waveform_cache, {})

    def test_final_projection_routes_each_column_to_physical_channel_window(self):
        widget = _WaveformRefreshWidget()
        widget._active_input_channels = [0, 2]
        widget.channel_workspace.set_live_channels((0, 2))
        waveform = np.asarray(
            [[1.0, 10.0], [2.0, 20.0]],
            dtype=np.float32,
        )

        widget.plot_waveform_to_workspace(
            waveform,
            2.0,
            channel_mapping=(0, 2),
        )

        self.assertEqual(
            [entry[0] for entry in widget.channel_workspace.direction_data],
            [0, 2],
        )
        self.assertTrue(
            np.array_equal(widget.channel_workspace.direction_data[0][2], waveform[:, 0])
        )
        self.assertTrue(
            np.array_equal(widget.channel_workspace.direction_data[1][2], waveform[:, 1])
        )
        self.assertEqual(widget._direction_waveform_cache, {})

    def test_final_waveform_display_downsamples_peaks_per_physical_channel(self):
        widget = _WaveformRefreshWidget()
        waveform = np.zeros(10_000, dtype=np.float32)
        waveform[1_234] = -9.0
        waveform[8_765] = 8.0

        widget.plot_waveform_to_workspace(
            waveform,
            1_000.0,
            channel_mapping=(0,),
        )

        channel, display_x, display_y = widget.channel_workspace.direction_data[-1]
        self.assertEqual(channel, 0)
        self.assertEqual(len(display_x), len(display_y))
        self.assertLessEqual(len(display_y), widget._WAVEFORM_DISPLAY_MAX_POINTS)
        self.assertEqual(min(display_y), -9.0)
        self.assertEqual(max(display_y), 8.0)
        self.assertEqual(display_x[0], 0.0)
        self.assertAlmostEqual(display_x[-1], 9.999)
        self.assertEqual(widget._direction_waveform_cache, {})

    def test_streaming_writer_receives_full_chunk_when_display_is_downsampled(self):
        class _Writer:
            def __init__(self):
                self.chunks = []

            def write_chunk(self, chunk):
                self.chunks.append(chunk)

        widget = _WaveformRefreshWidget()
        chunk = np.zeros((6_000, 2), dtype=np.float32)
        chunk[1_234] = -9.0
        chunk[5_678] = 8.0
        widget.streaming_wav_writer = _Writer()
        widget._active_product_condition_key = "01"
        widget._streaming_first_chunk_logged = True
        widget.default_logger = logging.getLogger(__name__)
        scheduled_callbacks = _begin_deferred_streaming_waveform_session(
            widget,
            sample_rate=48_000,
            startup_trim_samples=0,
            direction="01",
        )

        widget.on_audio_chunk_received({"multi": chunk})

        self.assertEqual(widget.channel_workspace.direction_data, [])
        self.assertEqual(len(scheduled_callbacks), 1)
        self.assertEqual(len(widget.streaming_wav_writer.chunks), 1)
        self.assertTrue(np.array_equal(widget.streaming_wav_writer.chunks[0], chunk))
        scheduled_callbacks.pop()()

        self.assertEqual(
            [entry[0] for entry in widget.channel_workspace.direction_data],
            [0, 1],
        )
        for channel, display_x, display_y in widget.channel_workspace.direction_data:
            self.assertLessEqual(len(display_y), widget._WAVEFORM_DISPLAY_MAX_POINTS)
            self.assertEqual(min(display_y), -9.0)
            self.assertEqual(max(display_y), 8.0)
            self.assertEqual(display_x[0], 0.0)
            self.assertAlmostEqual(display_x[-1], 5_999 / 48_000)
        self.assertNotIn("01", widget._direction_waveform_cache)

    def test_streaming_waveform_hides_startup_trim_but_writer_keeps_raw_chunks(self):
        class _Writer:
            def __init__(self):
                self.chunks = []

            def write_chunk(self, chunk):
                self.chunks.append(chunk)

        widget = _WaveformRefreshWidget()
        full_audio = np.arange(280, dtype=np.float32).reshape(140, 2)
        first_chunk = full_audio[:60]
        second_chunk = full_audio[60:]
        widget.streaming_wav_writer = _Writer()
        widget._active_product_condition_key = "01"
        widget._streaming_first_chunk_logged = True
        widget.default_logger = logging.getLogger(__name__)
        scheduled_callbacks = _begin_deferred_streaming_waveform_session(
            widget,
            sample_rate=1_000,
            startup_trim_samples=100,
            direction="01",
        )

        widget.on_audio_chunk_received({"multi": first_chunk})

        self.assertEqual(len(scheduled_callbacks), 1)
        scheduled_callbacks.pop()()
        self.assertEqual(
            widget.channel_workspace.direction_data,
            [(0, [], []), (1, [], [])],
        )
        self.assertNotIn("01", widget._direction_waveform_cache)
        self.assertTrue(np.array_equal(widget.streaming_wav_writer.chunks[0], first_chunk))

        widget.on_audio_chunk_received({"multi": second_chunk})

        self.assertEqual(len(scheduled_callbacks), 1)
        scheduled_callbacks.pop()()
        second_refresh = widget.channel_workspace.direction_data[-2:]
        self.assertEqual([entry[0] for entry in second_refresh], [0, 1])
        for channel, display_x, display_y in second_refresh:
            expected_waveform = full_audio[100:, channel]
            self.assertEqual(len(display_y), 40)
            self.assertEqual(display_x[0], 0.0)
            self.assertAlmostEqual(display_x[-1], 0.039)
            self.assertTrue(np.array_equal(display_y, expected_waveform))
        for accumulator in widget._streaming_waveform_session._accumulators.values():
            self.assertEqual(accumulator.raw_sample_count, 140)
            self.assertEqual(accumulator.display_sample_count, 40)
            self.assertEqual(accumulator.capacity, 0)
        self.assertNotIn("01", widget._direction_waveform_cache)
        self.assertEqual(len(widget.streaming_wav_writer.chunks), 2)
        self.assertTrue(np.array_equal(widget.streaming_wav_writer.chunks[0], first_chunk))
        self.assertTrue(np.array_equal(widget.streaming_wav_writer.chunks[1], second_chunk))

    def test_invalid_serial_recording_uses_whole_round_abort(self):
        calls = []

        class _InvalidRecordingHost:
            _serial_product_condition_executing = True

            def _on_serial_product_runtime_error(self, reason):
                calls.append(reason)
                return True

        SequenceWidgetStreamingOpsMixin._handle_invalid_recording(
            _InvalidRecordingHost(),
            "empty audio",
        )

        self.assertEqual(len(calls), 1)
        self.assertIn("empty audio", calls[0])

    def test_waveform_panel_resets_old_grid_columns_after_config_switch(self):
        panel = DirectionWaveformPanel(
            condition_configs=[
                {"condition_name": str(i), "trigger_state": str(i)}
                for i in range(5)
            ]
        )
        self.assertEqual(panel.grid.columnStretch(2), 1)

        panel.set_conditions(
            [
                {"condition_name": "6000", "trigger_state": "01"},
                {"condition_name": "7000", "trigger_state": "02"},
            ]
        )

        self.assertEqual(panel.grid.columnStretch(0), 1)
        self.assertEqual(panel.grid.columnStretch(1), 1)
        self.assertEqual(panel.grid.columnStretch(2), 0)
        self.assertEqual(panel.grid.rowStretch(1), 0)

    def test_clear_plot_area_clears_channel_plots_without_deleting_condition_state(self):
        widget = _DummySequenceWidget()
        widget.channel_workspace = _DummyChannelWorkspace(["01", "02", "03"])
        widget._waveform_display_override_direction = "03"
        widget._current_trigger_direction = ""
        widget._direction_waveform_cache = {
            "01": ("wave_6000", 1.0),
            "02": ("wave_7000", 1.0),
            "03": ("wave_8000", 1.0),
        }
        widget._condition_record_cache = {
            "01": {"recorded_path": "6000.wav"},
            "02": {"recorded_path": "7000.wav"},
            "03": {"recorded_path": "8000.wav"},
        }

        widget._clear_plot_area()

        self.assertEqual(widget._direction_waveform_cache["01"], ("wave_6000", 1.0))
        self.assertEqual(widget._direction_waveform_cache["02"], ("wave_7000", 1.0))
        self.assertEqual(widget._direction_waveform_cache["03"], ("wave_8000", 1.0))
        self.assertIn("01", widget._condition_record_cache)
        self.assertIn("02", widget._condition_record_cache)
        self.assertIn("03", widget._condition_record_cache)
        self.assertEqual(widget.channel_workspace.cleared_directions, [])
        self.assertEqual(widget.channel_workspace.clear_all_count, 1)

    def test_waveform_mark_does_not_update_left_condition_judgement(self):
        widget = _DummySequenceWidget()
        widget.count_board.mode = "mark"
        widget.channel_workspace = _DummyChannelWorkspace(["01"])
        widget.left_panel = _SpyLeftPanel()
        widget._condition_record_cache = {}
        widget.recorded_path = None
        widget.recorded_signal_info = {}

        with patch("ui.sequence.sequence_widget_streaming_ops.QMessageBox.warning") as warning:
            widget.on_waveform_condition_mark_clicked("01", "OK")

        warning.assert_called_once()
        self.assertEqual(widget.channel_workspace.results, [])
        self.assertEqual(widget.left_panel.results, [])

    def test_waveform_mark_updates_bound_recent_session_result(self):
        widget = _DummySequenceWidget()
        widget.count_board.mode = "mark"
        widget.channel_workspace = _DummyChannelWorkspace(["01"])

        with tempfile.TemporaryDirectory() as folder:
            old_path = os.path.join(folder, "old.wav")
            new_path = os.path.join(folder, "ok.wav")
            with open(old_path, "wb") as f:
                f.write(b"RIFF")
            widget._condition_record_cache = {
                "01": {
                    "recorded_path": old_path,
                    "recorded_signal_info": {"file_path": old_path, "labels": "not_labeled"},
                    "session_id": "recent_1",
                }
            }
            widget.recent_test_session_by_id = {
                "recent_1": {
                    "session_id": "recent_1",
                    "condition_key": "01",
                    "result_label": "not labeled",
                    "recorded_path": old_path,
                    "recorded_signal_info": {"file_path": old_path, "labels": "not_labeled"},
                }
            }
            widget._relabel_stored_audio_record = lambda _path, _info, label: (
                error_code.OK,
                "ok",
                new_path,
                {"file_path": new_path, "labels": label},
            )

            widget.on_waveform_condition_mark_clicked("01", "OK")

        session_record = widget.recent_test_session_by_id["recent_1"]
        self.assertEqual(session_record["result_label"], "ok")
        self.assertEqual(session_record["recorded_path"], new_path)
        self.assertEqual(session_record["recorded_signal_info"]["labels"], "OK")
        self.assertEqual(widget.channel_workspace.results, [("01", "OK")])

    def test_waveform_mark_with_missing_session_id_updates_matching_condition(self):
        widget = _DummySequenceWidget()
        widget.count_board.mode = "mark"
        widget.channel_workspace = _DummyChannelWorkspace(["01", "03"])
        widget._current_recent_session_id = "recent_3"

        with tempfile.TemporaryDirectory() as folder:
            old_path = os.path.join(folder, "6000.wav")
            new_path = os.path.join(folder, "6000_ng.wav")
            with open(old_path, "wb") as f:
                f.write(b"RIFF")
            widget._condition_record_cache = {
                "01": {
                    "recorded_path": old_path,
                    "recorded_signal_info": {
                        "file_path": old_path,
                        "labels": "not_labeled",
                    },
                    "session_id": "",
                }
            }
            widget.recent_test_sessions = ["old_1", "recent_1", "recent_3"]
            widget.recent_test_session_by_id = {
                "old_1": {
                    "session_id": "old_1",
                    "group_id": "old_group",
                    "condition_key": "01",
                    "result_label": "not labeled",
                    "recorded_path": "old_6000.wav",
                },
                "recent_1": {
                    "session_id": "recent_1",
                    "group_id": "group_1",
                    "condition_key": "01",
                    "result_label": "not labeled",
                    "recorded_path": old_path,
                    "recorded_signal_info": {
                        "file_path": old_path,
                        "labels": "not_labeled",
                    },
                },
                "recent_3": {
                    "session_id": "recent_3",
                    "group_id": "group_1",
                    "condition_key": "03",
                    "result_label": "not labeled",
                    "recorded_path": "7500.wav",
                    "recorded_signal_info": {
                        "file_path": "7500.wav",
                        "labels": "not_labeled",
                    },
                },
            }
            widget._relabel_stored_audio_record = lambda _path, _info, label: (
                error_code.OK,
                "ok",
                new_path,
                {"file_path": new_path, "labels": label},
            )

            widget.on_waveform_condition_mark_clicked("01", "NG")

        self.assertEqual(widget.recent_test_session_by_id["recent_1"]["result_label"], "ng")
        self.assertEqual(
            widget.recent_test_session_by_id["recent_3"]["result_label"],
            "not labeled",
        )
        self.assertEqual(
            widget.recent_test_session_by_id["old_1"]["result_label"],
            "not labeled",
        )
        self.assertEqual(widget._condition_record_cache["01"]["session_id"], "recent_1")
        self.assertEqual(widget.channel_workspace.results, [("01", "NG")])

    def test_waveform_mark_imported_record_does_not_bind_recording_session(self):
        widget = _DummySequenceWidget()
        widget.count_board.mode = "mark"
        widget.channel_workspace = _DummyChannelWorkspace(["01", "03"])
        widget._current_recent_session_id = "recent_3"

        with tempfile.TemporaryDirectory() as folder:
            old_path = os.path.join(folder, "imported_6000.wav")
            new_path = os.path.join(folder, "imported_6000_ok.wav")
            with open(old_path, "wb") as f:
                f.write(b"RIFF")
            widget._condition_record_cache = {
                "01": {
                    "recorded_path": old_path,
                    "recorded_signal_info": {
                        "file_path": old_path,
                        "labels": "not_labeled",
                        "source_type": "imported",
                    },
                    "session_id": "",
                }
            }
            widget.recorded_path = old_path
            widget.recorded_signal_info = dict(
                widget._condition_record_cache["01"]["recorded_signal_info"]
            )
            current_session_updates = []
            widget._update_current_recent_session_result = current_session_updates.append
            widget.recent_test_sessions = ["recent_1", "recent_3"]
            widget.recent_test_session_by_id = {
                "recent_1": {
                    "session_id": "recent_1",
                    "group_id": "group_1",
                    "condition_key": "01",
                    "result_label": "not labeled",
                    "recorded_path": old_path,
                },
                "recent_3": {
                    "session_id": "recent_3",
                    "group_id": "group_1",
                    "condition_key": "03",
                    "result_label": "not labeled",
                    "recorded_path": "7500.wav",
                },
            }
            widget._relabel_stored_audio_record = lambda _path, _info, label: (
                error_code.OK,
                "ok",
                new_path,
                {
                    "file_path": new_path,
                    "labels": label,
                    "source_type": "imported",
                },
            )

            widget.on_waveform_condition_mark_clicked("01", "OK")

        self.assertEqual(
            widget.recent_test_session_by_id["recent_1"]["result_label"],
            "not labeled",
        )
        self.assertEqual(
            widget.recent_test_session_by_id["recent_3"]["result_label"],
            "not labeled",
        )
        self.assertEqual(widget._condition_record_cache["01"]["session_id"], "")
        self.assertEqual(current_session_updates, [])
        self.assertEqual(widget.channel_workspace.results, [("01", "OK")])

    def test_waveform_mark_falls_back_to_recent_history_when_cache_missing(self):
        widget = _DummySequenceWidget()
        widget.count_board.mode = "mark"
        widget.channel_workspace = _DummyChannelWorkspace(["01", "02"])
        widget._condition_record_cache = {}
        widget._waveform_display_override_direction = ""
        widget._current_trigger_direction = ""
        widget.recorded_path = None
        widget.recorded_signal_info = {}

        with tempfile.TemporaryDirectory() as folder:
            old_path = os.path.join(folder, "history.wav")
            new_path = os.path.join(folder, "ok.wav")
            with open(old_path, "wb") as f:
                f.write(b"RIFF")
            widget.recent_test_sessions = ["recent_1"]
            widget.recent_test_session_by_id = {
                "recent_1": {
                    "session_id": "recent_1",
                    "group_id": "group_1",
                    "condition_key": "01",
                    "result_label": "not labeled",
                    "recorded_path": old_path,
                    "recorded_signal_info": {"file_path": old_path, "labels": "not_labeled"},
                }
            }
            widget._relabel_stored_audio_record = lambda _path, _info, label: (
                error_code.OK,
                "ok",
                new_path,
                {"file_path": new_path, "labels": label},
            )

            with patch("ui.sequence.sequence_widget_streaming_ops.QMessageBox.warning") as warning:
                widget.on_waveform_condition_mark_clicked("01", "OK")

        warning.assert_not_called()
        session_record = widget.recent_test_session_by_id["recent_1"]
        self.assertEqual(session_record["recorded_path"], new_path)
        self.assertEqual(session_record["recorded_signal_info"]["labels"], "OK")
        self.assertEqual(widget._condition_record_cache["01"]["recorded_path"], new_path)
        self.assertEqual(widget.channel_workspace.results, [("01", "OK")])


if __name__ == "__main__":
    unittest.main()
