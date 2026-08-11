import logging
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QComboBox, QSplitter, QVBoxLayout, QWidget

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
from ui.sequence.direction_waveform_panel import DirectionWaveformPanel
from ui.sequence.sequencement_count_board import SequenceCountBoard
from ui.sequence.sequence_widget_config_ops import SequenceWidgetConfigOpsMixin
from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin
from consts import error_code
from base.product_test_program_config import ProductTestProgramConfigManager


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

    def _apply_condition_mode_to_waveforms(self):
        return None

    def _get_active_product_condition_key(self):
        return self._active_product_condition_key


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
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_waveform_layout_uses_top_bottom_split_with_two_horizontal_rows(self):
        widget = _DummySequenceWidget()

        layout = widget.create_waveform_layout()

        self.assertIsInstance(layout, QVBoxLayout)
        self.assertEqual(layout.count(), 1)

        main_splitter = layout.itemAt(0).widget()
        self.assertIsInstance(main_splitter, QSplitter)
        self.assertEqual(main_splitter.orientation(), Qt.Vertical)
        self.assertEqual(main_splitter.count(), 2)

        top_row_splitter = main_splitter.widget(0)
        bottom_row_splitter = main_splitter.widget(1)
        self.assertIsInstance(top_row_splitter, QSplitter)
        self.assertIsInstance(bottom_row_splitter, QSplitter)
        self.assertEqual(top_row_splitter.orientation(), Qt.Horizontal)
        self.assertEqual(bottom_row_splitter.orientation(), Qt.Horizontal)

        self.assertIs(top_row_splitter.widget(0), widget.left_panel.ai_result_panel)
        self.assertIs(top_row_splitter.widget(1), widget.channel_workspace)
        self.assertIs(bottom_row_splitter.widget(0), widget.left_panel.summary_panel)
        self.assertIs(bottom_row_splitter.widget(1), widget.recent_session_panel)

    def test_common_window_size_keeps_waveform_and_history_readable(self):
        widget = _DummySequenceWidget()
        widget.setLayout(widget.create_waveform_layout())
        widget.resize(1400, 900)
        widget.show()
        self.app.processEvents()

        self.assertGreaterEqual(widget.channel_workspace.width(), 700)
        self.assertGreaterEqual(widget.channel_workspace.height(), 360)
        self.assertGreaterEqual(widget.recent_session_panel.width(), 700)
        self.assertGreaterEqual(widget.recent_session_panel.height(), 300)
        self.assertGreater(widget.channel_workspace.width(), widget.left_panel.ai_result_panel.width())
        self.assertGreater(widget.recent_session_panel.width(), widget.left_panel.summary_panel.width())

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
        summary_card = widget.left_panel.summary_panel.findChild(MotorSectionCard)

        self.assertIsNotNone(ai_card)
        self.assertIsNotNone(summary_card)
        self.assertEqual(ai_card.height(), widget.left_panel.ai_result_panel.height())
        self.assertEqual(summary_card.height(), widget.left_panel.summary_panel.height())

        recent_card = widget.recent_session_panel.layout().itemAt(0).widget()
        recent_title = recent_card.layout().itemAt(0).widget()
        summary_title = summary_card.layout().itemAt(0).widget()
        self.assertEqual(recent_title.sizeHint().height(), summary_title.sizeHint().height())

    def test_default_main_splitter_ratio_is_45_to_55(self):
        widget = _RealisticSequenceWidget()
        layout = widget.create_waveform_layout()
        widget.setLayout(layout)
        widget.resize(1400, 900)
        widget.show()
        self.app.processEvents()

        main_splitter = layout.itemAt(0).widget()
        top_height, bottom_height = main_splitter.sizes()
        total_height = top_height + bottom_height

        self.assertGreater(total_height, 0)
        self.assertAlmostEqual(top_height / total_height, 0.45, delta=0.03)
        self.assertAlmostEqual(bottom_height / total_height, 0.55, delta=0.03)

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

    def test_using_config_combobox_reads_product_program_registry(self):
        with tempfile.TemporaryDirectory() as folder:
            program_dir = os.path.join(folder, "product_test_programs")
            queue_dir = os.path.join(folder, "analysis_sequence_config")
            os.makedirs(program_dir)
            os.makedirs(queue_dir)
            manager = ProductTestProgramConfigManager(
                program_dir,
                os.path.join(program_dir, "program_registry.json"),
                os.path.join(queue_dir, "sequence_config_registry.json"),
            )
            manager.save_program(
                None,
                {
                    "name": "2条波形",
                    "sub_configs": [{"condition_name": "6000 rpm", "trigger_state": "01"}],
                },
            )
            manager.save_program(
                None,
                {
                    "name": "4条波形",
                    "sub_configs": [{"condition_name": "9000 rpm", "trigger_state": "08"}],
                },
            )
            widget = _DummyProductProgramWidget(manager)

            widget.add_file_to_using_file_combobox()

            texts = [widget.using_file_combobox.itemText(i) for i in range(widget.using_file_combobox.count())]
            self.assertEqual(texts, ["2条波形", "4条波形"])
            self.assertEqual(widget.using_file_combobox.currentText(), "4条波形")
            self.assertEqual(widget.using_file_combobox.currentData(), "4条波形.json")

    def test_active_product_program_reports_partial_threshold_configuration(self):
        manager = types.SimpleNamespace(
            load_registry=lambda: {
                "active_file": "motor.json",
                "configs": [{"file": "motor.json", "name": "motor"}],
            },
            load_program=lambda _file_name: (
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
            validate_program=lambda _program, _file_name: {
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

    def test_waveform_workspace_reconfigure_does_not_redraw_unchanged_conditions(self):
        widget = _WaveformRefreshWidget()
        widget._direction_waveform_cache = {
            "01": ([0.0, 0.1], 1.0),
            "02": ([0.0, 0.2], 1.0),
            "03": ([0.0, 0.3], 1.0),
        }

        widget._configure_direction_waveform_workspace()

        self.assertEqual(widget.channel_workspace.set_condition_calls, [])
        self.assertEqual(widget.channel_workspace.direction_data, [])

        widget.plot_waveform_to_workspace([0.0, 0.4], 1.0, direction="01")

        self.assertEqual([entry[0] for entry in widget.channel_workspace.direction_data], ["01"])

    def test_active_product_condition_routes_waveform_to_matching_card(self):
        widget = _WaveformRefreshWidget()
        widget._active_product_condition_key = "02"

        active_direction = widget._resolve_active_recording_waveform_direction(fallback="")
        widget.plot_waveform_to_workspace([0.0, 0.7], 1.0, direction=active_direction)

        self.assertEqual(active_direction, "02")
        self.assertEqual([entry[0] for entry in widget.channel_workspace.direction_data], ["02"])
        self.assertIn("02", widget._direction_waveform_cache)
        self.assertNotIn("01", widget._direction_waveform_cache)

    def test_waveform_display_downsamples_peaks_without_changing_cached_audio(self):
        widget = _WaveformRefreshWidget()
        waveform = np.zeros(10_000, dtype=np.float32)
        waveform[1_234] = -9.0
        waveform[8_765] = 8.0

        widget.plot_waveform_to_workspace(waveform, 1_000.0, direction="01")

        cached_waveform, cached_sample_rate = widget._direction_waveform_cache["01"]
        _, display_x, display_y = widget.channel_workspace.direction_data[-1]
        self.assertEqual(cached_sample_rate, 1_000.0)
        self.assertTrue(np.array_equal(cached_waveform, waveform))
        self.assertEqual(len(display_x), len(display_y))
        self.assertLessEqual(len(display_y), widget._WAVEFORM_DISPLAY_MAX_POINTS)
        self.assertEqual(min(display_y), -9.0)
        self.assertEqual(max(display_y), 8.0)
        self.assertEqual(display_x[0], 0.0)
        self.assertAlmostEqual(display_x[-1], 9.999)

    def test_streaming_writer_receives_full_chunk_when_display_is_downsampled(self):
        class _Writer:
            def __init__(self):
                self.chunks = []

            def write_chunk(self, chunk):
                self.chunks.append(chunk)

        widget = _WaveformRefreshWidget()
        chunk = np.arange(12_000, dtype=np.float32).reshape(6_000, 2)
        widget.streaming_buffer_multi = []
        widget.streaming_wav_writer = _Writer()
        widget.data_struct = types.SimpleNamespace(sample_rate=48_000)
        widget._active_product_condition_key = "01"
        widget._streaming_first_chunk_logged = True
        widget.default_logger = logging.getLogger(__name__)

        widget.on_audio_chunk_received({"multi": chunk})

        cached_waveform, cached_sample_rate = widget._direction_waveform_cache["01"]
        _, _, display_y = widget.channel_workspace.direction_data[-1]
        self.assertEqual(cached_sample_rate, 48_000.0)
        self.assertEqual(cached_waveform.shape[0], chunk.shape[0])
        self.assertEqual(len(widget.streaming_wav_writer.chunks), 1)
        self.assertTrue(np.array_equal(widget.streaming_wav_writer.chunks[0], chunk))
        self.assertLessEqual(len(display_y), widget._WAVEFORM_DISPLAY_MAX_POINTS)

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

    def test_clear_plot_area_only_clears_active_condition_card(self):
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
        self.assertIsNone(widget._direction_waveform_cache["03"])
        self.assertIn("01", widget._condition_record_cache)
        self.assertIn("02", widget._condition_record_cache)
        self.assertNotIn("03", widget._condition_record_cache)
        self.assertEqual(widget.channel_workspace.cleared_directions, ["03"])
        self.assertEqual(widget.channel_workspace.clear_all_count, 0)

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
