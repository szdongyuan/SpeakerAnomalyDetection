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
from ui.sequence.station_plot_grid import StationPlotGrid
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
        self.recent_test_session_by_id = {}
        self._last_recent_session_mode = ""

    def _resolve_recent_session(self, session_id: str):
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
        self._last_recent_session_mode = ""


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
        self.assertFalse(widget.count_board.isVisible())
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

    def test_threshold_analyses_can_output_ok_ng(self):
        widget = _DummySequenceWidget()

        for analysis_type in ("LOUD", "FBA", "FFT"):
            widget.analysis_config = {
                "display_sequence": ["item"],
                "item": {"type": analysis_type, "limit_checked": True},
            }

            self.assertEqual(widget._can_output_ok_ng(), (True, ""))


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
                with patch.object(widget, "_resolve_workspace_screen_size", return_value=(1920, 1080)):
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
                with patch.object(widget, "_resolve_workspace_screen_size", return_value=(1920, 1080)):
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

    def test_mode_switch_remains_available_without_summary_widgets(self):
        board = SequenceCountBoard({})

        board.on_mark_btn_clicked()

        self.assertEqual(board.mode, "mark")
        self.assertFalse(hasattr(board, "stacked_widget"))

        board.on_test_btn_clicked()

        self.assertEqual(board.mode, "test")
        self.assertFalse(hasattr(board, "stacked_widget"))

    def test_count_board_rejects_test_mode_with_condition_threshold_prompt(self):
        reason = (
            "以下工况未启用阈值判定，无法进入测试模式：\n"
            "- 7000 rpm 未启用可自动输出 OK/NG 的规则阈值：queue_7000\n"
            "请启用所有工况的阈值，或使用标记模式。"
        )
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
            # Registry entries are visible only when their product files exist.
            for name in ("2条波形.json", "4条波形.json"):
                with open(os.path.join(folder, name), "w", encoding="utf-8") as stream:
                    stream.write("{}")
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
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        manager.program_dir = directory.name
        for name in ("a.json", "b.json"):
            with open(os.path.join(directory.name, name), "w", encoding="utf-8") as stream:
                stream.write("{}")
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


    def test_streaming_writer_receives_full_chunk_without_display(self):
        widget = SequenceWidgetStreamingOpsMixin()
        widget._recording_input_channels = (0, 1)
        widget.streaming_wav_writer = types.SimpleNamespace(chunks=[])
        widget.streaming_wav_writer.write_chunk = widget.streaming_wav_writer.chunks.append
        widget._streaming_first_chunk_logged = True
        chunk = np.zeros((6000, 2), dtype=np.float32)
        chunk[1234] = -9.0
        chunk[5678] = 8.0
        widget.on_audio_chunk_received({'multi': chunk})
        self.assertEqual(len(widget.streaming_wav_writer.chunks), 1)
        np.testing.assert_array_equal(widget.streaming_wav_writer.chunks[0], chunk)

    def test_streaming_writer_keeps_raw_chunks_before_startup_trim(self):
        widget = SequenceWidgetStreamingOpsMixin()
        widget._recording_input_channels = (0, 1)
        widget.streaming_wav_writer = types.SimpleNamespace(chunks=[])
        widget.streaming_wav_writer.write_chunk = widget.streaming_wav_writer.chunks.append
        widget._streaming_first_chunk_logged = True
        full_audio = np.arange(280, dtype=np.float32).reshape(140, 2)
        widget.on_audio_chunk_received({'multi': full_audio[:60]})
        widget.on_audio_chunk_received({'multi': full_audio[60:]})
        self.assertEqual(len(widget.streaming_wav_writer.chunks), 2)
        np.testing.assert_array_equal(widget.streaming_wav_writer.chunks[0], full_audio[:60])
        np.testing.assert_array_equal(widget.streaming_wav_writer.chunks[1], full_audio[60:])

    def test_invalid_serial_recording_uses_whole_round_abort(self):
        calls = []

        class _InvalidRecordingHost:
            _serial_product_condition_executing = True

            def _recover_current_product_recording(self, reason):
                # This scenario exercises the serial whole-round fallback.
                return False

            def _on_serial_product_runtime_error(self, reason):
                calls.append(reason)
                return True

        SequenceWidgetStreamingOpsMixin._handle_invalid_recording(
            _InvalidRecordingHost(),
            "empty audio",
        )

        self.assertEqual(len(calls), 1)
        self.assertIn("empty audio", calls[0])


    def test_waveform_mark_does_not_update_left_condition_judgement(self):
        widget = _DummySequenceWidget()
        widget.count_board.mode = "mark"
        widget.left_panel = _SpyLeftPanel()
        widget._condition_record_cache = {}
        widget.recorded_path = None
        widget.recorded_signal_info = {}

        with patch("ui.sequence.sequence_widget_streaming_ops.QMessageBox.warning") as warning:
            widget.on_waveform_condition_mark_clicked("01", "OK")

        warning.assert_called_once()
        self.assertEqual(widget.left_panel.results, [])

    def test_waveform_mark_updates_bound_recent_session_result(self):
        widget = _DummySequenceWidget()
        widget.count_board.mode = "mark"

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

    def test_waveform_mark_with_missing_session_id_updates_matching_condition(self):
        widget = _DummySequenceWidget()
        widget.count_board.mode = "mark"
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


    def test_waveform_mark_falls_back_to_recent_history_when_cache_missing(self):
        widget = _DummySequenceWidget()
        widget.count_board.mode = "mark"
        widget._condition_record_cache = {}
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


if __name__ == "__main__":
    unittest.main()
