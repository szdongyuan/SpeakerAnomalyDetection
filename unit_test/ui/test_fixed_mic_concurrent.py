import os
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest import mock

import numpy as np
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem

from base.fixed_mic_capture import FixedMicCaptureController
from consts import error_code
from ui.sequence.fixed_mic import FixedMicSessionTablePanel
from ui.sequence.sequencement_count_board import SequenceCountBoard
from ui.sequence.sequence_widget import SequenceWindow


def build_chunk(start_value, frames, channels):
    chunk = np.arange(start_value, start_value + frames * channels, dtype=np.float32)
    return chunk.reshape(frames, channels)


def create_test_controller(max_sessions=4, buffer_duration=10.0):
    controller = FixedMicCaptureController(
        {
            "sample_rate": 10,
            "channels": 2,
            "window_duration": 3.0,
            "buffer_duration": buffer_duration,
            "max_sessions": max_sessions,
            "trigger_mode": "manual_click",
        }
    )
    controller.is_running = True
    return controller


class TestFixedMicConcurrent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication([])

    def test_fixed_mic_controller_supports_overlapping_sessions_and_independent_clips(self):
        controller = create_test_controller(max_sessions=4, buffer_duration=10.0)

        code_1, _, session_1 = controller.create_manual_session("car_a")
        self.assertEqual(code_1, error_code.OK)

        controller._audio_callback(build_chunk(0, 10, 2), 10, None, None)
        code_2, _, session_2 = controller.create_manual_session("car_b")
        self.assertEqual(code_2, error_code.OK)

        controller._audio_callback(build_chunk(20, 10, 2), 10, None, None)
        controller._audio_callback(build_chunk(40, 10, 2), 10, None, None)
        controller._audio_callback(build_chunk(60, 10, 2), 10, None, None)

        processed_chunks, completed_sessions = controller.process_audio_queue()

        self.assertEqual(processed_chunks, 4)
        self.assertEqual(
            [session.session_id for session in completed_sessions],
            [session_1.session_id, session_2.session_id],
        )
        self.assertEqual(session_1.audio_clip.shape, (30, 2))
        self.assertEqual(session_2.audio_clip.shape, (30, 2))
        self.assertTrue(np.array_equal(session_1.audio_clip, build_chunk(0, 30, 2)))
        self.assertTrue(np.array_equal(session_2.audio_clip, build_chunk(20, 30, 2)))

    def test_fixed_mic_controller_rejects_sessions_after_reaching_limit(self):
        controller = create_test_controller(max_sessions=2, buffer_duration=10.0)

        code_1, _, _ = controller.create_manual_session("car_a")
        code_2, _, _ = controller.create_manual_session("car_b")
        code_3, msg_3, session_3 = controller.create_manual_session("car_c")

        self.assertEqual(code_1, error_code.OK)
        self.assertEqual(code_2, error_code.OK)
        self.assertEqual(code_3, error_code.INVALID_ADD)
        self.assertIn("Maximum active fixed mic sessions reached", msg_3)
        self.assertIsNone(session_3)

    def test_fixed_mic_controller_cancels_session_when_audio_window_is_evicted(self):
        controller = create_test_controller(max_sessions=1, buffer_duration=2.0)

        code, _, session = controller.create_manual_session("car_a")
        self.assertEqual(code, error_code.OK)

        controller._audio_callback(build_chunk(0, 20, 2), 20, None, None)
        controller._audio_callback(build_chunk(40, 20, 2), 20, None, None)
        _, completed_sessions = controller.process_audio_queue()

        self.assertEqual(len(completed_sessions), 1)
        self.assertIs(completed_sessions[0], session)
        self.assertEqual(session.status, "cancelled")
        self.assertIsNone(session.audio_clip)
        self.assertEqual(
            session.metadata["cancel_reason"],
            "audio clip window is no longer available in ring buffer",
        )

    def test_fixed_mic_analysis_queue_runs_sessions_serially(self):
        handled_sessions = []

        fake_window = SimpleNamespace(
            fixed_mic_analysis_busy=False,
            fixed_mic_analysis_queue=["session_a", "session_b"],
            _update_fixed_mic_session_status=mock.Mock(),
        )

        def finalize(session):
            handled_sessions.append(session)

        fake_window._finalize_and_run_fixed_mic_session = finalize
        fake_window._process_next_fixed_mic_analysis = lambda: SequenceWindow._process_next_fixed_mic_analysis(
            fake_window
        )

        with mock.patch(
            "ui.sequence.sequence_widget.QTimer.singleShot",
            side_effect=lambda _delay, callback: callback(),
        ):
            SequenceWindow._process_next_fixed_mic_analysis(fake_window)

        self.assertEqual(handled_sessions, ["session_a", "session_b"])
        self.assertFalse(fake_window.fixed_mic_analysis_busy)
        self.assertEqual(fake_window.fixed_mic_analysis_queue, [])

    def test_fixed_mic_mode_switch_supports_mark_mode_and_restores_legacy_mode(self):
        with mock.patch.object(SequenceCountBoard, "set_test_text", new=lambda self: None), mock.patch.object(
            SequenceCountBoard, "set_mark_text", new=lambda self: None
        ):
            count_board = SequenceCountBoard(
                {
                    "default_ai": None,
                }
            )

        fake_window = SimpleNamespace(
            count_board=count_board,
            fixed_mic_pending_review_sessions=[],
            fixed_mic_current_review_session=None,
            is_fixed_mic_mode=lambda: True,
            _update_fixed_mic_toolbar_state=lambda: None,
            _activate_next_fixed_mic_review_session=mock.Mock(),
            _refresh_fixed_mic_review_session_display=mock.Mock(),
            _set_fixed_mic_current_review_session=mock.Mock(),
        )

        SequenceWindow._sync_fixed_mic_mode_state(fake_window, desired_mode="mark")

        self.assertEqual(count_board.mode, "mark")
        self.assertTrue(count_board.review_session_widget.isHidden())
        self.assertFalse(count_board.mark_btn.isEnabled())

        SequenceWindow._sync_fixed_mic_mode_state(fake_window, desired_mode="test")

        self.assertEqual(count_board.mode, "test")
        self.assertTrue(count_board.mark_btn.isEnabled())
        self.assertTrue(count_board.review_session_widget.isHidden())

        fake_window.is_fixed_mic_mode = lambda: False
        SequenceWindow._sync_fixed_mic_mode_state(fake_window)

        self.assertTrue(count_board.mark_btn.isEnabled())

    def test_clicked_ok_or_ng_uses_fixed_mic_mark_handler(self):
        fake_window = SimpleNamespace(
            is_fixed_mic_mode=lambda: True,
            count_board=SimpleNamespace(mode="mark"),
            _handle_fixed_mic_mark_result=mock.Mock(),
        )

        SequenceWindow.clicked_ok_or_ng(fake_window)

        fake_window._handle_fixed_mic_mark_result.assert_called_once()

    def test_enqueue_fixed_mic_review_session_keeps_session_pending(self):
        fake_session = SimpleNamespace(session_id="fixed_mic_session_001")
        fake_window = SimpleNamespace(
            fixed_mic_pending_review_sessions=[],
            fixed_mic_current_review_session=None,
            default_logger=SimpleNamespace(info=lambda *args, **kwargs: None),
            _refresh_fixed_mic_review_session_display=mock.Mock(),
            _activate_next_fixed_mic_review_session=mock.Mock(),
        )

        SequenceWindow._enqueue_fixed_mic_review_session(fake_window, fake_session)

        self.assertEqual(fake_window.fixed_mic_pending_review_sessions, [fake_session])
        fake_window._refresh_fixed_mic_review_session_display.assert_called_once()
        fake_window._activate_next_fixed_mic_review_session.assert_not_called()

    def test_activate_next_fixed_mic_review_session_keeps_all_pending_without_selection(self):
        fake_session = SimpleNamespace(
            session_id="fixed_mic_session_001",
            trigger_time=datetime(2026, 3, 19, 10, 55, 58),
        )
        fake_window = SimpleNamespace(
            fixed_mic_pending_review_sessions=[fake_session],
            fixed_mic_current_review_session=None,
            analysis_config={"auto_analysis": True},
            count_board=SimpleNamespace(set_review_session_text=mock.Mock()),
            _update_fixed_mic_session_status=mock.Mock(),
            _load_fixed_mic_review_session_context=mock.Mock(),
            _close_analysis_windows=mock.Mock(),
            _run_fixed_mic_session_analysis=mock.Mock(),
            _emit_display_update=mock.Mock(),
            _refresh_fixed_mic_review_session_display=mock.Mock(),
            _select_fixed_mic_session_row=mock.Mock(),
            _get_selected_fixed_mic_session=lambda: None,
        )
        fake_window._is_fixed_mic_session_pending = lambda session: SequenceWindow._is_fixed_mic_session_pending(
            fake_window, session
        )
        fake_window._set_fixed_mic_current_review_session = lambda session, sync_selection=False: SequenceWindow._set_fixed_mic_current_review_session(
            fake_window, session, sync_selection
        )

        SequenceWindow._activate_next_fixed_mic_review_session(fake_window)

        self.assertIsNone(fake_window.fixed_mic_current_review_session)
        fake_window._update_fixed_mic_session_status.assert_not_called()
        fake_window._select_fixed_mic_session_row.assert_not_called()
        fake_window._load_fixed_mic_review_session_context.assert_not_called()
        fake_window._close_analysis_windows.assert_not_called()
        fake_window._run_fixed_mic_session_analysis.assert_not_called()
        fake_window._emit_display_update.assert_not_called()
        fake_window._refresh_fixed_mic_review_session_display.assert_called_once()

    def test_activate_next_fixed_mic_review_session_prefers_selected_pending_session(self):
        session_a = SimpleNamespace(session_id="fixed_mic_session_001", trigger_time=datetime(2026, 3, 19, 10, 55, 58), metadata={})
        session_b = SimpleNamespace(session_id="fixed_mic_session_002", trigger_time=datetime(2026, 3, 19, 10, 56, 1), metadata={})
        fake_window = SimpleNamespace(
            fixed_mic_pending_review_sessions=[session_a, session_b],
            fixed_mic_current_review_session=None,
            count_board=SimpleNamespace(set_review_session_text=mock.Mock()),
            _update_fixed_mic_session_status=mock.Mock(),
            _refresh_fixed_mic_review_session_display=mock.Mock(),
            _select_fixed_mic_session_row=mock.Mock(),
            _get_selected_fixed_mic_session=lambda: session_b,
        )
        fake_window._is_fixed_mic_session_pending = lambda session: SequenceWindow._is_fixed_mic_session_pending(
            fake_window, session
        )
        fake_window._set_fixed_mic_current_review_session = lambda session, sync_selection=False: SequenceWindow._set_fixed_mic_current_review_session(
            fake_window, session, sync_selection
        )

        SequenceWindow._activate_next_fixed_mic_review_session(fake_window)

        self.assertIs(fake_window.fixed_mic_current_review_session, session_b)
        fake_window._update_fixed_mic_session_status.assert_called_once_with(session_b, "审核中")
        fake_window._select_fixed_mic_session_row.assert_not_called()

    def test_handle_fixed_mic_mark_result_does_not_auto_activate_next_session(self):
        ok_button = object()
        session_a = SimpleNamespace(session_id="fixed_mic_session_001", metadata={}, analysis_result=None)
        session_b = SimpleNamespace(session_id="fixed_mic_session_002", metadata={}, analysis_result=None)
        fake_window = SimpleNamespace(
            fixed_mic_current_review_session=session_a,
            fixed_mic_pending_review_sessions=[session_a, session_b],
            count_board=SimpleNamespace(
                ok_btn=ok_button,
                ng_btn=object(),
                set_mark_result_file=mock.Mock(),
                set_mark_text=mock.Mock(),
            ),
            recorded_signal_info={"file_path": "test.wav"},
            sender=lambda: ok_button,
            _load_fixed_mic_review_session_context=mock.Mock(),
            update_recorded_signal_info_to_db=mock.Mock(),
            _update_fixed_mic_session_status=mock.Mock(),
            _update_fixed_mic_session_result=mock.Mock(),
            _emit_display_update=mock.Mock(),
            _refresh_fixed_mic_review_session_display=mock.Mock(),
            default_logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        )

        SequenceWindow._handle_fixed_mic_mark_result(fake_window)

        self.assertEqual(fake_window.fixed_mic_pending_review_sessions, [session_b])
        self.assertEqual(fake_window.recorded_signal_info["labels"], "OK")
        fake_window._update_fixed_mic_session_status.assert_called_once_with(session_a, "已标记")
        fake_window._update_fixed_mic_session_result.assert_called_once_with(session_a, "OK")
        self.assertIsNone(fake_window.fixed_mic_current_review_session)
        fake_window._refresh_fixed_mic_review_session_display.assert_called_once()

    def test_handle_fixed_mic_mark_result_removes_pending_by_session_id(self):
        ok_button = object()
        selected_session = SimpleNamespace(session_id="fixed_mic_session_001", metadata={}, analysis_result=None)
        duplicated_pending_session = SimpleNamespace(session_id="fixed_mic_session_001", metadata={}, analysis_result=None)
        remaining_session = SimpleNamespace(session_id="fixed_mic_session_002", metadata={}, analysis_result=None)
        fake_window = SimpleNamespace(
            fixed_mic_current_review_session=selected_session,
            fixed_mic_pending_review_sessions=[duplicated_pending_session, remaining_session],
            count_board=SimpleNamespace(
                ok_btn=ok_button,
                ng_btn=object(),
                set_mark_result_file=mock.Mock(),
                set_mark_text=mock.Mock(),
            ),
            recorded_signal_info={"file_path": "test.wav"},
            sender=lambda: ok_button,
            _load_fixed_mic_review_session_context=mock.Mock(),
            update_recorded_signal_info_to_db=mock.Mock(),
            _update_fixed_mic_session_status=mock.Mock(),
            _update_fixed_mic_session_result=mock.Mock(),
            _emit_display_update=mock.Mock(),
            _refresh_fixed_mic_review_session_display=mock.Mock(),
            default_logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        )

        SequenceWindow._handle_fixed_mic_mark_result(fake_window)

        self.assertEqual(fake_window.fixed_mic_pending_review_sessions, [remaining_session])
        fake_window._update_fixed_mic_session_status.assert_called_once_with(selected_session, "已标记")

    def test_selection_changed_marks_selected_pending_session_as_reviewing(self):
        session_a = SimpleNamespace(session_id="fixed_mic_session_001", metadata={})
        session_b = SimpleNamespace(session_id="fixed_mic_session_002", metadata={})
        item_a = QTableWidgetItem("001")
        item_a.setData(Qt.UserRole, "fixed_mic_session_001")
        item_b = QTableWidgetItem("002")
        item_b.setData(Qt.UserRole, "fixed_mic_session_002")
        session_table = QTableWidget(2, 1)
        session_table.setItem(0, 0, item_a)
        session_table.setItem(1, 0, item_b)
        session_table.setCurrentCell(1, 0)

        fake_window = SimpleNamespace(
            is_fixed_mic_mode=lambda: True,
            count_board=SimpleNamespace(mode="mark", set_review_session_text=mock.Mock()),
            fixed_mic_session_table=session_table,
            fixed_mic_session_panel=SimpleNamespace(
                get_session=lambda session_id: {
                    "fixed_mic_session_001": session_a,
                    "fixed_mic_session_002": session_b,
                }.get(session_id)
            ),
            fixed_mic_pending_review_sessions=[session_a, session_b],
            fixed_mic_current_review_session=session_a,
            _update_fixed_mic_session_status=mock.Mock(),
            _refresh_fixed_mic_review_session_display=mock.Mock(),
            _select_fixed_mic_session_row=mock.Mock(),
        )
        fake_window._is_fixed_mic_session_pending = lambda session: SequenceWindow._is_fixed_mic_session_pending(
            fake_window, session
        )
        fake_window._set_fixed_mic_current_review_session = lambda session, sync_selection=False: SequenceWindow._set_fixed_mic_current_review_session(
            fake_window, session, sync_selection
        )

        SequenceWindow._on_fixed_mic_session_selection_changed(fake_window)

        self.assertIs(fake_window.fixed_mic_current_review_session, session_b)
        fake_window._update_fixed_mic_session_status.assert_any_call(session_a, "待审核")
        fake_window._update_fixed_mic_session_status.assert_any_call(session_b, "审核中")
        fake_window._select_fixed_mic_session_row.assert_not_called()

    def test_sync_mark_mode_does_not_auto_activate_pending_session_without_selection(self):
        fake_window = SimpleNamespace(
            count_board=SimpleNamespace(
                mode="mark",
                mark_btn=SimpleNamespace(setEnabled=mock.Mock()),
                test_btn=SimpleNamespace(setEnabled=mock.Mock()),
                set_review_session_visible=mock.Mock(),
                set_mark_mode_enabled=mock.Mock(),
            ),
            fixed_mic_pending_review_sessions=[SimpleNamespace(session_id="fixed_mic_session_001")],
            fixed_mic_current_review_session=SimpleNamespace(session_id="fixed_mic_session_000"),
            is_fixed_mic_mode=lambda: True,
            _update_fixed_mic_toolbar_state=mock.Mock(),
            _set_fixed_mic_current_review_session=mock.Mock(),
            _get_selected_fixed_mic_session=lambda: None,
            _is_fixed_mic_session_pending=lambda session: False,
        )

        SequenceWindow._sync_fixed_mic_mode_state(fake_window)

        fake_window._set_fixed_mic_current_review_session.assert_called_once_with(None)

    def test_load_fixed_mic_review_session_context_can_skip_plot(self):
        fake_session = SimpleNamespace(
            metadata={"recorded_path": "test.wav", "recorded_signal_info": {"file_path": "test.wav"}, "sample_rate": 10},
            audio_clip=np.array([[0.1], [0.2]], dtype=np.float32),
        )
        fake_window = SimpleNamespace(
            recorded_path=None,
            recorded_signal_info={},
            data_struct=SimpleNamespace(store_wave_data=None, sample_rate=1, update_channel_count=mock.Mock()),
            line_graph=mock.Mock(),
            plot_line_graph=mock.Mock(),
        )

        SequenceWindow._load_fixed_mic_review_session_context(fake_window, fake_session, update_plot=False)

        self.assertEqual(fake_window.recorded_path, "test.wav")
        self.assertEqual(fake_window.recorded_signal_info, {"file_path": "test.wav"})
        fake_window.data_struct.update_channel_count.assert_called_once()
        fake_window.plot_line_graph.assert_not_called()

    def test_handle_fixed_mic_manual_trigger_starts_capture_and_creates_session(self):
        class FakeTimer(object):
            def __init__(self):
                self.started_with = None

            def start(self, interval):
                self.started_with = interval

        class FakeButton(object):
            def __init__(self):
                self.disabled_value = None
                self.enabled_value = None

            def setDisabled(self, value):
                self.disabled_value = value

            def setEnabled(self, value):
                self.enabled_value = value

        class FakeLineGraph(object):
            def __init__(self):
                self.cleared = False

            def clear(self):
                self.cleared = True

        class FakeLineEdit(object):
            def text(self):
                return "BARCODE_001"

        fake_session = SimpleNamespace(session_id="fixed_mic_session_001", trigger_sample_index=0)
        fake_controller = SimpleNamespace(
            is_running=True,
            get_active_session_count=mock.Mock(return_value=0),
            buffer_duration=15.0,
            window_duration=3.0,
            start_capture=mock.Mock(return_value=(error_code.OK, "started")),
            create_manual_session=mock.Mock(return_value=(error_code.OK, "created", fake_session)),
            ring_buffer=SimpleNamespace(get_total_samples_written=mock.Mock(return_value=0)),
            sample_rate=10,
        )
        fake_timer = FakeTimer()
        fake_replayer_btn = FakeButton()
        fake_data_btn = FakeButton()
        fake_line_graph = FakeLineGraph()
        fake_window = SimpleNamespace(
            checked_work_status_message=lambda: False,
            clicked_player_flag=True,
            fixed_mic_controller=None,
            sequence_config=[{"seq1": {"acq": {"detail": {"sample_rate": 10, "channels": 2, "window_duration": 3.0}}}}],
            mic="fake_mic",
            fixed_mic_poll_timer=fake_timer,
            line_graph=fake_line_graph,
            fixed_mic_plot_item="old_plot",
            lineedit_s_or_n=FakeLineEdit(),
            replayer_btn=fake_replayer_btn,
            data_btn=fake_data_btn,
            default_logger=SimpleNamespace(info=lambda *args, **kwargs: None),
            player_status_flag=False,
            _update_fixed_mic_toolbar_state=mock.Mock(),
            _configure_fixed_mic_live_plot_view=mock.Mock(),
            _reset_fixed_mic_session_views=mock.Mock(),
            _register_fixed_mic_session=mock.Mock(),
        )

        with mock.patch("ui.sequence.sequence_widget.FixedMicCaptureController", return_value=fake_controller):
            SequenceWindow._handle_fixed_mic_manual_trigger(fake_window)

        self.assertIs(fake_window.fixed_mic_controller, fake_controller)
        self.assertEqual(fake_timer.started_with, 50)
        self.assertTrue(fake_line_graph.cleared)
        self.assertIsNone(fake_window.fixed_mic_plot_item)
        fake_controller.start_capture.assert_called_once()
        fake_controller.create_manual_session.assert_called_once_with("BARCODE_001")
        self.assertTrue(fake_window.player_status_flag)
        fake_window._update_fixed_mic_toolbar_state.assert_called_once()
        fake_window._configure_fixed_mic_live_plot_view.assert_called_once()
        fake_window._reset_fixed_mic_session_views.assert_called_once()
        fake_window._register_fixed_mic_session.assert_called_once_with(fake_session, status_text="采集中")
        self.assertFalse(fake_data_btn.enabled_value)

    def test_poll_fixed_mic_runtime_enqueues_frozen_session_and_skips_cancelled(self):
        class FakePlotItem(object):
            def __init__(self):
                self.data = None

            def setData(self, x_axis, y_axis):
                self.data = (x_axis, y_axis)

        class FakeLineGraph(object):
            def __init__(self):
                self.plot_calls = []
                self.plot_item = FakePlotItem()
                self.x_ranges = []
                self.y_ranges = []

            def plot(self, x_axis, y_axis, pen=None, **kwargs):
                self.plot_calls.append((x_axis, y_axis, pen, kwargs))
                return self.plot_item

            def setXRange(self, left, right, padding=0.0):
                self.x_ranges.append((left, right, padding))

            def setYRange(self, bottom, top, padding=0.0):
                self.y_ranges.append((bottom, top, padding))

        frozen_session = SimpleNamespace(
            status="frozen",
            session_id="fixed_mic_session_001",
            metadata={"audio_clip_shape": (30, 2)},
        )
        cancelled_session = SimpleNamespace(
            status="cancelled",
            session_id="fixed_mic_session_002",
            metadata={"cancel_reason": "buffer overflow"},
        )
        fake_controller = SimpleNamespace(
            sample_rate=10,
            window_duration=3.0,
            process_audio_queue=mock.Mock(return_value=(2, [frozen_session, cancelled_session])),
            consume_recent_plot_chunks=mock.Mock(return_value=[np.array([[0.0], [1.0], [0.5]], dtype=np.float32)]),
            ring_buffer=SimpleNamespace(get_total_samples_written=mock.Mock(return_value=3)),
        )
        fake_window = SimpleNamespace(
            fixed_mic_controller=fake_controller,
            line_graph=FakeLineGraph(),
            fixed_mic_plot_item=None,
            fixed_mic_analysis_queue=[],
            default_logger=SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
            _process_next_fixed_mic_analysis=mock.Mock(),
            _update_fixed_mic_session_status=mock.Mock(),
            build_live_plot_data=SequenceWindow.build_live_plot_data,
            fixed_mic_plot_interval_sec=0.0,
            fixed_mic_plot_window_sec=15.0,
            fixed_mic_last_plot_update_ts=0.0,
            fixed_mic_live_y_limit=0.01,
            fixed_mic_stream_buffer=[],
            _update_fixed_mic_live_plot_range=None,
            _append_fixed_mic_stream_chunks=None,
            _get_fixed_mic_stream_plot_audio=None,
        )
        fake_window._update_fixed_mic_live_plot_range = lambda plot_audio: SequenceWindow._update_fixed_mic_live_plot_range(
            fake_window, plot_audio
        )
        fake_window._append_fixed_mic_stream_chunks = lambda chunks, sample_rate: SequenceWindow._append_fixed_mic_stream_chunks(
            fake_window, chunks, sample_rate
        )
        fake_window._get_fixed_mic_stream_plot_audio = lambda: SequenceWindow._get_fixed_mic_stream_plot_audio(
            fake_window
        )

        SequenceWindow._poll_fixed_mic_runtime(fake_window)

        self.assertEqual(fake_window.fixed_mic_analysis_queue, [frozen_session])
        self.assertIsNotNone(fake_window.fixed_mic_plot_item)
        self.assertEqual(len(fake_window.line_graph.plot_calls), 1)
        fake_controller.consume_recent_plot_chunks.assert_called_once()
        fake_window._update_fixed_mic_session_status.assert_any_call(frozen_session, "分析中")
        fake_window._update_fixed_mic_session_status.assert_any_call(cancelled_session, "切片失败")
        fake_window._process_next_fixed_mic_analysis.assert_called_once()

    def test_poll_fixed_mic_runtime_keeps_live_plot_when_no_active_session(self):
        fake_controller = SimpleNamespace(
            sample_rate=10,
            window_duration=3.0,
            process_audio_queue=mock.Mock(return_value=(0, [])),
            consume_recent_plot_chunks=mock.Mock(return_value=[]),
        )
        class FakePlotItem(object):
            def __init__(self):
                self.data = None

            def setData(self, x_axis, y_axis):
                self.data = (x_axis, y_axis)

        class FakeLineGraph(object):
            def __init__(self):
                self.plot_calls = []
                self.plot_item = FakePlotItem()
                self.x_ranges = []
                self.y_ranges = []

            def plot(self, x_axis, y_axis, pen=None, **kwargs):
                self.plot_calls.append((x_axis, y_axis, pen, kwargs))
                return self.plot_item

            def setXRange(self, left, right, padding=0.0):
                self.x_ranges.append((left, right, padding))

            def setYRange(self, bottom, top, padding=0.0):
                self.y_ranges.append((bottom, top, padding))

        existing_plot_item = FakePlotItem()
        fake_window = SimpleNamespace(
            fixed_mic_controller=fake_controller,
            line_graph=FakeLineGraph(),
            fixed_mic_plot_item=existing_plot_item,
            fixed_mic_analysis_queue=[],
            default_logger=SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
            _process_next_fixed_mic_analysis=mock.Mock(),
            _update_fixed_mic_session_status=mock.Mock(),
            build_live_plot_data=SequenceWindow.build_live_plot_data,
            fixed_mic_plot_interval_sec=0.0,
            fixed_mic_plot_window_sec=15.0,
            fixed_mic_last_plot_update_ts=0.0,
            fixed_mic_live_y_limit=0.01,
            fixed_mic_stream_buffer=[np.array([0.0, 1.0, 0.5], dtype=np.float32)],
            _update_fixed_mic_live_plot_range=None,
            _append_fixed_mic_stream_chunks=None,
            _get_fixed_mic_stream_plot_audio=None,
        )
        fake_window._update_fixed_mic_live_plot_range = lambda plot_audio: SequenceWindow._update_fixed_mic_live_plot_range(
            fake_window, plot_audio
        )
        fake_window._append_fixed_mic_stream_chunks = lambda chunks, sample_rate: SequenceWindow._append_fixed_mic_stream_chunks(
            fake_window, chunks, sample_rate
        )
        fake_window._get_fixed_mic_stream_plot_audio = lambda: SequenceWindow._get_fixed_mic_stream_plot_audio(
            fake_window
        )

        SequenceWindow._poll_fixed_mic_runtime(fake_window)

        fake_controller.consume_recent_plot_chunks.assert_called_once()
        self.assertIs(fake_window.fixed_mic_plot_item, existing_plot_item)
        fake_window._update_fixed_mic_session_status.assert_not_called()
        fake_window._process_next_fixed_mic_analysis.assert_called_once()

    def test_append_fixed_mic_stream_chunks_trims_to_plot_window(self):
        fake_window = SimpleNamespace(
            fixed_mic_plot_window_sec=1.0,
            fixed_mic_stream_buffer=[],
        )

        SequenceWindow._append_fixed_mic_stream_chunks(
            fake_window,
            [
                np.arange(0, 8, dtype=np.float32),
                np.arange(8, 16, dtype=np.float32),
            ],
            sample_rate=10,
        )

        combined_audio = SequenceWindow._get_fixed_mic_stream_plot_audio(fake_window)
        self.assertEqual(len(combined_audio), 10)
        self.assertTrue(np.array_equal(combined_audio, np.arange(6, 16, dtype=np.float32)))

    def test_register_fixed_mic_session_populates_result_column_and_view_button(self):
        fake_session = SimpleNamespace(
            session_id="fixed_mic_session_001",
            trigger_time=datetime(2026, 3, 19, 10, 55, 58),
            vehicle_barcode="BARCODE_001",
            window_duration=3.0,
            metadata={},
        )
        panel = FixedMicSessionTablePanel(on_view_session=lambda _session_id: None)
        fake_window = SimpleNamespace(
            fixed_mic_session_panel=panel,
        )

        SequenceWindow._register_fixed_mic_session(fake_window, fake_session, status_text="采集中")

        self.assertEqual(panel.session_table.rowCount(), 1)
        self.assertEqual(panel.session_table.item(0, 2).text(), "10:56:01")
        self.assertEqual(panel.session_table.item(0, 3).text(), "3.00s")
        self.assertEqual(panel.session_table.item(0, 5).text(), "采集中")
        self.assertEqual(panel.session_table.item(0, 6).text(), "not_labeled")
        self.assertIsNotNone(panel.session_table.cellWidget(0, 7))

    def test_update_fixed_mic_session_result_updates_result_column(self):
        fake_session = SimpleNamespace(
            session_id="fixed_mic_session_001",
            trigger_time=datetime(2026, 3, 19, 10, 55, 58),
            vehicle_barcode="BARCODE_001",
            window_duration=3.0,
            metadata={},
        )
        panel = FixedMicSessionTablePanel(on_view_session=lambda _session_id: None)
        fake_window = SimpleNamespace(
            fixed_mic_session_panel=panel,
        )

        SequenceWindow._register_fixed_mic_session(fake_window, fake_session, status_text="采集中")
        SequenceWindow._update_fixed_mic_session_result(fake_window, fake_session, "OK")

        self.assertEqual(panel.session_table.item(0, 5).text(), "采集中")
        self.assertEqual(panel.session_table.item(0, 6).text(), "OK")

    def test_show_fixed_mic_session_result_by_id_opens_analysis_windows(self):
        fake_session = SimpleNamespace(
            session_id="fixed_mic_session_001",
            audio_clip=np.array([[0.1], [0.2]], dtype=np.float32),
            metadata={},
        )
        fake_window = SimpleNamespace(
            fixed_mic_session_panel=SimpleNamespace(get_session=mock.Mock(return_value=fake_session)),
            _select_fixed_mic_session_row=mock.Mock(),
            _show_fixed_mic_session_analysis_windows=mock.Mock(),
        )

        SequenceWindow._show_fixed_mic_session_result_by_id(fake_window, "fixed_mic_session_001")

        fake_window._select_fixed_mic_session_row.assert_called_once_with("fixed_mic_session_001")
        fake_window._show_fixed_mic_session_analysis_windows.assert_called_once_with(fake_session)

    def test_update_fixed_mic_live_plot_range_locks_x_axis_and_smooths_y_axis(self):
        fake_line_graph = SimpleNamespace(
            setXRange=mock.Mock(),
            setYRange=mock.Mock(),
        )
        fake_window = SimpleNamespace(
            line_graph=fake_line_graph,
            fixed_mic_plot_window_sec=3.0,
            fixed_mic_live_y_limit=0.01,
        )

        SequenceWindow._update_fixed_mic_live_plot_range(fake_window, np.array([0.0, 0.02, -0.03], dtype=np.float32))

        fake_line_graph.setXRange.assert_called_once_with(0.0, 3.0, padding=0.0)
        fake_line_graph.setYRange.assert_called_once()
        self.assertGreater(fake_window.fixed_mic_live_y_limit, 0.01)

    def test_on_clicked_replayer_btn_stops_fixed_mic_runtime(self):
        stop_mock = mock.Mock()
        fake_window = SimpleNamespace(
            is_fixed_mic_mode=lambda: True,
            fixed_mic_controller=SimpleNamespace(is_running=True),
            default_logger=SimpleNamespace(info=lambda *args, **kwargs: None),
            _stop_fixed_mic_runtime=stop_mock,
        )

        SequenceWindow.on_clicked_replayer_btn(fake_window)

        stop_mock.assert_called_once()

    def test_finalize_fixed_mic_session_does_not_replace_live_waveform(self):
        fake_session = SimpleNamespace(
            session_id="fixed_mic_session_001",
            audio_clip=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            metadata={"sample_rate": 10},
        )
        fake_finalize_pipeline = SimpleNamespace(
            save_fixed_mic_session=mock.Mock(
                return_value=(
                    error_code.OK,
                    "saved",
                    "test.wav",
                    {"file_path": "test.wav", "labels": "not_labeled"},
                )
            )
        )
        fake_lineedit_count = SimpleNamespace(setText=mock.Mock(), text=lambda: "1")
        fake_window = SimpleNamespace(
            current_recorded_count=0,
            lineedit_count=fake_lineedit_count,
            lineedit_type=SimpleNamespace(text=lambda: "MODEL_001"),
            lineedit_s_or_n=SimpleNamespace(text=lambda: "BARCODE_001"),
            barcode_scanner_box=SimpleNamespace(isChecked=lambda: False),
            fixed_mic_finalize_pipeline=fake_finalize_pipeline,
            recorded_path=None,
            recorded_signal_info={},
            data_struct=SimpleNamespace(
                store_wave_data=None,
                sample_rate=1,
                update_channel_count=mock.Mock(),
            ),
            count_board=SimpleNamespace(mode="test"),
            fixed_mic_plot_item="live_plot",
            line_graph=SimpleNamespace(clear=mock.Mock()),
            analysis_config={"auto_analysis": False},
            default_logger=SimpleNamespace(info=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None),
            _close_analysis_windows=mock.Mock(),
            _run_fixed_mic_session_analysis=mock.Mock(),
            _emit_display_update=mock.Mock(),
            _update_fixed_mic_session_status=mock.Mock(),
            _update_fixed_mic_session_result=mock.Mock(),
            _select_fixed_mic_session_row=mock.Mock(),
            plot_line_graph=mock.Mock(),
        )

        with mock.patch("ui.sequence.sequence_widget.save_recorded_data_to_json"):
            SequenceWindow._finalize_and_run_fixed_mic_session(fake_window, fake_session)

        self.assertEqual(fake_window.fixed_mic_plot_item, "live_plot")
        fake_window.plot_line_graph.assert_not_called()
        fake_window._update_fixed_mic_session_status.assert_called_once_with(fake_session, "已保存")
        fake_window._update_fixed_mic_session_result.assert_called_once_with(fake_session, "not_labeled")
        fake_window._select_fixed_mic_session_row.assert_called_once_with("fixed_mic_session_001")
        fake_window._emit_display_update.assert_called_once()

    def test_finalize_fixed_mic_session_in_mark_mode_enqueues_review_session(self):
        fake_session = SimpleNamespace(
            session_id="fixed_mic_session_001",
            audio_clip=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            metadata={"sample_rate": 10},
        )
        fake_finalize_pipeline = SimpleNamespace(
            save_fixed_mic_session=mock.Mock(
                return_value=(
                    error_code.OK,
                    "saved",
                    "test.wav",
                    {"file_path": "test.wav", "labels": "not_labeled"},
                )
            )
        )
        fake_window = SimpleNamespace(
            current_recorded_count=0,
            lineedit_count=SimpleNamespace(setText=mock.Mock(), text=lambda: "1"),
            lineedit_type=SimpleNamespace(text=lambda: "MODEL_001"),
            lineedit_s_or_n=SimpleNamespace(text=lambda: "BARCODE_001"),
            barcode_scanner_box=SimpleNamespace(isChecked=lambda: False),
            fixed_mic_finalize_pipeline=fake_finalize_pipeline,
            recorded_path=None,
            recorded_signal_info={},
            data_struct=SimpleNamespace(
                store_wave_data=None,
                sample_rate=1,
                update_channel_count=mock.Mock(),
            ),
            count_board=SimpleNamespace(mode="mark"),
            analysis_config={"auto_analysis": True},
            default_logger=SimpleNamespace(info=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None),
            _close_analysis_windows=mock.Mock(),
            _run_fixed_mic_session_analysis=mock.Mock(),
            _emit_display_update=mock.Mock(),
            _update_fixed_mic_session_status=mock.Mock(),
            _update_fixed_mic_session_result=mock.Mock(),
            _select_fixed_mic_session_row=mock.Mock(),
            _enqueue_fixed_mic_review_session=mock.Mock(),
        )

        with mock.patch("ui.sequence.sequence_widget.save_recorded_data_to_json"):
            SequenceWindow._finalize_and_run_fixed_mic_session(fake_window, fake_session)

        fake_window._update_fixed_mic_session_status.assert_called_once_with(fake_session, "待审核")
        fake_window._update_fixed_mic_session_result.assert_called_once_with(fake_session, "not_labeled")
        fake_window._select_fixed_mic_session_row.assert_not_called()
        fake_window._enqueue_fixed_mic_review_session.assert_called_once_with(fake_session)
        fake_window._run_fixed_mic_session_analysis.assert_not_called()

    def test_finalize_fixed_mic_analysis_result_uses_ai_result_without_default_ai(self):
        fake_session = SimpleNamespace(
            session_id="fixed_mic_session_001",
            metadata={},
            analysis_result=None,
        )
        fake_ai = SimpleNamespace(
            result="OK",
            calculate_ai_scores=mock.Mock(),
            ai_analyse_score_textedit=SimpleNamespace(toPlainText=lambda: "评分结果: OK"),
        )
        fake_window = SimpleNamespace(
            recorded_signal_info={"labels": "not_labeled", "file_path": "test.wav"},
            default_ai=None,
            analysis_window=[fake_ai],
            analysis_config={"default_ai": None, "display_sequence": ["AI 分析 1"], "AI 分析 1": {"type": "AI", "analyse_model_name": "demo_model"}},
            count_board=SimpleNamespace(
                mode="test",
                set_test_result_file=mock.Mock(),
                set_test_text=mock.Mock(),
            ),
            update_recorded_signal_info_to_db=mock.Mock(),
            default_logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        )

        SequenceWindow._finalize_fixed_mic_session_analysis_result(fake_window, fake_session, use_ai_result_as_label=True)

        self.assertEqual(fake_window.recorded_signal_info["labels"], "OK")
        self.assertEqual(fake_session.analysis_result["overall_result"], "OK")
        self.assertEqual(fake_session.metadata["ai_result_text"], "评分结果: OK")
        fake_window.update_recorded_signal_info_to_db.assert_called_once()
        fake_window.count_board.set_test_result_file.assert_called_once_with("OK", "demo_model")
        fake_window.count_board.set_test_text.assert_called_once()

    def test_run_fixed_mic_session_analysis_only_runs_ai_for_auto_label(self):
        fake_session = SimpleNamespace(
            session_id="fixed_mic_session_001",
            metadata={},
            analysis_result=None,
        )
        fake_ai = SimpleNamespace(
            result="OK",
            calculate_ai_scores=mock.Mock(),
            ai_analyse_score_textedit=SimpleNamespace(toPlainText=lambda: "评分结果: OK"),
        )
        fake_spec = SimpleNamespace(calculate_spec=mock.Mock())

        fake_window = SimpleNamespace(
            analysis_window=[],
            default_ai=None,
            analysis_config={"auto_analysis": True},
            count_board=SimpleNamespace(mode="test"),
            recorded_signal_info={"labels": "not_labeled", "file_path": "test.wav"},
            update_recorded_signal_info_to_db=mock.Mock(),
            default_logger=SimpleNamespace(info=lambda *args, **kwargs: None),
            _update_fixed_mic_session_status=mock.Mock(),
            _update_fixed_mic_session_result=mock.Mock(),
            _select_fixed_mic_session_row=mock.Mock(),
            _emit_display_update=mock.Mock(),
            _get_analysis_window_position=mock.Mock(return_value=(0, 0)),
            _execute_analysis_windows=mock.Mock(),
            _show_default_ai_window=mock.Mock(),
        )

        def prepare_instances():
            fake_window.analysis_window = [fake_spec, fake_ai]
            fake_window.default_ai = None

        fake_window._prepare_analysis_instances = mock.Mock(side_effect=prepare_instances)

        SequenceWindow._run_fixed_mic_session_analysis(fake_window, fake_session, use_ai_result_as_label=True)

        fake_ai.calculate_ai_scores.assert_called_once_with("test", fake_window.analysis_config)
        fake_spec.calculate_spec.assert_not_called()
        fake_window._execute_analysis_windows.assert_not_called()
        fake_window._show_default_ai_window.assert_not_called()
        self.assertEqual(fake_session.analysis_result["overall_result"], "OK")
        fake_window._update_fixed_mic_session_result.assert_called_once_with(fake_session, "OK")

    def test_stop_fixed_mic_runtime_cleans_runtime_state(self):
        class FakeTimer(object):
            def __init__(self):
                self.stopped = False

            def isActive(self):
                return True

            def stop(self):
                self.stopped = True

        class FakeController(object):
            def __init__(self):
                self.stopped = False

            def stop_capture(self):
                self.stopped = True

        class FakeCountBoard(object):
            def __init__(self):
                self.mode = "test"
                self.review_text = None
                self.review_visible = None

            def set_review_session_text(self, text):
                self.review_text = text

            def set_review_session_visible(self, visible):
                self.review_visible = visible

        fake_timer = FakeTimer()
        fake_controller = FakeController()
        fake_count_board = FakeCountBoard()
        fake_window = SimpleNamespace(
            fixed_mic_poll_timer=fake_timer,
            fixed_mic_controller=fake_controller,
            fixed_mic_plot_item="plot",
            line_graph=SimpleNamespace(clear=mock.Mock()),
            fixed_mic_analysis_queue=["session"],
            fixed_mic_analysis_busy=True,
            player_status_flag=True,
            fixed_mic_pending_review_sessions=["pending"],
            fixed_mic_current_review_session="current",
            count_board=fake_count_board,
            _update_fixed_mic_toolbar_state=mock.Mock(),
            fixed_mic_live_y_limit=0.02,
        )

        SequenceWindow._stop_fixed_mic_runtime(fake_window)

        self.assertTrue(fake_timer.stopped)
        self.assertTrue(fake_controller.stopped)
        self.assertIsNone(fake_window.fixed_mic_controller)
        self.assertIsNone(fake_window.fixed_mic_plot_item)
        self.assertEqual(fake_window.fixed_mic_analysis_queue, [])
        self.assertFalse(fake_window.fixed_mic_analysis_busy)
        self.assertFalse(fake_window.player_status_flag)
        self.assertEqual(fake_window.fixed_mic_live_y_limit, 0.01)
        self.assertEqual(fake_window.fixed_mic_pending_review_sessions, [])
        self.assertIsNone(fake_window.fixed_mic_current_review_session)
        self.assertEqual(fake_count_board.review_text, "无")
        self.assertFalse(fake_count_board.review_visible)
        fake_window._update_fixed_mic_toolbar_state.assert_called_once()

    def test_stop_fixed_mic_runtime_preserves_review_queue_in_mark_mode(self):
        class FakeTimer(object):
            def __init__(self):
                self.stopped = False

            def isActive(self):
                return True

            def stop(self):
                self.stopped = True

        class FakeController(object):
            def __init__(self):
                self.stopped = False

            def process_audio_queue(self):
                return 0, []

            def stop_capture(self):
                self.stopped = True

        fake_timer = FakeTimer()
        fake_controller = FakeController()
        fake_count_board = SimpleNamespace(
            mode="mark",
            set_review_session_text=mock.Mock(),
            set_review_session_visible=mock.Mock(),
        )
        fake_window = SimpleNamespace(
            fixed_mic_poll_timer=fake_timer,
            fixed_mic_controller=fake_controller,
            fixed_mic_plot_item="plot",
            line_graph=SimpleNamespace(clear=mock.Mock()),
            fixed_mic_analysis_queue=[],
            fixed_mic_analysis_busy=False,
            player_status_flag=True,
            fixed_mic_pending_review_sessions=["pending_a", "pending_b"],
            fixed_mic_current_review_session="current",
            count_board=fake_count_board,
            _update_fixed_mic_toolbar_state=mock.Mock(),
            fixed_mic_live_y_limit=0.02,
            _process_next_fixed_mic_analysis=mock.Mock(),
        )

        SequenceWindow._stop_fixed_mic_runtime(fake_window)

        self.assertTrue(fake_timer.stopped)
        self.assertTrue(fake_controller.stopped)
        self.assertIsNone(fake_window.fixed_mic_controller)
        self.assertEqual(fake_window.fixed_mic_pending_review_sessions, ["pending_a", "pending_b"])
        self.assertEqual(fake_window.fixed_mic_current_review_session, "current")
        self.assertEqual(fake_window.fixed_mic_analysis_queue, [])
        self.assertFalse(fake_window.fixed_mic_analysis_busy)
        fake_count_board.set_review_session_text.assert_not_called()
        fake_count_board.set_review_session_visible.assert_called_once_with(False)
        fake_window._update_fixed_mic_toolbar_state.assert_called_once()
