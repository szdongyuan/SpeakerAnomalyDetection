import time

import pyqtgraph as pg
from PyQt5.QtWidgets import QMessageBox

from consts import error_code


def handle_fixed_mic_manual_trigger(window, controller_cls):
    if window.checked_work_status_message():
        return

    if not window.clicked_player_flag:
        window.default_logger.info("固定麦并发模式阶段2仅接收手动点击触发。")
        return

    if window.fixed_mic_controller is None:
        acq_detail = window.sequence_config[0]["seq1"]["acq"]["detail"]
        window.fixed_mic_controller = controller_cls(acq_detail, input_device=window.mic)
        start_code, start_msg = window.fixed_mic_controller.start_capture()
        if start_code != error_code.OK:
            QMessageBox.warning(window, "提示", "固定麦持续采集启动失败: %s" % start_msg)
            window.fixed_mic_controller = None
            return
        window.fixed_mic_plot_window_sec = max(
            min(float(window.fixed_mic_controller.buffer_duration), 15.0),
            float(window.fixed_mic_controller.window_duration),
        )
        window.fixed_mic_poll_timer.start(50)
        window.line_graph.clear()
        window.fixed_mic_plot_item = None
        window.fixed_mic_last_plot_update_ts = 0.0
        window.fixed_mic_live_y_limit = 0.01
        window.fixed_mic_stream_buffer = []
        window._reset_fixed_mic_session_views()
        window._configure_fixed_mic_live_plot_view()

    barcode = window.lineedit_s_or_n.text().strip() or None
    create_code, create_msg, session = window.fixed_mic_controller.create_manual_session(barcode)
    if create_code != error_code.OK:
        QMessageBox.warning(window, "提示", create_msg)
        return

    window.player_status_flag = window.fixed_mic_controller.is_running
    window.data_btn.setEnabled(False)
    window._update_fixed_mic_toolbar_state()
    window._register_fixed_mic_session(session, status_text="采集中")
    window.default_logger.info("固定麦阶段2已创建会话: %s" % session.session_id)


def poll_fixed_mic_runtime(window):
    if window.fixed_mic_controller is None:
        return

    _, completed_sessions = window.fixed_mic_controller.process_audio_queue()
    plot_chunks = window.fixed_mic_controller.consume_recent_plot_chunks()
    window._append_fixed_mic_stream_chunks(plot_chunks, window.fixed_mic_controller.sample_rate)

    current_time = time.monotonic()
    plot_interval_sec = getattr(window, "fixed_mic_plot_interval_sec", 0.12)
    if plot_chunks and current_time - getattr(window, "fixed_mic_last_plot_update_ts", 0.0) >= plot_interval_sec:
        plot_audio = window._get_fixed_mic_stream_plot_audio()
        if plot_audio.size > 0:
            time_axis, plot_audio = window.build_live_plot_data(
                plot_audio,
                window.fixed_mic_controller.sample_rate,
                max_points=4000,
            )
            if window.fixed_mic_plot_item is None:
                window.fixed_mic_plot_item = window.line_graph.plot(
                    time_axis,
                    plot_audio,
                    pen=pg.mkPen(color=(30, 30, 30), width=1),
                )
            else:
                window.fixed_mic_plot_item.setData(time_axis, plot_audio)
            window._update_fixed_mic_live_plot_range(plot_audio)
        window.fixed_mic_last_plot_update_ts = current_time

    for session in completed_sessions:
        if session.status == "frozen":
            window.fixed_mic_analysis_queue.append(session)
            window._update_fixed_mic_session_status(session, "分析中")
            window.default_logger.info(
                "固定麦阶段3切片冻结完成: session_id=%s, clip_shape=%s",
                session.session_id,
                session.metadata.get("audio_clip_shape"),
            )
        elif session.status == "cancelled":
            window._update_fixed_mic_session_status(session, "切片失败")
            window.default_logger.warning(
                "固定麦阶段3切片失败: session_id=%s, reason=%s",
                session.session_id,
                session.metadata.get("cancel_reason", ""),
            )
    window._process_next_fixed_mic_analysis()


def stop_fixed_mic_runtime(window):
    current_mode = getattr(getattr(window, "count_board", None), "mode", "test")
    preserve_review_state = current_mode == "mark"
    if hasattr(window, "fixed_mic_poll_timer") and window.fixed_mic_poll_timer.isActive():
        window.fixed_mic_poll_timer.stop()
    if getattr(window, "fixed_mic_controller", None) is not None:
        try:
            _, completed_sessions = window.fixed_mic_controller.process_audio_queue()
            for session in completed_sessions:
                if session.status == "frozen":
                    window.fixed_mic_analysis_queue.append(session)
                    window._update_fixed_mic_session_status(session, "分析中")
                elif session.status == "cancelled":
                    window._update_fixed_mic_session_status(session, "切片失败")
        except Exception:
            pass
        window.fixed_mic_controller.stop_capture()
        window.fixed_mic_controller = None
    if hasattr(window, "line_graph"):
        window.line_graph.clear()
    window.fixed_mic_plot_item = None
    window.fixed_mic_last_plot_update_ts = 0.0
    window.fixed_mic_live_y_limit = 0.01
    window.fixed_mic_stream_buffer = []
    if preserve_review_state:
        if window.fixed_mic_analysis_queue and not window.fixed_mic_analysis_busy:
            try:
                window._process_next_fixed_mic_analysis()
            except Exception:
                pass
    else:
        window.fixed_mic_analysis_queue = []
        window.fixed_mic_analysis_busy = False
    if hasattr(window, "player_status_flag"):
        window.player_status_flag = False
    if not preserve_review_state:
        window.fixed_mic_pending_review_sessions = []
        window.fixed_mic_current_review_session = None
    if getattr(window, "count_board", None) is not None:
        if not preserve_review_state:
            window.count_board.set_review_session_text("无")
        window.count_board.set_review_session_visible(False)
    window._update_fixed_mic_toolbar_state()


def process_next_fixed_mic_analysis(window, schedule_next):
    if window.fixed_mic_analysis_busy:
        return
    if not window.fixed_mic_analysis_queue:
        return

    session = window.fixed_mic_analysis_queue.pop(0)
    window.fixed_mic_analysis_busy = True
    window._update_fixed_mic_session_status(session, "分析中")
    try:
        window._finalize_and_run_fixed_mic_session(session)
    finally:
        window.fixed_mic_analysis_busy = False
        if window.fixed_mic_analysis_queue:
            schedule_next(window._process_next_fixed_mic_analysis)
