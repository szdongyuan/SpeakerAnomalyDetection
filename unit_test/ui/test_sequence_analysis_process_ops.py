from types import SimpleNamespace

import ui.sequence.sequence_widget_analysis_process_ops as analysis_process_ops
from ui.sequence.sequence_widget_analysis_process_ops import (
    SequenceWidgetAnalysisProcessOpsMixin,
)


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))

    def error(self, message):
        self.messages.append(("error", message))


class _LeftPanel:
    def __init__(self):
        self.condition_updates = []
        self.stage_updates = []
        self.stage_clears = 0
        self.final_updates = []

    def set_condition_result(self, condition_key, text, tone=None):
        self.condition_updates.append((condition_key, text, tone))

    def set_current_stage(self, text, tone=None):
        self.stage_updates.append((text, tone))

    def clear_current_stage(self):
        self.stage_clears += 1

    def set_final_result(self, text, tone=None):
        self.final_updates.append((text, tone))


class _Workspace:
    def __init__(self):
        self.context_updates = []

    def set_condition_context(self, condition_key, **context):
        self.context_updates.append((condition_key, context))


def test_automatic_request_snapshots_current_channel_labels(monkeypatch):
    captured = {}

    def _build(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(instances=())

    monkeypatch.setattr(analysis_process_ops, "build_analysis_task_request", _build)

    class Host(SequenceWidgetAnalysisProcessOpsMixin):
        def __init__(self):
            self.channel_workspace = SimpleNamespace(
                channel_layout={"CH1": "前", "CH2": "后"}
            )
            self.sequence_config = []
            self.analysis_config = {}
            self._live_mic_channel_v2pa_factors = {}
            self.default_logger = _Logger()

    Host()._build_process_analysis_request(
        "0.3",
        "D:/audio/test.wav",
        "自动分析",
        {"recorded_signal_info": {"analysis_storage": {"project_name": "P"}}},
    )

    assert captured["storage_snapshot"]["project_name"] == "P"
    assert captured["storage_snapshot"]["channel_labels"] == {
        "CH1": "前",
        "CH2": "后",
    }


def test_automatic_admission_error_is_not_marked_result_incomplete():
    class Host(SequenceWidgetAnalysisProcessOpsMixin):
        def __init__(self):
            self.default_logger = _Logger()
            self.condition_updates = []
            self.admission_updates = []

        @staticmethod
        def _should_run_silent_analysis_after_recording():
            return True

        @staticmethod
        def _get_active_product_condition_key():
            return "group_1:condition_1"

        @staticmethod
        def _resolve_condition_record(_condition_key):
            return {"session_id": "session-1"}

        @staticmethod
        def _analysis_record_wav_path(_record):
            return "C:/record.wav"

        @staticmethod
        def _build_process_analysis_request(*_args):
            raise analysis_process_ops.AnalysisTaskBuildError("缺少 CH5")

        def _set_condition_analysis_stage(self, condition_key, text, tone):
            self.condition_updates.append((condition_key, text, tone))

        def _record_analysis_admission_state(self, record, **state):
            self.admission_updates.append((record, state))

    host = Host()

    assert host._enqueue_automatic_analysis_current_recording() is True
    assert host.condition_updates == [
        ("group_1:condition_1", "分析失败", "ng")
    ]
    assert host.admission_updates == [
        (
            {"session_id": "session-1"},
            {
                "state": "failed",
                "status": "分析失败",
                "error": "缺少 CH5",
            },
        )
    ]


def test_channel_results_keep_non_judging_spec_as_analysis_complete():
    task_result = SimpleNamespace(
        instance_results=(
            SimpleNamespace(
                raw_channel=0,
                analysis_type="SPL",
                execution_status="分析完成",
                contributes_to_final=True,
                judgement="OK",
            ),
            SimpleNamespace(
                raw_channel=0,
                analysis_type="Spec",
                execution_status="分析完成",
                contributes_to_final=False,
                judgement=None,
            ),
        )
    )

    rows = SequenceWidgetAnalysisProcessOpsMixin._build_process_channel_results(
        task_result
    )

    assert rows[0]["SPL"] == "OK"
    assert rows[0]["Spec"] == "分析完成"
    assert rows[0]["result"] == "OK"


def test_progress_keeps_short_analysis_status_in_header_and_row():
    progress = SimpleNamespace(
        stage="分析中",
        message="声压级 (SPL) 1 CH1",
    )

    class _Service:
        active = True

        @staticmethod
        def poll():
            return [("progress", progress)], []

    class Host(SequenceWidgetAnalysisProcessOpsMixin):
        def __init__(self):
            self.left_panel = _LeftPanel()
            self._analysis_process_service = _Service()
            self._analysis_active_request = SimpleNamespace(
                source="自动分析",
                condition_key="group_1:condition_2",
            )

        @staticmethod
        def _write_analysis_worker_log(_record):
            return None

        @staticmethod
        def _refresh_analysis_action_state():
            return None

    host = Host()

    host._poll_analysis_process_runtime()

    assert host.left_panel.condition_updates == [
        ("group_1:condition_2", "分析中", "running")
    ]
    assert host.left_panel.stage_updates == [
        ("分析中", "running")
    ]


def test_analysis_stage_is_synchronized_to_waveform_header():
    class Host(SequenceWidgetAnalysisProcessOpsMixin):
        def __init__(self):
            self.left_panel = _LeftPanel()
            self.channel_workspace = _Workspace()

    host = Host()

    host._set_condition_analysis_stage(
        "group_1:condition_1",
        "分析排队",
        "running",
    )

    assert host.channel_workspace.context_updates == [
        ("group_1:condition_1", {"status": "分析排队"})
    ]


def test_deferred_round_completion_replaces_stale_background_status():
    class Host(SequenceWidgetAnalysisProcessOpsMixin):
        def __init__(self):
            self.left_panel = _LeftPanel()
            self._analysis_round_completion_pending = True
            self._analysis_deferred_serial_close = False
            self._analysis_process_service = None
            self._analysis_active_request = None
            self._analysis_task_queue = ()
            self._manual_product_condition_group_id = "group-1"
            self._current_cycle_recorded_count = 2
            self.unlocked = False
            self.player_state_refreshed = False

        @staticmethod
        def _automatic_round_display_state():
            return "OK", "ok", True

        def _unlock_analysis_round_config(self):
            self.unlocked = True

        def update_player_btn_is_paused(self):
            self.player_state_refreshed = True

    host = Host()

    assert host._finish_deferred_analysis_round_if_ready() is True

    assert host.left_panel.final_updates == [("OK", "ok")]
    assert host.left_panel.stage_updates == [("本轮完成", "ok")]
    assert host._analysis_round_completion_pending is False
    assert host.unlocked is True
    assert host.player_state_refreshed is True


def test_deferred_serial_close_uses_normal_round_completion_title():
    class Host(SequenceWidgetAnalysisProcessOpsMixin):
        def __init__(self):
            self.left_panel = _LeftPanel()
            self._analysis_round_completion_pending = True
            self._analysis_deferred_serial_close = True
            self._analysis_process_service = None
            self._analysis_active_request = None
            self._analysis_task_queue = ()
            self._serial_product_waiting_for_close = False
            self.player_state_refreshed = False

        @staticmethod
        def _automatic_round_display_state():
            return "OK", "ok", True

        def update_player_btn_is_paused(self):
            self.player_state_refreshed = True

    host = Host()

    assert host._finish_deferred_analysis_round_if_ready() is True

    assert host.left_panel.final_updates == [("OK", "ok")]
    assert host.left_panel.stage_updates == [("本轮完成", "ok")]
    assert host._serial_product_waiting_for_close is True
    assert host.player_state_refreshed is True


def test_terminal_result_does_not_leave_background_analysis_stage():
    class _Service:
        active = True

    class Host(SequenceWidgetAnalysisProcessOpsMixin):
        def __init__(self):
            self.left_panel = _LeftPanel()
            self._analysis_process_service = _Service()
            self._analysis_active_request = SimpleNamespace(source="自动分析")
            self._analysis_task_queue = ()

    host = Host()
    result = SimpleNamespace(
        condition_key="group_1:condition_1",
        instance_results=(),
        final_judgement="OK",
    )

    host._sync_process_result_to_condition_panel(result, "OK", "ok")

    assert host.left_panel.condition_updates == [
        ("group_1:condition_1", "OK", "ok")
    ]
    assert host.left_panel.stage_updates == []


def test_completed_analysis_replaces_collecting_waveform_status():
    class Host(SequenceWidgetAnalysisProcessOpsMixin):
        def __init__(self):
            self.left_panel = _LeftPanel()
            self.channel_workspace = _Workspace()
            self._analysis_process_service = None
            self._analysis_active_request = None
            self._analysis_task_queue = ()

    host = Host()
    result = SimpleNamespace(
        condition_key="group_1:condition_1",
        instance_results=(),
        execution_status="分析完成",
        final_judgement="OK",
    )

    host._sync_process_result_to_condition_panel(result, "OK", "ok")

    assert host.channel_workspace.context_updates == [
        ("group_1:condition_1", {"status": "分析完成"})
    ]


def test_idle_automatic_analysis_restores_waiting_next_condition_stage():
    result = SimpleNamespace(task_id="auto-1")

    class _Service:
        active = False

        @staticmethod
        def poll():
            return [("result", result)], []

    class Host(SequenceWidgetAnalysisProcessOpsMixin):
        def __init__(self):
            self.left_panel = _LeftPanel()
            self._analysis_process_service = _Service()
            self._analysis_active_request = SimpleNamespace(
                task_id="auto-1",
                source="自动分析",
            )
            self._analysis_handled_terminal_task_ids = set()
            self._analysis_round_completion_pending = False
            self._analysis_task_queue = ()
            self.player_status_flag = False
            self._record_workflow_busy = False

        def _handle_analysis_terminal(self, terminal_result):
            self._analysis_handled_terminal_task_ids.add(terminal_result.task_id)

        @staticmethod
        def _start_next_queued_analysis():
            return False

        @staticmethod
        def _write_analysis_worker_log(_record):
            return None

        @staticmethod
        def _refresh_analysis_action_state():
            return None

    host = Host()

    host._poll_analysis_process_runtime()

    assert host.left_panel.stage_updates == [("等待下一档位", "pending")]
    assert host._analysis_active_request is None


def test_finished_analysis_does_not_overwrite_active_recording_stage():
    class Host(SequenceWidgetAnalysisProcessOpsMixin):
        def __init__(self):
            self.left_panel = _LeftPanel()
            self._analysis_process_service = None
            self._analysis_active_request = None
            self._analysis_task_queue = ()
            self.player_status_flag = True
            self._record_workflow_busy = True

    host = Host()

    assert host._restore_waiting_stage_after_automatic_analysis() is True
    assert host.left_panel.stage_updates == []
    assert host.left_panel.stage_clears == 1


def test_manual_terminal_waits_for_click_without_updating_official_state(monkeypatch):
    class Host(SequenceWidgetAnalysisProcessOpsMixin):
        def __init__(self):
            self._analysis_handled_terminal_task_ids = set()
            self._analysis_task_records = {"manual-1": {"recorded_path": "x.wav"}}
            self._analysis_manual_requested_at = {"manual-1": 100.0}
            self._analysis_manual_source_labels = {"manual-1": "A口 / 0.3"}
            self._analysis_active_request = SimpleNamespace(
                analysis_config_snapshot=SimpleNamespace(to_dict=lambda: {})
            )
            self.default_logger = _Logger()
            self.shown = None

        def _show_manual_analysis_result_windows(self, result, **options):
            self.shown = (result, options)
            return 1

        def _set_condition_analysis_stage(self, *_args):
            raise AssertionError("手动查看不得修改正式档位状态")

        @staticmethod
        def _refresh_analysis_action_state():
            return None

    host = Host()
    result = SimpleNamespace(
        task_id="manual-1",
        source="手动查看",
        condition_key="0.3",
        instance_results=(SimpleNamespace(),),
        execution_status="分析完成",
        error_message="",
    )

    monkeypatch.setattr(analysis_process_ops.time, "monotonic", lambda: 108.25)
    host._handle_analysis_terminal(result)

    assert host.shown is None
    assert host._analysis_pending_manual_view["result"] is result
    assert host._analysis_pending_manual_view["source_label"] == "A口 / 0.3"
    assert "manual-1" not in host._analysis_task_records
    assert "manual-1" not in host._analysis_manual_requested_at
    assert any(
        "manual_analysis_result_ready" in message
        and "duration_seconds=8.250" in message
        for _level, message in host.default_logger.messages
    )

    assert host._show_pending_manual_analysis_view() is True
    assert host.shown[0] is result
    assert host.shown[1]["source_label"] == "A口 / 0.3"
    assert host._analysis_pending_manual_view is None
    assert any(
        "manual_analysis_windows_opened" in message
        and "window_count=1" in message
        for _level, message in host.default_logger.messages
    )


def test_manual_progress_updates_analysis_button_without_touching_recording_stage():
    progress = SimpleNamespace(
        stage="分析中",
        message="已完成 7/20",
        completed_instances=7,
        total_instances=20,
    )

    class _Service:
        active = True

        @staticmethod
        def poll():
            return [("progress", progress)], []

    class _Button:
        def __init__(self):
            self.progress = None

        def set_analyzing(self, completed, total, source_label):
            self.progress = (completed, total, source_label)

    class Host(SequenceWidgetAnalysisProcessOpsMixin):
        def __init__(self):
            self.data_btn = _Button()
            self._analysis_process_service = _Service()
            self._analysis_active_request = SimpleNamespace(
                task_id="manual-1",
                source="手动查看",
                condition_key="0.1",
            )
            self._analysis_manual_source_labels = {"manual-1": "A口 / 0.1"}

        @staticmethod
        def _write_analysis_worker_log(_record):
            return None

        @staticmethod
        def _refresh_analysis_action_state():
            return None

        def _set_condition_analysis_stage(self, *_args, **_kwargs):
            raise AssertionError("手动分析进度不得覆盖录音状态栏")

    host = Host()
    host._poll_analysis_process_runtime()

    assert host.data_btn.progress == (7, 20, "A口 / 0.1")


def test_recording_only_blocks_manual_analysis_for_the_active_condition(tmp_path):
    active_wav = tmp_path / "active.wav"
    old_wav = tmp_path / "old.wav"
    active_wav.write_bytes(b"active")
    old_wav.write_bytes(b"old")

    class Host(SequenceWidgetAnalysisProcessOpsMixin):
        player_status_flag = True
        _record_workflow_busy = True
        recorded_path = str(active_wav)

        @staticmethod
        def _get_active_product_condition_key():
            return "0.3"

    host = Host()

    assert host._manual_analysis_target_is_recording("0.3", str(active_wav)) is True
    assert host._manual_analysis_target_is_recording("0.1", str(old_wav)) is False


def test_analysis_button_stays_available_for_completed_other_condition_during_recording(
    tmp_path,
):
    active_wav = tmp_path / "active.wav"
    old_wav = tmp_path / "old.wav"
    active_wav.write_bytes(b"active")
    old_wav.write_bytes(b"old")

    class _Button:
        def __init__(self):
            self.enabled = None

        def setEnabled(self, enabled):
            self.enabled = enabled

    class Host(SequenceWidgetAnalysisProcessOpsMixin):
        def __init__(self):
            self.data_btn = _Button()
            self.selected_condition = "0.1"
            self.player_status_flag = True
            self._record_workflow_busy = True
            self.recorded_path = str(active_wav)
            self._analysis_process_service = None
            self._analysis_active_request = None
            self._analysis_task_queue = ()
            self._analysis_pending_manual_view = None
            self.records = {
                "0.1": {"recorded_path": str(old_wav)},
                "0.3": {"recorded_path": str(active_wav)},
            }

        def _selected_analysis_condition_key(self):
            return self.selected_condition

        def _resolve_condition_record(self, condition_key):
            return self.records[condition_key]

        @staticmethod
        def _resolve_audio_path_to_abs(path):
            return str(path)

        @staticmethod
        def _get_active_product_condition_key():
            return "0.3"

    host = Host()

    host._refresh_analysis_action_state()
    assert host.data_btn.enabled is True

    host.selected_condition = "0.3"
    host._refresh_analysis_action_state()
    assert host.data_btn.enabled is False


def test_manual_result_windows_receive_current_channel_position_labels(monkeypatch):
    created = []

    class _Signal:
        @staticmethod
        def connect(_callback):
            return None

    class _Window:
        def __init__(self, config_key, instance_results, **options):
            created.append((config_key, tuple(instance_results), options))
            self.destroyed = _Signal()

        @staticmethod
        def setAttribute(*_args):
            return None

        @staticmethod
        def show():
            return None

        @staticmethod
        def raise_():
            return None

    monkeypatch.setattr(
        analysis_process_ops,
        "AnalysisMultichannelResultWindow",
        _Window,
    )

    class Host(SequenceWidgetAnalysisProcessOpsMixin):
        def __init__(self):
            self.analysis_window = []
            self.channel_workspace = SimpleNamespace(
                channel_layout={"CH1": "前", "CH2": "后"}
            )
            self._analysis_active_request = SimpleNamespace(
                analysis_config_snapshot=SimpleNamespace(
                    to_dict=lambda: {"声压级": {"type": "SPL"}}
                )
            )

    host = Host()
    instance = SimpleNamespace(config_key="声压级")
    host._show_manual_analysis_result_windows(
        SimpleNamespace(instance_results=(instance,))
    )

    assert created[0][2]["channel_labels"] == {
        "CH1": "前",
        "CH2": "后",
    }


def test_artifact_failure_is_visible_without_changing_official_label():
    class Host(SequenceWidgetAnalysisProcessOpsMixin):
        def __init__(self):
            self.default_logger = _Logger()
            self.synced = None
            self.updated_label = None

        def _update_analysis_record_label(
            self,
            _condition_key,
            _wav_path,
            record,
            label,
        ):
            self.updated_label = label
            return record

        def _update_process_result_session_snapshot(self, *_args):
            return None

        def _sync_process_result_to_condition_panel(
            self,
            _result,
            display_text,
            tone,
        ):
            self.synced = (display_text, tone)

    host = Host()
    result = SimpleNamespace(
        task_id="auto-1",
        condition_key="0.3",
        wav_path="C:/record.wav",
        execution_status="分析完成",
        judgement_status="已判定",
        final_judgement="OK",
        instance_results=(
            SimpleNamespace(
                artifacts=(SimpleNamespace(status="保存失败"),),
            ),
        ),
    )

    host._apply_automatic_analysis_result(result, {"recorded_path": "record.wav"})

    assert host.updated_label == "OK"
    assert host.synced == ("OK", "ok")
    assert any("artifact_save_failed" in message for _, message in host.default_logger.messages)


def test_task_level_failure_marks_report_snapshot_failed():
    class Host(SequenceWidgetAnalysisProcessOpsMixin):
        def __init__(self):
            self.product_test_pdf_report_config = {"enabled": True}
            self.analysis_config = {}
            self._analysis_active_request = None
            self.updated = None
            self.default_logger = _Logger()

        @staticmethod
        def _format_recent_session_result_label(label):
            return label

        def _update_recent_session(self, session_id, **changes):
            self.updated = (session_id, changes)

    host = Host()
    result = SimpleNamespace(
        task_id="auto-2",
        execution_status="分析失败",
        error_message="子进程异常退出",
        wav_path="C:/record.wav",
        instance_results=(),
    )

    host._update_process_result_session_snapshot(
        result,
        {
            "session_id": "session-1",
            "recorded_path": "C:/record.wav",
            "recorded_signal_info": {},
        },
        "not_labeled",
    )

    assert host.updated[0] == "session-1"
    changes = host.updated[1]
    assert changes["analysis_report_state"] == "failed"
    assert changes["analysis_report_items"][0]["error"] == "子进程异常退出"


def test_disabled_automatic_analysis_closes_report_wait_state():
    class Host(SequenceWidgetAnalysisProcessOpsMixin):
        product_test_pdf_report_config = {"enabled": True}

        @staticmethod
        def _format_recent_session_result_label(label):
            return label

        def _update_recent_session(self, session_id, **changes):
            self.updated = (session_id, changes)

    host = Host()
    host._record_analysis_admission_state(
        {
            "session_id": "session-2",
            "recorded_path": "C:/record.wav",
            "recorded_signal_info": {"labels": "not_labeled"},
        },
        state="not_required",
        status="未启用自动分析",
    )

    assert host.updated[0] == "session-2"
    assert host.updated[1]["analysis_report_state"] == "not_required"
    assert host.updated[1]["analysis_report_items"] == []


def test_worker_log_keeps_compact_item_summary_in_main_and_raw_detail_in_debug():
    class Host(SequenceWidgetAnalysisProcessOpsMixin):
        def __init__(self):
            self.default_logger = _Logger()
            self._analysis_debug_logger = _Logger()

    host = Host()
    record = {
        "level": "INFO",
        "event": "analysis_item_finished",
        "task_id": "task-1",
        "source": "自动分析",
        "condition_key": "group_1:condition_2",
        "wav_path": "D:/very/long/path/source.wav",
        "config_key": "声压级 (SPL) 1",
        "analysis_type": "SPL",
        "channel_count": 5,
        "successful_channels": 5,
        "failed_channels": 0,
        "duration_seconds": 3.453,
    }

    host._write_analysis_worker_log(record)

    assert len(host.default_logger.messages) == 1
    level, message = host.default_logger.messages[0]
    assert level == "info"
    assert message.startswith("analysis_item_finished |")
    assert "item=声压级 (SPL) 1" in message
    assert "success=5" in message
    assert "duration_seconds=3.453" in message
    assert "wav_path" not in message
    assert host._analysis_debug_logger.messages[0][0] == "info"
    assert '"wav_path": "D:/very/long/path/source.wav"' in (
        host._analysis_debug_logger.messages[0][1]
    )


def test_worker_log_routes_successful_instance_detail_to_debug_only():
    class Host(SequenceWidgetAnalysisProcessOpsMixin):
        def __init__(self):
            self.default_logger = _Logger()
            self._analysis_debug_logger = _Logger()

    host = Host()
    host._write_analysis_worker_log(
        {
            "level": "INFO",
            "event": "analysis_instance_finished",
            "task_id": "task-1",
            "source": "自动分析",
            "condition_key": "group_1:condition_2",
            "runtime_key": "声压级 (SPL) 1--通道1",
            "duration_seconds": 1.328,
        }
    )

    assert host.default_logger.messages == []
    assert len(host._analysis_debug_logger.messages) == 1
    assert "analysis_instance_finished" in host._analysis_debug_logger.messages[0][1]


def test_worker_log_keeps_compact_task_completion_in_main():
    class Host(SequenceWidgetAnalysisProcessOpsMixin):
        def __init__(self):
            self.default_logger = _Logger()
            self._analysis_debug_logger = _Logger()

    host = Host()
    host._write_analysis_worker_log(
        {
            "level": "INFO",
            "event": "analysis_task_finished",
            "task_id": "task-1",
            "source": "手动查看",
            "condition_key": "group_1:condition_2",
            "execution_status": "分析完成",
            "final_judgement": "OK",
            "duration_seconds": 7.156,
        }
    )

    assert host.default_logger.messages == [
        (
            "info",
            "analysis_task_finished | task_id=task-1 | source=手动查看 | "
            "condition=group_1:condition_2 | execution=分析完成 | result=OK | "
            "duration_seconds=7.156",
        )
    ]


def test_worker_log_keeps_save_failure_in_main_and_full_detail_in_debug():
    class Host(SequenceWidgetAnalysisProcessOpsMixin):
        def __init__(self):
            self.default_logger = _Logger()
            self._analysis_debug_logger = _Logger()

    host = Host()
    host._write_analysis_worker_log(
        {
            "level": "WARNING",
            "event": "analysis_image_save_failed",
            "task_id": "task-1",
            "source": "自动分析",
            "condition_key": "group_1:condition_2",
            "runtime_key": "声压级 (SPL) 1--通道1",
            "artifact_path": "D:/result/CH1.png",
            "error_message": "磁盘写入失败",
        }
    )

    assert host.default_logger.messages == [
        (
            "warning",
            "analysis_image_save_failed | task_id=task-1 | source=自动分析 | "
            "condition=group_1:condition_2 | runtime=声压级 (SPL) 1--通道1 | "
            "error=磁盘写入失败 | path=D:/result/CH1.png",
        )
    ]
    assert len(host._analysis_debug_logger.messages) == 1
