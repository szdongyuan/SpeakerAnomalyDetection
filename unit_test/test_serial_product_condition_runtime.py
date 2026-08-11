import ast
from pathlib import Path
from types import SimpleNamespace

from ui.sequence.sequence_widget_serial_trigger_ops import SequenceWidgetSerialTriggerOpsMixin


FRAME_6000 = "01 04 02 00 01 78 F0"
FRAME_7000 = "FE 02 01 02 91 9C"
FRAME_8000 = "01 04 02 00 03 F9 31"
FRAME_IDLE = "01 04 02 00 00 B9 30"
FRAME_CLOSE = "01 04 02 00 04 B8 F3"
ANALYSIS_OPS_PATH = (
    Path(__file__).resolve().parents[1]
    / "ui"
    / "sequence"
    / "sequence_widget_analysis_ops.py"
)


def _load_analysis_method(method_name):
    module_tree = ast.parse(ANALYSIS_OPS_PATH.read_text(encoding="utf-8"))
    mixin = next(
        node
        for node in module_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SequenceWidgetAnalysisOpsMixin"
    )
    method = next(
        node
        for node in mixin.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    test_class = ast.ClassDef(
        name="TestAnalysisMixin",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[test_class], type_ignores=[])),
            str(ANALYSIS_OPS_PATH),
            "exec",
        ),
        namespace,
    )
    return getattr(namespace["TestAnalysisMixin"], method_name)


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
        self.stages = []
        self.condition_results = []
        self.final_results = []

    def set_current_stage(self, text, tone=None):
        self.stages.append((text, tone))

    def set_condition_result(self, condition_key, text, tone=None):
        self.condition_results.append((condition_key, text, tone))

    def set_final_result(self, text, tone=None):
        self.final_results.append((text, tone))


class _Button:
    def __init__(self):
        self.disabled = False

    def setDisabled(self, disabled):
        self.disabled = bool(disabled)

    def setText(self, _text):
        return None

    def setToolTip(self, _text):
        return None

    def setAccessibleName(self, _text):
        return None

    def setAccessibleDescription(self, _text):
        return None

    def setStyleSheet(self, _style):
        return None


class _SerialProductHost(SequenceWidgetSerialTriggerOpsMixin):
    def __init__(self):
        self.product_test_condition_configs = [
            {
                "condition_name": "6000 rpm",
                "trigger_state": FRAME_6000,
                "test_queue": "queue_6000",
            },
            {
                "condition_name": "7000 rpm",
                "trigger_state": FRAME_7000,
                "test_queue": "queue_7000",
            },
            {
                "condition_name": "8000 rpm",
                "trigger_state": FRAME_8000,
                "test_queue": "queue_8000",
            },
        ]
        self._manual_product_condition_index = 0
        self._manual_product_condition_group_id = ""
        self._displayed_manual_product_condition_group_id = ""
        self._manual_product_condition_results = {}
        self._manual_product_condition_completed_keys = set()
        self._manual_product_condition_counted_group_labels = {}
        self._active_product_condition_key = ""
        self._active_product_condition_config = None
        self._waveform_display_override_direction = ""
        self._current_trigger_direction = ""
        self._current_cycle_recorded_count = None
        self._record_workflow_busy = False
        self.player_status_flag = False
        self._serial_product_condition_executing = False
        self._serial_product_session_started = False
        self._serial_product_latched_frame = ""
        self._serial_product_waiting_for_close = False
        self._serial_product_pending_close_frame = ""
        self.product_test_close_trigger_state = ""
        self._serial_product_error_dialog_open = False
        self._product_test_program_config_dialog_open = False
        self._queued_directional_trigger = ""
        self._pending_serial_trigger_direction = ""
        self.clicked_player_flag = False
        self.count_board = SimpleNamespace(mode="test")
        self.data_struct = SimpleNamespace(analysis_result_dict={})
        self.default_logger = _Logger()
        self.left_panel = _LeftPanel()
        self.started = []
        self.reset_count = 0
        self.cleanup_count = 0
        self.discard_count = 0
        self.discarded_groups = []
        self.pause_update_count = 0
        self.unlock_count = 0
        self.data_btn = _Button()
        self.replayer_btn = _Button()
        self.serial_trigger_btn = _Button()
        self._token_seq = 0
        self.cleared_waveforms = 0
        self.loaded_queues = []

    def _product_condition_sequence(self):
        return [dict(item) for item in self.product_test_condition_configs]

    def _prepare_next_manual_product_condition_recording(self):
        condition = self.product_test_condition_configs[self._manual_product_condition_index]
        self.loaded_queues.append(condition["test_queue"])
        group_id = self._manual_product_condition_group_id
        if not group_id:
            group_id = self._generate_recording_token()
            self._manual_product_condition_group_id = group_id
            self._displayed_manual_product_condition_group_id = group_id
            self._current_cycle_recorded_count = group_id
            self._manual_product_condition_results = {}
            self._manual_product_condition_completed_keys = set()
            self.clear_all_direction_waveforms()
        self._active_product_condition_key = condition["trigger_state"]
        self._active_product_condition_config = dict(condition)
        self._waveform_display_override_direction = self._active_product_condition_key
        self.left_panel.set_condition_result(
            self._active_product_condition_key,
            "采集中",
            tone="running",
        )
        return True

    def _generate_recording_token(self):
        self._token_seq += 1
        return f"round-{self._token_seq}"

    def clear_all_direction_waveforms(self):
        self.cleared_waveforms += 1

    def _is_import_audio_mode(self):
        return False

    def _get_active_product_condition_key(self):
        return self._active_product_condition_key

    def start_this_play(self, _label):
        self.started.append(self._active_product_condition_key)
        self._record_workflow_busy = True
        self.player_status_flag = True

    def complete_current(self, result):
        active_key = self._active_product_condition_key
        if result:
            self._manual_product_condition_results[active_key] = result
        self._manual_product_condition_completed_keys.add(active_key)
        assert self._finalize_serial_product_condition_after_analysis()
        condition_keys = [item["trigger_state"] for item in self.product_test_condition_configs]
        if set(condition_keys).issubset(self._manual_product_condition_completed_keys):
            self._manual_product_condition_index = 0
            if self.product_test_close_trigger_state:
                self._serial_product_waiting_for_close = True
            else:
                self._manual_product_condition_group_id = ""
                self._current_cycle_recorded_count = None
        else:
            self._manual_product_condition_index = next(
                index
                for index, condition_key in enumerate(condition_keys)
                if condition_key not in self._manual_product_condition_completed_keys
            )
        self._active_product_condition_key = ""
        self._active_product_condition_config = None
        self._waveform_display_override_direction = ""
        self._record_workflow_busy = False
        self.player_status_flag = False
        self._on_serial_product_condition_completed()

    def _cleanup_streaming_resources(self):
        self.cleanup_count += 1

    def _discard_current_recent_session(self):
        self.discard_count += 1

    def _delete_serial_product_round_records(self, group_id):
        self.discarded_groups.append((group_id, True))

    def _unlock_sn_after_recording_if_needed(self):
        self.unlock_count += 1

    def _unlock_sn_for_product_round(self, clear=False):
        if clear:
            self.unlock_count += 1

    def update_player_btn_is_paused(self):
        self.pause_update_count += 1

    def _reset_manual_product_condition_cycle(self, clear_waveforms=False):
        self.reset_count += 1
        self._manual_product_condition_index = 0
        self._manual_product_condition_group_id = ""
        self._displayed_manual_product_condition_group_id = ""
        self._manual_product_condition_results = {}
        self._manual_product_condition_completed_keys = set()
        self._active_product_condition_key = ""
        self._active_product_condition_config = None
        self._waveform_display_override_direction = ""
        self._current_cycle_recorded_count = None
        self._serial_product_waiting_for_close = False
        if clear_waveforms:
            self.clear_all_direction_waveforms()


def _payload(frame):
    return {"raw_hex": frame, "product_full_frame": True}


def test_serial_frames_follow_fixture_order_within_one_round():
    host = _SerialProductHost()

    host.on_serial_full_frame_received(_payload(FRAME_7000))
    first_group_id = host._manual_product_condition_group_id
    assert first_group_id == "round-1"
    assert host.started == [FRAME_7000]
    assert host.loaded_queues == ["queue_7000"]
    host.complete_current("OK")

    host.on_serial_full_frame_received(_payload(FRAME_6000))
    assert host._manual_product_condition_group_id == first_group_id
    assert host.started == [FRAME_7000, FRAME_6000]
    host.complete_current("NG")

    host.on_serial_full_frame_received(_payload(FRAME_8000))
    assert host._manual_product_condition_group_id == first_group_id
    assert host.started == [FRAME_7000, FRAME_6000, FRAME_8000]
    host.complete_current("OK")

    assert host._manual_product_condition_index == 0
    assert host._manual_product_condition_group_id == ""
    assert host._displayed_manual_product_condition_group_id == first_group_id


def test_production_cycle_methods_keep_fixture_order_in_one_group():
    prepare = _load_analysis_method("_prepare_next_manual_product_condition_recording")
    advance = _load_analysis_method("_advance_manual_product_condition_cycle_after_recording")
    host = _SerialProductHost()
    host._serial_product_condition_executing = True
    host._product_condition_runtime_key = (
        lambda condition, _index=0: condition["trigger_state"]
    )
    host._set_product_condition_round_pending = lambda: None

    def load_condition(condition):
        host.loaded_queues.append(condition["test_queue"])
        return True, ""

    host._load_sequence_config_for_product_condition = load_condition

    for index, frame in ((1, FRAME_7000), (0, FRAME_6000), (2, FRAME_8000)):
        host._manual_product_condition_index = index
        assert prepare(host) is True
        group_id = host._manual_product_condition_group_id
        assert group_id == "round-1"
        assert host._active_product_condition_key == frame
        host._manual_product_condition_completed_keys.add(frame)
        advance(host)

    assert host.loaded_queues == ["queue_7000", "queue_6000", "queue_8000"]
    assert host._manual_product_condition_completed_keys == {
        FRAME_6000,
        FRAME_7000,
        FRAME_8000,
    }
    assert host._manual_product_condition_group_id == ""
    assert host._manual_product_condition_index == 0


def test_production_cycle_waits_for_close_frame_when_configured():
    prepare = _load_analysis_method("_prepare_next_manual_product_condition_recording")
    advance = _load_analysis_method("_advance_manual_product_condition_cycle_after_recording")
    host = _SerialProductHost()
    host.product_test_close_trigger_state = FRAME_CLOSE
    host._serial_product_condition_executing = True
    host._product_condition_runtime_key = (
        lambda condition, _index=0: condition["trigger_state"]
    )
    host._set_product_condition_round_pending = lambda: None
    host._load_sequence_config_for_product_condition = lambda _condition: (True, "")

    for index, frame in ((2, FRAME_8000), (0, FRAME_6000), (1, FRAME_7000)):
        host._manual_product_condition_index = index
        assert prepare(host) is True
        host._manual_product_condition_completed_keys.add(frame)
        advance(host)

    assert host._manual_product_condition_group_id == "round-1"
    assert host._serial_product_waiting_for_close is True
    assert host.left_panel.stages[-1] == ("全部工况完成，等待关闭测试", "ok")


def test_manual_play_is_ignored_while_serial_round_waits_for_close():
    on_clicked_player_btn = _load_analysis_method("on_clicked_player_btn")
    host = _SerialProductHost()
    host._serial_product_waiting_for_close = True

    on_clicked_player_btn(host)

    assert host.loaded_queues == []
    assert host.started == []


def test_explicit_close_finishes_round_and_allows_same_condition_in_new_round():
    host = _SerialProductHost()
    host.product_test_close_trigger_state = FRAME_CLOSE

    for frame in (FRAME_8000, FRAME_6000, FRAME_7000):
        host.on_serial_full_frame_received(_payload(frame))
        host.complete_current("OK")

    assert host._manual_product_condition_group_id == "round-1"
    assert host._serial_product_waiting_for_close is True
    host.on_serial_full_frame_received(_payload(FRAME_CLOSE))

    assert host._manual_product_condition_group_id == ""
    assert host._serial_product_waiting_for_close is False
    assert host._displayed_manual_product_condition_group_id == "round-1"
    assert host.left_panel.stages[-1] == ("本轮测试已关闭", "ok")

    host.on_serial_full_frame_received(_payload(FRAME_7000))
    assert host._manual_product_condition_group_id == "round-2"
    assert host.started[-1] == FRAME_7000


def test_idle_close_frame_between_conditions_is_ignored_and_round_continues(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        "ui.sequence.sequence_widget_serial_trigger_ops.QMessageBox.warning",
        lambda *_args: warnings.append(_args[-1]),
    )
    host = _SerialProductHost()
    host.product_test_close_trigger_state = FRAME_CLOSE
    host.on_serial_full_frame_received(_payload(FRAME_6000))
    host.complete_current("OK")

    host.on_serial_full_frame_received(_payload(FRAME_CLOSE))
    host.on_serial_full_frame_received(_payload(FRAME_CLOSE))

    assert host.discarded_groups == []
    assert host._manual_product_condition_group_id == "round-1"
    assert host._serial_product_waiting_for_close is False
    assert host.cleared_waveforms == 1
    assert warnings == []
    assert sum(
        1
        for level, message in host.default_logger.messages
        if level == "info" and "serial_product_idle_ignored_incomplete_round" in message
    ) == 2

    host.on_serial_full_frame_received(_payload(FRAME_7000))

    assert host._manual_product_condition_group_id == "round-1"
    assert host.started == [FRAME_6000, FRAME_7000]


def test_idle_close_frame_during_nonfinal_condition_is_ignored():
    host = _SerialProductHost()
    host.product_test_close_trigger_state = FRAME_CLOSE
    host.on_serial_full_frame_received(_payload(FRAME_6000))

    host.on_serial_full_frame_received(_payload(FRAME_CLOSE))

    assert host._serial_product_pending_close_frame == ""
    assert host._manual_product_condition_group_id == "round-1"
    assert host.discarded_groups == []
    assert host.reset_count == 0
    assert host.started == [FRAME_6000]


def test_idle_close_frame_during_final_condition_closes_after_analysis():
    host = _SerialProductHost()
    host.product_test_close_trigger_state = FRAME_CLOSE
    for frame in (FRAME_6000, FRAME_7000):
        host.on_serial_full_frame_received(_payload(frame))
        host.complete_current("OK")

    host.on_serial_full_frame_received(_payload(FRAME_8000))
    # Recording completion marks the condition before synchronous analysis runs.
    host._manual_product_condition_completed_keys.add(FRAME_8000)
    host.on_serial_full_frame_received(_payload(FRAME_CLOSE))

    assert host._serial_product_pending_close_frame == FRAME_CLOSE
    assert host._manual_product_condition_group_id == "round-1"
    assert host.discarded_groups == []

    host.complete_current("OK")

    assert host._serial_product_pending_close_frame == ""
    assert host._manual_product_condition_group_id == ""
    assert host._serial_product_waiting_for_close is False
    assert host.discarded_groups == []
    assert host.left_panel.stages[-1] == ("本轮测试已关闭", "ok")


def test_close_frame_without_active_round_is_idempotently_ignored():
    host = _SerialProductHost()
    host.product_test_close_trigger_state = FRAME_CLOSE

    host.on_serial_full_frame_received(_payload(FRAME_CLOSE))
    host.on_serial_full_frame_received(_payload(FRAME_CLOSE))

    assert host.started == []
    assert host.discarded_groups == []
    assert sum(
        1
        for level, message in host.default_logger.messages
        if level == "info" and "serial_product_close_ignored_no_active_round" in message
    ) == 2


def test_any_frame_from_current_product_can_start_a_new_round():
    host = _SerialProductHost()

    host.on_serial_full_frame_received(_payload(FRAME_7000))

    assert host.started == [FRAME_7000]
    assert host._manual_product_condition_group_id == "round-1"
    assert host._active_product_condition_key == FRAME_7000
    assert host.left_panel.condition_results[-1] == (FRAME_7000, "采集中", "running")
    assert host.reset_count == 0


def test_different_configured_frame_after_round_completion_starts_a_new_group():
    host = _SerialProductHost()
    for frame in (FRAME_7000, FRAME_6000, FRAME_8000):
        host.on_serial_full_frame_received(_payload(frame))
        host.complete_current("OK")

    assert host._manual_product_condition_group_id == ""

    host.on_serial_full_frame_received(_payload(FRAME_6000))

    assert host._manual_product_condition_group_id == "round-2"
    assert host.started[-1] == FRAME_6000


def test_repeated_last_frame_does_not_start_a_new_group_or_clear_waveforms():
    host = _SerialProductHost()
    for frame in (FRAME_7000, FRAME_6000, FRAME_8000):
        host.on_serial_full_frame_received(_payload(frame))
        host.complete_current("OK")

    assert host._manual_product_condition_group_id == ""
    assert host.cleared_waveforms == 1

    host.on_serial_full_frame_received(_payload(FRAME_8000))

    assert host._manual_product_condition_group_id == ""
    assert host.cleared_waveforms == 1
    assert host.started == [FRAME_7000, FRAME_6000, FRAME_8000]


def test_periodic_8000_after_two_condition_round_keeps_6000_round_state():
    host = _SerialProductHost()
    host.product_test_condition_configs = [
        host.product_test_condition_configs[0],
        host.product_test_condition_configs[2],
    ]

    host.on_serial_full_frame_received(_payload(FRAME_6000))
    first_group_id = host._manual_product_condition_group_id
    host.complete_current("OK")
    host.on_serial_trigger_status_changed(
        {
            "connected": True,
            "running": True,
            "has_response": True,
            "mode": "full_frame",
            "raw_hex": FRAME_IDLE,
        }
    )
    host.on_serial_full_frame_received(_payload(FRAME_8000))
    host.complete_current("OK")

    host.on_serial_full_frame_received(_payload(FRAME_8000))

    assert first_group_id == "round-1"
    assert host._manual_product_condition_group_id == ""
    assert host.started == [FRAME_6000, FRAME_8000]
    assert host.cleared_waveforms == 1


def test_unconfigured_transport_state_releases_last_frame_for_a_new_round():
    host = _SerialProductHost()
    for frame in (FRAME_7000, FRAME_6000, FRAME_8000):
        host.on_serial_full_frame_received(_payload(frame))
        host.complete_current("OK")

    host.on_serial_trigger_status_changed(
        {
            "connected": True,
            "running": True,
            "has_response": True,
            "mode": "full_frame",
            "raw_hex": FRAME_IDLE,
        }
    )
    host.on_serial_full_frame_received(_payload(FRAME_8000))

    assert host._manual_product_condition_group_id == "round-2"
    assert host.started[-1] == FRAME_8000
    assert host.cleared_waveforms == 2


def test_duplicate_current_frame_is_ignored_while_recording():
    host = _SerialProductHost()
    host.on_serial_full_frame_received(_payload(FRAME_6000))

    host.on_serial_full_frame_received(_payload(FRAME_6000))

    assert host.started == [FRAME_6000]
    assert host.reset_count == 0


def test_serial_frame_does_not_abort_an_unrelated_manual_recording():
    host = _SerialProductHost()
    host._record_workflow_busy = True

    host.on_serial_full_frame_received(_payload(FRAME_7000))

    assert host.reset_count == 0
    assert host.cleanup_count == 0
    assert host.started == []


def test_completed_frame_does_not_repeat_after_advancing(monkeypatch):
    host = _SerialProductHost()
    host.on_serial_full_frame_received(_payload(FRAME_6000))
    host.complete_current("OK")

    host.on_serial_full_frame_received(_payload(FRAME_6000))

    assert host.started == [FRAME_6000]
    assert FRAME_6000 in host._manual_product_condition_completed_keys


def test_frame_outside_current_product_is_logged_and_ignored():
    host = _SerialProductHost()
    unknown_frame = "FE 02 01 09 11 22"

    host.on_serial_full_frame_received(_payload(unknown_frame))

    assert host.started == []
    assert host._manual_product_condition_group_id == ""
    assert any(
        level == "info" and "serial_product_frame_unconfigured" in message
        for level, message in host.default_logger.messages
    )


def test_other_condition_frame_ignored_during_recording_can_trigger_after_completion(
    monkeypatch,
):
    warnings = []
    monkeypatch.setattr(
        "ui.sequence.sequence_widget_serial_trigger_ops.QMessageBox.warning",
        lambda *_args: warnings.append(_args[-1]),
    )
    host = _SerialProductHost()
    host.on_serial_full_frame_received(_payload(FRAME_6000))

    host.on_serial_full_frame_received(_payload(FRAME_6000))
    host.on_serial_full_frame_received(_payload(FRAME_7000))

    assert host.reset_count == 0
    assert host.cleanup_count == 0
    assert host._manual_product_condition_index == 0
    assert host.started == [FRAME_6000]
    assert host._serial_product_latched_frame == FRAME_6000
    assert warnings == []
    assert any(
        level == "info" and "serial_product_duplicate_ignored" in message
        for level, message in host.default_logger.messages
    )
    assert any(
        level == "info" and "serial_product_other_condition_ignored" in message
        for level, message in host.default_logger.messages
    )

    host.complete_current("OK")
    host.on_serial_full_frame_received(_payload(FRAME_7000))

    assert host.started == [FRAME_6000, FRAME_7000]
    assert host._serial_product_latched_frame == FRAME_7000
    assert warnings == []


def test_idle_unfinished_round_can_start_untested_condition_despite_stale_latch():
    host = _SerialProductHost()
    host._manual_product_condition_group_id = "round-1"
    host._manual_product_condition_completed_keys = {FRAME_6000}
    host._manual_product_condition_index = 1
    host._serial_product_latched_frame = FRAME_7000

    host.on_serial_full_frame_received(_payload(FRAME_7000))

    assert host.started == [FRAME_7000]
    assert host._manual_product_condition_index == 1
    assert host._serial_product_condition_executing


def test_abort_deletes_the_whole_round_and_restores_idle_ui(monkeypatch):
    monkeypatch.setattr(
        "ui.sequence.sequence_widget_serial_trigger_ops.QMessageBox.warning",
        lambda *_args: None,
    )
    host = _SerialProductHost()
    host.on_serial_full_frame_received(_payload(FRAME_6000))
    host._serial_product_pending_close_frame = FRAME_CLOSE

    host._abort_serial_product_round("录音异常")

    assert host.discarded_groups == [("round-1", True)]
    assert host.clicked_player_flag is False
    assert host.player_status_flag is False
    assert host._record_workflow_busy is False
    assert host.pause_update_count == 1
    assert host.unlock_count == 1
    assert host._serial_product_pending_close_frame == ""
    assert host.data_btn.disabled is True
    assert host.replayer_btn.disabled is True


def test_config_dialog_open_ignores_serial_product_frames():
    host = _SerialProductHost()
    host._product_test_program_config_dialog_open = True

    host.on_serial_full_frame_received(_payload(FRAME_6000))

    assert host.started == []
    assert host.reset_count == 0


def test_serial_trigger_does_not_require_ok_ng_to_complete_condition():
    host = _SerialProductHost()
    host.on_serial_full_frame_received(_payload(FRAME_6000))

    host.complete_current("")

    assert host.reset_count == 0
    assert host.discarded_groups == []
    assert host._manual_product_condition_completed_keys == {FRAME_6000}
    assert host._manual_product_condition_index == 1
    assert not host._serial_product_condition_executing
    assert any(
        level == "info"
        and "serial_product_condition_finalize" in message
        and f"condition={FRAME_6000}" in message
        and "condition_results={}" in message
        for level, message in host.default_logger.messages
    )


def test_missing_runtime_entry_shows_specific_warning_without_round_cleanup(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        "ui.sequence.sequence_widget_serial_trigger_ops.QMessageBox.warning",
        lambda *_args: warnings.append((_args[-2], _args[-1])),
    )
    host = _SerialProductHost()
    host._prepare_next_manual_product_condition_recording = None

    host.on_serial_full_frame_received(_payload(FRAME_6000))

    assert host.reset_count == 0
    assert host.discarded_groups == []
    assert warnings == [
        (
            "产品测试无法开始",
            "产品工况运行入口不可用，请检查程序版本或重新打开测试页面。",
        )
    ]


def test_recording_workflow_start_failure_shows_reason_before_round_cleanup(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        "ui.sequence.sequence_widget_serial_trigger_ops.QMessageBox.warning",
        lambda *_args: warnings.append(_args[-1]),
    )
    host = _SerialProductHost()
    host.start_this_play = lambda _label: None

    host.on_serial_full_frame_received(_payload(FRAME_6000))

    assert host.reset_count == 1
    assert host.discarded_groups == [("round-1", True)]
    assert warnings == [
        f"录音流程未能启动\n\n{host.SERIAL_PRODUCT_ERROR_MESSAGE}"
    ]


def test_condition_without_judgement_configuration_advances_normally():
    host = _SerialProductHost()
    host._can_output_ok_ng = lambda: (False, "当前配置未启用阈值对比，无法产出OK/NG")
    host.on_serial_full_frame_received(_payload(FRAME_6000))

    host.complete_current("")

    assert host.reset_count == 0
    assert FRAME_6000 in host._manual_product_condition_completed_keys
    assert host.started == [FRAME_6000]


def test_condition_index_alone_does_not_make_a_round_active():
    host = _SerialProductHost()
    host._manual_product_condition_index = 2

    host.on_serial_trigger_status_changed(
        {
            "connected": False,
            "running": False,
            "error": "port disconnected",
            "message": "port disconnected",
        }
    )

    assert host.reset_count == 0
    assert host.discarded_groups == []


def test_connection_failure_aborts_an_active_round(monkeypatch):
    monkeypatch.setattr(
        "ui.sequence.sequence_widget_serial_trigger_ops.QMessageBox.warning",
        lambda *_args: None,
    )
    host = _SerialProductHost()
    host.on_serial_full_frame_received(_payload(FRAME_7000))

    host.on_serial_trigger_status_changed(
        {
            "connected": False,
            "running": False,
            "error": "port disconnected",
            "message": "port disconnected",
        }
    )

    assert host.reset_count == 1
    assert host.cleanup_count == 1
    assert host.discarded_groups == [("round-1", True)]
    assert host._manual_product_condition_group_id == ""


def test_invalid_full_frame_configuration_is_rejected():
    host = _SerialProductHost()
    host.product_test_condition_configs[0]["trigger_state"] = "01"

    try:
        host._serial_full_frame_candidates()
    except ValueError as error:
        assert "至少需要 2 个字节" in str(error)
    else:
        raise AssertionError("single-byte trigger state should be rejected")


def test_close_frame_is_included_in_match_candidates_and_cannot_duplicate_condition():
    host = _SerialProductHost()
    host.product_test_close_trigger_state = FRAME_CLOSE

    assert host._serial_full_frame_candidates() == (
        FRAME_6000,
        FRAME_7000,
        FRAME_8000,
        FRAME_CLOSE,
    )

    host.product_test_close_trigger_state = FRAME_6000
    try:
        host._serial_full_frame_candidates()
    except ValueError as error:
        assert "报文重复" in str(error)
    else:
        raise AssertionError("close frame must not duplicate a condition frame")


def test_serial_trigger_rejects_import_audio_queue(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        "ui.sequence.sequence_widget_serial_trigger_ops.QMessageBox.warning",
        lambda *_args: warnings.append(_args[-1]),
    )
    host = _SerialProductHost()
    host._is_import_audio_mode = lambda: True

    host.on_serial_full_frame_received(_payload(FRAME_6000))

    assert host.started == []
    assert host.reset_count == 0
    assert host.discarded_groups == []
    assert host._manual_product_condition_group_id == ""
    assert host._manual_product_condition_index == 0
    assert warnings == [
        "当前工况绑定了 IMPORT_AUDIO 测试队列，串口触发不支持导入音频，请更换为录音测试队列。"
    ]


def test_error_dialog_suppresses_reentrant_frames_and_duplicate_warning(monkeypatch):
    host = _SerialProductHost()
    warnings = []

    def show_warning(*_args):
        warnings.append(_args[-1])
        host.on_serial_full_frame_received(_payload(FRAME_6000))
        host._abort_serial_product_round("弹窗期间的重复异常")

    monkeypatch.setattr(
        "ui.sequence.sequence_widget_serial_trigger_ops.QMessageBox.warning",
        show_warning,
    )

    host._abort_serial_product_round("首次异常")

    assert warnings == [f"首次异常\n\n{host.SERIAL_PRODUCT_ERROR_MESSAGE}"]
    assert host.started == []
    assert host.reset_count == 1

    host.on_serial_full_frame_received(_payload(FRAME_6000))
    assert host.started == [FRAME_6000]
