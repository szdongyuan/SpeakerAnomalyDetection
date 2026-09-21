from types import SimpleNamespace

import pytest

from base.hardware_trigger.serial_full_frame_matcher import SerialFullFrameMatcher
from base.hardware_trigger.serial_product_port_plan import build_serial_product_port_plan
from base.load_config import LoadUiConfig
from unit_test.test_product_test_project_config import make_project, make_manager
from unit_test.test_serial_product_condition_runtime import (
    _SerialProductHost,
    _load_analysis_method,
    _payload,
)


A = "A5 5A 01 01 0D 0A"
B = "A5 5A 01 02 0D 0A"
C = "A5 5A 01 03 0D 0A"
IDLE = "A5 5A 00 00 0D 0A"


def conditions_for(*ports):
    return [
        {
            "group_name": f"端口{port_index}",
            "key": f"group_{port_index}:condition_{gear_index}",
            "condition_name": f"档位{gear_index}",
            "trigger_state": frame,
            "test_queue": f"queue-{port_index}-{gear_index}",
        }
        for port_index, frames in enumerate(ports, 1)
        for gear_index, frame in enumerate(frames, 1)
    ]


class PortHost(_SerialProductHost):
    _advance_manual_product_condition_cycle_after_recording = _load_analysis_method(
        "_advance_manual_product_condition_cycle_after_recording"
    )

    def __init__(self, *ports, idle=""):
        super().__init__()
        self.product_test_condition_configs = conditions_for(*ports)
        self._serial_trigger_config = {"port_switch_idle_code": idle}
        self.analysis_pending = False

    def _product_condition_runtime_key(self, condition, index=None):
        return condition["key"]

    def _prepare_next_manual_product_condition_recording(self):
        super()._prepare_next_manual_product_condition_recording()
        self._active_product_condition_key = self._active_product_condition_config["key"]
        return True

    def _analysis_has_pending_tasks(self):
        return self.analysis_pending

    def _product_group_result_state(self, group_id):
        return False, ""

    def complete_current(self, *, pending=False):
        assert self._active_product_condition_key
        self.analysis_pending = pending
        self._manual_product_condition_completed_keys.add(self._active_product_condition_key)
        self.player_status_flag = False
        self._advance_manual_product_condition_cycle_after_recording()
        self._on_serial_product_condition_completed()
        self._record_workflow_busy = False

    def feed(self, frame):
        self.on_serial_full_frame_received(_payload(frame))


@pytest.mark.parametrize("idle", ["", IDLE])
def test_ordered_ports_reuse_codes_without_repeating_previous_last_gear(idle):
    host = PortHost((A, B), (A, B), idle=idle)
    host.feed(B)  # Port 1 starts with A, not whichever code arrives first.
    assert host.started == []
    for frame in (A, B):
        host.feed(frame)
        host.complete_current()
    if idle:
        host.feed(B)
        assert len(host.started) == 2
        host.feed(IDLE)
    assert host._serial_product_port_index == 1
    host.feed(B)
    assert len(host.started) == 2
    host.feed(A)
    host.complete_current()
    host.feed(B)
    host.complete_current()
    assert host.started == [item["key"] for item in host.product_test_condition_configs]
    assert host._manual_product_condition_group_id == ""


def test_idle_allows_same_code_at_port_boundary_and_does_not_skip_ports():
    host = PortHost((A,), (A,), (A,), idle=IDLE)
    host.feed(IDLE)
    assert getattr(host, "_serial_product_port_index", 0) == 0
    host.feed(A)
    host.feed(IDLE)  # Early idle while recording is not retained.
    host.complete_current()
    host.feed(A)
    assert len(host.started) == 1
    host.feed(IDLE)
    host.feed(IDLE)
    assert host._serial_product_port_index == 1
    host.feed(A)
    host.complete_current()
    host.feed(IDLE)
    host.feed(A)
    host.complete_current()
    assert host.started == [item["key"] for item in host.product_test_condition_configs]


@pytest.mark.parametrize("idle", ["", IDLE])
def test_both_modes_match_only_current_port_and_require_gear_order(idle):
    host = PortHost((A, B), (C,), idle=idle)
    host.feed(C)
    assert not host.started
    host.feed(B)
    assert not host.started
    host.feed(A)
    host.complete_current()
    host.feed(IDLE)  # B is not recorded yet; cannot move on.
    host.feed(C)
    assert len(host.started) == 1
    host.feed(B)
    host.complete_current()
    if idle:
        host.feed(C)
        assert len(host.started) == 2
        host.feed(IDLE)
    host.feed(C)
    assert host.loaded_queues == ["queue-1-1", "queue-1-2", "queue-2-1"]


@pytest.mark.parametrize("idle", ["", IDLE])
def test_port_waits_for_analysis_and_requires_idle_after_analysis(idle):
    host = PortHost((A,), (B,), idle=idle)
    host.feed(A)
    host.complete_current(pending=True)
    host.feed(IDLE)
    host.feed(B)
    assert len(host.started) == 1
    assert host._serial_product_port_index == 0
    host.analysis_pending = False
    host._refresh_serial_product_port_state()
    host.feed(B)
    if idle:
        assert len(host.started) == 1
        host.feed(IDLE)
        host.feed(B)
    assert len(host.started) == 2


@pytest.mark.parametrize("split", range(1, 6))
def test_fragmented_idle_and_sticky_next_gear_use_complete_frames(split):
    host = PortHost((A,), (A,), idle=IDLE)
    matcher = SerialFullFrameMatcher(host._serial_full_frame_candidates())
    host.feed(A)
    host.complete_current()
    idle_bytes = bytes.fromhex(IDLE)
    assert matcher.feed(idle_bytes[:split]) == []
    for frame in matcher.feed(idle_bytes[split:] + bytes.fromhex(A)):
        host.feed(frame.hex())
    assert len(host.started) == 2


def test_same_port_duplicates_rejected_but_cross_port_duplicates_deduplicated():
    with pytest.raises(ValueError, match="端口1内状态码重复"):
        build_serial_product_port_plan(conditions_for((A, A)), IDLE)
    plan = build_serial_product_port_plan(conditions_for((A, B), (A, B)))
    assert plan.candidates == (A, B)
    assert plan.ports == ((0, 1), (2, 3))


def test_ordered_validation_checks_port_and_round_boundaries():
    build_serial_product_port_plan(conditions_for((A, B), (A, C)))
    with pytest.raises(ValueError, match="端口1末档与端口2首档"):
        build_serial_product_port_plan(conditions_for((A, B), (B, C)))
    with pytest.raises(ValueError, match="本轮末档与下一轮首档状态码相同"):
        build_serial_product_port_plan(conditions_for((A, B), (A,)))
    build_serial_product_port_plan(conditions_for((A, B), (B, C)), IDLE)


@pytest.mark.parametrize("ports,idle,allowed", [
    (((A,),), "", False),
    (((A,),), IDLE, True),
    (((A, B), (A,)), "", False),
    (((A, B), (A,)), IDLE, True),
    (((A, B), (C,)), "", True),
    (((A.lower().replace(" ", ""), B), (A,)), "", False),
    ((("", ""), ("",)), "", True),
])
def test_save_validates_first_and_last_gear_across_rounds(
    tmp_path, monkeypatch, ports, idle, allowed,
):
    serial_path = tmp_path / "serial.json"
    monkeypatch.setattr(LoadUiConfig, "get_serial_discrete_input_config_path", lambda: str(serial_path))
    assert LoadUiConfig.save_serial_discrete_input_config({"port_switch_idle_code": idle})
    manager = make_manager(tmp_path)
    project = make_project(tmp_path)
    project["test_groups"] = [
        {"group_name": f"端口{port_index}", "test_conditions": [
            {"condition_name": f"档位{gear_index}", "trigger_state": frame, "test_queue": "基础测试"}
            for gear_index, frame in enumerate(frames, 1)
        ]}
        for port_index, frames in enumerate(ports, 1)
    ]
    ok, message = manager.save_project(None, project)
    assert ok == allowed
    if not allowed:
        assert "本轮末档与下一轮首档状态码相同" in message
        assert "空闲码" in message


@pytest.mark.parametrize("ports", [((A,),), ((A, B), (A,))])
def test_idle_allows_same_first_and_last_code_in_consecutive_rounds(ports):
    host = PortHost(*ports, idle=IDLE)
    for _ in range(2):
        for port_index, frames in enumerate(ports):
            if port_index:
                host.feed(IDLE)
            for frame in frames:
                host.feed(frame)
                host.complete_current()
        assert not host._manual_product_condition_group_id
        host.feed(IDLE)
    assert host.started == [item["key"] for item in host.product_test_condition_configs] * 2


@pytest.mark.parametrize("idle", [A, "XX ZZ", 123])
def test_invalid_idle_is_rejected(idle):
    with pytest.raises(ValueError):
        build_serial_product_port_plan(conditions_for((A, B)), idle)


def test_product_save_reads_idle_from_serial_file(tmp_path, monkeypatch):
    path = tmp_path / "serial.json"
    monkeypatch.setattr(LoadUiConfig, "get_serial_discrete_input_config_path", staticmethod(lambda: str(path)))
    manager = make_manager(tmp_path)
    project = make_project(tmp_path)
    for port in project["test_groups"]:
        port["test_conditions"][0]["trigger_state"] = A
    assert not manager.save_project(None, project)[0]
    assert LoadUiConfig.save_serial_discrete_input_config({"port_switch_idle_code": IDLE})
    assert manager.save_project(None, project)[0]
    assert LoadUiConfig.save_serial_discrete_input_config({"serial_settings": {"baudrate": 19200}})
    assert LoadUiConfig.load_serial_discrete_input_config()[1]["port_switch_idle_code"] == IDLE


def test_runtime_rechecks_configuration_before_listening():
    host = PortHost((A,), (A,), idle=IDLE)
    with pytest.raises(ValueError, match="末档"):
        host._serial_full_frame_candidates({"port_switch_idle_code": ""})
    assert host._serial_full_frame_candidates({"port_switch_idle_code": IDLE}) == (A, IDLE)


@pytest.mark.parametrize("idle", ["", IDLE])
def test_analysis_completion_callback_advances_or_waits_for_idle(idle):
    from ui.sequence.sequence_widget_analysis_process_ops import SequenceWidgetAnalysisProcessOpsMixin

    host = PortHost((A,), (B,), idle=idle)
    host.feed(A)
    host.complete_current(pending=True)
    host.analysis_pending = False
    assert SequenceWidgetAnalysisProcessOpsMixin._restore_waiting_stage_after_automatic_analysis(host)
    if idle:
        assert host._serial_product_waiting_port_idle
        assert "空闲码" in host.left_panel.stages[-1][0]
    else:
        assert host._serial_product_port_index == 1
        assert "端口2" in host.left_panel.stages[-1][0]


def test_actual_cycle_reset_clears_port_wait_and_held_frame():
    host = PortHost((A,), (A,), idle=IDLE)
    host.feed(A)
    host.complete_current()
    assert host._serial_product_waiting_port_idle
    host._reset_product_condition_display_state = lambda: None
    _load_analysis_method("_reset_manual_product_condition_cycle")(host)
    assert host._serial_product_port_index == 0
    assert not host._serial_product_waiting_port_idle
    assert host._serial_product_latched_frame == ""
    host.feed(A)
    assert host.started == ["group_1:condition_1", "group_1:condition_1"]


def test_different_port_gear_counts_and_new_round_start_at_first_port():
    host = PortHost((A, B), (C,), (A, B, C))
    for _round in range(2):
        for frame in (A, B, C, A, B, C):
            host.feed(frame)
            host.complete_current()
    assert host.started == [item["key"] for item in host.product_test_condition_configs] * 2


def test_refresh_reloads_hand_edited_idle_and_disabling_stops_listener(tmp_path, monkeypatch):
    path = tmp_path / "serial.json"
    monkeypatch.setattr(LoadUiConfig, "get_serial_discrete_input_config_path", staticmethod(lambda: str(path)))
    host = PortHost((A,), (A,))
    calls = []
    host.hw_manager = SimpleNamespace(
        start_serial_discrete_input_listener=lambda cfg, **kwargs: (calls.append(kwargs["full_frame_candidates"]) or {"ok": True}),
        stop_serial_discrete_input_listener=lambda: calls.append("stopped"),
    )
    LoadUiConfig.save_serial_discrete_input_config({"enabled": True, "port_switch_idle_code": IDLE})
    assert host.refresh_serial_product_trigger_runtime()["ok"]
    assert calls == [(A, IDLE)]
    LoadUiConfig.save_serial_discrete_input_config({"enabled": False})
    assert host.refresh_serial_product_trigger_runtime()["message"] == "disabled"
    assert calls[-1] == "stopped"


@pytest.mark.parametrize("split", range(1, 6))
def test_worker_raw_chunks_cannot_release_held_frame(monkeypatch, split):
    from base.hardware_trigger import serial_discrete_input_worker as worker_module
    from unit_test.test_serial_product_condition_runtime import _Logger

    host = PortHost((A,), idle=IDLE)
    host.feed(A)
    host.complete_current()
    frame = bytes.fromhex(A)
    chunks = [frame[:split], frame[split:] + frame[:split], frame[split:]]
    monkeypatch.setattr(worker_module.LogManager, "set_log_handler", staticmethod(lambda _name: _Logger()))
    worker = worker_module.SerialDiscreteInputWorker(
        {"serial_settings": {"port": "FAKE"}},
        full_frame_candidates=host._serial_full_frame_candidates(),
    )

    class Port:
        is_open = True

        @property
        def in_waiting(self):
            return len(chunks[0]) if chunks else 0

        def reset_input_buffer(self):
            pass

        def read(self, _size):
            chunk = chunks.pop(0)
            if not chunks:
                worker._is_running = False
            return chunk

        def close(self):
            self.is_open = False

    monkeypatch.setattr(worker_module, "serial", SimpleNamespace(Serial=lambda **_kwargs: Port()))
    worker.sig_state_changed.connect(host.on_serial_full_frame_received)
    worker.sig_status.connect(host._update_serial_product_latch_from_status)
    worker.run()
    assert host.started == ["group_1:condition_1"]
