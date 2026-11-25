import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from base.hardware_trigger.serial_full_frame_matcher import SerialFullFrameMatcher
from base.hardware_trigger.serial_product_port_plan import build_serial_product_port_plan
from base.load_config import LoadUiConfig
from base.product_test_config_refresh import ProductTestConfigurationError, build_product_test_refresh_snapshot
from ui.sequence import sequence_widget_serial_trigger_ops as serial_ops
from unit_test.base.test_product_test_config_refresh import make_refresh_manager
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
        self._serial_round_received_after_ns = time.perf_counter_ns()
        self._refresh_serial_product_port_state()

    def feed(self, frame):
        self.on_serial_full_frame_received(dict(_payload(frame), received_perf_ns=time.perf_counter_ns()))


@pytest.mark.parametrize("entry", ["product", "queue", "nested", "reset_cancel", "reset_confirm"])
def test_only_fresh_serial_trigger_after_dialog_closes(qt_app, entry):
    from PyQt5.QtCore import QObject, Qt, pyqtSignal
    from ui.sequence.sequence_widget_round_reset_ops import SequenceWidgetRoundResetOpsMixin
    from unit_test.ui.test_product_test_program_runtime_refresh import _load_main_window_method

    class Wire(QObject):
        frame = pyqtSignal(object)

    host = PortHost((A,))
    wire = Wire()
    wire.frame.connect(host.on_serial_full_frame_received, Qt.QueuedConnection)

    def send():
        wire.frame.emit(dict(_payload(A), received_perf_ns=time.perf_counter_ns()))

    if entry.startswith("reset"):
        host._round_reset_in_progress = False
        host._round_reset_group_id = "previous"
        host._round_reset_busy = lambda: False
        host._round_data_records = {}
        host._round_analysis_records = {}
        host.toolsbar = Mock()
        host._reset_round_presentation = Mock()
        host._refresh_round_reset_button = Mock()

        def confirm(_group):
            send()
            return False if entry == "reset_confirm" else None

        host._confirm_round_reset = confirm
        SequenceWidgetRoundResetOpsMixin._on_reset_current_round(host)
    else:
        window = SimpleNamespace(sequence_window=host, mic=None, mic_channels=[])
        host.on_sequence_config_updated = Mock()
        host.on_product_test_program_updated = Mock()
        queue = Mock()
        queue.exec.side_effect = send
        open_queue = _load_main_window_method(
            "_open_analysis_model_select", {"AnalysisModelSelect": Mock(return_value=queue)},
        )
        window._open_analysis_model_select = lambda path: open_queue(window, path)

        def close_product():
            if entry == "nested":
                window._open_analysis_model_select("queue.json")
                qt_app.processEvents()
                assert host.started == []  # The product editor still owns admission.
            send()

        if entry == "queue":
            window._open_analysis_model_select("queue.json")
        else:
            product = Mock()
            product.exec.side_effect = close_product
            open_product = _load_main_window_method(
                "on_product_test_program_config", {"ProductTestProjectConfigDialog": Mock(return_value=product)},
            )
            open_product(window)

    qt_app.processEvents()
    assert host.started == []
    send()
    qt_app.processEvents()
    assert host.started == ["group_1:condition_1"]


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


def test_waiting_for_port_idle_displays_waiting_for_next_gear(qt_app):
    from ui.sequence.motor_result_panel import MotorResultPanel

    host = PortHost((A,), (A,), idle=IDLE)
    panel = MotorResultPanel()
    host.left_panel.set_current_stage = panel.set_current_stage
    host.feed(A)
    host.complete_current()
    assert host._serial_product_waiting_port_idle
    assert panel.stage_label.text() == "等待下一档位"


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
def test_next_port_accepts_fresh_frames_while_analysis_is_pending(idle):
    host = PortHost((A,), (B,), idle=idle)
    host.feed(A)
    host.complete_current(pending=True)
    host.feed(B)
    if idle:
        assert len(host.started) == 1
        host.feed(IDLE)
        host.feed(B)
    assert len(host.started) == 2
    assert host.analysis_pending


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
    with pytest.raises(ValueError, match="端口1的末档与端口2的首档"):
        build_serial_product_port_plan(conditions_for((A, B), (B, C)))
    with pytest.raises(ValueError, match="本轮末档与下一轮首档状态码相同"):
        build_serial_product_port_plan(conditions_for((A, B), (A,)))
    build_serial_product_port_plan(conditions_for((A, B), (B, C)), IDLE)


@pytest.mark.parametrize("ports,idle,allowed", [
    (((A,),), "", True),
    (((A,),), IDLE, True),
    (((A, B), (A,)), "", False),
    (((A, B), (A,)), IDLE, True),
    (((A, B), (C,)), "", True),
    (((A.lower().replace(" ", ""), B), (A,)), "", False),
    ((("", ""), ("",)), "", True),
])
def test_save_allows_drafts_but_use_validates_round_boundaries(
    tmp_path, monkeypatch, ports, idle, allowed,
):
    serial_path = tmp_path / "serial.json"
    monkeypatch.setattr(LoadUiConfig, "get_serial_discrete_input_config_path", lambda: str(serial_path))
    assert LoadUiConfig.save_serial_discrete_input_config({"enabled": True, "port_switch_idle_code": idle})
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
    assert ok
    validation = manager.validate_project(project, message)
    assert validation["can_save"]
    assert validation["is_usable"] == allowed
    if not allowed:
        assert any(
            "本轮末档与下一轮首档状态码相同" in error
            for error in validation["use_errors"]
        )
    assert manager.import_project(str(Path(manager.program_dir, message)))[0]


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


def test_out_of_order_gear_cannot_replace_idle_between_rounds():
    host = PortHost((A, B), (A,), idle=IDLE)
    host.feed(A)
    host.complete_current()
    host.feed(B)
    host.complete_current()
    host.feed(IDLE)
    host.feed(A)
    host.complete_current()
    first_round = [item["key"] for item in host.product_test_condition_configs]
    assert host.started == first_round

    for frame in (A, B, A):
        host.feed(frame)
        assert host.started == first_round

    host.feed(IDLE)
    assert host.started == first_round
    host.feed(A)
    assert host.started == first_round + [first_round[0]]


def test_single_condition_without_idle_restarts_only_after_round_completion():
    host = PortHost((A,))
    assert host._serial_full_frame_candidates() == (A,)
    for round_index in range(3):
        host.feed(A)
        assert len(host.started) == round_index + 1
        group_id = host._manual_product_condition_group_id
        for _ in range(3):
            host.feed(A)
        assert len(host.started) == round_index + 1
        assert host._manual_product_condition_group_id == group_id
        host.complete_current()
        assert not host._manual_product_condition_group_id


@pytest.mark.parametrize("pending_analysis", [False, True])
@pytest.mark.parametrize("boundary", ["gear", "port", "single-round", "multi-round"])
@pytest.mark.parametrize("idle", ["", IDLE], ids=["no-idle", "with-idle"])
def test_delayed_frames_cannot_start_next_recording(
    qt_app, monkeypatch, pending_analysis, boundary, idle,
):
    from PyQt5.QtCore import QObject, Qt, pyqtSignal
    from ui.sequence import sequence_widget_streaming_ops as streaming
    from ui.sequence.sequence_widget_analysis_process_ops import SequenceWidgetAnalysisProcessOpsMixin as Analysis

    class Wire(QObject):
        frame = pyqtSignal(object)

    ports = {"gear": ((A, B),), "port": ((A,), (B,)),
             "single-round": ((A,),), "multi-round": ((A, B),)}[boundary]
    host = PortHost(*ports, idle=idle)
    wire = Wire()
    wire.frame.connect(host.on_serial_full_frame_received, Qt.QueuedConnection)

    def send(frames):
        received = time.perf_counter_ns()  # Frames from one read have the same timestamp.
        for frame in frames:
            wire.frame.emit(dict(_payload(frame), received_perf_ns=received))

    host.feed(A)
    if boundary == "multi-round":
        host.complete_current()
        host.feed(B)
    previous = list(host.started)
    is_round = boundary.endswith("round")
    first = A if is_round else B
    needs_idle = idle and boundary in ("port", "single-round")
    frames = [IDLE, first] if needs_idle else [first]
    send(frames)
    host.recorded_signal_info = {}
    host.data_btn, host.replayer_btn = Mock(), Mock()
    host._recording_process_direction = ""
    host._schedule_raw_audio_csv_export = Mock()
    host._cache_condition_record = Mock()
    host._update_current_recent_session_result = Mock()
    host.barcode_scanner_box = SimpleNamespace(isChecked=lambda: False)
    host._is_manual_product_condition_cycle_active = lambda: True
    host._mark_manual_product_condition_recording_completed = lambda: (
        host._manual_product_condition_completed_keys.add(host._active_product_condition_key)
    )
    host.analysis_pending = pending_analysis
    host._lock_analysis_round_config = host._unlock_analysis_round_config = Mock()
    host._defer_round_completion_for_analysis = Analysis._defer_round_completion_for_analysis.__get__(host)
    database = Mock()
    database.save_signal_info_to_db.return_value = (0, "")
    monkeypatch.setattr(streaming, "RecordingManager", lambda: database)
    assert streaming.SequenceWidgetStreamingOpsMixin._complete_saved_recording_business(
        host, 48000, (0,), None, True, "test", lambda: None,
    )
    if pending_analysis and is_round:
        assert host._analysis_round_completion_pending
        send(frames)
        host.analysis_pending = False
        assert Analysis._finish_deferred_analysis_round_if_ready(host)
    qt_app.processEvents()
    assert host.started == previous
    if not is_round:
        assert host.analysis_pending == pending_analysis
    if needs_idle:
        send([first])  # An old idle frame must not have unlocked this start.
        qt_app.processEvents()
        assert host.started == previous
    send(frames)
    qt_app.processEvents()
    expected = {"gear": "group_1:condition_2", "port": "group_2:condition_1"}
    assert host.started == previous + [expected.get(boundary, "group_1:condition_1")]


@pytest.mark.parametrize("busy_flag", [
    "_record_workflow_busy", "_analysis_round_completion_pending",
])
def test_single_condition_without_idle_still_obeys_workflow_busy(busy_flag):
    host = PortHost((A,))
    host.feed(A)
    host.complete_current()
    setattr(host, busy_flag, True)
    host.feed(A)
    assert len(host.started) == 1
    setattr(host, busy_flag, False)
    host.feed(A)
    assert len(host.started) == 2


def test_single_condition_with_idle_still_requires_idle_between_rounds():
    host = PortHost((A,), idle=IDLE)
    host.feed(A)
    host.complete_current()
    host.feed(A)
    assert len(host.started) == 1
    host.feed(IDLE)
    host.feed(A)
    assert len(host.started) == 2


@pytest.mark.parametrize("idle", [A, "XX ZZ", 123])
def test_invalid_idle_is_rejected(idle):
    with pytest.raises(ValueError):
        build_serial_product_port_plan(conditions_for((A, B)), idle)


def test_product_save_reads_idle_from_serial_file(tmp_path, monkeypatch):
    path = tmp_path / "serial.json"
    monkeypatch.setattr(LoadUiConfig, "get_serial_discrete_input_config_path", staticmethod(lambda: str(path)))
    assert LoadUiConfig.save_serial_discrete_input_config({"enabled": True})
    manager = make_manager(tmp_path)
    project = make_project(tmp_path)
    for port in project["test_groups"]:
        port["test_conditions"][0]["trigger_state"] = A
    ok, filename = manager.save_project(None, project)
    assert ok
    assert not manager.validate_project(project, filename)["is_usable"]
    assert LoadUiConfig.save_serial_discrete_input_config({"port_switch_idle_code": IDLE})
    assert manager.save_project(filename, project)[0]
    assert manager.validate_project(project, filename)["is_usable"]
    assert LoadUiConfig.save_serial_discrete_input_config({"serial_settings": {"baudrate": 19200}})
    assert LoadUiConfig.load_serial_discrete_input_config()[1]["port_switch_idle_code"] == IDLE


def test_runtime_rechecks_configuration_before_listening():
    host = PortHost((A,), (A,), idle=IDLE)
    with pytest.raises(ValueError, match="末档"):
        host._serial_full_frame_candidates({"port_switch_idle_code": ""})
    assert host._serial_full_frame_candidates({"port_switch_idle_code": IDLE}) == (A, IDLE)


def test_manual_project_with_shared_port_code_can_start_and_rejects_serial_switch(tmp_path, monkeypatch):
    monkeypatch.setattr(LoadUiConfig, "get_serial_discrete_input_config_path",
                        lambda: str(tmp_path / "serial.json"))
    assert LoadUiConfig.save_serial_discrete_input_config({"enabled": False})
    manager, project, _ = make_refresh_manager(tmp_path, condition_count=1, port_count=2)
    for group in project["test_groups"]:
        group["test_conditions"][0]["trigger_state"] = A
    ok, filename = manager.save_project("A.json", project)
    assert ok
    snapshot = build_product_test_refresh_snapshot(manager, filename)
    host = PortHost((A,), (A,))
    host.product_test_condition_configs = snapshot.conditions
    host._serial_trigger_config = LoadUiConfig.load_serial_discrete_input_config()[1]
    click_play = _load_analysis_method("on_clicked_player_btn")
    click_play(host)
    first, second = snapshot.conditions
    assert host.started == [first["key"]]
    host.complete_current()
    host._sync_product_progress_after_trigger_switch()
    assert host._serial_product_port_index == 1

    host.hw_manager = Mock()
    dialog = Mock()
    dialog.exec.return_value = ("save", {"enabled": True})
    monkeypatch.setattr(serial_ops, "SerialDiscreteInputConfigDialog", lambda *args, **kwargs: dialog)
    warning = Mock()
    monkeypatch.setattr(serial_ops.QMessageBox, "warning", warning)
    host.on_serial_trigger_btn_clicked()
    message = warning.call_args.args[-1]
    assert first["group_name"] in message and second["group_name"] in message
    assert "功能 → 产品测试程序配置" in message and "空闲码" not in message
    assert not LoadUiConfig.load_serial_discrete_input_config()[1]["enabled"]
    assert not host._serial_trigger_config["enabled"]
    host.hw_manager.start_serial_discrete_input_listener.assert_not_called()
    click_play(host)
    assert host.started == [first["key"], second["key"]]

    assert LoadUiConfig.save_serial_discrete_input_config({"enabled": True})
    with pytest.raises(ProductTestConfigurationError, match="核对并修改状态码"):
        build_product_test_refresh_snapshot(manager, filename)
    assert LoadUiConfig.save_serial_discrete_input_config({"port_switch_idle_code": IDLE})
    assert build_product_test_refresh_snapshot(manager, filename).conditions == snapshot.conditions


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
