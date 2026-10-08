"""Gate-driven startup recovery tests: all native work stays on its owner."""
import threading
import json

import pytest

from base.ve3668n_resource import VeResourceController
from base.ve_startup_policy import VeStartupBudget
from base.vkinging_sdk import VkDaqError
from unit_test.base.ve3668n_fakes import CaptureSDK, capture_request


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class Recovery:
    def __init__(self, tmp_path, blocked="create_task", *, failures=()):
        self.clock = Clock()
        self.entered = threading.Event()
        self.gate = threading.Event()
        self.owners = []
        self.proofs = []
        self.blocks = []
        self.failures = []
        self.fatals = []
        self.result = []
        self.first = CaptureSDK(hooks={} if blocked == "sdk_open" else {
            blocked: self.block}, failures=failures)
        self.second = CaptureSDK()
        self.controller = VeResourceController(
            sdk_factory=self.factory, clock=self.clock, generation=7,
            fatal=lambda *args: self.fatals.append(args))
        self.blocked = blocked
        self.budget = VeStartupBudget.create(100, 5, 1, 1)
        self.adapter = self.controller.stream(
            request=capture_request(tmp_path / "retry.wav"),
            callback=lambda *args: self.blocks.append(args),
            fail=lambda *args: self.failures.append(args),
            stop_event=threading.Event(), startup_budget=self.budget,
            on_retry=self.retry)
        self.thread = threading.Thread(target=lambda: self.result.append(self.adapter.start()))

    def block(self, *args, **kwargs):
        self.entered.set()
        assert self.gate.wait(2), "test failed to release native gate"

    def factory(self):
        self.owners.append(threading.current_thread())
        if len(self.owners) == 1:
            if self.blocked == "sdk_open":
                self.block()
            return self.first
        assert len(self.owners) == 2
        assert not self.owners[0].is_alive()
        assert self.proofs
        return self.second

    def retry(self, proof):
        assert not self.owners or not self.owners[0].is_alive()
        self.proofs.append(proof)

    def revoke(self):
        self.thread.start()
        assert self.entered.wait(1)
        self.clock.now = 101.1
        assert self.controller._startup_attempt.revoked.wait(1)
        assert not self.adapter.started.is_set()
        assert not self.blocks and not self.failures and not self.fatals

    def finish(self):
        self.gate.set()
        if self.thread.ident is not None:
            self.thread.join(2)
            assert not self.thread.is_alive()
        self.adapter.stop_event.set()
        self.controller.close(.5)
        for owner in self.owners:
            owner.join(1)
            assert not owner.is_alive()


@pytest.mark.parametrize("blocked", [
    "sdk_open", "get_devices", "get_device_attribute", "get_channels",
    "create_task", "create_iepe_voltage_channel", "configure_sample_clock",
    "start_task", "verify_actual_sample_rate",
])
def test_late_return_cleans_on_old_owner_before_one_retry(tmp_path, blocked):
    case = Recovery(tmp_path, blocked)
    try:
        case.revoke()
        case.gate.set()
        case.thread.join(1)
        assert case.result == [True]
        assert case.adapter.completed.wait(1)
        assert len(case.owners) == 2
        assert len(case.proofs) == 1
        assert not case.failures and not case.fatals
        assert case.first.closed
        assert case.first.calls("read_task_data") == 0
        for sdk, owner in zip((case.first, case.second), case.owners):
            assert sdk.owner_thread_ids == {owner.ident}
        proof = case.proofs[0]
        assert proof.old_task_id != proof.new_task_id
        assert proof.startup_budget is case.budget
        assert proof.generation == 7
        assert case.adapter.started_at == 101.1
        operations = case.first.operations()
        prefix = () if blocked == "sdk_open" else operations[:operations.index(blocked) + 1]
        cleanup = ("stop_task", "clear_task", "close") if case.first.calls("create_task") else ("close",)
        assert operations == prefix + cleanup
    finally:
        case.finish()


@pytest.mark.parametrize("operation", ["stop_task", "clear_task", "close"])
def test_cleanup_failure_prevents_retry(tmp_path, operation):
    case = Recovery(tmp_path, failures=(operation,))
    try:
        case.revoke()
        case.gate.set()
        case.thread.join(1)
        assert case.result == [False]
        assert len(case.owners) == 1 and not case.proofs
        assert case.failures[0][0] == "startup_cleanup_failed"
        assert operation in case.failures[0][1]
        assert len(case.fatals) == 1
        case.owners[0].join(1)
        assert not case.adapter.handles_released
    finally:
        case.finish()


@pytest.mark.parametrize("operation", ["native_return", "stop_task", "clear_task", "close"])
def test_cleanup_window_includes_native_return_and_all_cleanup(tmp_path, operation):
    case = Recovery(tmp_path)
    entered, gate = threading.Event(), threading.Event()

    def blocked_cleanup(*args):
        entered.set()
        assert gate.wait(2)

    if operation != "native_return":
        case.first.hooks[operation] = blocked_cleanup
    try:
        case.revoke()
        if operation != "native_return":
            case.gate.set()
            assert entered.wait(1)
        case.clock.now = 102
        case.thread.join(1)
        assert case.result == [False]
        assert case.failures[0][0] == "startup_cleanup_timeout"
        assert not case.proofs and len(case.owners) == 1
        gate.set()
        case.gate.set()
        case.owners[0].join(1)
        assert not case.proofs and len(case.owners) == 1
        assert not case.adapter.started.is_set()
    finally:
        gate.set()
        case.finish()


def test_cleanup_failure_is_reported_even_if_later_cleanup_blocks(tmp_path):
    case = Recovery(tmp_path, failures=("stop_task",))
    entered, gate = threading.Event(), threading.Event()

    def blocked_clear(*args):
        entered.set()
        assert gate.wait(2)

    case.first.hooks["clear_task"] = blocked_clear
    try:
        case.revoke()
        case.gate.set()
        assert entered.wait(1)
        case.thread.join(.2)
        assert case.result == [False]
        assert case.failures[0][0] == "startup_cleanup_failed"
        assert not case.proofs
    finally:
        gate.set()
        case.finish()


@pytest.mark.parametrize("when", ["before_start", "first_native", "cleanup", "proof", "second_native"])
def test_cancellation_never_starts_a_later_attempt(tmp_path, when):
    case = Recovery(tmp_path)
    entered, gate = threading.Event(), threading.Event()

    def block(*args):
        entered.set()
        assert gate.wait(2)

    if when == "cleanup":
        case.first.hooks["clear_task"] = block
    if when == "second_native":
        case.second.hooks["create_task"] = block
    if when == "proof":
        original = case.adapter.on_retry
        def cancel(proof):
            original(proof)
            case.adapter.stop_event.set()
        case.adapter.on_retry = cancel
    try:
        if when == "before_start":
            case.adapter.stop_event.set()
            case.thread.start()
        elif when == "first_native":
            case.thread.start()
            assert case.entered.wait(1)
            case.adapter.stop_event.set()
        else:
            case.revoke()
            case.gate.set()
            if when != "proof":
                assert entered.wait(1)
                case.adapter.stop_event.set()
        case.thread.join(.5)
        assert case.result == [False]
        assert not case.adapter.started.is_set()
        assert not case.blocks
        assert len(case.owners) == (0 if when == "before_start" else 2 if when == "second_native" else 1)
        if when == "proof":
            assert case.adapter.stop_event.is_set()
            assert case.adapter.handles_released
            assert not case.controller.failed
            next_adapter = case.controller.stream(
                request=capture_request(tmp_path / "after-proof-cancel.wav"),
                callback=lambda *args: None,
                fail=lambda *args: case.failures.append(args), stop_event=threading.Event(),
                startup_budget=VeStartupBudget.create(case.clock(), 5, 1, 1))
            assert next_adapter.start()
            assert next_adapter.completed.wait(1)
            assert case.adapter.stop_event.is_set()
            assert len(case.owners) == 2 and len(case.proofs) == 1
            assert not case.failures and not case.fatals
    finally:
        gate.set()
        case.finish()


@pytest.mark.parametrize("operation", ["sdk_open", "create_task", "start_task", "verify_actual_sample_rate"])
def test_first_native_error_before_deadline_keeps_original_cause(tmp_path, operation):
    case = Recovery(tmp_path, operation)
    if operation == "sdk_open":
        def fail_factory():
            case.owners.append(threading.current_thread())
            raise VkDaqError("LoadLibrary", -17, "loader refused")
        case.controller._sdk_factory = fail_factory
    else:
        case.first.failures.add(operation)
    try:
        case.gate.set()
        case.thread.start()
        case.thread.join(1)
        assert case.result == [False]
        assert case.adapter.failure_snapshot.stage == operation
        assert case.adapter.failure_snapshot.code == -17
        assert not case.proofs and len(case.owners) == 1
        assert case.adapter.completed.wait(1)
    finally:
        case.finish()


def test_second_failure_keeps_native_cause_with_terminal_retry_stage(tmp_path):
    case = Recovery(tmp_path)
    case.second.failures.add("start_task")
    try:
        case.revoke()
        case.gate.set()
        case.thread.join(1)
        assert case.result == [False]
        assert case.adapter.failure_snapshot.stage == "startup_retry_failed"
        assert case.adapter.failure_snapshot.code == -17
        assert "start_task" in case.adapter.failure_snapshot.detail
        assert "first initialization timed out" in case.failures[0][1]
        assert len(case.owners) == 2 and len(case.proofs) == 1
    finally:
        case.finish()


def test_second_timeout_uses_total_deadline_and_never_retries_again(tmp_path):
    case = Recovery(tmp_path)
    entered, gate = threading.Event(), threading.Event()

    def block(*args):
        entered.set()
        assert gate.wait(2)

    case.second.hooks["create_task"] = block
    try:
        case.revoke()
        case.gate.set()
        assert entered.wait(1)
        case.clock.now = 105
        case.thread.join(1)
        assert case.result == [False]
        assert case.failures[0][0] == "startup_deadline"
        assert len(case.owners) == 2
    finally:
        gate.set()
        case.finish()


def test_retry_proof_delivery_failure_prevents_second_sdk(tmp_path):
    case = Recovery(tmp_path)
    def fail_delivery(proof):
        raise OSError("control queue closed")
    case.adapter.on_retry = fail_delivery
    try:
        case.revoke()
        case.gate.set()
        case.thread.join(1)
        assert case.result == [False]
        assert case.failures[0][0] == "startup_retry_failed"
        assert "control queue closed" in case.failures[0][1]
        assert len(case.owners) == 1
    finally:
        case.finish()


@pytest.mark.parametrize("now,expected", [(101, True), (101.5, True), (102, False), (105, False)])
def test_queued_expiry_only_retries_inside_cleanup_window(tmp_path, now, expected):
    case = Recovery(tmp_path)
    case.clock.now = now
    case.gate.set()
    # The first *logical* attempt has no SDK; factory sees only replacement.
    def replacement():
        case.owners.append(threading.current_thread())
        assert case.proofs
        return case.second
    case.controller._sdk_factory = replacement
    try:
        case.thread.start()
        case.thread.join(1)
        assert case.result == [expected]
        assert len(case.owners) == int(expected)
        if expected:
            proof = case.proofs[0]
            assert proof.before_counts == proof.after_cleanup_counts
        else:
            assert not case.proofs
    finally:
        case.finish()


def test_owner_exit_must_be_confirmed_before_cleanup_deadline(tmp_path):
    case = Recovery(tmp_path)
    exited_cleanup, exit_gate = threading.Event(), threading.Event()
    original = case.controller._run_startup

    def delay_exit(attempt):
        original(attempt)
        if attempt.number == 1:
            exited_cleanup.set()
            assert exit_gate.wait(2)

    case.controller._run_startup = delay_exit
    try:
        case.revoke()
        case.gate.set()
        assert exited_cleanup.wait(1)
        assert not case.proofs
        case.clock.now = 102
        case.thread.join(.5)
        assert case.result == [False]
        assert case.failures[0][0] == "startup_cleanup_timeout"
        assert len(case.owners) == 1
        assert not case.adapter.handles_released
    finally:
        exit_gate.set()
        case.finish()


@pytest.mark.parametrize("when", ["first_native", "cleanup", "proof", "second_native"])
def test_close_during_startup_prevents_late_bind_and_new_owner(tmp_path, when):
    case = Recovery(tmp_path)
    entered, gate = threading.Event(), threading.Event()
    def block(*args):
        entered.set()
        assert gate.wait(2)
    if when == "cleanup":
        case.first.hooks["clear_task"] = block
    if when == "second_native":
        case.second.hooks["create_task"] = block
    if when == "proof":
        original = case.adapter.on_retry
        def close(proof):
            original(proof)
            case.controller.close(.01)
        case.adapter.on_retry = close
    try:
        if when == "first_native":
            case.thread.start()
            assert case.entered.wait(1)
        else:
            case.revoke()
            case.gate.set()
            if when != "proof":
                assert entered.wait(1)
        if when != "proof":
            case.controller.close(.01)
        case.thread.join(.5)
        assert case.result == [False]
        assert not case.adapter.started.is_set()
        assert not case.blocks
        assert len(case.owners) == (2 if when == "second_native" else 1)
    finally:
        gate.set()
        case.finish()


def test_commit_rechecks_deadline_before_publishing_started(tmp_path):
    case = Recovery(tmp_path)
    original = case.adapter._accept_bind
    calls = []
    def delayed_accept(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            case.clock.now = 101
        return original(*args, **kwargs)
    case.adapter._accept_bind = delayed_accept
    case.gate.set()
    try:
        case.thread.start()
        case.thread.join(1)
        assert case.result == [True]
        assert len(case.owners) == 2
        assert case.first.calls("read_task_data") == 0
        assert len(case.proofs) == 1
    finally:
        case.finish()


def test_second_start_timestamp_is_sampled_before_rate_verification(tmp_path):
    case = Recovery(tmp_path, "start_task")
    case.second.hooks["start_task"] = lambda *args: setattr(case.clock, "now", 101.3)
    case.second.hooks["verify_actual_sample_rate"] = lambda *args: setattr(case.clock, "now", 101.8)
    try:
        case.revoke()
        case.gate.set()
        case.thread.join(1)
        assert case.result == [True]
        assert case.adapter.started_at == 101.3
        assert case.first.calls("verify_actual_sample_rate") == 0
    finally:
        case.finish()


def test_diagnostics_distinguish_retry_from_late_first_attempt(tmp_path):
    case = Recovery(tmp_path)
    entries = []
    class Logger:
        def log(self, level, message, payload):
            entries.append(json.loads(payload))
    case.controller._resource_diagnostics.logger = Logger()
    try:
        case.revoke()
        case.gate.set()
        case.thread.join(1)
        assert case.result == [True]
        first = [entry for entry in entries if entry.get("attempt") == 1]
        second = [entry for entry in entries if entry.get("attempt") == 2]
        assert any(entry["event"] == "DISCARD" and entry["after_timeout"] for entry in first)
        assert second and all(not entry["after_timeout"] for entry in second)
        assert all(entry["startup_budget"]["deadline"] == 105 for entry in second)
        proof = next(entry for entry in entries if entry["phase"] == "startup_cleanup" and entry["event"] == "END")
        assert proof["old_task_id"] != proof["new_task_id"]
        assert proof["after_cleanup_counts"]["sdk_close"] == 1
    finally:
        case.finish()


@pytest.mark.parametrize("phase", ["sdk_open", "get_devices", "create_task", "start_task", "verify_actual_sample_rate"])
def test_revocation_during_begin_diagnostic_skips_native_call(tmp_path, phase):
    case = Recovery(tmp_path)
    entered, gate = threading.Event(), threading.Event()
    class Logger:
        def log(self, level, message, payload):
            entry = json.loads(payload)
            if entry.get("attempt") == 1 and entry["phase"] == phase and entry["event"] == "BEGIN":
                entered.set()
                assert gate.wait(2)
    case.controller._resource_diagnostics.logger = Logger()
    case.gate.set()
    # Construction can be skipped before the factory sees the first attempt.
    def factory():
        owner = threading.current_thread()
        case.owners.append(owner)
        return case.second if case.proofs else case.first
    case.controller._sdk_factory = factory
    case.adapter.on_retry = case.proofs.append
    try:
        case.thread.start()
        assert entered.wait(1)
        case.clock.now = 101.1
        assert case.controller._startup_attempt.revoked.wait(1)
        gate.set()
        case.thread.join(1)
        assert case.result == [True]
        assert len(case.proofs) == 1
        assert case.first.calls(phase) == 0
        assert case.controller._resource_diagnostics._phases == []
        if phase == "sdk_open":
            assert len(case.owners) == 1
            assert case.proofs[0].before_counts == case.proofs[0].after_cleanup_counts
    finally:
        gate.set()
        case.finish()


def test_retry_callbacks_are_outside_controller_and_adapter_locks(tmp_path):
    case = Recovery(tmp_path)
    original = case.adapter.on_retry
    def retry(proof):
        assert not case.controller._lock._is_owned()
        assert not case.adapter._lock._is_owned()
        original(proof)
    case.adapter.on_retry = retry
    try:
        case.revoke()
        case.gate.set()
        case.thread.join(1)
        assert case.result == [True]
    finally:
        case.finish()


def test_late_native_error_preserves_call_counts_and_retries_once(tmp_path):
    case = Recovery(tmp_path, failures=("create_task",))
    try:
        case.revoke()
        case.gate.set()
        case.thread.join(1)
        assert case.result == [True]
        counts = case.proofs[0].after_cleanup_counts
        assert (counts.sdk_open, counts.task_create, counts.task_start,
                counts.task_stop, counts.task_clear, counts.sdk_close) == (1, 1, 0, 0, 0, 1)
        assert any("-17" in item for item in case.controller._diagnostics)
    finally:
        case.finish()


def test_first_success_and_healthy_reuse_do_not_retry_expired_budget(tmp_path):
    case = Recovery(tmp_path)
    case.gate.set()
    try:
        case.thread.start()
        case.thread.join(1)
        assert case.result == [True]
        assert case.adapter.completed.wait(1)
        case.clock.now = 106
        adapter = case.controller.stream(
            request=capture_request(tmp_path / "reuse.wav"), callback=lambda *args: None,
            fail=lambda *args: case.failures.append(args), stop_event=threading.Event(),
            startup_budget=case.budget, on_retry=case.proofs.append)
        assert adapter.start()
        assert adapter.completed.wait(1)
        assert len(case.owners) == 1 and not case.proofs and not case.failures
    finally:
        case.finish()


def test_another_bind_cannot_enter_replacement_command_queue(tmp_path):
    case = Recovery(tmp_path)
    original = case.adapter.on_retry
    def retry(proof):
        original(proof)
        other = case.controller.stream(
            request=capture_request(tmp_path / "other.wav"), callback=lambda *args: None,
            fail=lambda *args: None, stop_event=threading.Event())
        with pytest.raises(RuntimeError, match="initialization already in progress"):
            other.start()
    case.adapter.on_retry = retry
    try:
        case.revoke()
        case.gate.set()
        case.thread.join(1)
        assert case.result == [True]
        assert len(case.owners) == 2
    finally:
        case.finish()


@pytest.mark.parametrize("operation,stage", [("create_task", "create_task"), ("get_devices", "device")])
def test_native_error_before_deadline_is_not_reclassified_by_late_diagnostics(tmp_path, operation, stage):
    case = Recovery(tmp_path, failures=(operation,))
    entered, gate = threading.Event(), threading.Event()
    class Logger:
        def log(self, level, message, payload):
            entry = json.loads(payload)
            if entry["phase"] == operation and entry["event"] == "ERROR":
                entered.set()
                assert gate.wait(2)
    case.controller._resource_diagnostics.logger = Logger()
    case.gate.set()
    try:
        case.thread.start()
        assert entered.wait(1)
        case.clock.now = 101.1
        gate.set()
        case.thread.join(1)
        assert case.result == [False]
        assert case.adapter.failure_snapshot.stage == stage
        assert case.adapter.failure_snapshot.code == -17
        assert operation in case.failures[0][1]
        assert not case.proofs
    finally:
        gate.set()
        case.finish()


def test_cancelled_cleanup_does_not_confirm_ownership_before_owner_exit(tmp_path):
    case = Recovery(tmp_path)
    exited_cleanup, exit_gate = threading.Event(), threading.Event()
    original = case.controller._run_startup
    def delay_exit(attempt):
        original(attempt)
        exited_cleanup.set()
        assert exit_gate.wait(2)
    case.controller._run_startup = delay_exit
    try:
        case.thread.start()
        assert case.entered.wait(1)
        case.adapter.stop_event.set()
        case.thread.join(1)
        case.gate.set()
        assert exited_cleanup.wait(1)
        assert not case.adapter.handles_released
        exit_gate.set()
        case.owners[0].join(1)
        assert case.adapter.handles_released
    finally:
        exit_gate.set()
        case.finish()


def test_confirmed_cancelled_owner_does_not_poison_next_request(tmp_path):
    case = Recovery(tmp_path)
    try:
        case.thread.start()
        assert case.entered.wait(1)
        case.adapter.stop_event.set()
        case.thread.join(1)
        case.gate.set()
        case.owners[0].join(1)
        assert case.adapter.handles_released
        assert not case.controller.failed
        def factory():
            case.owners.append(threading.current_thread())
            return case.second
        case.controller._sdk_factory = factory
        adapter = case.controller.stream(
            request=capture_request(tmp_path / "next.wav"), callback=lambda *args: None,
            fail=lambda *args: case.failures.append(args), stop_event=threading.Event(),
            startup_budget=case.budget)
        assert adapter.start()
        assert adapter.completed.wait(1)
        assert not case.proofs and not case.failures
    finally:
        case.finish()


@pytest.mark.parametrize("attempt", [1, 2])
@pytest.mark.parametrize("blocked_close", [False, True])
def test_cancelled_owner_reports_cleanup_error_without_waiter(tmp_path, attempt, blocked_close):
    case = Recovery(tmp_path)
    native_entered, native_gate = threading.Event(), threading.Event()
    close_entered, close_gate = threading.Event(), threading.Event()
    fatal = threading.Event()
    def block_native(*args):
        native_entered.set()
        assert native_gate.wait(2)
    def block_close():
        close_entered.set()
        assert close_gate.wait(2)
    def report_fatal(*args):
        assert not case.controller._lock._is_owned()
        assert not case.adapter._lock._is_owned()
        case.fatals.append(args)
        fatal.set()
    case.controller._fatal_callback = report_fatal
    sdk = case.first if attempt == 1 else case.second
    sdk.failures.add("clear_task")
    if attempt == 2:
        sdk.hooks["create_task"] = block_native
    if blocked_close:
        sdk.hooks["close"] = block_close
    try:
        if attempt == 1:
            case.thread.start()
            assert case.entered.wait(1)
        else:
            case.revoke()
            case.gate.set()
            assert native_entered.wait(1)
        case.adapter.stop_event.set()
        case.thread.join(1)
        assert case.result == [False]
        assert not case.fatals
        case.gate.set()
        native_gate.set()
        if blocked_close:
            assert close_entered.wait(1)
        assert fatal.wait(.2)
        assert case.controller.failed
        assert case.controller.failure_snapshot.stage == "startup_cleanup_failed"
        assert case.controller.failure_snapshot.code == -17
        assert "clear_task" in case.fatals[0][1]
        assert not case.adapter.handles_released
        assert not case.adapter.started.is_set()
        assert len(case.proofs) == attempt - 1
        assert len(case.owners) == attempt
    finally:
        native_gate.set()
        close_gate.set()
        case.finish()


def test_optional_discovery_error_still_allows_first_attempt(tmp_path):
    case = Recovery(tmp_path)
    def optional_error(name, attribute):
        if attribute == "DeviceStatus":
            raise VkDaqError("get_device_attribute", -17, "optional status unsupported")
    case.first.hooks["get_device_attribute"] = optional_error
    case.gate.set()
    try:
        case.thread.start()
        case.thread.join(1)
        assert case.result == [True]
        assert not case.failures and not case.fatals and not case.proofs
    finally:
        case.finish()


def test_discovery_rejects_one_bad_candidate_without_failing_usable_device(tmp_path):
    case = Recovery(tmp_path)
    original_devices = case.first.get_devices
    original_attribute = case.first.get_device_attribute
    case.first.get_devices = lambda: (("192.0.2.2", "BrokenDev"), *original_devices())
    def attribute(name, key):
        if name == "BrokenDev":
            raise VkDaqError("get_device_attribute", -17, "candidate unavailable")
        return original_attribute(name, key)
    case.first.get_device_attribute = attribute
    case.gate.set()
    try:
        case.thread.start()
        case.thread.join(1)
        assert case.result == [True]
        assert not case.failures and not case.fatals and not case.proofs
    finally:
        case.finish()


def test_cleanup_error_preserves_preceding_native_failure(tmp_path):
    case = Recovery(tmp_path, failures=("start_task", "clear_task"))
    case.gate.set()
    try:
        case.thread.start()
        case.thread.join(1)
        case.owners[0].join(1)
        assert case.result == [False]
        assert case.controller.failure_snapshot.stage == "start_task"
        assert case.adapter.failure_snapshot.stage == "start_task"
        assert case.controller.failure_snapshot.code == -17
        assert len(case.fatals) == 1 and case.fatals[0][0] == "start_task"
        assert not case.adapter.handles_released
    finally:
        case.finish()


@pytest.mark.parametrize("failure,stage,detail", [
    ("empty_devices", "device", "unavailable"),
    ("wrong_identity", "device", "unavailable"),
    ("unavailable_channels", "device", "selected channels unavailable"),
    ("invalid_rate", "verify_actual_sample_rate", "actual sample rate 48000"),
])
def test_decisive_validation_failure_precedes_delayed_diagnostics(tmp_path, failure, stage, detail):
    case = Recovery(tmp_path)
    entered, gate = threading.Event(), threading.Event()
    class Logger:
        def log(self, level, message, payload):
            entry = json.loads(payload)
            if entry["phase"] == stage and entry["event"] == "ERROR":
                entered.set()
                assert gate.wait(2)
    case.controller._resource_diagnostics.logger = Logger()
    if failure == "empty_devices":
        case.first.get_devices = lambda: ()
    elif failure == "wrong_identity":
        case.first.record["MachineId"] = "other-machine"
    elif failure == "unavailable_channels":
        case.first.record["channels"] = ("FreshDev/AIN4",)
    else:
        case.first.verify_actual_sample_rate = lambda *args: 48000
    case.gate.set()
    try:
        case.thread.start()
        assert entered.wait(1)
        case.clock.now = 101.1
        gate.set()
        case.thread.join(1)
        assert case.result == [False]
        assert not case.adapter.started.is_set()
        assert case.adapter.failure_snapshot.stage == stage
        assert case.adapter.failure_snapshot.code is None
        assert detail in case.adapter.failure_snapshot.detail
        assert len(case.failures) == len(case.fatals) == 1
        assert case.fatals[0][0] == stage
        assert len(case.owners) == 1 and not case.proofs
    finally:
        gate.set()
        case.finish()


@pytest.mark.parametrize("interruption,stage", [("close", "close"), ("deadline", "startup_deadline")])
def test_proof_callback_terminal_interruption_keeps_controller_failed(tmp_path, interruption, stage):
    case = Recovery(tmp_path)
    original = case.adapter.on_retry
    def interrupt(proof):
        original(proof)
        if interruption == "close":
            case.controller.close(.01)
        else:
            case.clock.now = case.budget.deadline
    case.adapter.on_retry = interrupt
    try:
        case.revoke()
        case.gate.set()
        case.thread.join(1)
        assert case.result == [False]
        assert case.controller.failed
        assert case.controller.failure_snapshot.stage == stage
        assert case.adapter.handles_released
        assert len(case.owners) == 1 and len(case.proofs) == 1
        with pytest.raises(RuntimeError, match="permanently failed"):
            case.controller.stream(
                request=capture_request(tmp_path / "after-terminal.wav"),
                callback=lambda *args: None, fail=lambda *args: None,
                stop_event=threading.Event(),
                startup_budget=VeStartupBudget.create(case.clock(), 5, 1, 1))
    finally:
        case.finish()


@pytest.mark.parametrize("operation", ["stop_task", "clear_task", "close"])
@pytest.mark.parametrize("cancelled", [False, True])
def test_cleanup_error_precedes_delayed_diagnostics_and_deadline(tmp_path, operation, cancelled):
    case = Recovery(tmp_path, failures=(operation,))
    entered, gate = threading.Event(), threading.Event()
    entries = []
    class Logger:
        def log(self, level, message, payload):
            entry = json.loads(payload)
            entries.append(entry)
            if entry["phase"] == operation and entry["event"] == "ERROR":
                entered.set()
                assert gate.wait(2)
    case.controller._resource_diagnostics.logger = Logger()
    try:
        case.revoke()
        if cancelled:
            case.adapter.stop_event.set()
            case.thread.join(1)
            assert case.result == [False]
        case.gate.set()
        assert entered.wait(1)
        case.clock.now = case.budget.cleanup_deadline
        case.thread.join(1)
        assert case.result == [False]
        assert case.controller.failed
        assert case.controller.failure_snapshot.stage == "startup_cleanup_failed"
        assert case.controller.failure_snapshot.code == -17
        assert operation in case.controller.failure_snapshot.detail
        assert len(case.fatals) == len(case.failures) == 1
        assert case.fatals[0][0] == "startup_cleanup_failed"
        assert not any(entry["phase"] == "startup_cleanup_timeout" for entry in entries)
        assert not case.adapter.handles_released
        assert len(case.owners) == 1 and not case.proofs
    finally:
        gate.set()
        case.finish()


def test_delayed_cleanup_error_keeps_preceding_native_failure(tmp_path):
    case = Recovery(tmp_path, failures=("start_task", "clear_task"))
    entered, gate = threading.Event(), threading.Event()
    class Logger:
        def log(self, level, message, payload):
            entry = json.loads(payload)
            if entry["phase"] == "clear_task" and entry["event"] == "ERROR":
                entered.set()
                assert gate.wait(2)
    case.controller._resource_diagnostics.logger = Logger()
    case.gate.set()
    try:
        case.thread.start()
        assert entered.wait(1)
        case.clock.now = case.budget.cleanup_deadline
        case.thread.join(1)
        assert case.result == [False]
        assert case.controller.failure_snapshot.stage == "start_task"
        assert case.controller.failure_snapshot.code == -17
        assert len(case.fatals) == 1 and case.fatals[0][0] == "start_task"
        assert not case.adapter.handles_released
    finally:
        gate.set()
        case.finish()


@pytest.mark.parametrize("operation,next_operation", [("stop_task", "clear_task"), ("clear_task", "close")])
def test_cleanup_continues_on_owner_after_delayed_error_log(tmp_path, operation, next_operation):
    case = Recovery(tmp_path, failures=(operation,))
    log_entered, log_gate = threading.Event(), threading.Event()
    native_entered, native_gate = threading.Event(), threading.Event()
    class Logger:
        def log(self, level, message, payload):
            entry = json.loads(payload)
            if entry["phase"] == operation and entry["event"] == "ERROR":
                log_entered.set()
                assert log_gate.wait(2)
    def block_next(*args):
        native_entered.set()
        assert native_gate.wait(2)
    case.controller._resource_diagnostics.logger = Logger()
    case.first.hooks[next_operation] = block_next
    try:
        case.revoke()
        case.gate.set()
        assert log_entered.wait(1)
        case.clock.now = case.budget.cleanup_deadline
        case.thread.join(1)
        assert case.result == [False]
        log_gate.set()
        assert native_entered.wait(1)
        assert case.controller.failure_snapshot.stage == "startup_cleanup_failed"
        assert case.controller.failure_snapshot.code == -17
        assert operation in case.controller.failure_snapshot.detail
        assert case.first.owner_thread_ids == {case.owners[0].ident}
        assert len(case.failures) == len(case.fatals) == 1
        assert not case.adapter.handles_released and not case.proofs
    finally:
        log_gate.set()
        native_gate.set()
        case.finish()


@pytest.mark.parametrize("stale_snapshot", [False, True])
def test_cancelled_release_confirmation_survives_waiter_and_poll_races(tmp_path, stale_snapshot):
    case = Recovery(tmp_path)
    lookup_entered, lookup_gate = threading.Event(), threading.Event()
    original = case.controller._startup_handles_released
    poll_result = []
    delayed_thread = None
    def delayed_lookup(adapter):
        if threading.current_thread() is delayed_thread:
            released = original(adapter) if stale_snapshot else None
            lookup_entered.set()
            assert lookup_gate.wait(2)
            return released if stale_snapshot else original(adapter)
        return original(adapter)
    case.controller._startup_handles_released = delayed_lookup
    poll = None
    try:
        case.thread.start()
        assert case.entered.wait(1)
        if stale_snapshot:
            case.adapter.stop_event.set()
            case.thread.join(1)
            assert case.result == [False]
            poll = threading.Thread(target=lambda: poll_result.append(case.adapter.handles_released))
            delayed_thread = poll
            poll.start()
        else:
            delayed_thread = case.thread
            case.adapter.stop_event.set()
        assert lookup_entered.wait(1)
        case.gate.set()
        case.owners[0].join(1)
        assert not case.owners[0].is_alive()
        assert case.adapter.handles_released
        assert case.controller._startup_attempt is None
        lookup_gate.set()
        case.thread.join(1)
        if poll is not None:
            poll.join(1)
            assert not poll.is_alive()
            assert poll_result == [True]
        assert case.result == [False]
        assert case.adapter.handles_released
        assert not case.controller.failed
    finally:
        lookup_gate.set()
        case.finish()
        if poll is not None:
            poll.join(1)


def test_first_attempt_cleanup_is_not_terminal_adapter_release(tmp_path):
    case = Recovery(tmp_path)
    entered, gate = threading.Event(), threading.Event()
    original = case.adapter.on_retry
    def retry(proof):
        original(proof)
        assert not case.adapter.handles_released
    def block_second(*args):
        entered.set()
        assert gate.wait(2)
    case.adapter.on_retry = retry
    case.second.hooks["create_task"] = block_second
    case.second.failures.add("clear_task")
    try:
        case.revoke()
        case.gate.set()
        assert entered.wait(1)
        assert not case.adapter.handles_released
        case.adapter.stop_event.set()
        case.thread.join(1)
        assert case.result == [False]
        assert not case.adapter.handles_released
        gate.set()
        case.owners[1].join(1)
        assert case.controller.failed
        assert not case.adapter.handles_released
    finally:
        gate.set()
        case.finish()


@pytest.mark.parametrize("attempt", [1, 2])
@pytest.mark.parametrize("stale_poll", [False, True])
def test_release_polling_during_valid_initialization_does_not_change_binding(tmp_path, attempt, stale_poll):
    case = Recovery(tmp_path)
    native_entered, native_gate = threading.Event(), threading.Event()
    lookup_entered, lookup_gate = threading.Event(), threading.Event()
    poll = None
    original = case.controller._startup_handles_released
    def block_second(*args):
        native_entered.set()
        assert native_gate.wait(2)
    def delayed_lookup(adapter):
        released = original(adapter)
        if threading.current_thread() is poll:
            lookup_entered.set()
            assert lookup_gate.wait(2)
        return released
    if attempt == 2:
        case.second.hooks["create_task"] = block_second
    case.controller._startup_handles_released = delayed_lookup
    try:
        if attempt == 1:
            case.thread.start()
            assert case.entered.wait(1)
        else:
            case.revoke()
            case.gate.set()
            assert native_entered.wait(1)
        if stale_poll:
            poll = threading.Thread(target=lambda: case.adapter.handles_released)
            poll.start()
            assert lookup_entered.wait(1)
        else:
            for _ in range(3):
                assert not case.adapter.handles_released
        case.gate.set()
        native_gate.set()
        case.thread.join(1)
        assert case.result == [True]
        assert case.adapter.completed.wait(1)
        lookup_gate.set()
        if poll is not None:
            poll.join(1)
            assert not poll.is_alive()
        assert case.adapter.handles_released
        assert len(case.owners) == attempt and len(case.proofs) == attempt - 1
        assert not case.failures and not case.fatals
    finally:
        lookup_gate.set()
        native_gate.set()
        case.finish()
        if poll is not None:
            poll.join(1)


def test_declined_bind_is_not_reclassified_as_timeout_retry(tmp_path):
    case = Recovery(tmp_path)
    case.adapter._accept_bind = lambda *args: False
    case.gate.set()
    try:
        case.thread.start()
        case.thread.join(1)
        assert case.result == [False]
        case.owners[0].join(1)
        assert case.adapter.handles_released
        assert len(case.owners) == 1 and not case.proofs
        assert not case.adapter.started.is_set()
    finally:
        case.finish()


@pytest.mark.parametrize("method", ["release", "close"])
@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_cancelled_startup_release_entry_reconciles_without_status_poll(tmp_path, method, cleanup_failure):
    case = Recovery(tmp_path, failures=("clear_task",) if cleanup_failure else ())
    try:
        case.thread.start()
        assert case.entered.wait(1)
        case.adapter.stop_event.set()
        case.thread.join(1)
        assert case.result == [False]
        case.gate.set()
        case.owners[0].join(1)
        assert not case.owners[0].is_alive()
        # Do not poll adapter.handles_released: the lifecycle method must
        # reconcile the completed attempt itself.
        outcome = getattr(case.controller, method)(.1)
        assert outcome.success is not cleanup_failure
        assert case.controller.failed is cleanup_failure
        assert len(case.fatals) == int(cleanup_failure)
        if cleanup_failure:
            assert case.controller.failure_snapshot.stage == "startup_cleanup_failed"
        else:
            assert case.controller._startup_attempt is None
            assert case.controller._owner is None
        assert case.first.closed
    finally:
        case.finish()


@pytest.mark.parametrize("method", ["release", "close"])
def test_cancelled_startup_release_entry_requires_actual_owner_exit(tmp_path, method):
    case = Recovery(tmp_path)
    cleanup_finished, exit_gate = threading.Event(), threading.Event()
    original = case.controller._run_startup
    def delayed_exit(attempt):
        original(attempt)
        cleanup_finished.set()
        assert exit_gate.wait(2)
    case.controller._run_startup = delayed_exit
    try:
        case.thread.start()
        assert case.entered.wait(1)
        case.adapter.stop_event.set()
        case.thread.join(1)
        assert case.result == [False]
        case.gate.set()
        assert cleanup_finished.wait(1)
        assert case.owners[0].is_alive()
        outcome = getattr(case.controller, method)(.02)
        assert not outcome.success
        assert case.controller._owner is case.owners[0]
        assert case.controller.failed is (method == "close")
        assert len(case.fatals) == int(method == "close")
    finally:
        exit_gate.set()
        case.finish()


def test_release_during_retry_proof_gap_cannot_report_empty_success(tmp_path):
    case = Recovery(tmp_path)
    original = case.adapter.on_retry
    release_outcomes = []
    observations = []
    def retry(proof):
        original(proof)
        observations.append((case.controller._owner, case.controller._state))
        release_outcomes.append(case.controller.release(.1))
    case.adapter.on_retry = retry
    try:
        case.revoke()
        case.gate.set()
        case.thread.join(1)
        assert observations == [(None, case.controller._STARTING)]
        assert len(release_outcomes) == 1 and not release_outcomes[0].success
        assert "initialization in progress" in release_outcomes[0].diagnostics[0]
        assert case.result == [True]
        assert case.adapter.completed.wait(1)
        assert len(case.owners) == 2 and len(case.proofs) == 1
        assert case.second.calls("create_task") == 1
        assert not case.failures and not case.fatals
    finally:
        case.finish()


@pytest.mark.parametrize("phase", ["first", "cleanup", "proof", "second", "cancelled_live"])
def test_public_lifecycle_entrypoints_respect_pending_startup(tmp_path, phase):
    case = Recovery(tmp_path)
    entered, gate = threading.Event(), threading.Event()
    def new_adapter(name):
        return case.controller.stream(
            request=capture_request(tmp_path / (name + ".wav")),
            callback=lambda *args: None, fail=lambda *args: None,
            stop_event=threading.Event(), startup_budget=case.budget)
    precreated = new_adapter("precreated")
    def block(*args):
        entered.set()
        assert gate.wait(2)
    if phase == "cleanup":
        case.first.hooks["clear_task"] = block
    elif phase == "proof":
        original = case.adapter.on_retry
        def retry(proof):
            original(proof)
            block()
        case.adapter.on_retry = retry
    elif phase == "second":
        case.second.hooks["create_task"] = block
    try:
        if phase in ("first", "cancelled_live"):
            case.thread.start()
            assert case.entered.wait(1)
            if phase == "cancelled_live":
                case.adapter.stop_event.set()
                case.thread.join(1)
                assert case.result == [False]
        else:
            case.revoke()
            case.gate.set()
            assert entered.wait(1)
        assert not case.controller.release(.02).success
        assert not case.controller.failed
        assert not case.adapter.handles_released
        created_during_startup = new_adapter("pending")
        for adapter in (precreated, created_during_startup):
            with pytest.raises(RuntimeError, match="initialization already in progress"):
                adapter.start()
        assert len(case.owners) == (2 if phase == "second" else 1)
        assert not case.controller.close(.02).success
        assert case.controller.failed
        with pytest.raises(RuntimeError, match="permanently failed"):
            new_adapter("after-close")
        assert not case.adapter.started.is_set()
    finally:
        gate.set()
        case.finish()


@pytest.mark.parametrize("precreate", [False, True])
def test_adapter_and_bind_reconcile_cancelled_dead_owner_without_poll(tmp_path, precreate):
    case = Recovery(tmp_path)
    def make_adapter():
        return case.controller.stream(
            request=capture_request(tmp_path / "fresh-after-cancel.wav"),
            callback=lambda *args: None, fail=lambda *args: case.failures.append(args),
            stop_event=threading.Event(), startup_budget=case.budget)
    adapter = make_adapter() if precreate else None
    try:
        case.thread.start()
        assert case.entered.wait(1)
        case.adapter.stop_event.set()
        case.thread.join(1)
        case.gate.set()
        case.owners[0].join(1)
        assert case.result == [False]
        assert not case.owners[0].is_alive()
        def factory():
            assert not case.owners[0].is_alive()
            case.owners.append(threading.current_thread())
            return case.second
        case.controller._sdk_factory = factory
        if adapter is None:
            adapter = make_adapter()
        assert adapter.start()
        assert adapter.completed.wait(1)
        assert len(case.owners) == 2 and not case.proofs
        assert not case.failures and not case.fatals
        assert case.adapter.handles_released
    finally:
        case.finish()
