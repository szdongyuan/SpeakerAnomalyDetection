"""Focused MainWindow coordination tests; no native VE or dialog I/O."""

from copy import deepcopy
import logging
from types import SimpleNamespace

import pytest

from base.recording_process_protocol import VeLifecycleCounts
from base.recording_service import VePrewarmCompletion
from base.ve3668n_prewarm_lifetime import VePrewarmLifetime
from main_window import MainWindow
from unit_test.base.ve3668n_fakes import device_info, input_config


class _StatusBar:
    def __init__(self):
        self.messages = []

    def showMessage(self, message, *_args):
        self.messages.append(message)


class _Bridge:
    def __init__(self, status="accepted"):
        self.status = status
        self.hardware_busy = False
        self.calls = []
        self.callback = None
        self.release_calls = []

    def prewarm_ve(self, request, callback):
        self.calls.append(request)
        self.callback = callback
        return self.status

    def release_ve(self, signature, callback):
        self.release_calls.append((signature, callback))
        return "pending"


class _SynchronousBridge(_Bridge):
    def prewarm_ve(self, request, callback):
        self.calls.append(request)
        self.callback = callback
        callback(_completion(request))
        return "accepted"


class _WindowHarness:
    _ve_signature = MainWindow._ve_signature
    _ve_prewarm_signature = MainWindow._ve_prewarm_signature
    _show_ve_prewarm_status = MainWindow._show_ve_prewarm_status
    _try_start_ve_prewarm = MainWindow._try_start_ve_prewarm
    _on_ve_prewarm_complete = MainWindow._on_ve_prewarm_complete
    _render_ve_prewarm_failure = MainWindow._render_ve_prewarm_failure
    _on_ve_hardware_release_complete = MainWindow._on_ve_hardware_release_complete
    _on_ve_discovery_result = MainWindow._on_ve_discovery_result
    _hardware_busy = MainWindow._hardware_busy
    _hardware_selection_admission_available = (
        MainWindow._hardware_selection_admission_available)
    on_hardware_window_init = MainWindow.on_hardware_window_init

    def _refresh_ve_admission_controls(self):
        """The focused harness has no real recording or hardware controls."""

    def __init__(self, *, bridge=None, lifetime=None):
        self.default_logger = logging.getLogger("core")
        self.ve_prewarm_lifetime = lifetime or VePrewarmLifetime()
        self.recording_bridge = bridge or _Bridge()
        self.sequence_window = SimpleNamespace(
            player_status_flag=False,
            mic=None,
            speaker=None,
            mic_channels=[],
            speaker_channels=[],
            update_v2pa_factor=lambda: None,
            refresh_channel_windows=lambda: None,
        )
        self.mic = None
        self.speaker = None
        self.mic_channels = []
        self.speaker_channels = []
        self.ve_discovery = SimpleNamespace(cancel=lambda: None, start=lambda: None)
        self.ve_profile_store = SimpleNamespace(
            load=lambda _device, _calibrations: _device.get("input_config") or input_config())
        self.ve_calibration_store = object()
        self.sequence_window.ve_profile_store = self.ve_profile_store
        self.sequence_window.ve_calibration_store = self.ve_calibration_store
        self.ve_discovery_factory = None
        self.hardware_selection_path = None
        self._status = _StatusBar()
        self.status_updates = 0

    def statusBar(self):
        return self._status

    def update_statusbar(self):
        self.status_updates += 1


def _completion(request, *, success=True, stage="completed", code=None,
                detail="", ownership_safe=True, diagnostics=()):
    return VePrewarmCompletion(
        request.warmup_id,
        request.signature,
        success,
        stage,
        code,
        detail,
        tuple(diagnostics),
        VeLifecycleCounts(1, 1, 1, 1, 1, 1),
        ownership_safe,
    )


def _discovery(*devices, released=True, diagnostics=()):
    return SimpleNamespace(
        result=SimpleNamespace(devices=tuple(devices), diagnostics=tuple(diagnostics)),
        handles_released=released,
    )


def test_startup_discovery_waits_for_valid_selection_and_never_dispatches_twice():
    window = _WindowHarness()
    window.mic = device_info(available=False, input_config=None)
    window.mic_channels = [7, 1]

    window._on_ve_discovery_result(_discovery(diagnostics=("not connected",)))
    assert window.ve_prewarm_lifetime.snapshot().state == "available"
    assert window.recording_bridge.calls == []

    window._on_ve_discovery_result(_discovery(device_info()))
    assert window.ve_prewarm_lifetime.snapshot().state == "pending"
    assert len(window.recording_bridge.calls) == 1

    window._on_ve_discovery_result(_discovery(device_info(name="duplicate")))
    assert len(window.recording_bridge.calls) == 1


def test_valid_startup_discovery_while_busy_is_discarded_immediately():
    bridge = _Bridge()
    bridge.hardware_busy = True
    window = _WindowHarness(bridge=bridge)
    window.mic = device_info(available=False, input_config=None)
    window.mic_channels = [7, 1]

    window._on_ve_discovery_result(_discovery(device_info()))

    assert window.mic["available"]
    assert window.ve_prewarm_lifetime.snapshot().state == "skipped_busy"
    assert bridge.calls == []
    bridge.hardware_busy = False
    assert window._try_start_ve_prewarm(window.mic, [7, 1], "later") == "consumed"


def test_trigger_validates_before_claim_and_only_first_valid_ve_is_dispatched():
    window = _WindowHarness()
    ordinary = {"backend": "sounddevice", "name": "mic"}
    unavailable = device_info(available=False, input_config=None)

    assert window._try_start_ve_prewarm(ordinary, [0], "startup") == "not_ve"
    assert window._try_start_ve_prewarm(unavailable, [7, 1], "startup") == "invalid"
    assert window.ve_prewarm_lifetime.snapshot().state == "available"

    selected = device_info()
    window.mic, window.mic_channels = selected, [7, 1]
    assert window._try_start_ve_prewarm(selected, [7, 1], "startup") == "accepted"
    assert len(window.recording_bridge.calls) == 1
    request = window.recording_bridge.calls[0]
    assert request.channels == (7, 1) and request.frames_per_channel == 25600
    assert window.ve_prewarm_lifetime.snapshot().state == "pending"
    assert window._try_start_ve_prewarm(selected, [7, 1], "duplicate") == "consumed"
    assert len(window.recording_bridge.calls) == 1

    window.recording_bridge.callback(_completion(request))
    assert window.ve_prewarm_lifetime.snapshot().state == "succeeded"
    assert window.status_updates == 1


def test_synchronous_service_terminal_cannot_restore_pending_ui():
    window = _WindowHarness(bridge=_SynchronousBridge())
    selected = device_info()
    window.mic, window.mic_channels = selected, [7, 1]

    assert window._try_start_ve_prewarm(selected, [7, 1], "startup") == "accepted"
    assert window.ve_prewarm_lifetime.snapshot().state == "succeeded"
    assert "VE 设备正在初始化…" not in window._status.messages


@pytest.mark.parametrize("projected_busy,service_status", [(True, "accepted"), (False, "busy")])
def test_busy_consumes_opportunity_without_queue_or_later_prewarm(projected_busy, service_status):
    bridge = _Bridge(service_status)
    bridge.hardware_busy = projected_busy
    window = _WindowHarness(bridge=bridge)
    selected = device_info()

    assert window._try_start_ve_prewarm(selected, [7, 1], "startup") == "skipped_busy"
    assert window.ve_prewarm_lifetime.snapshot().state == "skipped_busy"
    assert len(bridge.calls) == (0 if projected_busy else 1)

    bridge.hardware_busy = False
    bridge.status = "accepted"
    assert window._try_start_ve_prewarm(selected, [7, 1], "later") == "consumed"
    assert len(bridge.calls) == (0 if projected_busy else 1)
    assert window.ve_prewarm_lifetime.admission_for(
        window._ve_prewarm_signature(selected, [7, 1])) == "allowed"


def test_completion_requires_exact_context_and_failure_uses_ownership_modal(monkeypatch):
    warnings = []
    monkeypatch.setattr("main_window.QMessageBox.critical",
                        lambda *args: warnings.append((args[-2], args[-1])))
    window = _WindowHarness()
    selected = device_info()
    window.mic, window.mic_channels = selected, [7, 1]
    assert window._try_start_ve_prewarm(selected, [7, 1], "hardware") == "accepted"
    request = window.recording_bridge.calls[0]

    stale = SimpleNamespace(**{
        **_completion(request, success=False, stage="start_task", code=-12001,
                      detail="first cause").__dict__,
        "warmup_id": "stale",
    })
    window.recording_bridge.callback(stale)
    assert window.ve_prewarm_lifetime.snapshot().state == "pending"
    assert warnings == []

    failure = _completion(request, success=False, stage="release_ve", code=-9,
                          detail="ClearTask failed", ownership_safe=False,
                          diagnostics=("secondary bind failure",))
    window.recording_bridge.callback(failure)
    snapshot = window.ve_prewarm_lifetime.snapshot()
    assert snapshot.state == "failed" and snapshot.failed_signature == request.signature
    assert snapshot.ownership_safe is False
    assert len(warnings) == 1
    assert warnings[0][0] == "Vkinging 设备初始化失败"
    assert "code=-9" in warnings[0][1] and "重启" in warnings[0][1]
    assert "ClearTask failed" not in warnings[0][1]
    assert window._status.messages[-1] == "VE 设备初始化失败"
    assert window._try_start_ve_prewarm(
        device_info(machine_id="later-machine"), [7, 1], "later") == "consumed"
    assert len(window.recording_bridge.calls) == 1


@pytest.mark.parametrize("success", [True, False])
def test_authenticated_stale_selection_terminal_consumes_without_touching_new_ui(
    monkeypatch, success,
):
    warnings = []
    monkeypatch.setattr("main_window.QMessageBox.critical",
                        lambda *args: warnings.append(args[-1]))
    window = _WindowHarness()
    original = device_info()
    window.mic, window.mic_channels = original, [7, 1]
    assert window._try_start_ve_prewarm(original, [7, 1], "startup") == "accepted"
    request = window.recording_bridge.calls[0]
    status_updates = window.status_updates
    messages = list(window._status.messages)
    current = device_info(machine_id="new-selection")
    window.mic, window.mic_channels = current, [7, 1]

    window.recording_bridge.callback(_completion(
        request, success=success,
        stage="completed" if success else "start_task",
        detail="" if success else "old selection failed"))

    snapshot = window.ve_prewarm_lifetime.snapshot()
    assert snapshot.state == ("succeeded" if success else "failed")
    assert warnings == []
    assert window.status_updates == status_updates
    assert window._status.messages == messages
    assert window._ve_prewarm_context is None
    assert window.ve_prewarm_lifetime.admission_for(
        window._ve_prewarm_signature(current, [7, 1])) == "allowed"
    expected = "allowed" if success else "failed_signature"
    assert window.ve_prewarm_lifetime.admission_for(request.signature) == expected


def test_closing_status_consumes_silently(monkeypatch):
    warnings = []
    monkeypatch.setattr("main_window.QMessageBox.critical", lambda *args: warnings.append(args))
    window = _WindowHarness(bridge=_Bridge("closing"))

    assert window._try_start_ve_prewarm(device_info(), [7, 1], "startup") == "closing"
    assert window.ve_prewarm_lifetime.snapshot().state == "skipped_busy"
    assert warnings == []


def test_ownership_safe_release_failure_suggests_hardware_reselection(monkeypatch):
    warnings = []
    monkeypatch.setattr("main_window.QMessageBox.critical",
                        lambda *args: warnings.append(args[-1]))
    window = _WindowHarness()
    selected = device_info()
    window.mic, window.mic_channels = selected, [7, 1]
    assert window._try_start_ve_prewarm(selected, [7, 1], "hardware") == "accepted"
    request = window.recording_bridge.calls[0]

    window.recording_bridge.callback(_completion(
        request, success=False, stage="release_ve", detail="release failed",
        ownership_safe=True))

    assert window.ve_prewarm_lifetime.snapshot().ownership_safe is True
    assert len(warnings) == 1
    assert "硬件设置" in warnings[0] and "重启应用" not in warnings[0]


def test_first_ve_confirmation_lets_prewarm_own_incompatible_release(monkeypatch):
    window = _WindowHarness()
    old = device_info(machine_id="old-machine")
    new = device_info(machine_id="new-machine", input_config=input_config(44100))
    window.mic, window.mic_channels = old, [7, 1]
    window.sequence_window.mic, window.sequence_window.mic_channels = old, [7, 1]
    monkeypatch.setattr("main_window.open_hardware_selection_window",
                        lambda **kwargs: (True, None, [], new, [7, 1]))

    window.on_hardware_window_init()

    assert len(window.recording_bridge.calls) == 1
    assert window.recording_bridge.calls[0].signature[3] == 44100
    assert window.recording_bridge.release_calls == []


def test_cancel_does_not_consume_and_consumed_or_ordinary_switch_keeps_release(monkeypatch):
    lifetime = VePrewarmLifetime()
    window = _WindowHarness(lifetime=lifetime)
    old = device_info(machine_id="old-machine")
    window.mic, window.mic_channels = old, [7, 1]
    window.sequence_window.mic, window.sequence_window.mic_channels = old, [7, 1]
    monkeypatch.setattr("main_window.open_hardware_selection_window",
                        lambda **kwargs: (False, None, [], None, []))
    window.on_hardware_window_init()
    assert lifetime.snapshot().state == "available"

    selected_signature = window._ve_prewarm_signature(old, [7, 1])
    assert lifetime.claim("already-used", selected_signature)
    assert lifetime.mark_succeeded("already-used", selected_signature)
    replacement = device_info(machine_id="new-machine")
    monkeypatch.setattr("main_window.open_hardware_selection_window",
                        lambda **kwargs: (True, None, [], replacement, [7, 1]))
    window.on_hardware_window_init()
    assert len(window.recording_bridge.calls) == 0
    assert len(window.recording_bridge.release_calls) == 1

    ordinary = {"backend": "sounddevice", "name": "mic", "hostapi": None}
    monkeypatch.setattr("main_window.save_if_changed", lambda *args, **kwargs: False)
    monkeypatch.setattr("main_window.open_hardware_selection_window",
                        lambda **kwargs: (True, None, [], ordinary, [0]))
    window.on_hardware_window_init()
    assert len(window.recording_bridge.release_calls) == 2
    assert lifetime.snapshot().state == "succeeded"


@pytest.mark.parametrize("code", [-9, None])
@pytest.mark.parametrize("ownership_safe", [False, True])
def test_prewarm_fault_popup_is_safe_but_original_and_identity_are_logged(
        monkeypatch, caplog, code, ownership_safe):
    from ui.vkinging_presentation import ve_failure_text
    warnings = []
    monkeypatch.setattr("main_window.QMessageBox.critical", lambda *args: warnings.append(args[-1]))
    window = _WindowHarness()
    window.mic, window.mic_channels = device_info(), [7, 1]
    window._try_start_ve_prewarm(window.mic, [7, 1], "startup")
    request = window.recording_bridge.calls[0]
    failure = _completion(request, success=False, stage="other-private-machine-id",
                          code=code, detail="unknown fault", ownership_safe=ownership_safe,
                          diagnostics=("test-machine-1 and other-private-machine-id",))
    before = deepcopy(failure)
    window.recording_bridge.callback(failure)
    window.recording_bridge.callback(failure)
    assert len(warnings) == 1
    assert ve_failure_text("prewarm", code=code) in warnings[0]
    assert "test-machine-1" not in warnings[0] and "other-private-machine-id" not in warnings[0]
    assert ("重启" in warnings[0]) is (not ownership_safe)
    assert failure == before
    assert "unknown fault" in caplog.text and failure.diagnostics[0] in caplog.text
    assert window.ve_prewarm_lifetime.snapshot().state == "failed"


@pytest.mark.parametrize("boundary", ["configuration", "admission", "completion", "closing", "stale"])
def test_prewarm_failure_logs_context_even_without_id_in_detail(monkeypatch, caplog, boundary):
    monkeypatch.setattr("main_window.QMessageBox.critical", lambda *args: None)
    window = _WindowHarness(bridge=_Bridge("rejected" if boundary == "admission" else "accepted"))
    window.mic, window.mic_channels = device_info(), [7, 1]
    if boundary == "configuration":
        window.ve_profile_store.load = lambda *args: (_ for _ in ()).throw(ValueError("unknown config fault"))
    window._try_start_ve_prewarm(window.mic, [7, 1], "startup")
    if boundary in ("completion", "closing", "stale"):
        request = window.recording_bridge.calls[0]
        window._recording_close_requested = boundary == "closing"
        if boundary == "stale":
            window.mic = {"backend": "sounddevice", "name": "ordinary"}
        window.recording_bridge.callback(_completion(request, success=False, detail="unknown fault"))
    assert "test-machine-1" in caplog.text
    assert "fault" in caplog.text or "rejected" in caplog.text
    assert all("test-machine-1" not in message for message in window._status.messages)


def test_prewarm_renderer_ignores_untrusted_string_code_without_logging(monkeypatch, caplog):
    from ui.vkinging_presentation import ve_failure_text
    warnings = []
    monkeypatch.setattr("main_window.QMessageBox.critical", lambda *args: warnings.append(args[-1]))
    window = _WindowHarness()
    fault = SimpleNamespace(code="other-private-machine-id", stage="test-machine-1",
                            detail="test-machine-1 and other-private-machine-id")
    window._render_ve_prewarm_failure(fault, ownership_safe=False)
    assert ve_failure_text("prewarm") in warnings[0]
    assert "test-machine-1" not in warnings[0] and "other-private-machine-id" not in warnings[0]
    assert "重启" in warnings[0]
    assert not caplog.records


@pytest.mark.parametrize("diagnostic", ["unknown configuration fault", "test-machine-1 and other-private-machine-id"])
def test_signature_status_hides_errors_and_logs_identity_once_until_repaired(
        monkeypatch, caplog, diagnostic):
    from ui.vkinging_presentation import ve_failure_text
    window = _WindowHarness()
    window.mic, window.mic_channels = device_info(), [7, 1]
    original_load = window.ve_profile_store.load
    def fail(*args):
        raise OSError(diagnostic)
    monkeypatch.setattr(window.ve_profile_store, "load", fail)
    for _ in range(3):
        assert window._ve_signature(window.mic, window.mic_channels) is None
    assert window._status.messages[-1] == ve_failure_text("configuration")
    errors = [record.getMessage() for record in caplog.records if diagnostic in record.getMessage()]
    assert len(errors) == 1 and "test-machine-1" in errors[0]
    monkeypatch.setattr(window.ve_profile_store, "load", original_load)
    assert window._ve_signature(window.mic, window.mic_channels) is not None
    monkeypatch.setattr(window.ve_profile_store, "load", fail)
    assert window._ve_signature(window.mic, window.mic_channels) is None
    assert len([record for record in caplog.records if diagnostic in record.getMessage()]) == 2
