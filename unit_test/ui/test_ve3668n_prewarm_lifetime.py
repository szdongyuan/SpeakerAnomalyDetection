"""Focused process-lifetime coverage independent of VE native hardware."""

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from base.ve3668n_prewarm_lifetime import VePrewarmLifetime


def signature(machine_id="test-machine-1", channels=(7, 1), rate=51200):
    return ("vkinging", machine_id, channels, rate)


class MutableHashable:
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return 1


def test_claim_is_atomic_and_snapshot_is_immutable():
    lifetime = VePrewarmLifetime()
    selected = signature()
    barrier = threading.Barrier(9)
    results = []

    def claim(index):
        barrier.wait()
        results.append((index, lifetime.claim(f"selection-{index}", selected)))

    threads = [threading.Thread(target=claim, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(2)

    assert sum(accepted for _, accepted in results) == 1
    snapshot = lifetime.snapshot()
    assert snapshot.state == "pending"
    assert snapshot.signature == selected
    assert snapshot.signature is not selected
    with pytest.raises(FrozenInstanceError):
        snapshot.state = "available"


@pytest.mark.parametrize("token", ["", 1, [], MutableHashable("selection")])
def test_claim_rejects_noncanonical_or_caller_mutable_tokens(token):
    with pytest.raises(ValueError, match="token"):
        VePrewarmLifetime().claim(token, signature())


@pytest.mark.parametrize(
    "selected",
    [
        ["vkinging", "test-machine-1", (7, 1), 51200],
        ("soundcard", "test-machine-1", (7, 1), 51200),
        ("vkinging", " test-machine-1 ", (7, 1), 51200),
        ("vkinging", "test-machine-1", [7, 1], 51200),
        ("vkinging", "test-machine-1", (7, 1), 96000),
        MutableHashable("signature"),
    ],
)
def test_claim_rejects_noncanonical_or_caller_mutable_signatures(selected):
    with pytest.raises(ValueError, match="signature"):
        VePrewarmLifetime().claim("selection", selected)


def test_stale_busy_terminal_rejects_identity_before_reading_detail():
    lifetime = VePrewarmLifetime()
    selected = signature()
    assert lifetime.claim("current", selected)
    assert not lifetime.mark_skipped_busy("stale", selected, object())
    assert lifetime.snapshot().state == "pending"


def test_stale_failure_rejects_identity_before_reading_fault_payload():
    lifetime = VePrewarmLifetime()
    selected = signature()
    assert lifetime.claim("current", selected)
    malformed = SimpleNamespace(stage="start_task", detail="stale", diagnostics=3)
    assert not lifetime.mark_failed(
        "stale", selected, malformed, ownership_safe=True)
    assert lifetime.snapshot().state == "pending"


def test_terminal_and_admission_boundaries_require_canonical_signatures():
    lifetime = VePrewarmLifetime()
    selected = signature()
    assert lifetime.claim("selection", selected)
    malformed = ("vkinging", "test-machine-1", [7, 1], 51200)
    for terminal in (
        lambda: lifetime.mark_succeeded("selection", malformed),
        lambda: lifetime.mark_skipped_busy("selection", malformed, "busy"),
        lambda: lifetime.mark_failed(
            "selection", malformed,
            SimpleNamespace(stage="start", code=None, detail="failed", diagnostics=()),
            ownership_safe=True),
        lambda: lifetime.admission_for(malformed),
    ):
        with pytest.raises(ValueError, match="signature"):
            terminal()
    assert lifetime.snapshot().state == "pending"


def test_succeeded_and_skipped_busy_consume_without_blocking_any_signature():
    for terminal in ("succeeded", "skipped_busy"):
        lifetime = VePrewarmLifetime()
        selected = signature()
        assert lifetime.claim("selection", selected)
        if terminal == "succeeded":
            assert lifetime.mark_succeeded("selection", selected)
        else:
            assert lifetime.mark_skipped_busy("selection", selected, "hardware busy")
        assert lifetime.snapshot().state == terminal
        assert lifetime.snapshot().failed_signature is None
        assert lifetime.admission_for(selected) == "allowed"
        assert lifetime.admission_for(signature(machine_id="other")) == "allowed"
        assert not lifetime.claim("later", signature(machine_id="other"))


def test_failure_blocks_only_exact_signature_and_freezes_diagnostics():
    lifetime = VePrewarmLifetime()
    selected = signature()
    diagnostics = ["attempt 1", "attempt 2"]
    fault = SimpleNamespace(
        stage="start_task", code=-12001,
        detail="iio_device_create_multi_buffer: invalid argument",
        diagnostics=diagnostics,
    )
    assert lifetime.claim("selection", selected)
    assert lifetime.mark_failed("selection", selected, fault, ownership_safe=False)
    diagnostics.append("late mutation")

    snapshot = lifetime.snapshot()
    assert snapshot.failed_signature == selected
    assert snapshot.failed_signature is not selected
    assert snapshot.failure_category == "start_task"
    assert snapshot.diagnostics == ("attempt 1", "attempt 2")
    assert snapshot.ownership_safe is False
    assert lifetime.admission_for(selected) == "failed_signature"
    assert lifetime.admission_for(signature(machine_id="other")) == "allowed"
    assert lifetime.admission_for(None) == "allowed"


def test_same_lifetime_outcome_survives_window_and_bridge_harness_reconstruction():
    lifetime = VePrewarmLifetime()
    selected = signature()
    first_bridge = SimpleNamespace(service=object())
    first_window = SimpleNamespace(
        ve_prewarm_lifetime=lifetime, recording_bridge=first_bridge)
    assert lifetime.claim("selection", selected)
    assert lifetime.mark_succeeded("selection", selected)
    del first_window, first_bridge

    second_bridge = SimpleNamespace(service=object())
    second_window = SimpleNamespace(
        ve_prewarm_lifetime=lifetime, recording_bridge=second_bridge)
    assert second_window.ve_prewarm_lifetime.snapshot().state == "succeeded"
    assert not second_window.ve_prewarm_lifetime.claim("second", selected)


def test_main_window_rejects_invalid_explicit_lifetime():
    from main_window import MainWindow

    with pytest.raises(TypeError, match="ve_prewarm_lifetime"):
        MainWindow.__init__(object(), ve_prewarm_lifetime=object())


def test_production_bootstraps_create_and_forward_one_lifetime():
    root = Path(__file__).resolve().parents[2]
    launcher_tree = ast.parse(
        (root / "main_window_Launcher.py").read_text(encoding="utf-8"))
    launcher = next(
        node for node in launcher_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MainWindowLauncher")
    lifetime_creations = [
        node for node in ast.walk(launcher)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute)
                and target.attr == "ve_prewarm_lifetime" for target in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "VePrewarmLifetime"
    ]
    injected_calls = [
        node for node in ast.walk(launcher)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "MainWindow"
        and any(keyword.arg == "ve_prewarm_lifetime" for keyword in node.keywords)
    ]
    assert len(lifetime_creations) == len(injected_calls) == 1

    direct_tree = ast.parse((root / "main_window.py").read_text(encoding="utf-8"))
    direct_creations = [
        node for block in direct_tree.body if isinstance(block, ast.If)
        for node in ast.walk(block)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name)
                and target.id == "ve_prewarm_lifetime" for target in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "VePrewarmLifetime"
    ]
    direct_calls = [
        node for node in ast.walk(direct_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "MainWindow"
        and any(keyword.arg == "ve_prewarm_lifetime" for keyword in node.keywords)
    ]
    assert len(direct_creations) == len(direct_calls) == 1
