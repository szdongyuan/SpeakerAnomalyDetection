"""Early verdict timing, save ownership and final result equivalence."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
import gc
import weakref

import pytest

from base.analysis import executor, storage
from base.analysis.contracts import AnalysisInstance
from base.audio_label_policy import judgement_summary, required_judgements
from unit_test.base.analysis.test_executor import task


def judged_task(task):
    parameters = {"weighting": "Z", "limit_checked": True, "limit_mode": "manual",
                  "manual_input_mode": "constant", "constant_upper_enabled": True,
                  "constant_upper_value": 200, "constant_lower_enabled": False}
    return replace(task, instances=(
        AnalysisInstance("SPL", "SPL", 0, parameters),
        AnalysisInstance("SPL", "SPL", 1, parameters),
    ))


def test_verdict_is_before_last_save_and_does_not_include_curves(task, monkeypatch):
    task = judged_task(task)
    sequence, messages = [], []
    original = executor.save_instance_artifacts

    def save(*args):
        sequence.append(args[1].channel_index)
        return original(*args)

    def ready(payload):
        sequence.append("verdict")
        messages.append(payload)

    monkeypatch.setattr(executor, "save_instance_artifacts", save)
    result = executor.execute_task(task, on_judgement_ready=ready)
    assert sequence == [0, "verdict", 1]
    assert len(messages) == 1
    assert judgement_summary(messages[0]["instance_results"], required_judgements(task.instances)) == result["judgement_summary"]
    assert all("display_payload" not in item and "metrics" not in item
               for item in messages[0]["instance_results"])


def test_no_limits_has_no_early_verdict(task):
    messages = []
    executor.execute_task(task, on_judgement_ready=messages.append)
    assert messages == []


def test_failed_required_instance_is_not_ok(task):
    task = judged_task(task)
    task = replace(task, instances=(*task.instances,
        AnalysisInstance("unsupported", "AI", 0, {"limit_checked": True})))
    messages = []
    result = executor.execute_task(task, on_judgement_ready=messages.append)
    assert len(messages) == 1
    assert judgement_summary(messages[0]["instance_results"], required_judgements(task.instances)) == result["judgement_summary"]
    assert result["judgement_summary"]["status"] == "结果不完整"


def test_blocked_save_does_not_block_verdict_but_blocks_terminal(task, monkeypatch):
    task = judged_task(task)
    baseline = executor.execute_task(task)
    started, release, ready = Event(), Event(), Event()
    original = storage.write_csv_atomic

    def save(*args):
        started.set()
        assert release.wait(10)
        return original(*args)

    monkeypatch.setattr(storage, "write_csv_atomic", save)
    with ThreadPoolExecutor(max_workers=1) as writer, ThreadPoolExecutor(max_workers=1) as runner:
        future = runner.submit(executor.execute_task, task, save_executor=writer,
                               on_judgement_ready=lambda payload: ready.set())
        try:
            assert started.wait(5)
            assert ready.wait(5)
            assert not future.done()
        finally:
            release.set()
        assert future.result(timeout=10) == baseline


def test_callback_failure_drains_already_submitted_saves(task, monkeypatch):
    task = judged_task(task)
    started, release, notified = Event(), Event(), Event()
    original = storage.write_csv_atomic

    def save(*args):
        started.set()
        assert release.wait(10)
        return original(*args)

    def ready(payload):
        notified.set()
        raise OSError("pipe closed")

    monkeypatch.setattr(storage, "write_csv_atomic", save)
    with ThreadPoolExecutor(max_workers=1) as writer, ThreadPoolExecutor(max_workers=1) as runner:
        future = runner.submit(executor.execute_task, task, save_executor=writer, on_judgement_ready=ready)
        try:
            assert started.wait(5) and notified.wait(5)
            assert not future.done()
        finally:
            release.set()
        with pytest.raises(OSError, match="pipe closed"):
            future.result(timeout=10)


def test_resident_writer_does_not_retain_previous_task_curves(task, monkeypatch):
    original = executor._execute_instance
    references = []

    def track(*args, **kwargs):
        result = original(*args, **kwargs)
        references.append(weakref.ref(result["display_payload"]["y"]))
        return result

    monkeypatch.setattr(executor, "_execute_instance", track)
    with ThreadPoolExecutor(max_workers=1) as writer:
        for _ in range(5):
            result = executor.execute_task(task, save_executor=writer)
            assert result["instance_results"][0]["artifacts"][0]["status"] == "saved"
            del result
            gc.collect()
            assert all(reference() is None for reference in references)
