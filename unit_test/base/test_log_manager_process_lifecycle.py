"""Real process exit and shared-file rotation, isolated from hardware/Qt."""
import json
import logging
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

import pytest

from base import log_manager
from base.log_manager import LogManager
from unit_test.logging_test_support import isolated_project_logger


def run_process(mode, directory, count=1, timeout=15, import_first=None):
    command = [sys.executable, "-m", "unit_test.base.log_manager_process_fakes",
               mode, str(directory), str(count)]
    if import_first is not None:
        script = (f"import {import_first}; "
                  "from unit_test.base.log_manager_process_fakes import main; main()")
        command = [sys.executable, "-c", script, mode, str(directory), str(count)]
    # subprocess.run kills and reaps the direct test process on timeout.
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not result.stderr
    return result


@pytest.mark.parametrize("mode", ["normal", "spawn"])
@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_normal_exit_flushes_tail_and_preserves_caller(tmp_path, mode, count):
    run_process(mode, tmp_path, count)
    text = (tmp_path / "main.log").read_text()
    assert text.count("|INFO|tail=") == count
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    for index in range(count):
        assert f"core|INFO|tail={index}|log_manager_process_fakes.py:{metadata['line']}|" in text


@pytest.mark.parametrize("mode", ["finalizer", "normal-finalizer"])
@pytest.mark.parametrize("import_first", ["logging", "multiprocessing.util"])
def test_shutdown_runs_after_ordinary_producer_finalizers(tmp_path, mode, import_first):
    run_process(mode, tmp_path, import_first=import_first)
    text = (tmp_path / "main.log").read_text()
    assert "tail=0" in text and "producer-finalizer" in text
    stats = json.loads((tmp_path / "finalized.json").read_text())
    assert stats["written"] == 2 and stats["pending"] == 0
    assert stats["dropped_closed"] == 0
    assert not stats["consumer_alive"]


@pytest.mark.parametrize("mode", ["blocked-normal", "blocked-explicit"])
@pytest.mark.parametrize("import_first", ["logging", "multiprocessing.util"])
def test_exit_uses_one_budget_with_blocked_real_file_lock(tmp_path, mode, import_first):
    run_process(mode, tmp_path, timeout=8, import_first=import_first)
    elapsed = time.monotonic() - float((tmp_path / "exit-start").read_text())
    assert elapsed < 1.0, f"six handlers accumulated shutdown waits: {elapsed:.2f}s"
    assert (tmp_path / "root-closed").exists()
    if mode == "blocked-explicit":
        stats = json.loads((tmp_path / "shutdown.json").read_text())
        assert stats["consumer_alive"] and stats["pending"] == 6
        assert stats["written"] == 0


def test_three_spawn_processes_rotate_shared_file_without_loss_or_duplicates(tmp_path):
    run_process("rotation", tmp_path, timeout=20)
    for child in range(3):
        stats = json.loads((tmp_path / f"child-{child}.json").read_text())
        assert stats["accepted"] == stats["written"] == 37
        assert stats["pending"] == stats["write_errors"] == stats["dropped_full"] == 0
        assert not stats["consumer_alive"]
    paths = list(tmp_path.glob("main.log*"))
    assert len(paths) > 1, "small threshold must exercise real rotation"
    records = [line for path in paths for line in path.read_text().splitlines()]
    pattern = re.compile(r"core\|INFO\|id=(\d+:\d+)\|log_manager_process_fakes.py:\d+\|\d+")
    matches = [pattern.fullmatch(record) for record in records]
    assert all(matches), records
    ids = [match.group(1) for match in matches]
    assert len(ids) == len(set(ids)) == 111
    assert set(ids) == {f"{child}:{index}" for child in range(3) for index in range(37)}


def test_hooks_do_not_accumulate_or_retain_old_runtimes(tmp_path):
    run_process("hooks", tmp_path)
    text = (tmp_path / "main.log").read_text()
    assert text.count("core.video|INFO|cycle=") == 4
    for index in range(4):
        assert text.count(f"cycle={index}|") == 1


def test_import_does_not_start_consumer_or_create_files(tmp_path):
    run_process("import", tmp_path)
    assert not list(tmp_path.iterdir())


def test_analysis_style_error_drains_preceding_debug_json_and_traceback(tmp_path, monkeypatch):
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        core = LogManager.set_log_handler("core")
        debug = LogManager.set_log_handler("debug")
        debug.info('{"event": "analysis-result"}')
        logging.getLogger("core.video").info("video-before-error")
        try:
            raise ValueError("analysis-error-probe")
        except ValueError:
            core.exception("analysis failed")
        runtime = LogManager._runtime
        with runtime._condition:
            assert runtime._condition.wait_for(lambda: runtime._written == 3, timeout=2)
        assert '{"event": "analysis-result"}' in state.path.with_name("debug.log").read_text()
        text = state.path.read_text()
        assert text.count("video-before-error") == 1
        assert text.index("video-before-error") < text.index("analysis failed")
        assert "Traceback" in text and "ValueError: analysis-error-probe" in text
        assert "test_log_manager_process_lifecycle.py" in text


def test_flush_watermark_across_debug_and_core_ignores_later_blocked_write(tmp_path, monkeypatch):
    first_entered, first_release = threading.Event(), threading.Event()
    later_entered, later_release = threading.Event(), threading.Event()
    original = log_manager._BatchFileHandler.do_write

    def gated_write(sink, text):
        if "first-gate" in text:
            first_entered.set()
            assert first_release.wait(5)
        if "later-gate" in text:
            later_entered.set()
            assert later_release.wait(5)
        return original(sink, text)

    monkeypatch.setattr(log_manager._BatchFileHandler, "do_write", gated_write)
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        core = LogManager.set_log_handler("core")
        debug = LogManager.set_log_handler("debug")
        core.error("first-gate")
        assert first_entered.wait(2)
        debug.info("captured-debug")
        runtime = LogManager._runtime
        finished = threading.Event()
        results = []

        def flush():
            results.append(LogManager.flush(2))
            finished.set()

        flusher = threading.Thread(target=flush)
        flusher.start()
        try:
            with runtime._condition:
                assert runtime._condition.wait_for(lambda: 2 in runtime._barriers, timeout=2)
            core.error("later-gate")
            first_release.set()
            assert later_entered.wait(2)
            assert finished.wait(1)
            assert results == [True]
            assert "captured-debug" in state.path.with_name("debug.log").read_text()
            assert LogManager.get_async_stats()["pending"] == 1
        finally:
            first_release.set()
            later_release.set()
            flusher.join(3)
