"""Spawnable forced-exit probes; deliberate crash probes remain abrupt."""
import json
import logging
import os
from pathlib import Path
import time

from unit_test.base.log_manager_process_fakes import configure


def tail(directory, count, ready=None, blocked=None):
    from base.log_exit import exit_with_log_drain

    manager = configure(directory)
    from base import log_manager
    log_manager.DEFAULT_LOG["log_format"] += "|%(created).9f"
    logger = manager.set_log_handler("core")
    records = []

    class Metadata(logging.Handler):
        def emit(self, record):
            records.append(dict(created=record.created, line=record.lineno))

    logger.addHandler(Metadata())
    for index in range(count):
        logger.info("tail=%d", index)
    Path(directory, "metadata.json").write_text(json.dumps(records))
    if ready is not None:
        ready.set()
        blocked.wait()
        Path(directory, "returned").touch()
    else:
        exit_with_log_drain(23)


def abrupt():
    os._exit(31)


def idle(ready, release):
    ready.set()
    release.wait()


def blocked_sink(directory, ready, self_exit, budget, release):
    import threading
    from base import log_manager
    from base.log_exit import exit_with_log_drain

    manager = configure(directory)
    entered = threading.Event()

    def gate(sink, message):
        assert sink.is_locked, "must block after real file lock acquisition"
        entered.set()
        threading.Event().wait()

    log_manager._BatchFileHandler.do_write = gate
    manager.set_log_handler("core").error("blocked-output")
    assert entered.wait(3)
    ready.set()
    if self_exit:
        release.wait()
        Path(directory, "exit-start").write_text(str(time.monotonic()))
        exit_with_log_drain(24, timeout=budget)
    release.wait()


def exercise_blocked(mode, directory, budget):
    import multiprocessing
    from base.log_exit import ProcessLogDrain, run_with_log_drain

    context = multiprocessing.get_context("spawn")
    drain = ProcessLogDrain.create(context)
    ready, release = context.Event(), context.Event()
    child = context.Process(target=run_with_log_drain,
                            args=(blocked_sink, (directory, ready, mode != "parent", budget, release),
                                  drain.child_endpoint))
    child.start()
    try:
        assert ready.wait(4)
        start = time.monotonic()
        if mode in ("parent", "race"):
            drain.begin("blocked-sink", timeout=budget)
        if mode != "parent":
            if mode == "race":
                time.sleep(budget * 0.6)
            release.set()
            child.join(budget + 1.5)
            assert child.exitcode == 24
            result = drain.poll()
        else:
            result = drain.wait()
            child.terminate()
            child.join(1.5)
        assert not child.is_alive()
        assert result.status == "timeout"
        elapsed = time.monotonic() - start
        assert elapsed < budget + 0.8
        Path(directory, "outcome.json").write_text(json.dumps(dict(
            elapsed=elapsed, status=result.status, stats=result.stats, exitcode=child.exitcode)))
    finally:
        if child.is_alive():
            child.kill()
        child.join(2)
        child.close()
        drain.close()


def import_and_reacquire(directory):
    import threading
    from base.log_exit import ProcessLogDrain, run_with_log_drain, _ProcessBinding
    import multiprocessing
    assert not any(t.name.startswith("project-log-") for t in threading.enumerate())
    assert not list(Path(directory).iterdir())
    manager = configure(directory)
    for index in range(4):
        drain = ProcessLogDrain.create(multiprocessing.get_context("spawn"))
        run_with_log_drain(lambda: manager.set_log_handler("core").info("cycle=%d", index),
                           (), drain.child_endpoint)
        assert _ProcessBinding.cycle is None
        assert not any(t.name == "project-log-control" for t in threading.enumerate())
        assert manager.shutdown_all(1)
        drain.close()


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "lifecycle":
        import_and_reacquire(sys.argv[2])
    else:
        exercise_blocked(sys.argv[1], sys.argv[2], float(sys.argv[3]))
