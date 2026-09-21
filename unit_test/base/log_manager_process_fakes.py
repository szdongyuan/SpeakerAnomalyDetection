"""Importable, hardware-free process targets; importing acquires no logger."""
import inspect
import json
import logging
import multiprocessing
from multiprocessing.util import Finalize
from pathlib import Path
import sys
import threading
import time


def configure(directory, *, rotation=1 << 20):
    from base import log_manager

    directory = Path(directory)
    config = dict(log_name=str(directory / "main.log"), max_size=rotation,
                  backup_count=100,
                  log_format="%(name)s|%(levelname)s|%(message)s|%(filename)s:%(lineno)d|%(process)d")
    log_manager.LOG_DIR = str(directory)
    log_manager.DEFAULT_LOG = config
    log_manager.LOG_MAPPING = {
        "core": config,
        "debug": dict(config, log_name=str(directory / "debug.log")),
    }
    return log_manager.LogManager


def tail_child(directory, count, producer_finalizer=False):
    manager = configure(directory)
    logger = manager.set_log_handler("core")
    for index in range(count):
        line = inspect.currentframe().f_lineno + 1
        logger.info("tail=%d", index)
    Path(directory, "metadata.json").write_text(json.dumps({"line": line}))
    if producer_finalizer:
        Finalize(None, logger.info, args=("producer-finalizer",), exitpriority=0)
        Finalize(None, observe_finalized, args=(directory,), exitpriority=-20)


def observe_finalized(directory):
    from base.log_manager import LogManager

    Path(directory, "finalized.json").write_text(json.dumps(LogManager.get_async_stats()))


def blocked_exit(directory, explicit):
    from base import log_manager

    manager = configure(directory)
    log_manager.LOG_SHUTDOWN_TIMEOUT = 0.3
    entered = threading.Event()
    never_release = threading.Event()

    def blocked_write(sink, message):
        # write_batch has already acquired the real cross-process file lock.
        assert sink.is_locked
        entered.set()
        never_release.wait()

    log_manager._BatchFileHandler.do_write = blocked_write

    class RootProbe(logging.Handler):
        def emit(self, record):
            pass

        def close(self):
            Path(directory, "root-closed").touch()
            super().close()

    probe = RootProbe()
    logging.getLogger().addHandler(probe)
    loggers = [manager.set_log_handler(f"route-{index}") for index in range(6)]
    for logger in loggers:
        logger.error("blocked")
    assert entered.wait(3), "consumer did not reach real locked write"
    Path(directory, "exit-start").write_text(str(time.monotonic()))
    if explicit:
        logging.shutdown()
        Path(directory, "shutdown.json").write_text(json.dumps(manager.get_async_stats()))


def rotation_child(directory, identity, barrier):
    manager = configure(directory, rotation=240)
    logger = manager.set_log_handler("core")
    Finalize(None, rotation_stats, args=(directory, identity), exitpriority=-20)
    barrier.wait(5)
    for index in range(35):
        logger.info("id=%d:%d", identity, index)
    # Windows msvcrt's blocking lock retries at one-second intervals. Observe
    # ordinary full-batch completion before testing the bounded exit tail, so
    # this integrity test does not conflate a permitted exit timeout with loss
    # during rotation. No flush or shutdown is used by the child target.
    runtime = manager._runtime
    with runtime._condition:
        assert runtime._condition.wait_for(lambda: runtime._written == 35, timeout=7)
    for index in range(35, 37):
        logger.info("id=%d:%d", identity, index)
    assert manager.get_async_stats()["accepted"] == 37


def rotation_stats(directory, identity):
    from base.log_manager import LogManager

    Path(directory, f"child-{identity}.json").write_text(json.dumps(LogManager.get_async_stats()))


def run_rotation(directory):
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(3)
    children = [context.Process(target=rotation_child, args=(directory, i, barrier))
                for i in range(3)]
    try:
        for child in children:
            child.start()
        deadline = time.monotonic() + 10
        for child in children:
            child.join(max(0, deadline - time.monotonic()))
        assert all(not child.is_alive() and child.exitcode == 0 for child in children)
        assert not multiprocessing.active_children()
    finally:
        for child in children:
            if child.pid is not None:
                if child.is_alive():
                    child.kill()
                child.join(3)
                child.close()


def hooks_and_reacquisition(directory):
    import atexit
    import gc
    import weakref
    from base import log_manager

    registrations = []
    register = atexit.register

    def observe(callback, *args, **kwargs):
        registrations.append(callback)
        return register(callback, *args, **kwargs)

    atexit.register = observe
    manager = configure(directory)
    old_runtimes = []
    finalizer = None
    for index in range(4):
        logger = manager.set_log_handler("core")
        manager.set_log_handler("core.video")
        if finalizer is None:
            finalizer = manager._process_finalizer
        assert manager._process_finalizer is finalizer
        assert len([t for t in threading.enumerate() if t.name == "project-log-consumer"]) == 1
        logging.getLogger("core.video").info("cycle=%d", index)
        assert manager.shutdown_all(2)
        assert not any(t.name == "project-log-consumer" for t in threading.enumerate())
        old_runtimes.append(weakref.ref(manager._runtime))
    gc.collect()
    assert all(reference() is None for reference in old_runtimes[:-1])
    assert registrations == [manager._shutdown_at_exit]
    assert len([h for h in logger.handlers if isinstance(h, log_manager._ProjectQueueHandler)]) == 1


def run_spawn_tail(directory, count, producer_finalizer=False):
    child = multiprocessing.get_context("spawn").Process(
        target=tail_child, args=(directory, count, producer_finalizer))
    child.start()
    try:
        child.join(10)
        assert not child.is_alive(), "tail child did not finish"
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.kill()
            child.join(5)
        child.close()


def main():
    mode, directory, count = sys.argv[1:]
    if mode == "spawn":
        run_spawn_tail(directory, int(count))
    elif mode == "finalizer":
        run_spawn_tail(directory, int(count), True)
    elif mode == "normal-finalizer":
        tail_child(directory, int(count), True)
    elif mode in ("blocked-normal", "blocked-explicit"):
        blocked_exit(directory, mode == "blocked-explicit")
    elif mode == "rotation":
        run_rotation(directory)
    elif mode == "hooks":
        hooks_and_reacquisition(directory)
    elif mode == "import":
        manager = configure(directory)
        assert manager._runtime is None
        assert not manager._exit_hooks_registered
        assert not any(t.name == "project-log-consumer" for t in threading.enumerate())
    else:
        tail_child(directory, int(count))


if __name__ == "__main__":
    main()
