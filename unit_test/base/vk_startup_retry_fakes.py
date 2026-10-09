"""Spawn-safe SDK and IPC gates for startup recovery integration tests."""
import json
import os
from pathlib import Path
import threading
import time

from base.streaming_file_writer import StreamingWavWriter
from unit_test.base.ve3668n_fakes import CaptureSDK


def _wait_for(path, *, observe=None):
    deadline = time.monotonic() + 10
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"test gate was not released: {path}")
        if observe is not None:
            observe()
        threading.Event().wait(.005)


def startup_retry_dependencies(*, trace_dir, gate_retry_send=False, cleanup_mode=None,
                               block_instance=1, pause_finalizers=(), cleanup_instance=1):
    """Each worker owns its counter and actual owner Thread references."""
    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    owners = []
    trace_lock = threading.Lock()

    def record(file, operation, **values):
        with trace_lock, (trace_dir / file).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(operation=operation, pid=os.getpid(),
                                         thread_id=threading.get_ident(),
                                         at=time.monotonic(), **values)) + "\n")

    class StartupSDK(CaptureSDK):
        def __init__(self):
            self.instance = len(owners) + 1
            previous = [owner.is_alive() for owner in owners]
            record("native.jsonl", "sdk_create", instance=self.instance,
                   previous_owners_alive=previous)
            assert not any(previous), "replacement SDK overlapped an old owner"
            owners.append(threading.current_thread())
            super().__init__(failures=("clear_task",) if cleanup_mode == "failure"
                             and self.instance == cleanup_instance else ())
            self.values = (.125, -.25)

        def _call(self, operation, *args, **kwargs):
            record("native.jsonl", operation, instance=self.instance, args=args)
            if self.instance == block_instance and operation == "create_task":
                (trace_dir / f"native-{self.instance}-entered").touch()
                _wait_for(trace_dir / f"release-native-{self.instance}")
            if self.instance == cleanup_instance and cleanup_mode is not None:
                if operation == ("clear_task" if cleanup_mode == "failure" else "stop_task"):
                    (trace_dir / "cleanup-entered").touch()
                if operation == ("close" if cleanup_mode == "failure" else "stop_task"):
                    _wait_for(trace_dir / "release-cleanup")
            return super()._call(operation, *args, **kwargs)

        def read_task_data(self, *args, **kwargs):
            # Pace continuous idle draining like the existing worker SDK fake.
            threading.Event().wait(.003)
            return super().read_task_data(*args, **kwargs)

        def close(self):
            super().close()
            record("native.jsonl", "cleanup_complete", instance=self.instance)

    class TracedWriter(StreamingWavWriter):
        def __init__(self, path, **kwargs):
            super().__init__(path, **kwargs)
            record("writer.jsonl", "open", path=str(path))

    def append_metadata(path, metadata, **kwargs):
        from base.wav_calibration_metadata import append_owned_recording_calibration_metadata_result

        identity = Path(path).stem
        if identity in pause_finalizers:
            (trace_dir / f"finalizer-{identity}-entered").touch()
            _wait_for(trace_dir / f"release-finalizer-{identity}")
        return append_owned_recording_calibration_metadata_result(path, metadata, **kwargs)

    return {"ve_sdk_factory": StartupSDK, "writer_factory": TracedWriter,
            "metadata_appender": append_metadata}


def gated_startup_worker(control, preview, generation, factory, options, *args):
    """Hold the real sender before queue selection, accumulating retry events."""
    from base import recording_worker as module

    trace_dir = Path(options["trace_dir"])
    lanes = {}
    original_send_loop = module._send_loop

    def observed_send_loop(connection, outgoing, broken, latest=None, wake=None,
                           urgent=None, ordered=None, diagnostics=None):
        if connection is wrapped:
            lanes.update(outgoing=outgoing, ordered=ordered, latest=latest)
        return original_send_loop(connection, outgoing, broken, latest, wake,
                                  urgent, ordered, diagnostics)

    def observe_backlog():
        if (trace_dir / "backlog-ready").exists():
            return
        snapshot = {}
        for name, lane in lanes.items():
            with lane.mutex:
                snapshot[name] = [event.kind for event in lane.queue if event is not None]
        if any(kind in ("completed", "ve_prewarm_terminal")
               for values in snapshot.values() for kind in values):
            (trace_dir / "backlog.json").write_text(json.dumps(snapshot))
            (trace_dir / "backlog-ready").touch()

    class Connection:
        def __getattr__(self, name):
            return getattr(control, name)

        def send(self, event):
            with (trace_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(kind=event.kind, request_id=event.request_id,
                                             payload=repr(event.payload))) + "\n")
            control.send(event)
            if options.get("gate_retry_send") and event.kind == "ready":
                # The parent can now send start. Keep this send call pending so
                # proof AND subsequent lifecycle events queue before the real
                # sender next chooses among outgoing/ordered/latest lanes.
                (trace_dir / "retry-send-entered").touch()
                _wait_for(trace_dir / "release-retry-send", observe=observe_backlog)

    wrapped = Connection()
    module._send_loop = observed_send_loop
    try:
        module.recording_worker(wrapped, preview, generation, factory, options, *args)
    finally:
        module._send_loop = original_send_loop
