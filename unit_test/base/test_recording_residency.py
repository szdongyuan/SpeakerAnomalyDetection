"""Resident process and soundcard probing through the real spawn boundary."""
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from base.recording_process_protocol import RecordingEvent, VePrewarmRequest
from base.recording_service import RecordingCallbacks, RecordingService
from unit_test.base.recording_process_fakes import (
    device_info, persistent_ve_worker_dependencies, process_dependencies,
)
from unit_test.base.test_recording_service import eventually, request
from unit_test.base.vkinging_fakes import capture_request


def resident_dependencies(**options):
    """Snapshot inventory only when PortAudio would initialize."""
    from base import recording_worker

    root = Path(options["trace_dir"])
    if options.get("fail_first") and not (root / "failed-once").exists():
        (root / "failed-once").touch()
        raise OSError("injected worker factory failure")
    if options.get("block_factory"):
        (root / "factory-entered").touch()
        eventually(lambda: (root / "release-factory").exists(), timeout=10)
    dependencies = process_dependencies(**options)
    backend = dependencies.pop("backend")
    dependencies["ve_sdk_factory"] = persistent_ve_worker_dependencies(**options)["ve_sdk_factory"]
    open_stream = backend.InputStream
    apis = [{"name": "Selected API"}, {"name": "Other API"}]
    inventory = []

    def trace(operation):
        with (root / "soundcard.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps((operation, os.getpid())) + "\n")

    def read_inventory():
        devices = json.loads((root / "inventory.json").read_text(encoding="utf-8"))
        backend.default = SimpleNamespace(_default_device=(devices[0]["index"], None))
        return devices

    def load():
        nonlocal inventory
        trace("load")
        inventory = read_inventory()
        return backend

    def initialize():
        nonlocal inventory
        trace("initialize")
        if (root / "fail-refresh").exists():
            (root / "fail-refresh").unlink()
            raise OSError("injected PortAudio refresh failure")
        inventory = read_inventory()

    def query_devices(index=None, kind=None):
        if options.get("block_query"):
            trace("query-blocked")
            threading.Event().wait(30)
        if kind == "input":
            return inventory[-1] if options.get("override_default") else inventory[0]
        if index is None:
            return inventory
        return next(device for device in inventory if device["index"] == index)

    def stream(**config):
        trace("stream")
        return open_stream(**config)

    backend.query_devices = query_devices
    backend.query_hostapis = lambda index=None: apis if index is None else apis[index]
    backend._initialize = initialize
    backend._terminate = lambda: trace("terminate")
    backend.InputStream = stream
    recording_worker.sounddevice_backend = load
    return dependencies


@pytest.fixture
def service(tmp_path):
    active = []

    def make(*, options=None, **kwargs):
        result = RecordingService(
            backend_factory="unit_test.base.test_recording_residency:resident_dependencies",
            backend_options={"trace_dir": str(tmp_path), **(options or {})}, **kwargs)
        active.append(result)
        return result

    yield make
    for item in active:
        item.shutdown()
        assert item.closed.wait(10)


def test_resident_ready_is_idempotent_and_reused_by_ve_and_capture(service, tmp_path):
    owner = service(options={"block_factory": True})
    statuses = []
    report = statuses.append
    assert owner.worker_pid is None
    assert owner.ensure_resident(report) == "accepted"
    eventually(lambda: (tmp_path / "factory-entered").exists())
    assert owner.ensure_resident(report) == "accepted"
    assert owner.worker_pid is not None and owner.worker_status.state == "starting"

    ve_request = capture_request(tmp_path / "ve.wav")
    warmup = VePrewarmRequest.create("warmup", ve_request.device, ve_request.channels,
                                     ve_request.sample_rate, attempt=1)
    completions = []
    assert owner.prewarm_ve(warmup, completions.append) == "accepted"
    (tmp_path / "release-factory").touch()
    eventually(lambda: completions)
    ready, = [status for status in statuses if status.state == "ready"]
    assert ready.pid == owner.worker_pid and ready.generation == owner.generation == 1
    assert completions[0].success
    session = owner.start(ve_request, RecordingCallbacks(
        result_ready=lambda session, _: session.accept_result()))
    assert session.released.wait(8)
    assert session.state == "completed"
    assert session.worker_pid == ready.pid
    assert session.generation == ready.generation
    assert not (tmp_path / "soundcard.jsonl").exists()


def test_repeated_residency_requests_coalesce_before_supervisor_dispatch(service, monkeypatch):
    owner = service()
    entered, release = threading.Event(), threading.Event()
    tick = owner._tick

    def gated_tick():
        entered.set()
        assert release.wait(8)
        tick()

    monkeypatch.setattr(owner, "_tick", gated_tick)
    assert entered.wait(3)
    statuses = []
    report = statuses.append
    try:
        for _ in range(owner._inbox.maxsize + 1):
            assert owner.ensure_resident(report) == "accepted"
        assert owner.worker_pid is None
    finally:
        release.set()
    eventually(lambda: any(status.state == "ready" for status in statuses))
    assert owner.generation == 1
    assert [status.state for status in statuses] == ["starting", "ready"]


def inventory(path, *devices):
    (path / "inventory.json").write_text(json.dumps(devices), encoding="utf-8")


def trace(path):
    target = path / "soundcard.jsonl"
    return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]


def soundcard(**changes):
    return {**device_info(), "default_samplerate": 48000.0, **changes}


def probe(owner, identity, *, name=None, hostapi_name=None):
    from base.recording_process_protocol import SoundcardPrepareRequest

    results = []
    assert owner.prepare_soundcard(
        SoundcardPrepareRequest(identity, name, hostapi_name), results.append) == "accepted"
    eventually(lambda: results)
    result, = results
    assert result.probe_id == identity
    return result


def test_probe_result_wire_rejects_mutable_device_after_unpickling():
    from base.recording_process_protocol import (
        FrozenConfig, SOUNDCARD_PREPARE_RESULT, SoundcardPrepareResult,
    )

    device = FrozenConfig.snapshot({"backend": "sounddevice", "index": 7, "name": "device",
                                    "hostapi": 0, "hostapi_name": "API", "max_input_channels": 1})
    result = SoundcardPrepareResult("probe", 1, device, (0,), 48000)
    event = RecordingEvent(1, "probe", SOUNDCARD_PREPARE_RESULT, result)
    object.__setattr__(device, "entries", list(device.entries))
    with pytest.raises(ValueError, match="immutable"):
        RecordingEvent.__post_init__(event)


def test_probe_refreshes_exact_name_and_hostapi_in_same_child(service, tmp_path):
    default = soundcard(name="system default")
    selected = soundcard(name="selected", index=12, max_input_channels=4,
                         default_samplerate=44100.0)
    other_api = {**selected, "index": 9, "hostapi": 1}
    inventory(tmp_path, default, other_api)
    owner = service(options={"override_default": True})
    owner.ensure_resident()
    first = probe(owner, "default")
    assert first.success and first.device["name"] == "system default"
    assert first.channels == (0, 1, 2) and first.sample_rate == 48000
    worker_pid = owner.worker_pid

    missing = probe(owner, "missing", name="selected", hostapi_name="Selected API")
    assert not missing.success and "missing" in missing.detail
    inventory(tmp_path, default, other_api, selected)
    added = probe(owner, "added", name="selected", hostapi_name="Selected API")
    assert added.success and added.device["index"] == 12
    assert added.device["hostapi_name"] == "Selected API"
    assert added.channels == (0, 1, 2, 3) and added.sample_rate == 44100
    assert type(added.sample_rate) is int

    inventory(tmp_path, default, other_api, soundcard(name="old index reused", index=12),
              {**selected, "index": 14, "max_input_channels": 2, "default_samplerate": 96000.0})
    moved = probe(owner, "moved", name="selected", hostapi_name="Selected API")
    assert moved.device["index"] == 14 and moved.channels == (0, 1)
    assert moved.sample_rate == 96000
    assert moved.generation == owner.generation == first.generation
    assert owner.worker_pid == worker_pid
    assert trace(tmp_path) == [[op, worker_pid] for op in (
        "load", "terminate", "initialize", "terminate", "initialize", "terminate", "initialize")]

    session = owner.start(request(tmp_path, device=moved.device, channels=moved.channels,
                                  sample_rate=moved.sample_rate), RecordingCallbacks(
        result_ready=lambda session, _: session.accept_result()))
    assert session.released.wait(8) and session.state == "completed"
    assert session.worker_pid == worker_pid
    assert trace(tmp_path)[-1] == ["stream", worker_pid]
    owner.shutdown()
    assert owner.closed.wait(8)
    assert trace(tmp_path)[-1] == ["terminate", worker_pid]


@pytest.mark.parametrize("rate", [0, -1, True, 44100.5, "44100", float("inf"), float("nan")])
def test_probe_rejects_invalid_actual_sample_rate(service, tmp_path, rate):
    inventory(tmp_path, soundcard(default_samplerate=rate))
    owner = service()
    owner.ensure_resident()
    result = probe(owner, "invalid-rate")
    assert not result.success and "sample rate" in result.detail
    assert result.device is None and result.channels == ()
    assert trace(tmp_path) == [["load", owner.worker_pid]]


def test_initial_factory_failure_and_idle_death_recover_without_restarting_on_shutdown(service):
    owner = service(options={"fail_first": True}, retry_delay=.1)
    statuses = []
    owner.ensure_resident(statuses.append)
    eventually(lambda: any(item.state == "ready" and item.generation == 2 for item in statuses))
    assert any(item.state == "unavailable" and item.generation == 1
               and "injected worker factory failure" in item.detail for item in statuses)
    previous = owner._worker
    previous.process.terminate()
    eventually(lambda: any(item.state == "ready" and item.generation == 3 for item in statuses))
    assert owner._worker is not previous
    assert any(item.state == "unavailable" and item.generation == 2 for item in statuses)
    owner.shutdown()
    assert owner.closed.wait(8)
    assert owner.ensure_resident() == "closing"
    assert owner.worker_pid is None and owner.generation == 3


def test_shutdown_from_replacement_starting_observer_prevents_spawn(service, monkeypatch):
    owner = service(retry_delay=.05, terminate_timeout=.15, shutdown_timeout=.15)
    spawned = []
    spawn = owner._spawn_process

    def record_spawn():
        spawn()
        spawned.append(owner.worker_pid)

    def shutdown_before_replacement(status):
        if status.state == "starting" and status.generation == 2:
            owner.shutdown()

    monkeypatch.setattr(owner, "_spawn_process", record_spawn)
    owner.ensure_resident(shutdown_before_replacement)
    eventually(lambda: owner.worker_status.state == "ready")
    first_pid = owner.worker_pid
    owner._worker.process.terminate()
    assert owner.closed.wait(8)
    assert spawned == [first_pid]
    assert owner.worker_status.state == "closing"


def test_shutdown_during_spawn_setup_prevents_process_start(service, monkeypatch):
    from multiprocessing.process import BaseProcess
    from base.log_exit import ProcessLogDrain

    owner = service(shutdown_timeout=.15)
    started = []
    create_drain = ProcessLogDrain.create
    start_process = BaseProcess.start

    def create_and_shutdown(*args, **kwargs):
        drain = create_drain(*args, **kwargs)
        owner.shutdown()
        return drain

    def record_start(process):
        started.append(process.name)
        start_process(process)

    monkeypatch.setattr(ProcessLogDrain, "create", create_and_shutdown)
    monkeypatch.setattr(BaseProcess, "start", record_start)
    owner.ensure_resident()
    assert owner.closed.wait(8)
    assert not started and owner.worker_pid is None


def test_shutdown_after_launch_admission_reaps_replacement_without_blocking(service, monkeypatch):
    from multiprocessing.process import BaseProcess

    owner = service(retry_delay=.05, terminate_timeout=.15)
    entered, release = threading.Event(), threading.Event()
    returned, reported = threading.Event(), threading.Event()
    launched = []
    start_process = BaseProcess.start

    def start_with_gate(process):
        launched.append(process)
        if len(launched) == 2:
            entered.set()
            assert release.wait(8)
        start_process(process)

    def shutdown():
        owner.shutdown(reported.set)
        returned.set()

    monkeypatch.setattr(BaseProcess, "start", start_with_gate)
    owner.ensure_resident()
    eventually(lambda: owner.worker_status.state == "ready")
    owner._worker.process.terminate()
    caller = threading.Thread(target=shutdown, daemon=True)
    try:
        assert entered.wait(8)
        caller.start()
        assert returned.wait(2), "shutdown waited for Process.start"
        with owner._lock:
            admitted = getattr(owner, "_launching_worker", None)
            assert owner._closing and admitted is not None
            assert admitted.generation == 2 and admitted.process is launched[1]
        assert owner.worker_pid is None and not owner.closed.is_set() and not reported.is_set()
    finally:
        release.set()
        if caller.ident is not None:
            caller.join(3)
    assert reported.wait(8) and owner.closed.is_set()
    assert owner.worker_pid is None and owner._launching_worker is None and owner.generation == 2
    assert len(launched) == 2 and all(process._closed for process in launched)


@pytest.mark.parametrize("work", ["capture", "prewarm", "probe"])
def test_shutdown_before_lazy_spawn_releases_admitted_work(service, tmp_path, monkeypatch, work):
    from base.recording_process_protocol import SoundcardPrepareRequest

    owner = service(shutdown_timeout=.15)
    started = []
    publish = owner._publish_worker_status
    spawn = owner._spawn_process

    def shutdown_on_starting(status):
        publish(status)
        if status.state == "starting":
            owner.shutdown()

    def record_spawn():
        started.append(True)
        spawn()

    monkeypatch.setattr(owner, "_publish_worker_status", shutdown_on_starting)
    monkeypatch.setattr(owner, "_spawn_process", record_spawn)
    completions = []
    if work == "capture":
        session = owner.start(request(tmp_path, purpose="calibration", channels=(0,), streaming=False))
    elif work == "prewarm":
        capture = capture_request(tmp_path / "ve.wav")
        warmup = VePrewarmRequest.create("warmup", capture.device, capture.channels,
                                         capture.sample_rate, attempt=1)
        assert owner.prewarm_ve(warmup, completions.append) == "accepted"
    else:
        assert owner.prepare_soundcard(SoundcardPrepareRequest("probe"), completions.append) == "accepted"
    assert owner.closed.wait(8)
    assert not started and not owner.busy
    if work == "capture":
        assert session.released.is_set() and session.state == "cancelled"
    assert not completions


@pytest.mark.parametrize("fault, phase, detail", [
    ("malformed_ready", "waiting_ready", "invalid worker event"),
    ("duplicate_ready", "starting", "duplicate worker ready event"),
    ("unexpected_command", "waiting_ready", "soundcard preparation command"),
    ("unexpected_command", "starting", "soundcard preparation command"),
    ("unexpected_command", "capturing", "soundcard preparation command"),
    ("unexpected_command", "detaching", "soundcard preparation command"),
])
def test_worker_protocol_fault_finishes_ve_prewarm_and_restores_residency(
        service, tmp_path, fault, phase, detail):
    from base.recording_process_protocol import (
        SOUNDCARD_PREPARE, SoundcardPrepareRequest,
        VE_PREWARM_DETACHING, VE_PREWARM_PROGRESS, VE_PREWARM_STARTED,
        VePrewarmProgress, VePrewarmStarted,
    )

    owner = service(options={"block_factory": True}, retry_delay=.05,
                    terminate_timeout=.15, shutdown_timeout=.15)
    owner.ensure_resident()
    eventually(lambda: (tmp_path / "factory-entered").exists())
    worker = owner._worker
    capture = capture_request(tmp_path / "ve.wav")
    warmup = VePrewarmRequest.create("warmup", capture.device, capture.channels,
                                     capture.sample_rate, attempt=1)
    completions = []
    assert owner.prewarm_ve(warmup, completions.append) == "accepted"
    eventually(lambda: owner._pending_ve_prewarm.phase == "waiting_ready")
    ready = RecordingEvent(worker.generation, "", "ready")
    if fault == "malformed_ready":
        object.__setattr__(ready, "request_id", "unexpected")
    if phase != "waiting_ready":
        owner._inbox.put(("event", worker, ready))
        eventually(lambda: owner._pending_ve_prewarm.phase == "starting")
    if phase in ("capturing", "detaching"):
        started_at = owner._pending_ve_prewarm.start_sent_at
        started = VePrewarmStarted("warmup", worker.generation, 1, warmup.signature, started_at)
        owner._inbox.put(("event", worker, RecordingEvent(
            worker.generation, "warmup", VE_PREWARM_STARTED, started)))
    if phase == "detaching":
        progress = VePrewarmProgress("warmup", worker.generation, 1, warmup.signature,
                                     started_at, warmup.frames_per_channel, owner._clock())
        for kind in (VE_PREWARM_PROGRESS, VE_PREWARM_DETACHING):
            owner._inbox.put(("event", worker, RecordingEvent(worker.generation, "warmup", kind, progress)))
    eventually(lambda: owner._pending_ve_prewarm.phase == phase)
    event = (RecordingEvent(worker.generation, "probe", SOUNDCARD_PREPARE, SoundcardPrepareRequest("probe"))
             if fault == "unexpected_command" else ready)
    owner._inbox.put(("event", worker, event))

    eventually(lambda: completions, timeout=4)
    failure, = completions
    assert not failure.success and failure.stage == "protocol" and failure.ownership_safe
    assert detail in failure.detail
    (tmp_path / "release-factory").touch()
    eventually(lambda: owner.generation == 2 and owner.worker_status.state == "ready")
    assert owner._worker is not worker and not owner.busy


def test_retiring_process_cleanup_does_not_block_start_admission(service, tmp_path, monkeypatch):
    owner = service()
    owner.ensure_resident()
    eventually(lambda: owner.worker_status.state == "ready")
    process = owner._worker.process
    close = process.close
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    failures = []

    def gated_close():
        entered.set()
        assert release.wait(8)
        close()

    def start():
        try:
            owner.start(request(tmp_path))
        except RuntimeError as exc:
            failures.append(str(exc))
        finally:
            returned.set()

    monkeypatch.setattr(process, "close", gated_close)
    process.terminate()
    caller = threading.Thread(target=start, daemon=True)
    try:
        assert entered.wait(8)
        caller.start()
        assert returned.wait(2), "start waited for process cleanup"
        assert len(failures) == 1 and "busy" in failures[0]
    finally:
        release.set()
        if caller.ident is not None:
            caller.join(3)


def test_in_flight_probe_excludes_hardware_and_shutdown_clears_it(service, tmp_path):
    from base.recording_process_protocol import SoundcardPrepareRequest

    inventory(tmp_path, soundcard())
    owner = service(options={"block_query": True}, shutdown_timeout=.1, terminate_timeout=.2)
    owner.ensure_resident()
    results = []
    assert owner.prepare_soundcard(SoundcardPrepareRequest("pending"), results.append) == "accepted"
    eventually(lambda: (tmp_path / "soundcard.jsonl").exists()
               and any(op == "query-blocked" for op, _ in trace(tmp_path)))
    assert owner.busy and not owner.can_start_recording
    assert owner.release_ve() == "busy"
    ve_request = capture_request(tmp_path / "ve.wav")
    warmup = VePrewarmRequest.create("warmup", ve_request.device, ve_request.channels,
                                     ve_request.sample_rate, attempt=1)
    assert owner.prewarm_ve(warmup, None) == "busy"
    with pytest.raises(RuntimeError, match="busy"):
        owner.start(request(tmp_path))
    assert owner.prepare_soundcard(SoundcardPrepareRequest("other"), None) == "busy"
    owner.shutdown()
    assert owner.closed.wait(8)
    assert owner.generation == 1 and owner.worker_pid is None
    assert not results


def test_old_generation_result_cannot_complete_current_probe(service, tmp_path):
    from base.recording_process_protocol import (
        SOUNDCARD_PREPARE_RESULT, SoundcardPrepareRequest, SoundcardPrepareResult,
    )

    inventory(tmp_path, soundcard())
    owner = service(options={"block_query": True}, retry_delay=.1,
                    shutdown_timeout=.1, terminate_timeout=.2)
    owner.ensure_resident()
    eventually(lambda: owner.worker_status.state == "ready")
    owner._worker.process.terminate()
    eventually(lambda: owner.worker_status.state == "ready" and owner.generation == 2)
    results = []
    assert owner.prepare_soundcard(SoundcardPrepareRequest("new"), results.append) == "accepted"
    eventually(lambda: (tmp_path / "soundcard.jsonl").exists()
               and any(op == "query-blocked" for op, _ in trace(tmp_path)))
    worker = owner._worker
    stale = SoundcardPrepareResult("new", 1, detail="stale")
    owner._inbox.put(("event", worker, RecordingEvent(1, "new", SOUNDCARD_PREPARE_RESULT, stale)))
    sentinel = threading.Event()
    owner._inbox.put(("release_callback", lambda *_: sentinel.set(), "released", ()))
    assert sentinel.wait(3)
    assert not results and owner.busy and not worker.retiring


def test_refresh_failure_retires_uncertain_native_generation(service, tmp_path):
    inventory(tmp_path, soundcard())
    owner = service(retry_delay=.1)
    owner.ensure_resident()
    assert probe(owner, "first").success
    old_pid = owner.worker_pid
    (tmp_path / "fail-refresh").touch()
    failed = probe(owner, "refresh-failure")
    assert not failed.success and "injected PortAudio refresh failure" in failed.detail
    eventually(lambda: owner.worker_status.state == "ready" and owner.generation == 2)
    assert trace(tmp_path) == [[op, old_pid] for op in ("load", "terminate", "initialize")]
    assert probe(owner, "after-recovery").success


def test_bridge_delivers_worker_and_probe_status_on_qt_thread(service, tmp_path, qt_app):
    from PyQt5.QtCore import QThread
    from base.recording_process_protocol import SoundcardPrepareRequest
    from ui.recording_service_bridge import RecordingServiceBridge

    inventory(tmp_path, soundcard())
    bridge = RecordingServiceBridge(service())
    statuses, results = [], []
    bridge.worker_status_changed.connect(lambda value: statuses.append((QThread.currentThread(), value)))
    bridge.ensure_resident()
    assert bridge.prepare_soundcard(SoundcardPrepareRequest("qt"),
                                    lambda value: results.append((QThread.currentThread(), value))) == "accepted"

    def deliver():
        qt_app.processEvents()
        return results and any(value.state == "ready" for _, value in statuses)

    eventually(deliver)
    assert all(thread is bridge.thread() for thread, _ in [*statuses, *results])
    assert results[0][1].success
