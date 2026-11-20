"""Compare verdict/terminal delivery using phase-6 inputs and production paths.

The optional legacy root is a read-only source snapshot, never an installed
alternate implementation. Run modes sequentially to avoid CPU/disk contention.
"""
import argparse
import ctypes
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from time import perf_counter, sleep

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from unit_test.performance.benchmark_analysis_pipeline import assert_result, distribution, payload_for


def private_mib(pid):
    """Windows private bytes without adding a benchmark dependency."""
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in (
                "peak_working", "working", "peak_paged", "paged", "peak_nonpaged",
                "nonpaged", "pagefile", "peak_pagefile", "private")]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x410, False, pid)
    if not handle:
        return None
    try:
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        query.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        if not query(handle, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.private / 1024**2
    finally:
        kernel.CloseHandle(handle)


def load_snapshot(root):
    import base.analysis
    for name in ("base.analysis_csv_exporter", "base.analysis.storage", "base.analysis.executor", "base.analysis.worker"):
        spec = importlib.util.spec_from_file_location(name, Path(root) / (name.replace(".", "/") + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        parent, attribute = name.rsplit(".", 1)
        setattr(sys.modules[parent], attribute, module)


def snapshot_worker(connection, station_id, generation):
    load_snapshot(os.environ["ANALYSIS_BENCHMARK_SNAPSHOT"])
    from base.analysis.worker import analysis_worker_main
    analysis_worker_main(connection, station_id, generation)


def signature(result):
    result = {**result, "task_id": "fixed"}
    for item in result["instance_results"]:
        item["artifacts"] = [{**artifact, "path": Path(artifact["path"]).name}
                             for artifact in item["artifacts"]]
    return hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()


def direct(args, paths):
    from base.analysis.input_adapter import build_analysis_batch
    from base.analysis.executor import execute_task
    task, = build_analysis_batch(payload_for(paths[:1], args.output, args.scenario, 0,
                                            "time_varying_iso532_1")).tasks
    rows, signatures = [], set()
    with ThreadPoolExecutor(max_workers=1) as writer:
        for index in range(args.repetitions + 1):
            early = []
            start = perf_counter()
            options = {} if args.mode == "legacy" else {"on_judgement_ready": lambda _: early.append(perf_counter())}
            if args.mode == "parallel":
                options["save_executor"] = writer
            result = execute_task(task, **options)
            end = perf_counter()
            assert_result(result)
            signatures.add(signature(result))
            del result
            gc.collect()
            if index:
                rows.append({"verdict_ms": ((early[0] if early else end) - start) * 1000,
                             "terminal_ms": (end - start) * 1000,
                             "private_mib": private_mib(os.getpid())})
            print(f"{args.mode}: {index}/{args.repetitions}", flush=True)
    assert len(signatures) == 1
    hashes = {str(path.relative_to(args.output / "exports")): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (args.output / "exports").rglob("*.csv")}
    return {"mode": args.mode, "samples": rows, "signature": signatures.pop(), "csv": hashes,
            **{key: distribution([row[key] for row in rows]) for key in ("verdict_ms", "terminal_ms", "private_mib")}}


def stations(args, paths):
    from base.analysis.process_manager import AnalysisProcessManager
    from base.analysis.service import AnalysisService
    managers = []

    def make_manager(sid, **kwargs):
        if args.legacy_root:
            kwargs["worker_target"] = snapshot_worker
        manager = AnalysisProcessManager(sid, **kwargs)
        managers.append(manager)
        return manager

    service = AnalysisService(task_timeout=180, drain_timeout=180, manager_factory=make_manager)
    ids = [f"station-{index}" for index in range(len(paths))]
    started, verdicts, ended, latencies, memory = {}, {}, set(), [], []
    pids, submitted = {}, {}

    def poll_until(predicate):
        deadline = perf_counter() + 240
        while not predicate():
            assert perf_counter() < deadline, service.station_states
            for event in service.poll_events():
                tid, kind, now = event["task_id"], event["event_type"], perf_counter()
                if kind == "task_started":
                    assert tid not in started
                    started[tid] = now
                elif kind == "judgement_ready":
                    assert tid not in verdicts and tid not in ended
                    verdicts[tid] = now
                elif kind == "task_result":
                    assert tid not in ended
                    ended.add(tid)
                    assert_result(event["payload"])
                    latencies.append({"task_id": tid, "station": event["station_id"],
                        "verdict_ms": (verdicts.get(tid, now) - submitted[tid]) * 1000,
                        "terminal_ms": (now - submitted[tid]) * 1000,
                        "queue_ms": (started[tid] - submitted[tid]) * 1000})
            sleep(0.005)

    try:
        service.sync_stations(ids)
        poll_until(lambda: all(s["state"] == "ready" for s in service.station_states.values()))
        pids = {sid: service.station_states[sid]["pid"] for sid in ids}
        for index in range(args.repetitions + 1):
            payload = payload_for(paths, args.output, args.scenario, 0, "time_varying_iso532_1", index, burst=2)
            payload["conditions"] = [{"condition_id": direction, "segmented_analysis": {"mode": "none"}}
                                     for direction in ("forward", "reverse")]
            for position, task in enumerate(payload["tasks"]):
                task["condition_id"] = "forward" if position % 2 == 0 else "reverse"
            target = len(ended) + len(payload["tasks"])
            now = perf_counter()
            submitted.update({task["task_id"]: now for task in payload["tasks"]})
            assert service.submit_batch(payload)["accepted"]
            poll_until(lambda: len(ended) == target)
            assert service.pending_task_count == 0
            assert {sid: service.station_states[sid]["pid"] for sid in ids} == pids
            memory.append({**{sid: private_mib(pid) for sid, pid in pids.items()},
                           "host": private_mib(os.getpid())})
            print(f"stations: {index}/{args.repetitions}, completed={len(ended)}", flush=True)
    finally:
        service.shutdown()
        poll_until(lambda: service.closed)
    # closed is set only after join(), process.close() and IPC thread cleanup.
    # Windows can still allow OpenProcess on an exited PID held by another handle.
    assert all(manager.closed and manager.pid is None for manager in managers)
    if not args.legacy_root:
        assert set(verdicts) == ended
    return {"latencies": latencies, "idle_private_mib": memory,
            "completed": len(ended), "early_messages": len(verdicts), "workers_reaped": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("legacy", "sync", "parallel", "stations"), required=True)
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument("--input-index", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--scenario", default="ALL_STFT")
    args = parser.parse_args()
    if args.repetitions < 1 or args.input_index < 0:
        parser.error("repetitions must be positive and input-index must be nonnegative")
    if args.mode == "legacy" and args.legacy_root is None:
        parser.error("legacy mode requires a source snapshot via --legacy-root")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.legacy_root:
        os.environ["ANALYSIS_BENCHMARK_SNAPSHOT"] = str(args.legacy_root.resolve())
        if args.mode != "stations":
            load_snapshot(args.legacy_root)
    paths = [Path(item["path"]) for item in json.loads(args.inputs.read_text(encoding="utf-8"))["inputs"]]
    result = stations(args, paths) if args.mode == "stations" else direct(args, paths[args.input_index:])
    (args.output / "measurements.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
