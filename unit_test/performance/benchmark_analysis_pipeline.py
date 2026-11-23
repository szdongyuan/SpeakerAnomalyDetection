"""Headless phase-6 measurements using the production executor and resident service.

Run each task scenario in a fresh interpreter, sequentially. Numerical tracing
and retention checks are separate from timing. No benchmark hooks enter product code.
"""
import argparse
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import replace
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import sys
from time import perf_counter, sleep
import tracemalloc
from unittest.mock import patch
import weakref

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def distribution(values):
    import numpy as np
    return {"n": len(values), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "min": min(values), "max": max(values)}


def payload_for(paths, output, scenario, channels, method, sequence=0, burst=1):
    import soundfile as sf
    kinds = {
        "LOUD": ["LOUD"], "SHRP": ["SHRP"], "ROUGH": ["ROUGH"],
        "LOUD_SHRP": ["LOUD", "SHRP"], "SHRP_LOUD": ["SHRP", "LOUD"],
        "SHRP_ROUGH": ["SHRP", "ROUGH"],
        "ALL_STFT": ["SPL", "FFT", "FBA", "Spec", "LOUD", "SHRP", "ROUGH"],
        "ALL_CQT": ["SPL", "FFT", "FBA", "Spec", "LOUD", "SHRP", "ROUGH"],
    }[scenario]
    window = {"analysis_time_range_enabled": True, "analysis_start_time_sec": 0.0,
              "analysis_end_time_sec": 3.0}
    stations, tasks = [], []
    for number, path in enumerate(paths):
        count = sf.info(path).channels
        sid = f"station-{number}"
        stations.append({"station_id": sid, "channels": [
            {"channel_name": f"CH{i + 1}", "signal_type": "sound_pressure"}
            for i in range(count)]})
        selected = list(range(count)) if channels == 0 else list(range(min(count, channels)))
        config = {"display_sequence": kinds}
        for kind in kinds:
            parameters = {"type": kind, "analysis_channels": selected, "limit_checked": kind != "Spec",
                          "limit_mode": "manual", "manual_input_mode": "constant",
                          "constant_upper_enabled": True, "constant_upper_value": 100000.0,
                          "constant_lower_enabled": False}
            if kind in {"LOUD", "SHRP", "ROUGH"}:
                parameters.update(advanced=dict(window), method=method if kind != "ROUGH" else "daniel_weber")
            else:
                parameters.update(window)
            if kind == "SPL":
                parameters.update(weighting="A", limit_metric="curve_y")
            if kind in {"FFT", "Spec"}:
                parameters.update(n_fft=1024, hop_length=256)
            if kind == "Spec":
                parameters["freq_scale_type"] = "log" if scenario == "ALL_CQT" else "linear"
            config[kind] = parameters
        for pending in range(burst):
            tasks.append({"task_id": f"run-{sequence}-{pending}-{sid}", "station_id": sid,
                          "condition_id": "condition", "wav_path": str(path),
                          "analysis_result_directory": str(output / "exports" / sid),
                          "analysis_config": config})
    return {"stations": stations, "conditions": [{"condition_id": "condition",
             "segmented_analysis": {"mode": "none"}}], "tasks": tasks}


def assert_result(result):
    assert result["execution_status"] == "分析完成", [(i["config_key"], i["error"]) for i in result["instance_results"]]
    assert all(a["status"] == "saved" for i in result["instance_results"] for a in i["artifacts"])


def measure_task(task, reuse):
    from base.analysis import executor
    from base.analysis_compute_context import TaskComputeContext
    from base.core_algorithm.sound_quality.loudness import LoudnessAnalyzer
    elapsed = defaultdict(float)
    calls = defaultdict(int)
    contexts = []

    def timed(name, function):
        def call(*args, **kwargs):
            started = perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                elapsed[name] += (perf_counter() - started) * 1000
                calls[name] += 1
        return call

    def context_factory():
        context = TaskComputeContext()
        contexts.append(context)
        return context

    with ExitStack() as patches:
        for name, stage in [("load_task_wav", "read_ms"), ("calculate_analysis_instance", "algorithm_ms"),
                            ("_judge", "judgement_ms"), ("plain_data", "plain_result_ms"),
                            ("save_instance_artifacts", "save_ms")]:
            patches.enter_context(patch.object(executor, name, timed(stage, getattr(executor, name))))
        patches.enter_context(patch.object(LoudnessAnalyzer, "compute", timed("loudness_raw_ms", LoudnessAnalyzer.compute)))
        patches.enter_context(patch.object(executor, "TaskComputeContext", context_factory))
        started = perf_counter()
        result = executor.execute_task(task, reuse_computations=reuse)
        elapsed["task_ms"] = (perf_counter() - started) * 1000
    assert_result(result)
    started = perf_counter()
    serialized = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    elapsed["pickle_ms"] = (perf_counter() - started) * 1000
    return {**elapsed, "raw_calls": calls["loudness_raw_ms"], "pickle_bytes": len(serialized),
            "cache": [{"hits": c.hits, "misses": c.misses, "computes": c.compute_count,
                       "lookup_us": c.lookup_seconds * 1e6, "compute_ms": c.compute_seconds * 1000,
                       "retained_entries": len(c)} for c in contexts]}, result


def task_benchmark(task, repetitions):
    # First shared execution includes first use of numerical paths in this interpreter.
    # It is not an empty-Numba-cache measurement.
    first, _ = measure_task(task, True)
    warmup, _ = measure_task(task, False)
    runs = {"shared": [], "independent": []}
    for repetition in range(repetitions):
        results = {}
        for reuse in ((True, False) if repetition % 2 == 0 else (False, True)):
            record, results[reuse] = measure_task(task, reuse)
            runs["shared" if reuse else "independent"].append(record)
        assert results[True] == results[False]
    summary = {mode: {key: distribution([r[key] for r in records]) for key in records[0]
                      if key.endswith("_ms")} for mode, records in runs.items()}
    return {"first_shared": first, "independent_warmup": warmup, "runs": runs,
            "summary": summary, "outputs_identical": True}


def memory_benchmark(task, repetitions):
    import psutil
    from base.analysis import executor
    from base.analysis_compute_context import TaskComputeContext
    from base.core_algorithm.sound_quality.loudness import LoudnessAnalyzer
    process = psutil.Process()
    # Warm all numerical paths and allocator pools before measuring idle growth.
    for _ in range(3):
        assert_result(executor.execute_task(task))
    raw_compute = LoudnessAnalyzer.compute
    raw_refs, context_refs, rows = [], [], []

    def observed_compute(self, *args, **kwargs):
        raw = raw_compute(self, *args, **kwargs)
        raw_refs.extend((weakref.ref(raw), weakref.ref(raw.specific_loudness)))
        return raw

    def observed_context():
        context = TaskComputeContext()
        context_refs.append(weakref.ref(context))
        return context

    with patch.object(LoudnessAnalyzer, "compute", observed_compute), patch.object(executor, "TaskComputeContext", observed_context):
        for index in range(repetitions):
            result = executor.execute_task(replace(task, task_id=f"memory-{index}"))
            assert_result(result)
            del result
            gc.collect()
            memory = process.memory_info()
            rows.append({"task": index, "rss": memory.rss, "private": memory.private,
                         "live_raw_refs": sum(r() is not None for r in raw_refs),
                         "live_context_refs": sum(r() is not None for r in context_refs)})
            assert rows[-1]["live_raw_refs"] == rows[-1]["live_context_refs"] == 0
    peaks = {}
    for reuse in (False, True):
        gc.collect()
        tracemalloc.start()
        result = executor.execute_task(task, reuse_computations=reuse)
        _, peak = tracemalloc.get_traced_memory()
        del result
        gc.collect()
        current, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks[str(reuse)] = {"peak_bytes": peak, "after_release_bytes": current}
    return {"idle_samples": rows, "checked_weak_references": len(raw_refs) + len(context_refs),
            "tracemalloc_separate_runs": peaks}


def station_benchmark(paths, output, scenario, channels, method, repetitions):
    import psutil
    from base.analysis.service import AnalysisService
    service = AnalysisService(waiting_capacity=4, startup_timeout=90, task_timeout=120)
    ids = [f"station-{i}" for i in range(len(paths))]
    parent = psutil.Process()
    samples, latencies, rounds, pids = [], [], [], {}
    last_sample = 0.0
    seen, active, starts = set(), set(), {}
    max_active = 0

    def sample(force=False):
        nonlocal last_sample
        now = perf_counter()
        if not force and now - last_sample < 0.05:
            return
        last_sample = now
        values = {}
        for name, pid in [("parent", parent.pid), *pids.items()]:
            try:
                process = psutil.Process(pid)
                m = process.memory_info()
                values[name] = {"rss": m.rss, "private": m.private, "peak_rss": m.peak_wset,
                                "handles": process.num_handles()}
            except psutil.NoSuchProcess:
                pass
        samples.append({"elapsed_s": now - beginning, "processes": values})

    def poll_until(predicate, timeout=150):
        nonlocal max_active
        deadline = perf_counter() + timeout
        while not predicate():
            assert perf_counter() < deadline, service.station_states
            for event in service.poll_events():
                tid = event.get("task_id")
                if event["event_type"] == "task_started":
                    assert event["station_id"] not in active
                    active.add(event["station_id"])
                    max_active = max(max_active, len(active))
                    starts[tid] = perf_counter()
                elif event["event_type"] == "task_result":
                    assert tid not in seen
                    seen.add(tid)
                    assert_result(event["payload"])
                    active.remove(event["station_id"])
                    latencies.append({"task_id": tid, "station": event["station_id"],
                                      "service_ms": (perf_counter() - starts.pop(tid)) * 1000})
            sample()
            sleep(0.005)

    beginning = perf_counter()
    try:
        assert service.sync_stations(ids)["accepted"]
        poll_until(lambda: all(s["state"] == "ready" for s in service.station_states.values()))
        startup_ms = (perf_counter() - beginning) * 1000
        pids.update({sid: service.station_states[sid]["pid"] for sid in ids})
        for index in range(repetitions):
            payload = payload_for(paths, output, scenario, channels, method, index, burst=2)
            target = len(seen) + len(payload["tasks"])
            start = perf_counter()
            assert service.submit_batch(payload)["accepted"]
            poll_until(lambda: len(seen) == target)
            rounds.append({"round": index, "wall_ms": (perf_counter() - start) * 1000})
            assert service.pending_task_count == 0
            assert {sid: service.station_states[sid]["pid"] for sid in ids} == pids
            sample(True)
            rounds[-1]["idle_memory"] = samples[-1]["processes"]
            print(f"stations={len(paths)} round={index + 1}/{repetitions}", flush=True)
        # Queue final work then close immediately: accepted work must drain exactly once.
        final = payload_for(paths, output, scenario, channels, method, repetitions, burst=1)
        target = len(seen) + len(final["tasks"])
        assert service.submit_batch(final)["accepted"]
        service.shutdown()
        poll_until(lambda: service.closed)
        assert len(seen) == target and not active and not starts
    finally:
        service.shutdown()
        deadline = perf_counter() + 45
        while not service.closed and perf_counter() < deadline:
            service.poll_events()
            sleep(0.01)
    assert service.closed
    assert all(not psutil.pid_exists(pid) for pid in pids.values())
    return {"startup_ms": startup_ms, "completed": len(seen), "pids": pids,
            "max_simultaneously_active": max_active, "latencies": latencies, "rounds": rounds,
            "memory_samples": samples, "closed": True, "workers_reaped": True}


def overhead_benchmark(payload):
    from base.analysis import build_analysis_batch
    from base.analysis_compute_context import TaskComputeContext
    from base.audio_label_policy import required_judgements
    times = []
    for _ in range(500):
        started = perf_counter()
        batch = build_analysis_batch(payload)
        [required_judgements(task.instances) for task in batch.tasks]
        times.append((perf_counter() - started) * 1e6)
    count = 20000
    start = perf_counter()
    for _ in range(count):
        TaskComputeContext().clear()
    create_us = (perf_counter() - start) * 1e6 / count
    context = TaskComputeContext()
    key = ("raw", ("task", 0, 0, 132300), 44100, 1.0, "free")
    context.get_or_compute(key, object)
    start = perf_counter()
    for _ in range(count):
        context.get_or_compute(key, object)
    return {"batch_and_requirements_us": distribution(times), "tasks": len(batch.tasks),
            "instances": sum(len(t.instances) for t in batch.tasks),
            "create_clear_us": create_us, "cache_hit_us": (perf_counter() - start) * 1e6 / count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["task", "memory", "stations", "overhead"], required=True)
    parser.add_argument("--scenario", default="LOUD_SHRP")
    parser.add_argument("--method", default="time_varying_iso532_1")
    parser.add_argument("--stations", type=int, default=1)
    parser.add_argument("--channels", type=int, default=1, help="0 selects all WAV channels")
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    paths = [Path(i["path"]) for i in manifest["inputs"][:args.stations]]
    original_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    started = perf_counter()
    from base.analysis import build_analysis_batch
    if args.mode in {"task", "memory"}:
        from base.analysis import executor  # Numerical imports belong to workers in service mode.
    import soundfile as sf
    import numpy as np
    imports_ms = (perf_counter() - started) * 1000
    payload = payload_for(paths, args.output, args.scenario, args.channels, args.method)
    task = build_analysis_batch(payload).tasks[0]
    if args.mode == "task":
        result = task_benchmark(task, args.repetitions)
    elif args.mode == "memory":
        result = memory_benchmark(task, args.repetitions)
    elif args.mode == "stations":
        result = station_benchmark(paths, args.output, args.scenario, args.channels, args.method, args.repetitions)
    else:
        result = overhead_benchmark(payload)
    assert original_hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    source_paths = [*ROOT.glob("base/analysis/*.py"), ROOT / "base/analysis_compute_context.py",
                    ROOT / "base/core_algorithm/sound_quality/service.py", ROOT / "base/sound_quality_analysis_adapter.py"]
    inputs = []
    for p in paths:
        samples, rate = sf.read(p, frames=int(sf.info(p).samplerate * 3), always_2d=True)
        inputs.append({"sha256": original_hashes[str(p)], "rate": rate,
                       "file_seconds": sf.info(p).duration, "analysed_seconds": len(samples) / rate,
                       "channels": samples.shape[1], "rms": np.sqrt(np.mean(samples**2, axis=0)).tolist()})
    result.update(mode=args.mode, scenario=args.scenario, method=args.method, imports_ms=imports_ms,
                  parameters=payload, inputs=inputs, original_files_unchanged=True,
                  versions={name: importlib.metadata.version(name) for name in ["numpy", "scipy", "numba", "psutil"]},
                  python=platform.python_version(), thread_environment={name: os.environ.get(name) for name in
                  ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]},
                  source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    dump(args.output / "measurements.json", result)
    print(args.output.name, "completed", flush=True)


if __name__ == "__main__":
    main()
