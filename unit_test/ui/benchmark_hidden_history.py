"""Controlled recent-history callback benchmark; no capture, playback or PDF I/O.

Run this file with --help. Each source version runs in a separate Qt subprocess.
Only the baseline constructs the real (hidden) RecentSessionPanel. Business
append/update and synchronous group-result callbacks are inherited unmodified.
This contract excludes retired automatic-PDF readiness/items; historical benchmark
reports retain their original, broader PDF measurement contract.
"""
import argparse
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter_ns
from types import SimpleNamespace


def normalize(value):
    """Compare public values, including timestamps, regardless of frozen containers."""
    if isinstance(value, Mapping):
        return {key: normalize(item) for key, item in value.items()
                if key not in {"_recent_group_metadata", "analysis_report_state", "analysis_report_items"}}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


def verify_equivalent(before, after):
    assert normalize(before) == normalize(after), "baseline/current semantics differ"


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def worker(root, output, repetitions, baseline):
    sys.path.insert(0, str(root))
    # Redirect before any application import can create a log handler. Working
    # directory and bytecode are also isolated by the parent process.
    from consts import running_consts
    running_consts.LOG_DIR = str(output.parent / "logs")
    for name, config in running_consts.LOG_MAPPING.items():
        config["log_name"] = str(output.parent / "logs" / (name + ".log"))

    from PyQt5.QtCore import QCoreApplication, QEvent, QT_VERSION_STR, PYQT_VERSION_STR
    from PyQt5.QtWidgets import QApplication, QWidget
    from ui.sequence import sequence_widget_analysis_ops as analysis

    app = QApplication.instance() or QApplication([])
    panel_class = None
    if baseline:
        from ui.sequence.recent_session_panel import RecentSessionPanel
        panel_class = RecentSessionPanel

    class Clock(datetime):
        instant = datetime(2026, 10, 9, 12)

        @classmethod
        def now(cls):
            return cls.instant

    analysis.datetime = Clock

    class Host(analysis.SequenceWidgetAnalysisOpsMixin):
        def __init__(self):
            self.recent_test_sessions = []
            self.recent_test_session_by_id = {}
            self._recent_session_max_items = 20
            self._recent_session_seq = 0
            self._condition_record_cache = {}
            self._manual_product_condition_group_id = "product-A"
            self._manual_product_condition_results = {}
            self._manual_product_condition_completed_keys = set()
            self.count_board = SimpleNamespace(mode="test")
            self.left_panel = None
            self.product_test_condition_configs = [
                {"key": f"{i:02}", "condition_name": f"Condition {i}"}
                for i in range(1, 17)]
            self.sequence_config = [{"seq1": {"sample_rate": 48000,
                "channels": [1, 2], "conditions": self.product_test_condition_configs}}]
            self.analysis_config = {"display_sequence": ["SPL", "FFT"],
                "FFT": {"window": "hann", "size": 4096}, "SPL": {"unit": "Pa"}}
            self._active_input_channels = [1, 2]
            self.using_config_path = "synthetic-config.json"
            self.data_struct = SimpleNamespace(sample_rate=48000,
                analysis_result_dict={"SPL": (True, 0.2)})
            self.lineedit_s_or_n = SimpleNamespace(text=lambda: "SYNTHETIC")
            self.lineedit_type = SimpleNamespace(text=lambda: "Model-16")
            self.panel_counts = Counter()
            if panel_class:
                self.recent_session_panel = panel_class(
                    condition_configs=self.product_test_condition_configs)
                self.panel_counts["construction"] += 1
                # Wrappers only count calls; all real implementations run.
                for name in ("upsert_session", "remove_session", "_populate_group_row"):
                    original = getattr(self.recent_session_panel, name)
                    def counted(*args, _original=original, _name=name, **kwargs):
                        self.panel_counts[_name] += 1
                        return _original(*args, **kwargs)
                    setattr(self.recent_session_panel, name, counted)

        def prepare(self, index):
            # Four older records, then a complete 16-condition product. The
            # 21st record retests A/01 and evicts B/01.
            group = "product-B" if index < 4 else "product-A"
            key = f"{index + 1:02}" if index < 4 else f"{(index - 4) % 16 + 1:02}"
            self._current_cycle_recorded_count = group
            self._current_trigger_direction = key
            self._active_product_condition_config = next(
                item for item in self.product_test_condition_configs if item["key"] == key)
            self.recorded_path = f"synthetic/{index:02}.wav"
            self.recorded_signal_info = {"file_path": self.recorded_path,
                "barcode": f"SN-{group}-{index}", "labels": "OK", "sample_number": index}
            Clock.instant = datetime(2026, 10, 9, 12) + timedelta(seconds=index)

        def semantics(self):
            return normalize({
                "order": self.recent_test_sessions,
                "records": self.recent_test_session_by_id,
                "current": self._current_recent_session_id,
                "groups": {group: {
                    "data": self._collect_product_condition_records(group),
                    "result": self._product_group_result_state(group),
                } for group in ("product-A", "product-B")},
            })

        def widgets(self):
            panel = getattr(self, "recent_session_panel", None)
            if panel is None:
                assert not any(type(widget).__name__ == "RecentSessionPanel"
                               for widget in app.allWidgets())
                return {"total": 0, "by_type": {}}
            widgets = [panel, *panel.findChildren(QWidget)]
            return {"total": len(widgets),
                    "by_type": dict(Counter(type(widget).__name__ for widget in widgets))}

    def drain():
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()

    def trial():
        host = Host()
        for index in range(19):
            host.prepare(index)
            host._append_recent_session_from_current_run("OK")
            drain()
        timings, states, widget_counts = {}, {}, {}
        for index in (19, 20):
            host.prepare(index)
            start = perf_counter_ns()
            host._append_recent_session_from_current_run("OK")
            timings[f"append_{index + 1}"] = (perf_counter_ns() - start) / 1e6
            assert len(host.recent_test_sessions) == len(host.recent_test_session_by_id) == 20
            if index == 20:
                assert "recent_000001" not in host.recent_test_session_by_id
            states[f"append_{index + 1}"] = host.semantics()
            drain()
            widget_counts[f"append_{index + 1}"] = host.widgets()
            fields = {"result_label": "ng", "recorded_signal_info": dict(
                host.recorded_signal_info, labels="NG"),
                "analysis_result_dict": {"SPL": (False, 0.5)}}
            start = perf_counter_ns()
            host._update_recent_session(host._current_recent_session_id, **fields)
            timings[f"update_{index + 1}"] = (perf_counter_ns() - start) / 1e6
            states[f"update_{index + 1}"] = host.semantics()
            assert host._product_group_result_state("product-A") == (True, "NG")
            drain()
            widget_counts[f"update_{index + 1}"] = host.widgets()
        counts = {name: host.panel_counts[name] for name in
                  ("construction", "upsert_session", "remove_session", "_populate_group_row")}
        if panel_class:
            host.recent_session_panel.close()
            host.recent_session_panel.deleteLater()
            drain()
        return dict(timings_ms=timings, semantics=states, widgets=widget_counts, calls=counts)

    # Untimed validation/warmup runs before the measured samples.
    warmup = trial()
    samples = [trial() for _ in range(repetitions)]
    for sample in samples:
        verify_equivalent(warmup["semantics"], sample["semantics"])
    modules = ("ui/sequence/sequence_widget_analysis_ops.py",
               "ui/sequence/product_condition_result_ops.py")
    if baseline:
        modules += ("ui/sequence/recent_session_panel.py",)
    result = dict(root=str(root), head=git(root, "rev-parse", "HEAD"),
        python=platform.python_version(), qt=QT_VERSION_STR, pyqt=PYQT_VERSION_STR,
        platform=platform.platform(), conditions=16, retention=20,
        source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                       for name in modules},
        semantics=warmup["semantics"], samples=samples)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    from base.log_manager import LogManager
    assert LogManager.shutdown_all(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--current", type=Path)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--with-panel", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repetitions < (0 if args.worker else 1):
        parser.error("repetitions must be positive")
    if args.worker:
        worker(args.worker.resolve(), args.output.resolve(), args.repetitions, args.with_panel)
        return
    if not (args.baseline and args.current and args.checkpoint):
        parser.error("--baseline, --current and --checkpoint are required")
    roots = {name: getattr(args, name).resolve() for name in ("baseline", "current")}
    # Fail closed if the readonly baseline code differs from its recorded checkpoint.
    git(roots["baseline"], "diff", "--exit-code", args.checkpoint, "--", "ui", "base", "consts")
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONDONTWRITEBYTECODE="1")
    script = Path(__file__).resolve()
    # Validate separate source processes before starting repeated measurements.
    for phase, repetitions in (("validation", 0), ("measurement", args.repetitions)):
        results = {}
        for name, root in roots.items():
            folder = args.output.resolve() / phase / name
            folder.mkdir(parents=True)
            command = [sys.executable, "-B", str(script), "--worker", str(root),
                       "--output", str(folder / "raw.json"), "--repetitions", str(repetitions)]
            if name == "baseline":
                command.append("--with-panel")
            with (folder / "process.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=folder, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            results[name] = json.loads((folder / "raw.json").read_text(encoding="utf-8"))
        verify_equivalent(results["baseline"]["semantics"], results["current"]["semantics"])
    summary = {"equivalent": True, "checkpoint": args.checkpoint,
               "repetitions": args.repetitions, "timings_ms": {}, "structural": {}}
    for name, result in results.items():
        summary["timings_ms"][name] = {}
        for stage in result["samples"][0]["timings_ms"]:
            values = [sample["timings_ms"][stage] for sample in result["samples"]]
            summary["timings_ms"][name][stage] = dict(
                median=statistics.median(values), minimum=min(values), maximum=max(values))
        summary["structural"][name] = {key: result["samples"][0][key]
                                        for key in ("widgets", "calls")}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
