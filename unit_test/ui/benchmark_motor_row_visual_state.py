"""Real Qt group callback benchmark (standalone, never collected by pytest).

Example: python benchmark_motor_row_visual_state.py --baseline-root <archive>
    --current-root . --output-dir <new-directory>
Each source runs sequentially in its own process, with one warmup and three
measured sequences per N/K/mode. No recording, analysis, or database I/O occurs.
"""
import argparse
from collections import Counter
from collections.abc import Mapping
from contextlib import ExitStack
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter_ns, process_time_ns
from types import SimpleNamespace
from unittest.mock import patch


BASELINE = "e8d4b3bab29a2778faa18e64476cfbf37c7bf342"
OPERATIONS = ("setStyleSheet", "setProperty", "polish", "unpolish", "update")


def plain(value):
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, set):
        return sorted(value)
    return value


def save(path, value):
    path.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2), encoding="utf-8")


def verify_target(result):
    """Structural regression gate, deliberately usable against the old source."""
    for scenario in result["scenarios"]:
        for sample in scenario["samples"]:
            for stage, measurement in sample.items():
                calls = measurement["calls"]
                for kind in ("button", "label"):
                    assert calls[kind]["setStyleSheet"] == 0, (scenario["name"], stage, kind, calls)
                    if stage == "unchanged" or scenario["mode"] == "test":
                        assert all(value == 0 for value in calls[kind].values()), calls
                expected = {"unchanged": 0, "one_changed": 1, "multiple_changed": 5}[stage]
                if scenario["mode"] == "test":
                    expected = 0
                assert all(value == 0 for value in calls["button"].values()), calls
                for operation in ("setProperty", "polish", "unpolish", "update"):
                    assert calls["label"][operation] == expected, (stage, operation, calls)


def verify_archive(baseline, current):
    """Validate every tracked Python source in the benchmark's application trees."""
    listing = subprocess.check_output([
        "git", "-C", str(current), "ls-tree", "-r", BASELINE, "--", "ui", "base", "consts"
    ], text=True)
    checked = 0
    for line in listing.splitlines():
        metadata, name = line.split("\t", 1)
        if not name.endswith(".py"):
            continue
        content = (baseline / name).read_bytes()
        # Windows git archive may apply core.autocrlf. Permit only that
        # byte-level conversion; all source content must match the commit.
        candidates = (content, content.replace(b"\r\n", b"\n"))
        hashes = {hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
                  for data in candidates}
        assert metadata.split()[2] in hashes, name
        checked += 1
    assert checked > 0
    return checked


def worker(root, output, sizes, completed_counts, modes, require_target):
    sys.path.insert(0, str(root))
    from consts import running_consts
    running_consts.LOG_DIR = str(output.parent / "logs")
    for name, config in running_consts.LOG_MAPPING.items():
        config["log_name"] = str(output.parent / "logs" / (name + ".log"))

    from PyQt5.QtCore import QCoreApplication, QEvent, Qt, QT_VERSION_STR, PYQT_VERSION_STR
    from PyQt5.QtWidgets import QApplication
    from ui.sequence import sequence_widget_analysis_ops as analysis
    from ui.sequence.motor_result_panel import MotorResultPanel

    app = QApplication.instance() or QApplication([])

    class Clock(datetime):
        @classmethod
        def now(cls):
            return cls(2026, 10, 10, 12)

    analysis.datetime = Clock

    def drain():
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()

    class Host(analysis.SequenceWidgetAnalysisOpsMixin):
        def __init__(self, size, completed, mode):
            self.recent_test_sessions = []
            self.recent_test_session_by_id = {}
            self._recent_session_max_items = 20
            self._recent_session_seq = 0
            self._condition_record_cache = {}
            self._manual_product_condition_group_id = "round-1"
            self._current_cycle_recorded_count = "round-1"
            self._manual_product_condition_results = {}
            self.count_board = SimpleNamespace(mode=mode)
            self.analysis_config = {"display_sequence": ["SPL"],
                                    "SPL": {"type": "SPL", "channels": [0, 1]}}
            self.product_test_condition_configs = [
                {"key": f"group_{i // 20 + 1}:condition_{i % 20 + 1}",
                 "group_name": f"Port {i // 20 + 1}", "condition_name": f"Condition {i + 1}",
                 "analysis_list": self.analysis_config} for i in range(size)]
            self.sequence_config = [{"seq1": {
                "acq": {"mode": "RECORD_ONLY", "detail": {"sample_rate": 44100, "record_time": 1}},
                "analysis_list": self.analysis_config}}]
            self._active_input_channels = [0, 1]
            self.using_config_path = "synthetic-config.json"
            self.data_struct = SimpleNamespace(sample_rate=44100, analysis_result_dict={})
            self.lineedit_s_or_n = SimpleNamespace(text=lambda: "SYNTHETIC")
            self.lineedit_type = SimpleNamespace(text=lambda: "Model")
            self.left_panel = MotorResultPanel(
                condition_configs=self.product_test_condition_configs, queue_catalog={})
            self.left_panel.setAttribute(Qt.WA_DontShowOnScreen)
            self.left_panel.resize(600, 800)
            self.left_panel.show()
            # Accumulated keys are independent of retained history. Exercise the
            # real completion method for the last key after seeding prior runs.
            self._manual_product_condition_completed_keys = {
                item["key"] for item in self.product_test_condition_configs[:completed - 1]}
            self.prepare(completed - 1)
            self._mark_manual_product_condition_recording_completed()
            assert len(self._manual_product_condition_completed_keys) == completed
            # 21 real appends prove eviction without performing K*N setup syncs.
            for index in [completed - 20, *range(completed - 20, completed)]:
                self.prepare(index)
                self._append_recent_session_from_current_run("not_labeled")
            assert len(self.recent_test_sessions) == len(self.recent_test_session_by_id) == 20
            assert "recent_000001" not in self.recent_test_session_by_id
            assert len(self._manual_product_condition_completed_keys) == completed
            assert len(self._manual_product_group_raw_results("round-1")) == completed
            assert len(self.left_panel.rows) == size
            self._update_current_recent_session_result("not_labeled")
            self.left_panel.select_condition(self.product_test_condition_configs[0]["key"], show_detail=True)
            # Nonempty channel state is included in equivalence snapshots and
            # protected from placeholder updates by the real panel behavior.
            first = self.product_test_condition_configs[0]["key"]
            self.left_panel.set_condition_channel_results(first, [
                {"raw_channel": 0, "result": "OK", "SPL": "OK"},
                {"raw_channel": 1, "result": "OK", "SPL": "OK"}])
            self.left_panel.set_condition_result(first, "OK", "ok")

        def prepare(self, index):
            condition = self.product_test_condition_configs[index]
            self._active_product_condition_key = condition["key"]
            self._current_trigger_direction = condition["key"]
            self._active_product_condition_config = condition
            self.recorded_path = f"synthetic/{index:03d}.wav"
            self.recorded_signal_info = {"file_path": self.recorded_path, "barcode": "SYNTHETIC",
                                         "labels": "not_labeled", "active_input_channels": [0, 1]}

        def snapshot(self):
            panel = self.left_panel
            return plain({
                "rows": {key: {**{field: row[field] for field in (
                    "result", "tone", "channel_results", "completed_channels", "analysis_completed",
                    "analysis_channels", "runtime_details")},
                    "labels": {name: label.text() for name, label in row["labels"].items()},
                    "hidden": row["button"].isHidden()}
                    for key, row in panel.rows.items()},
                "selection": panel.selected_key, "viewed": panel.viewed_key,
                "port": panel.current_port, "detail_owner": panel._detail_owner_key,
                "detail_hidden": panel.detail_frame.isHidden(),
                "summary": {"port": panel.port_result_value.text(), "round": panel.round_result_value.text(),
                            "final": panel.final_value.text(), "automatic": panel.get_automatic_round_result()},
                "history_order": self.recent_test_sessions, "history": self.recent_test_session_by_id,
                "current_session": self._current_recent_session_id,
                "completed": self._manual_product_condition_completed_keys,
                "manual_results": self._manual_product_condition_results,
                "group": self._collect_product_condition_records("round-1")})

    def run_scenario(size, completed, mode):
        host = Host(size, completed, mode)
        panel = host.left_panel
        counts = {kind: Counter() for kind in ("button", "label")}
        widgets = {id(widget): kind for row in panel.rows.values()
                   for kind, widget in (("button", row["button"]), ("label", row["labels"]["result"]))}
        styles = {id(widget.style()): widget.style() for row in panel.rows.values()
                  for widget in (row["button"], row["labels"]["result"])}
        with ExitStack() as stack:
            for row in panel.rows.values():
                for kind, widget in (("button", row["button"]), ("label", row["labels"]["result"])):
                    for operation in ("setStyleSheet", "setProperty", "update"):
                        original = getattr(widget, operation)
                        def counted(*args, _original=original, _kind=kind, _operation=operation, **kwargs):
                            counts[_kind][_operation] += 1
                            return _original(*args, **kwargs)
                        stack.enter_context(patch.object(widget, operation, counted))
            for style in styles.values():
                for operation in ("polish", "unpolish"):
                    original = getattr(style, operation)
                    def counted_style(*args, _original=original, _operation=operation, **kwargs):
                        if args and id(args[0]) in widgets:
                            counts[widgets[id(args[0])]][_operation] += 1
                        return _original(*args, **kwargs)
                    stack.enter_context(patch.object(style, operation, counted_style))

            def measure():
                drain()  # Includes pending paints, deliberately outside timing.
                for counter in counts.values():
                    counter.clear()
                cpu_start = process_time_ns()
                wall_start = perf_counter_ns()
                host._update_current_recent_session_result(host.recorded_signal_info["labels"])
                wall_ms = (perf_counter_ns() - wall_start) / 1e6
                cpu_ms = (process_time_ns() - cpu_start) / 1e6
                calls = {kind: {operation: counter[operation] for operation in OPERATIONS}
                         for kind, counter in counts.items()}
                snapshot = host.snapshot()
                drain()
                return {"wall_ms": wall_ms, "cpu_ms": cpu_ms, "calls": calls, "semantics": snapshot}

            def trial():
                for record in host.recent_test_session_by_id.values():
                    record["result_label"] = "not_labeled"
                    record["recorded_signal_info"]["labels"] = "not_labeled"
                host.recorded_signal_info["labels"] = "not_labeled"
                # Restore changed rows through the real public panel API. A
                # neutral state first clears completed-analysis protection.
                for condition in host.product_test_condition_configs[completed - 5:completed]:
                    panel.set_condition_result(condition["key"], "待检测", "pending")
                    if mode == "mark":
                        panel.set_condition_result(condition["key"], "待判定", "pending")
                result = {"unchanged": measure()}
                host.recorded_signal_info["labels"] = "OK"
                result["one_changed"] = measure()
                for session_id in host.recent_test_sessions[1:5]:
                    record = host.recent_test_session_by_id[session_id]
                    record["result_label"] = "ng"
                    record["recorded_signal_info"]["labels"] = "NG"
                host.recorded_signal_info["labels"] = "NG"
                result["multiple_changed"] = measure()
                return result

            warmup = trial()
            samples = [trial() for _ in range(3)]
            for sample in samples:
                for stage in warmup:
                    assert sample[stage]["semantics"] == warmup[stage]["semantics"], stage
        panel.close()
        panel.deleteLater()
        drain()
        return {"name": f"{mode}-{size}-{completed}", "mode": mode, "rows": size,
                "completed": completed, "retained": 20, "warmup": warmup, "samples": samples}

    modules = ("ui/sequence/motor_result_panel.py", "consts/ui_style_const.py",
               "ui/sequence/sequence_widget_analysis_ops.py", "ui/sequence/product_condition_result_ops.py")
    result = {"root": str(root), "python": platform.python_version(), "qt": QT_VERSION_STR,
              "pyqt": PYQT_VERSION_STR, "platform": platform.platform(),
              "environment": {name: os.environ.get(name) for name in (
                  "QT_QPA_PLATFORM", "QT_QPA_FONTDIR", "TEMP", "TMP", "PYTHONHASHSEED")},
              "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in modules},
              "scenarios": []}
    for mode in modes:
        for size in sizes:
            for completed in completed_counts:
                result["scenarios"].append(run_scenario(size, completed, mode))
                save(output, result)
                print(result["scenarios"][-1]["name"], "saved", flush=True)
    from base.log_manager import LogManager
    assert LogManager.shutdown_all(5)
    if require_target:
        verify_target(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--current-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, choices=(100, 200))
    parser.add_argument("--completed", type=int, choices=(20, 60, 100))
    parser.add_argument("--mode", choices=("mark", "test"))
    parser.add_argument("--worker-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--require-target", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    sizes = [args.rows] if args.rows else [100, 200]
    completed_counts = [args.completed] if args.completed else [20, 60, 100]
    modes = [args.mode] if args.mode else ["mark", "test"]
    output = args.output_dir.resolve()
    if args.worker_root:
        output.mkdir(parents=True, exist_ok=True)
        worker(args.worker_root.resolve(), output / "raw.json", sizes, completed_counts, modes, args.require_target)
        return
    if not args.baseline_root or not args.current_root:
        parser.error("--baseline-root and --current-root are required")
    baseline, current = args.baseline_root.resolve(), args.current_root.resolve()
    checked = verify_archive(baseline, current)
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    results = {}
    for name, root in (("baseline", baseline), ("current", current)):
        folder = output / name
        folder.mkdir()
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker-root", str(root),
                   "--output-dir", str(folder)]
        for option in ("rows", "completed", "mode"):
            if getattr(args, option) is not None:
                command.extend(["--" + option, str(getattr(args, option))])
        if name == "current":
            command.append("--require-target")
        save(folder / "command.json", command)
        with (folder / "process.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        results[name] = json.loads((folder / "raw.json").read_text(encoding="utf-8"))
        if name == "baseline":
            for scenario in results[name]["scenarios"]:
                for sample in scenario["samples"]:
                    for measurement in sample.values():
                        expected = scenario["completed"] if scenario["mode"] == "mark" else 0
                        assert measurement["calls"]["button"]["setStyleSheet"] == expected * scenario["rows"]
                        assert measurement["calls"]["label"]["setStyleSheet"] == expected
        print(name, "finished", flush=True)
    summary = {"baseline_commit": BASELINE, "verified_baseline_files": checked,
               "semantic_equivalence": True, "scenarios": []}
    for before, after in zip(results["baseline"]["scenarios"], results["current"]["scenarios"], strict=True):
        assert before["name"] == after["name"]
        stages = {}
        for stage in before["warmup"]:
            for old, new in zip([before["warmup"], *before["samples"]],
                                [after["warmup"], *after["samples"]], strict=True):
                assert old[stage]["semantics"] == new[stage]["semantics"], (before["name"], stage)
            stages[stage] = {}
            for name, scenario in (("baseline", before), ("current", after)):
                samples = scenario["samples"]
                stages[stage][name] = {
                    **{metric: [sample[stage][metric] for sample in samples] for metric in ("wall_ms", "cpu_ms")},
                    **{f"median_{metric}": statistics.median(sample[stage][metric] for sample in samples)
                       for metric in ("wall_ms", "cpu_ms")},
                    "calls": [sample[stage]["calls"] for sample in samples]}
            stages[stage]["wall_reduction_percent"] = 100 * (1 - stages[stage]["current"]["median_wall_ms"] /
                                                           stages[stage]["baseline"]["median_wall_ms"])
        summary["scenarios"].append({"name": before["name"], "stages": stages})
    save(output / "summary.json", summary)
    print("Semantics and structural gates passed; summary:", output / "summary.json")


if __name__ == "__main__":
    main()
