"""Isolated Qt save benchmark; no recording device or production config is used.

Run with the project Python: scripts/profile_product_config_refresh.py --output tmp/refresh-profile
Requires the test environment because the integration host isolates hardware via mocks.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "windows")

from PyQt5.QtCore import QCoreApplication, QEvent, Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QMessageBox, QPushButton

from base.load_config import LoadUiConfig
from ui.product_test_project_config_dialog import ProductTestProjectConfigDialog
from unit_test.base.test_product_test_config_refresh import make_refresh_manager
from unit_test.ui.test_product_test_program_runtime_refresh import ProductRefreshHost, save_b


def flush_events(app):
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()
    app.processEvents()


def measure(app, action):
    counters = {"queue_reads": 0, "button_styles": 0}
    read = LoadUiConfig.load_data_from_json
    style = QPushButton.setStyleSheet

    def counted_read(path):
        if Path(path).parent.name == "queues" and Path(path).name != "registry.json":
            counters["queue_reads"] += 1
        return read(path)

    def counted_style(button, stylesheet):
        counters["button_styles"] += 1
        return style(button, stylesheet)

    with patch.object(LoadUiConfig, "load_data_from_json", counted_read), \
            patch.object(QPushButton, "setStyleSheet", counted_style):
        started = perf_counter()
        action()
        callback_end = perf_counter()
        flush_events(app)
        events_end = perf_counter()
    return {"callback_seconds": callback_end - started,
            "events_seconds": events_end - started, **counters}


def run_size(app, output, size, repeats, baseline_path):
    folder = Path(tempfile.mkdtemp(prefix=f"data-{size}-", dir=output))
    manager, project, _ = make_refresh_manager(folder, condition_count=size // 10)
    host = ProductRefreshHost(manager)
    host.setAttribute(Qt.WA_DontShowOnScreen)
    layout = QHBoxLayout(host)
    layout.addWidget(host.left_panel, 1)
    layout.addWidget(host.channel_workspace, 2)
    host.resize(1280, 780)
    host.show()
    flush_events(app)
    result = {"conditions": size, "runs": {}}
    result["runs"]["panel_reset"] = [
        measure(app, host.left_panel.result_panel.reset) for _ in range(repeats + 1)
    ]
    if baseline_path:
        spec = importlib.util.spec_from_file_location("baseline_panel", baseline_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        panel = module.MotorResultPanel(condition_configs=host.product_test_condition_configs)
        panel.setAttribute(Qt.WA_DontShowOnScreen)
        panel.resize(480, 780)
        panel.show()
        flush_events(app)
        result["baseline_panel_reset"] = measure(app, panel.reset)
        panel.close()
        panel.deleteLater()
        flush_events(app)

    save_b(host, project)
    dialog = ProductTestProjectConfigDialog(manager=manager)
    dialog.setAttribute(Qt.WA_DontShowOnScreen)
    dialog.programs_changed.connect(host.on_product_test_program_updated)
    dialog.show()
    flush_events(app)
    host._product_test_program_config_dialog_open = True

    def save():
        assert dialog._save_project(close_dialog=False)
        assert host._product_config_refresh_state == "ready"

    for scenario in ("save_unchanged_a", "save_b", "save_changed_a"):
        dialog._load_project("B.json" if scenario == "save_b" else "A.json")
        samples = []
        for _ in range(repeats + 1):
            before = host._applied_product_snapshot
            if scenario == "save_changed_a":
                radio = (dialog.wav_only_radio if dialog.wav_and_csv_radio.isChecked()
                         else dialog.wav_and_csv_radio)
                radio.setChecked(True)
            samples.append(measure(app, save))
            if scenario == "save_changed_a":
                assert host._applied_product_snapshot.signature != before.signature
            else:
                assert host._applied_product_snapshot is before
        result["runs"][scenario] = samples
    host._product_test_program_config_dialog_open = False
    host.update_player_btn_is_paused()
    host.grab().save(str(output / f"after-save-{size}.png"))
    dialog.close()
    dialog.deleteLater()
    host.close()
    host.deleteLater()
    flush_events(app)
    result["warm_medians"] = {
        name: {field: statistics.median(run[field] for run in runs[1:])
               for field in runs[0]}
        for name, runs in result["runs"].items()
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 100, 200])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-panel", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 10 or size % 10 for size in args.sizes):
        parser.error("sizes must be positive multiples of 10; repeats must be at least 1")
    args.output.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])

    def unexpected_warning(_parent, title, message):
        raise RuntimeError(f"{title}: {message}")

    with patch.object(QMessageBox, "information", return_value=QMessageBox.Ok), \
            patch.object(QMessageBox, "warning", unexpected_warning):
        results = []
        for size in args.sizes:
            result = run_size(app, args.output, size, args.repeats, args.baseline_panel)
            results.append(result)
            print(json.dumps({"conditions": size, **result["warm_medians"]}), flush=True)
    (args.output / "measurements.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
