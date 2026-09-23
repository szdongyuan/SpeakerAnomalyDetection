"""Optional matched-component GUI benchmark; no machine-specific timing assertions.

Run directly from any directory. A revision loads only its waveform module.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import types

# Select one binding before either revision (or pyqtgraph) can import Qt.
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"

import numpy as np
from PyQt5.QtCore import PYQT_VERSION_STR, QT_VERSION_STR
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget
import pyqtgraph as pg

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
WARMUP = 5
FS = 48000


def load_component(revision):
    if revision is None:
        return importlib.import_module("ui.adaptive_waveform")
    source = subprocess.check_output(
        ["git", "show", revision + ":ui/adaptive_waveform.py"], cwd=ROOT
    )
    module = types.ModuleType("benchmark_reference_waveform")
    exec(compile(source, revision + ":ui/adaptive_waveform.py", "exec"), module.__dict__)
    return module


def benchmark_case(app, module, duration, channels, width, iterations):
    # Source and exact sample-time axis allocation are deliberately outside timings.
    x = np.arange(duration * FS, dtype=float) / FS
    signals = [np.sin(2 * np.pi * 997 * x + channel * 0.37)
               for channel in range(channels)]
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    plots, items, calls = [], [], []
    original_sinc = module.bandlimited_values

    def observed_sinc(y, positions):
        calls.append((len(y), len(positions)))
        return original_sinc(y, positions)

    module.bandlimited_values = observed_sinc
    try:
        for signal in signals:
            plot = pg.PlotWidget()
            plot.setFixedSize(width, 220)
            layout.addWidget(plot)
            plot.setDownsampling(auto=True, mode="peak")
            plot.setClipToView(True)
            plot.setYRange(-1.2, 1.2, padding=0)
            plot.setXRange(0, duration, padding=0)
            item = module.AdaptiveWaveformItem(x, signal, pen="w")
            plot.addItem(item)
            assert item.opts["autoDownsample"] and item.opts["clipToView"]
            plots.append(plot)
            items.append(item)
        host.show()
        app.processEvents()

        def paint():
            app.processEvents()
            for plot in plots:
                assert plot.isVisible()
                pixmap = plot.grab()
                assert not pixmap.isNull() and pixmap.width() == width

        paint()
        widths = [float(plot.getViewBox().width()) for plot in plots]
        timings = {}
        bounds = {}
        for scenario in ("dense_update", "sparse_pan", "streaming_update", "sparse_cache"):
            sparse = scenario != "dense_update"
            for plot, item, signal in zip(plots, items, signals):
                item.setData(x, signal, streaming=scenario == "streaming_update")
                plot.setXRange(1, 1 + 128 / FS, padding=0) if sparse else plot.setXRange(0, duration, padding=0)
            paint()
            calls.clear()
            samples = []
            for index in range(WARMUP + iterations):
                started = time.perf_counter_ns()
                for plot, item, signal in zip(plots, items, signals):
                    if scenario in ("dense_update", "streaming_update"):
                        item.setData(x, signal, streaming=scenario == "streaming_update")
                    elif scenario == "sparse_pan":
                        left = 1 + (index % 2) * 16 / FS
                        plot.setXRange(left, left + 128 / FS, padding=0)
                    else:
                        item._refresh_smoothing()
                paint()
                elapsed = (time.perf_counter_ns() - started) / 1e6
                if index >= WARMUP:
                    samples.append(elapsed)
            if scenario in ("streaming_update", "sparse_cache"):
                assert not calls, (scenario, calls)
            if scenario == "sparse_pan":
                assert calls, "Sparse view did not reconstruct"
                assert all(n <= 4096 + 2 * module.SINC_RADIUS and m <= 12288 for n, m in calls)
            bounds[scenario] = {
                "sinc_calls_including_warmup": len(calls),
                "max_input_samples": max((n for n, _ in calls), default=0),
                "max_output_points": max((m for _, m in calls), default=0),
            }
            timings[scenario] = {"median_ms": float(np.median(samples)),
                                 "p95_ms": float(np.percentile(samples, 95))}
        return {"duration_s": duration, "channels": channels, "widget_width_px": width,
                "viewbox_widths_px": widths, "timings": timings, "reconstruction": bounds}
    finally:
        host.close()
        host.deleteLater()
        app.processEvents()
        module.bandlimited_values = original_sinc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", help="Load only the waveform module from this git ref")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    module = load_component(args.revision)
    app = QApplication.instance() or QApplication([])
    cases = []
    for duration in (10, 60):
        for channels in (1, 4):
            for width in (800, 1600):
                result = benchmark_case(app, module, duration, channels, width, args.iterations)
                cases.append(result)
                print(f"{duration}s / {channels} channels / {width}px: {result['timings']}", flush=True)
    # Same view/width must use exactly the same local work for both source lengths.
    for short, long in zip(cases[:4], cases[4:]):
        for key in ("max_input_samples", "max_output_points"):
            assert short["reconstruction"]["sparse_pan"][key] == long["reconstruction"]["sparse_pan"][key]
    result = {
        "revision": args.revision or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source": "git module" if args.revision else "worktree module",
        "environment": {"python": sys.version, "numpy": np.__version__, "qt": QT_VERSION_STR,
                        "pyqt": PYQT_VERSION_STR, "pyqtgraph": pg.__version__, "os": platform.platform(),
                        "qt_platform": os.environ["QT_QPA_PLATFORM"], "qt_binding": os.environ["PYQTGRAPH_QT_LIB"]},
        "sample_rate_hz": FS, "iterations": args.iterations, "warmup": WARMUP,
        "timing_boundary": "One batch across every visible channel: operation + processEvents + PlotWidget.grab. Source/axis allocation excluded; setData replacement included. Fixed Y range.",
        "settings": "Both revisions: plot-level auto peak downsampling and clipToView. 128-sample sparse span, alternating 16-sample pan. Streaming uses sparse view to verify bypass. Cache explicitly refreshes unchanged sparse view.",
        "cases": cases,
    }
    encoded = json.dumps(result, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
