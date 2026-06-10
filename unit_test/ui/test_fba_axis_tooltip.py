import ast
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base.core_algorithm.response.dominant_tone_analyzer import (
    FrequencyInterval,
    find_dominant_fba_bands,
    parse_frequency_intervals,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SIGNAL_ANALYSIS_PATH = REPO_ROOT / "ui" / "signal_analysis_window.py"


def _load_class_method(method_name: str):
    source = SIGNAL_ANALYSIS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FrequencyBandAnalysis")
    method_node = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    module = ast.Module(body=[method_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "np": np,
        "FrequencyInterval": FrequencyInterval,
        "find_dominant_fba_bands": find_dominant_fba_bands,
        "parse_frequency_intervals": parse_frequency_intervals,
    }
    exec(compile(module, str(SIGNAL_ANALYSIS_PATH), "exec"), namespace)
    return namespace[method_name]


def test_build_sparse_x_ticks_keeps_axis_readable_and_preserves_edges():
    build_ticks = _load_class_method("_build_sparse_x_ticks")
    labels = [f"{i}Hz" for i in range(25)]

    ticks = build_ticks(labels, max_ticks=8)

    assert ticks[0] == (0, "0Hz")
    assert ticks[-1] == (24, "24Hz")
    assert len(ticks) <= 8
    assert len({pos for pos, _label in ticks}) == len(ticks)


def test_build_fba_hover_rows_contains_full_band_context():
    build_rows = _load_class_method("_build_fba_hover_rows")
    bands = [
        types.SimpleNamespace(label="20", f_low=17.8, f_high=22.4, f_center=20.0),
        types.SimpleNamespace(label="25", f_low=22.4, f_high=28.2, f_center=25.0),
    ]

    rows = build_rows(bands, np.array([1.25, np.nan]), {1})

    assert rows[0]["index"] == 0
    assert rows[0]["label"] == "20"
    assert rows[0]["level_text"] == "1.25 dB"
    assert rows[0]["status"] == "OK"
    assert rows[1]["level_text"] == "N/A"
    assert rows[1]["status"] == "NG"


def test_build_fba_display_levels_supports_delta_mode():
    build_levels = _load_class_method("_build_fba_display_levels")
    levels = np.array([30.0, 45.0])
    baseline = np.array([20.0, 40.0])

    result = build_levels(levels, baseline, "delta")

    assert np.allclose(result["plot_levels"], np.array([10.0, 5.0]))
    assert result["baseline_levels"] is baseline
    assert np.allclose(result["delta_levels"], np.array([10.0, 5.0]))


def test_fba_detect_dominant_tones_uses_highest_band_per_interval():
    detect = _load_class_method("_detect_dominant_tones")
    bands = [
        types.SimpleNamespace(label="100", f_low=80.0, f_high=125.0, f_center=100.0),
        types.SimpleNamespace(label="315", f_low=250.0, f_high=400.0, f_center=315.0),
        types.SimpleNamespace(label="1k", f_low=800.0, f_high=1250.0, f_center=1000.0),
    ]
    levels = np.array([20.0, 35.0, 30.0])
    config = {
        "dominant_tone_enabled": True,
        "dominant_tone_intervals_text": "50, 500, Low\n500, 2000, High",
    }

    tones = detect(bands, levels, config, fallback_low_hz=50, fallback_high_hz=2000)

    assert [tone["interval_label"] for tone in tones] == ["Low", "High"]
    assert tones[0]["frequency_hz"] == 315.0
    assert tones[0]["band_label"] == "315"
    assert tones[1]["frequency_hz"] == 1000.0
