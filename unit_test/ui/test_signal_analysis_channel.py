import ast
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from base.core_algorithm.response.spl_frequency_analyzer import SplFrequencyAnalyzer
from base.data_struct.data_deal_struct import DataDealStruct
from base.stimulus_signal.frequency_stepped import generate_frequency_stepped
from ui.signal_analysis_window import SplFrequency, _abs_deviation_curve


def _load_resolve_analysis_channel_signal():
    source_path = Path(__file__).parents[2] / "ui" / "signal_analysis_window.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function_node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "resolve_analysis_channel_signal"
    )
    module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(module)

    namespace = {"DataDealStruct": DataDealStruct, "np": np}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["resolve_analysis_channel_signal"]


def test_resolve_analysis_channel_signal_uses_multi_channel_data(capsys):
    resolve_analysis_channel_signal = _load_resolve_analysis_channel_signal()
    data_struct = DataDealStruct()
    original_mono = data_struct.store_wave_data
    original_multi = data_struct.store_wave_data_multi

    try:
        data_struct.store_wave_data = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        data_struct.store_wave_data_multi = np.array(
            [
                [1.0, 101.0],
                [2.0, 102.0],
                [3.0, 103.0],
            ],
            dtype=np.float32,
        )

        resolved = resolve_analysis_channel_signal(
            data_struct,
            {"analysis_channel": 1},
            "SPL",
        )

        np.testing.assert_array_equal(resolved, np.array([101.0, 102.0, 103.0], dtype=np.float32))
        assert capsys.readouterr().out == ""
    finally:
        data_struct.store_wave_data = original_mono
        data_struct.store_wave_data_multi = original_multi


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def _frequency_stepped_metadata_and_recording():
    sample_rate = 48000
    generated = generate_frequency_stepped(
        sample_rate=sample_rate,
        repeat_times=1,
        min_duration=0.012,
        min_cycles=8,
        frequency_mode="custom_linear",
        frequencies=[1000.0, 1000.0, 2000.0],
        generate_waveform=False,
    )
    recording = np.zeros(generated.metadata["alignment_sample_count"], dtype=float)
    amplitudes = {
        0: 1.0,
        1: 3.0,
        2: 2.0,
    }
    for segment in generated.segments:
        n = np.arange(segment.sample_count, dtype=float)
        recording[segment.start_sample:segment.end_sample] = amplitudes[segment.step_index] * np.sin(
            2.0 * np.pi * segment.frequency_hz * n / sample_rate
        )
    return generated.metadata, recording, sample_rate


def test_spl_frequency_step_sc_smoothing_preserves_duplicate_output_points(qapp, monkeypatch):
    metadata, recording, sample_rate = _frequency_stepped_metadata_and_recording()
    expected = SplFrequencyAnalyzer(sample_rate=sample_rate).compute(
        recording,
        stimulus_metadata=metadata,
        splf_calc_mode="total",
        v2pa_factor=1.0,
        eps=1e-12,
    )

    widget = SplFrequency("SPLF")
    monkeypatch.setattr(widget, "plot_spl_frequency", lambda *args, **kwargs: None)
    monkeypatch.setattr(widget, "plot_spl_frequency_with_limits", lambda *args, **kwargs: None)
    widget.v2pa_factor = 1.0
    widget.analysis_config = {"smooth_checked": True, "splf_calc_mode": "total"}
    widget.data_struct.sample_rate = sample_rate
    widget.data_struct.stimulus_info = metadata
    widget.data_struct.store_wave_data = recording.astype(np.float32)
    monkeypatch.setattr("ui.signal_analysis_window.MessageBox.warning", lambda *args, **kwargs: None)

    result = widget.calculate_spl()

    assert result["frequency_list"] == [1000.0, 1000.0, 2000.0]
    assert result["spl_db"] == pytest.approx(expected.spl_db.tolist(), abs=1e-6)
    assert result["spl_db_raw"] == pytest.approx(expected.spl_db.tolist(), abs=1e-6)


def test_golden_baseline_deviation_pairs_duplicate_frequency_points_by_position():
    deviation = _abs_deviation_curve(
        [1000.0, 1000.0, 2000.0],
        [10.0, 30.0, 20.0],
        [1000.0, 1000.0, 2000.0],
        [11.0, 29.0, 19.0],
    )

    assert deviation.tolist() == pytest.approx([-1.0, 1.0, 1.0])
