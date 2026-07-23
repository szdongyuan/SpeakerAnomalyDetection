import ast
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from base.core_algorithm.response.spl_frequency_analyzer import SplFrequencyAnalyzer
from base.data_struct.data_deal_struct import DataDealStruct
from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis
from base.stimulus_signal.frequency_stepped import generate_frequency_stepped
from consts.harmonic_detection_consts import (
    HARMONIC_DETECTION_METHOD_FOURIER,
    HARMONIC_DETECTION_METHOD_KEY,
    HARMONIC_DETECTION_METHOD_SYNCHRONOUS,
)
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


def _harmonic_tone(sample_rate, sample_count, f0, amplitudes, phase=0.0, dc=0.0):
    n = np.arange(sample_count, dtype=np.float64)
    y = np.full(sample_count, dc, dtype=np.float64)
    for order, amplitude in amplitudes.items():
        y += amplitude * np.sin(phase + 2.0 * np.pi * order * f0 * n / sample_rate)
    return y


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
        0: {1: 1.0, 2: 0.05},
        1: {1: 3.0, 2: 0.60},
        2: {1: 2.0, 3: 0.10},
    }
    for segment in generated.segments:
        recording[segment.start_sample:segment.end_sample] = _harmonic_tone(
            sample_rate,
            segment.sample_count,
            segment.frequency_hz,
            amplitudes[segment.step_index],
            phase=0.15 * (segment.step_index + 1),
            dc=0.02,
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
    widget.analysis_config = {"smooth_checked": True, "splf_calc_mode": "total"}
    widget.data_struct.sample_rate = sample_rate
    widget.data_struct.stimulus_info = metadata
    widget.data_struct.store_wave_data = recording.astype(np.float32)
    monkeypatch.setattr("ui.signal_analysis_window.resolve_analysis_v2pa_factor_for_channel", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr("ui.signal_analysis_window.MessageBox.warning", lambda *args, **kwargs: None)

    result = widget.calculate_spl()

    assert result["frequency_list"] == [1000.0, 1000.0, 2000.0]
    assert result["spl_db"] == pytest.approx(expected.spl_db.tolist(), abs=1e-6)
    assert result["spl_db_raw"] == pytest.approx(expected.spl_db.tolist(), abs=1e-6)


def test_public_thd_api_handles_frequency_stepped_metadata():
    metadata, recording, sample_rate = _frequency_stepped_metadata_and_recording()

    freq_value, harmonic, thd = AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
        recording,
        sample_rate,
        {"stimulus_metadata": metadata, "harmonic_orders": [2, 3]},
    )

    assert freq_value.tolist() == [1000.0, 1000.0, 2000.0]
    assert harmonic.shape == (6, len(freq_value))
    assert harmonic[0].tolist() == pytest.approx(freq_value.tolist())
    assert harmonic[1].tolist() == pytest.approx([1.0, 3.0, 2.0], rel=1e-8, abs=1e-10)
    assert harmonic[2].tolist() == pytest.approx([0.05, 0.60, 0.0], rel=1e-8, abs=1e-10)
    assert harmonic[3].tolist() == pytest.approx([0.0, 0.0, 0.10], rel=1e-8, abs=1e-10)
    np.testing.assert_allclose(harmonic[4:6], 0.0, rtol=1e-8, atol=1e-10)
    assert thd.tolist() == pytest.approx([5.0, 20.0, 5.0], rel=1e-8, abs=1e-10)


def test_public_thd_api_frequency_stepped_fourier_method_preserves_harmonic_plot_matrix_shape():
    metadata, recording, sample_rate = _frequency_stepped_metadata_and_recording()

    freq_value, harmonic, thd = AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
        recording,
        sample_rate,
        {
            "stimulus_metadata": metadata,
            "harmonic_orders": [2, 3],
            HARMONIC_DETECTION_METHOD_KEY: HARMONIC_DETECTION_METHOD_FOURIER,
        },
    )

    assert freq_value.tolist() == pytest.approx([1000.0, 1000.0, 2000.0])
    assert harmonic.shape == (6, len(freq_value))
    assert harmonic[0].tolist() == pytest.approx(freq_value.tolist())
    assert harmonic[1, 0] > 0.0
    assert harmonic[2, 0] / harmonic[1, 0] == pytest.approx(0.05, rel=1e-6, abs=1e-8)
    assert harmonic[2, 1] / harmonic[1, 1] == pytest.approx(0.20, rel=1e-6, abs=1e-8)
    assert harmonic[3, 2] / harmonic[1, 2] == pytest.approx(0.05, rel=1e-6, abs=1e-8)
    assert thd.tolist() == pytest.approx([5.0, 20.0, 5.0], rel=1e-6, abs=1e-8)


def test_public_thd_api_preserves_step_harmonic_plot_matrix():
    sample_rate = 48000
    metadata = {
        "stimulus_method": "steps",
        "stimulus_type": "linear",
        "start_freq": 1000,
        "stop_freq": 2000,
        "num_steps": 2,
        "repeat_times": 1,
        "total_time": 0.04,
    }
    step_samples = int(sample_rate * metadata["total_time"] / metadata["num_steps"])
    recording = np.concatenate(
        [
            _harmonic_tone(sample_rate, step_samples, 1000.0, {1: 1.2, 2: 0.12}, phase=0.3, dc=0.04),
            _harmonic_tone(sample_rate, step_samples, 2000.0, {1: 0.8, 3: 0.08}, phase=-0.9, dc=-0.02),
        ]
    )

    freq_value, harmonic, thd = AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
        recording,
        sample_rate,
        {
            "stimulus_metadata": metadata,
            "harmonic_orders": [2, 3],
            HARMONIC_DETECTION_METHOD_KEY: HARMONIC_DETECTION_METHOD_SYNCHRONOUS,
        },
    )

    assert freq_value.tolist() == pytest.approx([1000.0, 2000.0])
    assert harmonic.shape == (6, 2)
    assert harmonic.ndim == 2
    assert harmonic[0].tolist() == pytest.approx([1000.0, 2000.0])
    assert harmonic[1].tolist() == pytest.approx([1.2, 0.8], rel=1e-8, abs=1e-10)
    assert harmonic[2].tolist() == pytest.approx([0.12, 0.0], rel=1e-8, abs=1e-10)
    assert harmonic[3].tolist() == pytest.approx([0.0, 0.08], rel=1e-8, abs=1e-10)
    np.testing.assert_allclose(harmonic[4:6], 0.0, rtol=1e-8, atol=1e-10)
    assert thd.tolist() == pytest.approx([10.0, 10.0], rel=1e-8, abs=1e-10)


def test_public_thd_api_step_fourier_method_preserves_harmonic_plot_matrix_shape():
    sample_rate = 48000
    metadata = {
        "stimulus_method": "steps",
        "stimulus_type": "linear",
        "start_freq": 1000.0,
        "stop_freq": 1000.0,
        "num_steps": 1,
        "repeat_times": 1,
        "total_time": 0.05,
    }
    recording = _harmonic_tone(sample_rate, int(sample_rate * metadata["total_time"]), 1000.0, {1: 1.0, 2: 0.2})

    freq_value, harmonic, thd = AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
        recording,
        sample_rate,
        {"stimulus_metadata": metadata, "harmonic_orders": [2], HARMONIC_DETECTION_METHOD_KEY: "fourier"},
    )

    assert freq_value.tolist() == pytest.approx([1000.0])
    assert harmonic.shape == (6, 1)
    assert harmonic[1, 0] > 0.0
    assert harmonic[2, 0] / harmonic[1, 0] == pytest.approx(0.2, rel=1e-6, abs=1e-8)
    assert thd[0] == pytest.approx(20.0, rel=1e-6, abs=1e-8)


def test_signal_analysis_window_uses_public_thd_analysis_api():
    source_path = Path(__file__).parents[2] / "ui" / "signal_analysis_window.py"
    source = source_path.read_text(encoding="utf-8")

    assert "atfra.calculate_thd_three_phase(" in source
    assert "atfra.calculate_perceptual_thd_three_phase(" in source
    assert "atfra._calculate_thd_three_phase(" not in source
    assert "atfra._calculate_perceptual_thd_three_phase(" not in source


def test_frequency_stepped_thd_analysis_does_not_call_private_stft_api():
    source_path = Path(__file__).parents[2] / "base" / "pre_processing" / "audio_thd_frequency_response_analysis.py"
    source = source_path.read_text(encoding="utf-8")

    assert "._compute_stft(" not in source
    assert "max_harmonic_order=35" not in source
    assert "intermodulation" not in source.lower()


def test_golden_baseline_deviation_pairs_duplicate_frequency_points_by_position():
    deviation = _abs_deviation_curve(
        [1000.0, 1000.0, 2000.0],
        [10.0, 30.0, 20.0],
        [1000.0, 1000.0, 2000.0],
        [11.0, 29.0, 19.0],
    )

    assert deviation.tolist() == pytest.approx([-1.0, 1.0, 1.0])
