import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base.core_algorithm.response.spl_frequency_analyzer import SplFrequencyAnalyzer
from base.stimulus_signal.frequency_stepped import generate_frequency_stepped


def _spl_db(rms_pa, reference_pressure_pa=20.0e-6):
    return 20.0 * np.log10(float(rms_pa) / reference_pressure_pa)


def _harmonic_signal(sample_rate, sample_count, f0, *, fundamental_peak, second_peak=0.0, dc=0.0):
    n = np.arange(sample_count, dtype=np.float64)
    return (
        float(dc)
        + float(fundamental_peak) * np.sin(2.0 * np.pi * f0 * n / sample_rate)
        + float(second_peak) * np.sin(2.0 * np.pi * 2.0 * f0 * n / sample_rate)
    )


def test_step_splf_fundamental_and_total_modes_use_existing_real_analyzer_paths():
    sample_rate = 48000
    metadata = {
        "stimulus_method": "steps",
        "stimulus_type": "linear",
        "start_freq": 1000.0,
        "stop_freq": 1000.0,
        "num_steps": 1,
        "repeat_times": 1,
        "total_time": 0.04,
    }
    sample_count = int(sample_rate * metadata["total_time"])
    recording = _harmonic_signal(
        sample_rate,
        sample_count,
        1000.0,
        fundamental_peak=2.0,
        second_peak=0.5,
        dc=0.25,
    )

    analyzer = SplFrequencyAnalyzer(sample_rate)
    fundamental = analyzer.compute(recording, stimulus_metadata=metadata, v2pa_factor=1.0, splf_calc_mode="fundamental")
    total = analyzer.compute(recording, stimulus_metadata=metadata, v2pa_factor=1.0, splf_calc_mode="total")

    assert fundamental.frequencies_hz.tolist() == pytest.approx([1000.0])
    assert total.frequencies_hz.tolist() == pytest.approx([1000.0])
    assert fundamental.spl_db[0] == pytest.approx(_spl_db(2.0 / np.sqrt(2.0)), abs=1e-3)
    expected_total_rms = np.sqrt((2.0 / np.sqrt(2.0)) ** 2 + (0.5 / np.sqrt(2.0)) ** 2)
    assert total.spl_db[0] == pytest.approx(_spl_db(expected_total_rms), abs=1e-3)
    assert total.spl_db[0] > fundamental.spl_db[0]


def test_frequency_stepped_splf_fundamental_and_total_modes_use_existing_real_analyzer_paths():
    sample_rate = 48000
    generated = generate_frequency_stepped(
        sample_rate=sample_rate,
        repeat_times=1,
        min_duration=0.012,
        min_cycles=8,
        frequency_mode="custom_linear",
        frequencies=[1000.0],
        generate_waveform=False,
    )
    recording = np.zeros(generated.metadata["alignment_sample_count"], dtype=float)
    segment = generated.segments[0]
    recording[segment.start_sample:segment.end_sample] = _harmonic_signal(
        sample_rate,
        segment.sample_count,
        1000.0,
        fundamental_peak=2.0,
        second_peak=0.5,
        dc=0.25,
    )

    analyzer = SplFrequencyAnalyzer(sample_rate)
    fundamental = analyzer.compute(
        recording,
        stimulus_metadata=generated.metadata,
        v2pa_factor=1.0,
        splf_calc_mode="fundamental",
    )
    total = analyzer.compute(
        recording,
        stimulus_metadata=generated.metadata,
        v2pa_factor=1.0,
        splf_calc_mode="total",
    )

    assert fundamental.frequencies_hz.tolist() == pytest.approx([1000.0])
    assert total.frequencies_hz.tolist() == pytest.approx([1000.0])
    assert fundamental.spl_db[0] == pytest.approx(_spl_db(2.0 / np.sqrt(2.0)), abs=1e-3)
    expected_total_rms = np.sqrt((2.0 / np.sqrt(2.0)) ** 2 + (0.5 / np.sqrt(2.0)) ** 2)
    assert total.spl_db[0] == pytest.approx(_spl_db(expected_total_rms), abs=1e-3)
    assert total.spl_db[0] > fundamental.spl_db[0]
