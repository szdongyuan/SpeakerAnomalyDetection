import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base.core_algorithm.response.fft_analyzer import FftAnalyzer


def test_welch_fft_detects_sine_peak_near_frequency():
    fs = 48000
    n_fft = 4096
    duration = 1.0
    target_hz = 1000.0
    t = np.arange(int(fs * duration)) / fs
    signal = 0.1 * np.sin(2 * np.pi * target_hz * t)

    result = FftAnalyzer().analyze(
        signal,
        fs=fs,
        n_fft=n_fft,
        window="hann",
        overlap_ratio=0.5,
        weighting="Z",
        v2pa_factor=1.0,
    )

    assert result.frequencies_hz.size == n_fft // 2 + 1
    peak_hz = result.frequencies_hz[int(np.nanargmax(result.spectrum_db))]
    assert abs(peak_hz - target_hz) <= fs / n_fft


def test_a_weighting_reduces_low_frequency_more_than_z_weighting():
    fs = 48000
    n_fft = 4096
    t = np.arange(fs) / fs
    signal = 0.1 * np.sin(2 * np.pi * 100.0 * t)

    z_result = FftAnalyzer().analyze(signal, fs=fs, n_fft=n_fft, weighting="Z")
    a_result = FftAnalyzer().analyze(signal, fs=fs, n_fft=n_fft, weighting="A")

    idx = int(np.argmin(np.abs(z_result.frequencies_hz - 100.0)))
    assert a_result.spectrum_db[idx] < z_result.spectrum_db[idx] - 10.0


@pytest.mark.parametrize(
    ("n_fft", "overlap_ratio", "match"),
    [
        (511, 0.5, "FFT 点数"),
        (65536, 0.5, "FFT 点数"),
        (1024, 0.99, "重叠率"),
    ],
)
def test_welch_fft_rejects_invalid_configuration(n_fft, overlap_ratio, match):
    with pytest.raises(ValueError, match=match):
        FftAnalyzer().analyze(
            np.ones(4096),
            fs=48000,
            n_fft=n_fft,
            overlap_ratio=overlap_ratio,
        )
