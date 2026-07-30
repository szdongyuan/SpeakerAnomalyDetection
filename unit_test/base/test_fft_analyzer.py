import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base.core_algorithm.response.fft_analyzer import FftAnalyzer
from consts.acoustic_analysis.specific_consts.fft_consts import MAX_FFT_SIZE


def test_welch_fft_detects_sine_peak_near_frequency():
    fs = 48000
    n_fft = 4096
    target_hz = 1000.0
    time = np.arange(fs) / fs
    signal = 0.1 * np.sin(2 * np.pi * target_hz * time)

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


def test_welch_fft_accepts_positional_configuration_arguments():
    result = FftAnalyzer().analyze(np.ones(1024), 48000, 512)

    assert result.n_fft == 512


def test_welch_fft_accepts_maximum_fft_size():
    result = FftAnalyzer().analyze(np.ones(MAX_FFT_SIZE), 48000, MAX_FFT_SIZE)

    assert result.n_fft == MAX_FFT_SIZE
    assert result.frequencies_hz.size == MAX_FFT_SIZE // 2 + 1


def test_a_weighting_reduces_low_frequency_more_than_z_weighting():
    fs = 48000
    n_fft = 4096
    time = np.arange(fs) / fs
    signal = 0.1 * np.sin(2 * np.pi * 100.0 * time)

    z_result = FftAnalyzer().analyze(signal, fs=fs, n_fft=n_fft, weighting="Z")
    a_result = FftAnalyzer().analyze(signal, fs=fs, n_fft=n_fft, weighting="A")

    index = int(np.argmin(np.abs(z_result.frequencies_hz - 100.0)))
    assert a_result.spectrum_db[index] < z_result.spectrum_db[index] - 10.0


def test_v2pa_factor_changes_spectrum_by_expected_level():
    fs = 48000
    time = np.arange(fs) / fs
    signal = np.sin(2 * np.pi * 1000.0 * time)

    reference = FftAnalyzer().analyze(signal, fs=fs, n_fft=4096, v2pa_factor=1.0)
    doubled = FftAnalyzer().analyze(signal, fs=fs, n_fft=4096, v2pa_factor=2.0)

    peak_index = int(np.nanargmax(reference.spectrum_db))
    assert np.isclose(
        doubled.spectrum_db[peak_index] - reference.spectrum_db[peak_index],
        20.0 * np.log10(2.0),
    )


@pytest.mark.parametrize(
    ("n_fft", "overlap_ratio", "match"),
    [
        (511, 0.5, "FFT 点数"),
        (MAX_FFT_SIZE + 1, 0.5, "FFT 点数"),
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


def test_welch_fft_rejects_signal_shorter_than_fft_size():
    with pytest.raises(ValueError, match="不能大于信号长度"):
        FftAnalyzer().analyze(np.ones(511), fs=48000, n_fft=512)
