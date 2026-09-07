import numpy as np
import pytest

from base.core_algorithm.response import SpectrogramAnalyzer


def _sine_wave(frequency_hz, *, sample_rate=48_000, seconds=0.25):
    time_axis = np.arange(int(sample_rate * seconds)) / sample_rate
    return 0.01 * np.sin(2.0 * np.pi * frequency_hz * time_axis)


def test_linear_spectrogram_returns_aligned_frequency_time_matrix():
    result = SpectrogramAnalyzer().analyze(
        _sine_wave(1_000),
        fs=48_000,
        n_fft=256,
        hop_length=64,
        scale="linear",
        max_time_bins=2_000,
    )

    assert result.scale == "linear"
    assert result.values_db.shape == (
        len(result.frequencies_hz),
        len(result.times_s),
    )
    peak_index = int(np.nanargmax(np.nanmax(result.values_db, axis=1)))
    assert result.frequencies_hz[peak_index] == pytest.approx(937.5)


def test_linear_spectrogram_bounds_long_recording_time_bins():
    result = SpectrogramAnalyzer().analyze(
        _sine_wave(1_000, seconds=4.0),
        fs=48_000,
        n_fft=256,
        hop_length=64,
        scale="linear",
        max_time_bins=200,
    )

    assert result.values_db.shape[1] == 200
    assert result.times_s[-1] > 3.9


@pytest.mark.parametrize(
    ("n_fft", "hop_length"),
    [(0, 64), (256, 0), (256, 257)],
)
def test_spectrogram_rejects_invalid_frame_configuration(n_fft, hop_length):
    with pytest.raises(ValueError, match="FFT 点数或帧移"):
        SpectrogramAnalyzer().analyze(
            _sine_wave(1_000),
            fs=48_000,
            n_fft=n_fft,
            hop_length=hop_length,
        )
