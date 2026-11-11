"""Contracts of the unchanged snapshot algorithms with the existing loudness API."""

import numpy as np
import pytest

from base.core_algorithm.sound_quality.loudness import LoudnessAnalyzer, LoudnessMethod
from base.core_algorithm.sound_quality.roughness import compute_roughness_daniel_weber
from base.core_algorithm.sound_quality.sharpness import compute_sharpness_din45692


@pytest.mark.parametrize("method", list(LoudnessMethod))
@pytest.mark.parametrize("sample_rate", [44100, 48000])
def test_sharpness_consumes_existing_loudness_without_changing_its_arrays(method, sample_rate):
    t = np.arange(int(0.4 * sample_rate)) / sample_rate
    pressure = 0.2 * np.sin(2 * np.pi * 1000 * t)
    loudness = LoudnessAnalyzer(sample_rate).compute(pressure, method=method, v2pa_factor=1.0)
    original = loudness.specific_loudness.copy()
    bark = loudness.bark_axis.copy()
    times = loudness.time_s.copy()

    result = compute_sharpness_din45692(original, bark, times, method=method)

    assert result.sharpness_acum.shape == loudness.loudness_sone.shape
    np.testing.assert_array_equal(result.time_s, loudness.time_s)
    assert np.isfinite(result.summary.s_mean) and result.summary.s_mean > 0
    assert np.isfinite(result.summary.s_stationary)
    np.testing.assert_array_equal(original, loudness.specific_loudness)
    np.testing.assert_array_equal(bark, loudness.bark_axis)
    np.testing.assert_array_equal(times, loudness.time_s)
    assert not result.sharpness_acum.flags.writeable
    assert not result.time_s.flags.writeable


@pytest.mark.parametrize("bark", [10.0, 20.0])
def test_sharpness_single_band_analytic_value(bark):
    axis = np.linspace(0.1, 24.0, 240)
    specific = np.zeros((240, 3))
    specific[int(round(bark * 10)) - 1] = [2.0, 4.0, 8.0]
    result = compute_sharpness_din45692(specific, axis, np.array([0.0, 0.1, 0.2]))
    weighting = 1.0 if bark <= 15.8 else 0.15 * np.exp(0.42 * (bark - 15.8)) + 0.85
    expected = 0.11 * bark * weighting
    np.testing.assert_allclose(result.sharpness_acum, expected, rtol=1e-14)
    assert result.summary.s_stationary == pytest.approx(expected)
    assert result.metadata["skip_frames_for_summary"] == 1


def test_sharpness_silence_is_undefined_and_time_skip_only_changes_summary():
    axis = np.linspace(0.1, 24.0, 240)
    specific = np.zeros((240, 3))
    specific[199, 0] = 4.0
    specific[99, 2] = 4.0
    times = np.array([0.0, 0.05, 0.1])
    result = compute_sharpness_din45692(specific, axis, times)
    assert np.isnan(result.sharpness_acum[1])
    assert result.sharpness_acum[0] > result.sharpness_acum[2]
    assert result.summary.s_mean == pytest.approx(1.1)
    silent = compute_sharpness_din45692(np.zeros((240, 3)), axis, times)
    assert np.isnan(silent.sharpness_acum).all()
    assert np.isnan(silent.summary.s_mean)


@pytest.mark.parametrize("invalid", ["bark", "time", "nonfinite"])
def test_sharpness_rejects_invalid_input_contract(invalid):
    axis = np.linspace(0.1, 24.0, 240)
    specific = np.ones((240, 3))
    times = np.array([0.0, 0.1, 0.2])
    if invalid == "bark":
        axis += 0.01
    elif invalid == "time":
        times = times[:2]
    else:
        specific[0, 0] = np.nan
    with pytest.raises(ValueError):
        compute_sharpness_din45692(specific, axis, times)


@pytest.mark.parametrize("signal", [np.zeros(0), np.zeros(9599), np.ones((2, 9600)), np.full(9600, np.nan)])
def test_roughness_rejects_empty_short_multichannel_or_nonfinite_input(signal):
    with pytest.raises(ValueError):
        compute_roughness_daniel_weber(signal, 48000)


def test_roughness_minimum_frame_and_silence():
    result = compute_roughness_daniel_weber(np.zeros(9600), 48000)
    np.testing.assert_array_equal(result.roughness_asper, [0.0])
    assert result.specific_roughness_asper_per_bark.shape == (47, 1)
    assert result.summary.r5_asper == 0.0
