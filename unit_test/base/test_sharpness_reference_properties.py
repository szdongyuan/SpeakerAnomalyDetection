"""Analytical DIN weighting cases independent of the project's weight table."""
import math

import numpy as np
import pytest

from base.core_algorithm.sound_quality.sharpness import compute_sharpness_din45692


@pytest.mark.parametrize("bark", [10., 20.])
def test_single_bark_band_has_known_normalized_sharpness(bark):
    z = np.arange(1, 241) / 10
    specific = np.zeros((240, 3))
    specific[round(bark * 10) - 1] = [2., 20., 200.]
    result = compute_sharpness_din45692(specific, z, np.array([.1, .102, .104]))
    weight = 1. if bark <= 15.8 else .15 * math.exp(.42 * (bark - 15.8)) + .85
    expected = .11 * bark * weight
    np.testing.assert_allclose(result.sharpness_acum, expected, rtol=1e-14)
    assert result.summary.s_mean == pytest.approx(expected)
    assert result.summary.s_stationary == pytest.approx(expected)


def test_time_skip_excludes_startup_only_from_summary():
    z = np.arange(1, 241) / 10
    specific = np.zeros((240, 3))
    specific[199, 0] = 10.
    specific[99, 1:] = 10.
    result = compute_sharpness_din45692(specific, z, np.array([.05, .1, .102]))
    assert result.sharpness_acum[0] > 1.1
    assert result.summary.s_max == pytest.approx(1.1)
    assert result.summary.s_mean == pytest.approx(1.1)


def test_below_floor_and_short_signals_have_no_fabricated_summary():
    z = np.arange(1, 241) / 10
    specific = np.zeros((240, 3))
    specific[99] = [0, .5, 10]
    result = compute_sharpness_din45692(specific, z, np.array([0., .02, .04]))
    assert np.isnan(result.sharpness_acum[:2]).all()
    assert result.sharpness_acum[-1] == pytest.approx(1.1)
    assert np.isnan(result.summary.s_max) and np.isnan(result.summary.s_mean)
