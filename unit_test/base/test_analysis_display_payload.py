import numpy as np

from base.analysis_display_payload import min_max_envelope, sample_curve_at_interval


def test_min_max_envelope_is_bounded_and_retains_spike():
    x = np.arange(100_000, dtype=np.float64)
    y = np.zeros_like(x)
    y[51_234] = 99.0

    display_x, display_y = min_max_envelope(x, y, max_points=20_000)

    assert len(display_x) <= 20_000
    assert len(display_x) == len(display_y)
    assert 99.0 in display_y
    assert display_x == sorted(display_x)


def test_fixed_interval_sampling_uses_ten_milliseconds():
    x = np.arange(0.0, 1.0001, 0.001)
    y = x * 2.0
    sampled_x, sampled_y = sample_curve_at_interval(x, y)

    assert len(sampled_x) == 101
    assert np.allclose(np.diff(sampled_x), 0.01)
    assert np.allclose(sampled_y, np.asarray(sampled_x) * 2.0)
