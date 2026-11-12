import numpy as np
import pytest

from ui.sequence.curve_sampling import select_display_indices
from ui.sequence.sampled_plot_curve import SampledPlotCurve


MIN_MAX = {'method': 'preserve_min_max', 'pixel_bucket_width': 1}


@pytest.mark.parametrize('size', [0, 1, 2, 11, 100])
def test_none_preserves_all_points(size):
    x = np.arange(size, dtype=float)
    selected = select_display_indices(x, x, {'method': 'none'}, (0, 100), 10)
    np.testing.assert_array_equal(selected, np.arange(size))


def test_every_nth_keeps_original_grid_endpoints_and_missing_breaks():
    x = np.arange(26, dtype=float)
    y = x.copy()
    y[11:13] = np.nan
    selected = select_display_indices(x, y, {
        'method': 'keep_every_nth_point', 'keep_one_point_every_n_points': 10,
    }, (0, 25), 100)
    np.testing.assert_array_equal(selected, [0, 10, 11, 12, 13, 20, 25])
    assert np.isnan(y[selected][2:4]).all()


def test_pixel_buckets_preserve_spikes_original_positions_and_order():
    x = np.linspace(0, 10, 10001)
    y = np.zeros(len(x))
    y[1234], y[1235] = 100, -50
    y[5000] = np.nan
    selected = select_display_indices(x, y, MIN_MAX, (0, 10), 100)
    assert {0, 1234, 1235, 4999, 5000, 5001, 10000} <= set(selected)
    assert np.all(np.diff(selected) > 0)
    assert len(selected) < 450
    # Zoom onto 11 points: all can now occupy separate screen pixels.
    zoomed = select_display_indices(x, y, MIN_MAX, (1.23, 1.24), 100)
    assert set(range(1230, 1241)) <= set(zoomed)


def test_log_mapped_frequency_and_reverse_order_are_not_resorted():
    frequency = np.geomspace(20, 20000, 10001)
    mapped = np.log10(frequency)
    y = np.sin(np.arange(len(mapped)))
    selected = select_display_indices(mapped, y, MIN_MAX, (mapped[0], mapped[-1]), 30)
    assert len(selected) <= 4 * 32
    reverse = select_display_indices(mapped[::-1], y[::-1], MIN_MAX, (mapped[0], mapped[-1]), 30)
    assert np.all(np.diff(reverse) > 0)


def test_long_missing_region_only_needs_its_endpoints():
    x = np.arange(100000, dtype=float)
    selected = select_display_indices(x, np.full(x.size, np.nan), MIN_MAX, (0, 100000), 100)
    np.testing.assert_array_equal(selected, [0, 99999])


def test_full_bounds_remain_when_every_nth_omits_a_spike(qt_app):
    x = np.arange(101, dtype=float)
    y = np.ones(101)
    y[51] = 99
    line = SampledPlotCurve(x, y, {
        'method': 'keep_every_nth_point', 'keep_one_point_every_n_points': 10,
    })
    line.refresh_display((0, 100), 100)
    assert max(line.yData) == 1
    assert line.dataBounds(1) == (1, 99)
    line.setLogMode(False, True)
    assert line.dataBounds(1) == pytest.approx((0, np.log10(99)))
    np.testing.assert_array_equal(line.full_y, y)
