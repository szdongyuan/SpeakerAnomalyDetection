"""Viewport clipping preserves intersections without bridging missing data."""

import numpy as np
import pytest

from ui.plotting.geometry import clip_curve_x


@pytest.mark.parametrize('x,y,bounds,expected_x,expected_y', [
    ([0., 1.], [40., 40.], (.4, .6), [.4, .6], [40., 40.]),
    ([0., 1.], [0., 10.], (.4, .6), [.4, .6], [4., 6.]),
    ([1., 0.], [10., 0.], (.4, .6), [.6, .4], [6., 4.]),
    ([0., .5, 1.], [0., np.nan, 10.], (.4, .6), [.5], [np.nan]),
    ([0., 1.], [np.nan, 10.], (.4, .6), [], []),
    ([0., .5, 1.], [np.nan, 3., np.nan], (.4, .6), [.5], [3.]),
    ([.4, .6], [4., 6.], (.4, .6), [.4, .6], [4., 6.]),
    ([0., .2], [0., 2.], (.4, .6), [], []),
    ([], [], (.4, .6), [], []),
    ([.5], [3.], (.4, .6), [.5], [3.]),
    ([.5, 1., .5], [1., 2., 3.], (.4, .6), [.5, .6, .6, .6, .5],
     [1., 1.2, np.nan, 2.8, 3.]),
])
def test_clip_preserves_geometry_and_source(x, y, bounds, expected_x, expected_y):
    x, y = np.array(x), np.array(y)
    before_x, before_y = x.copy(), y.copy()
    actual_x, actual_y = clip_curve_x(x, y, bounds)
    np.testing.assert_allclose(actual_x, expected_x)
    np.testing.assert_allclose(actual_y, expected_y)
    np.testing.assert_array_equal(x, before_x)
    np.testing.assert_array_equal(y, before_y)


@pytest.mark.parametrize('log_x,log_y,bounds,expected_y', [
    (True, False, (10., 100.), [50.5, 100.]),
    (False, True, (50.5, 100.), [10., 100.]),
    (True, True, (10., 100.), [10., 100.]),
])
def test_clip_interpolates_in_display_axis_coordinates(log_x, log_y, bounds, expected_y):
    actual_x, actual_y = clip_curve_x(np.array([1., 100.]), np.array([1., 100.]),
                                      bounds, log_x=log_x, log_y=log_y)
    np.testing.assert_allclose(actual_x, bounds)
    np.testing.assert_allclose(actual_y, expected_y)
