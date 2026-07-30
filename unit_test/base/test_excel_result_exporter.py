import numpy as np

from base.excel_result_exporter import _export_unit, _extract_curve_xy


def test_extract_curve_xy_prefers_raw_keys_when_present():
    result = {
        "freq_value": [1.0, 2.0, 3.0],
        "thd": [10.0, 11.0, 12.0],
        "thd_raw": [100.0, 110.0, 120.0],
    }
    x, y = _extract_curve_xy(result)
    assert x == [1.0, 2.0, 3.0]
    assert y == [100.0, 110.0, 120.0]


def test_extract_curve_xy_falls_back_to_display_keys_when_raw_missing():
    result = {"freq_value": [1.0, 2.0], "thd": [10.0, 11.0]}
    x, y = _extract_curve_xy(result)
    assert x == [1.0, 2.0]
    assert y == [10.0, 11.0]


def test_extract_curve_xy_handles_numpy_arrays_and_prefers_raw():
    result = {
        "frequency_list": np.asarray([100.0, 200.0], dtype=float),
        "fr": np.asarray([-1.0, -2.0], dtype=float),
        "fr_raw": np.asarray([-10.0, -20.0], dtype=float),
    }
    x, y = _extract_curve_xy(result)
    assert x == [100.0, 200.0]
    assert y == [-10.0, -20.0]


def test_extract_curve_xy_supports_fba_weighted_band_levels():
    result = {
        "band_centers": [100.0, 1000.0],
        "band_levels_db": [60.0, 70.0],
        "band_levels_weighted_db": [41.0, 70.0],
    }

    x, y = _extract_curve_xy(result)

    assert x == [100.0, 1000.0]
    assert y == [41.0, 70.0]
    assert _export_unit("FBA", {}) == "dB"


def test_extract_curve_xy_supports_fft_display_curve():
    result = {
        "frequency_bins": [100.0, 1000.0],
        "fft_db": [60.0, 70.0],
        "plot_db": [-2.0, 3.0],
    }

    x, y = _extract_curve_xy(result)

    assert x == [100.0, 1000.0]
    assert y == [-2.0, 3.0]
    assert _export_unit("FFT", {}) == "dB"

