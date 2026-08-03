import numpy as np

from base.pre_processing.spl_runtime_config import (
    calculate_overall_spl,
    resolve_spl_unit,
)


def test_calculate_overall_spl_uses_rms_and_calibration_multiplier():
    signal = np.array([1.0, -1.0])

    overall_spl = calculate_overall_spl(
        signal,
        v2pa_factor=2.0,
    )

    assert np.isclose(overall_spl, 100.0)


def test_calculate_overall_spl_returns_nan_for_empty_signal():
    assert np.isnan(calculate_overall_spl([]))


def test_resolve_spl_unit_matches_frequency_weighting():
    assert resolve_spl_unit("Z") == "dB"
    assert resolve_spl_unit("A") == "dBA"
    assert resolve_spl_unit("B") == "dBB"
    assert resolve_spl_unit("C") == "dBC"
    assert resolve_spl_unit("D") == "dBD"
