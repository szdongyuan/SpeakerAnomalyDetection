import numpy as np
import pytest

from base.pre_processing.spl_runtime_config import (
    apply_spl_analysis_time_range,
    calculate_overall_spl,
    resolve_directional_additional_correction_db,
    resolve_free_field_distance_correction_db,
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


def test_free_field_distance_correction_uses_spherical_spreading():
    assert resolve_free_field_distance_correction_db(
        {
            "free_field_distance_enabled": True,
            "measurement_distance_m": 0.1,
            "target_distance_m": 1.0,
        }
    ) == pytest.approx(-20.0)
    assert resolve_free_field_distance_correction_db(
        {
            "free_field_distance_enabled": True,
            "measurement_distance_m": 1.0,
            "target_distance_m": 0.5,
        }
    ) == pytest.approx(6.020599913)
    assert resolve_free_field_distance_correction_db(
        {
            "free_field_distance_enabled": True,
            "measurement_distance_m": 1.0,
            "target_distance_m": 1.0,
        }
    ) == pytest.approx(0.0)


def test_free_field_distance_correction_is_backward_compatible_when_disabled():
    assert resolve_free_field_distance_correction_db({}) == 0.0
    assert resolve_free_field_distance_correction_db(
        {
            "free_field_distance_enabled": False,
            "measurement_distance_m": 0.0,
            "target_distance_m": float("nan"),
        }
    ) == 0.0


def test_directional_correction_is_independent_and_supports_signed_values():
    assert resolve_directional_additional_correction_db({}) == 0.0
    assert resolve_directional_additional_correction_db(
        {
            "free_field_distance_enabled": False,
            "directional_correction_enabled": True,
            "directional_additional_correction_db": -12.5,
        }
    ) == pytest.approx(-12.5)
    assert resolve_directional_additional_correction_db(
        {
            "directional_correction_enabled": True,
            "directional_additional_correction_db": 6.0,
        }
    ) == pytest.approx(6.0)
    assert resolve_directional_additional_correction_db(
        {
            "free_field_distance_enabled": True,
            "directional_correction_enabled": False,
            "directional_additional_correction_db": float("nan"),
        }
    ) == 0.0
    # 没有明确保存启用开关时，方向修正默认关闭。
    assert resolve_directional_additional_correction_db(
        {
            "free_field_distance_enabled": True,
            "directional_additional_correction_db": -3.0,
        }
    ) == 0.0
    assert resolve_directional_additional_correction_db(
        {
            "free_field_distance_enabled": False,
            "directional_additional_correction_db": float("nan"),
        }
    ) == 0.0

    with pytest.raises(ValueError, match="方向修正"):
        resolve_directional_additional_correction_db(
            {
                "free_field_distance_enabled": False,
                "directional_correction_enabled": True,
                "directional_additional_correction_db": float("nan"),
            }
        )


@pytest.mark.parametrize(
    ("measurement_distance_m", "target_distance_m"),
    [
        (0.0, 1.0),
        (-0.1, 1.0),
        (0.1, 0.0),
        (0.1, float("inf")),
        ("invalid", 1.0),
    ],
)
def test_free_field_distance_correction_rejects_invalid_enabled_distances(
    measurement_distance_m,
    target_distance_m,
):
    with pytest.raises(ValueError, match="距离"):
        resolve_free_field_distance_correction_db(
            {
                "free_field_distance_enabled": True,
                "measurement_distance_m": measurement_distance_m,
                "target_distance_m": target_distance_m,
            }
        )


def test_spl_analysis_time_range_slices_signal_and_returns_source_offset():
    signal = np.arange(100)
    config = {
        "analysis_time_range_enabled": True,
        "analysis_start_time_sec": 2.0,
        "analysis_end_time_sec": 6.0,
    }

    sliced, start_sample = apply_spl_analysis_time_range(
        signal,
        sample_rate=10,
        config=config,
    )

    assert start_sample == 20
    assert sliced.tolist() == list(range(20, 60))


def test_spl_analysis_time_range_keeps_original_signal_when_disabled_or_invalid():
    signal = np.arange(10)

    disabled, disabled_offset = apply_spl_analysis_time_range(
        signal,
        sample_rate=10,
        config={"analysis_time_range_enabled": False},
    )
    invalid, invalid_offset = apply_spl_analysis_time_range(
        signal,
        sample_rate=10,
        config={
            "analysis_time_range_enabled": True,
            "analysis_start_time_sec": 5.0,
            "analysis_end_time_sec": 2.0,
        },
    )

    assert disabled is signal
    assert disabled_offset == 0
    assert invalid is signal
    assert invalid_offset == 0


def test_spl_analysis_time_range_treats_zero_end_as_recording_end():
    signal = np.arange(10)

    sliced, start_sample = apply_spl_analysis_time_range(
        signal,
        sample_rate=10,
        config={
            "analysis_time_range_enabled": True,
            "analysis_start_time_sec": 0.3,
            "analysis_end_time_sec": 0.0,
        },
    )

    assert start_sample == 3
    assert sliced.tolist() == list(range(3, 10))
