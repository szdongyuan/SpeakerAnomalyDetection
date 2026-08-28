from fractions import Fraction

import pytest

from consts.harmonic_detection_consts import (
    HARMONIC_DETECTION_METHOD_DEFAULT,
    HARMONIC_DETECTION_METHOD_FOURIER,
    HARMONIC_DETECTION_METHOD_KEY,
    HARMONIC_DETECTION_METHOD_LABELS,
    HARMONIC_DETECTION_METHOD_SYNCHRONOUS,
    normalize_harmonic_detection_method,
)
from ui.ui_analysis_config.config_normalization import (
    CONFIG_CONCEPTS,
    normalize_analysis_channel,
    normalize_analysis_channels,
    normalize_octave_smoothing,
    normalize_time_smoothing,
    normalize_weighting,
    weighting_to_display_label,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "Z"),
        ("", "Z"),
        ("None", "Z"),
        ("Z（None）", "Z"),
        ("z", "Z"),
        ("A", "A"),
        ("b", "B"),
        ("bad", "Z"),
    ],
)
def test_normalize_weighting_returns_canonical_value(value, expected):
    assert normalize_weighting(value) == expected


def test_normalize_weighting_honors_valid_default():
    assert normalize_weighting("bad", default="A") == "A"


def test_weighting_to_display_label_uses_legacy_none_label():
    assert weighting_to_display_label("None") == "Z（None）"
    assert weighting_to_display_label("C") == "C"


def test_normalize_octave_smoothing_prefers_explicit_key():
    cfg = {"octave_smoothing": 3, "smooth_checked": True}

    assert normalize_octave_smoothing(cfg) == 3


def test_normalize_octave_smoothing_supports_legacy_smooth_checked():
    assert normalize_octave_smoothing({"smooth_checked": True}) == 6
    assert normalize_octave_smoothing({"smooth_checked": False}) == 0


def test_normalize_octave_smoothing_rejects_unsupported_values():
    assert normalize_octave_smoothing({"octave_smoothing": 5}, default=0) == 0
    assert normalize_octave_smoothing({"octave_smoothing": "bad"}, default=3) == 3


def test_normalize_time_smoothing_returns_stable_internal_shape():
    cfg = {
        "smooth_enabled": True,
        "smooth_unit": "points",
        "smooth_time_sec": "0.125",
        "smooth_points": "32",
        "smooth_algo": "3",
    }

    assert normalize_time_smoothing(cfg) == {
        "enabled": True,
        "unit": "points",
        "time_sec": 0.125,
        "points": 32,
        "algo": 3,
    }


def test_normalize_time_smoothing_accepts_legacy_smooth_checked():
    normalized = normalize_time_smoothing({"smooth_checked": True})

    assert normalized["enabled"] is True
    assert normalized["unit"] == "time"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (127, 127),
        (2.0, 2),
        (127.0, 127),
        ("2", 2),
        (" 127 ", 127),
    ],
)
def test_normalize_analysis_channel_accepts_compatible_zero_based_values(value, expected):
    assert normalize_analysis_channel({"analysis_channel": value}) == expected


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, [0]),
        ({"analysis_channel": "2"}, [2]),
        ({"analysis_channel": 7, "analysis_channels": [2, 0, 2]}, [0, 2]),
        ({"analysis_channel": 7, "analysis_channels": []}, [7]),
        ({"analysis_channel": 7, "analysis_channels": "0,2"}, [7]),
        ({"analysis_channels": [True, -1, 128, 1.5, "bad", "2", 127]}, [2, 127]),
        ({"analysis_channels": [float("nan"), float("inf"), None]}, [0]),
        ({"analysis_channels": (7, 2)}, [2, 7]),
    ],
)
def test_normalize_recorded_channel_list(config, expected):
    assert normalize_analysis_channels(config) == expected


def test_scalar_normalizer_ignores_recorded_channel_list():
    assert normalize_analysis_channel({
        "analysis_channel": 1,
        "analysis_channels": [0, 2],
    }) == 1


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        -1,
        128,
        2.5,
        float("inf"),
        float("-inf"),
        float("nan"),
        "+2",
        "-2",
        "",
        "   ",
        "2.0",
        "channel 2",
        None,
    ],
)
def test_normalize_analysis_channel_rejects_out_of_contract_values(value):
    assert normalize_analysis_channel({"analysis_channel": value}) == 0


def test_normalize_analysis_channel_defaults_missing_config_to_zero():
    assert normalize_analysis_channel({}) == 0
    assert normalize_analysis_channel(None) == 0


def test_normalize_analysis_channel_ignores_hardware_availability():
    assert normalize_analysis_channel({"analysis_channel": 127}, [0, 2]) == 127


def test_normalize_analysis_channel_rejects_oversized_decimal_without_raising():
    assert normalize_analysis_channel({"analysis_channel": "9" * 5000}) == 0


def test_normalize_analysis_channel_rejects_overflowing_real_without_raising():
    assert normalize_analysis_channel(
        {"analysis_channel": Fraction(10**1000, 1)}
    ) == 0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, HARMONIC_DETECTION_METHOD_SYNCHRONOUS),
        ("", HARMONIC_DETECTION_METHOD_SYNCHRONOUS),
        ("synchronous", HARMONIC_DETECTION_METHOD_SYNCHRONOUS),
        ("同步检波", HARMONIC_DETECTION_METHOD_SYNCHRONOUS),
        ("sync", HARMONIC_DETECTION_METHOD_SYNCHRONOUS),
        ("fourier", HARMONIC_DETECTION_METHOD_FOURIER),
        ("傅里叶变换", HARMONIC_DETECTION_METHOD_FOURIER),
        ("fft", HARMONIC_DETECTION_METHOD_FOURIER),
        ("bad", HARMONIC_DETECTION_METHOD_SYNCHRONOUS),
    ],
)
def test_normalize_harmonic_detection_method_returns_canonical_value(value, expected):
    assert normalize_harmonic_detection_method(value) == expected


def test_normalize_harmonic_detection_method_strict_rejects_invalid_non_empty_value():
    with pytest.raises(ValueError, match="Unsupported harmonic detection method"):
        normalize_harmonic_detection_method("bad", strict=True)


def test_normalize_harmonic_detection_method_strict_allows_missing_default():
    assert normalize_harmonic_detection_method(None, strict=True) == HARMONIC_DETECTION_METHOD_DEFAULT


def test_harmonic_detection_constants_are_stable_serialized_boundaries():
    assert HARMONIC_DETECTION_METHOD_KEY == "harmonic_detection_method"
    assert HARMONIC_DETECTION_METHOD_DEFAULT == HARMONIC_DETECTION_METHOD_SYNCHRONOUS
    assert HARMONIC_DETECTION_METHOD_LABELS[HARMONIC_DETECTION_METHOD_SYNCHRONOUS] == "同步检波"
    assert HARMONIC_DETECTION_METHOD_LABELS[HARMONIC_DETECTION_METHOD_FOURIER] == "傅里叶变换"


def test_config_concepts_document_step_one_taxonomy():
    expected_concepts = {
        "analysis_channel",
        "weighting",
        "frequency_smoothing",
        "time_smoothing",
        "threshold_curve",
        "reference_threshold",
        "golden_sample",
        "harmonic_selection",
        "harmonic_detection_method",
    }

    assert expected_concepts.issubset(CONFIG_CONCEPTS)
    assert "octave_smoothing" in CONFIG_CONCEPTS["frequency_smoothing"]["legacy_keys"]
    assert "smooth_enabled" in CONFIG_CONCEPTS["time_smoothing"]["legacy_keys"]
    assert HARMONIC_DETECTION_METHOD_KEY in CONFIG_CONCEPTS["harmonic_detection_method"]["legacy_keys"]
