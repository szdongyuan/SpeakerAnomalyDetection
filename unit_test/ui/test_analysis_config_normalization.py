import pytest

from ui.ui_analysis_config.config_normalization import (
    CONFIG_CONCEPTS,
    normalize_analysis_channel,
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


def test_normalize_analysis_channel_uses_selected_available_channel():
    assert normalize_analysis_channel({"analysis_channel": "2"}, [0, 2, 4]) == 2


@pytest.mark.parametrize(
    ("cfg", "available_channels", "expected"),
    [
        ({"analysis_channel": "bad"}, [3, 5], 3),
        ({"analysis_channel": 4}, [0, 1], 0),
        ({}, [2, 1], 1),
        ({"analysis_channel": 5}, ["bad", 5], 5),
        ({"analysis_channel": 5}, ["bad"], 0),
        ({"analysis_channel": 9}, [], 0),
        (None, None, 0),
    ],
)
def test_normalize_analysis_channel_falls_back_safely(cfg, available_channels, expected):
    assert normalize_analysis_channel(cfg, available_channels) == expected


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
    }

    assert expected_concepts.issubset(CONFIG_CONCEPTS)
    assert "octave_smoothing" in CONFIG_CONCEPTS["frequency_smoothing"]["legacy_keys"]
    assert "smooth_enabled" in CONFIG_CONCEPTS["time_smoothing"]["legacy_keys"]
