from types import MappingProxyType

import pytest

from ui.sequence.analysis_channel_preflight import (
    PASSIVE_CHANNEL_ANALYSIS_TYPES,
    REQUIRED_CHANNEL_ANALYSIS_TYPES,
    preflight_analysis_channels,
)


def _config(*items):
    return {
        "display_sequence": [key for key, _item_type, _params in items],
        **{
            key: {"type": item_type, **params}
            for key, item_type, params in items
        },
    }


def test_live_preflight_maps_physical_channels_and_returns_ordered_skips():
    result = preflight_analysis_channels(
        _config(
            ("spl-valid", "SPL", {"analysis_channel": 2}),
            ("spec-missing", "Spec", {"analysis_channel": 1}),
            ("fba-missing", "FBA", {"analysis_channel": 3}),
        ),
        active_input_channels=[0, 2],
    )

    assert result.local_channels == {"spl-valid": 1}
    assert isinstance(result.local_channels, MappingProxyType)
    assert [skip.item_key for skip in result.skipped] == [
        "spec-missing",
        "fba-missing",
    ]
    assert result.skipped[0].requested_channel == 1
    assert result.skipped[0].available_channels == (0, 2)
    assert "In2" in result.skipped[0].reason
    assert "In1、In3" in result.skipped[0].reason
    with pytest.raises(TypeError):
        result.local_channels["new"] = 0


def test_import_preflight_uses_wav_local_columns():
    result = preflight_analysis_channels(
        _config(
            ("spl-left", "SPL", {"analysis_channel": 0}),
            ("spec-right", "Spec", {"analysis_channel": 1}),
            ("fba-missing", "FBA", {"analysis_channel": 2}),
        ),
        active_input_channels=[9, 12],
        imported_channel_count=2,
    )

    assert result.local_channels == {"spl-left": 0, "spec-right": 1}
    assert result.skipped[0].available_channels == (0, 1)
    assert result.skipped[0].requested_channel == 2


def test_preflight_only_handles_explicit_channel_protocol_types():
    assert REQUIRED_CHANNEL_ANALYSIS_TYPES == frozenset({"SPL", "Spec", "FBA"})
    assert PASSIVE_CHANNEL_ANALYSIS_TYPES == frozenset({"AI", "FFT", "LOUD"})

    result = preflight_analysis_channels(
        _config(
            ("rsc", "RSC", {"analysis_channel": 9}),
            ("lp", "LP", {"analysis_channel": 9}),
            ("excel", "Excel", {"analysis_channel": 9}),
            ("future", "FUTURE", {"analysis_channel": 9}),
        ),
        active_input_channels=[0],
    )

    assert result.local_channels == {}
    assert result.skipped == ()


@pytest.mark.parametrize("item_type", ["SPL", "Spec", "FBA", "AI", "LP", "FFT", "LOUD"])
def test_recorded_channel_list_keeps_valid_channels_when_one_is_missing(item_type):
    result = preflight_analysis_channels(
        _config(("item", item_type, {"analysis_channels": [7, 0, 2]})),
        active_input_channels=[2, 0],
    )

    assert result.local_channels == {"item--通道1": 1, "item--通道3": 0}
    assert result.fully_skipped_items == ()
    assert len(result.skipped) == 1
    assert result.skipped[0].item_key == "item--通道8"
    assert result.skipped[0].config_key == "item"


def test_all_recorded_channels_missing_marks_base_item_as_fully_skipped():
    result = preflight_analysis_channels(
        _config(("item", "SPL", {"analysis_channels": [2, 7]})),
        active_input_channels=[0],
    )

    assert result.local_channels == {}
    assert result.fully_skipped_items == ("item",)
    assert [skip.item_key for skip in result.skipped] == ["item--通道3", "item--通道8"]


def test_import_preflight_ignores_recorded_channel_list():
    result = preflight_analysis_channels(
        _config(("item", "SPL", {"analysis_channel": 1, "analysis_channels": [0, 7]})),
        active_input_channels=[0, 7],
        imported_channel_count=2,
    )

    assert result.local_channels == {"item": 1}
    assert result.skipped == ()


@pytest.mark.parametrize(
    "malformed",
    [None, True, -1, 128, 1.5, float("inf"), "bad", object()],
)
def test_malformed_analysis_channel_uses_normalized_default(malformed):
    result = preflight_analysis_channels(
        _config(("spl", "SPL", {"analysis_channel": malformed})),
        active_input_channels=[0, 2],
    )

    assert result.local_channels == {"spl": 0}
    assert result.skipped == ()
