import pytest

from base.recording_preview_config import (
    resolve_recording_preview_time_mode,
    validate_recording_preview_time_mode,
)
from consts.recording_preview_consts import (
    PREVIEW_TIME_MODE_CUMULATIVE,
    PREVIEW_TIME_MODE_RELATIVE_LATEST,
    RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY,
)


def test_missing_preview_mode_defaults_to_relative_latest():
    assert (
        resolve_recording_preview_time_mode({})
        == PREVIEW_TIME_MODE_RELATIVE_LATEST
    )


@pytest.mark.parametrize(
    "value",
    [PREVIEW_TIME_MODE_RELATIVE_LATEST, PREVIEW_TIME_MODE_CUMULATIVE],
)
def test_legal_preview_mode_is_preserved(value):
    assert validate_recording_preview_time_mode(value) == value
    assert (
        resolve_recording_preview_time_mode(
            {RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY: value}
        )
        == value
    )


@pytest.mark.parametrize(
    "value",
    [None, True, False, 1, 1.0, "", "Relative_Latest", "CUMULATIVE", "unknown"],
)
def test_explicit_invalid_preview_mode_is_rejected(value):
    with pytest.raises(ValueError, match="recording_preview_time_mode"):
        resolve_recording_preview_time_mode(
            {RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY: value}
        )


@pytest.mark.parametrize("detail", [None, True, 1, "", [], ()])
def test_non_mapping_acquisition_detail_is_rejected(detail):
    with pytest.raises(ValueError, match="must be a mapping"):
        resolve_recording_preview_time_mode(detail)
