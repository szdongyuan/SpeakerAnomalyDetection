from collections.abc import Mapping

from consts.recording_preview_consts import (
    PREVIEW_TIME_MODE_CUMULATIVE,
    PREVIEW_TIME_MODE_RELATIVE_LATEST,
    RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY,
)


def validate_recording_preview_time_mode(value):
    if type(value) is not str or value not in (
        PREVIEW_TIME_MODE_RELATIVE_LATEST,
        PREVIEW_TIME_MODE_CUMULATIVE,
    ):
        raise ValueError("recording_preview_time_mode 配置无效")
    return value


def resolve_recording_preview_time_mode(detail):
    if not isinstance(detail, Mapping):
        raise ValueError("recording acquisition detail must be a mapping")
    value = (
        detail[RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY]
        if RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY in detail
        else PREVIEW_TIME_MODE_RELATIVE_LATEST
    )
    return validate_recording_preview_time_mode(value)
