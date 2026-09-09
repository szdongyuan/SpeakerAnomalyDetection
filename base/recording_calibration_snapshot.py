from collections.abc import Mapping
from numbers import Integral

from base.soundcard_calibration_manager import load_mic_channel_calibrations
from base.ve3668n_wav_metadata import build_ve_recording_metadata
from consts.ve3668n_consts import VE_BACKEND


def _normalize_input_channels(input_channels):
    if not isinstance(input_channels, (list, tuple)) or not input_channels:
        raise ValueError("Physical input channels must be a non-empty sequence.")

    normalized = []
    for channel in input_channels:
        if isinstance(channel, bool) or not isinstance(channel, Integral):
            raise ValueError(
                "Each physical input channel must be a non-negative integer."
            )
        channel_index = int(channel)
        if channel_index < 0 or channel_index in normalized:
            raise ValueError(
                "Physical input channels must be non-negative and unique."
            )
        normalized.append(channel_index)
    return normalized


def build_recording_wav_calibration_metadata(
    input_channels,
    input_device=None,
    calibration_path=None,
    *,
    ve_calibration_store=None,
):
    """Build one WAV-local snapshot from the current physical-channel registry."""
    if isinstance(input_device, Mapping) and input_device.get("backend") == VE_BACKEND:
        return build_ve_recording_metadata(
            input_device, input_channels, input_device.get("input_config"), ve_calibration_store,
        )
    normalized_channels = _normalize_input_channels(input_channels)
    calibrations = load_mic_channel_calibrations(input_device, calibration_path)

    recorded_channels = []
    for wav_channel_index, physical_channel in enumerate(normalized_channels):
        record = calibrations.get(physical_channel)
        if record is None:
            recorded_channels.append(
                {
                    "wav_channel_index": wav_channel_index,
                    "physical_input_channel": physical_channel,
                    "v2pa_factor": None,
                    "standard_spl": None,
                    "calibrated": False,
                }
            )
            continue

        recorded_channels.append(
            {
                "wav_channel_index": wav_channel_index,
                "physical_input_channel": physical_channel,
                "v2pa_factor": record["v2pa_factor"],
                "standard_spl": record["standard_spl_db"],
                "calibrated": True,
            }
        )

    return {"recorded_channels": recorded_channels}
