"""Validate recording-local physical channel provenance for completion."""
from base.ve3668n_wav_metadata import validate_ve_wav_metadata
from base.wav_calibration_metadata import WavCalibrationMetadataReadStatus

def verified_recorded_channels(source):
    """Expose physical mapping only after the complete source validates."""
    if source.status is not WavCalibrationMetadataReadStatus.VALID:
        return None
    try:
        normalized = validate_ve_wav_metadata(source.metadata)
    except ValueError:
        return None
    return tuple(item["physical_input_channel"] for item in sorted(
        normalized["recorded_channels"], key=lambda item: item["wav_channel_index"]))
