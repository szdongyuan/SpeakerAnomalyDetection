import math

from base.hardware_management import HardwareManagementRepository
from base.soundcard_calibration_manager import load_mic_channel_v2pa_factors


def _warn(logger, message):
    if logger is not None and hasattr(logger, "warning"):
        logger.warning(message)


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _finite_positive_float(value):
    number = _finite_float(value)
    if number is None or number <= 0:
        return None
    return number


def _uncalibrated_metadata(input_channels):
    return {
        "recorded_channels": [
            {
                "wav_channel_index": wav_channel_index,
                "v2pa_factor": None,
                "standard_spl": None,
                "calibrated": False,
            }
            for wav_channel_index, _channel_index in enumerate(input_channels)
        ]
    }


def load_mic_channel_standard_spl(hardware_id=None, db_path=None):
    rows = HardwareManagementRepository(db_path).list_channel_calibrations(
        hardware_id,
        "input",
        "mic_v2pa",
    )
    standard_spl = {}
    for row in rows:
        try:
            channel_index = int(row["channel_index"])
        except (KeyError, TypeError, ValueError):
            continue
        spl = _finite_float(row.get("standard_spl"))
        if spl is not None:
            standard_spl[channel_index] = spl
    return standard_spl


def build_recording_wav_calibration_metadata(input_channels, hardware_id=None, db_path=None, logger=None):
    try:
        normalized_channels = [int(channel) for channel in input_channels]
    except (TypeError, ValueError):
        normalized_channels = []

    try:
        v2pa_factors = load_mic_channel_v2pa_factors(hardware_id=hardware_id, db_path=db_path)
        standard_spl = load_mic_channel_standard_spl(hardware_id=hardware_id, db_path=db_path)
    except Exception as exc:
        _warn(logger, f"Failed to build WAV calibration metadata; channels marked uncalibrated. {exc}")
        return _uncalibrated_metadata(normalized_channels)

    recorded_channels = []
    for wav_channel_index, input_channel in enumerate(normalized_channels):
        factor = _finite_positive_float(v2pa_factors.get(input_channel))
        spl = _finite_float(standard_spl.get(input_channel))
        calibrated = factor is not None
        recorded_channels.append(
            {
                "wav_channel_index": wav_channel_index,
                "v2pa_factor": factor if calibrated else None,
                "standard_spl": spl if calibrated else None,
                "calibrated": calibrated,
            }
        )
    return {"recorded_channels": recorded_channels}
