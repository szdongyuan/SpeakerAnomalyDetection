import math
import os
from datetime import datetime

from base.data_struct.data_deal_struct import DataDealStruct
from base.recording_settings import (
    resolve_monitor_fade_in_ms,
    resolve_startup_trim_ms,
)
from base.system_intervction.hardware_intervction import get_mac_address
from base.streaming_audio_processor import StreamingAudioProcessor
from consts import error_code, model_consts

data_struct = DataDealStruct()


def _coerce_sample_rate(sample_rate) -> float:
    """Validate ``sample_rate`` for ms->samples conversion.

    Returns the rate as a positive finite float, or ``0.0`` to signal
    "bad input -- skip the conversion". Centralised so every resolver
    below treats an uninitialised / negative / NaN sample rate the
    same way (no samples emitted) instead of each one inventing its
    own guard.
    """
    try:
        sr = float(sample_rate)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(sr) or sr <= 0:
        return 0.0
    return sr


def resolve_startup_trim_samples(acq_detail, sample_rate) -> int:
    """Return the number of leading samples to drop from one recording.

    Resolves ``startup_trim_ms`` through the global recording-settings
    file (see :mod:`base.recording_settings`) plus an optional
    per-product override in ``acq_detail``. The shared resolver enforces
    the precedence rules; this helper just turns the resulting
    millisecond value into a sample count at the given rate.

    Returns ``0`` (no trim) when the resolved trim is zero or when the
    sample rate is missing / invalid -- in that case the recording is
    written verbatim, which is the safer failure mode than silently
    eating samples we cannot accurately count.
    """
    trim_ms = resolve_startup_trim_ms(acq_detail)
    if trim_ms <= 0:
        return 0
    sr = _coerce_sample_rate(sample_rate)
    if sr <= 0:
        return 0
    return int(round(trim_ms * sr / 1000.0))


def resolve_monitor_fade_in_samples(acq_detail, sample_rate) -> int:
    """Return the linear-ramp length, in samples, for monitor fade-in.

    The fade smooths the 0 -> signal transition at the end of the
    monitor-mute window (see ``startup_trim_ms``) so the operator does
    not hear a click when live monitoring resumes. Resolved with the
    same precedence as :func:`resolve_startup_trim_samples`.

    Returns ``0`` only when the resolver produced zero (operator
    explicitly opted out) or the sample rate is invalid. A positive
    value below the one-sample threshold is rounded up to ``1`` so a
    user setting e.g. ``monitor_fade_in_ms: 0.01`` still gets at least
    one ramped sample rather than silently degrading to a hard cut.
    """
    fade_ms = resolve_monitor_fade_in_ms(acq_detail)
    if fade_ms <= 0:
        return 0
    sr = _coerce_sample_rate(sample_rate)
    if sr <= 0:
        return 0
    return max(1, int(round(fade_ms * sr / 1000.0)))


def get_recorded_info(product_model, product_number, barcode, label, name_suffix=""):
    """
        Generate recorded information.

        This function generates a recording file name based on the current date, MAC address, product model,
    barcode, and direction suffix.
        It also constructs the path for the recording file. Additionally, it creates a dictionary containing the
    recording file path and product information.

        Returns:
            tuple: A tuple containing the recording file path and a dictionary with recording information.
    """
    now = datetime.now()
    recording_time = now.strftime("%Y-%m-%d")
    recording_time_for_name = now.strftime("%Y-%m-%d-%H-%M-%S")
    mac_address = get_mac_address()
    mac_address = mac_address.replace(":", "") if mac_address else None

    recorded_name = product_model + "_" + recording_time_for_name + "_" + mac_address
    if barcode:
        recorded_name = recorded_name + "_BC" + barcode
    else:
        barcode = None
    if name_suffix:
        recorded_name = recorded_name + str(name_suffix)
    recorded_name = recorded_name + ".wav"
    store_record_dir = model_consts.STORED_RECORDED_PATH + "/" + label
    if not os.path.exists(store_record_dir):
        os.makedirs(store_record_dir)
    recorded_path = store_record_dir + "/" + recorded_name
    recorded_signal_info = {
        "file_path": recorded_path,
        "product_model": product_model,
        "record_date": recording_time,
        "barcode": barcode,
        "labels": label,
        "record_name_suffix": str(name_suffix or ""),
    }

    return recorded_path, recorded_signal_info


def stream_record_without_play(recorded_dict, recorded_path, recorded_signal_info):
    """
    Start streaming recording (non-blocking).

    Returns StreamingAudioProcessor instance for UI to manage lifecycle.
    File writing, data collection, and database operations handled by UI layer.

    Args:
        recorded_dict (dict): Recording parameters containing:
            - 'num_frames': Total number of frames to record
            - 'sample_rate': Sample rate in Hz
            - 'channels': Number of channels (default: 1)
            - 'device': Input device (optional)
        recorded_path (str): Path where WAV file will be saved (managed by UI)
        recorded_signal_info (dict): Recording metadata (saved by UI after completion)

    Returns:
        tuple: (StreamingAudioProcessor instance, sample_rate)
    """
    sample_rate = recorded_dict.get("sample_rate", data_struct.sample_rate)
    num_frames = recorded_dict.get("num_frames", 441000)
    device = recorded_dict.get("device")
    input_channels = recorded_dict.get("input_channels")
    output_device = recorded_dict.get("output_device")
    raw_output_channels = recorded_dict.get("output_channels")
    if isinstance(raw_output_channels, (list, tuple)):
        output_channels = []
        for ch in raw_output_channels:
            try:
                output_channels.append(int(ch))
            except Exception:
                continue
    elif raw_output_channels is None:
        output_channels = []
    else:
        try:
            output_channels = [int(raw_output_channels)]
        except Exception:
            output_channels = []
    monitor_playback = recorded_dict.get("monitor_playback", False)
    monitor_gain_db = float(recorded_dict.get("monitor_gain_db", 0.0))
    # Mute the first N samples on the monitor output so the operator does
    # not hear the sound-card power-on pop during real-time monitoring.
    # Same sample count the caller uses for the post-recording WAV trim
    # (both derived from ``startup_trim_ms`` via
    # :func:`resolve_startup_trim_samples`), so monitor and stored WAV stay
    # consistent: if the config opts in to trimming, the pop is suppressed
    # everywhere it could be heard or read.
    monitor_mute_leading_samples = int(
        recorded_dict.get("monitor_mute_leading_samples", 0) or 0
    )
    # Linear-ramp length applied at the very end of the mute window. See
    # :func:`resolve_monitor_fade_in_samples` for the resolution rules and
    # ``StreamingAudioProcessor._apply_monitor_startup_mute`` for the
    # actual ramp logic. Forwarded here (rather than re-resolved inside
    # the processor) so the caller controls the precise sample count and
    # so the processor stays free of config-loading concerns.
    monitor_fade_in_samples = int(
        recorded_dict.get("monitor_fade_in_samples", 0) or 0
    )

    # Create streaming processor
    processor = StreamingAudioProcessor()

    # Start streaming recording (non-blocking) with exact sample count
    record_code, msg = processor.start_streaming_rec(
        sample_rate=sample_rate,
        target_samples=num_frames,  # Use exact sample count instead of duration
        device=device,
        input_channels=input_channels,
        output_device=output_device,
        output_channels=output_channels,
        monitor_playback=monitor_playback,
        monitor_gain_db=monitor_gain_db,
        monitor_mute_leading_samples=monitor_mute_leading_samples,
        monitor_fade_in_samples=monitor_fade_in_samples,
    )

    if record_code == error_code.OK:
        # Return processor for UI to manage (don't block!)
        return processor, sample_rate
    else:
        raise RuntimeError(f"Failed to start streaming recording: {msg}")
