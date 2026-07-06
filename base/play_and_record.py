import os
from datetime import datetime

from base.data_struct.data_deal_struct import DataDealStruct
from base.system_intervction.hardware_intervction import get_mac_address
from base.pre_processing.split_repeat_signal import SplitRepeatSignal
from base.recording_management import RecordingManager
from base.save_data import save_audio_simple
from base.soundcard_audio_processor import SoundcardAudioProcessor, alignment_reference_from_stimulus
from base.streaming_audio_processor import StreamingAudioProcessor
from consts import error_code, model_consts

data_struct = DataDealStruct()


def _coerce_runtime_sample_rate(sample_rate):
    if isinstance(sample_rate, bool):
        return None
    try:
        coerced = int(sample_rate)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        if not isinstance(sample_rate, str) and float(sample_rate) != float(coerced):
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return coerced if coerced > 0 else None


def _require_runtime_sample_rate(sample_rate, context):
    coerced = _coerce_runtime_sample_rate(sample_rate)
    if coerced is None:
        raise RuntimeError(f"Missing resolved {context} sample rate.")
    return coerced


def _require_recording_runtime_sample_rate(recorded_dict):
    values = {}
    for key in ("sample_rate", "sr"):
        if key not in recorded_dict or recorded_dict.get(key) in (None, ""):
            continue
        coerced = _coerce_runtime_sample_rate(recorded_dict.get(key))
        if coerced is None:
            raise RuntimeError(f"Invalid resolved recording sample rate: {key}.")
        values[key] = coerced
    if not values:
        raise RuntimeError("Missing resolved recording sample rate.")
    if len(set(values.values())) != 1:
        raise RuntimeError("Conflicting resolved recording sample rate values.")
    return next(iter(values.values()))


def _require_play_record_runtime_sample_rate(stimulus_dict, recorded_dict):
    sample_rate, validation_code, validation_msg = SoundcardAudioProcessor._resolve_playrec_sample_rate(
        recorded_dict,
        stimulus_dict,
    )
    if validation_code != error_code.OK:
        raise RuntimeError(validation_msg or "Invalid resolved play-and-record sample rate.")
    return sample_rate


def record_without_play(recorded_dict, recorded_path, recorded_signal_info):
    """
    Implements the complete workflow for the record-only mode.
    It records audio based on the `sample_rate` specified in`data_struct`, then
    saves the result to a .wav file and the database (with `stimulus_id` set to
    0). All subsequent operations,such as FFT/STFT, automatic analysis, and
    button state updates, are the same as in the play-and-record mode.
    """

    record_code, recorded_signal = SoundcardAudioProcessor.sd_rec(recorded_dict)

    if record_code != error_code.OK:
        return record_code, recorded_signal

    sample_rate = _require_recording_runtime_sample_rate(recorded_dict)
    save_audio_simple(recorded_path, recorded_signal, sample_rate)

    data_struct.store_wave_data = recorded_signal

    recorded_signal_info["sample_rate"] = sample_rate
    RecordingManager().save_signal_info_to_db(recorded_signal_info, None)
    return record_code, recorded_signal


def play_last_stimulus_wave(stimulus_dict, recorded_dict, recorded_path, recorded_signal_info):
    """
    Implements the complete workflow for the play and record mode.:
    1. Obtains the sample rate and generates dictionaries for the stimulus and recorded signals.
    2. Uses the soundcard audio processor to play the stimulus signal and record the response signal.
    3. If recording is successful, plots the recorded signal on the line graph and saves the signal information.
    4. Enables the data button and the replay button.
    """
    sample_rate = data_struct.sample_rate

    sap = SoundcardAudioProcessor()
    record_code, recorded_signal = sap.sd_play_rec(recorded_dict, stimulus_dict, recorded_path)
    if record_code != error_code.OK:
        return record_code, recorded_signal

    data_struct.store_wave_data = recorded_signal
    sample_rate = _require_play_record_runtime_sample_rate(stimulus_dict, recorded_dict)
    recorded_signal_info["sample_rate"] = sample_rate
    RecordingManager().save_signal_info_to_db(recorded_signal_info, data_struct.stimulus_info)

    repeat_times = data_struct.stimulus_info.get("repeat_times")
    if repeat_times > 1:
        kwargs = {"repeat_times": repeat_times}
        data_struct.split_repeat_data = SplitRepeatSignal().split_repeat_signal(
            data_struct.store_wave_data, sample_rate, **kwargs
        )
    return record_code, data_struct.store_wave_data


def get_recorded_info(product_model, product_number, barcode, label):
    """
        Generate recorded information.

        This function generates a unique recording file name based on the current date, MAC address, product model,
    and product number.
        It also constructs the path for the recording file. Additionally, it creates a dictionary containing the
    recording file path and product information.

        Returns:
            tuple: A tuple containing the recording file path and a dictionary with recording information.
    """
    recording_time = datetime.now().strftime("%Y-%m-%d")
    mac_address = get_mac_address()
    mac_address = mac_address.replace(":", "") if mac_address else None
    product_number = "{:03}".format(int(product_number))
    recorded_name = product_model + "_" + recording_time + "_" + mac_address + "_" + product_number
    if barcode:
        recorded_name = recorded_name + "_BC" + barcode
    else:
        barcode = None
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
    sample_rate = _require_recording_runtime_sample_rate(recorded_dict)
    num_frames = recorded_dict.get("num_frames", 441000)
    device = recorded_dict.get("device")
    output_device = recorded_dict.get("output_device")
    input_channels = recorded_dict.get("input_channels")
    monitor_playback = recorded_dict.get("monitor_playback", False)
    monitor_input_channel = recorded_dict.get("monitor_input_channel")
    monitor_gain_db = float(recorded_dict.get("monitor_gain_db", 0.0))
    recording_start_delay_frames = recorded_dict.get("recording_start_delay_frames", 0)

    # Create streaming processor
    processor = StreamingAudioProcessor()

    # Start streaming recording (non-blocking) with exact sample count
    record_code, msg = processor.start_streaming_rec(
        sample_rate=sample_rate,
        target_samples=num_frames,  # Use exact sample count instead of duration
        device=device,
        output_device=output_device,
        input_channels=input_channels,
        monitor_playback=monitor_playback,
        monitor_input_channel=monitor_input_channel,
        monitor_gain_db=monitor_gain_db,
        discard_initial_samples=recording_start_delay_frames,
    )

    if record_code == error_code.OK:
        # Return processor for UI to manage (don't block!)
        return processor, sample_rate
    else:
        raise RuntimeError(f"Failed to start streaming recording: {msg}")


def stream_play_and_record(stimulus_dict, recorded_dict, recorded_path, recorded_signal_info):
    """
    Start streaming play+record (non-blocking).

    Returns StreamingAudioProcessor instance and stimulus data for UI to manage lifecycle.
    Alignment, file writing, and database operations handled by UI layer after completion.

    Args:
        stimulus_dict (dict): Stimulus signal parameters containing:
            - 'data': Stimulus signal array
            - 'amplitude': Playback amplitude
            - 'sr': Sample rate
        recorded_dict (dict): Recording parameters containing:
            - 'prepare_frames': Silent frames before stimulus
            - 'prolong_frames': Silent frames after stimulus
            - 'input_device': Input device (optional)
            - 'output_device': Output device (optional)
        recorded_path (str): Path where WAV file will be saved (managed by UI)
        recorded_signal_info (dict): Recording metadata (saved by UI after completion)

    Returns:
        tuple: (StreamingAudioProcessor instance, stimulus_data, sample_rate)
    """
    sample_rate = _require_runtime_sample_rate(stimulus_dict.get("sr"), "stimulus")
    stimulus_data = stimulus_dict.get("data")
    prepare_frames = recorded_dict.get("prepare_frames", 1000)
    prolong_frames = recorded_dict.get("prolong_frames", 10000)

    # Calculate exact target samples
    target_samples = prepare_frames + len(stimulus_data) + prolong_frames

    input_device = recorded_dict.get("input_device")
    output_device = recorded_dict.get("output_device")
    recording_start_delay_frames = recorded_dict.get("recording_start_delay_frames", 0)

    # Create streaming processor
    processor = StreamingAudioProcessor()

    # Start streaming play+record (non-blocking) with exact sample count
    record_code, msg = processor.start_streaming_playrec(
        stimulus_dict=stimulus_dict,
        sample_rate=sample_rate,
        target_samples=target_samples,  # Use exact sample count instead of duration
        input_device=input_device,
        output_device=output_device,
        prepare_frames=prepare_frames,
        prolong_frames=prolong_frames,
        discard_initial_samples=recording_start_delay_frames,
    )

    if record_code == error_code.OK:
        # Return processor and stimulus data for UI to manage (don't block!)
        # UI will perform alignment after recording completes
        return processor, alignment_reference_from_stimulus(stimulus_dict), sample_rate
    else:
        raise RuntimeError(f"Failed to start streaming play+record: {msg}")
