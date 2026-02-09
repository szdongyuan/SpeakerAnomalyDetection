import os
from datetime import datetime

from base.data_struct.data_deal_struct import DataDealStruct
from base.system_intervction.hardware_intervction import get_mac_address
from base.recording_management import RecordingManager
from base.save_data import save_audio_simple
from base.soundcard_audio_processor import SoundcardAudioProcessor
from base.streaming_audio_processor import StreamingAudioProcessor
from consts import error_code, model_consts

data_struct = DataDealStruct()


def record_without_play(recorded_dict, recorded_path, recorded_signal_info):
    """
    Implements the complete workflow for the record-only mode.
    It records audio based on the `sample_rate` specified in`data_struct`, then
    saves the result to a .wav file and the database (with `stimulus_id` set to
    0). All subsequent operations,such as FFT/STFT, automatic analysis, and
    button state updates, are the same as in the play-and-record mode.
    """

    sample_rate = data_struct.sample_rate
    record_code, recorded_signal = SoundcardAudioProcessor.sd_rec(recorded_dict)

    if record_code == error_code.OK:
        recorded_multi = None
        try:
            recorded_multi = recorded_dict.get("_recorded_multi")
        except Exception:
            recorded_multi = None

        if recorded_multi is not None:
            save_audio_simple(recorded_path, recorded_multi, sample_rate)
            try:
                data_struct.store_wave_data_multi = recorded_multi
                data_struct.store_wave_data = recorded_multi.mean(axis=1)
            except Exception:
                data_struct.store_wave_data_multi = None
                data_struct.store_wave_data = recorded_signal
        else:
            save_audio_simple(recorded_path, recorded_signal, sample_rate)
            try:
                data_struct.store_wave_data_multi = recorded_signal.reshape(-1, 1)
            except Exception:
                data_struct.store_wave_data_multi = None
            data_struct.store_wave_data = recorded_signal

        recorded_signal_info["sample_rate"] = sample_rate
        RecordingManager().save_signal_info_to_db(recorded_signal_info, None)


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
    sample_rate = recorded_dict.get("sample_rate", data_struct.sample_rate)
    num_frames = recorded_dict.get("num_frames", 441000)
    device = recorded_dict.get("device")
    input_channels = recorded_dict.get("input_channels")
    output_device = recorded_dict.get("output_device")
    output_channels = recorded_dict.get("output_channels")
    monitor_playback = bool(output_device and output_channels)

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
    )

    if record_code == error_code.OK:
        # Return processor for UI to manage (don't block!)
        return processor, sample_rate
    else:
        raise RuntimeError(f"Failed to start streaming recording: {msg}")


