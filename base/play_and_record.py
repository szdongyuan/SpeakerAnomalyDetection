import os
from datetime import datetime

from base.data_struct.data_deal_struct import DataDealStruct
from base.system_intervction.hardware_intervction import get_mac_address
from base.streaming_audio_processor import StreamingAudioProcessor
from consts import error_code, model_consts

data_struct = DataDealStruct()


def get_recorded_info(product_model, product_number, barcode, label, name_suffix=""):
    """
        Generate recorded information.

        This function generates a unique recording file name based on the current date, MAC address, product model,
    and product number.
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
    product_token = str(product_number or "").strip()
    if not product_token:
        product_token = now.strftime("%H%M%S%f")
    sanitized_token = "".join(ch for ch in product_token if ch.isalnum() or ch in ("-", "_"))
    if not sanitized_token:
        sanitized_token = now.strftime("%H%M%S%f")

    recorded_name = product_model + "_" + recording_time_for_name + "_" + mac_address + "_" + sanitized_token
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
    )

    if record_code == error_code.OK:
        # Return processor for UI to manage (don't block!)
        return processor, sample_rate
    else:
        raise RuntimeError(f"Failed to start streaming recording: {msg}")
