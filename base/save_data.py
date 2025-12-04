import json
import os

from datetime import datetime
from scipy.io import wavfile

from consts.running_consts import DEFAULT_DIR


def save_audio_simple(save_path, audio, sr=44100):
    """
    Save audio to WAV file with multi-channel support.

    Args:
        save_path (str): Output WAV file path
        audio (np.ndarray): Audio data
            - Shape (samples,) for mono
            - Shape (samples, channels) for multi-channel
        sr (int): Sample rate (default: 44100)

    Note:
        scipy.io.wavfile.write() natively supports both mono and multi-channel.
    """
    audio = audio.astype("float32")
    wavfile.write(save_path, sr, audio)


def save_recorded_data_to_json(product_model, current_recorded_count, scanner_barcode, scanner_barcode_check):
    """
    Save the recorded number to a text file.

    This function writes the current recorded number and the current date to a specified text file.
    If the file exists and the date matches, it updates the recorded number.
    If the file does not exist or the date does not match, it creates a new file and writes the initial recorded number.
    """
    file_path = DEFAULT_DIR + "ui/ui_config/recorded_number.json"
    current_time = datetime.now().strftime("%Y-%m-%d")
    data = {
        "product_model": product_model,
        "current_recorded_count": int(current_recorded_count),
        "scanner_barcode": scanner_barcode,
        "scanner_barcode_check": scanner_barcode_check,
        "datetime": current_time,
    }
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def ensure_test_result_file(analysis_config):
    default_ai_model = analysis_config.get("default_ai")
    if default_ai_model:
        analyse_model_name = analysis_config.get(default_ai_model, {}).get("analyse_model_name", None)
    else:
        analyse_model_name = "null"
    current_time = datetime.now().strftime("%Y-%m-%d")
    test_result_path = DEFAULT_DIR + f"log/test_result_log/{current_time}.dat"
    if not os.path.exists(test_result_path):
        os.makedirs(os.path.dirname(test_result_path), exist_ok=True)
        with open(test_result_path, "w") as f:
            f.write(
                f"total: 0\n"
                f"ok: 0\n"
                f"ng: 0\n"
                f"ok_percent: 0\n"
                f"current_model: {analyse_model_name}\n"
                f"datatime: {current_time}\n"
            )
