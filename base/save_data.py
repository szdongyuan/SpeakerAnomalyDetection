import json
import os

from datetime import datetime
from scipy.io import wavfile
import numpy as np

from consts.running_consts import DEFAULT_DIR


def save_audio_simple(save_path, audio, sr=44100):
    """
    Save audio to WAV.

    Supports:
    - mono: shape (frames,)
    - multi-channel: shape (frames, channels)

    Data is written as float32.
    """
    if not save_path:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    audio_arr = np.asarray(audio, dtype=np.float32)
    if audio_arr.ndim == 2 and audio_arr.shape[1] == 1:
        audio_arr = audio_arr.reshape(-1)
    if audio_arr.ndim not in (1, 2):
        raise ValueError(f"Unsupported audio shape: {audio_arr.shape}")

    wavfile.write(save_path, int(sr), audio_arr)


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
    current_time = datetime.now().strftime("%Y-%m-%d")
    test_result_path = DEFAULT_DIR + f"log/test_result_log/{current_time}.dat"
    if not os.path.exists(test_result_path):
        os.makedirs(os.path.dirname(test_result_path), exist_ok=True)
        with open(test_result_path, "w") as f:
            f.write(
                f"total: 0\n"
                f"ok: 0\n"
                f"ng: 0\n"
                f"ok_percent: 0%\n"
                f"datatime: {current_time}\n"
            )
