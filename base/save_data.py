import json
import os

from datetime import datetime
import numpy as np
from scipy.io import wavfile

from base.file_ops import FileOps
from consts.audio_consts import bit_depth_to_dtype
from consts.running_consts import DEFAULT_DIR
from base.wav_calibration_metadata import append_wav_calibration_metadata


def save_audio_simple(save_path, audio, sr=44100, bit_depth=32):
    # we assume audio is mono channel
    audio = np.asarray(audio).astype(bit_depth_to_dtype(bit_depth), copy=False)
    FileOps.ensure_directory_exists(save_path)
    wavfile.write(save_path, sr, audio)


def save_audio_with_calibration_metadata(save_path, audio, sr=44100, calibration_metadata=None, logger=None, bit_depth=32):
    save_audio_simple(save_path, audio, sr, bit_depth=bit_depth)
    if not calibration_metadata:
        return
    try:
        appended = append_wav_calibration_metadata(save_path, calibration_metadata, logger=logger)
    except Exception as exc:
        appended = False
        if logger is not None:
            log_method = getattr(logger, "warning", None) or getattr(logger, "error", None)
            if log_method is not None:
                log_method(f"Failed to append WAV calibration metadata; audio file was kept. {exc}")
    if not appended and logger is not None:
        log_method = getattr(logger, "warning", None) or getattr(logger, "error", None)
        if log_method is not None:
            log_method("Failed to append WAV calibration metadata; audio file was kept.")


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
