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


def save_recorded_data_to_json(
    product_model=None,
    scanner_barcode=None,
    scanner_barcode_check=None,
    sequence_mode=None,
):
    """
    Persist lightweight sequence-page UI state with merge semantics.

    Each parameter is optional: pass None to leave the corresponding field
    untouched on disk. This avoids unrelated callers (e.g. mode switch,
    type/edit lose-focus) accidentally overwriting fields they don't own
    (most importantly ``scanner_barcode_check``).
    """
    file_path = DEFAULT_DIR + "ui/ui_config/recorded_number.json"

    data = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            data = {}

    if product_model is not None:
        data["product_model"] = product_model
    if scanner_barcode is not None:
        data["scanner_barcode"] = scanner_barcode
    if scanner_barcode_check is not None:
        data["scanner_barcode_check"] = bool(scanner_barcode_check)
    if sequence_mode is not None:
        normalized_mode = str(sequence_mode).strip().lower()
        if normalized_mode in ("test", "mark"):
            data["sequence_mode"] = normalized_mode

    data["datetime"] = datetime.now().strftime("%Y-%m-%d")

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
                f"not_labels: 0\n"
                f"ok_percent: 0%\n"
                f"datatime: {current_time}\n"
            )
