import numpy as np
from scipy.io import wavfile

from base.save_data import save_audio_simple


def test_save_audio_simple_creates_missing_parent_directory(tmp_path):
    save_path = tmp_path / "missing" / "nested" / "sample.wav"
    audio = np.asarray([0.0, 0.25, -0.25], dtype=np.float64)

    save_audio_simple(save_path, audio, sr=8000)

    assert save_path.is_file()
    sr, saved_audio = wavfile.read(save_path)
    assert sr == 8000
    assert saved_audio.dtype == np.float32


def test_save_audio_simple_writes_float64_when_requested(tmp_path):
    save_path = tmp_path / "float64.wav"
    audio = np.asarray([0.0, 0.25, -0.25], dtype=np.float32)

    save_audio_simple(save_path, audio, sr=8000, bit_depth=64)

    sr, saved_audio = wavfile.read(save_path)
    assert sr == 8000
    assert saved_audio.dtype == np.float64
