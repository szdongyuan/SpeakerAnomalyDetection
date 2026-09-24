"""Real-file PCM24 save contracts, independent of the codec implementation."""
import struct

import numpy as np
import pytest
import soundfile as sf

from base.save_data import save_audio_simple


@pytest.mark.parametrize("frames", [1, 3, 5])
@pytest.mark.parametrize("channels", [None, 1, 2, 3])
def test_save_pcm24_header_and_clipping(tmp_path, frames, channels):
    source = np.resize(np.array([.5, 2, -3, 1, -.5], dtype=np.float32),
                       frames if channels is None else (frames, channels))
    original = source.copy()
    expected = np.resize(np.array([.5, 8388607 / 8388608, -1,
                                   8388607 / 8388608, -.5], dtype=np.float32), source.shape)
    path = tmp_path / "saved.wav"
    save_audio_simple(str(path), source, 32000)
    info = sf.info(path)
    assert (info.subtype, info.samplerate, info.frames, info.channels) == ("PCM_24", 32000, frames, channels or 1)
    actual, _ = sf.read(path, dtype="float32", always_2d=channels is not None)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, original)
    raw = path.read_bytes()
    assert struct.unpack_from("<I", raw, 4)[0] + 8 == len(raw)
    fmt = raw.index(b"fmt ") + 8
    assert struct.unpack_from("<HHIIHH", raw, fmt) == (1, channels or 1, 32000, 32000 * (channels or 1) * 3, (channels or 1) * 3, 24)
    offset = raw.index(b"data") + 8
    size = struct.unpack_from("<I", raw, offset - 4)[0]
    assert size == frames * (channels or 1) * 3
    assert len(raw) == offset + size + size % 2


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_save_rejects_nonfinite(tmp_path, value):
    with pytest.raises(ValueError, match="finite"):
        save_audio_simple(str(tmp_path / "bad.wav"), [value])


def test_save_empty_path_and_invalid_shape(tmp_path):
    assert save_audio_simple("", object()) is None
    with pytest.raises(ValueError, match="shape"):
        save_audio_simple(str(tmp_path / "bad.wav"), np.zeros((2, 2, 2)))


def test_save_uses_bounded_writer_blocks(tmp_path, monkeypatch):
    from base import save_data
    blocks = []
    class Writer:
        def __init__(self, path, sample_rate, channels):
            assert sample_rate == 12345 and channels == 2
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def write_chunk(self, block):
            blocks.append(len(block))
    monkeypatch.setattr(save_data, "StreamingWavWriter", Writer, raising=False)
    save_audio_simple(str(tmp_path / "bounded.wav"), np.zeros((150000, 2), np.float32), 12345)
    assert sum(blocks) == 150000 and max(blocks) <= 65536
