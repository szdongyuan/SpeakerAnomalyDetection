import struct
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
from scipy.io import wavfile

from base import save_data
from base import wav_calibration_metadata as wav_metadata
from base.save_data import save_audio_with_calibration_metadata
from base.wav_calibration_metadata import (
    WavCalibrationResolution,
    append_wav_calibration_metadata,
    normalize_wav_calibration_metadata,
    read_wav_calibration_metadata,
    resolve_wav_channel_v2pa_factor,
)


def test_save_audio_with_calibration_metadata_writes_readable_wav(tmp_path):
    path = tmp_path / "record.wav"
    metadata = {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": 2.5, "standard_spl": 94.0, "calibrated": True}
        ]
    }

    save_audio_with_calibration_metadata(path, np.array([0.1, 0.2], dtype=np.float32), 48000, metadata)

    assert read_wav_calibration_metadata(path) == metadata


def test_save_audio_with_calibration_metadata_keeps_audio_when_metadata_append_fails(tmp_path, monkeypatch):
    path = tmp_path / "record.wav"
    messages = []
    logger = SimpleNamespace(
        warning=lambda message: messages.append(message),
        error=lambda message: messages.append(message),
    )
    monkeypatch.setattr(save_data, "append_wav_calibration_metadata", lambda *args, **kwargs: False)

    save_audio_with_calibration_metadata(
        path,
        np.array([0.1, 0.2], dtype=np.float32),
        48000,
        {"recorded_channels": [{"wav_channel_index": 0, "v2pa_factor": 2.5, "standard_spl": 94.0, "calibrated": True}]},
        logger=logger,
    )

    rate, audio = wavfile.read(path)
    assert rate == 48000
    assert audio.size == 2
    assert messages


def test_save_audio_with_calibration_metadata_keeps_audio_when_metadata_append_raises(tmp_path, monkeypatch):
    path = tmp_path / "record.wav"
    messages = []
    logger = SimpleNamespace(
        warning=lambda message: messages.append(message),
        error=lambda message: messages.append(message),
    )

    def fail_append(*args, **kwargs):
        raise OSError("metadata write failed")

    monkeypatch.setattr(save_data, "append_wav_calibration_metadata", fail_append)

    save_audio_with_calibration_metadata(
        path,
        np.array([0.1, 0.2], dtype=np.float32),
        48000,
        {"recorded_channels": [{"wav_channel_index": 0, "v2pa_factor": 2.5, "standard_spl": 94.0, "calibrated": True}]},
        logger=logger,
    )

    rate, audio = wavfile.read(path)
    assert rate == 48000
    assert audio.size == 2
    assert messages


def test_normalize_metadata_keeps_only_wav_channel_calibration_fields():
    payload = {
        "microphone": {"hardware_id": "must-not-survive"},
        "recorded_channels": [
            {
                "wav_channel_index": 0,
                "input_channel_index": 7,
                "input_channel_label": "In8",
                "calibration_type": "mic_v2pa",
                "v2pa_factor": 2.5,
                "standard_spl": 94.0,
                "calibrated": True,
            }
        ],
    }

    normalized = normalize_wav_calibration_metadata(payload)

    assert normalized == {
        "recorded_channels": [
            {
                "wav_channel_index": 0,
                "v2pa_factor": 2.5,
                "standard_spl": 94.0,
                "calibrated": True,
            }
        ]
    }


def test_manual_calibration_normalizes_serializes_and_resolves_as_authoritative(tmp_path):
    path = tmp_path / "manual_calibration.wav"
    wavfile.write(path, 48000, np.array([0.1, 0.2], dtype=np.float32))
    metadata = {
        "recorded_channels": [
            {
                "wav_channel_index": 0,
                "v2pa_factor": 1.234567,
                "standard_spl": None,
                "calibrated": True,
            }
        ]
    }

    normalized = normalize_wav_calibration_metadata(metadata)
    assert normalized == metadata
    assert append_wav_calibration_metadata(path, normalized) is True

    serialized = read_wav_calibration_metadata(path)
    assert serialized == metadata
    assert resolve_wav_channel_v2pa_factor(serialized, 0) == WavCalibrationResolution(
        factor=1.234567,
        has_valid_metadata=True,
        used_file_metadata=True,
    )


def test_resolve_wav_channel_factor_uses_one_for_missing_or_uncalibrated():
    metadata = {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": 2.5, "standard_spl": 94.0, "calibrated": True},
            {"wav_channel_index": 1, "v2pa_factor": None, "standard_spl": None, "calibrated": False},
        ]
    }

    assert resolve_wav_channel_v2pa_factor(metadata, 0).factor == 2.5
    assert resolve_wav_channel_v2pa_factor(metadata, 1).factor == 1.0
    assert resolve_wav_channel_v2pa_factor(metadata, 9).factor == 1.0


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {"recorded_channels": []},
        {"recorded_channels": "bad"},
        {"recorded_channels": [{"wav_channel_index": "bad", "v2pa_factor": 2.5, "calibrated": True}]},
    ],
)
def test_normalize_metadata_returns_none_for_missing_or_malformed_payload(payload):
    assert normalize_wav_calibration_metadata(payload) is None


def test_normalize_metadata_rejects_mixed_valid_and_malformed_channels():
    payload = {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": 2.5, "standard_spl": 94.0, "calibrated": True},
            {"wav_channel_index": "bad", "v2pa_factor": 3.5, "standard_spl": 94.0, "calibrated": True},
        ]
    }

    assert normalize_wav_calibration_metadata(payload) is None


def test_normalize_metadata_rejects_duplicate_wav_channel_indices():
    payload = {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": 2.5, "standard_spl": 94.0, "calibrated": True},
            {"wav_channel_index": 0, "v2pa_factor": 3.5, "standard_spl": 94.0, "calibrated": True},
        ]
    }

    assert normalize_wav_calibration_metadata(payload) is None
    assert resolve_wav_channel_v2pa_factor(payload, 0) == WavCalibrationResolution(1.0, False, False)


def test_resolve_wav_channel_factor_returns_structured_fallback_for_mixed_malformed_metadata():
    resolution = resolve_wav_channel_v2pa_factor(
        {
            "recorded_channels": [
                {"wav_channel_index": 0, "v2pa_factor": 2.5, "standard_spl": 94.0, "calibrated": True},
                {"wav_channel_index": "bad", "v2pa_factor": 3.5, "standard_spl": 94.0, "calibrated": True},
            ]
        },
        0,
    )

    assert resolution.factor == 1.0
    assert resolution.has_valid_metadata is False
    assert resolution.used_file_metadata is False


def test_resolve_wav_channel_factor_returns_structured_fallback_for_malformed_metadata():
    resolution = resolve_wav_channel_v2pa_factor(
        {"recorded_channels": [{"wav_channel_index": 0, "v2pa_factor": "bad", "calibrated": True}]},
        0,
    )

    assert resolution.factor == 1.0
    assert resolution.has_valid_metadata is False
    assert resolution.used_file_metadata is False


class _OverflowingFloat:
    def __float__(self):
        raise OverflowError("too large to convert to float")


def test_resolve_wav_channel_factor_returns_structured_fallback_for_overflowing_factor():
    resolution = resolve_wav_channel_v2pa_factor(
        {"recorded_channels": [{"wav_channel_index": 0, "v2pa_factor": _OverflowingFloat(), "calibrated": True}]},
        0,
    )

    assert resolution.factor == 1.0
    assert resolution.has_valid_metadata is False
    assert resolution.used_file_metadata is False


@pytest.mark.parametrize(
    "wav_channel_index",
    [True, False, 0.9, 1.8, "0.9", -1, float("nan"), float("inf")],
)
def test_normalize_metadata_rejects_non_lossless_wav_channel_indices(wav_channel_index):
    metadata = {
        "recorded_channels": [
            {
                "wav_channel_index": wav_channel_index,
                "v2pa_factor": 2.5,
                "standard_spl": 94.0,
                "calibrated": True,
            }
        ]
    }

    assert normalize_wav_calibration_metadata(metadata) is None
    resolution = resolve_wav_channel_v2pa_factor(metadata, 0)
    assert resolution.factor == 1.0
    assert resolution.has_valid_metadata is False
    assert resolution.used_file_metadata is False


@pytest.mark.parametrize("factor", [0, -1, float("nan"), float("inf"), "bad"])
def test_normalize_metadata_rejects_invalid_calibrated_factor(factor):
    with pytest.raises(ValueError):
        normalize_wav_calibration_metadata(
            {"recorded_channels": [{"wav_channel_index": 0, "v2pa_factor": factor, "calibrated": True}]}
        )


@pytest.mark.parametrize("factor", [None, _OverflowingFloat()])
def test_normalize_metadata_rejects_uncoercible_calibrated_factor_with_value_error(factor):
    with pytest.raises(ValueError, match="v2pa_factor must be a finite positive number"):
        normalize_wav_calibration_metadata(
            {"recorded_channels": [{"wav_channel_index": 0, "v2pa_factor": factor, "calibrated": True}]}
        )


def test_wav_metadata_roundtrip_preserves_audio_and_rate(tmp_path):
    path = tmp_path / "recording.wav"
    sample_rate = 48000
    audio = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    wavfile.write(path, sample_rate, audio)
    metadata = {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": 2.5, "standard_spl": 94.0, "calibrated": True},
            {"wav_channel_index": 1, "v2pa_factor": None, "standard_spl": None, "calibrated": False},
        ]
    }

    append_wav_calibration_metadata(path, metadata)

    assert read_wav_calibration_metadata(path) == metadata
    loaded_rate, loaded_audio = wavfile.read(path)
    assert loaded_rate == sample_rate
    np.testing.assert_allclose(loaded_audio, audio)


def test_append_writes_unprefixed_metadata_marker(tmp_path):
    path = tmp_path / "recording.wav"
    wavfile.write(path, 48000, np.array([0.1, 0.2], dtype=np.float32))
    metadata = {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": 2.5, "standard_spl": 94.0, "calibrated": True}
        ]
    }

    append_wav_calibration_metadata(path, metadata)

    wav_bytes = path.read_bytes()
    assert b"mic_calibration=" in wav_bytes
    assert b"dy_" + b"mic_calibration=" not in wav_bytes


def test_read_unprefixed_embedded_metadata_marker(tmp_path):
    path = tmp_path / "embedded.wav"
    wavfile.write(path, 48000, np.array([0.1, 0.2], dtype=np.float32))
    metadata = {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": 2.5, "standard_spl": 94.0, "calibrated": True}
        ]
    }
    _append_test_info_comment(
        path,
        'mic_calibration={"recorded_channels":[{"wav_channel_index":0,"v2pa_factor":2.5,'
        '"standard_spl":94.0,"calibrated":true}]}',
    )

    assert read_wav_calibration_metadata(path) == metadata


def test_read_missing_or_malformed_metadata_returns_none(tmp_path):
    path = tmp_path / "plain.wav"
    wavfile.write(path, 48000, np.array([0.1, 0.2], dtype=np.float32))

    assert read_wav_calibration_metadata(path) is None


def test_read_malformed_embedded_metadata_returns_none(tmp_path):
    path = tmp_path / "malformed.wav"
    wavfile.write(path, 48000, np.array([0.1, 0.2], dtype=np.float32))
    _append_test_info_comment(path, "mic_calibration={not-json")

    assert read_wav_calibration_metadata(path) is None


def test_append_rejects_duplicate_wav_channel_metadata(tmp_path):
    path = tmp_path / "duplicate_append.wav"
    messages = []
    logger = _ListLogger(messages)
    wavfile.write(path, 48000, np.array([0.1, 0.2], dtype=np.float32))
    metadata = {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": 2.5, "standard_spl": 94.0, "calibrated": True},
            {"wav_channel_index": 0, "v2pa_factor": 3.5, "standard_spl": 94.0, "calibrated": True},
        ]
    }

    assert append_wav_calibration_metadata(path, metadata, logger=logger) is False
    assert read_wav_calibration_metadata(path) is None
    assert messages


def test_append_logs_malformed_wav_channel_metadata(tmp_path):
    path = tmp_path / "malformed_append.wav"
    messages = []
    logger = _ListLogger(messages)
    wavfile.write(path, 48000, np.array([0.1, 0.2], dtype=np.float32))
    metadata = {
        "recorded_channels": [
            {"wav_channel_index": "bad", "v2pa_factor": 2.5, "standard_spl": 94.0, "calibrated": True},
        ]
    }

    assert append_wav_calibration_metadata(path, metadata, logger=logger) is False
    assert read_wav_calibration_metadata(path) is None
    assert messages


def test_append_logs_unsupported_file(tmp_path):
    path = tmp_path / "not_a_wave.txt"
    messages = []
    logger = _ListLogger(messages)
    path.write_text("not a riff wave", encoding="utf-8")

    assert append_wav_calibration_metadata(
        path,
        {"recorded_channels": [{"wav_channel_index": 0, "v2pa_factor": 2.5, "calibrated": True}]},
        logger=logger,
    ) is False
    assert messages


def test_read_rejects_duplicate_wav_channel_metadata(tmp_path):
    path = tmp_path / "duplicate_read.wav"
    wavfile.write(path, 48000, np.array([0.1, 0.2], dtype=np.float32))
    _append_test_info_comment(
        path,
        'mic_calibration={"recorded_channels":['
        '{"wav_channel_index":0,"v2pa_factor":2.5,"standard_spl":94.0,"calibrated":true},'
        '{"wav_channel_index":0,"v2pa_factor":3.5,"standard_spl":94.0,"calibrated":true}'
        "]}",
    )

    assert read_wav_calibration_metadata(path) is None


def test_read_ignores_metadata_appended_beyond_declared_riff_size(tmp_path):
    path = tmp_path / "stale_tail.wav"
    wavfile.write(path, 48000, np.array([0.1, 0.2], dtype=np.float32))
    _append_test_info_comment(
        path,
        'mic_calibration={"recorded_channels":[{"wav_channel_index":0,"v2pa_factor":2.5,'
        '"standard_spl":94.0,"calibrated":true}]}',
        update_riff_size=False,
    )

    assert read_wav_calibration_metadata(path) is None


def test_append_overwrites_stale_bytes_beyond_declared_riff_size(tmp_path):
    path = tmp_path / "stale_tail_append.wav"
    wavfile.write(path, 48000, np.array([0.1, 0.2], dtype=np.float32))
    _append_test_declared_oversize_chunk(path)
    metadata = {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": 3.5, "standard_spl": 94.0, "calibrated": True}
        ]
    }

    assert append_wav_calibration_metadata(path, metadata) is True

    assert read_wav_calibration_metadata(path) == metadata


def test_append_rejects_riff_size_overflow_without_modifying_file(monkeypatch):
    messages = []
    logger = _ListLogger(messages)
    declared_riff_size = 0xFFFFFFF0
    original_bytes = b"RIFF" + struct.pack("<I", declared_riff_size) + b"WAVE"
    fake_file = _LargeRiffFile(original_bytes, file_size=declared_riff_size + 8)
    metadata = {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": 3.5, "standard_spl": 94.0, "calibrated": True}
        ]
    }

    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: fake_file)
    monkeypatch.setattr(wav_metadata, "_scan_declared_riff_chunks", lambda *args, **kwargs: {})

    assert append_wav_calibration_metadata("overflow.wav", metadata, logger=logger) is False
    assert fake_file.bytes == original_bytes
    assert fake_file.truncate_calls == []
    assert fake_file.write_calls == []
    assert any("RIFF size exceeds 32-bit limit" in message for message in messages)


def test_read_logs_structurally_invalid_app_metadata(tmp_path):
    path = tmp_path / "invalid_app_metadata.wav"
    messages = []
    logger = _ListLogger(messages)
    wavfile.write(path, 48000, np.array([0.1, 0.2], dtype=np.float32))
    _append_test_info_comment(path, 'mic_calibration={"recorded_channels":[]}')

    assert read_wav_calibration_metadata(path, logger=logger) is None
    assert messages


def test_read_logs_top_level_chunk_declared_beyond_riff_parse_end(tmp_path):
    path = tmp_path / "truncated_chunk.wav"
    messages = []
    logger = _ListLogger(messages)
    wavfile.write(path, 48000, np.array([0.1, 0.2], dtype=np.float32))
    _append_test_declared_oversize_chunk(path, update_riff_size=True)

    assert read_wav_calibration_metadata(path, logger=logger) is None
    assert messages


def test_append_rejects_existing_chunk_declared_beyond_riff_parse_end(tmp_path):
    path = tmp_path / "truncated_chunk_append.wav"
    messages = []
    logger = _ListLogger(messages)
    wavfile.write(path, 48000, np.array([0.1, 0.2], dtype=np.float32))
    _append_test_declared_oversize_chunk(path, update_riff_size=True)
    metadata = {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": 3.5, "standard_spl": 94.0, "calibrated": True}
        ]
    }

    assert append_wav_calibration_metadata(path, metadata, logger=logger) is False
    assert read_wav_calibration_metadata(path) is None
    assert messages


def test_read_rejects_inflated_riff_size_with_readable_metadata(tmp_path):
    path = tmp_path / "inflated_riff_size.wav"
    messages = []
    logger = _ListLogger(messages)
    wavfile.write(path, 48000, np.array([0.1, 0.2], dtype=np.float32))
    _append_test_info_comment(
        path,
        'mic_calibration={"recorded_channels":[{"wav_channel_index":0,"v2pa_factor":2.5,'
        '"standard_spl":94.0,"calibrated":true}]}',
    )
    _inflate_test_riff_size(path)

    assert read_wav_calibration_metadata(path, logger=logger) is None
    assert any("declared RIFF size exceeds file size" in message for message in messages)


def test_read_logs_odd_sized_list_missing_top_level_padding(tmp_path):
    path = tmp_path / "unpadded_list.wav"
    messages = []
    logger = _ListLogger(messages)
    wavfile.write(path, 48000, np.array([0.1, 0.2], dtype=np.float32))
    _append_test_unpadded_odd_info_list(path)

    assert read_wav_calibration_metadata(path, logger=logger) is None
    assert messages


def test_read_logs_odd_sized_list_nonzero_top_level_padding(tmp_path):
    path = tmp_path / "nonzero_padded_list.wav"
    messages = []
    logger = _ListLogger(messages)
    wavfile.write(path, 48000, np.array([0.1, 0.2], dtype=np.float32))
    _append_test_nonzero_padded_odd_list(path)

    assert read_wav_calibration_metadata(path, logger=logger) is None
    assert any("nonzero padding byte after odd-sized chunk" in message for message in messages)


def test_read_logs_list_subchunk_declared_beyond_list_payload(tmp_path):
    path = tmp_path / "truncated_list_subchunk.wav"
    messages = []
    logger = _ListLogger(messages)
    wavfile.write(path, 48000, np.array([0.1, 0.2], dtype=np.float32))
    _append_test_truncated_info_comment(path)

    assert read_wav_calibration_metadata(path, logger=logger) is None
    assert messages


def test_read_logs_odd_sized_icmt_missing_list_padding(tmp_path):
    path = tmp_path / "unpadded_icmt.wav"
    messages = []
    logger = _ListLogger(messages)
    wavfile.write(path, 48000, np.array([0.1, 0.2], dtype=np.float32))
    _append_test_unpadded_odd_icmt(path)

    assert read_wav_calibration_metadata(path, logger=logger) is None
    assert messages


def test_read_logs_odd_sized_icmt_nonzero_list_padding(tmp_path):
    path = tmp_path / "nonzero_padded_icmt.wav"
    messages = []
    logger = _ListLogger(messages)
    wavfile.write(path, 48000, np.array([0.1, 0.2], dtype=np.float32))
    _append_test_nonzero_padded_odd_icmt(path)

    assert read_wav_calibration_metadata(path, logger=logger) is None
    assert any("nonzero padding byte after odd-sized subchunk" in message for message in messages)


def test_read_logs_list_level_trailing_junk_after_valid_metadata(tmp_path):
    path = tmp_path / "list_trailing_junk.wav"
    messages = []
    logger = _ListLogger(messages)
    wavfile.write(path, 48000, np.array([0.1, 0.2], dtype=np.float32))
    _append_test_info_comment_with_list_trailing_junk(
        path,
        'mic_calibration={"recorded_channels":[{"wav_channel_index":0,"v2pa_factor":2.5,'
        '"standard_spl":94.0,"calibrated":true}]}',
    )

    assert read_wav_calibration_metadata(path, logger=logger) is None
    assert messages


def test_read_logs_top_level_trailing_junk_after_valid_metadata(tmp_path):
    path = tmp_path / "top_level_trailing_junk.wav"
    messages = []
    logger = _ListLogger(messages)
    wavfile.write(path, 48000, np.array([0.1, 0.2], dtype=np.float32))
    _append_test_info_comment_with_top_level_trailing_junk(
        path,
        'mic_calibration={"recorded_channels":[{"wav_channel_index":0,"v2pa_factor":2.5,'
        '"standard_spl":94.0,"calibrated":true}]}',
    )

    assert read_wav_calibration_metadata(path, logger=logger) is None
    assert messages


class _ListLogger:
    def __init__(self, messages):
        self.messages = messages

    def warning(self, message):
        self.messages.append(message)


class _LargeRiffFile:
    def __init__(self, data, *, file_size):
        self._data = bytearray(data)
        self._file_size = file_size
        self._position = 0
        self.truncate_calls = []
        self.write_calls = []

    @property
    def bytes(self):
        return bytes(self._data)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def seek(self, offset, whence=0):
        if whence == 0:
            self._position = offset
        elif whence == 1:
            self._position += offset
        elif whence == 2:
            self._position = self._file_size + offset
        else:
            raise ValueError("unsupported whence")
        return self._position

    def tell(self):
        return self._position

    def read(self, size=-1):
        if size is None or size < 0:
            size = len(self._data) - self._position
        end = min(self._position + size, len(self._data))
        payload = bytes(self._data[self._position : end])
        self._position += len(payload)
        return payload

    def truncate(self, size=None):
        self.truncate_calls.append(self._position if size is None else size)
        return self._position if size is None else size

    def write(self, data):
        self.write_calls.append((self._position, bytes(data)))
        return len(data)


def _append_test_info_comment(path, text, *, update_riff_size=True):
    payload = text.encode("utf-8") + b"\x00"
    icmt_chunk = b"ICMT" + struct.pack("<I", len(payload)) + payload
    if len(payload) % 2:
        icmt_chunk += b"\x00"
    list_payload = b"INFO" + icmt_chunk
    list_chunk = b"LIST" + struct.pack("<I", len(list_payload)) + list_payload
    if len(list_payload) % 2:
        list_chunk += b"\x00"

    with open(path, "r+b") as wav_file:
        wav_file.seek(0, 2)
        wav_file.write(list_chunk)
        if update_riff_size:
            file_size = wav_file.tell()
            wav_file.seek(4)
            wav_file.write(struct.pack("<I", file_size - 8))


def _inflate_test_riff_size(path, extra_bytes=64):
    with open(path, "r+b") as wav_file:
        wav_file.seek(4)
        riff_size = struct.unpack("<I", wav_file.read(4))[0]
        wav_file.seek(4)
        wav_file.write(struct.pack("<I", riff_size + extra_bytes))


def _append_test_info_comment_with_list_trailing_junk(path, text):
    payload = text.encode("utf-8") + b"\x00"
    icmt_chunk = b"ICMT" + struct.pack("<I", len(payload)) + payload
    if len(payload) % 2:
        icmt_chunk += b"\x00"
    list_payload = b"INFO" + icmt_chunk + b"junk"
    list_chunk = b"LIST" + struct.pack("<I", len(list_payload)) + list_payload
    if len(list_payload) % 2:
        list_chunk += b"\x00"

    with open(path, "r+b") as wav_file:
        wav_file.seek(0, 2)
        wav_file.write(list_chunk)
        file_size = wav_file.tell()
        wav_file.seek(4)
        wav_file.write(struct.pack("<I", file_size - 8))


def _append_test_info_comment_with_top_level_trailing_junk(path, text):
    _append_test_info_comment(path, text)
    with open(path, "r+b") as wav_file:
        wav_file.seek(0, 2)
        wav_file.write(b"junk")
        file_size = wav_file.tell()
        wav_file.seek(4)
        wav_file.write(struct.pack("<I", file_size - 8))


def _append_test_declared_oversize_chunk(path, *, update_riff_size=False):
    payload = b"stale"
    chunk = b"JUNK" + struct.pack("<I", len(payload) + 64) + payload
    with open(path, "r+b") as wav_file:
        wav_file.seek(0, 2)
        wav_file.write(chunk)
        if update_riff_size:
            file_size = wav_file.tell()
            wav_file.seek(4)
            wav_file.write(struct.pack("<I", file_size - 8))


def _append_test_truncated_info_comment(path):
    payload = b"mic_calibration={not-json"
    icmt_chunk = b"ICMT" + struct.pack("<I", len(payload) + 16) + payload
    list_payload = b"INFO" + icmt_chunk
    list_chunk = b"LIST" + struct.pack("<I", len(list_payload)) + list_payload
    if len(list_payload) % 2:
        list_chunk += b"\x00"

    with open(path, "r+b") as wav_file:
        wav_file.seek(0, 2)
        wav_file.write(list_chunk)
        file_size = wav_file.tell()
        wav_file.seek(4)
        wav_file.write(struct.pack("<I", file_size - 8))


def _append_test_unpadded_odd_info_list(path):
    payload = (
        b'mic_calibration={"recorded_channels":[{"wav_channel_index":0,"v2pa_factor":2.5,'
        b'"standard_spl":94.0,"calibrated":true}]}'
    )
    if len(payload) % 2:
        payload += b"\x00"
    icmt_chunk = b"ICMT" + struct.pack("<I", len(payload)) + payload
    list_payload = b"INFO" + icmt_chunk + b"x"
    assert len(list_payload) % 2 == 1
    list_chunk = b"LIST" + struct.pack("<I", len(list_payload)) + list_payload

    with open(path, "r+b") as wav_file:
        wav_file.seek(0, 2)
        wav_file.write(list_chunk)
        file_size = wav_file.tell()
        wav_file.seek(4)
        wav_file.write(struct.pack("<I", file_size - 8))


def _append_test_nonzero_padded_odd_list(path):
    list_payload = b"adtlx"
    assert len(list_payload) % 2 == 1
    list_chunk = b"LIST" + struct.pack("<I", len(list_payload)) + list_payload + b"\x01"

    with open(path, "r+b") as wav_file:
        wav_file.seek(0, 2)
        wav_file.write(list_chunk)
        file_size = wav_file.tell()
        wav_file.seek(4)
        wav_file.write(struct.pack("<I", file_size - 8))


def _append_test_unpadded_odd_icmt(path):
    payload = (
        b'mic_calibration={"recorded_channels":[{"wav_channel_index":0,"v2pa_factor":2.5,'
        b'"standard_spl":94.0,"calibrated":true}]}'
    )
    if len(payload) % 2 == 0:
        payload += b"\x00"
    assert len(payload) % 2 == 1
    icmt_chunk = b"ICMT" + struct.pack("<I", len(payload)) + payload
    list_payload = b"INFO" + icmt_chunk
    list_chunk = b"LIST" + struct.pack("<I", len(list_payload)) + list_payload
    if len(list_payload) % 2:
        list_chunk += b"\x00"

    with open(path, "r+b") as wav_file:
        wav_file.seek(0, 2)
        wav_file.write(list_chunk)
        file_size = wav_file.tell()
        wav_file.seek(4)
        wav_file.write(struct.pack("<I", file_size - 8))


def _append_test_nonzero_padded_odd_icmt(path):
    payload = (
        b'mic_calibration={"recorded_channels":[{"wav_channel_index":0,"v2pa_factor":2.5,'
        b'"standard_spl":94.0,"calibrated":true}]}'
    )
    if len(payload) % 2 == 0:
        payload += b"\x00"
    assert len(payload) % 2 == 1
    icmt_chunk = b"ICMT" + struct.pack("<I", len(payload)) + payload + b"\x01"
    list_payload = b"INFO" + icmt_chunk
    list_chunk = b"LIST" + struct.pack("<I", len(list_payload)) + list_payload
    if len(list_payload) % 2:
        list_chunk += b"\x00"

    with open(path, "r+b") as wav_file:
        wav_file.seek(0, 2)
        wav_file.write(list_chunk)
        file_size = wav_file.tell()
        wav_file.seek(4)
        wav_file.write(struct.pack("<I", file_size - 8))
