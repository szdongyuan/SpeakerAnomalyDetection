"""Transactional metadata finalization of exclusively owned, unpublished WAVs."""
import struct

import numpy as np
import pytest
import soundfile as sf

from base import wav_calibration_metadata as module
from unit_test.base.test_wav_calibration_metadata import (
    _append_raw_chunk, _channel_with_physical, _chunk, _metadata, _metadata_list_chunk,
)
from unit_test.base.ve3668n_fakes import wav_metadata


@pytest.fixture
def recording(tmp_path):
    path = tmp_path / "new-recording.wav"
    audio = np.arange(20000, dtype=np.float32).reshape(-1, 2) / 32768
    sf.write(path, audio, 44100, subtype="PCM_24")
    metadata = _metadata(_channel_with_physical(0, 4), _channel_with_physical(1, 1))
    return path, audio, metadata


def append(path, metadata):
    return module.append_owned_recording_calibration_metadata_result(path, metadata)


@pytest.mark.parametrize("ve", [False, True])
def test_owned_append_preserves_audio_without_copy_or_audio_read(recording, monkeypatch, ve):
    path, audio, metadata = recording
    if ve:
        metadata = wav_metadata()
    original = path.read_bytes()
    files = set(path.parent.iterdir())
    data_offset = original.index(b"data") + 8
    data_end = data_offset + struct.unpack("<I", original[data_offset - 4:data_offset])[0]
    real_open = open
    reads = []

    class HeaderOnlyFile:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def read(self, size=-1):
            start = self.wrapped.tell()
            assert size >= 0
            assert start >= data_end or start + size <= data_offset, "audio payload read"
            reads.append(size)
            return self.wrapped.read(size)

    def forbidden(*args, **kwargs):
        pytest.fail("owned append must not copy/create a temporary")

    monkeypatch.setattr(module, "open", lambda *a, **k: HeaderOnlyFile(real_open(*a, **k)), raising=False)
    monkeypatch.setattr(module, "_copy_stream_exact", forbidden)
    monkeypatch.setattr(module.tempfile, "NamedTemporaryFile", forbidden)
    result = append(path, metadata)

    assert result.appended, result.primary_error
    assert result.handles_released
    assert not result.cleanup_paths
    assert not result.retained_handles
    assert sum(reads) < 10000
    assert set(path.parent.iterdir()) == files
    actual_bytes = path.read_bytes()
    assert actual_bytes[:4] == original[:4]
    assert actual_bytes[8:len(original)] == original[8:]
    assert struct.unpack("<I", actual_bytes[4:8])[0] + 8 == len(actual_bytes)
    actual, rate = sf.read(path, dtype="float32", always_2d=True)
    assert rate == 44100
    np.testing.assert_array_equal(actual, audio)
    assert module.read_wav_calibration_metadata(path) == metadata


@pytest.mark.parametrize("damage", ["not-riff", "truncated", "trailing", "bad-chunk", "bad-info",
                                    "metadata", "invalid-metadata"])
def test_ineligible_file_is_rejected_without_mutation(recording, damage):
    path, _, metadata = recording
    original = path.read_bytes()
    if damage == "not-riff":
        path.write_bytes(b"RF64" + original[4:])
    elif damage == "truncated":
        path.write_bytes(original[:-1])
    elif damage == "trailing":
        path.write_bytes(original + b"undeclared")
    elif damage == "bad-chunk":
        _append_raw_chunk(path, b"JUNK" + struct.pack("<I", 100))
    elif damage == "bad-info":
        _append_raw_chunk(path, _chunk(b"LIST", b"INFO" + b"ICMT" + struct.pack("<I", 100)))
    elif damage == "metadata":
        assert module.append_wav_calibration_metadata_result(path, metadata).appended
    else:
        _append_raw_chunk(path, _metadata_list_chunk(b"{invalid}"))
    before = path.read_bytes()

    result = append(path, metadata)

    assert not result.appended
    assert result.primary_error
    assert result.handles_released
    assert path.read_bytes() == before


def test_riff_limit_rejection_does_not_write(recording, monkeypatch):
    path, _, metadata = recording
    before = path.read_bytes()
    monkeypatch.setattr(module, "MAX_RIFF_SIZE", len(before) - 8)

    result = append(path, metadata)

    assert not result.appended
    assert "32-bit" in result.primary_error
    assert path.read_bytes() == before


@pytest.mark.parametrize("damage", ["unknown-tag", "float-tag", "extensible-tag", "pcm23",
                                    "double64", "incomplete-extension", "bad-byte-rate"])
def test_unsupported_owned_format_is_rejected_before_write(recording, monkeypatch, damage):
    path, _, metadata = recording
    raw = bytearray(path.read_bytes())
    fmt_start = raw.index(b"fmt ") + 8
    if damage in {"unknown-tag", "float-tag", "extensible-tag"}:
        tag = {"unknown-tag": 0x1234, "float-tag": 3, "extensible-tag": 0xFFFE}[damage]
        struct.pack_into("<H", raw, fmt_start, tag)
    elif damage == "pcm23":
        # Rounded bytes-per-sample still matches the original block alignment.
        struct.pack_into("<H", raw, fmt_start + 14, 23)
    elif damage == "double64":
        # Even a coherent DOUBLE header is outside the capture writer's PCM24 scope.
        struct.pack_into("<H", raw, fmt_start, 3)
        struct.pack_into("<IHH", raw, fmt_start + 8, 44100 * 16, 16, 64)
    elif damage == "incomplete-extension":
        raw[fmt_start + 16:fmt_start + 16] = b"\0\0"
        struct.pack_into("<I", raw, fmt_start - 4, 17)
        struct.pack_into("<I", raw, 4, len(raw) - 8)
    else:
        struct.pack_into("<I", raw, fmt_start + 8, 1)
    path.write_bytes(raw)
    writes = []
    real_write = module._write_all

    def track_write(destination, data):
        writes.append(bytes(data))
        return real_write(destination, data)

    monkeypatch.setattr(module, "_write_all", track_write)

    result = append(path, metadata)

    assert not result.appended
    assert result.primary_error
    assert result.handles_released
    assert result.rollback_succeeded is None
    assert writes == []
    assert path.read_bytes() == raw


@pytest.mark.parametrize("subtype", ["PCM_16", "FLOAT", "DOUBLE"])
def test_generic_append_retains_other_wav_format_support(recording, subtype):
    path, audio, metadata = recording
    sf.write(path, audio, 44100, subtype=subtype)
    before, _ = sf.read(path, always_2d=True)

    result = module.append_wav_calibration_metadata_result(path, metadata)

    assert result.appended, result.primary_error
    actual, rate = sf.read(path, always_2d=True)
    assert rate == 44100
    np.testing.assert_array_equal(actual, before)
    assert module.read_wav_calibration_metadata(path) == metadata


def test_unrelated_info_comment_is_eligible(recording):
    path, _, metadata = recording
    _append_raw_chunk(path, _chunk(b"LIST", b"INFO" + _chunk(b"ICMT", b"ordinary comment\0")))
    before = path.read_bytes()

    assert append(path, metadata).appended
    assert path.read_bytes()[8:len(before)] == before[8:]


class AppendFaults:
    """Inject one real file boundary fault while preserving real disk operations."""
    def __init__(self, monkeypatch, stage, *, rollback=None, close_fails=False):
        self.handles = []
        self.fsync_calls = 0
        self.fired = False
        self.rollback_fired = False
        real_open = open
        real_fsync = module.os.fsync
        real_scan = module._scan_declared_riff_chunks
        owner = self
        scans = 0

        class File:
            def __init__(self, wrapped):
                self.wrapped = wrapped
                self.writes = 0
                owner.handles.append(self)

            def __getattr__(self, name):
                return getattr(self.wrapped, name)

            def write(self, data):
                self.writes += 1
                if not owner.fired and ((stage == "append" and self.writes == 1)
                                        or (stage == "header" and self.writes == 2)):
                    owner.fired = True
                    self.wrapped.write(data[:2])
                    raise OSError(f"injected {stage} write")
                if owner.fired and rollback == "header":
                    owner.rollback_fired = True
                    raise OSError("injected rollback header")
                return self.wrapped.write(data)

            def truncate(self, *args):
                if rollback == "truncate":
                    owner.rollback_fired = True
                    raise OSError("injected rollback truncate")
                return self.wrapped.truncate(*args)

            def flush(self):
                if stage == "flush" and not owner.fired:
                    owner.fired = True
                    raise OSError("injected flush")
                return self.wrapped.flush()

            def close(self):
                if close_fails:
                    raise OSError("injected close")
                return self.wrapped.close()

        def fsync(fd):
            self.fsync_calls += 1
            if stage == "fsync" and not self.fired:
                self.fired = True
                raise OSError("injected fsync")
            if self.fired and rollback == "fsync":
                self.rollback_fired = True
                raise OSError("injected rollback fsync")
            return real_fsync(fd)

        def scan(*args, **kwargs):
            nonlocal scans
            scans += 1
            if stage == "validation" and scans == 2:
                self.fired = True
                raise ValueError("injected validation")
            if self.fired and rollback == "validation":
                self.rollback_fired = True
                raise ValueError("injected rollback validation")
            return real_scan(*args, **kwargs)

        monkeypatch.setattr(module, "open", lambda *a, **k: File(real_open(*a, **k)), raising=False)
        monkeypatch.setattr(module.os, "fsync", fsync)
        monkeypatch.setattr(module, "_scan_declared_riff_chunks", scan)

    def release(self):
        for handle in self.handles:
            handle.wrapped.close()


@pytest.mark.parametrize("stage", ["append", "header", "flush", "fsync", "validation"])
def test_failed_append_restores_original_file(recording, monkeypatch, stage):
    path, _, metadata = recording
    before = path.read_bytes()
    faults = AppendFaults(monkeypatch, stage)

    result = append(path, metadata)

    assert faults.fired
    assert not result.appended
    assert result.handles_released
    assert result.rollback_succeeded is True
    assert not result.rollback_errors
    assert f"injected {stage}" in result.primary_error
    assert faults.fsync_calls >= 1
    assert path.read_bytes() == before


@pytest.mark.parametrize("rollback", ["truncate", "header", "fsync", "validation"])
def test_uncertain_rollback_reports_original_and_recovery_errors(recording, monkeypatch, rollback):
    path, _, metadata = recording
    faults = AppendFaults(monkeypatch, "append", rollback=rollback)

    result = append(path, metadata)

    assert faults.fired and faults.rollback_fired
    assert not result.appended
    assert result.rollback_succeeded is False
    assert "injected append" in result.primary_error
    assert any(f"injected rollback {rollback}" in error for error in result.rollback_errors)
    assert result.handles_released
    assert path.exists()


@pytest.mark.parametrize("stage,rollback", [(None, None), ("append", None), ("append", "truncate")])
def test_close_failure_transfers_handle_and_preserves_all_diagnostics(recording, monkeypatch, stage, rollback):
    path, _, metadata = recording
    faults = AppendFaults(monkeypatch, stage, rollback=rollback, close_fails=True)
    try:
        result = append(path, metadata)

        assert not result.appended
        assert not result.handles_released
        assert result.retained_handles == ((str(path), faults.handles[0]),)
        assert "injected close" in result.close_errors[0]
        assert f"injected {stage or 'close'}" in result.primary_error
        if rollback:
            assert result.rollback_succeeded is False
            assert "injected rollback truncate" in result.rollback_errors[0]
        assert path.exists()
    finally:
        faults.release()


def test_success_fsyncs_metadata(recording, monkeypatch):
    path, _, metadata = recording
    faults = AppendFaults(monkeypatch, None)

    assert append(path, metadata).appended
    assert faults.fsync_calls == 1


@pytest.mark.parametrize("damage", ["no-data", "duplicate-data", "unaligned-data", "zero-rate",
                                    "zero-channels", "invalid-align"])
def test_invalid_audio_layout_is_rejected_without_mutation(recording, damage):
    path, _, metadata = recording
    raw = bytearray(path.read_bytes())
    data_start = raw.index(b"data")
    fmt_start = raw.index(b"fmt ") + 8
    if damage == "no-data":
        raw[data_start:data_start + 4] = b"JUNK"
    elif damage == "duplicate-data":
        raw.extend(_chunk(b"data", b""))
        raw[4:8] = struct.pack("<I", len(raw) - 8)
    elif damage == "unaligned-data":
        raw = raw[:-4]
        raw[data_start + 4:data_start + 8] = struct.pack("<I", len(raw) - data_start - 8)
        raw[4:8] = struct.pack("<I", len(raw) - 8)
    elif damage == "zero-rate":
        raw[fmt_start + 4:fmt_start + 8] = b"\0" * 4
    elif damage == "zero-channels":
        raw[fmt_start + 2:fmt_start + 4] = b"\0" * 2
    else:
        raw[fmt_start + 12:fmt_start + 14] = struct.pack("<H", 3)
    path.write_bytes(raw)

    result = append(path, metadata)

    assert not result.appended
    assert result.primary_error
    assert path.read_bytes() == raw


def test_ve_header_mismatch_restores_original(recording):
    path, _, _ = recording
    before = path.read_bytes()
    metadata = wav_metadata(sample_rate=51200)

    result = append(path, metadata)

    assert not result.appended
    assert result.rollback_succeeded is True
    assert path.read_bytes() == before


@pytest.mark.parametrize("invalid", [None, {}, {"recorded_channels": []}])
def test_invalid_metadata_is_rejected_before_open(recording, monkeypatch, invalid):
    path, _, _ = recording

    def forbidden(*args, **kwargs):
        pytest.fail("invalid metadata must not open the recording")

    monkeypatch.setattr(module, "open", forbidden, raising=False)
    result = append(path, invalid)
    assert not result.appended
    assert result.primary_error


def test_oversized_metadata_is_rejected_before_write(recording, monkeypatch):
    path, _, metadata = recording
    before = path.read_bytes()
    monkeypatch.setattr(module, "MAX_CALIBRATION_COMMENT_SIZE", 1)

    result = append(path, metadata)

    assert not result.appended
    assert "read limit" in result.primary_error
    assert path.read_bytes() == before


@pytest.mark.parametrize("frames", [1, 3, 5])
@pytest.mark.parametrize("fallback", [False, True])
def test_owned_append_preserves_odd_pcm24_payload_and_pad(tmp_path, monkeypatch, frames, fallback):
    import sys
    from base.streaming_file_writer import StreamingWavWriter
    if fallback:
        monkeypatch.setitem(sys.modules, "soundfile", None)
    path = tmp_path / "odd.wav"
    with StreamingWavWriter(str(path), 44100, 1) as writer:
        writer.write_chunk(np.full(frames, .5, dtype=np.float32))
    original = path.read_bytes()
    data_offset = original.index(b"data") + 8
    assert struct.unpack_from("<I", original, data_offset - 4)[0] == frames * 3
    assert original[data_offset + frames * 3:] == b"\0"
    metadata = _metadata(_channel_with_physical(0, 4))
    result = append(path, metadata)
    assert result.appended, result.primary_error
    raw = path.read_bytes()
    assert raw[8:len(original)] == original[8:]
    assert struct.unpack_from("<I", raw, 4)[0] + 8 == len(raw)
    np.testing.assert_array_equal(sf.read(path, dtype="float32")[0], np.full(frames, .5))
    assert module.read_wav_calibration_metadata(path) == metadata
