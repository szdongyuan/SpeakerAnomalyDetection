import json
import os
import struct
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.io import wavfile

from base import wav_calibration_metadata

WavCalibrationResolution = wav_calibration_metadata.WavCalibrationResolution
normalize_wav_calibration_metadata = wav_calibration_metadata.normalize_wav_calibration_metadata
resolve_wav_channel_v2pa_factor = wav_calibration_metadata.resolve_wav_channel_v2pa_factor


class _BoundedReadFile:
    def __init__(self, raw_file, max_read_size=65536):
        self._raw_file = raw_file
        self.max_read_size = max_read_size
        self.read_sizes = []

    def read(self, size=-1):
        self.read_sizes.append(size)
        if size < 0 or size > self.max_read_size:
            raise AssertionError(f"oversized or unbounded read: {size}")
        return self._raw_file.read(size)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._raw_file.__exit__(exc_type, exc_value, traceback)

    def __getattr__(self, name):
        return getattr(self._raw_file, name)


def _install_bounded_open(monkeypatch, path):
    real_open = open
    wrappers = []

    def bounded_open(file, mode="r", *args, **kwargs):
        raw_file = real_open(file, mode, *args, **kwargs)
        if os.fspath(file) != os.fspath(path):
            return raw_file
        wrapper = _BoundedReadFile(raw_file)
        wrappers.append(wrapper)
        return wrapper

    monkeypatch.setattr("builtins.open", bounded_open)
    return wrappers


def _metadata(*channels):
    return {"recorded_channels": list(channels)}


def _channel(index=0, factor=2.5, standard_spl=94.0, calibrated=True):
    return {
        "wav_channel_index": index,
        "v2pa_factor": factor,
        "standard_spl": standard_spl,
        "calibrated": calibrated,
    }


def _channel_with_physical(
    index=0,
    physical=0,
    factor=2.5,
    standard_spl=94.0,
    calibrated=True,
):
    channel = _channel(index, factor, standard_spl, calibrated)
    channel["physical_input_channel"] = physical
    return channel


def _chunk(chunk_id, payload, padding=b"\x00"):
    return chunk_id + struct.pack("<I", len(payload)) + payload + (padding if len(payload) % 2 else b"")


def _metadata_list_chunk(payload):
    comment = b"mic_calibration=" + payload + b"\x00"
    return _chunk(b"LIST", b"INFO" + _chunk(b"ICMT", comment))


def _append_raw_chunk(path, chunk):
    data = path.read_bytes()
    riff_end = struct.unpack("<I", data[4:8])[0] + 8
    updated = data[:riff_end] + chunk
    updated = updated[:4] + struct.pack("<I", len(updated) - 8) + updated[8:]
    path.write_bytes(updated)


def _write_wav(path, channels=1):
    audio = np.linspace(-0.25, 0.25, 32, dtype=np.float32)
    if channels > 1:
        audio = np.column_stack([audio * (index + 1) for index in range(channels)])
    wavfile.write(path, 16000, audio)
    return audio


def _append_temp_files(path):
    return list(path.parent.glob(f".{path.name}.*.tmp"))


def test_normalize_wav_metadata_accepts_develop_compatible_values():
    payload = _metadata(
        {
            "wav_channel_index": "2",
            "v2pa_factor": "2.5",
            "standard_spl": "94",
            "calibrated": 1,
            "ignored": "field",
        }
    )

    assert normalize_wav_calibration_metadata(payload) == _metadata(_channel(index=2))


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        _metadata(),
        _metadata("not-a-record"),
        _metadata({"calibrated": False}),
        _metadata(_channel(index=True)),
        _metadata(_channel(index=-1)),
        _metadata(_channel(index=1.5)),
        _metadata(_channel(index=" 1")),
        _metadata(_channel(index="+1")),
        _metadata(_channel(index=0), _channel(index="0")),
        _metadata(_channel(factor=None)),
        _metadata(_channel(factor=True)),
        _metadata(_channel(factor=0)),
        _metadata(_channel(factor=-1)),
        _metadata(_channel(factor=float("inf"))),
        _metadata(_channel(factor="not-a-number")),
    ],
)
def test_normalize_wav_metadata_rejects_invalid_payload(payload):
    assert normalize_wav_calibration_metadata(payload) is None


def test_normalize_wav_metadata_coerces_supported_indices_and_optional_fields():
    payload = _metadata(
        _channel(index=2.0, factor=99, standard_spl=True, calibrated=False),
        _channel(index=np.int64(3), factor=3, standard_spl="nan", calibrated=[]),
    )

    assert normalize_wav_calibration_metadata(payload) == _metadata(
        _channel(index=2, factor=None, standard_spl=None, calibrated=False),
        _channel(index=3, factor=None, standard_spl=None, calibrated=False),
    )


def test_normalize_wav_metadata_preserves_complete_physical_channel_identity():
    payload = _metadata(
        _channel_with_physical(index=0, physical=np.int64(2)),
        _channel_with_physical(index=1, physical=4, factor=None, calibrated=False),
    )

    assert normalize_wav_calibration_metadata(payload) == _metadata(
        _channel_with_physical(index=0, physical=2),
        _channel_with_physical(index=1, physical=4, factor=None, calibrated=False),
    )


@pytest.mark.parametrize(
    "payload",
    [
        _metadata(_channel_with_physical(index=0, physical=2), _channel(index=1)),
        _metadata(
            _channel_with_physical(index=0, physical=2),
            _channel_with_physical(index=1, physical=2),
        ),
        _metadata(_channel_with_physical(physical=True)),
        _metadata(_channel_with_physical(physical=-1)),
        _metadata(_channel_with_physical(physical=1.5)),
    ],
)
def test_normalize_wav_metadata_rejects_invalid_physical_channel_identity(payload):
    assert normalize_wav_calibration_metadata(payload) is None


def test_physical_channel_identity_does_not_change_calibration_resolution():
    legacy = _metadata(_channel(index=1, factor=3.25))
    physical = _metadata(_channel_with_physical(index=1, physical=7, factor=3.25))

    assert resolve_wav_channel_v2pa_factor(physical, 1) == resolve_wav_channel_v2pa_factor(
        legacy,
        1,
    )


@pytest.mark.parametrize(
    ("metadata", "index", "expected"),
    [
        (_metadata(_channel(index=1, factor=3.25)), 1, WavCalibrationResolution(3.25, True, True)),
        (_metadata(_channel(index=1)), 2, WavCalibrationResolution(1.0, True, False)),
        (
            _metadata(_channel(index=1, factor=None, calibrated=False)),
            1,
            WavCalibrationResolution(1.0, True, False),
        ),
        ({"recorded_channels": []}, 0, WavCalibrationResolution(1.0, False, False)),
        (_metadata(_channel()), True, WavCalibrationResolution(1.0, False, False)),
    ],
)
def test_resolve_wav_channel_v2pa_factor(metadata, index, expected):
    assert resolve_wav_channel_v2pa_factor(metadata, index) == expected


@pytest.mark.parametrize("channels", [1, 2])
def test_append_and_read_wav_calibration_metadata_round_trip(tmp_path, channels):
    append_wav_calibration_metadata = wav_calibration_metadata.append_wav_calibration_metadata
    read_wav_calibration_metadata = wav_calibration_metadata.read_wav_calibration_metadata
    path = tmp_path / f"{channels}-channel.wav"
    original_audio = _write_wav(path, channels)
    metadata = _metadata(*[_channel(index=i, factor=2.5 + i) for i in range(channels)])

    assert append_wav_calibration_metadata(path, metadata) is True
    assert read_wav_calibration_metadata(path) == metadata
    sample_rate, actual_audio = wavfile.read(path)
    assert sample_rate == 16000
    np.testing.assert_array_equal(actual_audio, original_audio)


@pytest.mark.parametrize("include_physical", [False, True])
def test_append_and_read_round_trip_preserves_old_and_new_schema(tmp_path, include_physical):
    append_wav_calibration_metadata = wav_calibration_metadata.append_wav_calibration_metadata
    read_wav_calibration_metadata = wav_calibration_metadata.read_wav_calibration_metadata
    path = tmp_path / f"schema-{include_physical}.wav"
    _write_wav(path, channels=2)
    if include_physical:
        metadata = _metadata(
            _channel_with_physical(index=0, physical=0),
            _channel_with_physical(index=1, physical=2),
        )
    else:
        metadata = _metadata(_channel(index=0), _channel(index=1))

    assert append_wav_calibration_metadata(path, metadata) is True
    assert read_wav_calibration_metadata(path) == metadata


def test_diagnostic_reader_distinguishes_absent_valid_and_invalid(tmp_path):
    from base.wav_calibration_metadata import (
        WavCalibrationMetadataReadStatus,
        inspect_wav_calibration_metadata,
    )

    absent_path = tmp_path / "absent.wav"
    valid_path = tmp_path / "valid.wav"
    invalid_path = tmp_path / "invalid.wav"
    for path in (absent_path, valid_path, invalid_path):
        _write_wav(path)

    valid_metadata = _metadata(_channel_with_physical(physical=2))
    _append_raw_chunk(
        valid_path,
        _metadata_list_chunk(json.dumps(valid_metadata).encode("utf-8")),
    )
    _append_raw_chunk(invalid_path, _metadata_list_chunk(b"{invalid"))

    absent = inspect_wav_calibration_metadata(absent_path)
    assert absent.status is WavCalibrationMetadataReadStatus.ABSENT
    assert absent.metadata is None

    valid = inspect_wav_calibration_metadata(valid_path)
    assert valid.status is WavCalibrationMetadataReadStatus.VALID
    assert valid.metadata == valid_metadata

    invalid = inspect_wav_calibration_metadata(invalid_path)
    assert invalid.status is WavCalibrationMetadataReadStatus.INVALID
    assert invalid.metadata is None

    assert wav_calibration_metadata.read_wav_calibration_metadata(absent_path) is None
    assert wav_calibration_metadata.read_wav_calibration_metadata(valid_path) == valid_metadata
    assert wav_calibration_metadata.read_wav_calibration_metadata(invalid_path) is None


def test_diagnostic_reader_treats_undecodable_nonmetadata_icmt_as_absent(tmp_path):
    path = tmp_path / "binary-comment.wav"
    _write_wav(path)
    _append_raw_chunk(path, _chunk(b"LIST", b"INFO" + _chunk(b"ICMT", b"\xffbinary")))

    result = wav_calibration_metadata.inspect_wav_calibration_metadata(path)

    assert result.status is wav_calibration_metadata.WavCalibrationMetadataReadStatus.ABSENT
    assert result.metadata is None


def test_diagnostic_reader_treats_prefixed_bad_utf8_as_invalid(tmp_path):
    path = tmp_path / "invalid-prefixed-comment.wav"
    _write_wav(path)
    payload = b"mic_calibration=" + b"\xff"
    _append_raw_chunk(path, _chunk(b"LIST", b"INFO" + _chunk(b"ICMT", payload)))

    result = wav_calibration_metadata.inspect_wav_calibration_metadata(path)

    assert result.status is wav_calibration_metadata.WavCalibrationMetadataReadStatus.INVALID
    assert result.metadata is None


@pytest.mark.parametrize(
    ("prefix", "expected_status"),
    [
        (b"ordinary_binary=", wav_calibration_metadata.WavCalibrationMetadataReadStatus.ABSENT),
        (b"mic_calibration=", wav_calibration_metadata.WavCalibrationMetadataReadStatus.INVALID),
    ],
)
def test_diagnostic_reader_probes_oversized_icmt_prefix_without_full_read(
    tmp_path,
    monkeypatch,
    prefix,
    expected_status,
):
    path = tmp_path / f"oversized-{expected_status.value}.wav"
    _write_wav(path)
    payload = prefix + (b"\xff" * wav_calibration_metadata.MAX_CALIBRATION_COMMENT_SIZE)
    _append_raw_chunk(path, _chunk(b"LIST", b"INFO" + _chunk(b"ICMT", payload)))
    wrappers = _install_bounded_open(monkeypatch, path)

    result = wav_calibration_metadata.inspect_wav_calibration_metadata(path)

    assert result.status is expected_status
    assert result.metadata is None
    assert wrappers
    assert all(0 <= size <= 65536 for wrapper in wrappers for size in wrapper.read_sizes)
    assert any(
        len(wav_calibration_metadata.CALIBRATION_COMMENT_PREFIX.encode("ascii"))
        in wrapper.read_sizes
        for wrapper in wrappers
    )


def test_append_writes_valid_chunk_lengths_and_odd_padding(tmp_path):
    append_wav_calibration_metadata = wav_calibration_metadata.append_wav_calibration_metadata
    path = tmp_path / "odd.wav"
    _write_wav(path)

    assert append_wav_calibration_metadata(path, _metadata(_channel(factor=2.55))) is True

    data = path.read_bytes()
    assert struct.unpack("<I", data[4:8])[0] == len(data) - 8
    list_offset = data.rfind(b"LIST")
    list_size = struct.unpack("<I", data[list_offset + 4 : list_offset + 8])[0]
    assert list_offset + 8 + list_size + (list_size % 2) == len(data)
    icmt_offset = list_offset + 12
    assert data[icmt_offset : icmt_offset + 4] == b"ICMT"
    icmt_size = struct.unpack("<I", data[icmt_offset + 4 : icmt_offset + 8])[0]
    icmt_end = icmt_offset + 8 + icmt_size
    assert icmt_size % 2 == 1
    assert data[icmt_end : icmt_end + 1] == b"\x00"


def test_read_uses_last_valid_comment_and_ignores_unrelated_list(tmp_path):
    append_wav_calibration_metadata = wav_calibration_metadata.append_wav_calibration_metadata
    read_wav_calibration_metadata = wav_calibration_metadata.read_wav_calibration_metadata
    path = tmp_path / "multiple.wav"
    _write_wav(path)
    first = _metadata(_channel(factor=1.5))
    last = _metadata(_channel(factor=4.5))
    assert append_wav_calibration_metadata(path, first)
    _append_raw_chunk(path, _chunk(b"LIST", b"adtl" + _chunk(b"labl", b"other")))
    _append_raw_chunk(path, _metadata_list_chunk(b"{invalid"))
    _append_raw_chunk(path, _metadata_list_chunk(json.dumps(last).encode("utf-8")))

    assert read_wav_calibration_metadata(path) == last


@pytest.mark.parametrize(
    ("payload", "diagnostic"),
    [
        (b"{invalid", "Invalid WAV calibration metadata"),
        (json.dumps({"recorded_channels": []}).encode("utf-8"), "Invalid WAV calibration metadata"),
    ],
)
def test_read_invalid_json_or_payload_returns_none_and_logs(tmp_path, payload, diagnostic):
    read_wav_calibration_metadata = wav_calibration_metadata.read_wav_calibration_metadata
    path = tmp_path / "invalid-metadata.wav"
    _write_wav(path)
    _append_raw_chunk(path, _metadata_list_chunk(payload))
    logger = Mock()

    assert read_wav_calibration_metadata(path, logger=logger) is None
    assert any(diagnostic in call.args[0] for call in logger.warning.call_args_list)
    assert logger.warning.call_count == 1


def test_read_wav_without_calibration_comment_returns_none_and_logs(tmp_path):
    read_wav_calibration_metadata = wav_calibration_metadata.read_wav_calibration_metadata
    path = tmp_path / "no-calibration-comment.wav"
    _write_wav(path)
    logger = Mock()

    assert read_wav_calibration_metadata(path, logger=logger) is None
    assert logger.warning.call_count == 1
    assert "no valid" in logger.warning.call_args.args[0].lower()


@pytest.mark.parametrize("kind", ["declared_past_eof", "truncated_chunk", "nonzero_padding", "trailing_declared"])
def test_read_rejects_corrupt_riff_and_logs(tmp_path, kind):
    read_wav_calibration_metadata = wav_calibration_metadata.read_wav_calibration_metadata
    path = tmp_path / f"{kind}.wav"
    _write_wav(path)
    data = bytearray(path.read_bytes())
    if kind == "declared_past_eof":
        data[4:8] = struct.pack("<I", len(data) + 100)
    elif kind == "truncated_chunk":
        chunk = b"JUNK" + struct.pack("<I", 10) + b"abc"
        data.extend(chunk)
        data[4:8] = struct.pack("<I", len(data) - 8)
    elif kind == "nonzero_padding":
        chunk = _chunk(b"JUNK", b"x", padding=b"!")
        data.extend(chunk)
        data[4:8] = struct.pack("<I", len(data) - 8)
    else:
        data.extend(b"x")
        data[4:8] = struct.pack("<I", len(data) - 8)
    path.write_bytes(data)
    logger = Mock()

    assert read_wav_calibration_metadata(path, logger=logger) is None
    logger.warning.assert_called()


def test_read_ignores_bytes_outside_declared_riff_and_append_truncates_them(tmp_path):
    append_wav_calibration_metadata = wav_calibration_metadata.append_wav_calibration_metadata
    read_wav_calibration_metadata = wav_calibration_metadata.read_wav_calibration_metadata
    path = tmp_path / "trailing.wav"
    _write_wav(path)
    original = path.read_bytes()
    path.write_bytes(original + b"untrusted-tail")

    assert read_wav_calibration_metadata(path) is None
    assert append_wav_calibration_metadata(path, _metadata(_channel())) is True
    assert b"untrusted-tail" not in path.read_bytes()
    assert read_wav_calibration_metadata(path) == _metadata(_channel())
    assert _append_temp_files(path) == []


def test_non_wav_read_and_append_fail_with_diagnostics(tmp_path):
    append_wav_calibration_metadata = wav_calibration_metadata.append_wav_calibration_metadata
    read_wav_calibration_metadata = wav_calibration_metadata.read_wav_calibration_metadata
    path = tmp_path / "not.wav"
    original = b"not a wave"
    path.write_bytes(original)
    logger = Mock()

    assert read_wav_calibration_metadata(path, logger=logger) is None
    assert append_wav_calibration_metadata(path, _metadata(_channel()), logger=logger) is False
    assert path.read_bytes() == original
    logger.warning.assert_called()


def test_append_rejects_malformed_info_subchunk_without_mutating_wav(tmp_path):
    append_wav_calibration_metadata = wav_calibration_metadata.append_wav_calibration_metadata
    path = tmp_path / "malformed-info.wav"
    _write_wav(path)
    malformed_info = b"INFO" + b"ICMT" + struct.pack("<I", 5) + b"ab"
    _append_raw_chunk(path, _chunk(b"LIST", malformed_info))
    original_bytes = path.read_bytes()
    logger = Mock()

    assert append_wav_calibration_metadata(path, _metadata(_channel()), logger=logger) is False
    assert path.read_bytes() == original_bytes
    assert any("Invalid WAV LIST/INFO" in call.args[0] for call in logger.warning.call_args_list)


def test_append_streams_large_list_validation_without_oversized_reads(tmp_path, monkeypatch):
    append_wav_calibration_metadata = wav_calibration_metadata.append_wav_calibration_metadata
    read_wav_calibration_metadata = wav_calibration_metadata.read_wav_calibration_metadata
    path = tmp_path / "large-lists.wav"
    _write_wav(path)
    _append_raw_chunk(path, _chunk(b"LIST", b"adtl" + _chunk(b"labl", b"x" * 131072)))
    large_info = b"INFO" + _chunk(b"JUNK", b"y" * 131072)
    _append_raw_chunk(path, _chunk(b"LIST", large_info))
    wrappers = _install_bounded_open(monkeypatch, path)

    assert append_wav_calibration_metadata(path, _metadata(_channel())) is True
    assert read_wav_calibration_metadata(path) == _metadata(_channel())
    assert wrappers
    assert all(0 <= size <= 65536 for wrapper in wrappers for size in wrapper.read_sizes)


def test_append_rejects_malformed_non_info_list_without_mutating_wav(tmp_path):
    append_wav_calibration_metadata = wav_calibration_metadata.append_wav_calibration_metadata
    path = tmp_path / "malformed-adtl.wav"
    _write_wav(path)
    malformed_adtl = b"adtl" + b"labl" + struct.pack("<I", 5) + b"ab"
    _append_raw_chunk(path, _chunk(b"LIST", malformed_adtl))
    original_bytes = path.read_bytes()
    logger = Mock()

    assert append_wav_calibration_metadata(path, _metadata(_channel()), logger=logger) is False
    assert path.read_bytes() == original_bytes
    assert any("Invalid WAV LIST" in call.args[0] for call in logger.warning.call_args_list)


def test_read_skips_oversized_icmt_with_bounded_reads_and_uses_later_valid_comment(
    tmp_path,
    monkeypatch,
):
    read_wav_calibration_metadata = wav_calibration_metadata.read_wav_calibration_metadata
    path = tmp_path / "oversized-icmt.wav"
    _write_wav(path)
    oversized_comment = b"mic_calibration=" + (b"x" * (1024 * 1024))
    _append_raw_chunk(path, _chunk(b"LIST", b"INFO" + _chunk(b"ICMT", oversized_comment)))
    expected = _metadata(_channel(factor=7.25))
    _append_raw_chunk(path, _metadata_list_chunk(json.dumps(expected).encode("utf-8")))
    logger = Mock()
    wrappers = _install_bounded_open(monkeypatch, path)

    assert read_wav_calibration_metadata(path, logger=logger) == expected
    assert all(0 <= size <= 65536 for wrapper in wrappers for size in wrapper.read_sizes)
    assert any("exceeds" in call.args[0] and "limit" in call.args[0] for call in logger.warning.call_args_list)


def test_append_temp_fsync_failure_keeps_original_readable_and_cleans_temp(
    tmp_path,
    monkeypatch,
):
    append_wav_calibration_metadata = wav_calibration_metadata.append_wav_calibration_metadata
    path = tmp_path / "large-tail.wav"
    original_audio = _write_wav(path)
    path.write_bytes(path.read_bytes() + (b"T" * 196608))
    original_bytes = path.read_bytes()
    logger = Mock()
    wrappers = _install_bounded_open(monkeypatch, path)
    monkeypatch.setattr(
        "base.wav_calibration_metadata.os.fsync",
        Mock(side_effect=OSError("injected temp fsync failure")),
    )

    assert append_wav_calibration_metadata(path, _metadata(_channel()), logger=logger) is False
    assert path.read_bytes() == original_bytes
    _, actual_audio = wavfile.read(path)
    np.testing.assert_array_equal(actual_audio, original_audio)
    assert _append_temp_files(path) == []
    assert wrappers
    assert all(0 <= size <= 65536 for wrapper in wrappers for size in wrapper.read_sizes)
    assert any("injected temp fsync failure" in call.args[0] for call in logger.warning.call_args_list)


def test_append_temp_write_failure_keeps_original_bytes_and_cleans_temp(
    tmp_path,
    monkeypatch,
):
    append_wav_calibration_metadata = wav_calibration_metadata.append_wav_calibration_metadata
    path = tmp_path / "write-failure.wav"
    original_audio = _write_wav(path)
    original_bytes = path.read_bytes()
    logger = Mock()
    real_write_all = wav_calibration_metadata._write_all

    def fail_metadata_write(destination, data):
        if bytes(data[:4]) == b"LIST":
            destination.write(data[:17])
            raise OSError("injected temporary write failure")
        return real_write_all(destination, data)

    monkeypatch.setattr(
        "base.wav_calibration_metadata._write_all",
        fail_metadata_write,
    )

    assert append_wav_calibration_metadata(path, _metadata(_channel()), logger=logger) is False
    assert path.read_bytes() == original_bytes
    _, actual_audio = wavfile.read(path)
    np.testing.assert_array_equal(actual_audio, original_audio)
    assert _append_temp_files(path) == []
    assert any("temporary write failure" in call.args[0] for call in logger.warning.call_args_list)


def test_append_replace_failure_keeps_original_bytes_and_cleans_temp(
    tmp_path,
    monkeypatch,
):
    append_wav_calibration_metadata = wav_calibration_metadata.append_wav_calibration_metadata
    path = tmp_path / "replace-failure.wav"
    original_audio = _write_wav(path)
    original_bytes = path.read_bytes()
    logger = Mock()
    monkeypatch.setattr(
        "base.wav_calibration_metadata.os.replace",
        Mock(side_effect=OSError("injected replace failure")),
    )

    assert append_wav_calibration_metadata(path, _metadata(_channel()), logger=logger) is False
    assert path.read_bytes() == original_bytes
    _, actual_audio = wavfile.read(path)
    np.testing.assert_array_equal(actual_audio, original_audio)
    assert _append_temp_files(path) == []
    assert any("injected replace failure" in call.args[0] for call in logger.warning.call_args_list)


def test_append_temp_validation_failure_keeps_original_bytes_and_cleans_temp(
    tmp_path,
    monkeypatch,
):
    append_wav_calibration_metadata = wav_calibration_metadata.append_wav_calibration_metadata
    path = tmp_path / "validation-failure.wav"
    original_audio = _write_wav(path)
    original_bytes = path.read_bytes()
    logger = Mock()
    monkeypatch.setattr(
        "base.wav_calibration_metadata._validate_appended_wav",
        Mock(return_value=False),
    )

    assert append_wav_calibration_metadata(path, _metadata(_channel()), logger=logger) is False
    assert path.read_bytes() == original_bytes
    _, actual_audio = wavfile.read(path)
    np.testing.assert_array_equal(actual_audio, original_audio)
    assert _append_temp_files(path) == []
    assert any("validation failed" in call.args[0] for call in logger.warning.call_args_list)


@pytest.mark.parametrize("stage", ["source", "temporary", "validation"])
def test_structured_append_reports_uncertain_file_ownership(tmp_path, monkeypatch, stage):
    from unit_test.base.recording_process_fakes import MetadataFileFaults
    path = tmp_path / "ownership.wav"
    _write_wav(path)
    original = path.read_bytes()
    faults = MetadataFileFaults(stage)
    faults.install(monkeypatch)
    try:
        result = wav_calibration_metadata.append_wav_calibration_metadata_result(path, _metadata(_channel()))
        assert not result.appended and not result.handles_released
        assert result.cleanup_paths == tuple(faults.temporary_paths)
        assert result.retained_handles and result.close_errors
        assert all(os.path.exists(temporary) for temporary in result.cleanup_paths)
        assert path.read_bytes() == original
    finally:
        faults.release_all()


def test_structured_append_safe_rejection_preserves_legacy_bool_api(tmp_path):
    path = tmp_path / "rejected.wav"
    _write_wav(path)
    result = wav_calibration_metadata.append_wav_calibration_metadata_result(path, None)
    assert not result.appended and result.handles_released
    assert result.cleanup_paths == () and result.retained_handles == ()
    assert wav_calibration_metadata.append_wav_calibration_metadata(path, None) is False
    assert wav_calibration_metadata.append_wav_calibration_metadata(path, _metadata(_channel())) is True


@pytest.mark.parametrize("stage", ["source", "temporary", "validation"])
def test_legacy_append_close_failure_still_returns_false(tmp_path, monkeypatch, stage):
    from unit_test.base.recording_process_fakes import MetadataFileFaults
    path = tmp_path / "legacy-close.wav"
    _write_wav(path)
    faults = MetadataFileFaults(stage)
    faults.install(monkeypatch)
    logger = Mock()
    try:
        assert wav_calibration_metadata.append_wav_calibration_metadata(path, _metadata(_channel()), logger) is False
        assert any("close failed" in call.args[0] for call in logger.warning.call_args_list)
    finally:
        faults.release_all()


def test_append_riff_size_overflow_does_not_mutate_readable_wav(tmp_path, monkeypatch):
    append_wav_calibration_metadata = wav_calibration_metadata.append_wav_calibration_metadata
    path = tmp_path / "overflow.wav"
    original_audio = _write_wav(path)
    original_bytes = path.read_bytes()
    logger = Mock()
    monkeypatch.setattr("base.wav_calibration_metadata.MAX_RIFF_SIZE", len(original_bytes) - 8)

    assert append_wav_calibration_metadata(path, _metadata(_channel()), logger=logger) is False
    assert path.read_bytes() == original_bytes
    _, actual_audio = wavfile.read(path)
    np.testing.assert_array_equal(actual_audio, original_audio)
    assert any("32-bit" in call.args[0] for call in logger.warning.call_args_list)


def test_append_rejects_generated_comment_over_reader_limit_without_mutation(tmp_path):
    append_wav_calibration_metadata = wav_calibration_metadata.append_wav_calibration_metadata
    path = tmp_path / "oversized-generated-comment.wav"
    _write_wav(path)
    original_bytes = path.read_bytes()
    metadata = _metadata(
        *[
            _channel(index=index, factor=2.5 + (index % 10))
            for index in range(14000)
        ]
    )
    logger = Mock()

    assert append_wav_calibration_metadata(path, metadata, logger=logger) is False
    assert path.read_bytes() == original_bytes
    assert any("exceeds" in call.args[0] and "limit" in call.args[0] for call in logger.warning.call_args_list)


def test_append_serialization_failure_returns_false_without_mutation(tmp_path):
    append_wav_calibration_metadata = wav_calibration_metadata.append_wav_calibration_metadata
    path = tmp_path / "serialization-failure.wav"
    _write_wav(path)
    original_bytes = path.read_bytes()
    metadata = _metadata(_channel(index=10**5000))
    logger = Mock()

    assert append_wav_calibration_metadata(path, metadata, logger=logger) is False
    assert path.read_bytes() == original_bytes
    assert logger.warning.call_count == 1
    assert "serializ" in logger.warning.call_args.args[0].lower()


def test_save_audio_with_calibration_metadata_success_and_no_metadata(tmp_path):
    from base.save_data import save_audio_with_calibration_metadata

    read_wav_calibration_metadata = wav_calibration_metadata.read_wav_calibration_metadata
    audio = np.column_stack(
        [np.linspace(-0.2, 0.2, 16, dtype=np.float32), np.linspace(0.2, -0.2, 16, dtype=np.float32)]
    )
    metadata = _metadata(_channel(index=0), _channel(index=1, factor=None, calibrated=False))
    with_metadata = tmp_path / "nested" / "with.wav"
    without_metadata = tmp_path / "without.wav"

    assert save_audio_with_calibration_metadata(with_metadata, audio, 22050, metadata) is True
    assert read_wav_calibration_metadata(with_metadata) == metadata
    assert save_audio_with_calibration_metadata(without_metadata, audio, 22050, None) is True
    assert read_wav_calibration_metadata(without_metadata) is None


def test_save_audio_with_metadata_append_failure_preserves_audio(tmp_path):
    from base.save_data import save_audio_with_calibration_metadata

    path = tmp_path / "append-failure.wav"
    audio = np.linspace(-0.5, 0.5, 24, dtype=np.float32)
    logger = Mock()
    invalid_metadata = {"recorded_channels": []}

    assert save_audio_with_calibration_metadata(path, audio, 8000, invalid_metadata, logger) is False
    sample_rate, actual_audio = wavfile.read(path)
    assert sample_rate == 8000
    np.testing.assert_array_equal(actual_audio, audio)
    logger.warning.assert_called()


@pytest.mark.parametrize("invalid_metadata", [{}, []])
def test_save_audio_with_falsey_invalid_metadata_returns_false(tmp_path, invalid_metadata):
    from base.save_data import save_audio_with_calibration_metadata

    path = tmp_path / "falsey-invalid.wav"
    audio = np.linspace(-0.5, 0.5, 24, dtype=np.float32)
    logger = Mock()

    assert save_audio_with_calibration_metadata(path, audio, 8000, invalid_metadata, logger) is False
    _, actual_audio = wavfile.read(path)
    np.testing.assert_array_equal(actual_audio, audio)
    logger.warning.assert_called()
