import json
import math
import os
import struct
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Optional

from consts.audio_consts import CHUNK_HEADER_SIZE, MAX_RIFF_SIZE, RIFF_HEADER_SIZE


@dataclass(frozen=True)
class WavCalibrationResolution:
    factor: float
    has_valid_metadata: bool
    used_file_metadata: bool


def normalize_wav_calibration_metadata(payload: Any) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None

    recorded_channels = payload.get("recorded_channels")
    if not isinstance(recorded_channels, list):
        return None

    normalized_channels = []
    seen_wav_channel_indices = set()
    for channel in recorded_channels:
        normalized_channel = _normalize_channel(channel)
        if normalized_channel is None:
            return None
        wav_channel_index = normalized_channel["wav_channel_index"]
        if wav_channel_index in seen_wav_channel_indices:
            return None
        seen_wav_channel_indices.add(wav_channel_index)
        normalized_channels.append(normalized_channel)

    if not normalized_channels:
        return None

    return {"recorded_channels": normalized_channels}


def resolve_wav_channel_v2pa_factor(metadata: Any, wav_channel_index: int) -> WavCalibrationResolution:
    try:
        normalized = normalize_wav_calibration_metadata(metadata)
        selected_index = _coerce_wav_channel_index(wav_channel_index)
    except (OverflowError, TypeError, ValueError):
        return WavCalibrationResolution(1.0, False, False)

    if normalized is None:
        return WavCalibrationResolution(1.0, False, False)

    for channel in normalized["recorded_channels"]:
        if channel["wav_channel_index"] != selected_index:
            continue
        if channel["calibrated"] and channel["v2pa_factor"] is not None:
            return WavCalibrationResolution(float(channel["v2pa_factor"]), True, True)
        return WavCalibrationResolution(1.0, True, False)

    return WavCalibrationResolution(1.0, True, False)


def append_wav_calibration_metadata(path, metadata, logger=None) -> bool:
    try:
        normalized = normalize_wav_calibration_metadata(metadata)
        if normalized is None:
            _log_metadata_issue(
                logger,
                "Invalid WAV calibration metadata",
                ValueError("metadata payload was rejected"),
            )
            return False

        comment = "mic_calibration=" + json.dumps(normalized, ensure_ascii=True, separators=(",", ":"))
        list_chunk = _build_info_comment_list_chunk(comment)

        with open(path, "r+b") as wav_file:
            if not _is_riff_wave_file(wav_file):
                _log_metadata_issue(
                    logger,
                    "Unsupported WAV file",
                    ValueError("file is not a RIFF/WAVE file"),
                )
                return False
            wav_file.seek(0, os.SEEK_END)
            file_size = wav_file.tell()
            append_offset = _read_authoritative_riff_end(wav_file, file_size, logger)
            if append_offset is None:
                return False
            if _scan_declared_riff_chunks(wav_file, append_offset, logger, read_metadata=False) is None:
                return False

            new_riff_size = append_offset + len(list_chunk) - 8
            if new_riff_size > MAX_RIFF_SIZE:
                _log_metadata_issue(
                    logger,
                    "Failed to append WAV calibration metadata",
                    ValueError("RIFF size exceeds 32-bit limit"),
                )
                return False

            wav_file.seek(append_offset)
            wav_file.truncate()
            wav_file.write(list_chunk)
            file_size = wav_file.tell()
            wav_file.seek(4)
            wav_file.write(struct.pack("<I", file_size - 8))
        return True
    except Exception as exc:
        _log_metadata_issue(logger, "Failed to append WAV calibration metadata", exc)
        return False


def read_wav_calibration_metadata(path, logger=None) -> Optional[dict]:
    last_valid_metadata = None
    try:
        with open(path, "rb") as wav_file:
            if not _is_riff_wave_file(wav_file):
                return None
            wav_file.seek(0, os.SEEK_END)
            file_size = wav_file.tell()
            wav_file.seek(4)
            riff_size = struct.unpack("<I", wav_file.read(4))[0]
            if riff_size + 8 > file_size:
                _log_metadata_issue(
                    logger,
                    "Invalid WAV RIFF structure",
                    ValueError("declared RIFF size exceeds file size"),
                )
                return None
            parse_end = riff_size + 8
            last_valid_metadata = _scan_declared_riff_chunks(wav_file, parse_end, logger, read_metadata=True)
            if last_valid_metadata is None:
                return None
    except Exception as exc:
        _log_metadata_issue(logger, "Failed to read WAV calibration metadata", exc)
        return None

    return last_valid_metadata


def _normalize_channel(channel: Any) -> Optional[dict]:
    if not isinstance(channel, dict):
        return None

    try:
        wav_channel_index = _coerce_wav_channel_index(channel.get("wav_channel_index"))
    except (TypeError, ValueError):
        return None
    calibrated = bool(channel.get("calibrated", False))

    if calibrated:
        v2pa_factor = _coerce_positive_finite_float(channel.get("v2pa_factor"))
    else:
        v2pa_factor = None

    return {
        "wav_channel_index": wav_channel_index,
        "v2pa_factor": v2pa_factor,
        "standard_spl": _coerce_optional_finite_float(channel.get("standard_spl")),
        "calibrated": calibrated,
    }


def _build_info_comment_list_chunk(comment: str) -> bytes:
    comment_payload = comment.encode("utf-8") + b"\x00"
    icmt_chunk = _build_chunk(b"ICMT", comment_payload)
    return _build_chunk(b"LIST", b"INFO" + icmt_chunk)


def _build_chunk(chunk_id: bytes, payload: bytes) -> bytes:
    chunk = chunk_id + struct.pack("<I", len(payload)) + payload
    if len(payload) % 2:
        chunk += b"\x00"
    return chunk


def _is_riff_wave_file(wav_file) -> bool:
    wav_file.seek(0)
    header = wav_file.read(RIFF_HEADER_SIZE)
    return len(header) == RIFF_HEADER_SIZE and header[:4] == b"RIFF" and header[8:12] == b"WAVE"


def _read_authoritative_riff_end(wav_file, file_size: int, logger=None) -> Optional[int]:
    wav_file.seek(4)
    riff_size_data = wav_file.read(4)
    if len(riff_size_data) != 4:
        _log_metadata_issue(logger, "Invalid WAV RIFF structure", ValueError("missing RIFF size"))
        return None

    riff_end = struct.unpack("<I", riff_size_data)[0] + 8
    if riff_end < RIFF_HEADER_SIZE:
        _log_metadata_issue(logger, "Invalid WAV RIFF structure", ValueError("declared RIFF size is too small"))
        return None
    if riff_end > file_size:
        _log_metadata_issue(logger, "Invalid WAV RIFF structure", ValueError("declared RIFF size exceeds file size"))
        return None
    return riff_end


def _scan_declared_riff_chunks(wav_file, parse_end: int, logger=None, *, read_metadata: bool) -> Optional[dict]:
    last_valid_metadata = None
    wav_file.seek(RIFF_HEADER_SIZE)

    while wav_file.tell() + CHUNK_HEADER_SIZE <= parse_end:
        chunk_header = wav_file.read(CHUNK_HEADER_SIZE)
        if len(chunk_header) < CHUNK_HEADER_SIZE:
            break
        chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
        chunk_data_start = wav_file.tell()
        chunk_data_end = chunk_data_start + chunk_size
        if chunk_data_end > parse_end:
            _log_metadata_issue(
                logger,
                "Invalid WAV chunk structure",
                ValueError("chunk payload extends beyond declared RIFF data"),
            )
            return None
        if chunk_size % 2 and chunk_data_end + 1 > parse_end:
            _log_metadata_issue(
                logger,
                "Invalid WAV chunk structure",
                ValueError("missing padding byte after odd-sized chunk"),
            )
            return None

        if read_metadata and chunk_id == b"LIST" and chunk_size >= 4:
            list_data = wav_file.read(chunk_size)
            valid_metadata = _read_info_comment_metadata(list_data, logger)
            if valid_metadata is not None:
                last_valid_metadata = valid_metadata
        else:
            wav_file.seek(chunk_size, os.SEEK_CUR)

        if chunk_size % 2:
            padding = wav_file.read(1)
            if padding != b"\x00":
                _log_metadata_issue(
                    logger,
                    "Invalid WAV chunk structure",
                    ValueError("nonzero padding byte after odd-sized chunk"),
                )
                return None

    if wav_file.tell() != parse_end:
        _log_metadata_issue(
            logger,
            "Invalid WAV RIFF structure",
            ValueError("unconsumed trailing bytes in declared RIFF data"),
        )
        return None

    return last_valid_metadata if read_metadata else {}


def _read_info_comment_metadata(list_data: bytes, logger=None) -> Optional[dict]:
    if len(list_data) < 4 or list_data[:4] != b"INFO":
        return None

    last_valid_metadata = None
    offset = 4
    while offset + CHUNK_HEADER_SIZE <= len(list_data):
        subchunk_id, subchunk_size = struct.unpack("<4sI", list_data[offset : offset + CHUNK_HEADER_SIZE])
        offset += CHUNK_HEADER_SIZE
        payload_end = offset + subchunk_size
        if payload_end > len(list_data):
            _log_metadata_issue(
                logger,
                "Invalid WAV LIST/INFO structure",
                ValueError("subchunk payload extends beyond LIST data"),
            )
            return None
        if subchunk_size % 2 and payload_end + 1 > len(list_data):
            _log_metadata_issue(
                logger,
                "Invalid WAV LIST/INFO structure",
                ValueError("missing padding byte after odd-sized subchunk"),
            )
            return None
        if subchunk_size % 2 and list_data[payload_end : payload_end + 1] != b"\x00":
            _log_metadata_issue(
                logger,
                "Invalid WAV LIST/INFO structure",
                ValueError("nonzero padding byte after odd-sized subchunk"),
            )
            return None

        if subchunk_id == b"ICMT":
            metadata = _parse_info_comment_payload(list_data[offset:payload_end], logger)
            if metadata is not None:
                last_valid_metadata = metadata

        offset = payload_end + (subchunk_size % 2)

    if offset != len(list_data):
        _log_metadata_issue(
            logger,
            "Invalid WAV LIST/INFO structure",
            ValueError("unconsumed trailing bytes in LIST data"),
        )
        return None

    return last_valid_metadata


def _parse_info_comment_payload(payload: bytes, logger=None) -> Optional[dict]:
    try:
        comment = payload.rstrip(b"\x00").decode("utf-8")
    except UnicodeDecodeError as exc:
        _log_metadata_issue(logger, "Failed to decode WAV calibration metadata", exc)
        return None

    if not comment.startswith("mic_calibration="):
        return None

    try:
        raw_metadata = json.loads(comment[len("mic_calibration=") :])
        normalized = normalize_wav_calibration_metadata(raw_metadata)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        _log_metadata_issue(logger, "Invalid WAV calibration metadata", exc)
        return None
    if normalized is None:
        _log_metadata_issue(logger, "Invalid WAV calibration metadata", ValueError("metadata payload was rejected"))
        return None
    return normalized


def _log_metadata_issue(logger, message: str, exc: Exception) -> None:
    if logger is None:
        return
    log_message = f"{message}: {exc}"
    log_method = getattr(logger, "warning", None) or getattr(logger, "error", None)
    if log_method is not None:
        log_method(log_message)


def _coerce_wav_channel_index(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("wav_channel_index must be a non-negative integer")
    if isinstance(value, Integral):
        index = int(value)
    elif isinstance(value, Real):
        if not math.isfinite(value) or not float(value).is_integer():
            raise ValueError("wav_channel_index must be a non-negative integer")
        index = int(value)
    elif isinstance(value, str):
        if not value.isdecimal():
            raise ValueError("wav_channel_index must be a non-negative integer")
        index = int(value)
    else:
        raise ValueError("wav_channel_index must be a non-negative integer")
    if index < 0:
        raise ValueError("wav_channel_index must be a non-negative integer")
    return index


def _coerce_positive_finite_float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("v2pa_factor must be a finite positive number")
    try:
        factor = float(value)
    except (TypeError, OverflowError) as exc:
        raise ValueError("v2pa_factor must be a finite positive number") from exc
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("v2pa_factor must be a finite positive number")
    return factor


def _coerce_optional_finite_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number
