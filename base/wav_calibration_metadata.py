import json
import math
import os
import shutil
import struct
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral, Real
from typing import Any, Optional


RIFF_ID = b"RIFF"
WAVE_ID = b"WAVE"
LIST_ID = b"LIST"
INFO_ID = b"INFO"
ICMT_ID = b"ICMT"
CALIBRATION_COMMENT_PREFIX = "mic_calibration="
RIFF_HEADER_SIZE = 12
CHUNK_HEADER_SIZE = 8
MAX_RIFF_SIZE = 0xFFFFFFFF
MAX_CALIBRATION_COMMENT_SIZE = 1024 * 1024
PHYSICAL_INPUT_CHANNEL_FIELD = "physical_input_channel"


@dataclass(frozen=True)
class WavCalibrationResolution:
    factor: float
    has_valid_metadata: bool
    used_file_metadata: bool


class WavCalibrationMetadataReadStatus(Enum):
    ABSENT = "absent"
    VALID = "valid"
    INVALID = "invalid"


@dataclass(frozen=True)
class WavCalibrationMetadataReadResult:
    status: WavCalibrationMetadataReadStatus
    metadata: Optional[dict]


@dataclass(frozen=True)
class WavCalibrationMetadataAppendResult:
    """Local ownership handoff, not an IPC payload (may retain open file objects)."""
    appended: bool
    handles_released: bool
    cleanup_paths: tuple[str, ...] = ()
    close_errors: tuple[str, ...] = ()
    retained_handles: tuple[tuple[str, Any], ...] = field(default=(), repr=False, compare=False)


class _AppendFileOwnership:
    def __init__(self):
        self.cleanup_paths = set()
        self.retained_handles = []
        self.close_errors = []

    @property
    def handles_released(self):
        return not self.retained_handles

    @contextmanager
    def hold(self, handle, *, temporary=False):
        path = os.fspath(handle.name)
        if temporary:
            self.cleanup_paths.add(path)
        try:
            yield handle
        finally:
            try:
                handle.close()
            except Exception as exc:
                # Actual file-wrapper close boundary: a failed close cannot prove
                # release. Keep references and paths for the caller's retirement
                # policy, and normalize arbitrary wrapper failures for bool callers.
                self.retained_handles.append((path, handle))
                detail = f"WAV metadata file close failed for {path}: {exc}"
                self.close_errors.append(detail)
                raise OSError(detail) from exc


def normalize_wav_calibration_metadata(payload: Any) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None

    recorded_channels = payload.get("recorded_channels")
    if not isinstance(recorded_channels, list) or not recorded_channels:
        return None

    normalized_channels = []
    seen_indices = set()
    seen_physical_channels = set()
    channels_with_physical_identity = 0
    for channel in recorded_channels:
        normalized = _normalize_channel(channel)
        if normalized is None or normalized["wav_channel_index"] in seen_indices:
            return None
        seen_indices.add(normalized["wav_channel_index"])
        if PHYSICAL_INPUT_CHANNEL_FIELD in normalized:
            physical_channel = normalized[PHYSICAL_INPUT_CHANNEL_FIELD]
            if physical_channel in seen_physical_channels:
                return None
            seen_physical_channels.add(physical_channel)
            channels_with_physical_identity += 1
        normalized_channels.append(normalized)

    if channels_with_physical_identity not in (0, len(normalized_channels)):
        return None

    return {"recorded_channels": normalized_channels}


def resolve_wav_channel_v2pa_factor(
    metadata: Any,
    wav_channel_index: int,
) -> WavCalibrationResolution:
    normalized = normalize_wav_calibration_metadata(metadata)
    try:
        selected_index = _coerce_wav_channel_index(wav_channel_index)
    except (TypeError, ValueError, OverflowError):
        return WavCalibrationResolution(1.0, False, False)

    if normalized is None:
        return WavCalibrationResolution(1.0, False, False)

    for channel in normalized["recorded_channels"]:
        if channel["wav_channel_index"] != selected_index:
            continue
        if channel["calibrated"]:
            return WavCalibrationResolution(channel["v2pa_factor"], True, True)
        break
    return WavCalibrationResolution(1.0, True, False)


def append_wav_calibration_metadata(path, metadata, logger=None) -> bool:
    """Legacy warning/bool API; recording capture uses the ownership result below."""
    return append_wav_calibration_metadata_result(path, metadata, logger).appended


def append_wav_calibration_metadata_result(path, metadata, logger=None) -> WavCalibrationMetadataAppendResult:
    ownership = _AppendFileOwnership()
    appended = _append_wav_calibration_metadata(path, metadata, logger, ownership)
    return WavCalibrationMetadataAppendResult(
        appended=appended,
        handles_released=ownership.handles_released,
        cleanup_paths=tuple(sorted(ownership.cleanup_paths)),
        close_errors=tuple(ownership.close_errors),
        retained_handles=tuple(ownership.retained_handles),
    )


def _append_wav_calibration_metadata(path, metadata, logger, ownership) -> bool:
    normalized = normalize_wav_calibration_metadata(metadata)
    if normalized is None:
        _log_metadata_issue(logger, "Invalid WAV calibration metadata", "metadata payload was rejected")
        return False

    try:
        comment = CALIBRATION_COMMENT_PREFIX + json.dumps(
            normalized,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        comment_payload = comment.encode("utf-8") + b"\x00"
        if len(comment_payload) > MAX_CALIBRATION_COMMENT_SIZE:
            _log_metadata_issue(
                logger,
                "Invalid WAV calibration metadata",
                f"generated ICMT comment exceeds {MAX_CALIBRATION_COMMENT_SIZE}-byte read limit",
            )
            return False
        list_chunk = _build_chunk(LIST_ID, INFO_ID + _build_chunk(ICMT_ID, comment_payload))
    except (MemoryError, OverflowError, TypeError, ValueError, struct.error) as exc:
        _log_metadata_issue(logger, "Failed to serialize WAV calibration metadata", exc)
        return False

    temp_path = None
    try:
        target_path = os.path.abspath(os.fspath(path))
        with ownership.hold(open(target_path, "rb")) as wav_file:
            if not _is_riff_wave_file(wav_file):
                _log_metadata_issue(logger, "Unsupported WAV file", "file is not a RIFF/WAVE file")
                return False

            wav_file.seek(0, os.SEEK_END)
            file_size = wav_file.tell()
            riff_end = _read_authoritative_riff_end(wav_file, file_size, logger)
            if riff_end is None:
                return False
            structure_valid, _, _ = _scan_declared_riff_chunks(
                wav_file,
                riff_end,
                logger,
                read_metadata=False,
            )
            if not structure_valid:
                return False

            new_riff_size = riff_end + len(list_chunk) - 8
            if new_riff_size > MAX_RIFF_SIZE:
                _log_metadata_issue(
                    logger,
                    "Failed to append WAV calibration metadata",
                    "RIFF size exceeds 32-bit limit",
                )
                return False

            with ownership.hold(tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=os.path.dirname(target_path),
                prefix=f".{os.path.basename(target_path)}.",
                suffix=".tmp",
                delete=False,
            ), temporary=True) as temp_file:
                temp_path = temp_file.name
                wav_file.seek(0)
                _copy_stream_exact(wav_file, temp_file, riff_end)
                _write_all(temp_file, list_chunk)
                temp_file.seek(4)
                _write_all(temp_file, struct.pack("<I", new_riff_size))
                temp_file.flush()
                os.fsync(temp_file.fileno())

        if not _validate_appended_wav(
            temp_path,
            normalized,
            logger,
            ownership=ownership,
        ):
            raise OSError("temporary WAV validation failed")

        shutil.copystat(target_path, temp_path)
        os.replace(temp_path, target_path)
        ownership.cleanup_paths.discard(temp_path)
        temp_path = None
        return True
    except (OSError, OverflowError, struct.error) as exc:
        _log_metadata_issue(logger, "Failed to append WAV calibration metadata", exc)
        return False
    finally:
        if temp_path is not None and ownership.handles_released:
            try:
                os.unlink(temp_path)
                ownership.cleanup_paths.discard(temp_path)
            except OSError as exc:
                _log_metadata_issue(
                    logger,
                    "Failed to clean temporary WAV metadata file",
                    exc,
                )


def inspect_wav_calibration_metadata(
    path,
    logger=None,
) -> WavCalibrationMetadataReadResult:
    try:
        with open(path, "rb") as wav_file:
            if not _is_riff_wave_file(wav_file):
                _log_metadata_issue(logger, "Unsupported WAV file", "file is not a RIFF/WAVE file")
                return WavCalibrationMetadataReadResult(
                    WavCalibrationMetadataReadStatus.INVALID,
                    None,
                )
            wav_file.seek(0, os.SEEK_END)
            file_size = wav_file.tell()
            riff_end = _read_authoritative_riff_end(wav_file, file_size, logger)
            if riff_end is None:
                return WavCalibrationMetadataReadResult(
                    WavCalibrationMetadataReadStatus.INVALID,
                    None,
                )
            structure_valid, metadata, metadata_comment_seen = _scan_declared_riff_chunks(
                wav_file,
                riff_end,
                logger,
                read_metadata=True,
            )
            if not structure_valid:
                return WavCalibrationMetadataReadResult(
                    WavCalibrationMetadataReadStatus.INVALID,
                    None,
                )
            if metadata is not None:
                return WavCalibrationMetadataReadResult(
                    WavCalibrationMetadataReadStatus.VALID,
                    metadata,
                )
            if metadata_comment_seen:
                return WavCalibrationMetadataReadResult(
                    WavCalibrationMetadataReadStatus.INVALID,
                    None,
                )
            _log_metadata_issue(
                logger,
                "No valid WAV calibration metadata",
                "calibration comment was not found",
            )
            return WavCalibrationMetadataReadResult(
                WavCalibrationMetadataReadStatus.ABSENT,
                None,
            )
    except (OSError, OverflowError, struct.error) as exc:
        _log_metadata_issue(logger, "Failed to read WAV calibration metadata", exc)
        return WavCalibrationMetadataReadResult(
            WavCalibrationMetadataReadStatus.INVALID,
            None,
        )


def read_wav_calibration_metadata(path, logger=None) -> Optional[dict]:
    result = inspect_wav_calibration_metadata(path, logger=logger)
    if result.status is WavCalibrationMetadataReadStatus.VALID:
        return result.metadata
    return None


def _normalize_channel(channel: Any) -> Optional[dict]:
    if not isinstance(channel, dict):
        return None
    try:
        wav_channel_index = _coerce_wav_channel_index(channel.get("wav_channel_index"))
        calibrated = bool(channel.get("calibrated", False))
        v2pa_factor = (
            _coerce_positive_finite_float(channel.get("v2pa_factor"))
            if calibrated
            else None
        )
        physical_input_channel = (
            _coerce_physical_input_channel(channel[PHYSICAL_INPUT_CHANNEL_FIELD])
            if PHYSICAL_INPUT_CHANNEL_FIELD in channel
            else None
        )
    except (TypeError, ValueError, OverflowError):
        return None

    normalized = {
        "wav_channel_index": wav_channel_index,
        "v2pa_factor": v2pa_factor,
        "standard_spl": _coerce_optional_finite_float(channel.get("standard_spl")),
        "calibrated": calibrated,
    }
    if physical_input_channel is not None:
        normalized[PHYSICAL_INPUT_CHANNEL_FIELD] = physical_input_channel
    return normalized


def _coerce_physical_input_channel(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("physical_input_channel must be a non-negative integer")
    channel = int(value)
    if channel < 0:
        raise ValueError("physical_input_channel must be a non-negative integer")
    return channel


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
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("v2pa_factor must be a finite positive number") from exc
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("v2pa_factor must be a finite positive number")
    return factor


def _coerce_optional_finite_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _build_chunk(chunk_id: bytes, payload: bytes) -> bytes:
    chunk = chunk_id + struct.pack("<I", len(payload)) + payload
    return chunk + (b"\x00" if len(payload) % 2 else b"")


def _is_riff_wave_file(wav_file) -> bool:
    wav_file.seek(0)
    header = wav_file.read(RIFF_HEADER_SIZE)
    return (
        len(header) == RIFF_HEADER_SIZE
        and header[:4] == RIFF_ID
        and header[8:12] == WAVE_ID
    )


def _read_authoritative_riff_end(wav_file, file_size: int, logger=None) -> Optional[int]:
    wav_file.seek(4)
    size_bytes = wav_file.read(4)
    if len(size_bytes) != 4:
        _log_metadata_issue(logger, "Invalid WAV RIFF structure", "missing RIFF size")
        return None
    riff_end = struct.unpack("<I", size_bytes)[0] + 8
    if riff_end < RIFF_HEADER_SIZE:
        _log_metadata_issue(logger, "Invalid WAV RIFF structure", "declared RIFF size is too small")
        return None
    if riff_end > file_size:
        _log_metadata_issue(logger, "Invalid WAV RIFF structure", "declared RIFF size exceeds file size")
        return None
    return riff_end


def _scan_declared_riff_chunks(wav_file, parse_end: int, logger=None, *, read_metadata: bool):
    last_valid_metadata = None
    metadata_comment_seen = False
    wav_file.seek(RIFF_HEADER_SIZE)

    while wav_file.tell() + CHUNK_HEADER_SIZE <= parse_end:
        chunk_id, chunk_size = struct.unpack("<4sI", wav_file.read(CHUNK_HEADER_SIZE))
        payload_start = wav_file.tell()
        payload_end = payload_start + chunk_size
        padded_end = payload_end + (chunk_size % 2)
        if payload_end > parse_end:
            _log_metadata_issue(
                logger,
                "Invalid WAV chunk structure",
                "chunk payload extends beyond declared RIFF data",
            )
            return False, None, metadata_comment_seen
        if padded_end > parse_end:
            _log_metadata_issue(
                logger,
                "Invalid WAV chunk structure",
                "missing padding byte after odd-sized chunk",
            )
            return False, None, metadata_comment_seen

        if chunk_id == LIST_ID:
            if chunk_size < 4:
                _log_metadata_issue(
                    logger,
                    "Invalid WAV LIST structure",
                    "LIST chunk is missing its list type",
                )
                return False, None, metadata_comment_seen
            list_type = wav_file.read(4)
            valid_list, metadata, list_metadata_comment_seen = _scan_list_subchunks(
                wav_file,
                payload_end,
                list_type,
                logger,
                read_metadata=read_metadata,
            )
            if not valid_list:
                return False, None, metadata_comment_seen
            metadata_comment_seen = metadata_comment_seen or list_metadata_comment_seen
            if metadata is not None:
                last_valid_metadata = metadata
        else:
            wav_file.seek(chunk_size, os.SEEK_CUR)

        if chunk_size % 2 and wav_file.read(1) != b"\x00":
            _log_metadata_issue(
                logger,
                "Invalid WAV chunk structure",
                "nonzero padding byte after odd-sized chunk",
            )
            return False, None, metadata_comment_seen

    if wav_file.tell() != parse_end:
        _log_metadata_issue(
            logger,
            "Invalid WAV RIFF structure",
            "unconsumed trailing bytes in declared RIFF data",
        )
        return False, None, metadata_comment_seen
    return True, last_valid_metadata, metadata_comment_seen


def _scan_list_subchunks(
    wav_file,
    list_end: int,
    list_type: bytes,
    logger=None,
    *,
    read_metadata: bool,
):
    last_valid_metadata = None
    metadata_comment_seen = False
    structure_name = "WAV LIST/INFO" if list_type == INFO_ID else "WAV LIST"
    while wav_file.tell() + CHUNK_HEADER_SIZE <= list_end:
        subchunk_id, subchunk_size = struct.unpack(
            "<4sI",
            wav_file.read(CHUNK_HEADER_SIZE),
        )
        payload_end = wav_file.tell() + subchunk_size
        padded_end = payload_end + (subchunk_size % 2)
        if payload_end > list_end:
            _log_metadata_issue(
                logger,
                f"Invalid {structure_name} structure",
                "subchunk payload extends beyond LIST data",
            )
            return False, None, metadata_comment_seen
        if padded_end > list_end:
            _log_metadata_issue(
                logger,
                f"Invalid {structure_name} structure",
                "missing padding byte after odd-sized subchunk",
            )
            return False, None, metadata_comment_seen

        should_read_comment = read_metadata and list_type == INFO_ID and subchunk_id == ICMT_ID
        if should_read_comment:
            metadata_prefix = CALIBRATION_COMMENT_PREFIX.encode("ascii")
            prefix_size = min(subchunk_size, len(metadata_prefix))
            prefix_probe = wav_file.read(prefix_size)
            remaining_size = subchunk_size - prefix_size
            is_metadata_comment = (
                prefix_size == len(metadata_prefix)
                and prefix_probe == metadata_prefix
            )
            if not is_metadata_comment:
                wav_file.seek(remaining_size, os.SEEK_CUR)
            elif subchunk_size > MAX_CALIBRATION_COMMENT_SIZE:
                metadata_comment_seen = True
                _log_metadata_issue(
                    logger,
                    "Invalid WAV calibration metadata",
                    f"ICMT comment exceeds {MAX_CALIBRATION_COMMENT_SIZE}-byte read limit",
                )
                wav_file.seek(remaining_size, os.SEEK_CUR)
            else:
                metadata, comment_seen = _parse_info_comment_payload(
                    prefix_probe + wav_file.read(remaining_size),
                    logger,
                )
                metadata_comment_seen = metadata_comment_seen or comment_seen
                if metadata is not None:
                    last_valid_metadata = metadata
        else:
            wav_file.seek(subchunk_size, os.SEEK_CUR)

        if subchunk_size % 2 and wav_file.read(1) != b"\x00":
            _log_metadata_issue(
                logger,
                f"Invalid {structure_name} structure",
                "nonzero padding byte after odd-sized subchunk",
            )
            return False, None, metadata_comment_seen

    if wav_file.tell() != list_end:
        _log_metadata_issue(
            logger,
            f"Invalid {structure_name} structure",
            "unconsumed trailing bytes in LIST data",
        )
        return False, None, metadata_comment_seen
    return True, last_valid_metadata, metadata_comment_seen


def _parse_info_comment_payload(payload: bytes, logger=None):
    metadata_prefix = CALIBRATION_COMMENT_PREFIX.encode("ascii")
    if not payload.startswith(metadata_prefix):
        return None, False
    try:
        serialized_metadata = (
            payload[len(metadata_prefix) :].rstrip(b"\x00").decode("utf-8")
        )
    except UnicodeDecodeError as exc:
        _log_metadata_issue(logger, "Failed to decode WAV calibration metadata", exc)
        return None, True

    try:
        raw_metadata = json.loads(serialized_metadata)
    except (json.JSONDecodeError, TypeError) as exc:
        _log_metadata_issue(logger, "Invalid WAV calibration metadata", exc)
        return None, True
    normalized = normalize_wav_calibration_metadata(raw_metadata)
    if normalized is None:
        _log_metadata_issue(logger, "Invalid WAV calibration metadata", "metadata payload was rejected")
    return normalized, True


def _copy_stream_exact(source, destination, byte_count: int) -> None:
    remaining = byte_count
    while remaining:
        data = source.read(min(64 * 1024, remaining))
        if not data:
            raise OSError("source WAV ended before its declared RIFF boundary")
        _write_all(destination, data)
        remaining -= len(data)


def _write_all(destination, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = destination.write(view)
        if not written:
            raise OSError("temporary WAV write did not make progress")
        view = view[written:]


def _validate_appended_wav(path, expected_metadata, logger=None, *, ownership=None) -> bool:
    if ownership is None:
        ownership = _AppendFileOwnership()
    try:
        with ownership.hold(open(path, "rb")) as wav_file:
            if not _is_riff_wave_file(wav_file):
                _log_metadata_issue(
                    logger,
                    "Invalid temporary WAV",
                    "file is not a RIFF/WAVE file",
                )
                return False
            wav_file.seek(0, os.SEEK_END)
            file_size = wav_file.tell()
            riff_end = _read_authoritative_riff_end(
                wav_file,
                file_size,
                logger,
            )
            if riff_end is None or riff_end != file_size:
                _log_metadata_issue(
                    logger,
                    "Invalid temporary WAV",
                    "declared RIFF size does not match temporary file size",
                )
                return False
            structure_valid, actual_metadata, _ = _scan_declared_riff_chunks(
                wav_file,
                riff_end,
                logger,
                read_metadata=True,
            )
            if not structure_valid or actual_metadata != expected_metadata:
                _log_metadata_issue(
                    logger,
                    "Invalid temporary WAV",
                    "appended calibration metadata did not validate",
                )
                return False
            return True
    except (OSError, OverflowError, struct.error) as exc:
        _log_metadata_issue(logger, "Failed to validate temporary WAV", exc)
        return False


def _log_metadata_issue(logger, message: str, detail) -> None:
    if logger is None:
        return
    log_method = getattr(logger, "warning", None) or getattr(logger, "error", None)
    if log_method is not None:
        log_method(f"{message}: {detail}")
