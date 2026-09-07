import json
import math
import os
import shutil
import struct
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from numbers import Integral, Real
from typing import Any, Optional

from base.ve3668n_wav_metadata import (
    resolve_ve_wav_channel_v2pa_factor, validate_ve_wav_metadata,
)
from consts.ve3668n_consts import VE_BACKEND


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
MAX_BACKEND_HINT_SIZE = 64 * 1024
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
    """Read provenance plus local ownership; retained handles are not IPC data."""
    status: WavCalibrationMetadataReadStatus
    metadata: Optional[dict]
    declared_backend: Optional[str] = None
    handles_released: bool = True
    close_errors: tuple[str, ...] = ()
    retained_handles: tuple[tuple[str, Any], ...] = field(default=(), repr=False, compare=False)
    primary_error: Optional[str] = None


@dataclass
class _MetadataScanState:
    """Per-read provenance survives malformed chunks and file-boundary errors."""
    declared_backend: Optional[str] = None
    invalid_ve: bool = False
    ve_metadata: Optional[dict] = None
    wav_format: Optional[tuple[int, int]] = None  # sample rate, WAV column count
    invalid_format: bool = False
    provenance_only: bool = False

    def select(self, legacy_metadata):
        if self.invalid_ve or self.provenance_only:
            return None
        if self.declared_backend == VE_BACKEND:
            return self.ve_metadata
        return legacy_metadata


@dataclass(frozen=True)
class WavCalibrationMetadataAppendResult:
    """Local ownership handoff, not an IPC payload (may retain open file objects)."""
    appended: bool
    handles_released: bool
    cleanup_paths: tuple[str, ...] = ()
    close_errors: tuple[str, ...] = ()
    retained_handles: tuple[tuple[str, Any], ...] = field(default=(), repr=False, compare=False)
    primary_error: Optional[str] = None


class _AppendFileOwnership:
    def __init__(self):
        self.cleanup_paths = set()
        self.retained_handles = []
        self.close_errors = []
        self.primary_error = None

    def record_error(self, error):
        if self.primary_error is None:
            self.primary_error = str(error)

    @property
    def handles_released(self):
        return not self.retained_handles

    @contextmanager
    def hold(self, handle, *, temporary=False, path=None, use_context=False):
        path = os.fspath(handle.name if path is None else path)
        if temporary:
            self.cleanup_paths.add(path)
        original = None
        entered = False
        try:
            source = handle.__enter__() if use_context else handle
            entered = True
            yield source
        except Exception as exc:
            # File/context-manager boundary: preserve the exception arguments
            # for reader wrappers' __exit__, including arbitrary wrapper faults.
            original = exc
            self.record_error(exc)
            raise
        finally:
            try:
                if use_context and entered:
                    handle.__exit__(type(original) if original is not None else None,
                                    original, original.__traceback__ if original is not None else None)
                else:
                    handle.close()
            except Exception as exc:
                # Actual file-wrapper close boundary: a failed close cannot prove
                # release. Keep references and paths for the caller's retirement
                # policy, and normalize arbitrary wrapper failures for bool callers.
                self.retained_handles.append((path, handle))
                detail = f"WAV metadata file close failed for {path}: {exc}"
                self.close_errors.append(detail)
                self.record_error(detail)
                if original is None:
                    raise OSError(detail) from exc
                # Nested handles must all attempt cleanup without replacing the
                # first processing/close failure already propagating outward.
                original.add_note(detail)


def normalize_wav_calibration_metadata(payload: Any) -> Optional[dict]:
    if isinstance(payload, Mapping) and payload.get("backend") == VE_BACKEND:
        try:
            return validate_ve_wav_metadata(payload)
        except ValueError:
            return None
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
    read_result = isinstance(metadata, WavCalibrationMetadataReadResult)
    payload = metadata.metadata if read_result else metadata
    declared_ve = read_result and metadata.declared_backend == VE_BACKEND
    if declared_ve or (isinstance(payload, Mapping) and payload.get("backend") == VE_BACKEND):
        if read_result and metadata.status is not WavCalibrationMetadataReadStatus.VALID:
            payload = None
        resolution = resolve_ve_wav_channel_v2pa_factor(payload, wav_channel_index)
        if resolution.factor is None:
            raise ValueError(f"VE calibration {resolution.state}: {resolution.diagnostic}")
        return WavCalibrationResolution(resolution.factor, True, True)
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
    for detail in ownership.close_errors:
        _log_metadata_issue(logger, "WAV metadata cleanup failed", detail)
    return WavCalibrationMetadataAppendResult(
        appended=appended,
        handles_released=ownership.handles_released,
        cleanup_paths=tuple(sorted(ownership.cleanup_paths)),
        close_errors=tuple(ownership.close_errors),
        retained_handles=tuple(ownership.retained_handles),
        primary_error=ownership.primary_error,
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
        ownership.record_error(exc)
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
    except Exception as exc:
        # File-operation boundary: wrappers and validation can raise any ordinary
        # exception. Always return the owner ledger, including failed-close handles.
        ownership.record_error(exc)
        _log_metadata_issue(logger, "Failed to append WAV calibration metadata", ownership.primary_error)
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
    """Preserve read status/provenance and transfer any uncertain reader handle."""
    ownership = _AppendFileOwnership()
    result = _inspect_wav_calibration_metadata(path, logger, ownership)
    for detail in ownership.close_errors:
        _log_metadata_issue(logger, "WAV metadata cleanup failed", detail)
    return replace(result, handles_released=ownership.handles_released,
                   close_errors=tuple(ownership.close_errors),
                   retained_handles=tuple(ownership.retained_handles),
                   primary_error=ownership.primary_error)


def _inspect_wav_calibration_metadata(path, logger, ownership):
    scan_state = _MetadataScanState()
    try:
        with ownership.hold(open(path, "rb"), path=path, use_context=True) as wav_file:
            if not _is_riff_wave_file(wav_file):
                _log_metadata_issue(logger, "Unsupported WAV file", "file is not a RIFF/WAVE file")
                return WavCalibrationMetadataReadResult(
                    WavCalibrationMetadataReadStatus.INVALID,
                    None,
                )
            wav_file.seek(0, os.SEEK_END)
            file_size = wav_file.tell()
            riff_end = _read_authoritative_riff_end(
                wav_file, file_size, logger, allow_truncated=True,
            )
            if riff_end is None:
                return WavCalibrationMetadataReadResult(
                    WavCalibrationMetadataReadStatus.INVALID,
                    None,
                )
            # A truncated RIFF is never trustworthy, but its reachable comments
            # may still identify voltage data that must not use legacy factor=1.
            scan_state.provenance_only = riff_end > file_size
            structure_valid, metadata, metadata_comment_seen = _scan_declared_riff_chunks(
                wav_file,
                min(riff_end, file_size),
                logger,
                read_metadata=True,
                scan_state=scan_state,
            )
            if scan_state.provenance_only or not structure_valid or scan_state.invalid_ve:
                return WavCalibrationMetadataReadResult(
                    WavCalibrationMetadataReadStatus.INVALID,
                    None,
                    scan_state.declared_backend,
                )
            if metadata is not None:
                return WavCalibrationMetadataReadResult(
                    WavCalibrationMetadataReadStatus.VALID,
                    metadata,
                    scan_state.declared_backend,
                )
            if metadata_comment_seen:
                return WavCalibrationMetadataReadResult(
                    WavCalibrationMetadataReadStatus.INVALID,
                    None,
                    scan_state.declared_backend,
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
    except Exception as exc:
        # Reader boundary: an ordinary read/parse failure must not bypass the
        # ownership handoff or discard the backend already observed in this scan.
        ownership.record_error(exc)
        _log_metadata_issue(logger, "Failed to read WAV calibration metadata", exc)
        return WavCalibrationMetadataReadResult(
            WavCalibrationMetadataReadStatus.INVALID,
            None,
            scan_state.declared_backend,
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


def _read_authoritative_riff_end(wav_file, file_size: int, logger=None, *,
                                 allow_truncated: bool = False) -> Optional[int]:
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
        if not allow_truncated:
            return None
    return riff_end


def _scan_declared_riff_chunks(wav_file, parse_end: int, logger=None, *, read_metadata: bool,
                               scan_state=None):
    # Keep the historical triple-return contract, including private callers.
    if scan_state is None:
        scan_state = _MetadataScanState()
    last_valid_metadata = None
    metadata_comment_seen = False
    wav_file.seek(RIFF_HEADER_SIZE)

    while wav_file.tell() + CHUNK_HEADER_SIZE <= parse_end:
        chunk_id, chunk_size = struct.unpack("<4sI", wav_file.read(CHUNK_HEADER_SIZE))
        payload_start = wav_file.tell()
        payload_end = payload_start + chunk_size
        padded_end = payload_end + (chunk_size % 2)
        if (padded_end > parse_end and read_metadata and chunk_id == LIST_ID
                and min(payload_end, parse_end) - payload_start >= 4):
            # Even a damaged LIST can declare VE, but never inspect beyond RIFF.
            _, _, comment_seen = _scan_list_subchunks(
                wav_file, min(payload_end, parse_end), wav_file.read(4), logger,
                read_metadata=True, scan_state=scan_state,
            )
            metadata_comment_seen |= comment_seen
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
                scan_state=scan_state,
            )
            metadata_comment_seen = metadata_comment_seen or list_metadata_comment_seen
            if not valid_list:
                return False, None, metadata_comment_seen
            if metadata is not None:
                last_valid_metadata = metadata
        elif chunk_id == b"fmt " and read_metadata and not scan_state.provenance_only:
            if chunk_size < 16 or scan_state.wav_format is not None:
                scan_state.invalid_format = True
                wav_file.seek(chunk_size, os.SEEK_CUR)
            else:
                # The common prefix supports PCM, IEEE FLOAT and extensible WAV;
                # no decoder handle and no audio/data scan are needed.
                _, channels, sample_rate = struct.unpack("<HHI", wav_file.read(8))
                scan_state.wav_format = (sample_rate, channels)
                wav_file.seek(chunk_size - 8, os.SEEK_CUR)
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
    if scan_state.ve_metadata is not None:
        expected_format = (
            scan_state.ve_metadata["acquisition"]["sample_rate"],
            len(scan_state.ve_metadata["recorded_channels"]),
        )
        if scan_state.invalid_format or scan_state.wav_format != expected_format:
            scan_state.invalid_ve = True
            _log_metadata_issue(logger, "Invalid VE WAV header",
                                "recording sample_rate/channel count must match the WAV fmt header")
    return True, scan_state.select(last_valid_metadata), metadata_comment_seen


def _scan_list_subchunks(
    wav_file,
    list_end: int,
    list_type: bytes,
    logger=None,
    *,
    read_metadata: bool,
    scan_state=None,
):
    if scan_state is None:
        scan_state = _MetadataScanState()
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
        should_read_comment = read_metadata and list_type == INFO_ID and subchunk_id == ICMT_ID
        if padded_end > list_end and should_read_comment:
            # Probe only bytes bounded by both the containing LIST and RIFF.
            available = min(subchunk_size, list_end - wav_file.tell(), MAX_BACKEND_HINT_SIZE)
            probe = _read_metadata_bytes(wav_file, available)
            prefix = CALIBRATION_COMMENT_PREFIX.encode("ascii")
            if probe.startswith(prefix):
                metadata_comment_seen = True
                if _has_ve_backend_hint(probe[len(prefix):]):
                    scan_state.declared_backend = VE_BACKEND
                    scan_state.invalid_ve = True
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
            elif scan_state.provenance_only or subchunk_size > MAX_CALIBRATION_COMMENT_SIZE:
                metadata_comment_seen = True
                probe_size = min(remaining_size, MAX_BACKEND_HINT_SIZE - prefix_size)
                if _has_ve_backend_hint(wav_file.read(probe_size)):
                    scan_state.declared_backend = VE_BACKEND
                    scan_state.invalid_ve = True
                if subchunk_size > MAX_CALIBRATION_COMMENT_SIZE:
                    _log_metadata_issue(
                        logger,
                        "Invalid WAV calibration metadata",
                        f"ICMT comment exceeds {MAX_CALIBRATION_COMMENT_SIZE}-byte read limit",
                    )
                wav_file.seek(remaining_size - probe_size, os.SEEK_CUR)
            else:
                probe_size = min(remaining_size, MAX_BACKEND_HINT_SIZE - prefix_size)
                body_probe = _read_metadata_bytes(wav_file, probe_size)
                if _has_ve_backend_hint(body_probe):
                    # Retain provenance even if the remaining bounded read fails.
                    scan_state.declared_backend = VE_BACKEND
                metadata, comment_seen = _parse_info_comment_payload(
                    prefix_probe + body_probe + _read_metadata_bytes(
                        wav_file, remaining_size - probe_size,
                    ),
                    logger,
                    scan_state=scan_state,
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
    return True, scan_state.select(last_valid_metadata), metadata_comment_seen


def _read_metadata_bytes(wav_file, size):
    """Only called for size-limited ICMT bodies; never allocate from audio sizes."""
    parts = []
    while size:
        part = wav_file.read(min(65536, size))
        if not part:
            raise OSError("WAV metadata ended before its declared boundary")
        parts.append(part)
        size -= len(part)
    return b"".join(parts)


def _has_ve_backend_hint(body):
    """Recognize a top-level JSON backend token, even after malformed UTF-8.

    The caller supplies only a bounded prefixed ICMT body. Tokenizing strings
    keeps nested keys and quoted examples from becoming backend declarations.
    Escaped JSON keys/values are allowed; no damaged text is used as metadata.
    """
    depth, previous, backend_key, expect_value = 0, None, False, False
    cursor = 0
    while cursor < len(body):
        start = cursor
        byte = body[cursor]
        cursor += 1
        if byte == ord('"'):
            # Consume each string once. Restarting at its escaped quotes after
            # a missing closing quote makes even a bounded probe quadratic.
            while cursor < len(body):
                byte = body[cursor]
                cursor += 1
                if byte == ord("\\"):
                    cursor += 1
                elif byte == ord('"'):
                    break
            else:
                return False
            token = body[start:cursor]
        elif byte in b"{}[]:,":
            token = body[start:cursor]
        else:
            continue
        if previous is None and token != b"{":
            return False
        if depth == 1 and token.startswith(b'"'):
            try:
                value = json.loads(token)
            except (ValueError, UnicodeDecodeError):
                value = None
            if expect_value and value == VE_BACKEND:
                return True
            backend_key = previous in (b"{", b",") and value == "backend"
            expect_value = False
        elif depth == 1 and token == b":":
            expect_value = backend_key
        else:
            backend_key = expect_value = False
        if token in (b"{", b"["):
            depth += 1
        elif token in (b"}", b"]"):
            depth -= 1
            if depth <= 0:
                return False
        previous = token
    return False


def _unique_ve_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate VE metadata field: {key!r}")
        result[key] = value
    return result


def _parse_info_comment_payload(payload: bytes, logger=None, *, scan_state=None):
    if scan_state is None:
        scan_state = _MetadataScanState()
    metadata_prefix = CALIBRATION_COMMENT_PREFIX.encode("ascii")
    if not payload.startswith(metadata_prefix):
        return None, False
    body = payload[len(metadata_prefix):].rstrip(b"\x00")
    declared_ve = _has_ve_backend_hint(body)
    if declared_ve:
        scan_state.declared_backend = VE_BACKEND
    try:
        serialized_metadata = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        scan_state.invalid_ve |= declared_ve
        _log_metadata_issue(logger, "Failed to decode WAV calibration metadata", exc)
        return None, True

    try:
        raw_metadata = json.loads(serialized_metadata)
        if declared_ve:
            # Strict only for identified VE: old duplicate-key behavior remains.
            raw_metadata = json.loads(serialized_metadata, object_pairs_hook=_unique_ve_json_object)
    except (ValueError, TypeError, RecursionError) as exc:
        scan_state.invalid_ve |= declared_ve
        _log_metadata_issue(logger, "Invalid WAV calibration metadata", exc)
        return None, True
    normalized = normalize_wav_calibration_metadata(raw_metadata)
    if normalized is None:
        scan_state.invalid_ve |= declared_ve
        _log_metadata_issue(logger, "Invalid WAV calibration metadata", "metadata payload was rejected")
    elif declared_ve:
        previous = scan_state.ve_metadata
        if previous is not None and (
            previous["acquisition"]["sample_rate"] != normalized["acquisition"]["sample_rate"]
            or len(previous["recorded_channels"]) != len(normalized["recorded_channels"])
        ):
            scan_state.invalid_ve = True
            _log_metadata_issue(logger, "Invalid VE WAV metadata",
                                "VE comments disagree about the recording sample_rate/channel count")
        scan_state.ve_metadata = normalized
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
    except Exception as exc:
        # Temporary-reader boundary shares the append owner. Preserve its first
        # failure and uncertain handles before returning the legacy bool result.
        ownership.record_error(exc)
        _log_metadata_issue(logger, "Failed to validate temporary WAV", exc)
        return False


def _log_metadata_issue(logger, message: str, detail) -> None:
    if logger is None:
        return
    log_method = getattr(logger, "warning", None) or getattr(logger, "error", None)
    if log_method is not None:
        log_method(f"{message}: {detail}")
