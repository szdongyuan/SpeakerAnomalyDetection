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
    from collections.abc import Mapping
    if isinstance(payload, Mapping) and payload.get("backend") == "vkinging":
        try:
            return normalize_vkinging_metadata(payload)
        except (ValueError, TypeError, OverflowError, RecursionError):
            return None
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
    from collections.abc import Mapping
    if isinstance(metadata, Mapping) and metadata.get("backend") == "vkinging":
        normalized = normalize_vkinging_metadata(metadata)
        selected_index = _coerce_wav_channel_index(wav_channel_index)
        for channel in normalized["recorded_channels"]:
            if channel["wav_channel_index"] == selected_index and channel["calibrated"]:
                return WavCalibrationResolution(channel["v2pa_factor"], True, True)
        raise ValueError("vkinging WAV channel is absent or uncalibrated")
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
    from collections.abc import Mapping
    if isinstance(metadata, Mapping) and metadata.get("backend") == "vkinging":
        # This historical API rewrites in place and cannot establish the
        # independent before-data marker. Use VkingingVoltageWriter instead.
        return False
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


def _read_legacy_wav_calibration_metadata(path, logger=None) -> Optional[dict]:
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

@dataclass(frozen=True)
class WavCalibrationInspection:
    source: str
    status: str
    metadata: Optional[dict] = None
    reason: Optional[str] = None
    unit: Optional[str] = None
    structure_valid: bool = True


class UnsupportedVkingingSchema(ValueError):
    pass


def json_snapshot(value):
    """Own JSON data, including the mapping proxies/tuples in capture requests."""
    from collections.abc import Mapping
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError('metadata keys must be strings')
        return {key: json_snapshot(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_snapshot(child) for child in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError('metadata must contain JSON values')


def validate_vkinging_conditions(value):
    from collections.abc import Mapping
    from consts.vkinging_consts import VKINGING_RANGE_LIMITS
    if not isinstance(value, Mapping) or set(value) != {'input_mode', 'unit', 'range_min', 'range_max'}:
        raise ValueError('calibration conditions require exactly input_mode, unit, range_min, range_max')
    if value['input_mode'] != 'IEPE' or value['unit'] != 'V':
        raise ValueError('calibration conditions require IEPE/V')
    lo, hi = value['range_min'], value['range_max']
    if (type(lo) not in (int, float) or type(hi) not in (int, float)
            or hi not in VKINGING_RANGE_LIMITS or lo != -hi):
        raise ValueError('unsupported voltage range')
    return dict(value)


def normalize_vkinging_metadata(payload):
    from base.vkinging_input import validate_bit_depth, validate_physical_channels, validate_sample_rate
    from consts.vkinging_consts import VKINGING_WAV_SCHEMA_VERSION
    data = json_snapshot(payload)
    if not isinstance(data, dict) or data.get('backend') != 'vkinging':
        raise ValueError('metadata backend must be vkinging')
    version = data.get('schema_version')
    if type(version) is not int or version != VKINGING_WAV_SCHEMA_VERSION:
        raise UnsupportedVkingingSchema(f'unsupported vkinging schema_version {version!r}')
    acquisition = data.get('acquisition')
    if not isinstance(acquisition, dict):
        raise ValueError('acquisition must be an object')
    hardware_id = acquisition.get('hardware_id')
    if not isinstance(hardware_id, str) or not hardware_id.strip():
        raise ValueError('acquisition.hardware_id must be nonempty')
    if acquisition.get('model') is not None and not isinstance(acquisition['model'], str):
        raise ValueError('acquisition.model must be text or null')
    validate_sample_rate(acquisition.get('sample_rate'))
    validate_bit_depth(acquisition.get('bit_depth'))
    validate_vkinging_conditions({key: acquisition.get(key) for key in
                                  ('input_mode', 'unit', 'range_min', 'range_max')})
    channels = data.get('recorded_channels')
    if not isinstance(channels, list) or not channels:
        raise ValueError('recorded_channels must be nonempty')
    physical = []
    for index, channel in enumerate(channels):
        if not isinstance(channel, dict) or type(channel.get('wav_channel_index')) is not int or channel['wav_channel_index'] != index:
            raise ValueError('WAV channel indices must be compact and ordered')
        physical.append(channel.get('physical_channel_index'))
        if type(channel.get('calibrated')) is not bool:
            raise ValueError('calibrated must be boolean')
        if channel['calibrated']:
            if type(channel.get('v2pa_factor')) not in (int, float):
                raise ValueError('v2pa_factor must be a finite positive JSON number')
            channel['v2pa_factor'] = _coerce_positive_finite_float(channel.get('v2pa_factor'))
            spl = channel.get('standard_spl')
            if spl is not None and (type(spl) not in (float, int) or not math.isfinite(spl)):
                raise ValueError('standard_spl must be finite or null')
            channel['calibration_conditions'] = validate_vkinging_conditions(channel.get('calibration_conditions'))
        elif any(channel.get(key) is not None for key in ('v2pa_factor', 'standard_spl', 'calibration_conditions')):
            raise ValueError('uncalibrated channel cannot declare a factor or conditions')
    validate_physical_channels(physical)
    return data


def scan_riff_layout(wav_file):
    """Return bounded chunk descriptors, never audio payloads; require exact RIFF."""
    wav_file.seek(0, os.SEEK_END)
    file_size = wav_file.tell()
    wav_file.seek(0)
    header = wav_file.read(12)
    if len(header) != 12 or header[:4] != b'RIFF' or header[8:] != b'WAVE':
        raise ValueError('unsupported RIFF/WAVE header')
    if struct.unpack_from('<I', header, 4)[0] + 8 != file_size:
        raise ValueError('declared RIFF size does not match file size')
    chunks = []
    offset = 12
    while offset < file_size:
        wav_file.seek(offset)
        chunk = wav_file.read(8)
        if len(chunk) != 8:
            raise ValueError('truncated RIFF chunk header')
        name, size = struct.unpack('<4sI', chunk)
        end = offset + 8 + size
        if end + size % 2 > file_size:
            raise ValueError('RIFF chunk extends beyond file')
        if size % 2:
            wav_file.seek(end)
            if wav_file.read(1) != b'\0':
                raise ValueError('nonzero RIFF padding')
        chunks.append((name, offset + 8, size))
        offset = end + size % 2
    return chunks


def _provenance_evidence(wav_file):
    """Conservative evidence recovery for damaged structures, in bounded blocks.

    Used only to preserve declared origin; evidence alone never validates a file.
    """
    import re
    wav_file.seek(0)
    tail = b''
    while True:
        block = wav_file.read(65536)
        if not block:
            return False
        data = tail + block
        if b'vkin' in data or re.search(br'"backend"\s*:\s*"vkinging"', data):
            return True
        tail = data[-128:]


def _inspection_comments(data):
    """Parse LIST/INFO comments, rejecting malformed nested lengths/padding."""
    if data[:4] != b'INFO':
        return []
    offset, comments = 4, []
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError('truncated INFO subchunk')
        name, size = struct.unpack_from('<4sI', data, offset)
        offset += 8
        end = offset + size
        if end + size % 2 > len(data):
            raise ValueError('INFO subchunk extends beyond LIST')
        if size % 2 and data[end:end + 1] != b'\0':
            raise ValueError('nonzero INFO padding')
        if name == b'ICMT' and data[offset:end].startswith(b'mic_calibration='):
            comments.append(data[offset:end][16:].rstrip(b'\0'))
        offset = end + size % 2
    return comments


def inspect_wav_calibration_metadata(path, logger=None):
    """Inspect provenance before normalization; never substitutes live factors.

    source is legacy or vkinging; status carries parse/calibration validity.
    I/O errors produce an invalid inspection with a diagnostic reason.
    """
    from consts.vkinging_consts import VKINGING_WAV_MARKER
    source, declared = 'legacy', False
    structure_valid = False
    try:
        with open(path, 'rb') as wav_file:
            try:
                layout = scan_riff_layout(wav_file)
            except ValueError:
                # Recovery is needed when a damaged length hides later chunks.
                declared = _provenance_evidence(wav_file)
                source = 'vkinging' if declared else 'legacy'
                raise
            # Malformed calibration LIST/JSON must not make otherwise readable
            # voltage samples unusable. RIFF/audio structure is independent.
            structure_valid = (sum(name == b'fmt ' and size >= 16 for name, _, size in layout) == 1
                               and sum(name == b'data' for name, _, _ in layout) == 1)
            declared = any(name == b'vkin' for name, _, _ in layout)
            source = 'vkinging' if declared else 'legacy'
            markers, comments, formats, data_chunks, facts = [], [], [], [], []
            for name, offset, size in layout:
                if name in (b'vkin', b'LIST', b'fmt ', b'fact'):
                    if size > 1048576:
                        raise ValueError('metadata/header chunk exceeds supported size')
                    wav_file.seek(offset)
                    payload = wav_file.read(size)
                    if name == b'vkin':
                        markers.append((offset, payload))
                    elif name == b'LIST':
                        import re
                        if re.search(br'"backend"\s*:\s*"vkinging"', payload):
                            declared, source = True, 'vkinging'
                        comments.extend(_inspection_comments(payload))
                    elif name == b'fmt ':
                        formats.append(payload)
                    else:
                        facts.append(payload)
                if name == b'data':
                    data_chunks.append((offset, size))
            structure_valid = (len(formats) == 1 and len(formats[0]) >= 16 and len(data_chunks) == 1)
            for comment in comments:
                try:
                    decoded = json.loads(comment)
                except (ValueError, UnicodeDecodeError):
                    continue  # Invalid comments are diagnosed below, never used.
                if isinstance(decoded, dict) and decoded.get('backend') == 'vkinging':
                    declared, source = True, 'vkinging'
            if not declared:
                if len(formats) != 1 or len(formats[0]) < 16 or len(data_chunks) != 1:
                    raise ValueError('WAV requires exactly one format and data chunk')
                legacy = _read_legacy_wav_calibration_metadata(path, logger)
                return WavCalibrationInspection('legacy', 'legacy_valid' if legacy else
                                                'legacy_invalid' if comments else 'legacy_absent', legacy,
                                                'invalid legacy calibration metadata' if comments and not legacy else None)
            if len(markers) != 1 or len(comments) != 1:
                raise ValueError('vkinging requires exactly one marker and calibration JSON')
            marker_offset, marker = markers[0]
            import re
            version = re.fullmatch(br'vkinging;unit=V;schema=([0-9]+)\x00', marker)
            if version is not None and version.group(1) != b'1':
                raise UnsupportedVkingingSchema('unsupported vkinging marker schema')
            if marker != VKINGING_WAV_MARKER:
                raise ValueError('invalid vkinging provenance marker')
            metadata = normalize_vkinging_metadata(json.loads(comments[0]))
            if len(formats) != 1 or len(data_chunks) != 1 or marker_offset >= data_chunks[0][0]:
                raise ValueError('vkinging requires fmt and data, marker before data')
            fmt = formats[0]
            if len(fmt) < 16:
                raise ValueError('truncated WAV format')
            tag, count, rate, byte_rate, alignment, bits = struct.unpack_from('<HHIIHH', fmt)
            acq = metadata['acquisition']
            if (tag != 3 or count != len(metadata['recorded_channels']) or rate != acq['sample_rate']
                    or bits != acq['bit_depth'] or alignment != count * bits // 8
                    or byte_rate != rate * alignment or data_chunks[0][1] % alignment):
                raise ValueError('WAV format/count/rate differs from vkinging snapshot')
            if len(facts) > 1 or any(len(fact) < 4 or struct.unpack_from('<I', fact)[0] !=
                                     data_chunks[0][1] // alignment for fact in facts):
                raise ValueError('WAV fact frame count differs from data')
            calibrated = all(ch['calibrated'] for ch in metadata['recorded_channels'])
            return WavCalibrationInspection('vkinging', 'vkinging_valid' if calibrated else 'vkinging_uncalibrated',
                                            metadata, unit='V')
    except UnsupportedVkingingSchema as exc:
        return WavCalibrationInspection(source, 'unsupported', reason=str(exc), unit='V' if declared else None, structure_valid=structure_valid)
    except (OSError, ValueError, TypeError, OverflowError, RecursionError, struct.error) as exc:
        return WavCalibrationInspection(source, 'vkinging_invalid' if declared else 'legacy_invalid',
                                        reason=str(exc), unit='V' if declared else None, structure_valid=structure_valid)


def read_wav_calibration_metadata(path, logger=None):
    inspection = inspect_wav_calibration_metadata(path, logger)
    if inspection.source == 'vkinging':
        return inspection.metadata
    # Keep all historical soundcard tolerance and logger behavior intact.
    return _read_legacy_wav_calibration_metadata(path, logger)
