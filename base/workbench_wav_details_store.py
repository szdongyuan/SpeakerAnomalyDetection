"""Validated, atomic persistence for the workbench's daily WAV details."""

import json
import os
import tempfile
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path

from base.motor_daily_wav_records import DailyWavRecord, normalize_daily_wav_record
from base.record_analysis_reference import RecordAnalysisReference
from base.workbench_recording_protocol import StationWavChannel
from consts.running_consts import DEFAULT_DIR


_SCHEMA_VERSION = 1
_CHANNEL_FIELDS = (
    "channel_id", "channel_name", "signal_type", "device_ref",
    "physical_channel_index", "wav_channel_index",
)


@dataclass(frozen=True)
class WavDetailsRow:
    request_id: str
    source_station_id: str
    record: DailyWavRecord

    def __post_init__(self):
        for field in ("request_id", "source_station_id"):
            value = getattr(self, field)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{field} must be a nonempty string")
        if not isinstance(self.record, DailyWavRecord):
            raise TypeError("record must be a DailyWavRecord")
        reference = self.record.analysis_reference
        if isinstance(reference, RecordAnalysisReference) and reference.record_key != self.key:
            raise ValueError("analysis_reference identity must match the row identity")

    @property
    def key(self) -> tuple[str, str]:
        return self.request_id, self.source_station_id

    def to_display_dict(self) -> dict:
        record = self.record
        row = {
            "completed_at": record.completed_at,
            "station_id": record.station_id,
            "file_name": record.file_name,
            "condition": record.condition,
            "result": record.result,
        }
        if record.analysis_reference is not None:
            row["analysis_reference"] = record.analysis_reference
        return row


@dataclass(frozen=True)
class WorkbenchWavDetailsSnapshot:
    day: date
    rows: tuple[WavDetailsRow, ...]

    def __post_init__(self):
        if type(self.day) is not date:
            raise ValueError("date must be a local calendar date")
        if not isinstance(self.rows, (tuple, list)):
            raise TypeError("rows must be a list or tuple")

        seen = set()
        rows = []
        for index, row in enumerate(self.rows):
            if not isinstance(row, WavDetailsRow):
                raise TypeError(f"records[{index}] must be a WavDetailsRow")
            if row.key in seen:
                raise ValueError(f"records[{index}] has duplicate row identity {row.key!r}")
            seen.add(row.key)
            record = normalize_daily_wav_record(row.to_display_dict(), self.day, index)
            rows.append(row if record == row.record else replace(row, record=record))
        object.__setattr__(self, "rows", tuple(rows))


class WorkbenchWavDetailsStateError(Exception):
    """The daily WAV details could not be loaded or committed."""


def _reject_non_json_constant(token: str) -> None:
    raise ValueError(f"invalid JSON constant: {token}")


def _validate_fields(data, fields, location):
    if not isinstance(data, dict):
        raise ValueError(f"{location} must be a JSON object")
    for field in fields:
        if field not in data:
            raise ValueError(f"{location}.{field} is required")
        value = data[field]
        if isinstance(value, str):
            # JSON escapes can decode to lone surrogates that cannot be saved as UTF-8.
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError(f"{location}.{field} must be UTF-8 text: {exc}") from exc


def _decode_reference(data, request_id, source_station_id, location):
    _validate_fields(data, ("condition_id", "wav_path", "channels"), location)
    channels = data["channels"]
    if not isinstance(channels, list) or not channels:
        raise ValueError(f"{location}.channels must be a nonempty JSON array")

    decoded_channels = []
    for index, channel in enumerate(channels):
        _validate_fields(channel, _CHANNEL_FIELDS, f"{location}.channels[{index}]")
        decoded_channels.append(StationWavChannel(*(channel[field] for field in _CHANNEL_FIELDS)))
    return RecordAnalysisReference(
        request_id, source_station_id, data["condition_id"], data["wav_path"], tuple(decoded_channels),
    )


def _decode_row(data, index):
    location = f"records[{index}]"
    _validate_fields(
        data,
        ("completed_at", "station_id", "file_name", "condition", "result", "request_id", "source_station_id"),
        location,
    )
    completed_at = data["completed_at"]
    if not isinstance(completed_at, str) or ("T" not in completed_at and " " not in completed_at):
        raise ValueError(f"{location}.completed_at must be an ISO datetime")
    try:
        parsed_time = datetime.fromisoformat(completed_at)
    except ValueError as exc:
        raise ValueError(f"{location}.completed_at must be an ISO datetime: {exc}") from exc

    request_id, source_station_id = data["request_id"], data["source_station_id"]
    reference = None
    if "analysis_reference" in data:
        reference = _decode_reference(
            data["analysis_reference"], request_id, source_station_id, f"{location}.analysis_reference",
        )
    return WavDetailsRow(
        request_id, source_station_id,
        DailyWavRecord(
            parsed_time, data["station_id"], data["file_name"], data["condition"], data["result"], reference,
        ),
    )


def _decode_snapshot(data):
    _validate_fields(data, ("schema_version", "date", "records"), "state")
    version = data["schema_version"]
    if type(version) is not int or version != _SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")

    day = data["date"]
    if not isinstance(day, str):
        raise ValueError("date must be an ISO calendar date")
    try:
        parsed_day = date.fromisoformat(day)
    except ValueError as exc:
        raise ValueError(f"date must use YYYY-MM-DD format: {exc}") from exc
    if parsed_day.isoformat() != day:
        raise ValueError("date must use YYYY-MM-DD format")

    records = data["records"]
    if not isinstance(records, list):
        raise ValueError("records must be a JSON array")
    return WorkbenchWavDetailsSnapshot(parsed_day, tuple(
        _decode_row(data, index) for index, data in enumerate(records)
    ))


def _encode_row(row):
    data = row.to_display_dict()
    data["completed_at"] = row.record.completed_at.isoformat()
    data["request_id"], data["source_station_id"] = row.key
    reference = row.record.analysis_reference
    if reference is not None:
        data["analysis_reference"] = {
            "condition_id": reference.condition_id,
            "wav_path": reference.wav_path,
            "channels": [
                {field: getattr(channel, field) for field in _CHANNEL_FIELDS}
                for channel in reference.channels
            ],
        }
    return data


class WorkbenchWavDetailsStore:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else self.default_path()

    @staticmethod
    def default_path() -> Path:
        return Path(DEFAULT_DIR) / "ui/ui_config/workbench_wav_details_state.json"

    def load(self) -> WorkbenchWavDetailsSnapshot | None:
        error_message = f"Cannot load workbench WAV details state at {self.path}"
        try:
            try:
                stream = self.path.open(encoding="utf-8")
            except FileNotFoundError:
                return None
            with stream:
                text = stream.read()
        except (OSError, UnicodeError) as exc:
            raise WorkbenchWavDetailsStateError(f"{error_message}: {exc}") from exc

        try:
            data = json.loads(text, parse_constant=_reject_non_json_constant)
        except (ValueError, RecursionError) as exc:
            raise WorkbenchWavDetailsStateError(f"{error_message}: {exc}") from exc

        try:
            return _decode_snapshot(data)
        except (TypeError, ValueError) as exc:
            raise WorkbenchWavDetailsStateError(f"{error_message}: {exc}") from exc

    def save(self, snapshot: WorkbenchWavDetailsSnapshot) -> None:
        if not isinstance(snapshot, WorkbenchWavDetailsSnapshot):
            raise TypeError("snapshot must be a WorkbenchWavDetailsSnapshot")
        error_message = f"Cannot save workbench WAV details state at {self.path}"
        state = {
            "schema_version": _SCHEMA_VERSION,
            "date": snapshot.day.isoformat(),
            "records": [_encode_row(row) for row in snapshot.rows],
        }
        try:
            text = json.dumps(state, ensure_ascii=False, indent=2)
        except ValueError as exc:
            # Python can reject conversion of integers above its digit limit.
            raise WorkbenchWavDetailsStateError(f"{error_message}: {exc}") from exc

        temporary_path = None
        diagnostics = []
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            stream = tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent,
                prefix=f".{self.path.name}.", suffix=".tmp", delete=False,
            )
            try:
                temporary_path = Path(stream.name)
                stream.write(text)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                # Close before cleanup while preserving any error from writing or syncing.
                try:
                    stream.close()
                except Exception as close_error:
                    diagnostics.append(f"Could not close temporary file {temporary_path}: {close_error}")
                raise
            stream.close()
            os.replace(temporary_path, self.path)
        except BaseException as exc:
            # Discard an uncommitted file even when an unexpected error must propagate.
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except Exception as cleanup_error:
                    diagnostics.append(f"Could not remove temporary file {temporary_path}: {cleanup_error}")

            if isinstance(exc, (OSError, UnicodeError)):
                message = f"{error_message}: {exc}"
                raise WorkbenchWavDetailsStateError(". ".join([message, *diagnostics])) from exc
            for detail in diagnostics:
                exc.add_note(detail)
            raise
