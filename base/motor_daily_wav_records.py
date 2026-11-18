"""Immutable daily saved-WAV snapshots, independent of Qt and round records."""

from dataclasses import dataclass
from datetime import date, datetime

from base.motor_daily_records import DailyRecordSummary
from base.record_analysis_reference import RecordAnalysisReference


@dataclass(frozen=True)
class DailyWavRecord:
    completed_at: datetime
    station_id: int
    file_name: str
    condition: str
    result: str
    analysis_reference: RecordAnalysisReference | None = None


@dataclass(frozen=True)
class DailyWavRecordPage:
    rows: tuple[DailyWavRecord, ...]
    matched_count: int
    page: int
    page_count: int


def normalize_daily_wav_record(row, day, index) -> DailyWavRecord:
    """Validate and detach one row from a daily WAV snapshot."""
    if not isinstance(row, dict):
        raise TypeError(f"records[{index}] must be a dict")
    for field in ("completed_at", "station_id", "file_name", "condition", "result"):
        if field not in row:
            raise ValueError(f"records[{index}].{field} is required")
    completed_at = row["completed_at"]
    if not isinstance(completed_at, datetime):
        raise TypeError(f"records[{index}].completed_at must be a datetime")
    if completed_at.tzinfo is not None:
        raise ValueError(f"records[{index}].completed_at must be timezone-naive")
    if completed_at.date() != day:
        raise ValueError(f"records[{index}].completed_at must match record_date")
    station_id = row["station_id"]
    if not isinstance(station_id, int) or isinstance(station_id, bool):
        raise TypeError(f"records[{index}].station_id must be an int, not bool")
    if station_id <= 0:
        raise ValueError(f"records[{index}].station_id must be positive")
    for field in ("file_name", "condition", "result"):
        if not isinstance(row[field], str):
            raise TypeError(f"records[{index}].{field} must be a str")
    if not row["file_name"].strip():
        raise ValueError(f"records[{index}].file_name must not be empty")
    if not row["condition"].strip():
        raise ValueError(f"records[{index}].condition must not be empty")
    result = row["result"].strip().upper()
    if result not in ("OK", "NG", "未标记"):
        raise ValueError(f"records[{index}].result must be OK, NG or 未标记")
    analysis_reference = row.get("analysis_reference")
    if analysis_reference is not None and not isinstance(analysis_reference, RecordAnalysisReference):
        raise TypeError(f"records[{index}].analysis_reference must be a RecordAnalysisReference or None")
    return DailyWavRecord(
        completed_at, station_id, row["file_name"], row["condition"], result,
        analysis_reference,
    )


class DailyWavRecordStore:
    """Own a complete WAV day; validate and copy before replacing state."""

    def __init__(self):
        self.record_date = date.today()
        self.records = ()
        self.summary = DailyRecordSummary(0, 0, 0, 0, "--")
        self.station_ids = ()

    def replace(self, records, record_date=None):
        """Atomically publish a validated snapshot, detached from its caller."""
        if not isinstance(records, (list, tuple)):
            raise TypeError("records must be a list or tuple")
        day = date.today() if record_date is None else record_date
        if not isinstance(day, date) or isinstance(day, datetime):
            raise TypeError("record_date must be a date, not datetime")
        normalized = [
            normalize_daily_wav_record(row, day, index)
            for index, row in enumerate(records)
        ]
        normalized = tuple(sorted(normalized, key=lambda row: row.completed_at, reverse=True))
        total = len(normalized)
        ok = sum(row.result == "OK" for row in normalized)
        summary = DailyRecordSummary(
            total, ok, sum(row.result == "NG" for row in normalized),
            sum(row.result == "未标记" for row in normalized),
            f"{ok / total * 100:.1f}%" if total else "--",
        )
        stations = tuple(sorted({row.station_id for row in normalized}))
        self.record_date, self.records = day, normalized
        self.summary, self.station_ids = summary, stations

    def select_page(self, *, station_id=None, result=None, query="", page=1):
        """Filter filename, station and result before selecting up to 20 rows."""
        needle = query.strip().casefold()
        matched = tuple(
            row for row in self.records
            if (station_id is None or row.station_id == station_id)
            and (result is None or row.result == result)
            and needle in row.file_name.casefold()
        )
        page_count = max(1, (len(matched) + 19) // 20)
        current = min(max(1, page), page_count)
        start = (current - 1) * 20
        return DailyWavRecordPage(matched[start:start + 20], len(matched), current, page_count)
