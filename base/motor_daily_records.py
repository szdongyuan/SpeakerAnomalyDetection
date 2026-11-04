"""Immutable, process-local daily motor record snapshots, independent of Qt."""

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class DailyRecord:
    completed_at: datetime
    station_id: int
    product_code: str
    forward: str
    reverse: str
    result: str


@dataclass(frozen=True)
class DailyRecordSummary:
    total: int
    ok: int
    ng: int
    unmarked: int
    pass_rate: str


@dataclass(frozen=True)
class DailyRecordPage:
    rows: tuple[DailyRecord, ...]
    matched_count: int
    page: int
    page_count: int


class DailyRecordStore:
    """Own one complete day; replacements are validated before publication."""

    def __init__(self):
        self.record_date = date.today()
        self.records = ()
        self.summary = DailyRecordSummary(0, 0, 0, 0, "--")
        self.station_ids = ()

    def replace(self, records, record_date=None):
        """Replace the whole day, leaving the previous snapshot intact on error."""
        if not isinstance(records, (list, tuple)):
            raise TypeError("records must be a list or tuple")
        day = date.today() if record_date is None else record_date
        if not isinstance(day, date) or isinstance(day, datetime):
            raise TypeError("record_date must be a date, not datetime")
        normalized = []
        for index, row in enumerate(records):
            if not isinstance(row, dict):
                raise TypeError(f"records[{index}] must be a dict")
            for field in ("completed_at", "station_id", "product_code",
                          "forward", "reverse", "result"):
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
            if not isinstance(row["product_code"], str):
                raise TypeError(f"records[{index}].product_code must be a str")
            statuses = {}
            for field in ("forward", "reverse", "result"):
                value = row[field]
                if not isinstance(value, str):
                    raise TypeError(f"records[{index}].{field} must be a str")
                allowed = (
                    ("OK", "NG", "未标记") if field == "result"
                    else ("--", "OK", "NG")
                )
                value = value.strip().upper()
                if value not in allowed:
                    raise ValueError(f"records[{index}].{field} must be one of {allowed}")
                statuses[field] = value
            normalized.append(DailyRecord(
                completed_at, station_id, row["product_code"],
                statuses["forward"], statuses["reverse"], statuses["result"],
            ))
        normalized = tuple(sorted(
            normalized, key=lambda row: row.completed_at, reverse=True,
        ))
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
        """Select normalized options without changing the daily snapshot."""
        needle = query.strip().casefold()
        matched = tuple(
            row for row in self.records
            if (station_id is None or row.station_id == station_id)
            and (result is None or row.result == result)
            and needle in row.product_code.casefold()
        )
        page_count = max(1, (len(matched) + 19) // 20)
        current = min(max(1, page), page_count)
        start = (current - 1) * 20
        return DailyRecordPage(matched[start:start + 20], len(matched), current, page_count)
