"""Explicit-date motor records shared by daily-details tests."""

from datetime import datetime, time, timedelta


def make_daily_records(count, record_date):
    """Return independent dictionaries with deterministic, same-day times."""
    return [
        {
            "completed_at": datetime.combine(record_date, time())
            + timedelta(microseconds=index),
            "station_id": index % 3 + 1,
            "product_code": f"CASE-{index + 1:03d}",
            "forward": "OK",
            "reverse": "--",
            "result": ("OK", "NG", "未标记")[index % 3],
        }
        for index in range(count)
    ]
