"""Qt-free single-channel display snapshots and paired recording-day results."""

import math
from dataclasses import dataclass, field
from datetime import date, datetime

from base.audio_label_policy import aggregate_judgements
from ui.sequence.workbench_units import format_workbench_spl_unit


def _format_instance_value(item):
    if item.get("execution_status") != "分析完成":
        return "--"
    metrics = item.get("metrics", {})
    key = {"SPL": "overall_spl", "FBA": "overall_weighted_db"}.get(
        item.get("analysis_type"),
    )
    value = metrics.get(key) if key else None
    if type(value) not in (int, float) or not math.isfinite(value):
        return "分析完成"
    unit = (format_workbench_spl_unit(metrics["weighting"])
            if "weighting" in metrics else (metrics.get("unit") or ""))
    return f"{value:.2f} {unit}".strip()


@dataclass(frozen=True)
class WorkbenchStationResult:
    """One complete station display; each snapshot owns its row containers."""

    forward: str
    reverse: str
    result: str
    rows: list[list[str]]
    complete: bool


@dataclass
class _Station:
    ordinal: int
    metric_keys: tuple[str, ...]
    received: dict[str, tuple[str, dict[str, str]]] = field(default_factory=dict)

    def snapshot(self, condition_ids):
        forward, forward_values = self.received.get(condition_ids[0], ("--", {}))
        reverse, reverse_values = self.received.get(condition_ids[1], ("--", {}))
        complete = all(direction in self.received for direction in condition_ids)
        return WorkbenchStationResult(
            forward, reverse, aggregate_judgements((forward, reverse)) if complete else "--",
            [[key, forward_values.get(key, "--"), reverse_values.get(key, "--")]
             for key in self.metric_keys],
            complete,
        )


@dataclass(frozen=True)
class _Round:
    analysis_version: str
    source_identity: str
    recorded_at: datetime
    condition_ids: tuple[str, str]
    stations: dict[str, _Station]
    product_code: str


class WorkbenchResultState:
    """Own all rounds and complete records for one workbench process lifetime."""

    def __init__(self):
        self._rounds: dict[str, _Round] = {}
        self._records: list[dict] = []

    def open_round(self, round_id, *, analysis_version, source_identity,
                   recorded_at, condition_ids, stations, product_code):
        """Freeze display metadata once; reopening an existing round is a no-op."""
        if round_id in self._rounds:
            return
        self._rounds[round_id] = _Round(
            analysis_version, source_identity, recorded_at, tuple(condition_ids),
            {station_id: _Station(ordinal, tuple(keys))
             for station_id, ordinal, keys in stations},
            product_code,
        )

    def receive(self, round_id, station_id, condition_id, payload):
        """Accept the first direction terminal and return its complete station view.

        The controller authenticates task IDs against submitted task contexts.
        This boundary checks the supplied station and condition against the round
        and payload, before publishing any result or recording a completed pair.
        """
        if round_id not in self._rounds:
            raise ValueError(
                f"Unknown round {round_id!r} for station {station_id!r}, "
                f"condition {condition_id!r}",
            )
        round_state = self._rounds[round_id]
        if station_id not in round_state.stations:
            raise ValueError(f"Unknown station {station_id!r} in round {round_id!r}")
        if condition_id not in round_state.condition_ids:
            raise ValueError(
                f"Unknown condition {condition_id!r} in round {round_id!r}, "
                f"station {station_id!r}",
            )
        for field_name, expected in (("station_id", station_id), ("condition_id", condition_id)):
            actual = payload.get(field_name)
            if actual != expected:
                raise ValueError(
                    f"Task {payload.get('task_id')!r} in round {round_id!r}: "
                    f"{field_name} {actual!r} does not match {expected!r}",
                )
        station = round_state.stations[station_id]
        if condition_id in station.received:
            return None
        items = payload.get("instance_results", ())
        status = aggregate_judgements(item.get("judgement") for item in items)
        values = {item["config_key"]: _format_instance_value(item) for item in items}
        station.received[condition_id] = status, values
        snapshot = station.snapshot(round_state.condition_ids)
        # Each direction is accepted once, so only the second can complete a pair.
        if snapshot.complete:
            self._records.append({
                "completed_at": round_state.recorded_at,
                "station_id": station.ordinal,
                "product_code": round_state.product_code,
                "forward": snapshot.forward,
                "reverse": snapshot.reverse,
                "result": "未标记" if snapshot.result == "--" else snapshot.result,
            })
        return snapshot

    def records_for(self, day: date) -> list[dict]:
        """Return independent complete records belonging to the recording day."""
        return [dict(record) for record in self._records
                if record["completed_at"].date() == day]
