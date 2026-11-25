"""Qt-free condition display snapshots and station recording-day results."""

import math
from dataclasses import dataclass, field
from datetime import date, datetime

from consts.acoustic_analysis.common_consts import SOUND_QUALITY_ANALYSES
from base.audio_label_policy import aggregate_judgements, result_status
from ui.sequence.workbench_units import format_workbench_spl_unit


def _format_instance_value(item):
    if item.get("execution_status") != "分析完成":
        return "--"
    metrics = item.get("metrics", {})
    key = {"SPL": "overall_spl", "FBA": "overall_weighted_db"}.get(
        item.get("analysis_type"),
    )
    if item.get("analysis_type") in SOUND_QUALITY_ANALYSES:
        spec = SOUND_QUALITY_ANALYSES[item["analysis_type"]]
        key = metrics.get("limit_metric", spec["default_metric"])
        if key == "curve_y":
            key = spec["default_metric"]
    value = metrics.get(key) if key else None
    if type(value) not in (int, float) or not math.isfinite(value):
        return "无有效数值" if item.get("analysis_type") in SOUND_QUALITY_ANALYSES else "分析完成"
    unit = (format_workbench_spl_unit(metrics["weighting"])
            if "weighting" in metrics else (metrics.get("unit") or ""))
    return f"{value:.2f} {unit}".strip()


@dataclass(frozen=True)
class WorkbenchStationResult:
    """All configured columns; legacy direction fields only serve daily-record consumers."""

    condition_statuses: tuple[str, ...]
    result: str
    rows: list[list[str]]
    complete: bool
    judgement_complete: bool = False

    @property
    def forward(self):
        return self.condition_statuses[0] if self.condition_statuses else "--"

    @property
    def reverse(self):
        return self.condition_statuses[1] if len(self.condition_statuses) > 1 else "--"


@dataclass
class _Station:
    ordinal: int
    metric_keys: tuple[str, ...]
    expected_conditions: tuple[str, ...]
    received: dict[str, dict[str, str]] = field(default_factory=dict)
    judgements: dict[str, str] = field(default_factory=dict)
    cancelled: bool = False

    def snapshot(self, condition_ids):
        if self.cancelled:
            return WorkbenchStationResult(
                ("--",) * len(condition_ids), "--",
                [[key, *(["--"] * len(condition_ids))] for key in self.metric_keys], False,
            )
        statuses = tuple(self.judgements.get(key, "--") for key in condition_ids)
        complete = bool(self.expected_conditions) and all(
            key in self.received for key in self.expected_conditions)
        judgement_complete = bool(self.expected_conditions) and all(
            key in self.judgements for key in self.expected_conditions)
        return WorkbenchStationResult(
            statuses,
            aggregate_judgements(self.judgements[key] for key in self.expected_conditions)
            if judgement_complete else "--",
            [[key, *(self.received.get(condition, {}).get(key, "--") for condition in condition_ids)]
             for key in self.metric_keys],
            complete, judgement_complete,
        )


@dataclass(frozen=True)
class _Round:
    analysis_version: str
    source_identity: str
    recorded_at: datetime
    condition_ids: tuple[str, ...]
    stations: dict[str, _Station]
    product_code: str


class WorkbenchResultState:
    """Own all rounds and complete records for one workbench process lifetime."""

    def __init__(self):
        self._rounds: dict[str, _Round] = {}
        self._records: list[dict] = []

    def open_round(self, round_id, *, analysis_version, source_identity,
                   recorded_at, condition_ids, stations, product_code, expected_analysis_keys=None):
        """Freeze display metadata once; reopening an existing round is a no-op."""
        if round_id in self._rounds:
            return
        self._rounds[round_id] = _Round(
            analysis_version, source_identity, recorded_at, tuple(condition_ids),
            {station_id: _Station(ordinal, tuple(keys), tuple(
                condition for condition in condition_ids
                if expected_analysis_keys is None or (condition, station_id) in expected_analysis_keys))
             for station_id, ordinal, keys in stations},
            product_code,
        )

    def _locate(self, round_id, station_id, condition_id, payload):
        """Validate shared round/condition identities for both event types."""
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
        if condition_id not in station.expected_conditions:
            raise ValueError(f"Unexpected condition {condition_id!r} for station {station_id!r}")
        return round_state, station

    def cancel_round(self, round_id):
        """Retire unfinished station attempts; retain already completed records."""
        state = self._rounds[round_id]
        cancelled = {}
        for station_id, station in state.stations.items():
            if not station.cancelled and not station.snapshot(state.condition_ids).complete:
                station.cancelled = True
                cancelled[station_id] = station.snapshot(state.condition_ids)
        return cancelled

    def receive_judgement(self, round_id, station_id, condition_id, payload):
        """Update verdicts only; terminal metrics and daily records stay pending."""
        round_state, station = self._locate(round_id, station_id, condition_id, payload)
        if station.cancelled or condition_id in station.judgements:
            return None
        station.judgements[condition_id] = result_status(payload)
        return station.snapshot(round_state.condition_ids)

    def receive(self, round_id, station_id, condition_id, payload):
        """Accept each terminal once, independently of an earlier verdict."""
        round_state, station = self._locate(round_id, station_id, condition_id, payload)
        if condition_id in station.received:
            return None
        items = payload.get("instance_results", ())
        status = result_status(payload)
        grouped = {}
        for item in items:
            grouped.setdefault(item["config_key"], []).append(item)
        values = {key: (_format_instance_value(group[0]) if len(group) == 1 else
                        " / ".join(f"CH{i['channel_index'] + 1}: {_format_instance_value(i)}"
                                   for i in group)) for key, group in grouped.items()}
        station.judgements[condition_id] = status
        station.received[condition_id] = values
        snapshot = station.snapshot(round_state.condition_ids)
        # Only the last unreceived condition can complete this round once.
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

    def station_snapshots(self, round_id):
        """Return the frozen station identities and current results for a round."""
        state = self._rounds[round_id]
        return {key: (station.ordinal, station.expected_conditions, station.snapshot(state.condition_ids))
                for key, station in state.stations.items()}
