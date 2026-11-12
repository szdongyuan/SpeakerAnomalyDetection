"""Immutable view data; no Qt widgets, queue ownership or calculation logic."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ConditionSegmentDisplay:
    condition_id: str
    name: str
    duration_seconds: float | None = None
    channels: tuple[tuple[int, str], ...] = ()
    x_unit: str = ''
    y_unit: str = ''
    warning: str = ''


@dataclass(frozen=True)
class ConditionPlotDisplay:
    settings: Mapping
    conditions: tuple[ConditionSegmentDisplay, ...]
    station_id: str = ''
    summary_rows: int = 0


@dataclass(frozen=True)
class WorkbenchPlotCellDisplay:
    title: str = ''
    unit: str = ''
    first_condition: str = ''
    second_condition: str = ''
    warning: str = ''
    warning_detail: str = ''
    condition_plot: ConditionPlotDisplay | None = None
