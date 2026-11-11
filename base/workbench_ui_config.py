"""Workbench layout and recording preview display settings only."""

from collections.abc import Mapping
from dataclasses import dataclass

from base.workbench_config_values import _SnapshotValue, _freeze_config


@dataclass(frozen=True, eq=False)
class WorkbenchPlotCell(_SnapshotValue):
    title: str
    analysis_item: str
    analysis_type: str


@dataclass(frozen=True, eq=False)
class WorkbenchPlotLayout(_SnapshotValue):
    rows: int
    columns: int
    cells: tuple[tuple[WorkbenchPlotCell, ...], ...]

    def __post_init__(self):
        object.__setattr__(self, "cells", tuple(tuple(row) for row in self.cells))


@dataclass(frozen=True, eq=False)
class WorkbenchUiConfig(_SnapshotValue):
    layout: WorkbenchPlotLayout | None
    preview_time_modes: Mapping[str, str]

    def __post_init__(self):
        object.__setattr__(self, "preview_time_modes", _freeze_config(
            self.preview_time_modes, "preview_time_modes",
        ))
