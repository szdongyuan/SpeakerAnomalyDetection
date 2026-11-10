"""Analysis test queues and independent static descriptions, without file I/O."""

from collections.abc import Mapping
from dataclasses import dataclass

from base.workbench_config_values import (
    WorkbenchContext, WorkbenchStation, _SnapshotValue, _freeze_config, _thaw_json,
)


@dataclass(frozen=True, eq=False)
class AnalysisTestQueue(_SnapshotValue):
    analysis_channel_basis: str | None
    analysis_config: Mapping

    def __post_init__(self):
        object.__setattr__(self, "analysis_config", _freeze_config(
            self.analysis_config, "analysis_config",
        ))


@dataclass(frozen=True, eq=False)
class WorkbenchAnalysisConfig(_SnapshotValue):
    context: WorkbenchContext
    test_queues: Mapping[str, AnalysisTestQueue]

    def __post_init__(self):
        object.__setattr__(self, "test_queues", _freeze_config(self.test_queues, "test_queues"))

    def stations_for(self, condition_id: str) -> tuple[WorkbenchStation, ...]:
        condition = self.context.get_condition(condition_id)
        if condition is None:
            raise ValueError(f"Unknown workbench condition {condition_id!r}")
        return tuple(station for station in self.context.stations
                     if station.station_id in condition.applicable_station_ids)

    def describe(self) -> dict:
        """Export the agreed analysis contract, without inventing recording tasks."""
        participating = {station_id for condition in self.context.conditions
                         for station_id in condition.applicable_station_ids}
        stations = [{
            "station_id": station.station_id,
            "station_name": station.station_name,
            "device_ref": station.channels[0].hardware_channel.device_ref,
            "channels": [{
                "channel_id": channel.channel_id,
                "channel_name": channel.channel_name,
                "signal_type": channel.signal_type,
                "physical_channel_index": channel.hardware_channel.physical_channel_index,
            } for channel in station.channels],
        } for station in self.context.stations
            if station.station_id in participating and station.channels]
        return _thaw_json({
            "stations": stations,
            "conditions": [{
                "condition_id": condition.condition_id,
                "condition_name": condition.condition_name,
                "test_queue": condition.test_queue,
                "segmented_analysis": condition.segmented_analysis,
            } for condition in self.context.conditions],
            "test_queues": {name: {
                "analysis_channel_basis": test_queue.analysis_channel_basis,
                "analysis_config": test_queue.analysis_config,
            } for name, test_queue in self.test_queues.items()},
            "tasks": [],
        })
