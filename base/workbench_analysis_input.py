"""Assemble analysis parameters from request-bound, successful recordings."""

from dataclasses import dataclass
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from base.recording_process_protocol import RecordingRequest
from base.workbench_analysis_config import WorkbenchAnalysisConfig
from base.workbench_config_values import _same_snapshot_value
from base.workbench_recording_protocol import StationRecordingPlan, WorkbenchRecordingCompletion


class WorkbenchAnalysisInputError(ValueError):
    """A recording cannot be associated with its captured analysis configuration."""


@dataclass(frozen=True)
class AnalysisRecordingContext:
    request_id: str
    analysis_version: str
    analysis: WorkbenchAnalysisConfig
    project_id: str
    condition_id: str
    station_plans: tuple[StationRecordingPlan, ...]

    def __post_init__(self):
        object.__setattr__(self, 'station_plans', tuple(self.station_plans))


@dataclass(frozen=True)
class WorkbenchAnalysisInput:
    context: AnalysisRecordingContext
    completion: WorkbenchRecordingCompletion


def make_analysis_recording_context(
    analysis: WorkbenchAnalysisConfig, analysis_version: str, request: RecordingRequest,
) -> AnalysisRecordingContext:
    """Capture the final request's identities and immutable station plans."""
    group = request.station_group
    if group is None:
        raise WorkbenchAnalysisInputError(f'request {request.request_id!r}: missing station_group')
    context = AnalysisRecordingContext(
        request.request_id, analysis_version, analysis, group.project_id,
        group.condition_id, group.stations,
    )
    _validate_context(context)
    return context


def _error(context, message):
    return WorkbenchAnalysisInputError(
        f'request {context.request_id!r}, condition {context.condition_id!r}: {message}',
    )


def _validate_station_ids(context, stations, expected, source):
    seen = set()
    for station in stations:
        station_id = station.station_id
        if station_id in seen:
            raise _error(context, f'{source} duplicate station {station_id!r}')
        if station_id not in expected:
            raise _error(context, f'{source} unknown station {station_id!r}')
        seen.add(station_id)
    missing = [station_id for station_id in expected if station_id not in seen]
    if missing:
        raise _error(context, f'{source} missing stations {missing!r}')


def _validate_context(context):
    for name in ('request_id', 'analysis_version', 'project_id', 'condition_id'):
        value = getattr(context, name)
        if type(value) is not str or not value.strip():
            raise _error(context, f'{name} must be a nonempty string')
    shared = context.analysis.context
    if context.project_id != shared.project_id:
        raise _error(context, 'project_id differs from bound analysis')
    condition = shared.get_condition(context.condition_id)
    if condition is None:
        raise _error(context, 'condition_id is unknown in bound analysis')
    if condition.test_queue not in context.analysis.test_queues:
        raise _error(context, f'missing test_queue {condition.test_queue!r}')
    for station_id in condition.applicable_station_ids:
        if shared.get_station(station_id) is None:
            raise _error(context, f'bound analysis references unknown station {station_id!r}')
    stations = context.analysis.stations_for(context.condition_id)
    expected = [station.station_id for station in stations]
    _validate_station_ids(context, context.station_plans, expected, 'request plan')
    if [plan.station_id for plan in context.station_plans] != expected:
        raise _error(context, 'request plan station order differs from bound analysis')
    for station, plan in zip(stations, context.station_plans):
        if len({channel.hardware_channel.device_ref for channel in station.channels}) > 1:
            raise _error(context, f'station {station.station_id!r} has inconsistent device_ref values')
        expected_channels = tuple((
            channel.channel_id, channel.channel_name, channel.signal_type,
            channel.hardware_channel.device_ref, channel.hardware_channel.physical_channel_index, index,
        ) for index, channel in enumerate(station.channels))
        planned_channels = tuple((
            channel.channel_id, channel.channel_name, channel.signal_type,
            channel.device_ref, channel.physical_channel_index, channel.wav_channel_index,
        ) for channel in plan.channels)
        if not expected_channels or not _same_snapshot_value(planned_channels, expected_channels):
            raise _error(context, f'station {station.station_id!r} channel plan differs from bound analysis')


def _validate_input(item):
    context, completion = item.context, item.completion
    _validate_context(context)
    for name in ('request_id', 'project_id', 'condition_id'):
        if getattr(completion, name) != getattr(context, name):
            raise _error(context, f'completion {name} {getattr(completion, name)!r} differs from request')
    plans = {plan.station_id: plan for plan in context.station_plans}
    _validate_station_ids(context, completion.stations, plans, 'completion')
    for station in completion.stations:
        if not _same_snapshot_value(station.channels, plans[station.station_id].channels):
            raise _error(context, f'station {station.station_id!r} channels differ from request plan')
        if type(station.path) is not str or not station.path.strip() or not Path(station.path).is_absolute():
            raise _error(context, f'station {station.station_id!r} WAV path must be nonempty and absolute')


def build_analysis_payload(inputs, *, result_directories=None) -> dict:
    """Export only actual task references; do not read WAVs or create directories."""
    inputs = tuple(inputs)
    directories = {} if result_directories is None else result_directories
    valid_keys = {(item.context.request_id, plan.station_id)
                  for item in inputs for plan in item.context.station_plans}
    for key in directories:
        if key not in valid_keys:
            raise WorkbenchAnalysisInputError(f'result_directories has unknown request/station key {key!r}')
    if not inputs:
        return {'stations': [], 'conditions': [], 'test_queues': {}, 'tasks': []}
    analysis = inputs[0].context.analysis
    version = inputs[0].context.analysis_version
    request_ids = set()
    for item in inputs:
        _validate_input(item)
        context = item.context
        if context.request_id in request_ids:
            raise _error(context, f'duplicate completion for request {context.request_id!r}')
        request_ids.add(context.request_id)
        if context.analysis_version != version:
            raise _error(context, 'analysis version differs; assemble separate batches')
        if not _same_snapshot_value(analysis, context.analysis):
            raise _error(context, 'same version is bound to different analysis configurations')
    tasks = []
    for condition in analysis.context.conditions:
        for item in inputs:
            context = item.context
            if context.condition_id != condition.condition_id:
                continue
            completed = {station.station_id: station for station in item.completion.stations}
            for station in analysis.stations_for(condition.condition_id):
                tasks.append({
                    'task_id': str(uuid5(NAMESPACE_URL, json.dumps(
                        [context.request_id, station.station_id], ensure_ascii=False, separators=(',', ':'),
                    ))),
                    'station_id': station.station_id,
                    'condition_id': condition.condition_id,
                    'wav_path': completed[station.station_id].path,
                    'analysis_result_directory': directories.get((context.request_id, station.station_id)),
                })
    station_ids = {task['station_id'] for task in tasks}
    condition_ids = {task['condition_id'] for task in tasks}
    description = analysis.describe()
    conditions = [condition for condition in description['conditions']
                  if condition['condition_id'] in condition_ids]
    queue_names = {condition['test_queue'] for condition in conditions}
    return {
        'stations': [station for station in description['stations'] if station['station_id'] in station_ids],
        'conditions': conditions,
        'test_queues': {name: test_queue for name, test_queue in description['test_queues'].items() if name in queue_names},
        'tasks': tasks,
    }
