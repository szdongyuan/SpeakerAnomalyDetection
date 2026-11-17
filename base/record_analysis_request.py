"""Build a detached, compute-only request for one historical WAV, without I/O."""

from copy import deepcopy
from dataclasses import dataclass

from base.analysis.contracts import AnalysisInputError
from base.analysis.input_adapter import build_analysis_batch
from base.record_analysis_reference import RecordAnalysisReference
from base.workbench_analysis_config import WorkbenchAnalysisConfig


class RecordAnalysisRequestError(ValueError):
    """The selected recording cannot be analyzed using the current config."""


@dataclass(frozen=True)
class RecordAnalysisItem:
    config_key: str
    analysis_type: str
    channel_indices: tuple[int, ...]
    parameters: dict

    def __post_init__(self):
        object.__setattr__(self, 'channel_indices', tuple(self.channel_indices))
        object.__setattr__(self, 'parameters', deepcopy(self.parameters))


@dataclass(frozen=True)
class RecordAnalysisRequest:
    payload: dict
    items: tuple[RecordAnalysisItem, ...]

    def __post_init__(self):
        object.__setattr__(self, 'payload', deepcopy(self.payload))
        object.__setattr__(self, 'items', tuple(self.items))


def build_record_analysis_request(reference, analysis, task_id):
    """Match stable source identity and validate the entire referenced queue.

    Hardware connectivity and calibration stores are not consulted. Channel
    calibration remains the executor's WAV-metadata responsibility; preparation
    issues, including unsupported algorithms, do not remove navigation items.
    """
    if not isinstance(reference, RecordAnalysisReference):
        raise RecordAnalysisRequestError('reference: 缺少有效的音频定位信息')
    location = (f'record {reference.request_id!r}, station_id={reference.station_id!r}, '
                f'condition_id={reference.condition_id!r}')
    if not isinstance(analysis, WorkbenchAnalysisConfig):
        raise RecordAnalysisRequestError(f'{location}: analysis: 当前没有有效的分析配置')

    station = analysis.context.get_station(reference.station_id)
    if station is None:
        raise RecordAnalysisRequestError(f'{location}: station_id: 当前配置不存在该源工位')
    condition = analysis.context.get_condition(reference.condition_id)
    if condition is None:
        raise RecordAnalysisRequestError(f'{location}: condition_id: 当前配置不存在该源工况')
    if station.station_id not in condition.applicable_station_ids:
        raise RecordAnalysisRequestError(
            f'{location}: applicable_station_ids: 当前源工位不适用于该工况',
        )
    recorded = tuple((channel.wav_channel_index, channel.channel_id, channel.signal_type)
                     for channel in reference.channels)
    current = tuple((index, channel.channel_id, channel.signal_type)
                    for index, channel in enumerate(station.channels))
    if recorded != current:
        raise RecordAnalysisRequestError(
            f'{location}: channels: 当前配置的 WAV 通道身份与该记录不兼容',
        )

    description = analysis.describe()
    queue_name = condition.test_queue
    if queue_name not in description['test_queues']:
        raise RecordAnalysisRequestError(
            f'{location}: test_queue: 当前配置不存在引用的测试队列 {queue_name!r}',
        )
    payload = {
        'stations': [item for item in description['stations']
                     if item['station_id'] == reference.station_id],
        'conditions': [item for item in description['conditions']
                       if item['condition_id'] == reference.condition_id],
        'test_queues': {queue_name: description['test_queues'][queue_name]},
        'tasks': [{
            'task_id': task_id,
            'station_id': reference.station_id,
            'condition_id': reference.condition_id,
            'wav_path': reference.wav_path,
            'analysis_result_directory': None,
            'persist_artifacts': False,
        }],
    }
    try:
        batch = build_analysis_batch(payload)
    except AnalysisInputError as error:
        raise RecordAnalysisRequestError(f'{location}: {error}') from error

    grouped = {}
    for instance in batch.tasks[0].instances:
        grouped.setdefault(instance.config_key, []).append(instance)
    items = tuple(RecordAnalysisItem(
        key, instances[0].analysis_type,
        tuple(instance.channel_index for instance in instances),
        instances[0].parameters,
    ) for key, instances in grouped.items())
    if not items:
        payload['tasks'] = []
    return RecordAnalysisRequest(payload, items)
