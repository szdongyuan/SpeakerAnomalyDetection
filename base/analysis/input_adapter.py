"""Resolve controller batches without files, devices, algorithms, UI or IPC."""

from copy import deepcopy
import json
import math
from pathlib import Path

from .contracts import (
    AnalysisBatch,
    AnalysisChannel,
    AnalysisInputError,
    AnalysisInputIssue,
    AnalysisInstance,
    AnalysisTask,
)


# These existing calculation paths are acoustic, not generic vibration analyzers.
_ACOUSTIC_ANALYSIS_TYPES = frozenset({"SPL", "FFT", "FBA", "Spec"})


def _fail(path, message, task_id=None, error_type="InvalidField"):
    raise AnalysisInputError(AnalysisInputIssue(task_id, path, error_type, message))


def _mapping(value, path, task_id=None):
    if not isinstance(value, dict):
        _fail(path, "必须提供配置对象", task_id)
    return value


def _list(value, path, task_id=None):
    if not isinstance(value, list):
        _fail(path, "必须提供列表", task_id)
    return value


def _text(value, path, task_id=None):
    if not isinstance(value, str) or not value.strip():
        _fail(path, "必须提供非空字符串", task_id)
    return value.strip()


def _optional_text(value, path, task_id=None):
    return None if value is None else _text(value, path, task_id)


def _absolute_path(value, path, task_id):
    value = _text(value, path, task_id)
    if "\x00" in value or not Path(value).is_absolute():
        _fail(path, "必须提供绝对路径", task_id)
    return value


def _number(value, path, task_id):
    if type(value) not in (int, float):
        _fail(path, "必须提供有限数值", task_id)
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        _fail(path, "必须提供有限数值", task_id)
    return value


def _snapshot(value, path, task_id):
    # JSON is the controller boundary; this also rejects runtime objects and NaN.
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, RecursionError) as error:
        _fail(path, f"配置必须为可序列化的 JSON 数据：{error}", task_id)


def _index(values, collection, id_field):
    indexed = {}
    for index, value in enumerate(_list(values, collection)):
        path = f"{collection}[{index}]"
        item = _mapping(value, path)
        identifier = _text(item.get(id_field), f"{path}.{id_field}")
        if identifier in indexed:
            _fail(f"{path}.{id_field}", "标识重复", identifier if id_field == "task_id" else None,
                  "DuplicateId")
        indexed[identifier] = (path, item)
    return indexed


def _resolve_reference(task, field, index, path, task_id):
    identifier = _text(task.get(field), f"{path}.{field}", task_id)
    if identifier not in index:
        _fail(f"{path}.{field}", "引用的工位或工况不存在", task_id, "UnknownReference")
    return identifier, index[identifier]


def _build_channels(station, path, task_id):
    raw_channels = _list(station.get("channels"), f"{path}.channels", task_id)
    if not raw_channels:
        _fail(f"{path}.channels", "工位至少需要一个 WAV 通道", task_id)
    device_ref = _optional_text(station.get("device_ref"), f"{path}.device_ref", task_id)
    channels = []
    for index, value in enumerate(raw_channels):
        channel_path = f"{path}.channels[{index}]"
        channel = _mapping(value, channel_path, task_id)
        physical = channel.get("physical_channel_index")
        if physical is not None and (type(physical) is not int or physical < 0):
            _fail(f"{channel_path}.physical_channel_index", "物理通道号必须为非负整数", task_id)
        factor = channel.get("v2pa_factor")
        if factor is not None:
            factor = _number(factor, f"{channel_path}.v2pa_factor", task_id)
            if factor <= 0:
                _fail(f"{channel_path}.v2pa_factor", "标定系数必须大于零", task_id)
            factor = float(factor)
        channels.append(AnalysisChannel(
            channel_index=index,
            channel_name=_text(channel.get("channel_name"), f"{channel_path}.channel_name", task_id),
            signal_type=_text(channel.get("signal_type"), f"{channel_path}.signal_type", task_id),
            channel_id=_optional_text(channel.get("channel_id"), f"{channel_path}.channel_id", task_id),
            device_ref=device_ref,
            physical_channel_index=physical,
            v2pa_factor=factor,
        ))
    return tuple(channels)


def _selected_channels(config, count, path, task_id):
    if "analysis_channels" in config:
        field = f"{path}.analysis_channels"
        selected = _list(config["analysis_channels"], field, task_id)
    else:
        field = f"{path}.analysis_channel"
        selected = [config.get("analysis_channel")]
    if not selected:
        _fail(field, "分析项至少需要选择一个通道", task_id)
    seen = set()
    for index in selected:
        if type(index) is not int or not 0 <= index < count:
            _fail(field, f"WAV 通道索引必须为 0～{count - 1} 的整数", task_id)
        if index in seen:
            _fail(field, "分析通道不能重复", task_id)
        seen.add(index)
    return selected


def _validate_time_range(config, path, task_id):
    for field in ("analysis_time_range_enabled", "limit_checked", "smooth_checked"):
        if field in config and type(config[field]) is not bool:
            _fail(f"{path}.{field}", "开关必须为布尔值", task_id)
    if not config.get("analysis_time_range_enabled", False):
        return
    start = _number(config.get("analysis_start_time_sec"), f"{path}.analysis_start_time_sec", task_id)
    end = _number(config.get("analysis_end_time_sec"), f"{path}.analysis_end_time_sec", task_id)
    if start < 0:
        _fail(f"{path}.analysis_start_time_sec", "分析开始时间不能小于零", task_id)
    # The existing SPL configuration uses zero to mean the actual WAV end.
    if end != 0 and end <= start:
        _fail(f"{path}.analysis_end_time_sec", "结束时间必须大于开始时间，或为0表示音频末尾", task_id)


def _build_instances(config, channels, path, task_id):
    config = _mapping(config, path, task_id)
    sequence = _list(config.get("display_sequence"), f"{path}.display_sequence", task_id)
    instances = []
    seen = set()
    for number, key in enumerate(sequence):
        sequence_path = f"{path}.display_sequence[{number}]"
        _text(key, sequence_path, task_id)
        if key in seen:
            _fail(sequence_path, "分析项名称重复", task_id)
        seen.add(key)
        item_path = f"{path}[{json.dumps(key, ensure_ascii=False)}]"
        item = _mapping(config.get(key), item_path, task_id)
        analysis_type = _text(item.get("type"), f"{item_path}.type", task_id)
        selected = _selected_channels(item, len(channels), item_path, task_id)
        _validate_time_range(item, item_path, task_id)
        parameters = _snapshot(item, item_path, task_id)
        for index in selected:
            instances.append(AnalysisInstance(key, analysis_type, index, deepcopy(parameters)))
    return tuple(instances)


def _resolve_analysis_config(payload, task, condition, task_path, condition_path, task_id):
    inline_path = f"{task_path}.analysis_config"
    if "test_queue" not in condition or ("analysis_config" in task and "test_queues" not in payload):
        return _mapping(task.get("analysis_config"), inline_path, task_id), inline_path
    name = _text(condition["test_queue"], f"{condition_path}.test_queue", task_id)
    queues = _mapping(payload.get("test_queues"), "test_queues", task_id)
    if name not in queues:
        _fail(f"{condition_path}.test_queue", f"本次任务未提供测试队列：{name}", task_id, "UnknownReference")
    queue_path = f"test_queues[{json.dumps(name, ensure_ascii=False)}]"
    queue = _mapping(queues[name], queue_path, task_id)
    if queue.get("analysis_channel_basis") not in (None, "wav"):
        _fail(f"{queue_path}.analysis_channel_basis", "分析通道必须对应各工位 WAV 的列", task_id)
    config_path = f"{queue_path}.analysis_config"
    config = _mapping(queue.get("analysis_config"), config_path, task_id)
    if "analysis_config" in task and task["analysis_config"] != config:
        _fail(inline_path, "任务内分析配置与引用的测试队列不一致", task_id, "ConflictingAnalysisConfig")
    return config, config_path


def _preparation_issues(task, task_path, station_path, analysis_path):
    issues = []

    def add(path, kind, message):
        issues.append(AnalysisInputIssue(task.task_id, path, kind, message))

    if task.analysis_result_directory is None and any(
        instance.analysis_type == "SPL" for instance in task.instances
    ):
        add(f"{task_path}.analysis_result_directory", "MissingOutputDirectory",
            "总体 SPL CSV 的保存目录尚未提供；不会自动推导目录或改为不保存")
    checked_calibration = set()
    for instance in task.instances:
        path = f"{analysis_path}[{json.dumps(instance.config_key, ensure_ascii=False)}]"
        channel = task.channels[instance.channel_index]
        if instance.analysis_type not in _ACOUSTIC_ANALYSIS_TYPES:
            add(f"{path}.type", "UnsupportedAnalysisType", f"分析类型尚未接入：{instance.analysis_type}")
        elif channel.signal_type != "sound_pressure":
            selection_field = (
                "analysis_channels" if "analysis_channels" in instance.parameters
                else "analysis_channel"
            )
            add(f"{path}.{selection_field}", "UnsupportedSignalType",
                f"{instance.analysis_type} 的现有声学算法不能直接用于 WAV 第{channel.channel_index + 1}通道"
                f"（{channel.signal_type}）")
        elif channel.channel_index not in checked_calibration:
            checked_calibration.add(channel.channel_index)
            if channel.v2pa_factor is None:
                add(f"{station_path}.channels[{channel.channel_index}].v2pa_factor",
                    "CalibrationNotProvided", "输入未提供标定系数，执行前须核对 WAV 标定元数据及样本单位")
    return issues


def build_analysis_batch(payload: dict) -> AnalysisBatch:
    """Build isolated tasks or raise AnalysisInputError before returning any batch.

    This does not admit tasks to a queue or certify that they can execute.
    Queue references resolve only against this payload; recording profiles and
    configuration files are never consulted. File-dependent
    validation belongs to the executor; preparation_issues remain explicit.
    """
    payload = _mapping(payload, "$input")
    stations = _index(payload.get("stations"), "stations", "station_id")
    conditions = _index(payload.get("conditions"), "conditions", "condition_id")
    tasks = _index(payload.get("tasks"), "tasks", "task_id")
    built = []
    issues = []
    for task_id, (path, raw_task) in tasks.items():
        station_id, (station_path, station) = _resolve_reference(
            raw_task, "station_id", stations, path, task_id,
        )
        condition_id, (condition_path, condition) = _resolve_reference(
            raw_task, "condition_id", conditions, path, task_id,
        )
        channels = _build_channels(station, station_path, task_id)
        config, analysis_path = _resolve_analysis_config(
            payload, raw_task, condition, path, condition_path, task_id)
        instances = _build_instances(config, channels, analysis_path, task_id)
        segment_path = f"{condition_path}.segmented_analysis"
        segmentation = _mapping(condition.get("segmented_analysis", {"mode": "none"}),
                                segment_path, task_id)
        if segmentation.get("mode") != "none":
            _fail(f"{segment_path}.mode", "当前阶段仅支持 mode=none 的整段分析", task_id,
                  "UnsupportedSegmentation")
        output = raw_task.get("analysis_result_directory")
        if output is not None:
            output = _absolute_path(output, f"{path}.analysis_result_directory", task_id)
        task = AnalysisTask(
            task_id=task_id,
            station_id=station_id,
            station_name=_optional_text(
                station.get("station_name"), f"{station_path}.station_name", task_id,
            ),
            condition_id=condition_id,
            condition_name=_optional_text(
                condition.get("condition_name"), f"{condition_path}.condition_name", task_id,
            ),
            wav_path=_absolute_path(raw_task.get("wav_path"), f"{path}.wav_path", task_id),
            analysis_result_directory=output,
            channels=channels,
            instances=instances,
            segmented_analysis=_snapshot(segmentation, segment_path, task_id),
        )
        built.append(task)
        issues.extend(_preparation_issues(task, path, station_path, analysis_path))
    return AnalysisBatch(tuple(built), tuple(issues))
