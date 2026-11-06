"""Validate product data at the file/UI boundary; no device protocol or Qt."""

import ntpath
from uuid import UUID

from base.analysis_segments import normalize_segmented_analysis, segment_count
from base.product_test.config_model import SIGNAL_TYPES


def validate_structure(project):
    errors = []
    ids = set()

    def fields(obj, required, location):
        if not isinstance(obj, dict):
            errors.append(f"{location}必须是对象")
            return False
        missing = set(required) - obj.keys()
        extra = obj.keys() - set(required)
        if missing or extra:
            errors.append(f"{location}字段错误：缺少 {sorted(missing)}；不支持 {sorted(extra)}")
            return False
        return True

    def identity(obj, key, location):
        value = obj.get(key)
        try:
            if not isinstance(value, str) or str(UUID(value)) != value:
                raise ValueError()
        except (ValueError, TypeError, AttributeError):
            errors.append(f"{location}的 {key} 必须是规范 UUID")
            return
        if value in ids:
            errors.append(f"{location}的 ID 重复")
        ids.add(value)

    def text(obj, key, location, optional=False):
        value = obj.get(key)
        if not isinstance(value, str) or (not optional and not value.strip()):
            errors.append(f"{location}的 {key} 必须是{'文字' if optional else '非空文字'}")

    if isinstance(project, dict) and ("test_groups" in project or "sub_configs" in project):
        return ["旧产品配置结构不再支持，请重新配置；原文件未修改。"]
    if not fields(project, ("project_id", "project_name", "result_root_directory", "stations", "conditions"), "项目"):
        return errors
    identity(project, "project_id", "项目")
    text(project, "project_name", "项目")
    text(project, "result_root_directory", "项目")
    root = project["result_root_directory"]
    if isinstance(root, str) and (not ntpath.isabs(root) or "\0" in root):
        errors.append("测试结果根目录必须是有效的绝对路径")
    stations = project["stations"]
    conditions = project["conditions"]
    if not isinstance(stations, list) or not stations:
        return errors + ["至少配置一个工位，stations 必须是列表"]
    if not isinstance(conditions, list) or not conditions:
        return errors + ["至少配置一个工况，conditions 必须是列表"]
    station_ids = set()
    for index, station in enumerate(stations, 1):
        loc = f"工位{index}"
        if not fields(station, ("station_id", "station_name", "channels"), loc):
            continue
        identity(station, "station_id", loc)
        if isinstance(station["station_id"], str):
            station_ids.add(station["station_id"])
        text(station, "station_name", loc)
        if not isinstance(station["channels"], list) or not station["channels"]:
            errors.append(f"{loc}至少配置一个通道")
            continue
        for col, channel in enumerate(station["channels"], 1):
            where = f"{loc} WAV第{col}通道"
            if not fields(channel, ("channel_id", "channel_name", "signal_type", "hardware_channel"), where):
                continue
            identity(channel, "channel_id", where)
            text(channel, "channel_name", where)
            if channel["signal_type"] == "":
                errors.append(f"{where}：请选择信号类型")
            elif not isinstance(channel["signal_type"], str) or channel["signal_type"] not in SIGNAL_TYPES:
                errors.append(f"{where}信号类型无效")
            hw = channel["hardware_channel"]
            if fields(hw, ("device_ref", "physical_channel_index"), where):
                if isinstance(hw["device_ref"], str) and not hw["device_ref"].strip():
                    errors.append(f"{where}：请选择硬件通道")
                else:
                    text(hw, "device_ref", where)
                value = hw["physical_channel_index"]
                if type(value) is not int or value < 0:
                    errors.append(f"{where}物理通道下标必须是非负整数")
    for index, condition in enumerate(conditions, 1):
        loc = f"工况{index}"
        if not fields(condition, ("condition_id", "condition_name", "applicable_station_ids", "trigger_state", "test_queue", "segmented_analysis"), loc):
            continue
        identity(condition, "condition_id", loc)
        for key in ("condition_name", "test_queue", "trigger_state"):
            text(condition, key, loc, optional=(key == "trigger_state"))
        targets = condition["applicable_station_ids"]
        if not isinstance(targets, list) or not targets or not all(isinstance(key, str) for key in targets):
            errors.append(f"{loc}必须选择适用工位")
        elif len(set(targets)) != len(targets) or not set(targets) <= station_ids:
            errors.append(f"{loc}的适用工位重复或不存在")
        try:
            normalized = normalize_segmented_analysis(condition)
            if normalized["mode"] not in {"none", "time"}:
                errors.append(f"{loc}分段方式仅支持不分段或按时间")
            if set(condition["segmented_analysis"]) != set(normalized):
                errors.append(f"{loc}分段分析字段与所选模式不一致")
        except (ValueError, TypeError) as error:
            errors.append(f"{loc}：{error}")
    return errors


def selected_wav_channels(item):
    values = item.get("analysis_channels", [item.get("analysis_channel", 0)])
    if not isinstance(values, list) or not values or any(type(v) is not int or v < 0 for v in values):
        raise ValueError("分析通道必须是非空的非负整数列表")
    if len(set(values)) != len(values):
        raise ValueError("分析通道不能重复")
    return values


def validate_references(project, catalog, *, allow_unassigned_channels=False):
    errors = []
    stations = {item["station_id"]: item for item in project["stations"]}
    for condition in project["conditions"]:
        loc = condition["condition_name"]
        info = catalog.get(condition["test_queue"])
        if not info or not info.get("available"):
            errors.append(f"{loc}：测试队列不存在或不可用：{condition['test_queue']}")
            continue
        data = info["data"]
        if len(data) != 1 or len(data[0]) != 1:
            errors.append(f"{loc}：产品工况需要一个录音序列")
            continue
        seq = next(iter(data[0].values()))
        try:
            segment_count(normalize_segmented_analysis(condition), info["duration"])
        except ValueError as error:
            errors.append(f"{loc}：{error}")
        analyses = seq.get("analysis_list", {})
        for name in info["analysis_items"]:
            item = analyses[name]
            if item.get("type") == "Excel":
                continue
            try:
                columns = selected_wav_channels(item)
            except ValueError as error:
                errors.append(f"{loc}/{name}：{error}")
                continue
            for column in columns:
                types = set()
                for key in condition["applicable_station_ids"]:
                    station = stations[key]
                    label = f"{loc}/{name}/{station['station_name']} WAV第{column + 1}通道"
                    if column >= len(station["channels"]):
                        errors.append(f"{label}不存在")
                        continue
                    signal_type = station["channels"][column]["signal_type"]
                    if allow_unassigned_channels and signal_type == "":
                        continue
                    types.add(signal_type)
                    if item.get("type") in {"SPL", "LOUD", "LOUDNESS"} and signal_type != "sound_pressure":
                        errors.append(f"{label}需要声学信号")
                if len(types) > 1:
                    errors.append(f"{loc}/{name} WAV第{column + 1}通道在适用工位的信号类型不一致")
    return errors
