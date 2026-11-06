"""Station/condition configuration drafts. WAV order is the channel list order."""

from copy import deepcopy
from uuid import uuid4

from base.analysis_segments import normalize_segmented_analysis


SIGNAL_TYPES = {"sound_pressure": "声学（麦克风）", "acceleration": "振动（加速度）"}
RUNTIME_UNAVAILABLE = "新产品配置的采集与分析执行流程尚未接通；当前可编辑、保存配置。"


def new_channel(number):
    return {"channel_id": str(uuid4()), "channel_name": f"CH{number}",
            "signal_type": "", "hardware_channel": {
                "device_ref": "", "physical_channel_index": 0}}


def new_station(number):
    return {"station_id": str(uuid4()), "station_name": f"工位{number}", "channels": [new_channel(1)]}


def new_condition(number, station_ids=()):
    return {"condition_id": str(uuid4()), "condition_name": f"工况{number}",
            "applicable_station_ids": list(station_ids), "trigger_state": "",
            "test_queue": "", "segmented_analysis": {"mode": "none"}}


def default_project():
    station = new_station(1)
    return {"project_id": str(uuid4()), "project_name": "", "result_root_directory": "",
            "stations": [station], "conditions": [new_condition(1, [station["station_id"]])]}


def resize_stations(project, count):
    """Update an in-memory draft; removing stations never removes conditions."""
    if type(count) is not int or count < 1:
        raise ValueError("至少需要一个工位")
    stations = project["stations"]
    while len(stations) < count:
        stations.append(new_station(len(stations) + 1))
    del stations[count:]
    ids = {station["station_id"] for station in stations}
    for condition in project["conditions"]:
        condition["applicable_station_ids"] = [
            key for key in condition["applicable_station_ids"] if key in ids]


def copy_project(project, name):
    result = deepcopy(project)
    result["project_id"] = str(uuid4())
    result["project_name"] = name.strip()
    station_ids = {}
    for station in result["stations"]:
        old_id = station["station_id"]
        station["station_id"] = str(uuid4())
        station_ids[old_id] = station["station_id"]
        for channel in station["channels"]:
            channel["channel_id"] = str(uuid4())
    for condition in result["conditions"]:
        condition["condition_id"] = str(uuid4())
        condition["applicable_station_ids"] = [station_ids[key] for key in condition["applicable_station_ids"]]
    return result


def normalize_project(project):
    """Normalize a validated draft without generating IDs or reordering channels."""
    result = deepcopy(project)
    for key in ("project_name", "result_root_directory"):
        result[key] = result[key].strip()
    for station in result["stations"]:
        station["station_name"] = station["station_name"].strip()
        for channel in station["channels"]:
            channel["channel_name"] = channel["channel_name"].strip()
    for condition in result["conditions"]:
        for key in ("condition_name", "trigger_state", "test_queue"):
            condition[key] = condition[key].strip()
        condition["segmented_analysis"] = normalize_segmented_analysis(condition)
    return result


def hardware_channels(device):
    """Read the existing device snapshot only; never open or rescan hardware."""
    if not device:
        return []
    if device.get("backend") == "vkinging":
        identity = device.get("machine_id")
        if not identity:
            return []
        ref = f"vkinging:{identity}"
    else:
        name = device.get("name")
        if not name:
            return []
        ref = f"sounddevice:{device.get('hostapi', '')}:{name}"
    indices = device.get("physical_channels")
    if indices is None:
        indices = range(int(device.get("max_input_channels", 0)))
    return [{"device_ref": ref, "physical_channel_index": index,
             "label": f"{device.get('name', identity if device.get('backend') == 'vkinging' else ref)} / 物理CH{index + 1}"}
            for index in indices]
