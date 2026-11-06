"""Isolated station configuration fixtures; never use the user's configuration directory."""
from base.load_config import LoadUiConfig
from base.product_test_project_config import ProductTestProjectConfigManager
from base.product_test.config_model import default_project, new_channel, resize_stations


def make_manager(tmp_path, queue_specs=None, **kwargs):
    products = tmp_path / "products"
    queues = tmp_path / "queues"
    specs = queue_specs or {"测试队列A": {}}
    registry = {}
    for name, spec in specs.items():
        path = queues / (name + ".json")
        queue = [{"seq1": {"acq": {
            "name": "录制音频", "mode": "RECORD_ONLY", "detail": {"total_time": 3.0, "sample_rate": 44100}},
            "analysis_list": {"display_sequence": ["声压级"], "auto_analysis": True,
                              "声压级": {"type": "SPL", "analysis_channels": [0], "analysis_channel": 0,
                                         "limit_checked": spec.get("limit_checked", True)}}}}]
        assert LoadUiConfig.save_data_to_json(queue, str(path))
        registry[name] = str(path)
    assert LoadUiConfig.save_data_to_json(registry, str(queues / "registry.json"))
    return ProductTestProjectConfigManager(str(products), str(products / "program_registry.json"),
                                           str(queues / "registry.json"), **kwargs)


def make_project(tmp_path, *, project_name="电机A", condition_count=1, station_count=2):
    from base.product_test.config_model import new_condition
    project = default_project()
    project["project_name"] = project_name
    project["result_root_directory"] = str(tmp_path / "results")
    resize_stations(project, station_count)
    for i, station in enumerate(project["stations"]):
        station["channels"] = []
        for col in range(2):
            channel = new_channel(col + 1)
            channel["signal_type"] = "sound_pressure" if col == 0 else "acceleration"
            channel["hardware_channel"] = {"device_ref": "vkinging:TEST", "physical_channel_index": i * 4 + col}
            station["channels"].append(channel)
    project["conditions"] = []
    for number in range(condition_count):
        condition = new_condition(number + 1, [s["station_id"] for s in project["stations"]])
        condition["test_queue"] = "测试队列A"
        project["conditions"].append(condition)
    return project
