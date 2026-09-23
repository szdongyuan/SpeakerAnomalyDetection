import json
from copy import deepcopy
from pathlib import Path

import pytest

from base.load_config import LoadUiConfig
from base.product_test_config_refresh import build_product_test_refresh_snapshot
from base.product_test_project_config import ProductTestProjectConfigManager


def make_refresh_manager(tmp_path, condition_count=20, port_count=10):
    queue_path = tmp_path / "queues" / "test.json"
    queue = [{"seq1": {
        "acq": {"mode": "RECORD_ONLY", "detail": {"total_time": 10, "sample_rate": 48000}},
        "analysis_list": {"display_sequence": ["SPL"], "SPL": {"type": "SPL", "limit_checked": True}},
    }}]
    assert LoadUiConfig.save_data_to_json(queue, str(queue_path))
    registry_path = tmp_path / "queues" / "registry.json"
    assert LoadUiConfig.save_data_to_json({"test": str(queue_path)}, str(registry_path))
    manager = ProductTestProjectConfigManager(
        str(tmp_path / "projects"), str(tmp_path / "projects" / "program_registry.json"),
        str(registry_path),
    )
    project = {
        "project_name": "A", "result_root_directory": str(tmp_path / "results"),
        "export_raw_audio_csv": False,
        "test_groups": [{"group_name": f"端口{port}", "test_conditions": [
            {"condition_name": f"档位{index}", "test_queue": "test", "trigger_state": ""}
            for index in range(condition_count)
        ]} for port in range(port_count)],
    }
    saved = manager.save_project(None, project)
    assert saved[0], saved
    return manager, project, queue_path


def test_snapshot_reads_shared_queue_once_for_200_conditions(tmp_path, monkeypatch):
    manager, _, queue_path = make_refresh_manager(tmp_path)
    original = LoadUiConfig.load_data_from_json
    reads = []

    def read(path):
        reads.append(Path(path))
        return original(path)

    monkeypatch.setattr(LoadUiConfig, "load_data_from_json", read)
    snapshot = build_product_test_refresh_snapshot(manager, "A.json")
    assert len(snapshot.conditions) == 200
    assert reads.count(queue_path) == 1
    assert snapshot.conditions[0]["display_name"] == "端口0 / 档位0"


def test_snapshot_ignores_json_formatting_and_unused_queues(tmp_path):
    manager, _, queue_path = make_refresh_manager(tmp_path)
    initial = build_product_test_refresh_snapshot(manager, "A.json")
    data = json.loads(queue_path.read_text(encoding="utf-8"))
    queue_path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    registry = json.loads(Path(manager.queue_registry_path).read_text(encoding="utf-8"))
    registry["unused"] = str(tmp_path / "missing.json")
    LoadUiConfig.save_data_to_json(registry, manager.queue_registry_path)
    assert build_product_test_refresh_snapshot(manager, "A.json").signature == initial.signature


def test_shared_queue_content_change_is_detected(tmp_path):
    manager, _, queue_path = make_refresh_manager(tmp_path)
    initial = build_product_test_refresh_snapshot(manager, "A.json")
    data = json.loads(queue_path.read_text(encoding="utf-8"))
    data[0]["seq1"]["acq"]["detail"]["total_time"] = 20
    LoadUiConfig.save_data_to_json(data, str(queue_path))
    updated = build_product_test_refresh_snapshot(manager, "A.json")
    assert updated.signature != initial.signature
    assert updated.queue_catalog["test"]["duration"] == 20


@pytest.mark.parametrize("field", ["name", "voltage", "directory", "csv", "order"])
def test_effective_project_changes_are_detected(tmp_path, field):
    manager, project, _ = make_refresh_manager(tmp_path)
    initial = build_product_test_refresh_snapshot(manager, "A.json")
    changed = deepcopy(project)
    first = changed["test_groups"][0]["test_conditions"][0]
    if field == "name":
        first["condition_name"] = "新档位"
    elif field == "voltage":
        first["input_voltage"] = "12"
    elif field == "directory":
        changed["result_root_directory"] = str(tmp_path / "other")
    elif field == "csv":
        changed["export_raw_audio_csv"] = True
    else:
        changed["test_groups"].reverse()
    assert manager.save_project("A.json", changed)[0]
    assert build_product_test_refresh_snapshot(manager, "A.json").signature != initial.signature


def test_empty_configuration_never_loads_default_or_queues(tmp_path, monkeypatch):
    manager, _, _ = make_refresh_manager(tmp_path)
    monkeypatch.setattr(manager, "load_queue_catalog", lambda **_: pytest.fail("unexpected queue read"))
    monkeypatch.setattr(manager, "load_project", lambda *_: pytest.fail("unexpected project read"))
    snapshot = build_product_test_refresh_snapshot(manager, None)
    assert snapshot.conditions == []
    assert snapshot.context == {}


def test_invalid_queue_rejected_before_runtime_application(tmp_path):
    manager, _, queue_path = make_refresh_manager(tmp_path)
    queue_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="不可用"):
        build_product_test_refresh_snapshot(manager, "A.json")


def test_condition_data_keeps_existing_progress_signature_contract(tmp_path):
    manager, _, _ = make_refresh_manager(tmp_path)
    snapshot = build_product_test_refresh_snapshot(manager, "A.json")
    existing = LoadUiConfig.load_product_test_program_condition_configs(
        str(Path(manager.program_dir) / "A.json")
    )
    assert snapshot.conditions == existing
