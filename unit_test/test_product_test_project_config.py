"""Product configuration service acceptance tests for stations/conditions."""
import copy
import json
import os
from pathlib import Path
from uuid import UUID

import pytest
from base.load_config import LoadUiConfig
from base.product_test.config_model import resize_stations, hardware_channels
from base.product_test.config_validation import validate_structure
from base.product_test_config_refresh import build_product_test_refresh_snapshot, ProductTestConfigurationError
from base.config_deletion import ConfigDeletionService, ConfigDeletionError
from base.sequence_queue_references import SequenceQueueReferenceScanner, QueueReferenceDraft
from base.product_test.queue_context import validate_queue_save
from consts import error_code
from unit_test.product_config_fixtures import make_manager, make_project


def test_saved_station_configuration_loads_in_current_workbench(tmp_path):
    from base.motor_workbench_config import MotorWorkbenchConfigRepository

    manager = make_manager(tmp_path)
    project = make_project(tmp_path, condition_count=2)
    saved, filename = manager.save_project(None, project)
    assert saved, filename
    layout = tmp_path / "plots.json"
    cell = {"title": "声压级", "analysis_item": "声压级", "analysis_type": "SPL"}
    layout.write_text(json.dumps({"rows": 1, "columns": 2, "cells": [[cell, cell]]}), encoding="utf-8")
    repository = MotorWorkbenchConfigRepository(manager.registry_path, manager.queue_registry_path, layout)
    snapshot = repository.load_active()
    assert snapshot.project_id == project["project_id"]
    assert snapshot.active_file == filename
    assert [s.station_id for s in snapshot.stations] == [s["station_id"] for s in project["stations"]]
    assert [c.condition_id for c in snapshot.conditions] == [c["condition_id"] for c in project["conditions"]]
    assert snapshot.stations[1].channels[0].hardware_channel.physical_channel_index == 4
    assert snapshot.conditions[0].queue.analysis["声压级"]["analysis_channels"] == (0,)


def test_roundtrip_rename_updates_filename_keeps_identity_and_order(tmp_path):
    manager, project = make_manager(tmp_path), make_project(tmp_path, condition_count=2)
    ok, filename = manager.save_project(None, project)
    assert ok, filename
    assert filename == "电机A.json"
    assert manager.load_project(filename) == (error_code.OK, project)
    project["project_name"] = "改名"
    project["stations"][0]["channels"][0]["channel_name"] = "麦克风"
    assert manager.save_project(filename, project) == (True, "改名.json")
    assert not Path(manager.program_dir, filename).exists()
    assert manager.load_project("改名.json")[1] == project
    assert manager.load_registry()["active_file"] == "改名.json"
    assert manager.load_registry()["configs"][0]["project_name"] == "改名"
    assert not (tmp_path / "results").exists()


def test_save_as_remaps_ids_and_keeps_original_active(tmp_path):
    manager, project = make_manager(tmp_path), make_project(tmp_path)
    ok, original = manager.save_project(None, project)
    assert ok
    snapshot = copy.deepcopy(project)
    ok, filename = manager.save_as(project, "副本")
    assert ok, filename
    copied = manager.load_project(filename)[1]
    assert copied["project_id"] != project["project_id"]
    assert copied["conditions"][0]["applicable_station_ids"] == [s["station_id"] for s in copied["stations"]]
    assert copied["stations"][0]["channels"][0]["channel_id"] != project["stations"][0]["channels"][0]["channel_id"]
    assert project == snapshot
    assert manager.load_registry()["active_file"] == original


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(project_id="invalid"),
    lambda p: p["stations"][1].update(station_id=p["stations"][0]["station_id"]),
    lambda p: p["conditions"][0].update(applicable_station_ids=["missing"]),
    lambda p: p["conditions"][0].update(applicable_station_ids=[]),
    lambda p: p["stations"][0]["channels"][0]["hardware_channel"].update(physical_channel_index=True),
    lambda p: p["stations"][0]["channels"][0].update(signal_type=[]),
    lambda p: p.update(export_raw_audio_csv=False),
    lambda p: p["conditions"][0].update(input_voltage="12"),
    lambda p: p.update(stations=[]),
])
def test_invalid_structure_never_writes(tmp_path, mutation):
    manager, project = make_manager(tmp_path), make_project(tmp_path)
    mutation(project)
    ok, error = manager.save_project(None, project)
    assert not ok and error
    assert not (tmp_path / "products").exists()


def test_trigger_is_protocol_neutral(tmp_path):
    manager, project = make_manager(tmp_path), make_project(tmp_path)
    project["conditions"][0]["trigger_state"] = "PLC.motorReady"
    assert manager.save_project(None, project)[0]


def test_product_condition_rejects_removed_output_load_mode(tmp_path):
    manager, project = make_manager(tmp_path), make_project(tmp_path)
    project["conditions"][0]["segmented_analysis"] = {
        "mode": "output_load", "load_values": [0, 1], "load_unit": "A", "analysis_seconds": 1,
    }
    assert any("仅支持不分段或按时间" in error
               for error in manager.validate_project(project)["save_errors"])


def test_resize_removes_references_not_conditions(tmp_path):
    project = make_project(tmp_path, condition_count=2)
    ids = [c["condition_id"] for c in project["conditions"]]
    resize_stations(project, 1)
    assert [c["condition_id"] for c in project["conditions"]] == ids
    assert all(c["applicable_station_ids"] == [project["stations"][0]["station_id"]] for c in project["conditions"])


def test_queue_channel_missing_and_type_mismatch(tmp_path):
    manager, project = make_manager(tmp_path), make_project(tmp_path)
    project["stations"][1]["channels"][0]["signal_type"] = "acceleration"
    assert any("需要声学" in e for e in manager.validate_project(project)["save_errors"])
    catalog = manager.load_queue_catalog()
    catalog["测试队列A"]["data"][0]["seq1"]["analysis_list"]["声压级"]["analysis_channels"] = [2]
    assert any("不存在" in e for e in manager.validate_project(make_project(tmp_path), queue_catalog=catalog)["save_errors"])


def test_original_queue_can_be_used_without_conversion(tmp_path):
    manager, project = make_manager(tmp_path), make_project(tmp_path)
    catalog = manager.load_queue_catalog()
    assert "analysis_channel_basis" not in catalog["测试队列A"]["data"][0]["seq1"]
    assert not manager.validate_project(project, queue_catalog=catalog)["save_errors"]


@pytest.mark.parametrize("analysis_list", [{"display_sequence": [], "auto_analysis": True}, {}])
def test_recording_only_queue_is_available_and_product_can_be_saved(tmp_path, analysis_list):
    manager, project = make_manager(tmp_path), make_project(tmp_path)
    info = manager.load_queue_catalog()["测试队列A"]
    info["data"][0]["seq1"]["analysis_list"] = analysis_list
    assert LoadUiConfig.save_data_to_json(info["data"], info["path"])
    current = manager.load_queue_catalog()["测试队列A"]
    assert current["available"] and not current["can_auto_judge"]
    assert current["analysis_items"] == []
    assert manager.save_project(None, project)[0]


@pytest.mark.parametrize("fault", ["duration", "sample_rate", "missing_analysis", "malformed_order"])
def test_recording_only_support_does_not_accept_invalid_queue(tmp_path, fault):
    manager = make_manager(tmp_path)
    info = manager.load_queue_catalog()["测试队列A"]
    sequence = info["data"][0]["seq1"]
    sequence["analysis_list"] = {"display_sequence": []}
    if fault in {"duration", "sample_rate"}:
        sequence["acq"]["detail"]["total_time" if fault == "duration" else "sample_rate"] = 0
    else:
        sequence["analysis_list"]["display_sequence"] = ["missing"] if fault == "missing_analysis" else "bad"
    assert LoadUiConfig.save_data_to_json(info["data"], info["path"])
    assert not manager.load_queue_catalog()["测试队列A"]["available"]


@pytest.mark.parametrize("rename", [True, False])
def test_registry_failure_restores_original(tmp_path, monkeypatch, rename):
    manager, project = make_manager(tmp_path), make_project(tmp_path)
    ok, filename = manager.save_project(None, project)
    assert ok
    before = (tmp_path / "products" / filename).read_bytes()
    if rename:
        project["project_name"] = "修改"
    project["stations"][0]["station_name"] = "已修改工位"
    monkeypatch.setattr(manager, "save_registry", lambda _: False)
    assert not manager.save_project(filename, project)[0]
    assert (tmp_path / "products" / filename).read_bytes() == before
    assert not (tmp_path / "products" / "修改.json").exists()
    assert manager.load_registry()["active_file"] == filename


@pytest.mark.parametrize("name", ["a/b", "a\\b", "a:b", 'a"b', "a?b", "a*b", "a|b", "a<b>",
                                   "a\nb", "CON", "nul", "COM1", "LPT9", "CON.txt", "项目.",
                                   "program_registry", "PROGRAM_REGISTRY", "名" * 251])
def test_invalid_file_names_are_rejected_before_writing(tmp_path, name):
    manager, project = make_manager(tmp_path), make_project(tmp_path, project_name=name)
    assert not manager.validate_project(project)["can_save"]
    assert not manager.save_project(None, project)[0]
    assert not Path(manager.program_dir).exists()


@pytest.mark.parametrize("registered", [True, False])
def test_case_insensitive_name_collision_does_not_overwrite(tmp_path, registered):
    manager = make_manager(tmp_path)
    original = make_project(tmp_path, project_name="Motor")
    assert manager.save_project(None, original)[0]
    before = Path(manager.program_dir, "Motor.json").read_bytes()
    if not registered:
        manager.save_registry({"active_file": None, "configs": []})
    registry_before = Path(manager.registry_path).read_bytes()
    candidate = make_project(tmp_path, project_name="motor")
    assert not manager.save_project(None, candidate)[0]
    assert Path(manager.program_dir, "Motor.json").read_bytes() == before
    assert Path(manager.registry_path).read_bytes() == registry_before


def test_rename_collision_preserves_both_projects(tmp_path):
    manager = make_manager(tmp_path)
    project_a = make_project(tmp_path, project_name="A")
    project_b = make_project(tmp_path, project_name="B")
    assert manager.save_project(None, project_a)[0]
    assert manager.save_project(None, project_b)[0]
    before = {p.name: p.read_bytes() for p in Path(manager.program_dir).iterdir()}
    project_b["project_name"] = "A"
    assert not manager.save_project("B.json", project_b)[0]
    assert {p.name: p.read_bytes() for p in Path(manager.program_dir).iterdir()} == before


def test_uuid_filename_remains_readable_and_migrates_on_save(tmp_path):
    manager, project = make_manager(tmp_path), make_project(tmp_path)
    legacy_file = project["project_id"] + ".json"
    LoadUiConfig.save_data_to_json(project, str(Path(manager.program_dir, legacy_file)))
    manager.save_registry({"active_file": legacy_file, "configs": [{
        "project_id": project["project_id"], "project_name": project["project_name"], "file": legacy_file,
    }]})
    assert manager.load_project(legacy_file) == (error_code.OK, project)
    assert manager.save_project(legacy_file, project, update_active=False) == (True, "电机A.json")
    assert not Path(manager.program_dir, legacy_file).exists()
    assert manager.load_project("电机A.json") == (error_code.OK, project)
    assert manager.load_registry()["active_file"] == "电机A.json"


@pytest.mark.parametrize("registry_fails", [False, True])
def test_case_only_rename_and_rollback(tmp_path, monkeypatch, registry_fails):
    manager, project = make_manager(tmp_path), make_project(tmp_path, project_name="Motor")
    assert manager.save_project(None, project)[0]
    before = {p.name: p.read_bytes() for p in Path(manager.program_dir).iterdir()}
    project["project_name"] = "motor"
    if registry_fails:
        monkeypatch.setattr(manager, "save_registry", lambda _: False)
    result = manager.save_project("Motor.json", project)
    if registry_fails:
        assert not result[0]
        assert {p.name: p.read_bytes() for p in Path(manager.program_dir).iterdir()} == before
    else:
        assert result == (True, "motor.json")
        assert "motor.json" in os.listdir(manager.program_dir)
        assert "Motor.json" not in os.listdir(manager.program_dir)
        assert manager.load_registry()["active_file"] == "motor.json"


def test_old_file_cleanup_failure_rolls_back_registry_and_new_file(tmp_path, monkeypatch):
    manager, project = make_manager(tmp_path), make_project(tmp_path)
    ok, filename = manager.save_project(None, project)
    assert ok
    before = {p.name: p.read_bytes() for p in Path(manager.program_dir).iterdir()}
    original_remove = os.remove

    def fail_old_removal(path):
        if Path(path).name == filename:
            raise PermissionError("file in use")
        return original_remove(path)

    monkeypatch.setattr(os, "remove", fail_old_removal)
    project["project_name"] = "新名称"
    assert not manager.save_project(filename, project)[0]
    assert {p.name: p.read_bytes() for p in Path(manager.program_dir).iterdir()} == before


def test_same_name_save_and_nonactive_rename_preserve_active_project(tmp_path):
    manager, project = make_manager(tmp_path), make_project(tmp_path)
    assert manager.save_project(None, project) == (True, "电机A.json")
    project["stations"][0]["station_name"] = "修改工位"
    assert manager.save_project("电机A.json", project) == (True, "电机A.json")
    assert manager.load_project("电机A.json")[1] == project
    second = make_project(tmp_path, project_name="B")
    assert manager.save_project(None, second)[0]
    second["project_name"] = "C"
    assert manager.save_project("B.json", second) == (True, "C.json")
    assert manager.load_registry()["active_file"] == "电机A.json"


@pytest.mark.parametrize("failure", ["return_false", "raise_error"])
def test_project_write_failure_keeps_original_and_registry(tmp_path, monkeypatch, failure):
    manager, project = make_manager(tmp_path), make_project(tmp_path)
    ok, filename = manager.save_project(None, project)
    assert ok
    before = {p.name: p.read_bytes() for p in Path(manager.program_dir).iterdir()}

    def fail_write(*args):
        if failure == "raise_error":
            raise OSError("write failed")
        return False

    monkeypatch.setattr(LoadUiConfig, "save_data_to_json", fail_write)
    project["project_name"] = "新名称"
    assert not manager.save_project(filename, project)[0]
    assert {p.name: p.read_bytes() for p in Path(manager.program_dir).iterdir()} == before


def test_unregistered_file_cannot_be_overwritten_with_another_identity(tmp_path):
    manager, project = make_manager(tmp_path), make_project(tmp_path)
    LoadUiConfig.save_data_to_json(project, str(Path(manager.program_dir, "旧文件.json")))
    original = Path(manager.program_dir, "旧文件.json").read_bytes()
    other = make_project(tmp_path, project_name="其他项目")
    assert not manager.save_project("旧文件.json", other)[0]
    assert Path(manager.program_dir, "旧文件.json").read_bytes() == original
    assert not Path(manager.registry_path).exists()


def test_save_as_requires_different_name(tmp_path):
    manager, project = make_manager(tmp_path), make_project(tmp_path)
    assert manager.save_project(None, project)[0]
    original = Path(manager.program_dir, "电机A.json").read_bytes()
    assert not manager.save_as(project, "电机A")[0]
    assert Path(manager.program_dir, "电机A.json").read_bytes() == original


def test_corrupt_registry_is_not_overwritten(tmp_path):
    manager, project = make_manager(tmp_path), make_project(tmp_path)
    path = tmp_path / "products" / "program_registry.json"
    path.parent.mkdir()
    path.write_text("broken")
    assert not manager.save_project(None, project)[0]
    assert path.read_text() == "broken"


def test_import_is_draft_and_duplicate_identity_requires_explicit_target(tmp_path):
    manager, project = make_manager(tmp_path), make_project(tmp_path)
    source = tmp_path / "import.json"
    source.write_text(json.dumps(project), encoding="utf-8")
    assert manager.import_project(str(source)) == (True, project)
    assert not (tmp_path / "products").exists()
    assert manager.save_project(None, project)[0]
    assert not manager.save_project(None, project)[0]


def test_legacy_format_rejected_without_modification(tmp_path):
    manager = make_manager(tmp_path)
    source = tmp_path / "legacy.json"
    source.write_text('{"test_groups": []}', encoding="utf-8")
    before = source.read_bytes()
    assert not manager.import_project(str(source))[0]
    assert source.read_bytes() == before


def test_runtime_rejects_new_configuration(tmp_path):
    manager, project = make_manager(tmp_path), make_project(tmp_path)
    ok, filename = manager.save_project(None, project)
    assert ok
    assert manager.validate_project(project)["can_save"]
    assert not manager.validate_project(project)["is_usable"]
    with pytest.raises(ProductTestConfigurationError, match="尚未接通"):
        build_product_test_refresh_snapshot(manager, filename)
    assert not build_product_test_refresh_snapshot(manager, None).conditions


def test_saved_and_draft_queue_references_protect_deletion(tmp_path):
    manager, project = make_manager(tmp_path), make_project(tmp_path)
    assert manager.save_project(None, project)[0]
    service = ConfigDeletionService(manager)
    target = service.list_targets("queue")[0]
    with pytest.raises(ConfigDeletionError, match="正在使用"):
        service.check(target)
    scanner = service.scanner
    draft = QueueReferenceDraft(project)
    result = scanner.find_references(target.path, drafts=(draft,))
    assert len(result.references) == 2 and not result.issues


def test_queue_candidate_validates_all_referenced_stations(tmp_path):
    manager, project = make_manager(tmp_path), make_project(tmp_path)
    assert manager.save_project(None, project)[0]
    info = manager.load_queue_catalog()["测试队列A"]
    payload = copy.deepcopy(info["data"])
    payload[0]["seq1"]["analysis_list"]["声压级"]["analysis_channels"] = [1]
    assert any("需要声学" in e for e in validate_queue_save(payload, info["path"], manager))


def test_queue_save_checks_saved_mapping_even_when_editor_draft_changes_it(tmp_path):
    manager, project = make_manager(tmp_path), make_project(tmp_path)
    ok, filename = manager.save_project(None, project)
    assert ok
    draft = copy.deepcopy(project)
    for station in draft["stations"]:
        station["channels"][1]["signal_type"] = "sound_pressure"
    info = manager.load_queue_catalog()["测试队列A"]
    payload = copy.deepcopy(info["data"])
    payload[0]["seq1"]["analysis_list"]["声压级"]["analysis_channels"] = [1]
    references = (QueueReferenceDraft(draft, product_path=tmp_path / "products" / filename),)
    assert any("需要声学" in e for e in validate_queue_save(payload, info["path"], manager, references))


def test_product_delete_keeps_results(tmp_path):
    manager, project = make_manager(tmp_path), make_project(tmp_path)
    ok, filename = manager.save_project(None, project)
    assert ok
    results = tmp_path / "results"
    results.mkdir()
    (results / "keep.wav").write_bytes(b"test")
    assert manager.delete_project(filename)[0]
    assert (results / "keep.wav").exists()
    assert not manager.load_registry()["configs"]


def test_hardware_snapshot_preserves_physical_indices():
    channels = hardware_channels({"backend": "vkinging", "machine_id": "TEST", "name": "设备A", "physical_channels": [0, 4]})
    assert [c["physical_channel_index"] for c in channels] == [0, 4]
    assert channels[0]["device_ref"] == "vkinging:TEST"


def test_known_hardware_range_and_offline_warning(tmp_path):
    manager = make_manager(tmp_path, input_device_provider=lambda: {"backend": "vkinging", "machine_id": "TEST", "physical_channels": [0]})
    project = make_project(tmp_path)
    assert any("超出" in e for e in manager.validate_project(project)["save_errors"])
    manager.input_device_provider = None
    validation = manager.validate_project(project)
    assert validation["can_save"] and validation["use_warnings"]
