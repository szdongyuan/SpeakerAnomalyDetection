import json
import os

from base.load_config import LoadUiConfig
from base.product_test_program_config import ProductTestProgramConfigManager
from consts import error_code


def make_manager(tmp_path):
    program_dir = tmp_path / "product_test_programs"
    registry_path = program_dir / "program_registry.json"
    queue_dir = tmp_path / "analysis_sequence_config"
    queue_registry_path = queue_dir / "sequence_config_registry.json"
    program_dir.mkdir()
    queue_dir.mkdir()
    return ProductTestProgramConfigManager(
        str(program_dir),
        str(registry_path),
        str(queue_registry_path),
    )


def make_program(name="默认配置"):
    return {
        "name": name,
        "close_trigger_state": "",
        "pdf_report": {
            "enabled": False,
            "save_dir": "",
        },
        "sub_configs": [
            {
                "condition_name": "6000 rpm",
                "trigger_state": "01",
                "test_queue": "queue_6000",
            }
        ],
    }


def make_queue_config():
    return [
        {
            "seq1": {
                "acq": {
                    "mode": "RECORD_ONLY",
                    "detail": {
                        "total_time": 5.0,
                        "sample_rate": 44100,
                    }
                },
                "analysis_list": {
                    "display_sequence": [
                        "声压级 (SPL) 1",
                        "频谱分析 (Spec) 1",
                    ],
                    "声压级 (SPL) 1": {
                        "type": "SPL",
                        "limit_checked": True,
                    },
                    "频谱分析 (Spec) 1": {
                        "type": "Spec",
                    },
                },
            }
        }
    ]


def make_import_queue_config():
    config = make_queue_config()
    config[0]["seq1"]["acq"] = {
        "mode": "IMPORT_AUDIO",
        "detail": {
            "sample_rate": 44100,
        },
    }
    return config


def test_save_data_to_json_is_atomic(tmp_path):
    file_path = tmp_path / "config.json"
    assert LoadUiConfig.save_data_to_json({"name": "测试"}, str(file_path))
    assert not os.path.exists(str(file_path) + ".tmp")
    with open(file_path, "r", encoding="utf-8") as stream:
        assert json.load(stream) == {"name": "测试"}


def test_load_condition_configs_makes_duplicate_test_queue_keys_unique(tmp_path):
    program_path = tmp_path / "program.json"
    assert LoadUiConfig.save_data_to_json(
        {
            "name": "duplicate queue",
            "sub_configs": [
                {"condition_name": "6000", "trigger_state": "", "test_queue": "queue"},
                {"condition_name": "7000", "trigger_state": "", "test_queue": "3"},
                {"condition_name": "8000", "trigger_state": "", "test_queue": "3"},
            ],
        },
        str(program_path),
    )

    configs = LoadUiConfig.load_product_test_program_condition_configs(str(program_path))

    assert [item["condition_name"] for item in configs] == ["6000", "7000", "8000"]
    assert len({item["key"] for item in configs}) == 3


def test_load_pdf_report_config_defaults_to_disabled_for_legacy_program(tmp_path):
    program_path = tmp_path / "legacy_program.json"
    assert LoadUiConfig.save_data_to_json(
        {"name": "legacy", "sub_configs": []},
        str(program_path),
    )

    assert LoadUiConfig.load_product_test_program_pdf_report_config(
        str(program_path)
    ) == {
        "enabled": False,
        "save_dir": "",
    }


def test_load_close_trigger_state_defaults_to_empty_for_legacy_program(tmp_path):
    program_path = tmp_path / "legacy_program.json"
    assert LoadUiConfig.save_data_to_json(
        {"name": "legacy", "sub_configs": []},
        str(program_path),
    )

    assert (
        LoadUiConfig.load_product_test_program_close_trigger_state(
            str(program_path)
        )
        == ""
    )


def test_save_product_program_normalizes_close_trigger_state(tmp_path):
    manager = make_manager(tmp_path)
    program = make_program()
    program["close_trigger_state"] = "0104020000b930"

    success, file_name = manager.save_program(None, program)

    assert success
    load_code, saved_program = manager.load_program(file_name)
    assert load_code == error_code.OK
    assert saved_program["close_trigger_state"] == "01 04 02 00 00 B9 30"
    assert LoadUiConfig.load_product_test_program_close_trigger_state(
        os.path.join(manager.program_dir, file_name)
    ) == "01 04 02 00 00 B9 30"


def test_save_product_program_normalizes_top_level_pdf_report(tmp_path):
    manager = make_manager(tmp_path)
    report_dir = tmp_path / "reports"
    program = make_program()
    program["pdf_report"] = {
        "enabled": True,
        "save_dir": str(report_dir),
    }

    success, file_name = manager.save_program(None, program)

    assert success
    load_code, saved_program = manager.load_program(file_name)
    assert load_code == error_code.OK
    assert saved_program["pdf_report"] == {
        "enabled": True,
        "save_dir": os.path.abspath(str(report_dir)),
    }


def test_validate_rejects_invalid_pdf_report_shape(tmp_path):
    manager = make_manager(tmp_path)
    program = make_program()
    program["pdf_report"] = {"enabled": "yes", "save_dir": []}

    validation = manager.validate_program(program, None, {})

    assert not validation["can_save"]
    assert "pdf_report.enabled 必须是布尔值" in validation["save_errors"]
    assert "pdf_report.save_dir 必须是字符串" in validation["save_errors"]


def test_save_and_load_product_program(tmp_path):
    manager = make_manager(tmp_path)

    success, file_name = manager.save_program(None, make_program())

    assert success
    assert file_name == "默认配置.json"
    load_code, saved_program = manager.load_program(file_name)
    assert load_code == error_code.OK
    assert saved_program == make_program()
    assert manager.load_registry() == {
        "active_file": "默认配置.json",
        "configs": [
            {
                "file": "默认配置.json",
                "name": "默认配置",
            }
        ],
    }


def test_save_as_does_not_switch_existing_active_program(tmp_path):
    manager = make_manager(tmp_path)
    success, active_file = manager.save_program(None, make_program())
    assert success

    success, saved_file = manager.save_as(
        make_program(),
        "第二配置",
    )

    assert success
    assert saved_file == "第二配置.json"
    assert manager.load_registry()["active_file"] == active_file


def test_save_inactive_program_does_not_switch_active_program(tmp_path):
    manager = make_manager(tmp_path)
    success, active_file = manager.save_program(None, make_program())
    assert success
    success, inactive_file = manager.save_as(
        make_program(),
        "第二配置",
    )
    assert success

    updated_program = make_program("第二配置")
    updated_program["sub_configs"][0]["condition_name"] = "7000 rpm"
    success, saved_file = manager.save_program(
        inactive_file,
        updated_program,
    )

    assert success
    assert saved_file == inactive_file
    assert manager.load_registry()["active_file"] == active_file


def test_rename_inactive_program_does_not_switch_active_program(tmp_path):
    manager = make_manager(tmp_path)
    success, active_file = manager.save_program(None, make_program())
    assert success
    success, inactive_file = manager.save_as(
        make_program(),
        "第二配置",
    )
    assert success

    success, renamed_file = manager.save_program(
        inactive_file,
        make_program("已重命名配置"),
    )

    assert success
    assert renamed_file == "已重命名配置.json"
    assert manager.load_registry()["active_file"] == active_file


def test_import_program_copies_external_file_to_program_directory(tmp_path):
    manager = make_manager(tmp_path)
    source_dir = tmp_path / "external_configs"
    source_dir.mkdir()
    source_path = source_dir / "客户产品配置.json"
    source_program = make_program("客户产品配置")
    assert LoadUiConfig.save_data_to_json(source_program, str(source_path))

    success, file_name = manager.import_program(str(source_path))

    assert success
    assert file_name == "客户产品配置.json"
    local_path = os.path.join(manager.program_dir, file_name)
    assert os.path.isfile(local_path)
    assert os.path.abspath(local_path) != os.path.abspath(source_path)
    os.remove(source_path)

    load_code, imported_program = manager.load_program(file_name)
    assert load_code == error_code.OK
    assert imported_program == source_program
    assert manager.load_registry() == {
        "active_file": file_name,
        "configs": [
            {
                "file": file_name,
                "name": "客户产品配置",
            }
        ],
    }


def test_import_program_registers_matching_unregistered_file(tmp_path):
    manager = make_manager(tmp_path)
    success, active_file = manager.save_program(None, make_program())
    assert success

    orphan_program = make_program("模拟4")
    orphan_path = os.path.join(manager.program_dir, "模拟4.json")
    assert LoadUiConfig.save_data_to_json(orphan_program, orphan_path)

    success, file_name = manager.import_program(orphan_path)

    assert success
    assert file_name == "模拟4.json"
    assert manager.load_registry() == {
        "active_file": active_file,
        "configs": [
            {
                "file": active_file,
                "name": "默认配置",
            },
            {
                "file": "模拟4.json",
                "name": "模拟4",
            },
        ],
    }


def test_import_program_rejects_different_unregistered_file(tmp_path):
    manager = make_manager(tmp_path)
    assert manager.save_registry({"active_file": None, "configs": []})
    target_program = make_program("模拟4")
    target_program["sub_configs"][0]["condition_name"] = "existing condition"
    target_path = os.path.join(manager.program_dir, "模拟4.json")
    assert LoadUiConfig.save_data_to_json(target_program, target_path)

    source_dir = tmp_path / "external_configs"
    source_dir.mkdir()
    source_path = source_dir / "模拟4.json"
    source_program = make_program("模拟4")
    assert LoadUiConfig.save_data_to_json(source_program, str(source_path))

    success, message = manager.import_program(str(source_path))

    assert not success
    assert message == "同名配置文件已存在但内容不同：模拟4.json"
    load_code, stored_program = manager.load_program("模拟4.json")
    assert load_code == error_code.OK
    assert stored_program == target_program
    assert manager.load_registry() == {
        "active_file": None,
        "configs": [],
    }


def test_import_program_rejects_invalid_sub_config_structure(tmp_path):
    manager = make_manager(tmp_path)
    source_path = tmp_path / "invalid.json"
    assert LoadUiConfig.save_data_to_json(
        {"name": "无效配置", "sub_configs": None},
        str(source_path),
    )

    success, message = manager.import_program(str(source_path))

    assert not success
    assert "sub_configs 必须是列表" in message


def test_rebuild_registry_from_program_files(tmp_path):
    manager = make_manager(tmp_path)
    program_path = os.path.join(manager.program_dir, "默认配置.json")
    assert LoadUiConfig.save_data_to_json(make_program(), program_path)

    registry = manager.load_registry()

    assert registry["active_file"] == "默认配置.json"
    assert registry["configs"] == [
        {
            "file": "默认配置.json",
            "name": "默认配置",
        }
    ]


def test_registry_ignores_missing_program_file(tmp_path):
    manager = make_manager(tmp_path)
    assert LoadUiConfig.save_data_to_json(
        {
            "active_file": "missing.json",
            "configs": [
                {
                    "file": "missing.json",
                    "name": "已删除配置",
                }
            ],
        },
        manager.registry_path,
    )

    assert manager.load_registry() == {
        "active_file": None,
        "configs": [],
    }


def test_registry_ignores_file_name_that_does_not_match_config_name(
    tmp_path,
):
    manager = make_manager(tmp_path)
    mismatched_file = "internal_name.json"
    assert LoadUiConfig.save_data_to_json(
        make_program(),
        os.path.join(manager.program_dir, mismatched_file),
    )
    assert LoadUiConfig.save_data_to_json(
        {
            "active_file": mismatched_file,
            "configs": [
                {
                    "file": mismatched_file,
                    "name": "默认配置",
                }
            ],
        },
        manager.registry_path,
    )

    assert manager.load_registry() == {
        "active_file": None,
        "configs": [],
    }


def test_validate_rejects_duplicate_condition_and_trigger(tmp_path):
    manager = make_manager(tmp_path)
    program = make_program()
    program["sub_configs"].append(
        {
            "condition_name": "6000 rpm",
            "trigger_state": "01",
            "test_queue": "queue_7000",
        }
    )

    result = manager.validate_program(program, None)

    assert not result["can_save"]
    assert "工况名称重复：6000 rpm" in result["save_errors"]
    assert "触发状态重复：01" in result["save_errors"]


def test_validate_rejects_invalid_or_conflicting_close_trigger_state(tmp_path):
    manager = make_manager(tmp_path)
    program = make_program()
    program["sub_configs"][0]["trigger_state"] = "01 04 02 00 01 78 F0"
    program["close_trigger_state"] = "010402000178f0"

    conflict_result = manager.validate_program(program, None)

    assert not conflict_result["can_save"]
    assert (
        "关闭测试报文不能与工况状态报文重复：01 04 02 00 01 78 F0"
        in conflict_result["save_errors"]
    )

    program["close_trigger_state"] = "01"
    invalid_result = manager.validate_program(program, None)
    assert not invalid_result["can_save"]
    assert any(
        error.startswith("关闭测试报文格式错误：")
        for error in invalid_result["save_errors"]
    )


def test_incomplete_program_can_save_but_is_not_usable(tmp_path):
    manager = make_manager(tmp_path)
    program = make_program()
    program["sub_configs"][0]["trigger_state"] = ""
    program["sub_configs"][0]["test_queue"] = ""

    result = manager.validate_program(program, None)

    assert result["can_save"]
    assert not result["is_usable"]
    assert "6000 rpm 尚未绑定触发状态" in result["use_errors"]
    assert "6000 rpm 尚未选择测试队列" in result["use_errors"]


def test_queue_catalog_reads_relative_queue_path(tmp_path):
    manager = make_manager(tmp_path)
    queue_dir = os.path.dirname(manager.queue_registry_path)
    queue_path = os.path.join(queue_dir, "queue_6000.json")
    assert LoadUiConfig.save_data_to_json(make_queue_config(), queue_path)
    assert LoadUiConfig.save_data_to_json(
        {
            "queue_6000": "queue_6000.json",
            "using_config_path": queue_path,
        },
        manager.queue_registry_path,
    )

    catalog = manager.load_queue_catalog()

    assert catalog["queue_6000"]["available"]
    assert catalog["queue_6000"]["acquisition_mode"] == "RECORD_ONLY"
    assert catalog["queue_6000"]["duration"] == 5.0
    assert catalog["queue_6000"]["analysis_items"] == [
        "声压级 (SPL) 1",
        "频谱分析 (Spec) 1",
    ]


def test_import_audio_queue_is_available_without_total_time(tmp_path):
    manager = make_manager(tmp_path)
    queue_dir = os.path.dirname(manager.queue_registry_path)
    queue_path = os.path.join(queue_dir, "queue_import.json")
    assert LoadUiConfig.save_data_to_json(make_import_queue_config(), queue_path)
    assert LoadUiConfig.save_data_to_json(
        {
            "queue_import": "queue_import.json",
            "using_config_path": queue_path,
        },
        manager.queue_registry_path,
    )

    catalog = manager.load_queue_catalog()

    assert catalog["queue_import"]["available"]
    assert catalog["queue_import"]["acquisition_mode"] == "IMPORT_AUDIO"
    assert catalog["queue_import"]["duration"] is None
    assert catalog["queue_import"]["analysis_items"] == [
        "声压级 (SPL) 1",
        "频谱分析 (Spec) 1",
    ]


def test_mixed_import_and_record_queues_are_rejected(tmp_path):
    manager = make_manager(tmp_path)
    queue_dir = os.path.dirname(manager.queue_registry_path)
    record_path = os.path.join(queue_dir, "queue_record.json")
    import_path = os.path.join(queue_dir, "queue_import.json")
    assert LoadUiConfig.save_data_to_json(make_queue_config(), record_path)
    assert LoadUiConfig.save_data_to_json(
        make_import_queue_config(),
        import_path,
    )
    assert LoadUiConfig.save_data_to_json(
        {
            "queue_record": record_path,
            "queue_import": import_path,
        },
        manager.queue_registry_path,
    )
    program = make_program("混合采集配置")
    program["sub_configs"] = [
        {
            "condition_name": "6000 rpm",
            "trigger_state": "01",
            "test_queue": "queue_record",
        },
        {
            "condition_name": "7000 rpm",
            "trigger_state": "02",
            "test_queue": "queue_import",
        },
    ]

    validation = manager.validate_program(program, None)
    success, message = manager.save_program(None, program)

    expected = (
        "同一产品测试配置不能同时包含导入音频和录制音频工况，"
        "请统一各工况测试队列的采集模式后重试"
    )
    assert not validation["can_save"]
    assert expected in validation["save_errors"]
    assert not success
    assert message == expected


def test_all_import_queues_remain_usable(tmp_path):
    manager = make_manager(tmp_path)
    queue_dir = os.path.dirname(manager.queue_registry_path)
    first_path = os.path.join(queue_dir, "queue_import_1.json")
    second_path = os.path.join(queue_dir, "queue_import_2.json")
    assert LoadUiConfig.save_data_to_json(make_import_queue_config(), first_path)
    assert LoadUiConfig.save_data_to_json(make_import_queue_config(), second_path)
    assert LoadUiConfig.save_data_to_json(
        {
            "queue_import_1": first_path,
            "queue_import_2": second_path,
        },
        manager.queue_registry_path,
    )
    program = make_program("全导入配置")
    program["sub_configs"] = [
        {
            "condition_name": "6000 rpm",
            "trigger_state": "01",
            "test_queue": "queue_import_1",
        },
        {
            "condition_name": "7000 rpm",
            "trigger_state": "02",
            "test_queue": "queue_import_2",
        },
    ]

    validation = manager.validate_program(program, None)

    assert validation["can_save"]
    assert validation["is_usable"]


def test_queue_catalog_recovers_local_file_from_stale_absolute_path(tmp_path):
    manager = make_manager(tmp_path)
    queue_dir = os.path.dirname(manager.queue_registry_path)
    queue_path = os.path.join(queue_dir, "queue_6000.json")
    assert LoadUiConfig.save_data_to_json(make_queue_config(), queue_path)
    assert LoadUiConfig.save_data_to_json(
        {"queue_6000": "D:/removed-project/queue_6000.json"},
        manager.queue_registry_path,
    )

    catalog = manager.load_queue_catalog()

    assert catalog["queue_6000"]["path"] == os.path.abspath(queue_path)
    assert catalog["queue_6000"]["duration"] == 5.0
    assert catalog["queue_6000"]["analysis_items"] == [
        "声压级 (SPL) 1",
        "频谱分析 (Spec) 1",
    ]


def test_save_as_rejects_duplicate_name(tmp_path):
    manager = make_manager(tmp_path)
    success, message = manager.save_program(None, make_program())
    assert success, message

    success, message = manager.save_as(make_program(), "默认配置")

    assert not success
    assert "配置名称已存在：默认配置" in message


def test_save_as_uses_config_name_for_file_and_active_registry(tmp_path):
    manager = make_manager(tmp_path)

    success, file_name = manager.save_as(
        make_program(),
        "S004-1四转速测试",
    )

    assert success
    assert file_name == "S004-1四转速测试.json"
    load_code, saved_program = manager.load_program(file_name)
    assert load_code == error_code.OK
    assert saved_program["name"] == "S004-1四转速测试"
    assert manager.load_registry() == {
        "active_file": "S004-1四转速测试.json",
        "configs": [
            {
                "file": "S004-1四转速测试.json",
                "name": "S004-1四转速测试",
            }
        ],
    }


def test_save_renames_file_and_updates_active_registry(tmp_path):
    manager = make_manager(tmp_path)
    success, current_file = manager.save_program(None, make_program())
    assert success

    renamed_program = make_program("S004-1四转速测试")
    success, renamed_file = manager.save_program(
        current_file,
        renamed_program,
    )

    assert success
    assert renamed_file == "S004-1四转速测试.json"
    assert not os.path.exists(
        os.path.join(manager.program_dir, "默认配置.json")
    )
    assert os.path.isfile(
        os.path.join(manager.program_dir, renamed_file)
    )
    assert manager.load_registry() == {
        "active_file": renamed_file,
        "configs": [
            {
                "file": renamed_file,
                "name": "S004-1四转速测试",
            }
        ],
    }


def test_rename_registry_failure_preserves_original_file(tmp_path, monkeypatch):
    manager = make_manager(tmp_path)
    success, current_file = manager.save_program(None, make_program())
    assert success
    original_path = os.path.join(manager.program_dir, current_file)
    renamed_program = make_program("S004-1四转速测试")
    renamed_path = os.path.join(
        manager.program_dir,
        "S004-1四转速测试.json",
    )
    monkeypatch.setattr(manager, "save_registry", lambda _registry: False)

    success, message = manager.save_program(current_file, renamed_program)

    assert not success
    assert "注册表更新失败" in message
    assert os.path.isfile(original_path)
    assert not os.path.exists(renamed_path)


def test_rename_cleanup_failure_restores_registry(tmp_path, monkeypatch):
    manager = make_manager(tmp_path)
    success, current_file = manager.save_program(None, make_program())
    assert success
    original_path = os.path.join(manager.program_dir, current_file)
    renamed_program = make_program("S004-1四转速测试")
    renamed_path = os.path.join(
        manager.program_dir,
        "S004-1四转速测试.json",
    )
    real_remove = os.remove

    def fail_for_original(path):
        if os.path.abspath(path) == os.path.abspath(original_path):
            raise PermissionError("file is in use")
        real_remove(path)

    monkeypatch.setattr(os, "remove", fail_for_original)

    success, message = manager.save_program(current_file, renamed_program)

    assert not success
    assert "文件重命名失败" in message
    assert os.path.isfile(original_path)
    assert not os.path.exists(renamed_path)
    assert manager.load_registry()["active_file"] == current_file


def test_save_as_strips_json_extension_from_config_name(tmp_path):
    manager = make_manager(tmp_path)

    success, file_name = manager.save_as(
        make_program(),
        "S004-1四转速测试.json",
    )

    assert success
    assert file_name == "S004-1四转速测试.json"
    load_code, saved_program = manager.load_program(file_name)
    assert load_code == error_code.OK
    assert saved_program["name"] == "S004-1四转速测试"


def test_save_rejects_invalid_file_name_character(tmp_path):
    manager = make_manager(tmp_path)

    success, message = manager.save_as(
        make_program(),
        "S004-1/四转速测试",
    )

    assert not success
    assert "配置名称不能包含以下字符" in message
    assert os.listdir(manager.program_dir) == ["program_registry.json"]


def test_ai_only_queue_is_available_without_automatic_judgment(tmp_path):
    manager = make_manager(tmp_path)
    queue_path = os.path.join(
        os.path.dirname(manager.queue_registry_path),
        "ai_only.json",
    )
    queue_data = make_queue_config()
    analysis_list = queue_data[0]["seq1"]["analysis_list"]
    analysis_list["display_sequence"] = ["AI 分析 1"]
    analysis_list["AI 分析 1"] = {"type": "AI"}
    assert LoadUiConfig.save_data_to_json(queue_data, queue_path)
    assert LoadUiConfig.save_data_to_json(
        {"ai_only": queue_path},
        manager.queue_registry_path,
    )

    catalog = manager.load_queue_catalog()

    assert catalog["ai_only"]["available"]
    assert not catalog["ai_only"]["can_auto_judge"]
    assert catalog["ai_only"]["reason"] == ""
    assert (
        catalog["ai_only"]["judgment_reason"]
        == "未配置可输出 OK/NG 的规则阈值"
    )
    assert catalog["ai_only"]["duration"] == 5.0
    assert catalog["ai_only"]["analysis_items"] == ["AI 分析 1"]

    program = make_program()
    program["sub_configs"][0]["test_queue"] = "ai_only"
    validation = manager.validate_program(
        program,
        None,
        catalog,
    )
    assert validation["is_usable"]
    assert "不能自动输出 OK/NG" in validation["use_warnings"][0]


def test_trigger_states_include_non_idle_hardware_states(tmp_path, monkeypatch):
    manager = make_manager(tmp_path)
    serial_config = {
        "decoder": {"mode": "state_byte"},
        "state_maps": {
            "state_byte": {
                "00": {"action": "idle"},
                "01": {"action": "start_record"},
                "02": {"action": "ignore"},
                "0F": {"action": "cycle_complete"},
            }
        },
    }
    monkeypatch.setattr(
        LoadUiConfig,
        "load_serial_discrete_input_config",
        lambda: (error_code.OK, serial_config),
    )

    assert manager.load_trigger_states() == ["01", "02"]
