import json
import os

import pytest

from base.load_config import LoadUiConfig
from base.product_test_project_config import (
    ProductTestProjectConfigManager,
    ProductTestProjectValidator,
    classify_project_trigger_mode,
    flatten_test_conditions,
    iter_test_conditions,
)
from consts import error_code
from consts.product_test_project_consts import (
    GROUP_NAME_KEY,
    PROJECT_NAME_KEY,
    RESULT_ROOT_DIRECTORY_KEY,
    TEST_CONDITIONS_KEY,
    TEST_GROUPS_KEY,
)


def make_manager(tmp_path, queue_specs=None):
    program_dir = tmp_path / "projects"
    registry_path = program_dir / "program_registry.json"
    queue_dir = tmp_path / "queues"
    queue_registry_path = queue_dir / "registry.json"
    queue_registry = {}
    for queue_name, spec in (queue_specs or {"基础测试": {}}).items():
        queue_path = queue_dir / f"{queue_name}.json"
        analysis_type = spec.get("analysis_type", "SPL")
        analysis_config = {
            "type": analysis_type,
            "limit_checked": spec.get("limit_checked", True),
        }
        if analysis_type == "AI":
            analysis_config["analyse_model_name"] = spec.get(
                "analyse_model_name",
                "demo_model",
            )
        queue_data = [
            {
                "录制与分析": {
                    "acq": {
                        "mode": spec.get("mode", "RECORD_ONLY"),
                        "detail": {
                            "total_time": spec.get("duration", 600),
                            "sample_rate": 48000,
                        },
                    },
                    "analysis_list": {
                        "display_sequence": ["分析项1"],
                        "分析项1": analysis_config,
                    },
                }
            }
        ]
        assert LoadUiConfig.save_data_to_json(queue_data, str(queue_path))
        queue_registry[queue_name] = str(queue_path)
    assert LoadUiConfig.save_data_to_json(
        queue_registry,
        str(queue_registry_path),
    )
    return ProductTestProjectConfigManager(
        program_dir=str(program_dir),
        registry_path=str(registry_path),
        queue_registry_path=str(queue_registry_path),
    )


def make_project(tmp_path, *, project_name="PB-A01充电宝", condition_count=1):
    conditions = [
        {
            "condition_name": f"档位{index}",
            "trigger_state": "",
            "test_queue": "基础测试",
        }
        for index in range(1, condition_count + 1)
    ]
    return {
        PROJECT_NAME_KEY: project_name,
        RESULT_ROOT_DIRECTORY_KEY: str(tmp_path / "results"),
        TEST_GROUPS_KEY: [
            {
                GROUP_NAME_KEY: "USB-C输出口",
                TEST_CONDITIONS_KEY: conditions,
            },
            {
                GROUP_NAME_KEY: "USB-A输出口",
                TEST_CONDITIONS_KEY: [
                    {
                        "condition_name": "档位1",
                        "trigger_state": "",
                        "test_queue": "基础测试",
                    }
                ],
            },
        ],
    }


def test_runtime_loader_flattens_groups_with_composite_keys(tmp_path):
    project_path = tmp_path / "PB-A01充电宝.json"
    project = make_project(tmp_path)
    project[TEST_GROUPS_KEY][0][TEST_CONDITIONS_KEY][0][
        "trigger_state"
    ] = "01 04 02 00 01 78 f0"
    project[TEST_GROUPS_KEY][1][TEST_CONDITIONS_KEY][0][
        "trigger_state"
    ] = "01 04 02 01 01 29 30"
    assert LoadUiConfig.save_data_to_json(project, str(project_path))

    conditions = LoadUiConfig.load_product_test_program_condition_configs(
        str(project_path)
    )

    assert conditions == [
        {
            "key": "group_1:condition_1",
            "group_name": "USB-C输出口",
            "condition_name": "档位1",
            "display_name": "USB-C输出口 / 档位1",
            "trigger_state": "01 04 02 00 01 78 F0",
            "test_queue": "基础测试",
        },
        {
            "key": "group_2:condition_1",
            "group_name": "USB-A输出口",
            "condition_name": "档位1",
            "display_name": "USB-A输出口 / 档位1",
            "trigger_state": "01 04 02 01 01 29 30",
            "test_queue": "基础测试",
        },
    ]


def test_default_project_contains_only_new_schema_fields():
    project = ProductTestProjectConfigManager.default_project()

    assert set(project) == {
        PROJECT_NAME_KEY,
        RESULT_ROOT_DIRECTORY_KEY,
        TEST_GROUPS_KEY,
    }
    assert "name" not in project
    assert "sub_configs" not in project
    assert "config_version" not in project
    assert "close_trigger_state" not in project
    assert "pdf_report" not in project


def test_save_project_keeps_more_than_twenty_conditions_and_writes_registry(
    tmp_path,
):
    manager = make_manager(tmp_path)
    project = make_project(tmp_path, condition_count=21)

    success, file_name = manager.save_project(None, project)

    assert success is True
    assert file_name == "PB-A01充电宝.json"
    load_code, saved = manager.load_project(file_name)
    assert load_code == error_code.OK
    assert len(saved[TEST_GROUPS_KEY][0][TEST_CONDITIONS_KEY]) == 21
    assert set(saved) == {
        PROJECT_NAME_KEY,
        RESULT_ROOT_DIRECTORY_KEY,
        TEST_GROUPS_KEY,
    }
    assert os.path.isdir(tmp_path / "results" / "PB-A01充电宝")
    assert manager.load_registry() == {
        "active_file": file_name,
        "configs": [
            {"file": file_name, PROJECT_NAME_KEY: "PB-A01充电宝"}
        ],
    }


@pytest.mark.parametrize(
    ("mutator", "expected_message"),
    [
        (
            lambda project: project.update({PROJECT_NAME_KEY: "PB_A01"}),
            "项目名称不能包含",
        ),
        (
            lambda project: project.update(
                {RESULT_ROOT_DIRECTORY_KEY: "relative/results"}
            ),
            "测试结果根目录必须是绝对路径",
        ),
        (
            lambda project: project[TEST_GROUPS_KEY][1].update(
                {GROUP_NAME_KEY: "USB-C输出口"}
            ),
            "端口名称重复",
        ),
        (
            lambda project: project[TEST_GROUPS_KEY][0][
                TEST_CONDITIONS_KEY
            ].append(
                {
                    "condition_name": "档位1",
                    "trigger_state": "",
                    "test_queue": "基础测试",
                }
            ),
            "工况名称重复",
        ),
    ],
)
def test_project_validation_reports_project_group_and_condition_errors(
    tmp_path,
    mutator,
    expected_message,
):
    project = make_project(tmp_path)
    mutator(project)

    errors = ProductTestProjectValidator.validate_for_save(
        project,
        {"active_file": None, "configs": []},
        None,
    )

    assert any(expected_message in error for error in errors)


def test_status_codes_are_global_and_must_be_all_empty_or_all_present(tmp_path):
    project = make_project(tmp_path)
    conditions = flatten_test_conditions(project)
    assert classify_project_trigger_mode(conditions) == "manual"

    project[TEST_GROUPS_KEY][0][TEST_CONDITIONS_KEY][0]["trigger_state"] = (
        "01 04 02 00 01 78 F0"
    )
    mixed_errors = ProductTestProjectValidator.validate_for_save(
        project,
        {"active_file": None, "configs": []},
        None,
    )
    assert "所有工况状态码必须全部配置或全部留空" in mixed_errors

    duplicate_state = "01 04 02 00 01 78 F0"
    for _group_index, _group, _condition_index, condition in iter_test_conditions(
        project
    ):
        condition["trigger_state"] = duplicate_state
    duplicate_errors = ProductTestProjectValidator.validate_for_save(
        project,
        {"active_file": None, "configs": []},
        None,
    )
    assert any("状态码重复" in error for error in duplicate_errors)


def test_registry_keeps_missing_file_record_until_explicit_delete(tmp_path):
    manager = make_manager(tmp_path)
    registry = {
        "active_file": "缺失配置.json",
        "configs": [
            {
                "file": "缺失配置.json",
                PROJECT_NAME_KEY: "缺失配置",
            }
        ],
    }
    assert manager.save_registry(registry)

    assert manager.load_registry() == registry
    load_code, message = manager.load_project("缺失配置.json")
    assert load_code == error_code.INVALID_DATA_LOADING
    assert message == "产品测试配置文件不存在，请选择其他配置。"
    assert manager.load_registry() == registry


def test_rename_project_updates_file_and_registry_but_keeps_old_results(tmp_path):
    manager = make_manager(tmp_path)
    project = make_project(tmp_path)
    success, original_file = manager.save_project(None, project)
    assert success

    project[PROJECT_NAME_KEY] = "PB-A02充电宝"
    success, renamed_file = manager.save_project(original_file, project)

    assert success is True
    assert renamed_file == "PB-A02充电宝.json"
    assert not os.path.exists(os.path.join(manager.program_dir, original_file))
    assert os.path.isfile(os.path.join(manager.program_dir, renamed_file))
    assert os.path.isdir(tmp_path / "results" / "PB-A01充电宝")
    assert os.path.isdir(tmp_path / "results" / "PB-A02充电宝")
    assert manager.load_registry()["active_file"] == renamed_file


def test_registry_failure_restores_previous_project_file(tmp_path, monkeypatch):
    manager = make_manager(tmp_path)
    project = make_project(tmp_path)
    success, file_name = manager.save_project(None, project)
    assert success
    file_path = os.path.join(manager.program_dir, file_name)
    with open(file_path, "rb") as file_handle:
        original_bytes = file_handle.read()

    project[TEST_GROUPS_KEY][0][TEST_CONDITIONS_KEY][0][
        "condition_name"
    ] = "修改后档位"
    monkeypatch.setattr(manager, "save_registry", lambda _registry: False)

    success, message = manager.save_project(file_name, project)

    assert success is False
    assert "注册表更新失败" in message
    with open(file_path, "rb") as file_handle:
        assert file_handle.read() == original_bytes


def test_existing_project_is_not_overwritten_when_snapshot_read_fails(
    tmp_path, monkeypatch
):
    manager = make_manager(tmp_path)
    project = make_project(tmp_path)
    success, file_name = manager.save_project(None, project)
    assert success
    file_path = os.path.join(manager.program_dir, file_name)
    with open(file_path, "rb") as file_handle:
        original_bytes = file_handle.read()

    def fail_snapshot(_file_path):
        raise OSError("access denied")

    monkeypatch.setattr(manager, "_read_file_snapshot", fail_snapshot)
    project[TEST_GROUPS_KEY][0][TEST_CONDITIONS_KEY][0][
        "condition_name"
    ] = "修改后档位"

    success, message = manager.save_project(file_name, project)

    assert success is False
    assert "无法读取原产品测试配置" in message
    with open(file_path, "rb") as file_handle:
        assert file_handle.read() == original_bytes


def test_delete_project_removes_file_and_registration_without_results(tmp_path):
    manager = make_manager(tmp_path)
    project = make_project(tmp_path)
    success, file_name = manager.save_project(None, project)
    assert success
    result_directory = tmp_path / "results" / "PB-A01充电宝"

    success, deleted_file = manager.delete_project(file_name)

    assert success is True
    assert deleted_file == file_name
    assert not os.path.exists(os.path.join(manager.program_dir, file_name))
    assert manager.load_registry() == {"active_file": None, "configs": []}
    assert result_directory.is_dir()


def test_delete_missing_project_only_removes_registry_record(tmp_path):
    manager = make_manager(tmp_path)
    assert manager.save_registry(
        {
            "active_file": "缺失配置.json",
            "configs": [
                {
                    "file": "缺失配置.json",
                    PROJECT_NAME_KEY: "缺失配置",
                }
            ],
        }
    )

    success, _file_name = manager.delete_project("缺失配置.json")

    assert success is True
    assert manager.load_registry() == {"active_file": None, "configs": []}


def test_queue_catalog_exposes_duration_analysis_and_ai_judgment(tmp_path):
    manager = make_manager(
        tmp_path,
        {
            "AI测试": {
                "analysis_type": "AI",
                "analyse_model_name": "demo_model",
                "duration": 600,
            }
        },
    )

    info = manager.load_queue_catalog()["AI测试"]

    assert info["available"] is True
    assert info["duration"] == 600
    assert info["analysis_items"] == ["分析项1"]
    assert info["can_auto_judge"] is True


def test_import_rejects_legacy_or_duplicate_project(tmp_path):
    manager = make_manager(tmp_path)
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(
        json.dumps({"name": "旧配置", "sub_configs": []}, ensure_ascii=False),
        encoding="utf-8",
    )

    success, message = manager.import_project(str(legacy_path))
    assert success is False
    assert "项目名称不能为空" in message

    project = make_project(tmp_path)
    success, _file_name = manager.save_project(None, project)
    assert success
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(
        json.dumps(project, ensure_ascii=False),
        encoding="utf-8",
    )

    success, message = manager.import_project(str(duplicate_path))
    assert success is False
    assert message == "项目名称已存在：PB-A01充电宝"
