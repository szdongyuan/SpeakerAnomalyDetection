"""Prepare and compare product configurations without touching runtime or widgets."""

import json
from dataclasses import dataclass

from base.analysis_segments import normalize_segmented_analysis
from base.product_test_project_config import flatten_test_conditions, iter_test_conditions
from consts import error_code
from consts.product_test_project_consts import (
    EXPORT_RAW_AUDIO_CSV_KEY,
    PROJECT_NAME_KEY,
    RESULT_ROOT_DIRECTORY_KEY,
    TEST_QUEUE_KEY,
)


@dataclass(frozen=True)
class ProductTestRefreshSnapshot:
    active_file: str
    signature: str
    conditions: list
    context: dict
    queue_catalog: dict
    validation: dict


def build_product_test_refresh_snapshot(manager, active_file):
    """Read each referenced queue once; the signature stays separate from live data."""
    active_file = str(active_file or "").strip()
    if not active_file:
        return ProductTestRefreshSnapshot("", "", [], {}, {}, {})

    load_code, project = manager.load_project(active_file)
    if load_code != error_code.OK or not isinstance(project, dict):
        raise ValueError(f"产品配置无法读取：{active_file}")
    queue_names = {
        str(condition.get(TEST_QUEUE_KEY) or "").strip()
        for _, _, _, condition in iter_test_conditions(project)
    }
    catalog = manager.load_queue_catalog(queue_names=queue_names)
    validation = manager.validate_project(project, active_file, queue_catalog=catalog)
    if not validation["is_usable"]:
        raise ValueError("\n".join(validation["use_errors"]) or "产品配置不可用")
    for name, info in catalog.items():
        if not isinstance(info["data"][0].get("seq1"), dict):
            raise ValueError(f"测试队列缺少 seq1：{name}")

    project = manager._normalize_project(project)
    conditions = flatten_test_conditions(project)
    # Opening and saving legacy rows materializes these defaults in the editor.
    # Their effective value is unchanged even when the old file omitted the keys.
    for _, _, _, condition in iter_test_conditions(project):
        condition.setdefault("input_voltage", "")
        condition["segmented_analysis"] = normalize_segmented_analysis(condition)
    for condition in conditions:
        condition["display_name"] = (
            f"{condition['group_name']} / {condition['condition_name']}"
        )
    context = {
        "project_name": project[PROJECT_NAME_KEY],
        "result_root_directory": project[RESULT_ROOT_DIRECTORY_KEY],
        EXPORT_RAW_AUDIO_CSV_KEY: project[EXPORT_RAW_AUDIO_CSV_KEY],
        "active_file": active_file,
    }
    signature = json.dumps(
        {
            "active_file": active_file,
            "project": project,
            "queues": {
                name: {"path": info["path"], "data": info["data"]}
                for name, info in catalog.items()
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return ProductTestRefreshSnapshot(
        active_file, signature, conditions, context, catalog, validation
    )
