"""Legacy execution policies, excluded from the product configuration editor."""
from base.analysis_segments import segment_condition_fields
from base.hardware_trigger.serial_full_frame_matcher import normalize_hex_frame
from consts.product_test_project_consts import (
    TEST_GROUPS_KEY, TEST_CONDITIONS_KEY, GROUP_NAME_KEY, CONDITION_NAME_KEY,
    TRIGGER_STATE_KEY, TEST_QUEUE_KEY, PRODUCT_TRIGGER_MODE_MANUAL,
    PRODUCT_TRIGGER_MODE_SERIAL, PRODUCT_TRIGGER_MODE_MIXED,
)

def normalize_trigger_state(value):
    return " ".join(str(value or "").strip().upper().split())

def normalize_optional_hex_frame(value):
    normalized = normalize_trigger_state(value)
    if not normalized:
        return ""
    try:
        return normalize_hex_frame(normalized)
    except ValueError:
        return normalized

def iter_test_conditions(project_data):
    if not isinstance(project_data, dict):
        return
    for group_index, group in enumerate(project_data.get(TEST_GROUPS_KEY, []), 1):
        if not isinstance(group, dict):
            continue
        for condition_index, condition in enumerate(
            group.get(TEST_CONDITIONS_KEY, []),
            1,
        ):
            if isinstance(condition, dict):
                yield group_index, group, condition_index, condition

def flatten_test_conditions(project_data):
    result = []
    for group_index, group, condition_index, condition in iter_test_conditions(
        project_data
    ):
        group_name = str(group.get(GROUP_NAME_KEY, "") or "").strip()
        condition_name = str(condition.get(CONDITION_NAME_KEY, "") or "").strip()
        result.append(
            {
                "key": f"group_{group_index}:condition_{condition_index}",
                GROUP_NAME_KEY: group_name,
                **segment_condition_fields(condition),
                CONDITION_NAME_KEY: condition_name,
                TRIGGER_STATE_KEY: normalize_trigger_state(
                    condition.get(TRIGGER_STATE_KEY, "")
                ),
                TEST_QUEUE_KEY: str(
                    condition.get(TEST_QUEUE_KEY, "") or ""
                ).strip(),
            }
        )
    return result

def classify_project_trigger_mode(conditions_or_project):
    if isinstance(conditions_or_project, dict):
        conditions = flatten_test_conditions(conditions_or_project)
    else:
        conditions = conditions_or_project or []
    trigger_flags = [
        bool(normalize_trigger_state(item.get(TRIGGER_STATE_KEY, "")))
        for item in conditions
        if isinstance(item, dict)
    ]
    if not any(trigger_flags):
        return PRODUCT_TRIGGER_MODE_MANUAL
    if all(trigger_flags):
        return PRODUCT_TRIGGER_MODE_SERIAL
    return PRODUCT_TRIGGER_MODE_MIXED

def is_manual_project_play_allowed(conditions_or_project, *, serial_enabled=True):
    trigger_mode = classify_project_trigger_mode(conditions_or_project)
    return trigger_mode == PRODUCT_TRIGGER_MODE_MANUAL or (
        trigger_mode == PRODUCT_TRIGGER_MODE_SERIAL and not serial_enabled
    )
