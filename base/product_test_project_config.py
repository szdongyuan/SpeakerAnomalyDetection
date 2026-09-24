"""Project-level product test configuration management."""

import copy
import math
import ntpath
import os

from base.analysis_segments import segment_condition_fields, normalize_segmented_analysis, segment_count
from base.hardware_trigger.serial_full_frame_matcher import normalize_hex_frame
from base.hardware_trigger.serial_product_port_plan import build_serial_product_port_plan
from base.load_config import LoadUiConfig
from base.analysis_config_validation import validate_analysis_config
from base.recording_preview_config import resolve_recording_preview_time_mode
from consts import error_code
from consts.product_test_project_consts import (
    CONDITION_NAME_KEY,
    EXPORT_RAW_AUDIO_CSV_KEY,
    GROUP_NAME_KEY,
    INVALID_PROJECT_NAME_CHARS,
    LIMIT_RULE_ANALYSIS_TYPES,
    OUTPUT_LOAD_KEY,
    PRODUCT_TRIGGER_MODE_MANUAL,
    PRODUCT_TRIGGER_MODE_MIXED,
    PRODUCT_TRIGGER_MODE_SERIAL,
    PROGRAM_REGISTRY_FILE,
    PROJECT_NAME_KEY,
    REGISTRY_ACTIVE_FILE_KEY,
    REGISTRY_CONFIGS_KEY,
    REGISTRY_FILE_KEY,
    RESULT_ROOT_DIRECTORY_KEY,
    TEST_CONDITIONS_KEY,
    TEST_GROUPS_KEY,
    TEST_QUEUE_KEY,
    TRIGGER_STATE_KEY,
)
from consts.running_consts import (
    PRODUCT_TEST_PROGRAM_DIR,
    PRODUCT_TEST_PROGRAM_REGISTRY_PATH,
    SEQUENCE_CONFIG_REGISTRY_PATH,
)


def normalize_project_name(name):
    project_name = str(name or "").strip()
    if project_name.lower().endswith(".json"):
        project_name = project_name[:-5].strip()
    return project_name


def project_file_name(project_name):
    return normalize_project_name(project_name) + ".json"


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


class ProductTestProjectValidator(object):
    @staticmethod
    def validate_test_queue_references(project_data, queue_catalog):
        errors = []
        for group_index, group, condition_index, condition in iter_test_conditions(
            project_data
        ):
            location = ProductTestProjectValidator.condition_location(
                group_index,
                group,
                condition_index,
                condition,
            )
            test_queue_value = condition.get(TEST_QUEUE_KEY, "")
            if not isinstance(test_queue_value, str):
                continue
            test_queue = test_queue_value.strip()
            if not test_queue:
                continue
            queue_info = queue_catalog.get(test_queue)
            if not queue_info:
                errors.append(f"{location}引用的测试队列不存在：{test_queue}")
            elif not queue_info.get("available", False):
                errors.append(f"{location}引用的测试队列不可用：{test_queue}")
            if isinstance(queue_info, dict) and queue_info.get("available"):
                try:
                    segment_count(normalize_segmented_analysis(condition), queue_info.get("duration"))
                except ValueError as error:
                    errors.append(str(error))
        return errors

    @staticmethod
    def validate_for_save(project_data, registry, current_file, *, port_switch_idle_code=""):
        if not isinstance(project_data, dict):
            return ["产品测试配置必须是 JSON 对象"]

        errors = []
        project_name = normalize_project_name(
            project_data.get(PROJECT_NAME_KEY, "")
        )
        if not project_name:
            errors.append("项目名称不能为空")
        elif any(char in project_name for char in INVALID_PROJECT_NAME_CHARS):
            errors.append('项目名称不能包含以下字符：\\ / : * ? " < > | _')
        elif project_name.endswith("."):
            errors.append("项目名称不能以句点结尾")
        elif project_file_name(project_name).lower() == PROGRAM_REGISTRY_FILE:
            errors.append("项目名称不能使用系统保留名称：program_registry")

        result_root_value = project_data.get(RESULT_ROOT_DIRECTORY_KEY, "")
        if not isinstance(result_root_value, str) or not result_root_value.strip():
            errors.append("测试结果根目录不能为空")
        elif not ntpath.isabs(result_root_value.strip()):
            errors.append("测试结果根目录必须是绝对路径")

        export_raw_audio_csv = project_data.get(EXPORT_RAW_AUDIO_CSV_KEY, False)
        if type(export_raw_audio_csv) is not bool:
            errors.append("原始音频 CSV 保存配置必须是布尔值")

        groups = project_data.get(TEST_GROUPS_KEY)
        if not isinstance(groups, list):
            errors.append("test_groups 必须是列表")
            return errors
        if not groups:
            errors.append("至少需要配置一个端口")

        group_names = set()
        all_conditions = []
        for group_index, group in enumerate(groups, 1):
            trigger_states = set()
            if not isinstance(group, dict):
                errors.append(f"第 {group_index} 个端口格式错误")
                continue
            group_name = str(group.get(GROUP_NAME_KEY, "") or "").strip()
            group_label = group_name or f"第 {group_index} 个端口"
            if not group_name:
                errors.append(f"第 {group_index} 个端口名称不能为空")
            elif group_name in group_names:
                errors.append(f"端口名称重复：{group_name}")
            else:
                group_names.add(group_name)

            conditions = group.get(TEST_CONDITIONS_KEY)
            if not isinstance(conditions, list):
                errors.append(f"{group_label}的 test_conditions 必须是列表")
                continue
            if not conditions:
                errors.append(f"{group_label}至少需要配置一个工况")
                continue

            condition_names = set()
            for condition_index, condition in enumerate(conditions, 1):
                if not isinstance(condition, dict):
                    errors.append(f"{group_label}的第 {condition_index} 个工况格式错误")
                    continue
                all_conditions.append(condition)
                condition_name = str(
                    condition.get(CONDITION_NAME_KEY, "") or ""
                ).strip()
                row_name = condition_name or f"第 {condition_index} 个工况"
                location = f"{group_label}/{row_name}"
                if not condition_name:
                    errors.append(f"{group_label}的第 {condition_index} 个工况名称不能为空")
                elif condition_name in condition_names:
                    errors.append(
                        f"{group_label}中的工况名称重复：{condition_name}"
                    )
                else:
                    condition_names.add(condition_name)

                trigger_value = condition.get(TRIGGER_STATE_KEY, "")
                if not isinstance(trigger_value, str):
                    errors.append(f"{location}的状态码必须是字符串")
                else:
                    trigger_state = normalize_trigger_state(trigger_value)
                    if trigger_state:
                        try:
                            trigger_state = normalize_hex_frame(trigger_state)
                        except ValueError as error:
                            errors.append(f"{location}的状态码格式错误：{error}")
                        else:
                            if trigger_state in trigger_states:
                                errors.append(f"状态码重复：{trigger_state}")
                            else:
                                trigger_states.add(trigger_state)

                try:
                    segment_condition_fields(condition)
                except ValueError as error:
                    errors.append(str(error))

                if OUTPUT_LOAD_KEY in condition:
                    load = condition[OUTPUT_LOAD_KEY]
                    if not isinstance(load, dict):
                        errors.append(f"{location}的输出负载配置必须是对象")
                    else:
                        values = load.get("values")
                        seconds = load.get("analysis_seconds")
                        if type(load.get("enabled")) is not bool:
                            errors.append(f"{location}的输出负载启用状态必须是布尔值")
                        if not isinstance(values, list) or any(
                            type(value) not in (int, float)
                            or not math.isfinite(value)
                            or value < 0
                            for value in values
                        ):
                            errors.append(f"{location}的输出负载必须是非负数列表")
                        elif load.get("enabled") and not values:
                            errors.append(f"{location}的输出负载不能为空")
                        if (
                            type(seconds) not in (int, float)
                            or not math.isfinite(seconds)
                            or seconds <= 0
                        ):
                            errors.append(f"{location}的每段分析时长必须大于零")

                test_queue = condition.get(TEST_QUEUE_KEY, "")
                if not isinstance(test_queue, str):
                    errors.append(f"{location}的测试队列名称格式错误")
                elif not test_queue.strip():
                    errors.append(f"{location}尚未选择测试队列")

        if (
            all_conditions
            and classify_project_trigger_mode(all_conditions)
            == PRODUCT_TRIGGER_MODE_MIXED
        ):
            errors.append("所有工况状态码必须全部配置或全部留空")

        if (
            not errors and all_conditions
            and classify_project_trigger_mode(all_conditions) != PRODUCT_TRIGGER_MODE_MANUAL
        ):
            try:
                build_serial_product_port_plan(
                    flatten_test_conditions(project_data), port_switch_idle_code
                )
            except ValueError as error:
                errors.append(str(error))

        for item in (registry or {}).get(REGISTRY_CONFIGS_KEY, []):
            if not isinstance(item, dict):
                continue
            file_name = str(item.get(REGISTRY_FILE_KEY, "") or "").strip()
            registered_name = normalize_project_name(
                item.get(PROJECT_NAME_KEY, "")
            )
            if (
                file_name.casefold() != str(current_file or "").casefold()
                and registered_name.casefold() == project_name.casefold()
            ):
                errors.append(f"项目名称已存在：{project_name}")
                break
        return errors

    @staticmethod
    def validate_for_use(project_data, queue_catalog):
        if not flatten_test_conditions(project_data):
            return ["至少需要配置一个工况"]
        return ProductTestProjectValidator.validate_test_queue_references(
            project_data,
            queue_catalog,
        )

    @staticmethod
    def condition_location(
        group_index,
        group,
        condition_index,
        condition,
    ):
        group_name = str(group.get(GROUP_NAME_KEY, "") or "").strip()
        condition_name = str(condition.get(CONDITION_NAME_KEY, "") or "").strip()
        return (
            f"{group_name or f'第 {group_index} 个端口'}/"
            f"{condition_name or f'第 {condition_index} 个工况'}"
        )


class ProductTestProjectConfigManager(object):
    def __init__(
        self,
        program_dir=PRODUCT_TEST_PROGRAM_DIR,
        registry_path=PRODUCT_TEST_PROGRAM_REGISTRY_PATH,
        queue_registry_path=SEQUENCE_CONFIG_REGISTRY_PATH,
    ):
        self.program_dir = os.path.abspath(program_dir)
        self.registry_path = os.path.abspath(registry_path)
        self.queue_registry_path = os.path.abspath(queue_registry_path)

    @staticmethod
    def default_project():
        return {
            PROJECT_NAME_KEY: "",
            RESULT_ROOT_DIRECTORY_KEY: "",
            EXPORT_RAW_AUDIO_CSV_KEY: False,
            TEST_GROUPS_KEY: [
                {
                    GROUP_NAME_KEY: "新端口1",
                    TEST_CONDITIONS_KEY: [
                        {
                            CONDITION_NAME_KEY: "档位1",
                            TRIGGER_STATE_KEY: "",
                            TEST_QUEUE_KEY: "",
                        }
                    ],
                }
            ],
        }

    def load_registry(self):
        load_code, data = LoadUiConfig.load_data_from_json(self.registry_path)
        if load_code != error_code.OK or not isinstance(data, dict):
            return {
                REGISTRY_ACTIVE_FILE_KEY: None,
                REGISTRY_CONFIGS_KEY: [],
            }

        configs = []
        for item in data.get(REGISTRY_CONFIGS_KEY, []):
            if not isinstance(item, dict):
                continue
            file_name = str(item.get(REGISTRY_FILE_KEY, "") or "").strip()
            project_name = str(item.get(PROJECT_NAME_KEY, "") or "").strip()
            if (
                self._is_safe_file_name(file_name)
                and project_name
                and file_name == project_file_name(project_name)
            ):
                configs.append(
                    {REGISTRY_FILE_KEY: file_name, PROJECT_NAME_KEY: project_name}
                )

        active_file = (
            str(data.get(REGISTRY_ACTIVE_FILE_KEY, "") or "").strip() or None
        )
        known_files = {item[REGISTRY_FILE_KEY] for item in configs}
        if active_file not in known_files:
            active_file = None
        return {
            REGISTRY_ACTIVE_FILE_KEY: active_file,
            REGISTRY_CONFIGS_KEY: configs,
        }

    def save_registry(self, registry):
        normalized = {
            REGISTRY_ACTIVE_FILE_KEY: (
                registry or {}
            ).get(REGISTRY_ACTIVE_FILE_KEY) or None,
            REGISTRY_CONFIGS_KEY: [
                {
                    REGISTRY_FILE_KEY: str(
                        item.get(REGISTRY_FILE_KEY, "") or ""
                    ).strip(),
                    PROJECT_NAME_KEY: str(
                        item.get(PROJECT_NAME_KEY, "") or ""
                    ).strip(),
                }
                for item in (registry or {}).get(REGISTRY_CONFIGS_KEY, [])
                if isinstance(item, dict)
            ],
        }
        return LoadUiConfig.save_data_to_json(normalized, self.registry_path)

    def load_project(self, file_name):
        if not self._is_safe_file_name(file_name):
            return error_code.INVALID_DATA_LOADING, "产品测试配置文件名不合法"
        file_path = os.path.join(self.program_dir, file_name)
        if not os.path.isfile(file_path):
            return (
                error_code.INVALID_DATA_LOADING,
                "产品测试配置文件不存在，请选择其他配置。",
            )
        load_code, data = LoadUiConfig.load_data_from_json(file_path)
        if load_code != error_code.OK or not isinstance(data, dict):
            return error_code.INVALID_DATA_LOADING, data
        return error_code.OK, data

    def save_project(self, current_file, project_data):
        if current_file and not self._is_safe_file_name(current_file):
            return False, "产品测试配置文件名不合法"
        if current_file:
            current_file = self.existing_project_file(current_file) or current_file

        registry = self.load_registry()
        errors = self._collect_save_errors(project_data, registry, current_file)
        if errors:
            return False, "\n".join(errors)

        normalized_project = self._normalize_project(project_data)
        project_name = normalized_project[PROJECT_NAME_KEY]
        target_file = project_file_name(project_name)
        # A case-only change updates the same project, keeping its saved spelling.
        if current_file and target_file.casefold() == current_file.casefold():
            target_file = current_file
            project_name = os.path.splitext(current_file)[0]
            normalized_project[PROJECT_NAME_KEY] = project_name
        current_path = (
            os.path.join(self.program_dir, current_file) if current_file else None
        )
        target_path = os.path.join(self.program_dir, target_file)
        if target_file != current_file and (
            self.existing_project_file(project_name) or os.path.exists(target_path)
        ):
            return False, f"项目名称已存在：{project_name}"

        try:
            os.makedirs(
                os.path.join(
                    normalized_project[RESULT_ROOT_DIRECTORY_KEY],
                    project_name,
                ),
                exist_ok=True,
            )
        except OSError as error:
            return False, f"无法创建项目测试结果目录：{error}"

        try:
            target_snapshot = self._read_file_snapshot(target_path)
        except OSError as error:
            return False, f"无法读取原产品测试配置：{error}"
        if not LoadUiConfig.save_data_to_json(normalized_project, target_path):
            return False, "产品测试配置保存失败"

        original_registry = self._copy_registry(registry)
        active_file_before_save = registry.get(REGISTRY_ACTIVE_FILE_KEY)
        self._replace_registry_entry(
            registry,
            current_file,
            target_file,
            project_name,
        )
        if not active_file_before_save or active_file_before_save == current_file:
            registry[REGISTRY_ACTIVE_FILE_KEY] = target_file
        if not self.save_registry(registry):
            restored = self._restore_file_snapshot(target_path, target_snapshot)
            message = "产品测试配置保存失败：注册表更新失败"
            if not restored:
                message += "，且原配置恢复失败"
            return False, message

        if current_path and current_path != target_path and os.path.isfile(current_path):
            try:
                os.remove(current_path)
            except OSError as error:
                self.save_registry(original_registry)
                restored = self._restore_file_snapshot(target_path, target_snapshot)
                message = f"产品测试配置文件重命名失败：{error}"
                if not restored:
                    message += "，且目标文件恢复失败"
                return False, message
        return True, target_file

    def save_as(self, project_data, new_name):
        if not isinstance(project_data, dict):
            return False, "产品测试配置必须是 JSON 对象"
        copied_project = self._normalize_project(project_data)
        copied_project[PROJECT_NAME_KEY] = normalize_project_name(new_name)
        return self.save_project(None, copied_project)

    def import_project(self, source_path):
        """Read an import draft; persistence is handled only by save_project."""
        load_code, project_data = LoadUiConfig.load_data_from_json(source_path)
        if load_code != error_code.OK or not isinstance(project_data, dict):
            return False, "导入文件不是有效的产品测试配置 JSON"
        errors = ProductTestProjectValidator.validate_for_save(
            project_data, {}, None
        )
        if errors:
            return False, "\n".join(errors)
        return True, project_data

    def existing_project_file(self, project_name):
        """Find the managed save target, including a missing registered file."""
        target_file = project_file_name(project_name)
        if not self._is_safe_file_name(target_file):
            return None
        for item in self.load_registry().get(REGISTRY_CONFIGS_KEY, []):
            if item[REGISTRY_FILE_KEY].casefold() == target_file.casefold():
                return item[REGISTRY_FILE_KEY]
        if os.path.isdir(self.program_dir):
            with os.scandir(self.program_dir) as entries:
                for entry in entries:
                    if entry.name.casefold() == target_file.casefold() and entry.is_file():
                        return entry.name
        return None

    def delete_project(self, file_name):
        if not self._is_safe_file_name(file_name):
            return False, "产品测试配置文件名不合法"
        registry = self.load_registry()
        original_registry = self._copy_registry(registry)
        registry[REGISTRY_CONFIGS_KEY] = [
            item
            for item in registry.get(REGISTRY_CONFIGS_KEY, [])
            if item.get(REGISTRY_FILE_KEY) != file_name
        ]
        if registry.get(REGISTRY_ACTIVE_FILE_KEY) == file_name:
            registry[REGISTRY_ACTIVE_FILE_KEY] = None
        if not self.save_registry(registry):
            return False, "删除配置失败：注册表更新失败"

        file_path = os.path.join(self.program_dir, file_name)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except OSError as error:
                self.save_registry(original_registry)
                return False, f"删除产品测试配置文件失败：{error}"
        return True, file_name

    def validate_project(self, project_data, current_file, queue_catalog=None):
        registry = self.load_registry()
        if queue_catalog is None:
            queue_catalog = self.load_queue_catalog()
        save_errors = self._collect_save_errors(
            project_data,
            registry,
            current_file,
            queue_catalog,
        )
        use_errors = list(save_errors)
        if not save_errors:
            use_errors.extend(
                ProductTestProjectValidator.validate_for_use(
                    project_data,
                    queue_catalog,
                )
            )

        use_warnings = []
        test_mode_errors = []
        if not save_errors:
            for group_index, group, condition_index, condition in iter_test_conditions(
                project_data
            ):
                test_queue = str(
                    condition.get(TEST_QUEUE_KEY, "") or ""
                ).strip()
                queue_info = queue_catalog.get(test_queue)
                if (
                    queue_info
                    and queue_info.get("available", False)
                    and not queue_info.get("can_auto_judge", False)
                ):
                    location = ProductTestProjectValidator.condition_location(
                        group_index,
                        group,
                        condition_index,
                        condition,
                    )
                    reason = str(queue_info.get("judgment_reason", "") or "")
                    warning = (
                        f"{location}的测试队列不能自动输出 OK/NG："
                        f"{test_queue}（{reason}）"
                    )
                    use_warnings.append(warning)
        return {
            "can_save": not save_errors,
            "is_usable": not use_errors,
            "is_test_mode_usable": not use_errors,
            "save_errors": save_errors,
            "use_errors": use_errors,
            "use_warnings": use_warnings,
            "test_mode_errors": test_mode_errors,
        }

    def _collect_save_errors(
        self,
        project_data,
        registry,
        current_file,
        queue_catalog=None,
    ):
        _serial_error, serial_config = LoadUiConfig.load_serial_discrete_input_config()
        idle_code = (
            serial_config.get("port_switch_idle_code", "")
            if isinstance(serial_config, dict) else ""
        )
        errors = ProductTestProjectValidator.validate_for_save(
            project_data,
            registry,
            current_file,
            port_switch_idle_code=idle_code,
        )
        if errors:
            return errors
        if queue_catalog is None:
            queue_catalog = self.load_queue_catalog()
        errors.extend(
            ProductTestProjectValidator.validate_test_queue_references(
                project_data,
                queue_catalog,
            )
        )
        return errors

    def load_queue_catalog(self, queue_names=None):
        registry = LoadUiConfig._load_sequence_config_registry(
            self.queue_registry_path
        )
        catalog = {}
        registry_dir = os.path.dirname(self.queue_registry_path)
        for name, registered_path in registry.items():
            if name == "using_config_path" or not isinstance(registered_path, str):
                continue
            if queue_names is not None and name not in queue_names:
                continue
            file_path = self._resolve_queue_path(registered_path, registry_dir)
            catalog[name] = self._load_queue_info(file_path)
        return catalog

    @staticmethod
    def _resolve_queue_path(registered_path, registry_dir):
        if not os.path.isabs(registered_path):
            return os.path.abspath(os.path.join(registry_dir, registered_path))
        absolute_path = os.path.abspath(registered_path)
        if os.path.isfile(absolute_path):
            return absolute_path
        file_name = os.path.basename(absolute_path)
        for candidate in (
            os.path.join(registry_dir, file_name),
            os.path.join(os.path.dirname(registry_dir), file_name),
        ):
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
        return absolute_path

    @staticmethod
    def _normalize_project(project_data):
        test_groups = []
        for group in project_data.get(TEST_GROUPS_KEY, []):
            if not isinstance(group, dict):
                continue
            conditions = []
            for condition in group.get(TEST_CONDITIONS_KEY, []):
                if not isinstance(condition, dict):
                    continue
                conditions.append(
                    {
                        CONDITION_NAME_KEY: str(
                            condition.get(CONDITION_NAME_KEY, "") or ""
                        ).strip(),
                        TRIGGER_STATE_KEY: normalize_optional_hex_frame(
                            condition.get(TRIGGER_STATE_KEY, "")
                        ),
                        TEST_QUEUE_KEY: str(
                            condition.get(TEST_QUEUE_KEY, "") or ""
                        ).strip(),
                        **(
                            segment_condition_fields(condition)
                        ),
                    }
                )
            test_groups.append(
                {
                    GROUP_NAME_KEY: str(
                        group.get(GROUP_NAME_KEY, "") or ""
                    ).strip(),
                    TEST_CONDITIONS_KEY: conditions,
                }
            )
        result_root = str(
            project_data.get(RESULT_ROOT_DIRECTORY_KEY, "") or ""
        ).strip()
        return {
            PROJECT_NAME_KEY: normalize_project_name(
                project_data.get(PROJECT_NAME_KEY, "")
            ),
            RESULT_ROOT_DIRECTORY_KEY: ntpath.normpath(result_root),
            EXPORT_RAW_AUDIO_CSV_KEY: (
                project_data.get(EXPORT_RAW_AUDIO_CSV_KEY, False) is True
            ),
            TEST_GROUPS_KEY: test_groups,
        }

    @staticmethod
    def _replace_registry_entry(
        registry,
        current_file,
        target_file,
        project_name,
    ):
        for item in registry.get(REGISTRY_CONFIGS_KEY, []):
            if item.get(REGISTRY_FILE_KEY) == current_file:
                item[REGISTRY_FILE_KEY] = target_file
                item[PROJECT_NAME_KEY] = project_name
                return
        registry.setdefault(REGISTRY_CONFIGS_KEY, []).append(
            {REGISTRY_FILE_KEY: target_file, PROJECT_NAME_KEY: project_name}
        )

    @staticmethod
    def _load_queue_info(file_path):
        load_code, queue_data = LoadUiConfig.load_data_from_json(file_path)
        return ProductTestProjectConfigManager._queue_info_from_data(
            file_path, queue_data if load_code == error_code.OK else None
        )

    @staticmethod
    def _queue_info_from_data(file_path, queue_data):
        info = {
            "path": file_path,
            "available": False,
            "acquisition_mode": "",
            "duration": None,
            "analysis_items": [],
            "reason": "",
            "can_auto_judge": False,
            "judgment_reason": "",
        }
        if not isinstance(queue_data, list) or not queue_data:
            info["reason"] = "测试队列文件无法读取"
            return info
        first_sequence_group = queue_data[0]
        if not isinstance(first_sequence_group, dict) or not first_sequence_group:
            info["reason"] = "测试队列没有可执行序列"
            return info
        sequence_data = next(iter(first_sequence_group.values()))
        if not isinstance(sequence_data, dict):
            info["reason"] = "测试队列序列格式错误"
            return info

        acquisition = sequence_data.get("acq", {})
        acquisition_mode = str(acquisition.get("mode") or "RECORD_ONLY").strip().upper()
        info["acquisition_mode"] = acquisition_mode
        acquisition_detail = acquisition.get("detail", {})
        analysis_list = sequence_data.get("analysis_list", {})
        info["data"] = queue_data
        info["analysis_list"] = analysis_list
        try:
            validate_analysis_config(analysis_list)
        except ValueError as exc:
            info["reason"] = str(exc)
            return info
        display_sequence = analysis_list.get("display_sequence", [])
        if acquisition_mode != "RECORD_ONLY":
            info["reason"] = f"不支持的采集模式：{acquisition_mode or '-'}"
            return info
        try:
            resolve_recording_preview_time_mode(acquisition_detail)
        except ValueError as exc:
            info["reason"] = str(exc)
            return info
        duration = acquisition_detail.get("total_time")
        sample_rate = acquisition_detail.get("sample_rate")
        if not isinstance(duration, (int, float)) or duration <= 0:
            info["reason"] = "录音时长无效"
            return info
        info["duration"] = duration
        if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
            info["reason"] = "采样率无效"
            return info
        if isinstance(display_sequence, list):
            display_sequence = [
                name for name in display_sequence
                if not (
                    isinstance(analysis_list.get(name), dict)
                    and analysis_list[name].get("type") == "Excel"
                )
            ]
        if not isinstance(display_sequence, list) or not display_sequence:
            info["reason"] = "未配置分析项"
            return info
        info["analysis_items"] = [str(item) for item in display_sequence]
        for item_name in display_sequence:
            if not isinstance(analysis_list.get(item_name), dict):
                info["reason"] = f"分析项不存在：{item_name}"
                return info
        info["available"] = True
        if ProductTestProjectConfigManager._has_automatic_judgment(
            analysis_list,
            display_sequence,
        ):
            info["can_auto_judge"] = True
        else:
            info["judgment_reason"] = "未配置可输出 OK/NG 的规则阈值"
        return info

    @staticmethod
    def _has_automatic_judgment(analysis_list, display_sequence):
        for item_name in display_sequence:
            item_config = analysis_list.get(item_name, {})
            normalized_type = str(item_config.get("type", "") or "").upper()
            if normalized_type == "RSC":
                has_reference = bool(
                    str(item_config.get("reference_source_path", "") or "").strip()
                )
                current_only = bool(
                    item_config.get("view_current_only_without_reference", False)
                )
                threshold_enabled = bool(
                    item_config.get("enable_threshold_judgment", True)
                )
                if has_reference and not current_only and threshold_enabled:
                    return True
            elif (
                normalized_type in LIMIT_RULE_ANALYSIS_TYPES
                and item_config.get("limit_checked")
            ):
                return True
        return False

    @staticmethod
    def _copy_registry(registry):
        return {
            REGISTRY_ACTIVE_FILE_KEY: (registry or {}).get(
                REGISTRY_ACTIVE_FILE_KEY
            ),
            REGISTRY_CONFIGS_KEY: [
                dict(item)
                for item in (registry or {}).get(REGISTRY_CONFIGS_KEY, [])
            ],
        }

    @staticmethod
    def _read_file_snapshot(file_path):
        if not os.path.isfile(file_path):
            return None
        with open(file_path, "rb") as file_handle:
            return file_handle.read()

    @staticmethod
    def _restore_file_snapshot(file_path, snapshot):
        if snapshot is None:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError:
                return False
            return True
        temp_path = file_path + ".rollback"
        try:
            with open(temp_path, "wb") as file_handle:
                file_handle.write(snapshot)
            os.replace(temp_path, file_path)
            return True
        except OSError:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            return False

    @staticmethod
    def _is_safe_file_name(file_name):
        if not isinstance(file_name, str) or not file_name.lower().endswith(".json"):
            return False
        return (
            os.path.basename(file_name) == file_name
            and file_name.lower() != PROGRAM_REGISTRY_FILE
        )
