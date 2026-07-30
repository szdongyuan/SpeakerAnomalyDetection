import os

from base.load_config import LoadUiConfig
from consts import error_code
from consts.running_consts import (
    PRODUCT_TEST_PROGRAM_DIR,
    PRODUCT_TEST_PROGRAM_REGISTRY_PATH,
    SEQUENCE_CONFIG_REGISTRY_PATH,
)


PROGRAM_REGISTRY_FILE = "program_registry.json"
DEFAULT_PROGRAM_NAME = "默认配置"
INVALID_CONFIG_NAME_CHARS = '<>:"/\\|?*'
LIMIT_RULE_ANALYSIS_TYPES = {
    "SPL",
    "SPLF",
    "FR",
    "HD",
    "RB",
    "PRB",
    "FFT",
    "FBA",
    "Loudness",
}


def normalize_config_name(name):
    config_name = str(name or "").strip()
    if config_name.lower().endswith(".json"):
        config_name = config_name[:-5].strip()
    return config_name


def config_file_name(config_name):
    return normalize_config_name(config_name) + ".json"


class ProductTestProgramValidator(object):
    @staticmethod
    def validate_for_save(program_data, registry, current_file):
        errors = []
        if not isinstance(program_data, dict):
            return ["产品测试程序必须是 JSON 对象"]

        name = normalize_config_name(program_data.get("name", ""))
        if not name:
            errors.append("配置名称不能为空")
        elif any(char in name for char in INVALID_CONFIG_NAME_CHARS):
            errors.append("配置名称不能包含以下字符：\\ / : * ? \" < > |")
        elif name.endswith("."):
            errors.append("配置名称不能以句点结尾")
        elif config_file_name(name) == PROGRAM_REGISTRY_FILE:
            errors.append("配置名称不能使用系统保留名称：program_registry")

        sub_configs = program_data.get("sub_configs")
        if not isinstance(sub_configs, list):
            errors.append("sub_configs 必须是列表")
            return errors

        condition_names = set()
        trigger_states = set()
        for index, sub_config in enumerate(sub_configs, 1):
            if not isinstance(sub_config, dict):
                errors.append(f"第 {index} 个子配置格式错误")
                continue

            condition_name = str(sub_config.get("condition_name", "") or "").strip()
            if not condition_name:
                errors.append(f"第 {index} 个子配置的工况名称不能为空")
            elif condition_name in condition_names:
                errors.append(f"工况名称重复：{condition_name}")
            else:
                condition_names.add(condition_name)

            trigger_state = str(sub_config.get("trigger_state", "") or "").strip()
            if trigger_state:
                if trigger_state in trigger_states:
                    errors.append(f"触发状态重复：{trigger_state}")
                else:
                    trigger_states.add(trigger_state)

            test_queue = sub_config.get("test_queue", "")
            if not isinstance(test_queue, str):
                errors.append(f"第 {index} 个子配置的测试队列名称格式错误")

        for item in registry.get("configs", []):
            file_name = str(item.get("file", "") or "")
            registered_name = normalize_config_name(item.get("name", ""))
            if file_name != current_file and registered_name == name:
                errors.append(f"配置名称已存在：{name}")
                break
        return errors

    @staticmethod
    def validate_for_use(program_data, queue_catalog):
        errors = []
        sub_configs = program_data.get("sub_configs", [])
        if not sub_configs:
            return ["至少需要一个子配置"]

        for index, sub_config in enumerate(sub_configs, 1):
            condition_name = str(sub_config.get("condition_name", "") or "").strip()
            trigger_state = str(sub_config.get("trigger_state", "") or "").strip()
            test_queue = str(sub_config.get("test_queue", "") or "").strip()
            row_name = condition_name or f"第 {index} 个子配置"

            if not condition_name:
                errors.append(f"第 {index} 个子配置的工况名称不能为空")
            if not trigger_state:
                errors.append(f"{row_name} 尚未绑定触发状态")
            if not test_queue:
                errors.append(f"{row_name} 尚未选择测试队列")
                continue

            queue_info = queue_catalog.get(test_queue)
            if not queue_info:
                errors.append(f"{row_name} 引用的测试队列不存在：{test_queue}")
            elif not queue_info.get("available", False):
                errors.append(f"{row_name} 引用的测试队列不可用：{test_queue}")
        return errors


class ProductTestProgramConfigManager(object):
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
    def default_program():
        return {"name": DEFAULT_PROGRAM_NAME, "sub_configs": []}

    def load_registry(self):
        load_code, data = LoadUiConfig.load_data_from_json(self.registry_path)
        if load_code != error_code.OK or not isinstance(data, dict):
            return self.rebuild_registry()

        configs = []
        for item in data.get("configs", []):
            if not isinstance(item, dict):
                continue
            file_name = str(item.get("file", "") or "").strip()
            name = str(item.get("name", "") or "").strip()
            file_path = os.path.join(self.program_dir, file_name)
            if (
                self._is_safe_file_name(file_name)
                and name
                and file_name == config_file_name(name)
                and os.path.isfile(file_path)
            ):
                configs.append({"file": file_name, "name": name})

        active_file = str(data.get("active_file", "") or "").strip()
        known_files = {item["file"] for item in configs}
        if active_file not in known_files:
            active_file = None
        return {"active_file": active_file, "configs": configs}

    def rebuild_registry(self):
        os.makedirs(self.program_dir, exist_ok=True)
        configs = []
        for file_name in sorted(os.listdir(self.program_dir)):
            if file_name == PROGRAM_REGISTRY_FILE or not file_name.lower().endswith(".json"):
                continue
            load_code, program_data = self.load_program(file_name)
            if load_code != error_code.OK:
                continue
            name = str(program_data.get("name", "") or "").strip()
            if name and file_name == config_file_name(name):
                configs.append({"file": file_name, "name": name})

        active_file = None
        for item in configs:
            if item["name"] == DEFAULT_PROGRAM_NAME:
                active_file = item["file"]
                break
        if active_file is None and configs:
            active_file = configs[0]["file"]

        registry = {"active_file": active_file, "configs": configs}
        LoadUiConfig.save_data_to_json(registry, self.registry_path)
        return registry

    def save_registry(self, registry):
        return LoadUiConfig.save_data_to_json(registry, self.registry_path)

    def load_program(self, file_name):
        if not self._is_safe_file_name(file_name):
            return error_code.INVALID_DATA_LOADING, "产品测试程序文件名不合法"
        file_path = os.path.join(self.program_dir, file_name)
        load_code, data = LoadUiConfig.load_data_from_json(file_path)
        if load_code != error_code.OK or not isinstance(data, dict):
            return error_code.INVALID_DATA_LOADING, data
        return error_code.OK, data

    def save_program(self, current_file, program_data):
        if current_file and not self._is_safe_file_name(current_file):
            return False, "产品测试程序文件名不合法"

        normalized_program = self._normalize_program(program_data)
        target_file = config_file_name(normalized_program["name"])
        registry = self.load_registry()
        errors = ProductTestProgramValidator.validate_for_save(
            normalized_program,
            registry,
            current_file,
        )
        if errors:
            return False, "\n".join(errors)

        current_path = (
            os.path.join(self.program_dir, current_file)
            if current_file
            else None
        )
        target_path = os.path.join(self.program_dir, target_file)
        if target_file != current_file and os.path.exists(target_path):
            return False, f"配置名称已存在：{normalized_program['name']}"

        if not LoadUiConfig.save_data_to_json(normalized_program, target_path):
            return False, "产品测试程序保存失败"

        if (
            current_path
            and current_path != target_path
            and os.path.isfile(current_path)
        ):
            try:
                os.remove(current_path)
            except OSError as error:
                try:
                    os.remove(target_path)
                except OSError as cleanup_error:
                    return False, (
                        "产品测试程序文件重命名失败，且新文件无法清理："
                        f"{cleanup_error}"
                    )
                return False, f"产品测试程序文件重命名失败：{error}"

        self._replace_registry_entry(
            registry,
            current_file,
            target_file,
            normalized_program["name"],
        )
        registry["active_file"] = target_file
        if not self.save_registry(registry):
            return False, "产品测试程序已保存，但注册表更新失败"
        return True, target_file

    def save_as(self, program_data, new_name):
        copied_program = self._normalize_program(program_data)
        copied_program["name"] = normalize_config_name(new_name)
        return self.save_program(None, copied_program)

    def import_program(self, source_path):
        load_code, program_data = LoadUiConfig.load_data_from_json(source_path)
        if load_code != error_code.OK or not isinstance(program_data, dict):
            return False, "导入文件不是有效的产品测试程序 JSON"
        name = str(program_data.get("name", "") or "").strip()
        return self.save_as(program_data, name)

    def validate_program(self, program_data, current_file, queue_catalog=None):
        registry = self.load_registry()
        if queue_catalog is None:
            queue_catalog = self.load_queue_catalog()
        save_errors = ProductTestProgramValidator.validate_for_save(
            program_data,
            registry,
            current_file,
        )
        use_errors = list(save_errors)
        if not save_errors:
            use_errors.extend(
                ProductTestProgramValidator.validate_for_use(
                    program_data,
                    queue_catalog,
                )
            )
        return {
            "can_save": not save_errors,
            "is_usable": not use_errors,
            "save_errors": save_errors,
            "use_errors": use_errors,
        }

    def load_queue_catalog(self):
        registry = LoadUiConfig._load_sequence_config_registry(self.queue_registry_path)
        catalog = {}
        registry_dir = os.path.dirname(self.queue_registry_path)
        for name, registered_path in registry.items():
            if name == "using_config_path" or not isinstance(registered_path, str):
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
        local_candidates = (
            os.path.join(registry_dir, file_name),
            os.path.join(os.path.dirname(registry_dir), file_name),
        )
        for candidate in local_candidates:
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
        return absolute_path

    def load_trigger_states(self):
        load_code, serial_config = LoadUiConfig.load_serial_discrete_input_config()
        if load_code != error_code.OK or not isinstance(serial_config, dict):
            serial_config = LoadUiConfig.get_default_serial_discrete_input_config()

        decoder_mode = str(serial_config.get("decoder", {}).get("mode", "full_frame"))
        state_map = serial_config.get("state_maps", {}).get(decoder_mode, {})
        trigger_states = []
        for state_code, state_config in state_map.items():
            action = str(state_config.get("action", "") or "")
            if action not in ("idle", "cycle_complete"):
                trigger_states.append(str(state_code))
        return sorted(trigger_states)

    @staticmethod
    def _normalize_program(program_data):
        normalized_sub_configs = []
        for sub_config in program_data.get("sub_configs", []):
            normalized_sub_configs.append(
                {
                    "condition_name": str(sub_config.get("condition_name", "") or "").strip(),
                    "trigger_state": str(sub_config.get("trigger_state", "") or "").strip(),
                    "test_queue": str(sub_config.get("test_queue", "") or "").strip(),
                }
            )
        return {
            "name": normalize_config_name(program_data.get("name", "")),
            "sub_configs": normalized_sub_configs,
        }

    @staticmethod
    def _replace_registry_entry(registry, current_file, target_file, name):
        for item in registry.get("configs", []):
            if item.get("file") == current_file:
                item["file"] = target_file
                item["name"] = name
                return
        registry.setdefault("configs", []).append(
            {"file": target_file, "name": name}
        )

    @staticmethod
    def _load_queue_info(file_path):
        info = {
            "path": file_path,
            "available": False,
            "duration": None,
            "analysis_items": [],
            "reason": "",
        }
        load_code, queue_data = LoadUiConfig.load_data_from_json(file_path)
        if load_code != error_code.OK or not isinstance(queue_data, list) or not queue_data:
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

        acquisition_detail = sequence_data.get("acq", {}).get("detail", {})
        analysis_list = sequence_data.get("analysis_list", {})
        display_sequence = analysis_list.get("display_sequence", [])
        duration = acquisition_detail.get("total_time")
        sample_rate = acquisition_detail.get("sample_rate")
        if not isinstance(duration, (int, float)) or duration <= 0:
            info["reason"] = "录音时长无效"
            return info
        info["duration"] = duration
        if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
            info["reason"] = "采样率无效"
            return info
        if not isinstance(display_sequence, list) or not display_sequence:
            info["reason"] = "未配置分析项"
            return info
        info["analysis_items"] = [str(item) for item in display_sequence]
        for item_name in display_sequence:
            if not isinstance(analysis_list.get(item_name), dict):
                info["reason"] = f"分析项不存在：{item_name}"
                return info
        if not ProductTestProgramConfigManager._has_rule_judgment(
            analysis_list,
            display_sequence,
        ):
            info["reason"] = "未配置可输出 OK/NG 的规则阈值"
            return info

        info["available"] = True
        return info

    @staticmethod
    def _has_rule_judgment(analysis_list, display_sequence):
        for item_name in display_sequence:
            item_config = analysis_list.get(item_name, {})
            analysis_type = str(item_config.get("type", "") or "")
            if analysis_type == "RSC":
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
                analysis_type in LIMIT_RULE_ANALYSIS_TYPES
                and item_config.get("limit_checked")
            ):
                return True
        return False

    @staticmethod
    def _is_safe_file_name(file_name):
        if not isinstance(file_name, str) or not file_name.lower().endswith(".json"):
            return False
        return os.path.basename(file_name) == file_name and file_name != PROGRAM_REGISTRY_FILE
