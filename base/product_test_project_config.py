"""Product configuration service. Persistence reuses the existing JSON utilities."""
import copy
import os
from pathlib import PureWindowsPath

from base.load_config import LoadUiConfig
from base.product_test.config_model import (
    RUNTIME_UNAVAILABLE, copy_project, default_project, hardware_channels, normalize_project,
)
from base.product_test.config_validation import validate_structure, validate_references
from base.test_queue_catalog import load_queue_info, resolve_queue_path
from consts import error_code
from consts.running_consts import (
    PRODUCT_TEST_PROGRAM_DIR, PRODUCT_TEST_PROGRAM_REGISTRY_PATH, SEQUENCE_CONFIG_REGISTRY_PATH,
)
from consts.product_test_project_consts import PROGRAM_REGISTRY_FILE


def iter_test_conditions(project):
    """Reference-scanner view; it does not expand conditions into execution tasks."""
    names = {s["station_id"]: s["station_name"] for s in project["stations"]}
    for index, condition in enumerate(project["conditions"], 1):
        group = {"group_name": "、".join(names.get(key, key) for key in condition["applicable_station_ids"])}
        yield 0, group, index, condition


class ProductTestProjectConfigManager:
    def __init__(self, program_dir=PRODUCT_TEST_PROGRAM_DIR,
                 registry_path=PRODUCT_TEST_PROGRAM_REGISTRY_PATH,
                 queue_registry_path=SEQUENCE_CONFIG_REGISTRY_PATH, *, input_device_provider=None):
        self.program_dir = os.path.abspath(program_dir)
        self.registry_path = os.path.abspath(registry_path)
        self.queue_registry_path = os.path.abspath(queue_registry_path)
        self.input_device_provider = input_device_provider

    default_project = staticmethod(default_project)

    def load_registry(self):
        if not os.path.exists(self.registry_path):
            return {"active_file": None, "configs": []}
        code, data = LoadUiConfig.load_data_from_json(self.registry_path)
        if code != error_code.OK or not isinstance(data, dict) or not isinstance(data.get("configs"), list):
            raise ValueError("配置列表损坏，未创建或覆盖任何配置")
        seen = set()
        project_ids = set()
        for item in data["configs"]:
            if (not isinstance(item, dict) or not self._is_safe_file_name(item.get("file"))
                    or not isinstance(item.get("project_name", item.get("name")), str)):
                raise ValueError("配置列表包含无效记录")
            if item["file"].casefold() in seen:
                raise ValueError("配置列表包含重复文件")
            seen.add(item["file"].casefold())
            project_id = item.get("project_id")
            if project_id is not None:
                if not isinstance(project_id, str) or not project_id or project_id in project_ids:
                    raise ValueError("配置列表包含无效或重复项目 ID")
                project_ids.add(project_id)
        return data

    def save_registry(self, registry):
        return LoadUiConfig.save_data_to_json(registry, self.registry_path)

    def set_active_product(self, file_name) -> tuple[bool, str]:
        try:
            registry = self.load_registry()
            if not self._is_safe_file_name(file_name) or not any(
                item["file"] == file_name for item in registry["configs"]
            ):
                return False, "目标产品配置已不在配置列表中"
            updated = copy.deepcopy(registry)
            updated["active_file"] = file_name
            if not self.save_registry(updated):
                return False, "当前配置记录保存失败，请检查配置目录权限"
        except (OSError, ValueError) as error:
            return False, str(error)
        return True, file_name

    def load_project(self, file_name):
        if not self._is_safe_file_name(file_name):
            return error_code.INVALID_DATA_LOADING, "配置文件名无效"
        code, data = LoadUiConfig.load_data_from_json(os.path.join(self.program_dir, file_name))
        if code != error_code.OK:
            return code, data
        errors = validate_structure(data)
        if errors:
            return error_code.INVALID_DATA_LOADING, "\n".join(errors)
        return error_code.OK, data

    def get_hardware_channels(self):
        device = self.input_device_provider() if self.input_device_provider else None
        return hardware_channels(device)

    def load_queue_catalog(self, queue_names=None):
        registry = LoadUiConfig._load_sequence_config_registry(self.queue_registry_path)
        directory = os.path.dirname(self.queue_registry_path)
        return {name: load_queue_info(resolve_queue_path(path, directory))
                for name, path in registry.items()
                if name != "using_config_path" and isinstance(path, str)
                and (queue_names is None or name in queue_names)}

    def validate_project(self, project, current_file=None, queue_catalog=None, *, check_acquisition_consistency=False):
        errors = validate_structure(project)
        warnings = []
        if not errors:
            name_error = self._project_name_error(project["project_name"].strip())
            if name_error:
                errors.append(name_error)
            catalog = self.load_queue_catalog() if queue_catalog is None else queue_catalog
            errors.extend(validate_references(project, catalog))
            available = self.get_hardware_channels()
            keys = {(x["device_ref"], x["physical_channel_index"]) for x in available}
            devices = {x["device_ref"] for x in available}
            for station in project["stations"]:
                for col, channel in enumerate(station["channels"], 1):
                    hw = channel["hardware_channel"]
                    key = (hw["device_ref"], hw["physical_channel_index"])
                    label = f"{station['station_name']} WAV第{col}通道"
                    if hw["device_ref"] in devices and key not in keys:
                        errors.append(f"{label}超出已知硬件通道范围")
                    elif key not in keys:
                        warnings.append(f"{label}硬件引用未核实（设备未连接或不在当前硬件目录中）")
        return {"can_save": not errors, "save_errors": errors,
                "is_usable": False, "is_test_mode_usable": False,
                "use_errors": errors + [RUNTIME_UNAVAILABLE], "use_warnings": warnings,
                "test_mode_errors": [RUNTIME_UNAVAILABLE]}

    def save_project(self, current_file, project_data, *, update_active=True):
        validation = self.validate_project(project_data, current_file)
        if validation["save_errors"]:
            return False, "\n".join(validation["save_errors"])
        if current_file and not self._is_safe_file_name(current_file):
            return False, "配置文件名无效"
        try:
            registry = self.load_registry()
        except ValueError as error:
            return False, str(error)
        project = normalize_project(project_data)
        project_id = project["project_id"]
        target_file = project["project_name"] + ".json"
        for entry in registry["configs"]:
            if entry["file"] == current_file and entry.get("project_id") != project_id:
                return False, "不能用另一项目身份覆盖当前配置，请另存为"
            if entry["file"] != current_file and entry.get("project_id") == project_id:
                return False, "项目 ID 已存在，请另存为独立配置"
            if entry["file"] != current_file and entry["file"].casefold() == target_file.casefold():
                return False, "同名产品配置已存在，请修改项目名称"
        if current_file:
            code, existing = self.load_project(current_file)
            if code != error_code.OK:
                return False, "原配置无法读取，未覆盖原文件"
            if existing["project_id"] != project_id:
                return False, "不能用另一项目身份覆盖当前配置，请另存为"
        try:
            files = os.listdir(self.program_dir) if os.path.isdir(self.program_dir) else []
        except OSError as error:
            return False, f"配置目录无法读取：{error}"
        if any(name != current_file and name.casefold() == target_file.casefold() for name in files):
            return False, "同名文件已存在，请修改项目名称"
        updated = copy.deepcopy(registry)
        updated["configs"] = [e for e in updated["configs"] if e["file"] != current_file]
        updated["configs"].append({"project_id": project_id, "project_name": project["project_name"], "file": target_file})
        if ((current_file and updated.get("active_file") == current_file)
                or (update_active and not updated.get("active_file"))):
            updated["active_file"] = target_file
        return self._write_project(project, current_file, target_file, updated)

    def _write_project(self, project, current_file, target_file, registry):
        path = os.path.join(self.program_dir, target_file)
        original_path = os.path.join(self.program_dir, current_file) if current_file else path
        same_path = os.path.normcase(original_path) == os.path.normcase(path)
        try:
            snapshot = self._read_file_snapshot(path)
            registry_snapshot = self._read_file_snapshot(self.registry_path)
            if not LoadUiConfig.save_data_to_json(project, path):
                return False, "产品配置写入失败"
        except OSError as error:
            return False, f"产品配置写入失败：{error}"
        failure = "配置列表保存失败"
        try:
            saved = self.save_registry(registry)
            if saved and current_file and not same_path:
                failure = "原配置文件移除失败"
                os.remove(original_path)
        except OSError:
            saved = False
        if not saved:
            # Keep the old file until the registry is saved. Roll back both files
            # on handled failures, including failure to remove the old filename.
            restored = self._restore_file_snapshot(original_path if same_path else path, snapshot)
            try:
                registry_restored = (self._read_file_snapshot(self.registry_path) == registry_snapshot
                                     or self._restore_file_snapshot(self.registry_path, registry_snapshot))
            except OSError:
                registry_restored = False
            return False, failure + ("，原配置已恢复" if restored and registry_restored else "，原配置恢复失败")
        return True, target_file

    @staticmethod
    def _project_name_error(name):
        if any(char in '<>:"/\\|?*' or ord(char) < 32 for char in name):
            return '项目名称用于文件名，不能包含以下字符：< > : " / \\ | ? * 或控制字符'
        if name.endswith(".") or PureWindowsPath(name + ".json").is_reserved():
            return "项目名称不能以句点结尾，也不能使用 Windows 保留名称（如 CON、NUL、COM1）"
        if (name + ".json").casefold() == PROGRAM_REGISTRY_FILE.casefold():
            return "项目名称与配置列表文件重名，请更换名称"
        if len((name + ".json").encode("utf-16-le")) // 2 > 255:
            return "项目名称过长，请缩短名称"
        return ""

    def import_project(self, source_path):
        code, project = LoadUiConfig.load_data_from_json(source_path)
        if code != error_code.OK:
            return False, project
        errors = validate_structure(project)
        return (False, "\n".join(errors)) if errors else (True, project)

    def save_as(self, project, new_name):
        errors = validate_structure(project)
        if errors:
            return False, "\n".join(errors)
        return self.save_project(None, copy_project(project, new_name), update_active=False)

    def existing_project_file(self, identity):
        for entry in self.load_registry()["configs"]:
            if identity in (entry.get("project_id"), entry["file"]):
                return entry["file"]
        return None

    def delete_project(self, file_name):
        from base.config_deletion import ConfigDeletionService, ConfigDeletionError
        service = ConfigDeletionService(self)
        try:
            target = next((t for t in service.list_targets("product") if t.key == file_name), None)
            if target is None:
                return False, "配置不存在"
            service.delete(target)
        except ConfigDeletionError as error:
            return False, str(error)
        return True, ""

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
