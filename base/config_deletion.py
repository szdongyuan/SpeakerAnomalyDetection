"""Deletion boundary for registered product and queue configurations.

Loading may relocate old queue paths. Deletion intentionally never does.
"""

import copy
import json
import os
import stat
from dataclasses import dataclass

from base.load_config import LoadUiConfig
from base.product_test_project_config import ProductTestProjectConfigManager
from base.sequence_queue_references import SequenceQueueReferenceScanner, queue_path_key
from consts.running_consts import DEFAULT_DIR


class ConfigDeletionError(ValueError):
    def __init__(self, message, *, inconsistent=False, references=()):
        super().__init__(message)
        self.inconsistent = inconsistent
        self.references = tuple(references)


@dataclass(frozen=True)
class DeletionTarget:
    kind: str
    key: str
    path: str
    names: tuple[str, ...]
    file_stamp: tuple | None
    registry_stamp: str
    remove_file: bool
    error: str = ""

    @property
    def action(self):
        return "删除" if self.remove_file else "移除"


def exact_path(path, directory):
    if not isinstance(path, str) or not path.strip() or "\0" in path:
        raise ConfigDeletionError("配置路径无效")
    return os.path.abspath(os.path.join(directory, path))


def _read_registry(path, *, missing=None):
    try:
        with open(path, encoding="utf-8") as stream:
            registry = json.load(stream)
    except FileNotFoundError:
        if missing is not None:
            return copy.deepcopy(missing)
        raise ConfigDeletionError("配置列表不存在，请重新打开后检查") from None
    except (OSError, UnicodeError, ValueError) as error:
        raise ConfigDeletionError(f"无法读取配置列表：{error}") from error
    if not isinstance(registry, dict):
        raise ConfigDeletionError("配置列表格式错误，不能执行删除")
    return registry


def _stamp(registry):
    return json.dumps(registry, ensure_ascii=False, sort_keys=True)


def _write_registry(registry, path):
    try:
        return LoadUiConfig.save_data_to_json(registry, path)
    except OSError:
        # Directory creation and failed temporary-file cleanup may also raise.
        return False


class ConfigDeletionService:
    def __init__(self, manager=None):
        self.manager = manager or ProductTestProjectConfigManager()
        self.scanner = SequenceQueueReferenceScanner(
            self.manager.program_dir, self.manager.registry_path,
            self.manager.queue_registry_path,
        )
        self.protected_paths = {
            queue_path_key(path, DEFAULT_DIR) for path in (
                self.manager.registry_path, self.manager.queue_registry_path,
                os.path.join(DEFAULT_DIR, "ui", "ui_config", "sequence_config.json"),
            )
        }

    def _product_registry(self, *, allow_missing=False):
        registry = _read_registry(
            self.manager.registry_path,
            missing={"configs": [], "active_file": None} if allow_missing else None,
        )
        entries = registry.get("configs")
        if not isinstance(entries, list):
            raise ConfigDeletionError("产品配置列表格式错误")
        seen = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ConfigDeletionError("产品配置列表存在无效记录")
            filename = entry.get("file")
            name = entry.get("project_name", entry.get("name"))
            if (not self.manager._is_safe_file_name(filename)
                    or "\0" in filename or ":" in filename
                    or not isinstance(name, str) or not name.strip()
                    or filename.casefold() in seen):
                raise ConfigDeletionError("产品配置列表存在无效或重复记录")
            seen.add(filename.casefold())
        return registry

    def _queue_registry(self):
        registry = _read_registry(self.manager.queue_registry_path, missing={})
        return registry

    def _target(self, kind, key, path, names, registry):
        root = (self.manager.program_dir if kind == "product"
                else os.path.dirname(self.manager.queue_registry_path))
        path = exact_path(path, root)
        identity = queue_path_key(path, root)
        if identity in self.protected_paths:
            raise ConfigDeletionError("内置模板或配置列表不能删除")
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            info = None
        except OSError as error:
            raise ConfigDeletionError(f"无法访问配置文件：{error}") from error
        if info is not None and not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
            raise ConfigDeletionError("目标不是普通配置文件")
        linked = (identity != os.path.normcase(path)
                  or (info is not None and bool(getattr(info, "st_file_attributes", 0)
                                               & stat.FILE_ATTRIBUTE_REPARSE_POINT)))
        root_identity = queue_path_key(root, root)
        try:
            inside = os.path.commonpath([identity, root_identity]) == root_identity
        except ValueError:
            inside = False
        if kind == "product" and (linked or not inside):
            raise ConfigDeletionError("产品配置路径超出管理目录或为链接")
        file_stamp = None if info is None else (
            info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
        )
        return DeletionTarget(
            kind, key, path, tuple(names), file_stamp, _stamp(registry),
            info is not None and inside and not linked and path.lower().endswith(".json"),
        )

    def list_targets(self, kind):
        if kind == "product":
            registry = self._product_registry(allow_missing=True)
            return [self._target(kind, entry["file"], entry["file"],
                                 [entry.get("project_name", entry.get("name"))], registry)
                    for entry in registry["configs"]]
        if kind != "queue":
            raise ValueError("Unknown configuration kind")
        registry = self._queue_registry()
        directory = os.path.dirname(self.manager.queue_registry_path)
        groups = {}
        targets = []
        for name, value in registry.items():
            if name == "using_config_path":
                continue
            try:
                if not name.strip():
                    raise ConfigDeletionError("队列名称无效")
                path = exact_path(value, directory)
            except ConfigDeletionError as error:
                targets.append(DeletionTarget(
                    kind, f"invalid:{name}", "", (name,), None, _stamp(registry),
                    False, str(error),
                ))
                continue
            groups.setdefault(os.path.normcase(path), (path, []))[1].append(name)
        for key, (path, names) in groups.items():
            try:
                if queue_path_key(path, directory) in self.protected_paths:
                    continue
                targets.append(self._target(kind, key, path, names, registry))
            except ConfigDeletionError as error:
                targets.append(DeletionTarget(
                    kind, key, path, tuple(names), None, _stamp(registry),
                    False, str(error),
                ))
        return targets

    def check(self, target, *, drafts=()):
        current = next((item for item in self.list_targets(target.kind)
                        if item.key == target.key), None)
        if current != target:
            raise ConfigDeletionError("配置或文件已变化，请重新选择并确认")
        if target.error:
            raise ConfigDeletionError(target.error)
        if target.kind == "queue":
            # Registered products and editor drafts define queue references.
            result = self.scanner.find_references(target.path, drafts=drafts, aliases=target.names)
            blocking_issues = [issue for issue in result.issues if issue.blocks_deletion]
            if blocking_issues:
                names = list(dict.fromkeys(issue.product_name or os.path.basename(issue.path or "")
                                           for issue in blocking_issues))
                raise ConfigDeletionError("无法完整检查队列引用，请检查配置：" + "、".join(names))
            if result.references:
                rows = list(dict.fromkeys(
                    f"{name.product_name} / {name.group_name} / {name.condition_name}"
                    for reference in result.references for name in reference.display_names
                ))
                raise ConfigDeletionError(
                    "该队列正在使用，请先更换相关工况的队列并保存：\n" + "\n".join(rows),
                    references=rows,
                )

    def delete(self, target, *, drafts=()):
        self.check(target, drafts=drafts)
        registry_path = (self.manager.registry_path if target.kind == "product"
                         else self.manager.queue_registry_path)
        original = _read_registry(registry_path)
        if _stamp(original) != target.registry_stamp:
            raise ConfigDeletionError("配置列表已变化，请重新选择并确认")
        updated = copy.deepcopy(original)
        if target.kind == "product":
            updated["configs"] = [entry for entry in updated["configs"] if entry["file"] != target.key]
            active = updated.get("active_file")
            remaining = {entry["file"].casefold() for entry in updated["configs"]}
            if not isinstance(active, str) or active.casefold() not in remaining:
                updated["active_file"] = None
        else:
            for name in target.names:
                del updated[name]
            selection = updated.get("using_config_path")
            directory = os.path.dirname(registry_path)
            if selection is not None and (
                not isinstance(selection, str) or not selection.strip() or "\0" in selection
                or queue_path_key(selection, directory) == queue_path_key(target.path, directory)
            ):
                updated["using_config_path"] = None
        if not _write_registry(updated, registry_path):
            raise ConfigDeletionError("配置列表更新失败，未删除文件")
        if target.remove_file:
            try:
                os.remove(target.path)
            except OSError as error:
                if not _write_registry(original, registry_path):
                    raise ConfigDeletionError(
                        "文件未能删除，配置列表也未恢复。请检查配置列表后重启软件。",
                        inconsistent=True,
                    ) from error
                raise ConfigDeletionError(f"文件删除失败，已恢复配置列表：{error}") from error
        # The next checked target may advance only to the registry we just wrote.
        return _stamp(updated)
