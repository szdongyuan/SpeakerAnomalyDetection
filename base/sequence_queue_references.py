"""Read-only, uncached lookup of saved and draft product queue references."""

import json
import os
from dataclasses import dataclass, replace

from base.product_test_project_config import (
    ProductTestProjectConfigManager,
    iter_test_conditions,
)
from consts.product_test_project_consts import (
    CONDITION_NAME_KEY,
    GROUP_NAME_KEY,
    PROJECT_NAME_KEY,
    REGISTRY_CONFIGS_KEY,
    REGISTRY_FILE_KEY,
    TEST_CONDITIONS_KEY,
    TEST_GROUPS_KEY,
    TEST_QUEUE_KEY,
)
from consts.running_consts import (
    PRODUCT_TEST_PROGRAM_DIR,
    PRODUCT_TEST_PROGRAM_REGISTRY_PATH,
    SEQUENCE_CONFIG_REGISTRY_PATH,
)


def queue_path_key(path, registry_dir):
    """Canonical identity of an actual target (without catalog relocation)."""
    path = os.fspath(path)
    candidate = path if os.path.isabs(path) else os.path.join(registry_dir, path)
    return os.path.normcase(os.path.realpath(os.path.abspath(candidate)))


@dataclass(frozen=True, eq=False)
class QueueReferenceDraft:
    """Parent editor data and its original saved file, if any.

    Pass the current in-memory project/program dict as ``data``. ``product_path``
    may be absolute or relative to product_dir and must identify the original
    disk file even when the draft renames the product. For a new product, retain
    one draft instance per editor; separate instances have separate identities.
    The scanner never changes or saves data.
    """

    data: dict
    product_path: str | os.PathLike | None = None


@dataclass(frozen=True)
class QueueReferenceNames:
    source: str
    product_name: str
    group_name: str
    condition_name: str


@dataclass(frozen=True)
class QueueReference:
    product_path: str | None
    product_identity: str
    product_name: str
    group_index: int
    group_name: str
    condition_index: int
    condition_name: str
    sources: frozenset[str]
    display_names: tuple[QueueReferenceNames, ...]


@dataclass(frozen=True)
class QueueReferenceIssue:
    path: str | None
    product_name: str
    source: str
    message: str


@dataclass(frozen=True)
class QueueReferenceResult:
    references: tuple[QueueReference, ...]
    issues: tuple[QueueReferenceIssue, ...]


class SequenceQueueReferenceScanner:
    def __init__(
        self,
        product_dir=PRODUCT_TEST_PROGRAM_DIR,
        product_registry_path=PRODUCT_TEST_PROGRAM_REGISTRY_PATH,
        queue_registry_path=SEQUENCE_CONFIG_REGISTRY_PATH,
    ):
        self.product_dir = os.path.abspath(product_dir)
        self.product_registry_path = os.path.abspath(product_registry_path)
        self.queue_registry_path = os.path.abspath(queue_registry_path)

    def find_references(self, target_path, *, drafts=()):
        """Return immutable positional references and diagnostics, without writes.

        Sources are ``saved`` and/or ``draft``. ``display_names`` preserves each
        source's names, including names changed by an unsaved parent editor.
        Any issue means the result may be incomplete. Missing registries are
        empty on first use; missing registered files and unresolved nonempty
        aliases are issues. Each call reads the current disk versions again.
        """
        registry_dir = os.path.dirname(self.queue_registry_path)
        target_key = queue_path_key(target_path, registry_dir)
        issues = []
        queue_registry = self._read_json(
            self.queue_registry_path, issues, missing_ok=True
        )
        catalog = {}
        if queue_registry is not None:
            if not isinstance(queue_registry, dict):
                self._issue(issues, self.queue_registry_path, "", "saved",
                            "Queue registry must be an object")
            else:
                for alias, path in queue_registry.items():
                    if alias == "using_config_path":
                        continue
                    if not isinstance(path, str) or not path.strip() or "\0" in path:
                        self._issue(issues, self.queue_registry_path, "", "saved",
                                    f"Invalid queue path for {alias!r}")
                        # Preserve the invalid alias so it cannot be treated as a path.
                        catalog[alias] = None
                        continue
                    catalog[alias] = self._queue_key(path.strip(), registry_dir)

        references = {}
        registry = self._read_json(self.product_registry_path, issues, missing_ok=True)
        if registry is not None:
            if not isinstance(registry, dict) or not isinstance(
                registry.get(REGISTRY_CONFIGS_KEY), list
            ):
                self._issue(issues, self.product_registry_path, "", "saved",
                            "Product registry configs must be a list")
            else:
                for entry in registry[REGISTRY_CONFIGS_KEY]:
                    file_name = (entry.get(REGISTRY_FILE_KEY)
                                 if isinstance(entry, dict) else None)
                    if (not ProductTestProjectConfigManager._is_safe_file_name(file_name)
                            or "\0" in file_name):
                        self._issue(issues, self.product_registry_path, "", "saved",
                                    "Invalid registered product file")
                        continue
                    path = os.path.join(self.product_dir, file_name)
                    name = entry.get(PROJECT_NAME_KEY, entry.get("name", ""))
                    if not isinstance(name, str):
                        self._issue(issues, self.product_registry_path,
                                    file_name, "saved",
                                    "Registered product name must be a string")
                        name = ""
                    data = self._read_json(path, issues, product_name=name)
                    if data is not None:
                        self._collect(data, path, queue_path_key(path, self.product_dir),
                                      name, "saved", catalog, target_key,
                                      references, issues)

        for draft in drafts:
            if not isinstance(draft, QueueReferenceDraft):
                raise TypeError("drafts must contain QueueReferenceDraft instances")
            path = (os.path.abspath(os.path.join(self.product_dir, draft.product_path))
                    if draft.product_path else None)
            identity = (queue_path_key(path, self.product_dir)
                        if path else f"draft:{id(draft)}")
            self._collect(draft.data, path, identity, "", "draft", catalog,
                          target_key, references, issues)
        return QueueReferenceResult(tuple(references.values()), tuple(issues))

    @staticmethod
    def _issue(issues, path, name, source, message):
        issues.append(QueueReferenceIssue(path, name, source, message))

    def _read_json(self, path, issues, *, missing_ok=False, product_name=""):
        # This is the file/JSON boundary. Failures remain visible and no repair
        # is attempted; independent registered products can still be scanned.
        try:
            with open(path, "r", encoding="utf-8") as stream:
                data = json.load(stream)
        except FileNotFoundError as error:
            if not missing_ok:
                self._issue(issues, path, product_name, "saved", str(error))
            return None
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            self._issue(issues, path, product_name, "saved", str(error))
            return None
        if data is None:
            self._issue(issues, path, product_name, "saved", "Expected JSON object, got null")
        return data

    @staticmethod
    def _queue_key(path, registry_dir):
        resolved = ProductTestProjectConfigManager._resolve_queue_path(path, registry_dir)
        return queue_path_key(resolved, registry_dir)

    def _collect(self, data, path, identity, registered_name, source, catalog,
                 target_key, references, issues):
        name = registered_name or (os.path.basename(path) if path else "Unsaved product")
        if not isinstance(data, dict):
            self._issue(issues, path, name, source, "Product must be an object")
            return
        name = self._name(data, PROJECT_NAME_KEY if PROJECT_NAME_KEY in data else "name",
                          name, path, name, source, issues)
        for group_index, group, condition_index, condition in self._conditions(
            data, path, name, source, issues
        ):
            group_name = self._name(group, GROUP_NAME_KEY, f"Group {group_index}",
                                    path, name, source, issues)
            condition_name = self._name(condition, CONDITION_NAME_KEY,
                                        f"Condition {condition_index}",
                                        path, name, source, issues)
            queue = condition.get(TEST_QUEUE_KEY, "")
            location = f"Group {group_index}, condition {condition_index}"
            if not isinstance(queue, str) or "\0" in queue:
                self._issue(issues, path, name, source, f"{location}: invalid test_queue")
                continue
            queue = queue.strip()
            if not queue:
                continue
            if queue in catalog:
                key = catalog[queue]
            elif (os.path.isabs(queue) or "/" in queue or "\\" in queue
                  or queue.lower().endswith(".json")):
                key = self._queue_key(queue, os.path.dirname(self.queue_registry_path))
            else:
                self._issue(issues, path, name, source,
                            f"{location}: unresolved queue alias {queue!r}")
                continue
            if key != target_key:
                continue
            position = (identity, group_index, condition_index)
            names = QueueReferenceNames(source, name, group_name, condition_name)
            previous = references.get(position)
            if previous is None:
                references[position] = QueueReference(
                    path, identity, name, group_index, group_name,
                    condition_index, condition_name, frozenset({source}), (names,),
                )
            else:
                references[position] = replace(
                    previous,
                    sources=previous.sources | {source},
                    display_names=(previous.display_names if names in previous.display_names
                                   else previous.display_names + (names,)),
                )

    def _name(self, data, field, fallback, path, product_name, source, issues):
        value = data.get(field, "")
        if not isinstance(value, str):
            self._issue(issues, path, product_name, source, f"{field} must be a string")
            return fallback
        return value.strip() or fallback

    def _conditions(self, data, path, name, source, issues):
        if TEST_GROUPS_KEY in data:
            groups = data[TEST_GROUPS_KEY]
        elif "sub_configs" in data:
            groups = [{GROUP_NAME_KEY: "", TEST_CONDITIONS_KEY: data["sub_configs"]}]
        else:
            self._issue(issues, path, name, source, "Missing test_groups or sub_configs")
            return
        if not isinstance(groups, list):
            self._issue(issues, path, name, source, "test_groups must be a list")
            return
        # Retain positions while validating the shapes the shared iterator uses.
        checked = []
        for group_index, group in enumerate(groups, 1):
            if not isinstance(group, dict):
                self._issue(issues, path, name, source, f"Group {group_index} must be an object")
                checked.append({})
                continue
            conditions = group.get(TEST_CONDITIONS_KEY)
            if not isinstance(conditions, list):
                self._issue(issues, path, name, source,
                            f"Group {group_index}: test_conditions must be a list")
                checked.append({})
                continue
            for condition_index, condition in enumerate(conditions, 1):
                if not isinstance(condition, dict):
                    self._issue(issues, path, name, source,
                                f"Group {group_index}, condition {condition_index} must be an object")
            checked.append(group)
        yield from iter_test_conditions({TEST_GROUPS_KEY: checked})
