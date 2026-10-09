import json
import os

import pytest

from base.config_deletion import ConfigDeletionError, ConfigDeletionService
from base.load_config import LoadUiConfig
from base.product_test_project_config import ProductTestProjectConfigManager
from base.sequence_queue_references import QueueReferenceDraft


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def product(queue="", name="A"):
    return {"project_name": name, "test_groups": [{"group_name": "A口", "test_conditions": [
        {"condition_name": "档位1", "test_queue": queue},
    ]}]}


@pytest.fixture
def files(tmp_path):
    products = tmp_path / "products"
    registry = products / "program_registry.json"
    queues = tmp_path / "queues" / "registry.json"
    target = queues.parent / "Q.json"
    write(target, [])
    write(queues, {"Q": str(target), "using_config_path": str(target)})
    write(registry, {"active_file": "A.json", "configs": [
        {"file": "A.json", "project_name": "A"}, {"file": "B.json", "project_name": "B"},
    ], "extra": "keep"})
    write(products / "A.json", product())
    write(products / "B.json", product(name="B"))
    service = ConfigDeletionService(ProductTestProjectConfigManager(str(products), str(registry), str(queues)))
    return service, products, registry, queues, target


def queue_target(service):
    return next(item for item in service.list_targets("queue") if "Q" in item.names)


def test_product_delete_preserves_other_files_metadata_and_results(files):
    service, products, registry, queues, target = files
    history = products / "results" / "record.wav"
    history.parent.mkdir()
    history.write_bytes(b"recording")
    before = (products / "B.json").read_bytes(), queues.read_bytes(), target.read_bytes()
    service.delete(next(item for item in service.list_targets("product") if item.key == "A.json"))
    assert not (products / "A.json").exists()
    assert json.loads(registry.read_text())["active_file"] is None
    assert json.loads(registry.read_text())["extra"] == "keep"
    assert before == ((products / "B.json").read_bytes(), queues.read_bytes(), target.read_bytes())
    assert history.read_bytes() == b"recording"


@pytest.mark.parametrize("registry_kind", ["product", "queue"])
def test_corrupt_registry_never_writes_or_deletes(files, registry_kind):
    service, products, registry, queues, target = files
    broken = registry if registry_kind == "product" else queues
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigDeletionError):
        service.list_targets(registry_kind)
    assert broken.read_text() == "{"
    assert target.exists() and (products / "A.json").exists()


@pytest.mark.parametrize("source", ["saved", "draft", "both", "inactive"])
def test_any_product_or_draft_reference_blocks_queue(files, source):
    service, products, registry, queues, target = files
    if source in {"saved", "both", "inactive"}:
        write(products / ("B.json" if source == "inactive" else "A.json"), product("Q"))
    drafts = (QueueReferenceDraft(product("Q")),) if source in {"draft", "both"} else ()
    original = queues.read_bytes()
    with pytest.raises(ConfigDeletionError, match="档位1"):
        service.delete(queue_target(service), drafts=drafts)
    assert queues.read_bytes() == original and target.exists()


def test_unsaved_replacement_keeps_saved_reference(files):
    service, products, _, _, _ = files
    write(products / "A.json", product("Q"))
    with pytest.raises(ConfigDeletionError, match="正在使用"):
        service.delete(queue_target(service), drafts=(QueueReferenceDraft(product(), products / "A.json"),))


@pytest.mark.parametrize("state", ["unregistered", "invalid", "missing_registry"])
def test_unregistered_product_files_do_not_block_queue_deletion(files, state):
    service, products, registry, queues, target = files
    unregistered = products / "1.json"
    write(unregistered, product("Q"))
    if state == "invalid":
        unregistered.write_text("{", encoding="utf-8")
    if state == "missing_registry":
        registry.unlink()
    before = {path.name: path.read_bytes() for path in products.iterdir()}
    service.delete(queue_target(service))
    assert not target.exists()
    assert json.loads(queues.read_text()) == {"using_config_path": None}
    assert before == {path.name: path.read_bytes() for path in products.iterdir()}


@pytest.mark.parametrize("broken", ["registry", "registered_file", "missing_file"])
def test_incomplete_registered_references_block_queue_deletion(files, broken):
    service, products, registry, queues, target = files
    if broken == "missing_file":
        (products / "A.json").unlink()
    else:
        path = registry if broken == "registry" else products / "A.json"
        path.write_text("{", encoding="utf-8")
    before = queues.read_bytes()
    with pytest.raises(ConfigDeletionError):
        service.delete(queue_target(service))
    assert target.exists() and queues.read_bytes() == before


def test_missing_queue_alias_is_blocked_even_when_loader_relocates(files):
    service, products, _, queues, target = files
    missing = target.parent.parent / "old" / target.name
    write(queues, {"Q": str(missing)})
    write(products / "A.json", product("Q"))
    with pytest.raises(ConfigDeletionError, match="正在使用"):
        service.delete(queue_target(service))
    assert target.exists()
    write(products / "A.json", product())
    selected = queue_target(service)
    assert not selected.remove_file and selected.path == str(missing)
    service.delete(selected)
    assert target.exists() and json.loads(queues.read_text()) == {}


def test_relocated_reference_blocks_deleting_actual_local_file(files):
    service, products, _, queues, target = files
    write(queues, {"Q": str(target), "old": str(target.parent.parent / "old" / target.name)})
    write(products / "A.json", product("old"))
    with pytest.raises(ConfigDeletionError, match="正在使用"):
        service.delete(queue_target(service))


def test_same_target_aliases_are_removed_together_and_selection_cleared(files):
    service, _, _, queues, target = files
    other = target.parent / "elsewhere" / target.name
    write(other, [])
    write(queues, {"Q": str(target), "alias": "Q.json", "different": str(other), "using_config_path": str(target)})
    selected = queue_target(service)
    assert set(selected.names) == {"Q", "alias"}
    service.delete(selected)
    assert not target.exists() and other.exists()
    assert json.loads(queues.read_text()) == {"different": str(other), "using_config_path": None}


@pytest.mark.parametrize("exists", [False, True])
def test_builtin_default_is_hidden_by_path_without_hiding_user_names(
    files, tmp_path, monkeypatch, exists,
):
    service, _, _, queues, target = files
    default = tmp_path / "ui" / "ui_config" / "sequence_config.json"
    if exists:
        write(default, [])
    monkeypatch.setattr("base.config_deletion.DEFAULT_DIR", str(tmp_path))
    service = ConfigDeletionService(service.manager)
    same_filename = target.with_name("sequence_config.json")
    write(same_filename, [])
    registry = {
        "builtin": str(default),
        "builtin_alias": os.path.relpath(default, queues.parent),
        "默认配置": str(target),
        "sequence_config": str(same_filename),
        "using_config_path": str(default),
    }
    write(queues, registry)
    before = queues.read_bytes()
    targets = service.list_targets("queue")
    assert {item.names for item in targets} == {("默认配置",), ("sequence_config",)}
    assert all(item.remove_file for item in targets)
    assert queues.read_bytes() == before
    service.delete(next(item for item in targets if "默认配置" in item.names))
    registry.pop("默认配置")
    assert json.loads(queues.read_text()) == registry
    assert default.exists() is exists
    if exists:
        assert json.loads(default.read_text()) == []


def test_external_target_only_removed(files, tmp_path):
    service, _, _, queues, _ = files
    external = tmp_path / "external" / "Q.json"
    write(external, {"keep": 1})
    write(queues, {"Q": str(external), "using_config_path": str(external)})
    selected = queue_target(service)
    assert selected.action == "移除"
    service.delete(selected)
    assert json.loads(external.read_text()) == {"keep": 1}
    assert json.loads(queues.read_text()) == {"using_config_path": None}


@pytest.mark.parametrize("change", ["file", "registry"])
def test_changed_target_requires_new_confirmation(files, change):
    service, _, _, queues, target = files
    selected = queue_target(service)
    if change == "file":
        write(target, {"changed": True})
    else:
        write(queues, {"Q": str(target), "another": str(target.parent / "R.json")})
    before = queues.read_bytes()
    with pytest.raises(ConfigDeletionError, match="变化"):
        service.delete(selected)
    assert target.exists() and queues.read_bytes() == before


@pytest.mark.parametrize("restore", [True, False])
def test_file_failure_and_restore_failure_are_distinct(files, monkeypatch, restore):
    service, _, _, queues, target = files
    selected = queue_target(service)
    original = json.loads(queues.read_text())
    real_save = LoadUiConfig.save_data_to_json
    calls = []
    def save(data, path):
        calls.append(data)
        return real_save(data, path) if len(calls) == 1 or restore else False
    monkeypatch.setattr(LoadUiConfig, "save_data_to_json", save)
    def fail(path):
        raise PermissionError("locked")
    monkeypatch.setattr("base.config_deletion.os.remove", fail)
    with pytest.raises(ConfigDeletionError) as error:
        service.delete(selected)
    assert error.value.inconsistent is (not restore)
    assert target.exists()
    assert (json.loads(queues.read_text()) == original) is restore


def test_registry_write_failure_does_not_delete_file(files, monkeypatch):
    service, _, _, queues, target = files
    before = queues.read_bytes()
    monkeypatch.setattr(LoadUiConfig, "save_data_to_json", lambda *args: False)
    with pytest.raises(ConfigDeletionError, match="未删除文件"):
        service.delete(queue_target(service))
    assert target.exists() and queues.read_bytes() == before


def test_builtin_and_registry_are_never_candidates(files):
    service, _, registry, queues, _ = files
    protected = next(iter(service.protected_paths))
    write(queues, {"protected": protected, "registry": str(registry)})
    assert service.list_targets("queue") == []


def test_external_reference_also_blocks_removal(files, tmp_path):
    service, products, _, queues, _ = files
    external = tmp_path / "external.json"
    write(external, [])
    write(queues, {"Q": str(external)})
    write(products / "A.json", product("Q"))
    with pytest.raises(ConfigDeletionError, match="正在使用"):
        service.delete(queue_target(service))
    assert external.exists()


def test_unreadable_file_is_not_treated_as_missing(files, monkeypatch):
    service, _, _, queues, target = files
    real_lstat = os.lstat
    def denied(path, *args, **kwargs):
        if os.path.normcase(str(path)) == os.path.normcase(str(target)):
            raise PermissionError("denied")
        return real_lstat(path, *args, **kwargs)
    monkeypatch.setattr("base.config_deletion.os.lstat", denied)
    before = queues.read_bytes()
    selected = queue_target(service)
    assert selected.error and not selected.remove_file
    with pytest.raises(ConfigDeletionError):
        service.delete(selected)
    assert queues.read_bytes() == before


def test_write_exception_keeps_file(files, monkeypatch):
    service, _, _, queues, target = files
    before = queues.read_bytes()
    def fail(*args):
        raise PermissionError("directory denied")
    monkeypatch.setattr(LoadUiConfig, "save_data_to_json", fail)
    with pytest.raises(ConfigDeletionError, match="未删除文件"):
        service.delete(queue_target(service))
    assert target.exists() and queues.read_bytes() == before


def test_managed_directory_prefix_does_not_own_sibling(files):
    service, _, _, queues, target = files
    sibling = queues.parent.with_name(queues.parent.name + "_external") / "Q.json"
    write(sibling, [])
    write(queues, {"Q": str(sibling)})
    selected = queue_target(service)
    assert not selected.remove_file
    service.delete(selected)
    assert sibling.exists() and target.exists()


def test_damaged_unreferenced_queue_can_be_deleted(files):
    service, _, _, _, target = files
    target.write_text("{", encoding="utf-8")
    service.delete(queue_target(service))
    assert not target.exists()


def test_first_use_without_products_allows_deletion(tmp_path):
    target = tmp_path / "queues" / "Q.json"
    queues = target.parent / "registry.json"
    write(target, [])
    write(queues, {"Q": str(target)})
    manager = ProductTestProjectConfigManager(str(tmp_path / "products"),
        str(tmp_path / "products" / "program_registry.json"), str(queues))
    service = ConfigDeletionService(manager)
    service.delete(queue_target(service))
    assert not target.exists()


def test_link_in_managed_directory_never_deletes_external_file(files, tmp_path):
    service, _, _, queues, target = files
    external = tmp_path / "external.json"
    write(external, ["original"])
    link = target.with_name("link.json")
    try:
        link.symlink_to(external)
    except OSError as error:
        pytest.skip(f"Symbolic links not available: {error}")
    write(queues, {"Q": str(link)})
    selected = queue_target(service)
    assert not selected.remove_file
    service.delete(selected)
    assert link.is_symlink() and json.loads(external.read_text()) == ["original"]
