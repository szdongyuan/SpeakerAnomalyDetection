import json
from pathlib import Path

import pytest

from base.config_deletion import ConfigDeletionError
from base.sequence_queue_references import QueueReferenceDraft
from unit_test.base.test_config_deletion import files, product, write, queue_target


@pytest.mark.parametrize("draft", [False, True])
def test_unresolved_name_does_not_block_unrelated_queue(files, draft):
    service, products, _, _, target = files
    data = product("old_queue")
    drafts = (QueueReferenceDraft(data),) if draft else ()
    if not draft:
        write(products / "A.json", data)
    result = service.scanner.find_references(str(target), drafts=drafts)
    assert result.issues  # Shared-save warnings must remain available.
    service.delete(queue_target(service), drafts=drafts)
    assert not target.exists()


@pytest.mark.parametrize("source", ["saved", "draft", "path"])
def test_unresolved_name_does_not_hide_actual_reference(files, source):
    service, products, _, queues, target = files
    write(products / "A.json", product("old_queue"))
    data = product(str(target) if source == "path" else "Q")
    drafts = (QueueReferenceDraft(data),) if source == "draft" else ()
    if source != "draft":
        write(products / "B.json", data)
    before = queues.read_bytes(), target.read_bytes()
    with pytest.raises(ConfigDeletionError) as error:
        service.delete(queue_target(service), drafts=drafts)
    assert error.value.references
    assert before == (queues.read_bytes(), target.read_bytes())


@pytest.mark.parametrize("failure", ["json", "missing", "schema", "registry"])
def test_unresolved_name_cannot_mask_unknown_references(files, failure):
    service, products, registry, queues, target = files
    write(products / "A.json", product("old_queue"))
    if failure == "json":
        (products / "B.json").write_text("{", encoding="utf-8")
    elif failure == "missing":
        (products / "B.json").unlink()
    elif failure == "schema":
        write(products / "B.json", {"test_groups": False})
    else:
        registry.write_text("{", encoding="utf-8")
    before = queues.read_bytes(), target.read_bytes()
    with pytest.raises(ConfigDeletionError):
        service.delete(queue_target(service))
    assert before == (queues.read_bytes(), target.read_bytes())


@pytest.mark.parametrize("reference", ["none", "alias", "path"])
def test_legacy_suffix_is_record_only_and_still_checks_references(files, reference):
    service, products, _, queues, target = files
    legacy = target.with_name("legacy.txt")
    write(legacy, [])
    write(queues, {"Q": str(target), "legacy": str(legacy), "using_config_path": str(legacy)})
    if reference != "none":
        write(products / "A.json", product("legacy" if reference == "alias" else str(legacy)))
    selected = next(t for t in service.list_targets("queue") if "legacy" in t.names)
    assert not selected.remove_file
    before = legacy.read_bytes()
    if reference == "none":
        service.delete(selected)
        assert json.loads(queues.read_text()) == {"Q": str(target), "using_config_path": None}
    else:
        with pytest.raises(ConfigDeletionError) as error:
            service.delete(selected)
        assert error.value.references
    assert legacy.read_bytes() == before and target.exists()


@pytest.mark.parametrize("bad", [None, {}, "bad\x00.json"])
def test_unused_bad_queue_row_does_not_block_normal_target(files, bad):
    service, _, _, queues, target = files
    write(queues, {"Q": str(target), "bad": bad})
    selected = queue_target(service)
    service.delete(selected)
    assert not target.exists()
    assert json.loads(queues.read_text()) == {"bad": bad}


def test_referenced_bad_mapping_still_blocks_unknown_references(files):
    service, products, _, queues, target = files
    write(queues, {"Q": str(target), "bad": None})
    write(products / "A.json", product("bad"))
    with pytest.raises(ConfigDeletionError):
        service.delete(queue_target(service))
    assert target.exists()


@pytest.mark.parametrize("active", ["missing.json", "", [], 42])
def test_stale_product_selection_is_read_only_until_success(files, active):
    service, products, registry, _, target = files
    data = json.loads(registry.read_text())
    data["active_file"] = active
    write(registry, data)
    before = registry.read_bytes()
    selected = next(t for t in service.list_targets("product") if t.key == "B.json")
    service.check(selected)
    assert registry.read_bytes() == before
    service.delete(selected)
    saved = json.loads(registry.read_text())
    assert saved["active_file"] is None and saved["extra"] == "keep"
    assert saved["configs"] == [{"file": "A.json", "project_name": "A"}]
    assert target.exists() and (products / "A.json").exists()


def test_stale_selection_and_metadata_restore_when_file_delete_fails(files, monkeypatch):
    service, _, registry, _, _ = files
    data = json.loads(registry.read_text())
    data["active_file"] = "missing.json"
    write(registry, data)
    selected = service.list_targets("product")[0]
    before = registry.read_bytes()
    def fail(path):
        raise PermissionError("locked")
    monkeypatch.setattr("base.config_deletion.os.remove", fail)
    with pytest.raises(ConfigDeletionError) as error:
        service.delete(selected)
    assert not error.value.inconsistent
    assert json.loads(registry.read_text()) == json.loads(before)


def test_active_normalization_must_not_hide_concurrent_change(files):
    service, _, registry, _, _ = files
    data = json.loads(registry.read_text())
    data["active_file"] = "old_missing.json"
    write(registry, data)
    selected = service.list_targets("product")[0]
    data["active_file"] = "new_missing.json"
    write(registry, data)
    with pytest.raises(ConfigDeletionError):
        service.delete(selected)
    assert json.loads(registry.read_text())["active_file"] == "new_missing.json"


def test_default_queue_absent_selection_stays_absent_after_unrelated_deletion(files):
    service, _, _, queues, target = files
    write(queues, {"Q": str(target), "using_config_path": None})
    service.delete(queue_target(service))
    assert json.loads(queues.read_text()) == {"using_config_path": None}


@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("queue_name", ["13212", "other"])
def test_legacy_product_reference_scope(files, registered, queue_name):
    service, products, registry, queues, target = files
    legacy = products / "1.json"
    write(legacy, {
        "name": "1", "close_trigger_state": "",
        "pdf_report": {"enabled": False, "save_dir": ""},
        "sub_configs": [{"condition_name": "1", "trigger_state": "", "test_queue": "13212"}],
    })
    original = legacy.read_bytes()
    referenced = target.with_name("13212.json")
    write(referenced, [])
    write(queues, {"13212": str(referenced), "other": str(target)})
    if registered:
        data = json.loads(registry.read_text())
        data["configs"].append({"file": "1.json", "name": "1"})
        write(registry, data)
    selected = next(t for t in service.list_targets("queue") if queue_name in t.names)
    if registered and queue_name == "13212":
        with pytest.raises(ConfigDeletionError) as error:
            service.delete(selected)
        assert error.value.references
        assert referenced.exists()
    else:
        service.delete(selected)
        assert not Path(selected.path).exists()
    assert legacy.read_bytes() == original
