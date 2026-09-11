import builtins
import copy
import json
import os
from dataclasses import FrozenInstanceError

import pytest

from base.load_config import LoadUiConfig
from base.product_test_program_config import ProductTestProgramConfigManager
from base.sequence_queue_references import (
    QueueReferenceDraft,
    SequenceQueueReferenceScanner,
)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def project(*queues, name="Product", group="Port"):
    return {"project_name": name, "test_groups": [{
        "group_name": group,
        "test_conditions": [
            {"condition_name": chr(65 + index), "test_queue": queue}
            for index, queue in enumerate(queues)
        ],
    }]}


@pytest.fixture
def files(tmp_path):
    product_dir = tmp_path / "products"
    product_registry = product_dir / "program_registry.json"
    queue_registry = tmp_path / "queues" / "registry.json"
    target = queue_registry.parent / "Q.json"
    write_json(target, [])
    write_json(queue_registry, {"Q": "Q.json", "alias": str(target), "R": "R.json"})
    write_json(product_registry, {"active_file": "one.json", "configs": [
        {"file": "one.json", "project_name": "Product"},
    ]})
    write_json(product_dir / "one.json", project("Q", "alias"))
    scanner = SequenceQueueReferenceScanner(product_dir, product_registry, queue_registry)
    return scanner, product_dir, product_registry, queue_registry, target


def test_saved_union_draft_preserves_redirected_reference_and_new_condition(files):
    scanner, product_dir, _, _, target = files
    data = project("Q", "R", "Q")
    before = copy.deepcopy(data)
    draft = QueueReferenceDraft(data, product_path=product_dir / "one.json")
    result = scanner.find_references(target, drafts=(draft,))
    assert not result.issues
    assert [(r.condition_name, r.sources) for r in result.references] == [
        ("A", frozenset({"saved", "draft"})),
        ("B", frozenset({"saved"})),
        ("C", frozenset({"draft"})),
    ]
    assert data == before
    with pytest.raises(FrozenInstanceError):
        result.references[0].condition_name = "changed"
    with pytest.raises(FrozenInstanceError):
        result.issues = ()
    # A later scan reflects the newly persisted parent, with no cached disk refs.
    write_json(product_dir / "one.json", data)
    assert [r.condition_name for r in scanner.find_references(target).references] == ["A", "C"]


def test_registered_inactive_legacy_and_duplicate_names_have_positional_identity(files):
    scanner, product_dir, registry, _, target = files
    data = project("Q", "Q")
    data["test_groups"][0]["test_conditions"][1]["condition_name"] = "A"
    data["test_groups"].append(copy.deepcopy(data["test_groups"][0]))
    write_json(product_dir / "one.json", data)
    write_json(product_dir / "old.json", {"name": "Legacy", "sub_configs": [
        {"condition_name": "Old", "test_queue": "alias"},
    ]})
    write_json(registry, {"active_file": "one.json", "configs": [
        {"file": "one.json", "project_name": "Product"},
        {"file": "old.json", "name": "Legacy"},
        {"file": "one.json", "project_name": "duplicate registry entry"},
    ]})
    write_json(product_dir / "unregistered.json", project("Q"))
    result = scanner.find_references(target)
    assert not result.issues
    assert [(r.product_name, r.group_index, r.condition_index) for r in result.references] == [
        ("Product", 1, 1), ("Product", 1, 2),
        ("Product", 2, 1), ("Product", 2, 2), ("Legacy", 1, 1),
    ]


def test_new_draft_instance_identity_and_renamed_display_names(files):
    scanner, product_dir, _, _, target = files
    renamed = QueueReferenceDraft(project("Q", name="Renamed", group="New port"),
                                  product_path=product_dir / "one.json")
    renamed.data["test_groups"][0]["test_conditions"][0]["condition_name"] = "New condition"
    first = QueueReferenceDraft(project("Q", name="New"))
    second = QueueReferenceDraft(project("Q", name="New"))
    result = scanner.find_references(target, drafts=(renamed, first, first, second))
    assert not result.issues
    assert len(result.references) == 4
    reference = result.references[0]
    assert reference.sources == frozenset({"saved", "draft"})
    assert [(n.source, n.product_name, n.group_name, n.condition_name)
            for n in reference.display_names] == [
        ("saved", "Product", "Port", "A"),
        ("draft", "Renamed", "New port", "New condition"),
    ]
    assert result.references[2].product_identity != result.references[3].product_identity
    assert result.references[2].product_path is None


def test_alias_relative_absolute_and_catalog_relocation(files):
    scanner, product_dir, _, registry, target = files
    stale = target.parent.parent / "old_install" / target.name
    write_json(registry, {"alias": str(stale), "relative": "./sub/../Q.json"})
    write_json(product_dir / "one.json", project("alias", "relative", str(target), "Q.json"))
    result = scanner.find_references(target)
    assert not result.issues
    assert len(result.references) == 4
    assert len(scanner.find_references(target.parent / "new.json").references) == 0


@pytest.mark.skipif(os.name != "nt", reason="Windows path case semantics")
def test_windows_case_difference_matches_queue_and_draft_product(files):
    scanner, product_dir, _, registry, target = files
    write_json(registry, {"Q": str(target).upper(), "alias": str(target).lower()})
    draft = QueueReferenceDraft(project("Q", "alias"),
                                product_path=str(product_dir / "one.json").upper())
    result = scanner.find_references(str(target).lower(), drafts=(draft,))
    assert not result.issues
    assert len(result.references) == 2
    assert all(r.sources == frozenset({"saved", "draft"}) for r in result.references)


@pytest.mark.parametrize("bad", [None, [], {"test_groups": None},
    {"test_groups": [None, {"test_conditions": "bad"}]}, {"sub_configs": {}}])
def test_invalid_product_schema_reports_issue_and_keeps_other_known_refs(files, bad):
    scanner, product_dir, registry, _, target = files
    write_json(product_dir / "bad.json", bad)
    write_json(registry, {"configs": [{"file": "one.json"}, {"file": "bad.json"}]})
    result = scanner.find_references(target)
    assert len(result.references) == 2
    assert result.issues
    assert all(issue.path == str(product_dir / "bad.json") for issue in result.issues)
    assert all(issue.message for issue in result.issues)


def test_malformed_rows_do_not_discard_valid_siblings(files):
    scanner, product_dir, registry, queue_registry, target = files
    data = project("Q", "Q")
    data["test_groups"][0]["test_conditions"].extend([None, {"test_queue": 42}])
    data["test_groups"].extend([False, {"test_conditions": None}])
    write_json(product_dir / "one.json", data)
    write_json(registry, {"configs": [None, {"file": "one.json"}, {"file": 42}]})
    write_json(queue_registry, {"Q": "Q.json", "bad": {}, "using_config_path": None})
    result = scanner.find_references(target)
    assert len(result.references) == 2
    assert len(result.issues) == 7


@pytest.mark.parametrize("failure", ["missing", "json", "encoding", "denied"])
def test_registered_product_read_failure_is_diagnostic_and_query_never_writes(files, monkeypatch, failure):
    scanner, product_dir, registry, _, target = files
    broken = product_dir / "broken.json"
    write_json(registry, {"configs": [{"file": "one.json"},
                                      {"file": "broken.json", "name": "Broken"}]})
    if failure == "json":
        broken.write_text("{", encoding="utf-8")
    elif failure == "encoding":
        broken.write_bytes(b"\xff")
    elif failure == "denied":
        write_json(broken, project("Q"))
    original_open = builtins.open
    def read_only_open(path, mode="r", *args, **kwargs):
        assert not any(flag in mode for flag in "wax+")
        if failure == "denied" and os.fspath(path) == str(broken):
            raise PermissionError("fixture read denied")
        return original_open(path, mode, *args, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail("Query attempted a mutating manager/loader operation")
    snapshot = {p: p.read_bytes() for p in target.parent.parent.rglob("*.json")}
    monkeypatch.setattr(builtins, "open", read_only_open)
    monkeypatch.setattr(LoadUiConfig, "save_data_to_json", forbidden)
    monkeypatch.setattr(LoadUiConfig, "_load_sequence_config_registry", forbidden)
    monkeypatch.setattr(ProductTestProgramConfigManager, "load_registry", forbidden)
    monkeypatch.setattr(ProductTestProgramConfigManager, "rebuild_registry", forbidden)
    result = scanner.find_references(target)
    assert len(result.references) == 2
    assert len(result.issues) == 1
    assert result.issues[0].product_name == "Broken"
    assert result.issues[0].path == str(broken)
    assert snapshot == {p: p.read_bytes() for p in target.parent.parent.rglob("*.json")}


@pytest.mark.parametrize("registry_kind", ["product", "queue"])
@pytest.mark.parametrize("bad", ["{", "[]", "null"])
def test_invalid_registry_is_diagnostic_and_preserves_draft_or_direct_refs(files, registry_kind, bad):
    scanner, product_dir, product_registry, queue_registry, target = files
    write_json(product_dir / "one.json", project(str(target)))
    registry = product_registry if registry_kind == "product" else queue_registry
    registry.write_text(bad, encoding="utf-8")
    draft = QueueReferenceDraft(project(str(target)), product_path=product_dir / "one.json")
    result = scanner.find_references(target, drafts=(draft,))
    assert len(result.references) == 1
    assert any(issue.path == str(registry) for issue in result.issues)


def test_absent_registries_empty_draft_fields_and_unresolved_alias_are_distinct(files):
    scanner, _, product_registry, queue_registry, target = files
    product_registry.unlink()
    queue_registry.unlink()
    assert scanner.find_references(target).issues == ()
    blank = QueueReferenceDraft(project(""))
    assert scanner.find_references(target, drafts=(blank,)).issues == ()
    unknown = QueueReferenceDraft(project("unregistered alias"))
    assert scanner.find_references(target, drafts=(unknown,)).issues


@pytest.mark.parametrize("registry_kind", ["product", "queue"])
def test_unreadable_registry_reports_issue_and_keeps_known_draft(files, monkeypatch, registry_kind):
    scanner, _, product_registry, queue_registry, target = files
    denied = product_registry if registry_kind == "product" else queue_registry
    original_open = builtins.open
    def denied_open(path, *args, **kwargs):
        if os.fspath(path) == str(denied):
            raise PermissionError("fixture registry denied")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", denied_open)
    draft = QueueReferenceDraft(project(str(target)))
    result = scanner.find_references(target, drafts=(draft,))
    assert any(r.sources == frozenset({"draft"}) for r in result.references)
    assert any(issue.path == str(denied) and "denied" in issue.message
               for issue in result.issues)


def test_invalid_registered_filename_is_an_issue_without_attempting_to_open_it(files):
    scanner, _, registry, _, target = files
    write_json(registry, {"configs": [{"file": "one.json"}, {"file": "bad\u0000.json"}]})
    result = scanner.find_references(target)
    assert len(result.references) == 2
    assert len(result.issues) == 1
    assert result.issues[0].path == str(registry)


def test_invalid_registry_display_name_reports_issue_without_losing_file_references(files):
    scanner, _, registry, _, target = files
    write_json(registry, {"configs": [{"file": "one.json", "project_name": []}]})
    result = scanner.find_references(target)
    assert len(result.references) == 2
    assert len(result.issues) == 1
    assert result.issues[0].path == str(registry)
