"""Configuration-only phase: saved stations cannot enter the legacy runtime."""
from pathlib import Path

import pytest

from base.product_test_config_refresh import (
    ProductTestConfigurationError, build_product_test_refresh_snapshot,
)
from unit_test.product_config_fixtures import make_manager, make_project


def make_refresh_manager(tmp_path, condition_count=2, port_count=2):
    manager = make_manager(tmp_path)
    project = make_project(tmp_path, station_count=port_count, condition_count=condition_count)
    ok, filename = manager.save_project(None, project)
    assert ok, filename
    queue_path = Path(manager.load_queue_catalog()["测试队列A"]["path"])
    return manager, project, queue_path


def test_empty_configuration_never_loads_default_or_queues(tmp_path, monkeypatch):
    manager = make_manager(tmp_path)
    monkeypatch.setattr(manager, "load_project", lambda *_: pytest.fail("unexpected project read"))
    snapshot = build_product_test_refresh_snapshot(manager, None)
    assert snapshot.conditions == [] and snapshot.context == {} and snapshot.active_file == ""


def test_saved_station_configuration_cannot_enter_old_runtime(tmp_path, monkeypatch):
    manager, project, queue_path = make_refresh_manager(tmp_path)
    before = queue_path.read_bytes()
    monkeypatch.setattr(manager, "load_queue_catalog", lambda **_: pytest.fail("unexpected execution queue read"))
    with pytest.raises(ProductTestConfigurationError, match="尚未"):
        build_product_test_refresh_snapshot(manager, project["project_name"] + ".json")
    assert queue_path.read_bytes() == before


@pytest.mark.parametrize("content", [None, "{", '{"test_groups": []}'])
def test_missing_corrupt_and_legacy_configurations_are_rejected(tmp_path, content):
    manager, project, _ = make_refresh_manager(tmp_path)
    filename = project["project_name"] + ".json"
    path = Path(manager.program_dir, filename)
    if content is None:
        path.rename(path.with_suffix(".moved"))
    else:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(ProductTestConfigurationError, match="无法读取"):
        build_product_test_refresh_snapshot(manager, filename)
