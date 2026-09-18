import numpy as np
import pytest

from base import product_test_progress as progress


def test_store_round_trip_and_clear(tmp_path):
    store = progress.ProductTestProgressStore(tmp_path / "config" / "progress.json")
    assert store.load() is None
    store.save({"round": np.int64(3), "completed": {"b", "a"}})
    assert store.load() == {"round": 3, "completed": ["a", "b"]}
    store.clear()
    assert store.load() is None
    assert store.path.exists()


def test_failed_replace_keeps_previous_snapshot_and_cleans_temporary(tmp_path, monkeypatch):
    store = progress.ProductTestProgressStore(tmp_path / "progress.json")
    store.save({"round": 3})

    def fail(*args):
        raise PermissionError("read only")

    monkeypatch.setattr(progress.os, "replace", fail)
    with pytest.raises(PermissionError):
        store.save({"round": 4})
    assert store.load() == {"round": 3}
    assert list(tmp_path.iterdir()) == [store.path]


@pytest.mark.parametrize("data", ["{", "[]", '{"version":2,"progress":null}', '{"version":1,"progress":[]}'])
def test_invalid_files_are_not_silently_treated_as_empty(tmp_path, data):
    store = progress.ProductTestProgressStore(tmp_path / "progress.json")
    store.path.write_text(data, encoding="utf-8")
    with pytest.raises(ValueError):
        store.load()
    store.clear()
    assert store.load() is None


def test_frozen_path_is_beside_executable_not_extraction_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(progress.sys, "frozen", True, raising=False)
    monkeypatch.setattr(progress.sys, "executable", str(tmp_path / "app" / "testing.exe"))
    monkeypatch.setattr(progress, "DEFAULT_DIR", str(tmp_path / "temporary-extraction"))
    assert progress.product_test_progress_path() == tmp_path / "app/ui/ui_config/product_test_progress.json"


def test_source_path_uses_existing_config_root(tmp_path, monkeypatch):
    monkeypatch.setattr(progress.sys, "frozen", False, raising=False)
    monkeypatch.setattr(progress, "DEFAULT_DIR", str(tmp_path))
    assert progress.product_test_progress_path() == tmp_path / "ui/ui_config/product_test_progress.json"
