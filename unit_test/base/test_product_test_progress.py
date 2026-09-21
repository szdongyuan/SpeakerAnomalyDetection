import json
import os

import numpy as np
import pytest

from base import product_test_progress as progress


def test_analysis_groups_round_trip_preserves_exact_registered_paths(tmp_path):
    wav = (tmp_path / "audio.wav").as_posix()
    raw = (tmp_path / "raw.csv").as_posix()
    analysis = {
        (tmp_path / "csv" / "audio" / "SPL_总体.csv").as_posix(),
        (tmp_path / "csv" / "audio" / "FBA.csv").as_posix(),
        (tmp_path / "images" / "audio" / "CH1.PNG").as_posix(),
        (tmp_path / "images" / "audio" / "CH2.jpg").as_posix(),
    }
    groups = progress.group_progress_analysis_files(wav, analysis | {wav, raw}, {raw})
    assert groups["csv"] == {
        "directory": (tmp_path / "csv" / "audio").as_posix(),
        "files": ["FBA.csv", "SPL_总体.csv"],
    }
    assert groups["images"]["files"] == ["CH1.PNG", "CH2.jpg"]
    assert progress.expand_progress_analysis_files(groups) == set(map(os.path.normpath, analysis))
    # Files need not exist: missing/moved outputs still belong in the saved ledger.
    assert list(tmp_path.iterdir()) == []


def test_empty_analysis_groups_have_no_inferred_directories():
    groups = progress.group_progress_analysis_files("a.wav", {"a.wav"}, set())
    assert groups == {
        "csv": {"directory": "", "files": []},
        "images": {"directory": "", "files": []},
    }
    assert progress.expand_progress_analysis_files(groups) == set()


@pytest.mark.parametrize("files,raw", [
    (set(), set()),
    ({"a.wav"}, {"raw.csv"}),
    ({"a.wav", "other.wav"}, set()),
    ({"a.wav", "one/a.csv", "two/b.csv"}, set()),
])
def test_grouping_rejects_records_that_cannot_be_preserved(files, raw):
    with pytest.raises(ValueError):
        progress.group_progress_analysis_files("a.wav", files, raw)


@pytest.mark.parametrize("name", [
    "../other.csv", r"..\other.csv", "/other.csv", "D:/other.csv",
    r"D:\other.csv", "D:other.csv", "file.csv:other.csv", "", "..",
    "plot.png", None,
])
def test_grouped_filenames_cannot_escape_directory_or_change_type(name):
    groups = {
        "csv": {"directory": "results", "files": [name]},
        "images": {"directory": "", "files": []},
    }
    with pytest.raises(ValueError):
        progress.expand_progress_analysis_files(groups)


@pytest.mark.parametrize("csv", [None, {}, {"directory": "", "files": ["a.csv"]},
    {"directory": "results", "files": []}, {"directory": [], "files": ["a.csv"]},
    {"directory": "results", "files": "a.csv"},
])
def test_invalid_result_groups_are_rejected(csv):
    with pytest.raises(ValueError):
        progress.expand_progress_analysis_files({
            "csv": csv, "images": {"directory": "", "files": []},
        })


def test_store_round_trip_and_clear(tmp_path):
    store = progress.ProductTestProgressStore(tmp_path / "config" / "progress.json")
    assert store.load() is None
    store.save({"round": np.int64(3), "completed": {"b", "a"}})
    assert set(json.loads(store.path.read_text(encoding="utf-8"))) == {"progress"}
    assert store.load() == {"round": 3, "completed": ["a", "b"]}
    store.clear()
    assert store.load() is None
    assert json.loads(store.path.read_text(encoding="utf-8")) == {"progress": None}
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


@pytest.mark.parametrize("data", [
    "{", "[]", '{"version":4,"progress":null}', '{"version":1,"progress":[]}',
    '{"version":true,"progress":null}', '{"version":2}',
    '{"version":null,"progress":null}', '{}', '{"progress":[]}',
])
def test_invalid_files_are_not_silently_treated_as_empty(tmp_path, data):
    store = progress.ProductTestProgressStore(tmp_path / "progress.json")
    store.path.write_text(data, encoding="utf-8")
    with pytest.raises(ValueError):
        store.load()
    store.clear()
    assert store.load() is None


@pytest.mark.parametrize("version", [None, 1, 2, 3])
def test_tombstone_is_read_without_rewriting(tmp_path, version):
    store = progress.ProductTestProgressStore(tmp_path / "progress.json")
    payload = {"progress": None}
    if version is not None:
        payload["version"] = version
    content = json.dumps(payload)
    store.path.write_text(content, encoding="utf-8")
    assert store.load() is None
    assert store.loaded_version == version
    assert store.path.read_text(encoding="utf-8") == content


def test_frozen_path_is_beside_executable_not_extraction_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(progress.sys, "frozen", True, raising=False)
    monkeypatch.setattr(progress.sys, "executable", str(tmp_path / "app" / "testing.exe"))
    monkeypatch.setattr(progress, "DEFAULT_DIR", str(tmp_path / "temporary-extraction"))
    assert progress.product_test_progress_path() == tmp_path / "app/ui/ui_config/product_test_progress.json"


def test_source_path_uses_existing_config_root(tmp_path, monkeypatch):
    monkeypatch.setattr(progress.sys, "frozen", False, raising=False)
    monkeypatch.setattr(progress, "DEFAULT_DIR", str(tmp_path))
    assert progress.product_test_progress_path() == tmp_path / "ui/ui_config/product_test_progress.json"
