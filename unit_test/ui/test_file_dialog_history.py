import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

from ui import file_dialog_history as history


def test_first_read_does_not_create_history(tmp_path, isolate_file_dialog_history):
    assert history.get_initial_directory(history.QUEUE_IMPORT, str(tmp_path)) == str(tmp_path)
    assert not isolate_file_dialog_history.exists()


def test_independent_entries_survive_new_reads_and_same_directory(tmp_path, isolate_file_dialog_history):
    audio = tmp_path / "音频 含空格"
    audio.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    for key in (history.REFERENCE_AUDIO_SELECT, history.AUDIO_CLIP_SAVE):
        assert history.remember_directory(key, str(audio))
    assert history.remember_directory(history.AUDIO_CLIP_SAVE, str(other))
    assert history.get_initial_directory(history.REFERENCE_AUDIO_SELECT, "") == str(audio)
    assert history.get_initial_directory(history.AUDIO_CLIP_SAVE, "") == str(other)
    data = json.loads(isolate_file_dialog_history.read_text(encoding="utf-8"))
    assert set(data) == {"last_directories"}
    assert len(data["last_directories"]) == 2
    # Read again from disk instead of a process cache.
    data["last_directories"][history.REFERENCE_AUDIO_SELECT] = str(other)
    isolate_file_dialog_history.write_text(json.dumps(data), encoding="utf-8")
    assert history.get_initial_directory(history.REFERENCE_AUDIO_SELECT, "") == str(other)


@pytest.mark.parametrize("contents", ["{", "[]", '{"last_directories": []}', "{}"])
def test_invalid_document_falls_back(contents, tmp_path, isolate_file_dialog_history, caplog):
    isolate_file_dialog_history.write_text(contents, encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="core"):
        assert history.get_initial_directory(history.REPORT_PDF_SAVE, str(tmp_path)) == str(tmp_path)
    assert caplog.records


@pytest.mark.parametrize("invalid", [None, 5, [], "", "relative/path", "Z:/missing/directory", "C:/bad\x00path"])
def test_invalid_entry_does_not_hide_other_entry(invalid, tmp_path, isolate_file_dialog_history):
    isolate_file_dialog_history.write_text(json.dumps({"last_directories": {
        history.QUEUE_IMPORT: invalid, history.QUEUE_SAVE_AS: str(tmp_path),
    }}), encoding="utf-8")
    assert history.get_initial_directory(history.QUEUE_IMPORT, str(tmp_path)) == str(tmp_path)
    assert history.get_initial_directory(history.QUEUE_SAVE_AS, "") == str(tmp_path)


def test_removed_history_falls_back_without_recreating(tmp_path):
    removed = tmp_path / "removed"
    removed.mkdir()
    history.remember_directory(history.QUEUE_IMPORT, str(removed))
    removed.rmdir()
    assert history.get_initial_directory(history.QUEUE_IMPORT, str(tmp_path)) == str(tmp_path)
    assert not removed.exists()


def test_documents_then_home_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(history.QStandardPaths, "writableLocation", lambda _: str(tmp_path))
    assert history.get_initial_directory(history.QUEUE_IMPORT, "") == str(tmp_path)
    monkeypatch.setattr(history.QStandardPaths, "writableLocation", lambda _: "")
    monkeypatch.setattr(history.QDir, "homePath", lambda: str(tmp_path))
    assert history.get_initial_directory(history.QUEUE_IMPORT, "") == str(tmp_path)


def test_unchanged_directory_does_not_write(tmp_path, monkeypatch):
    history.remember_directory(history.QUEUE_IMPORT, str(tmp_path))
    def unexpected_write(*args, **kwargs):
        pytest.fail("unchanged history must not be rewritten")
    monkeypatch.setattr(history.tempfile, "NamedTemporaryFile", unexpected_write)
    assert history.remember_directory(history.QUEUE_IMPORT, str(tmp_path))


def test_failed_replace_preserves_previous_file(tmp_path, isolate_file_dialog_history, monkeypatch, caplog):
    history.remember_directory(history.QUEUE_IMPORT, str(tmp_path))
    before = isolate_file_dialog_history.read_bytes()
    def fail_replace(*args):
        raise PermissionError("simulated write failure")
    monkeypatch.setattr(history.os, "replace", fail_replace)
    with caplog.at_level(logging.WARNING, logger="core"):
        assert not history.remember_directory(history.REPORT_PDF_SAVE, str(tmp_path))
    assert isolate_file_dialog_history.read_bytes() == before
    assert not list(tmp_path.glob(".file_dialog_history.json.*.tmp"))
    assert caplog.records


def test_directory_creation_failure_is_nonfatal(tmp_path, monkeypatch):
    blocking_file = tmp_path / "file"
    blocking_file.write_text("not a directory")
    monkeypatch.setattr(history, "_history_path", lambda: blocking_file / "history.json")
    assert not history.remember_directory(history.REPORT_PDF_SAVE, str(tmp_path))


def test_unknown_key_is_a_programming_error(tmp_path):
    with pytest.raises(ValueError):
        history.get_initial_directory("typo", str(tmp_path))
    with pytest.raises(ValueError):
        history.remember_directory("typo", str(tmp_path))


def test_default_history_path_is_independent_of_cwd(tmp_path, monkeypatch):
    # Use the real resolver despite the automatic isolation fixture.
    import importlib.util
    spec = importlib.util.spec_from_file_location("history_path_probe", history.__file__)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    before = probe._history_path().resolve()
    monkeypatch.chdir(tmp_path)
    assert probe._history_path().resolve() == before
    assert before.name == "file_dialog_history.json"


def test_persistence_across_process_restart(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    target = tmp_path / "restart-history.json"
    chosen = tmp_path / "confirmed"
    chosen.mkdir()
    prefix = (
        "import sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(repository)!r}); "
        "from ui import file_dialog_history as h; "
        f"h._history_path = lambda: Path({str(target)!r}); "
    )
    subprocess.run(
        [sys.executable, "-c", prefix +
         f"assert h.remember_directory(h.REPORT_PDF_SAVE, {str(chosen)!r})"],
        cwd=tmp_path, check=True, capture_output=True, timeout=20,
    )
    subprocess.run(
        [sys.executable, "-c", prefix +
         f"assert h.get_initial_directory(h.REPORT_PDF_SAVE, '') == {str(chosen)!r}"],
        cwd=chosen, check=True, capture_output=True, timeout=20,
    )
