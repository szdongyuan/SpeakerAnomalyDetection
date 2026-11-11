import logging
import os
from pathlib import Path
import stat
import subprocess
import sys
from unittest.mock import DEFAULT, Mock, call

import pytest
from PyQt5.QtCore import QIODevice, QSettings


def _preferences():
    from ui.sequence import workbench_toolbar_preferences

    return workbench_toolbar_preferences


@pytest.fixture
def logger():
    return logging.getLogger(__name__)


@pytest.mark.parametrize("content", [None, b"[General]\nunrelated=value\n"])
def test_missing_mode_defaults_to_mark_without_writing(tmp_path, logger, caplog, content):
    path = tmp_path / "preferences.ini"
    if content is not None:
        path.write_bytes(content)

    assert _preferences().load_mode(logger, path=path) == "mark"

    if content is None:
        assert not path.exists()
    else:
        assert path.read_bytes() == content
    assert not caplog.records


def test_mode_round_trips_in_qt_ini_format(tmp_path, logger):
    preferences = _preferences()
    path = tmp_path / "preferences.ini"

    preferences.save_mode("test", logger, path=path)
    assert preferences.load_mode(logger, path=path) == "test"
    preferences.save_mode("mark", logger, path=path)
    assert preferences.load_mode(logger, path=path) == "mark"

    settings = QSettings(str(path), QSettings.IniFormat)
    assert settings.value("mode") == "mark"
    assert settings.status() == QSettings.NoError


@pytest.mark.parametrize("content", [b"[General]\nmode=other\n", b"[General\nmode=test\n"])
def test_invalid_ini_defaults_to_mark_and_warns(tmp_path, logger, caplog, content):
    path = tmp_path / "preferences.ini"
    path.write_bytes(content)

    with caplog.at_level(logging.WARNING):
        assert _preferences().load_mode(logger, path=path) == "mark"

    assert str(path) in caplog.text
    assert path.read_bytes() == content


def test_unreadable_ini_defaults_to_mark_and_warns(tmp_path, logger, caplog):
    path = tmp_path / "preferences.ini"
    path.mkdir()

    with caplog.at_level(logging.WARNING):
        assert _preferences().load_mode(logger, path=path) == "mark"

    assert str(path) in caplog.text
    assert path.is_dir()


def test_commit_failure_is_diagnosed_and_next_save_uses_fresh_file(tmp_path, logger, caplog, monkeypatch):
    preferences = _preferences()
    path = tmp_path / "preferences.ini"
    preferences.save_mode("test", logger, path=path)
    original = path.read_bytes()
    original_mtime = path.stat().st_mtime_ns
    payload = b"[General]\nmode=mark\n"
    output = Mock()
    output.open.return_value = True
    output.write.return_value = len(payload)
    output.commit.return_value = False
    output.errorString.return_value = "simulated commit failure"
    make_output = Mock(side_effect=[output, DEFAULT], wraps=preferences.QSaveFile)
    monkeypatch.setattr(preferences, "QSaveFile", make_output)

    with caplog.at_level(logging.WARNING):
        preferences.save_mode("mark", logger, path=path)

    make_output.assert_called_once_with(str(path))
    output.assert_has_calls([
        call.setDirectWriteFallback(False),
        call.open(QIODevice.WriteOnly),
        call.write(payload),
        call.commit(),
    ])
    assert str(path) in caplog.text
    assert "simulated commit failure" in caplog.text
    assert path.read_bytes() == original
    assert path.stat().st_mtime_ns == original_mtime
    assert preferences.load_mode(logger, path=path) == "test"

    preferences.save_mode("mark", logger, path=path)

    assert make_output.call_args_list == [call(str(path)), call(str(path))]
    assert preferences.load_mode(logger, path=path) == "mark"


@pytest.mark.skipif(sys.platform != "win32", reason="Requires Windows read-only file attributes")
def test_failed_publish_keeps_last_successful_value(tmp_path, logger, caplog):
    preferences = _preferences()
    path = tmp_path / "preferences.ini"
    preferences.save_mode("test", logger, path=path)
    original = path.read_bytes()
    original_mtime = path.stat().st_mtime_ns

    path.chmod(stat.S_IREAD)
    try:
        with caplog.at_level(logging.WARNING):
            preferences.save_mode("mark", logger, path=path)
    finally:
        path.chmod(stat.S_IREAD | stat.S_IWRITE)

    assert str(path) in caplog.text
    assert path.read_bytes() == original
    assert path.stat().st_mtime_ns == original_mtime
    assert preferences.load_mode(logger, path=path) == "test"
    assert list(tmp_path.iterdir()) == [path]

    preferences.save_mode("mark", logger, path=path)

    assert preferences.load_mode(logger, path=path) == "mark"


def test_mode_reads_replacement_with_same_size_and_mtime(tmp_path, logger):
    preferences = _preferences()
    path = tmp_path / "preferences.ini"
    preferences.save_mode("test", logger, path=path)
    assert preferences.load_mode(logger, path=path) == "test"
    original = path.stat()

    preferences.save_mode("mark", logger, path=path)
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))

    replacement = path.stat()
    assert replacement.st_size == original.st_size
    assert replacement.st_mtime_ns == original.st_mtime_ns
    assert path.read_bytes() == b"[General]\nmode=mark\n"
    assert preferences.load_mode(logger, path=path) == "mark"


def test_mode_restores_on_a_later_date_in_fresh_process(tmp_path, logger, monkeypatch):
    preferences = _preferences()
    monkeypatch.setattr(preferences, "DEFAULT_DIR", tmp_path)
    preferences.save_mode("test", logger)
    path = tmp_path / "ui" / "ui_config" / "workbench_toolbar_preferences.ini"
    original = path.read_bytes()
    original_mtime = path.stat().st_mtime_ns

    child_code = """
import datetime
import logging
import sys

now = datetime.datetime.now()
shift = datetime.timedelta(days=2)
later = now + shift

class LaterDate(datetime.date):
    @classmethod
    def today(cls):
        return later.date()

class LaterDatetime(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return later if tz is None else later.astimezone(tz)

datetime.date = LaterDate
datetime.datetime = LaterDatetime
assert datetime.date.today() == now.date() + shift
assert datetime.datetime.now() == now + shift

from ui.sequence import workbench_toolbar_preferences as preferences

preferences.DEFAULT_DIR = sys.argv[1]
assert preferences.load_mode(logging.getLogger(__name__)) == "test"
"""
    child = subprocess.run(
        [sys.executable, "-c", child_code, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert child.returncode == 0, child.stderr
    assert path.read_bytes() == original
    assert path.stat().st_mtime_ns == original_mtime
