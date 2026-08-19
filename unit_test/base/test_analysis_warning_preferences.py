import json
import os
from unittest.mock import Mock, patch

import pytest

import base.analysis_warning_preferences as preferences
from base.analysis_warning_preferences import (
    is_uncalibrated_microphone_warning_suppressed,
    save_uncalibrated_microphone_warning_suppressed,
)


def test_missing_file_defaults_to_unsuppressed_without_creating_file(tmp_path):
    path = tmp_path / "ui" / "ui_config" / "analysis_warning_preferences.json"

    assert is_uncalibrated_microphone_warning_suppressed(path=path) is False
    assert path.exists() is False


@pytest.mark.parametrize("value", [True, False])
def test_loads_strict_versioned_boolean(tmp_path, value):
    path = tmp_path / "analysis_warning_preferences.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "suppress_uncalibrated_microphone_warning": value,
            }
        ),
        encoding="utf-8",
    )

    assert is_uncalibrated_microphone_warning_suppressed(path=path) is value


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"version": 1},
        {"version": True, "suppress_uncalibrated_microphone_warning": True},
        {"version": 1.0, "suppress_uncalibrated_microphone_warning": True},
        {"version": 2, "suppress_uncalibrated_microphone_warning": True},
        {"version": 1, "suppress_uncalibrated_microphone_warning": 1},
        {"version": 1, "suppress_uncalibrated_microphone_warning": 0},
        {"version": 1, "suppress_uncalibrated_microphone_warning": "true"},
    ],
)
def test_invalid_schema_defaults_to_unsuppressed_and_logs(tmp_path, payload):
    path = tmp_path / "analysis_warning_preferences.json"
    original_bytes = json.dumps(payload).encode("utf-8")
    path.write_bytes(original_bytes)
    logger = Mock()

    assert (
        is_uncalibrated_microphone_warning_suppressed(path=path, logger=logger)
        is False
    )
    assert path.read_bytes() == original_bytes
    logger.warning.assert_called_once()
    assert str(path) in logger.warning.call_args.args[0]


def test_invalid_json_defaults_to_unsuppressed_without_modifying_file_and_logs(tmp_path):
    path = tmp_path / "analysis_warning_preferences.json"
    original_bytes = b"{invalid json"
    path.write_bytes(original_bytes)
    logger = Mock()

    assert (
        is_uncalibrated_microphone_warning_suppressed(path=path, logger=logger)
        is False
    )
    assert path.read_bytes() == original_bytes
    logger.warning.assert_called_once()
    assert str(path) in logger.warning.call_args.args[0]


def test_invalid_utf8_defaults_to_unsuppressed_without_modifying_file_and_logs(tmp_path):
    path = tmp_path / "analysis_warning_preferences.json"
    original_bytes = b"\xff\xfe\xfa"
    path.write_bytes(original_bytes)
    logger = Mock()

    assert (
        is_uncalibrated_microphone_warning_suppressed(path=path, logger=logger)
        is False
    )
    assert path.read_bytes() == original_bytes
    logger.warning.assert_called_once()
    assert str(path) in logger.warning.call_args.args[0]


def test_file_disappearing_before_open_defaults_to_unsuppressed_without_logging(tmp_path):
    path = tmp_path / "analysis_warning_preferences.json"
    original_bytes = b'{"version":1}'
    path.write_bytes(original_bytes)
    logger = Mock()

    with patch("builtins.open", side_effect=FileNotFoundError("file disappeared")):
        result = is_uncalibrated_microphone_warning_suppressed(
            path=path,
            logger=logger,
        )

    assert result is False
    assert path.read_bytes() == original_bytes
    logger.warning.assert_not_called()


def test_read_failure_defaults_to_unsuppressed_without_modifying_file_and_logs(tmp_path):
    path = tmp_path / "analysis_warning_preferences.json"
    original_bytes = b'{"version":1}'
    path.write_bytes(original_bytes)
    logger = Mock()

    with patch("builtins.open", side_effect=OSError("read failed")):
        result = is_uncalibrated_microphone_warning_suppressed(
            path=path,
            logger=logger,
        )

    assert result is False
    assert path.read_bytes() == original_bytes
    logger.warning.assert_called_once()
    assert str(path) in logger.warning.call_args.args[0]
    assert "read failed" in logger.warning.call_args.args[0]


def test_save_creates_parents_and_writes_versioned_true_payload(tmp_path):
    path = tmp_path / "ui" / "ui_config" / "analysis_warning_preferences.json"
    logger = Mock()

    result = save_uncalibrated_microphone_warning_suppressed(
        path=path,
        logger=logger,
    )

    assert result is True
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "version": 1,
        "suppress_uncalibrated_microphone_warning": True,
    }
    assert is_uncalibrated_microphone_warning_suppressed(path=path) is True
    logger.warning.assert_not_called()


def test_replace_failure_preserves_existing_file_and_removes_temporary_file(tmp_path):
    path = tmp_path / "ui" / "ui_config" / "analysis_warning_preferences.json"
    path.parent.mkdir(parents=True)
    original_bytes = json.dumps(
        {
            "version": 1,
            "suppress_uncalibrated_microphone_warning": False,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(original_bytes)
    logger = Mock()

    with patch.object(preferences.os, "replace", side_effect=OSError("replace failed")):
        result = save_uncalibrated_microphone_warning_suppressed(
            path=path,
            logger=logger,
        )

    assert result is False
    assert path.read_bytes() == original_bytes
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []
    logger.warning.assert_called_once()
    assert str(path) in logger.warning.call_args.args[0]
    assert "replace failed" in logger.warning.call_args.args[0]


def test_write_failure_preserves_existing_file_and_removes_temporary_file(tmp_path):
    path = tmp_path / "ui" / "ui_config" / "analysis_warning_preferences.json"
    path.parent.mkdir(parents=True)
    original_bytes = json.dumps(
        {
            "version": 1,
            "suppress_uncalibrated_microphone_warning": False,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(original_bytes)
    logger = Mock()

    with patch.object(preferences.json, "dump", side_effect=OSError("write failed")):
        result = save_uncalibrated_microphone_warning_suppressed(
            path=path,
            logger=logger,
        )

    assert result is False
    assert path.read_bytes() == original_bytes
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []
    logger.warning.assert_called_once()
    assert str(path) in logger.warning.call_args.args[0]
    assert "write failed" in logger.warning.call_args.args[0]


def test_cleanup_logs_descriptor_close_failure_without_modifying_target(tmp_path):
    target_path = tmp_path / "analysis_warning_preferences.json"
    original_bytes = b'{"version":1}'
    target_path.write_bytes(original_bytes)
    logger = Mock()

    with patch.object(preferences.os, "close", side_effect=OSError("close failed")):
        preferences._cleanup_temporary_file(123, None, target_path, logger)

    assert target_path.read_bytes() == original_bytes
    logger.warning.assert_called_once()
    assert str(target_path) in logger.warning.call_args.args[0]
    assert "close failed" in logger.warning.call_args.args[0]


def test_cleanup_logs_unlink_failure_without_modifying_target(tmp_path):
    target_path = tmp_path / "analysis_warning_preferences.json"
    original_bytes = b'{"version":1}'
    target_path.write_bytes(original_bytes)
    temp_path = tmp_path / f".{target_path.name}.leftover.tmp"
    temp_path.write_bytes(b"temporary")
    logger = Mock()

    try:
        with patch.object(
            preferences.os,
            "unlink",
            side_effect=OSError("unlink failed"),
        ):
            preferences._cleanup_temporary_file(
                None,
                temp_path,
                target_path,
                logger,
            )

        assert target_path.read_bytes() == original_bytes
        assert temp_path.exists() is True
        logger.warning.assert_called_once()
        assert str(temp_path) in logger.warning.call_args.args[0]
        assert "unlink failed" in logger.warning.call_args.args[0]
    finally:
        if temp_path.exists():
            os.unlink(temp_path)
