import json
from unittest.mock import Mock

import pytest

import base.input_calibration_preferences as preferences
from base.input_calibration_preferences import (
    load_input_calibration_mode,
    save_input_calibration_mode,
)
from consts.calibration_consts import (
    INPUT_CALIBRATION_MODE_MANUAL,
    INPUT_CALIBRATION_MODE_STANDARD_SPL,
)


def test_missing_preference_defaults_to_standard_without_creating_file(tmp_path):
    path = tmp_path / "input_calibration_preferences.json"
    logger = Mock()

    assert load_input_calibration_mode(path=path, logger=logger) == INPUT_CALIBRATION_MODE_STANDARD_SPL
    assert not path.exists()
    logger.warning.assert_not_called()


@pytest.mark.parametrize(
    "mode",
    [INPUT_CALIBRATION_MODE_STANDARD_SPL, INPUT_CALIBRATION_MODE_MANUAL],
)
def test_legal_preference_loads(tmp_path, mode):
    path = tmp_path / "input_calibration_preferences.json"
    path.write_text(json.dumps({"version": 1, "mode": mode}), encoding="utf-8")

    assert load_input_calibration_mode(path=path) == mode


@pytest.mark.parametrize(
    "contents",
    [
        "{not-json",
        "[]",
        '{"version": 2, "mode": "manual"}',
        '{"version": true, "mode": "manual"}',
        '{"version": 1, "mode": 1}',
        '{"version": 1, "mode": "unknown"}',
    ],
)
def test_invalid_preference_defaults_to_standard_and_warns(tmp_path, contents):
    path = tmp_path / "input_calibration_preferences.json"
    path.write_text(contents, encoding="utf-8")
    logger = Mock()

    result = load_input_calibration_mode(path=path, logger=logger)

    assert result == INPUT_CALIBRATION_MODE_STANDARD_SPL
    logger.warning.assert_called_once()


def test_unreadable_preference_defaults_to_standard_and_warns(tmp_path, monkeypatch):
    path = tmp_path / "input_calibration_preferences.json"
    path.write_text('{"version": 1, "mode": "manual"}', encoding="utf-8")
    logger = Mock()
    original_open = open

    def failing_open(target, *args, **kwargs):
        if str(target) == str(path):
            raise OSError("read failed")
        return original_open(target, *args, **kwargs)

    monkeypatch.setattr("builtins.open", failing_open)

    assert load_input_calibration_mode(path=path, logger=logger) == INPUT_CALIBRATION_MODE_STANDARD_SPL
    logger.warning.assert_called_once()


@pytest.mark.parametrize(
    "mode",
    [INPUT_CALIBRATION_MODE_MANUAL, INPUT_CALIBRATION_MODE_STANDARD_SPL],
)
def test_save_creates_parent_and_round_trips_mode(tmp_path, mode):
    path = tmp_path / "ui" / "ui_config" / "input_calibration_preferences.json"

    assert save_input_calibration_mode(mode, path=path)
    assert load_input_calibration_mode(path=path) == mode
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 1, "mode": mode}


def test_save_overwrites_previous_mode(tmp_path):
    path = tmp_path / "input_calibration_preferences.json"

    assert save_input_calibration_mode(INPUT_CALIBRATION_MODE_STANDARD_SPL, path=path)
    assert save_input_calibration_mode(INPUT_CALIBRATION_MODE_MANUAL, path=path)

    assert load_input_calibration_mode(path=path) == INPUT_CALIBRATION_MODE_MANUAL


def test_save_rejects_unsupported_caller_mode_without_touching_file(tmp_path):
    path = tmp_path / "input_calibration_preferences.json"

    with pytest.raises(ValueError, match="Unsupported input calibration mode"):
        save_input_calibration_mode("unknown", path=path)

    assert not path.exists()


def test_replace_failure_preserves_existing_file_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "input_calibration_preferences.json"
    existing_payload = {"version": 1, "mode": INPUT_CALIBRATION_MODE_STANDARD_SPL}
    path.write_text(json.dumps(existing_payload), encoding="utf-8")
    logger = Mock()

    def fail_replace(*_args):
        raise OSError("replace failed")

    monkeypatch.setattr(preferences.os, "replace", fail_replace)

    assert save_input_calibration_mode(INPUT_CALIBRATION_MODE_MANUAL, path=path, logger=logger) is False
    assert json.loads(path.read_text(encoding="utf-8")) == existing_payload
    assert list(tmp_path.glob(".input_calibration_preferences.json.*.tmp")) == []
    logger.warning.assert_called_once()


def test_dump_failure_preserves_existing_file_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "input_calibration_preferences.json"
    existing_payload = {"version": 1, "mode": INPUT_CALIBRATION_MODE_STANDARD_SPL}
    path.write_text(json.dumps(existing_payload), encoding="utf-8")
    logger = Mock()

    def fail_dump(*_args, **_kwargs):
        raise OSError("write failed")

    monkeypatch.setattr(preferences.json, "dump", fail_dump)

    assert save_input_calibration_mode(INPUT_CALIBRATION_MODE_MANUAL, path=path, logger=logger) is False
    assert json.loads(path.read_text(encoding="utf-8")) == existing_payload
    assert list(tmp_path.glob(".input_calibration_preferences.json.*.tmp")) == []
    logger.warning.assert_called_once()


def test_save_warning_does_not_expose_existing_file_contents(tmp_path, monkeypatch):
    path = tmp_path / "input_calibration_preferences.json"
    secret = "private-existing-content"
    path.write_text(secret, encoding="utf-8")
    logger = Mock()
    monkeypatch.setattr(preferences.os, "replace", Mock(side_effect=OSError("replace failed")))

    assert save_input_calibration_mode(INPUT_CALIBRATION_MODE_MANUAL, path=path, logger=logger) is False

    warning = logger.warning.call_args.args[0]
    assert secret not in warning
