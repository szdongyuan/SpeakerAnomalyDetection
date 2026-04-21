import json
import logging
import sys
import types
from unittest.mock import Mock, patch

import pytest

if "concurrent_log_handler" not in sys.modules:
    concurrent_log_handler = types.ModuleType("concurrent_log_handler")

    class ConcurrentRotatingFileHandler(logging.Handler):
        def __init__(self, *args, **kwargs):
            super().__init__()

    concurrent_log_handler.ConcurrentRotatingFileHandler = ConcurrentRotatingFileHandler
    sys.modules["concurrent_log_handler"] = concurrent_log_handler

from base.load_config import LoadUiConfig


class TestSnRegexRules(object):

    def test_load_sn_regex_rules_creates_default_when_missing(self, tmp_path):
        json_file_path = tmp_path / "ui" / "ui_config" / "sn_regex_rules.json"

        result = LoadUiConfig.load_sn_regex_rules_from_json(str(json_file_path))
        default_payload = LoadUiConfig.build_default_sn_regex_rules_payload()

        assert result == default_payload
        assert json_file_path.exists()
        assert json.loads(json_file_path.read_text(encoding="utf-8")) == default_payload

    def test_load_sn_regex_rules_logs_error_when_missing_file_writeback_fails(self, tmp_path):
        json_file_path = tmp_path / "ui" / "ui_config" / "sn_regex_rules.json"
        default_payload = LoadUiConfig.build_default_sn_regex_rules_payload()
        mock_logger = Mock()

        with (
            patch.object(LoadUiConfig, "save_sn_regex_rules_to_json", return_value=False) as mock_save,
            patch.object(LoadUiConfig, "_get_sn_regex_rules_logger", return_value=mock_logger),
        ):
            result = LoadUiConfig.load_sn_regex_rules_from_json(str(json_file_path))

        assert result == default_payload
        mock_save.assert_called_once_with(default_payload, str(json_file_path))
        mock_logger.error.assert_called_once()
        assert "Failed to persist recovered default SN regex rules" in mock_logger.error.call_args[0][0]

    @pytest.mark.parametrize(
        "selected_rule_id",
        [
            "missing-rule",
            ["missing-rule"],
            {"id": "missing-rule"},
        ],
    )
    def test_load_sn_regex_rules_recovers_from_invalid_selected_rule(self, tmp_path, selected_rule_id):
        json_file_path = tmp_path / "sn_regex_rules.json"
        default_payload = LoadUiConfig.build_default_sn_regex_rules_payload()
        custom_payload = {
            "version": 1,
            "selected_rule_id": selected_rule_id,
            "rules": [
                default_payload["rules"][0],
                {
                    "id": "custom-rule",
                    "name": "自定义规则",
                    "pattern": r"^SN-\d+$",
                    "is_default": False,
                },
            ],
        }
        json_file_path.write_text(
            json.dumps(custom_payload, indent=6, ensure_ascii=False),
            encoding="utf-8",
        )

        result = LoadUiConfig.load_sn_regex_rules_from_json(str(json_file_path))
        selected_rule = LoadUiConfig.get_selected_sn_regex_rule(result)

        assert result["selected_rule_id"] == "default-match-all"
        assert selected_rule["id"] == "default-match-all"
        assert json.loads(json_file_path.read_text(encoding="utf-8"))["selected_rule_id"] == "default-match-all"

    def test_load_sn_regex_rules_recovers_from_invalid_json(self, tmp_path):
        json_file_path = tmp_path / "sn_regex_rules.json"
        default_payload = LoadUiConfig.build_default_sn_regex_rules_payload()
        json_file_path.write_text("{invalid json", encoding="utf-8")

        result = LoadUiConfig.load_sn_regex_rules_from_json(str(json_file_path))

        assert result == default_payload
        assert json.loads(json_file_path.read_text(encoding="utf-8")) == default_payload

    def test_load_sn_regex_rules_logs_when_json_load_fails(self, tmp_path):
        json_file_path = tmp_path / "sn_regex_rules.json"
        default_payload = LoadUiConfig.build_default_sn_regex_rules_payload()
        mock_logger = Mock()
        json_file_path.write_text("{invalid json", encoding="utf-8")

        with patch.object(LoadUiConfig, "_get_sn_regex_rules_logger", return_value=mock_logger):
            result = LoadUiConfig.load_sn_regex_rules_from_json(str(json_file_path))

        assert result == default_payload
        mock_logger.warning.assert_called_once()
        assert "Failed to load SN regex rules" in mock_logger.warning.call_args[0][0]

    def test_load_sn_regex_rules_recovers_from_empty_rules(self, tmp_path):
        json_file_path = tmp_path / "sn_regex_rules.json"
        default_payload = LoadUiConfig.build_default_sn_regex_rules_payload()
        json_file_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "selected_rule_id": "default-match-all",
                    "rules": [],
                },
                indent=6,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        result = LoadUiConfig.load_sn_regex_rules_from_json(str(json_file_path))

        assert result == default_payload
        assert json.loads(json_file_path.read_text(encoding="utf-8")) == default_payload

    def test_load_sn_regex_rules_recovers_from_invalid_regex_pattern(self, tmp_path):
        json_file_path = tmp_path / "sn_regex_rules.json"
        default_payload = LoadUiConfig.build_default_sn_regex_rules_payload()
        payload_with_invalid_rule = {
            "version": 1,
            "selected_rule_id": "broken-rule",
            "rules": [
                default_payload["rules"][0],
                {
                    "id": "customer-rule",
                    "name": "客户规则",
                    "pattern": r"^SN-\d{6}$",
                    "is_default": False,
                },
                {
                    "id": "broken-rule",
                    "name": "坏规则",
                    "pattern": r"[",
                    "is_default": False,
                },
            ],
        }
        json_file_path.write_text(
            json.dumps(payload_with_invalid_rule, indent=6, ensure_ascii=False),
            encoding="utf-8",
        )

        result = LoadUiConfig.load_sn_regex_rules_from_json(str(json_file_path))
        selected_rule = LoadUiConfig.get_selected_sn_regex_rule(result)

        assert result["selected_rule_id"] == "default-match-all"
        assert [rule["id"] for rule in result["rules"]] == ["default-match-all", "customer-rule"]
        assert selected_rule["id"] == "default-match-all"
        assert json.loads(json_file_path.read_text(encoding="utf-8")) == result

    def test_load_sn_regex_rules_recovers_from_pure_literal_regex_pattern(self, tmp_path):
        json_file_path = tmp_path / "sn_regex_rules.json"
        default_payload = LoadUiConfig.build_default_sn_regex_rules_payload()
        payload_with_literal_rule = {
            "version": 1,
            "selected_rule_id": "literal-rule",
            "rules": [
                default_payload["rules"][0],
                {
                    "id": "customer-rule",
                    "name": "客户规则",
                    "pattern": r"^SN-\d{6}$",
                    "is_default": False,
                },
                {
                    "id": "literal-rule",
                    "name": "固定文本规则",
                    "pattern": r"^321321$",
                    "is_default": False,
                },
            ],
        }
        json_file_path.write_text(
            json.dumps(payload_with_literal_rule, indent=6, ensure_ascii=False),
            encoding="utf-8",
        )

        result = LoadUiConfig.load_sn_regex_rules_from_json(str(json_file_path))
        selected_rule = LoadUiConfig.get_selected_sn_regex_rule(result)

        assert result["selected_rule_id"] == "default-match-all"
        assert [rule["id"] for rule in result["rules"]] == ["default-match-all", "customer-rule"]
        assert selected_rule["id"] == "default-match-all"
        assert json.loads(json_file_path.read_text(encoding="utf-8")) == result

    def test_load_sn_regex_rules_restores_tampered_default_rule_and_logs_writeback(self, tmp_path):
        json_file_path = tmp_path / "sn_regex_rules.json"
        default_payload = LoadUiConfig.build_default_sn_regex_rules_payload()
        mock_logger = Mock()
        tampered_default_payload = {
            "version": 1,
            "selected_rule_id": "customer-rule",
            "rules": [
                {
                    "id": "default-match-all",
                    "name": "被篡改的默认规则",
                    "pattern": r"^SN-\d{6}$",
                    "is_default": False,
                },
                {
                    "id": "customer-rule",
                    "name": "客户规则",
                    "pattern": r"^CUST-\d{4}$",
                    "is_default": False,
                },
            ],
        }
        json_file_path.write_text(
            json.dumps(tampered_default_payload, indent=6, ensure_ascii=False),
            encoding="utf-8",
        )

        with patch.object(LoadUiConfig, "_get_sn_regex_rules_logger", return_value=mock_logger):
            result = LoadUiConfig.load_sn_regex_rules_from_json(str(json_file_path))

        saved_payload = json.loads(json_file_path.read_text(encoding="utf-8"))
        default_rule = next(rule for rule in result["rules"] if rule["id"] == "default-match-all")
        custom_rule = next(rule for rule in result["rules"] if rule["id"] == "customer-rule")

        assert default_rule == default_payload["rules"][0]
        assert custom_rule == tampered_default_payload["rules"][1]
        assert result["selected_rule_id"] == "customer-rule"
        assert saved_payload == result
        mock_logger.warning.assert_called_once()
        assert "Normalized SN regex rules" in mock_logger.warning.call_args[0][0]
        assert "restored built-in default rule definition" in mock_logger.warning.call_args[0][0]

    def test_load_sn_regex_rules_resets_tampered_custom_is_default_flag_and_persists(self, tmp_path):
        json_file_path = tmp_path / "sn_regex_rules.json"
        default_payload = LoadUiConfig.build_default_sn_regex_rules_payload()
        mock_logger = Mock()
        tampered_payload = {
            "version": 1,
            "selected_rule_id": "customer-rule",
            "rules": [
                default_payload["rules"][0],
                {
                    "id": "customer-rule",
                    "name": "客户规则",
                    "pattern": r"^CUST-\d{4}$",
                    "is_default": True,
                },
            ],
        }
        json_file_path.write_text(
            json.dumps(tampered_payload, indent=6, ensure_ascii=False),
            encoding="utf-8",
        )

        with patch.object(LoadUiConfig, "_get_sn_regex_rules_logger", return_value=mock_logger):
            result = LoadUiConfig.load_sn_regex_rules_from_json(str(json_file_path))

        saved_payload = json.loads(json_file_path.read_text(encoding="utf-8"))
        custom_rule = next(rule for rule in result["rules"] if rule["id"] == "customer-rule")
        default_rule = next(rule for rule in result["rules"] if rule["id"] == "default-match-all")

        assert result["selected_rule_id"] == "customer-rule"
        assert default_rule == default_payload["rules"][0]
        assert custom_rule["name"] == "客户规则"
        assert custom_rule["pattern"] == r"^CUST-\d{4}$"
        assert custom_rule["is_default"] is False
        assert saved_payload == result
        mock_logger.warning.assert_called_once()
        assert "Normalized SN regex rules" in mock_logger.warning.call_args[0][0]
        assert "reset non-default rule 'customer-rule' is_default flag" in mock_logger.warning.call_args[0][0]

    def test_save_sn_regex_rules_to_json_and_get_selected_rule(self, tmp_path):
        json_file_path = tmp_path / "ui" / "ui_config" / "sn_regex_rules.json"
        default_payload = LoadUiConfig.build_default_sn_regex_rules_payload()
        custom_payload = {
            "version": 1,
            "selected_rule_id": "custom-rule",
            "rules": [
                default_payload["rules"][0],
                {
                    "id": "custom-rule",
                    "name": "客户规则",
                    "pattern": r"^SN-\d{6}$",
                    "is_default": False,
                },
            ],
        }

        save_result = LoadUiConfig.save_sn_regex_rules_to_json(custom_payload, str(json_file_path))
        saved_payload = json.loads(json_file_path.read_text(encoding="utf-8"))
        selected_rule = LoadUiConfig.get_selected_sn_regex_rule(saved_payload)

        assert save_result is True
        assert saved_payload == custom_payload
        assert selected_rule["id"] == "custom-rule"
        assert selected_rule["pattern"] == r"^SN-\d{6}$"

    def test_save_sn_regex_rules_to_json_rejects_invalid_payload_without_overwriting_existing_file(self, tmp_path):
        json_file_path = tmp_path / "ui" / "ui_config" / "sn_regex_rules.json"
        default_payload = LoadUiConfig.build_default_sn_regex_rules_payload()
        existing_payload = {
            "version": 1,
            "selected_rule_id": "customer-rule",
            "rules": [
                default_payload["rules"][0],
                {
                    "id": "customer-rule",
                    "name": "客户规则",
                    "pattern": r"^SN-\d{4}$",
                    "is_default": False,
                },
            ],
        }
        json_file_path.parent.mkdir(parents=True, exist_ok=True)
        json_file_path.write_text(
            json.dumps(existing_payload, indent=6, ensure_ascii=False),
            encoding="utf-8",
        )
        invalid_payload = {
            "version": 1,
            "selected_rule_id": "broken-rule",
            "rules": [
                default_payload["rules"][0],
                {
                    "id": "broken-rule",
                    "name": "坏规则",
                    "pattern": r"[",
                    "is_default": False,
                },
            ],
        }

        save_result = LoadUiConfig.save_sn_regex_rules_to_json(invalid_payload, str(json_file_path))
        saved_payload = json.loads(json_file_path.read_text(encoding="utf-8"))

        assert save_result is False
        assert saved_payload == existing_payload

    def test_save_sn_regex_rules_to_json_keeps_existing_file_when_temp_write_fails(self, tmp_path):
        json_file_path = tmp_path / "ui" / "ui_config" / "sn_regex_rules.json"
        default_payload = LoadUiConfig.build_default_sn_regex_rules_payload()
        existing_payload = {
            "version": 1,
            "selected_rule_id": "customer-rule",
            "rules": [
                default_payload["rules"][0],
                {
                    "id": "customer-rule",
                    "name": "客户规则",
                    "pattern": r"^SN-\d{4}$",
                    "is_default": False,
                },
            ],
        }
        next_payload = {
            "version": 1,
            "selected_rule_id": "next-rule",
            "rules": [
                default_payload["rules"][0],
                {
                    "id": "next-rule",
                    "name": "新规则",
                    "pattern": r"^NEXT-\d{4}$",
                    "is_default": False,
                },
            ],
        }
        json_file_path.parent.mkdir(parents=True, exist_ok=True)
        json_file_path.write_text(
            json.dumps(existing_payload, indent=6, ensure_ascii=False),
            encoding="utf-8",
        )

        with patch("base.load_config.json.dump", side_effect=OSError("write failed")):
            save_result = LoadUiConfig.save_sn_regex_rules_to_json(next_payload, str(json_file_path))

        saved_payload = json.loads(json_file_path.read_text(encoding="utf-8"))

        assert save_result is False
        assert saved_payload == existing_payload
        assert list(json_file_path.parent.glob(".sn_regex_rules.json.*.tmp")) == []

    def test_save_sn_regex_rules_to_json_keeps_existing_file_when_atomic_replace_fails(self, tmp_path):
        json_file_path = tmp_path / "ui" / "ui_config" / "sn_regex_rules.json"
        default_payload = LoadUiConfig.build_default_sn_regex_rules_payload()
        existing_payload = {
            "version": 1,
            "selected_rule_id": "customer-rule",
            "rules": [
                default_payload["rules"][0],
                {
                    "id": "customer-rule",
                    "name": "客户规则",
                    "pattern": r"^SN-\d{4}$",
                    "is_default": False,
                },
            ],
        }
        next_payload = {
            "version": 1,
            "selected_rule_id": "next-rule",
            "rules": [
                default_payload["rules"][0],
                {
                    "id": "next-rule",
                    "name": "新规则",
                    "pattern": r"^NEXT-\d{4}$",
                    "is_default": False,
                },
            ],
        }
        json_file_path.parent.mkdir(parents=True, exist_ok=True)
        json_file_path.write_text(
            json.dumps(existing_payload, indent=6, ensure_ascii=False),
            encoding="utf-8",
        )

        with patch("base.load_config.os.replace", side_effect=OSError("replace failed")):
            save_result = LoadUiConfig.save_sn_regex_rules_to_json(next_payload, str(json_file_path))

        saved_payload = json.loads(json_file_path.read_text(encoding="utf-8"))

        assert save_result is False
        assert saved_payload == existing_payload
        assert list(json_file_path.parent.glob(".sn_regex_rules.json.*.tmp")) == []

    def test_can_compile_sn_regex_pattern_rejects_invalid_regex(self):
        assert LoadUiConfig.can_compile_sn_regex_pattern(r"[") is False
        assert LoadUiConfig.can_compile_sn_regex_pattern(r"^.+$") is True

    @pytest.mark.parametrize("pattern", [r"321321", r"^321321$"])
    def test_is_pure_literal_sn_regex_pattern_detects_literal_only_patterns(self, pattern):
        assert LoadUiConfig.is_pure_literal_sn_regex_pattern(pattern) is True

    @pytest.mark.parametrize("pattern", [r"^SN-\d+$", r"SN.*", r"[A-Z]+"])
    def test_is_pure_literal_sn_regex_pattern_allows_real_regex_features(self, pattern):
        assert LoadUiConfig.is_pure_literal_sn_regex_pattern(pattern) is False

    def test_save_sn_regex_rules_to_json_rejects_pure_literal_pattern(self, tmp_path):
        json_file_path = tmp_path / "ui" / "ui_config" / "sn_regex_rules.json"
        default_payload = LoadUiConfig.build_default_sn_regex_rules_payload()
        invalid_payload = {
            "version": 1,
            "selected_rule_id": "literal-rule",
            "rules": [
                default_payload["rules"][0],
                {
                    "id": "literal-rule",
                    "name": "固定文本规则",
                    "pattern": r"^321321$",
                    "is_default": False,
                },
            ],
        }

        save_result = LoadUiConfig.save_sn_regex_rules_to_json(invalid_payload, str(json_file_path))

        assert save_result is False
        assert json_file_path.exists() is False
