import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QPushButton

from base.product_test_project_config import is_manual_project_play_allowed
from ui.sequence import sequence_widget_serial_trigger_ops as serial_ops_module
from ui.sequence.sequence_widget_serial_trigger_ops import (
    SequenceWidgetSerialTriggerOpsMixin,
)
from ui.sequence.sequence_widget_ui_ops import SequenceWidgetUiOpsMixin


_APP = QApplication.instance() or QApplication([])


EMPTY_CONDITIONS = [
    {"condition_name": "6000", "trigger_state": ""},
    {"condition_name": "7000", "trigger_state": ""},
]
COMPLETE_CONDITIONS = [
    {"condition_name": "6000", "trigger_state": "01"},
    {"condition_name": "7000", "trigger_state": "02"},
]
MIXED_CONDITIONS = [
    {"condition_name": "6000", "trigger_state": "01"},
    {"condition_name": "7000", "trigger_state": ""},
]


class _ButtonSpy:
    def __init__(self):
        self.disabled = None
        self.tooltip = ""

    def setIcon(self, _icon):
        return None

    def setIconSize(self, _size):
        return None

    def setDisabled(self, disabled):
        self.disabled = bool(disabled)

    def setToolTip(self, tooltip):
        self.tooltip = str(tooltip)


class _PlaybackPermissionHost(SequenceWidgetUiOpsMixin):
    def __init__(self, condition_configs, serial_enabled=False, player_btn=None):
        self.toolsbar = SimpleNamespace(
            player_btn=player_btn or _ButtonSpy(),
            replayer_btn=_ButtonSpy(),
            data_btn=_ButtonSpy(),
        )
        self.product_test_condition_configs = [
            dict(condition) for condition in condition_configs
        ]
        self.product_test_close_trigger_state = ""
        self.player_status_flag = False
        self._record_workflow_busy = False
        self._serial_trigger_config = {"enabled": bool(serial_enabled)}
        self.data_struct = SimpleNamespace(
            store_wave_data="recorded",
            store_wave_data_multi="recorded_multi",
            wav_calibration_metadata={"old": True},
            wav_calibration_metadata_authoritative=True,
            wav_calibration_warning_shown=True,
        )

    def _load_sequence_config_for_product_condition(self, _condition_config):
        return True, ""

    def clear_all_direction_waveforms(self):
        return None

    def _close_analysis_windows(self):
        return None

    def _reset_barcode_commit_dedup(self):
        return None


@pytest.mark.parametrize(
    ("conditions", "expected"),
    [
        (EMPTY_CONDITIONS, True),
        (COMPLETE_CONDITIONS, False),
        (MIXED_CONDITIONS, False),
    ],
)
def test_manual_playback_permission_matrix(conditions, expected):
    assert is_manual_project_play_allowed(conditions) is expected


@pytest.mark.parametrize(
    ("conditions", "serial_enabled", "expected_enabled", "tooltip_text"),
    [
        (EMPTY_CONDITIONS, False, True, "开始录制"),
        (EMPTY_CONDITIONS, True, True, "开始录制"),
        (COMPLETE_CONDITIONS, False, False, "只能由状态码触发测试"),
        (COMPLETE_CONDITIONS, True, False, "只能由状态码触发测试"),
        (MIXED_CONDITIONS, False, False, "必须全部配置或全部留空"),
        (MIXED_CONDITIONS, True, False, "必须全部配置或全部留空"),
    ],
)
def test_play_button_matches_permission_matrix(
    conditions,
    serial_enabled,
    expected_enabled,
    tooltip_text,
):
    host = _PlaybackPermissionHost(conditions, serial_enabled)

    host.update_player_btn_is_paused()

    assert host.player_btn.disabled is not expected_enabled
    assert tooltip_text in host.player_btn.tooltip


def test_actual_qt_button_ignores_serial_toggle_for_complete_config():
    player_btn = QPushButton()
    host = _PlaybackPermissionHost(
        COMPLETE_CONDITIONS,
        serial_enabled=True,
        player_btn=player_btn,
    )

    host.update_player_btn_is_paused()
    assert player_btn.isEnabled() is False

    host._serial_trigger_config = {"enabled": False}
    host.update_player_btn_is_paused()
    assert player_btn.isEnabled() is False


def test_actual_qt_button_allows_empty_codes_when_serial_is_enabled():
    player_btn = QPushButton()
    host = _PlaybackPermissionHost(
        EMPTY_CONDITIONS,
        serial_enabled=True,
        player_btn=player_btn,
    )

    host.update_player_btn_is_paused()

    assert player_btn.isEnabled() is True
    assert player_btn.toolTip() == "开始录制：6000"


def test_complete_config_stays_disabled_without_serial_config():
    host = _PlaybackPermissionHost(COMPLETE_CONDITIONS)
    del host._serial_trigger_config

    host.update_player_btn_is_paused()

    assert host.player_btn.disabled is True
    assert "只能由状态码触发测试" in host.player_btn.tooltip


@pytest.mark.parametrize("busy_attribute", ["player_status_flag", "_record_workflow_busy"])
def test_runtime_busy_state_still_disables_manual_play(busy_attribute):
    host = _PlaybackPermissionHost(EMPTY_CONDITIONS, serial_enabled=False)
    setattr(host, busy_attribute, True)

    host.update_player_btn_is_paused()

    assert host.player_btn.disabled is True


def test_close_frame_does_not_change_empty_status_code_permission():
    host = _PlaybackPermissionHost(EMPTY_CONDITIONS, serial_enabled=True)
    host.product_test_close_trigger_state = "01 04 02 00 04 B8 F3"

    host.update_player_btn_is_paused()

    assert host.player_btn.disabled is False
    assert host.player_btn.tooltip == "开始录制：6000"


def test_play_button_tooltip_follows_next_queue_condition():
    host = _PlaybackPermissionHost(EMPTY_CONDITIONS, serial_enabled=False)
    host._manual_product_condition_index = 1

    host.update_player_btn_is_paused()

    assert host.player_btn.disabled is False
    assert host.player_btn.tooltip == "开始录制：7000"


def test_mark_reset_does_not_bypass_playback_permission():
    host = _PlaybackPermissionHost(COMPLETE_CONDITIONS, serial_enabled=True)

    host.on_mark_btn_clicked()

    assert host.player_btn.disabled is True
    assert host.replayer_btn.disabled is True
    assert host.data_btn.disabled is True
    assert host.data_struct.wav_calibration_metadata is None
    assert host.data_struct.wav_calibration_metadata_authoritative is False
    assert host.data_struct.wav_calibration_warning_shown is False


def test_disabling_serial_trigger_does_not_enable_complete_config(monkeypatch):
    class _Dialog:
        def __init__(self, *_args, **_kwargs):
            return None

        def exec(self):
            return "save", {"enabled": False}

    class _HardwareManager:
        def __init__(self):
            self.stop_calls = 0

        def get_serial_discrete_input_status(self):
            return {"running": False, "connected": False}

        def stop_serial_discrete_input_listener(self):
            self.stop_calls += 1

    monkeypatch.setattr(
        serial_ops_module,
        "SerialDiscreteInputConfigDialog",
        _Dialog,
    )
    monkeypatch.setattr(
        serial_ops_module.LoadUiConfig,
        "save_serial_discrete_input_config",
        staticmethod(lambda _config: True),
    )
    host = _PlaybackPermissionHost(COMPLETE_CONDITIONS, serial_enabled=True)
    host.hw_manager = _HardwareManager()
    host._test_serial_trigger_connection = lambda _config: {}
    host.on_serial_trigger_status_changed = lambda _status: None
    host.update_player_btn_is_paused()
    assert host.player_btn.disabled is True

    SequenceWidgetSerialTriggerOpsMixin.on_serial_trigger_btn_clicked(host)

    assert host._serial_trigger_config["enabled"] is False
    assert host.hw_manager.stop_calls == 1
    assert host.player_btn.disabled is True


def test_serial_config_save_failure_preserves_permission_and_runtime(monkeypatch):
    class _Dialog:
        def __init__(self, *_args, **_kwargs):
            return None

        def exec(self):
            return "save", {"enabled": False}

    class _HardwareManager:
        def __init__(self):
            self.stop_calls = 0

        def get_serial_discrete_input_status(self):
            return {"running": True, "connected": True}

        def stop_serial_discrete_input_listener(self):
            self.stop_calls += 1

    warnings = []
    monkeypatch.setattr(
        serial_ops_module,
        "SerialDiscreteInputConfigDialog",
        _Dialog,
    )
    monkeypatch.setattr(
        serial_ops_module.LoadUiConfig,
        "save_serial_discrete_input_config",
        staticmethod(lambda _config: False),
    )
    monkeypatch.setattr(
        serial_ops_module.QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )
    host = _PlaybackPermissionHost(COMPLETE_CONDITIONS, serial_enabled=True)
    host.hw_manager = _HardwareManager()
    host._test_serial_trigger_connection = lambda _config: {}
    host.on_serial_trigger_status_changed = lambda _status: None
    host.update_player_btn_is_paused()

    SequenceWidgetSerialTriggerOpsMixin.on_serial_trigger_btn_clicked(host)

    assert host._serial_trigger_config["enabled"] is True
    assert host.hw_manager.stop_calls == 0
    assert host.player_btn.disabled is True
    assert warnings == [("保存失败", "无法保存串口离散输入触发配置。")]
