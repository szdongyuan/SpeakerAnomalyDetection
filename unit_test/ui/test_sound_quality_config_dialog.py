"""Round-trip and semantic presentation of sound quality configuration."""
import pytest
from PyQt5.QtWidgets import QApplication

from ui.operation_sequence import SUPPORTED_ANALYSIS_TYPES
from ui.ui_analysis_config.common_widgets import MultiChannelSelectorWidget
from ui.ui_analysis_config.sound_quality_config_dialog import SoundQualityConfigWindow
from ui.sequence.workbench_ui_projection import _plot_unit
from unit_test.ui.test_loudness_config_dialog import _ConfigManager


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    return app


@pytest.mark.parametrize("kind,unit", [("SHRP", "acum"), ("ROUGH", "asper")])
def test_defaults_channel_selection_and_reload(qapp, kind, unit):
    manager = _ConfigManager({kind: {"analysis_channels": [0, 2], "analysis_channel": 0,
                                       "advanced": {"analysis_time_range_enabled": True,
                                                    "analysis_start_time_sec": 0.2, "analysis_end_time_sec": 2.8},
                                       "display": {"show_threshold_lines": False}}})
    window = SoundQualityConfigWindow(manager, kind, kind, [0, 1, 2], True)
    try:
        assert isinstance(window.channel_selector, MultiChannelSelectorWidget)
        assert window.semantic_group_keys() == ["input", "compute", "display", "judgment"]
        window.limit_enabled.setChecked(True)
        window.metric_combo.setCurrentIndex(1)
        window.scalar_controls["upper"][1].setValue(1.234)
        if kind == "ROUGH":
            window.overlap_spin.setValue(0.25)
        assert window.validate_config()
        saved = window.get_default_config()
        assert saved["analysis_channels"] == [0, 2]
        assert saved["advanced"]["analysis_end_time_sec"] == 2.8
        assert saved["scalar_upper_value"] == 1.234
        assert saved["display"]["show_threshold_lines"] is False
        manager.save_default_config(kind, saved)
        again = SoundQualityConfigWindow(manager, kind, kind, [0, 1, 2], True)
        try:
            assert again.get_default_config() == saved
        finally:
            again.close()
        assert kind in SUPPORTED_ANALYSIS_TYPES and _plot_unit({"type": kind}) == unit
    finally:
        window.close()


@pytest.mark.parametrize("kind", ["SHRP", "ROUGH"])
def test_legacy_empty_defaults_and_threshold_toggle(qapp, kind):
    window = SoundQualityConfigWindow(_ConfigManager({}), kind, kind)
    try:
        assert window.validate_config()
        assert window.get_default_config()["limit_checked"] is False
        window.limit_enabled.setChecked(True)
        assert window.validate_config()
        assert window.get_default_config()["limit_metric"] == "curve_y"
        window.metric_combo.setCurrentIndex(1)
        window.scalar_controls["upper"][1].setValue(0.5)
        window.metric_combo.setCurrentIndex(0)
        assert window.get_default_config()["scalar_upper_value"] == 0.5
        assert window.threshold_widget.constant_limit_unit == ("acum" if kind == "SHRP" else "asper")
    finally:
        window.close()


def test_sharpness_fixed_settings_are_not_exposed(qapp):
    window = SoundQualityConfigWindow(_ConfigManager({}), "SHRP", "SHRP")
    try:
        assert not hasattr(window, "method_combo")
        assert not hasattr(window, "advanced_toggle")
        saved = window.get_default_config()
        assert saved["method"] == "time_varying_iso532_1"
        assert saved["advanced"]["output_time_resolution_ms"] == 2.0
        assert "stationary_frame_duration_s" not in saved["advanced"]
        assert window.validate_config()
    finally:
        window.close()


@pytest.mark.parametrize("old", [
    {"method": "per_segment", "advanced": {"stationary_frame_duration_s": .2}},
    {"method": "time_varying_iso532_1", "advanced": {"output_time_resolution_ms": 4}},
])
def test_sharpness_legacy_conversion_requires_save_confirmation(qapp, monkeypatch, old):
    from ui.ui_analysis_config.sound_quality_config_dialog import MessageBox
    window = SoundQualityConfigWindow(_ConfigManager({"SHRP": old}), "SHRP", "SHRP")
    try:
        saved = window.get_default_config()
        assert saved["method"] == "time_varying_iso532_1"
        assert saved["advanced"]["output_time_resolution_ms"] == 2.0
        assert window.load_config["method"] == old["method"]
        answers = iter((MessageBox.No, MessageBox.Yes))
        monkeypatch.setattr(MessageBox, "question", lambda *args: next(answers))
        accepted = []
        monkeypatch.setattr(window, "accept_config", accepted.append)
        window.on_click_ok_btn()
        assert not accepted
        window.on_click_ok_btn()
        assert accepted == [saved]
    finally:
        window.close()


def test_sharpness_parameter_views_render(qapp, main_ui_font, tmp_path):
    window = SoundQualityConfigWindow(_ConfigManager({}), "SHRP", "SHRP", [0], True)
    try:
        window.show()
        qapp.processEvents()
        assert window.grab().save(str(tmp_path / "fixed_time_varying.png"))
    finally:
        window.close()
