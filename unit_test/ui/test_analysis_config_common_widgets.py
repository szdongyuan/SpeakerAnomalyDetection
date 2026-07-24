import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout

from consts.harmonic_detection_consts import (
    HARMONIC_DETECTION_METHOD_FOURIER,
    HARMONIC_DETECTION_METHOD_KEY,
    HARMONIC_DETECTION_METHOD_SYNCHRONOUS,
)
from ui.ui_analysis_config.common_widgets import (
    ChannelSelectorWidget,
    GoldenSampleWidget,
    HarmonicDetectionMethodSelectorWidget,
    HarmonicSelectorWidget,
    OctaveSmoothingSelectorWidget,
    SemanticAnalysisConfigDialogBase,
    TimeSmoothingWidget,
    WeightingSelectorWidget,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_channel_selector_uses_available_channel(qapp):
    widget = ChannelSelectorWidget({"analysis_channel": "2"}, [0, 2, 4])

    assert widget.current_channel() == 2
    assert widget.get_config() == {"analysis_channel": 2}


@pytest.mark.parametrize(
    ("cfg", "available_channels", "expected"),
    [
        ({"analysis_channel": "bad"}, [3, 5], 3),
        ({"analysis_channel": 4}, [0, 1], 0),
        ({}, [], 0),
        (None, None, 0),
    ],
)
def test_channel_selector_falls_back_safely(qapp, cfg, available_channels, expected):
    widget = ChannelSelectorWidget(cfg, available_channels)

    assert widget.current_channel() == expected


def test_weighting_selector_saves_z_for_none_display(qapp):
    widget = WeightingSelectorWidget({"weighting": "Z（None）"}, allowed_options=("Z", "A", "C"), default="A")

    assert widget.combo_box.currentText() == "Z（None）"
    assert widget.current_weighting() == "Z"
    assert widget.get_config() == {"weighting": "Z"}


def test_weighting_selector_respects_allowed_options(qapp):
    widget = WeightingSelectorWidget({"weighting": "B"}, allowed_options=("Z", "A", "C"), default="A")

    assert widget.current_weighting() == "A"


def test_harmonic_detection_method_selector_defaults_to_synchronous(qapp):
    widget = HarmonicDetectionMethodSelectorWidget({})

    assert widget.combo_box.currentText() == "同步检波"
    assert widget.current_method() == HARMONIC_DETECTION_METHOD_SYNCHRONOUS
    assert widget.get_config() == {HARMONIC_DETECTION_METHOD_KEY: HARMONIC_DETECTION_METHOD_SYNCHRONOUS}


def test_harmonic_detection_method_selector_loads_fourier(qapp):
    widget = HarmonicDetectionMethodSelectorWidget({HARMONIC_DETECTION_METHOD_KEY: "fourier"})

    assert widget.combo_box.currentText() == "傅里叶变换"
    assert widget.current_method() == HARMONIC_DETECTION_METHOD_FOURIER
    assert widget.get_config() == {HARMONIC_DETECTION_METHOD_KEY: HARMONIC_DETECTION_METHOD_FOURIER}


def test_harmonic_detection_method_selector_normalizes_invalid_saved_value(qapp):
    widget = HarmonicDetectionMethodSelectorWidget({HARMONIC_DETECTION_METHOD_KEY: "bad"})

    assert widget.current_method() == HARMONIC_DETECTION_METHOD_SYNCHRONOUS


def test_octave_smoothing_selector_reads_explicit_key(qapp):
    widget = OctaveSmoothingSelectorWidget({"octave_smoothing": 3})

    assert widget.current_octave_smoothing() == 3
    assert widget.get_config() == {"octave_smoothing": 3}


def test_octave_smoothing_selector_reads_legacy_smooth_checked(qapp):
    widget = OctaveSmoothingSelectorWidget({"smooth_checked": True})

    assert widget.current_octave_smoothing() == 6


def test_octave_smoothing_selector_supports_option_subset(qapp):
    widget = OctaveSmoothingSelectorWidget({"octave_smoothing": 12}, allowed_options=(0, 3, 6), default=0)

    assert widget.current_octave_smoothing() == 0


def test_time_smoothing_widget_outputs_legacy_keys(qapp):
    widget = TimeSmoothingWidget(
        {
            "smooth_enabled": True,
            "smooth_unit": "points",
            "smooth_time_sec": 0.125,
            "smooth_points": 32,
            "smooth_algo": 3,
        }
    )

    assert widget.get_config() == {
        "smooth_enabled": True,
        "smooth_unit": "points",
        "smooth_time_sec": 0.125,
        "smooth_points": 32,
        "smooth_algo": 3,
    }
    assert widget.time_spin.isHidden() is True
    assert widget.points_spin.isHidden() is False


def test_time_smoothing_widget_can_hide_algorithm(qapp):
    widget = TimeSmoothingWidget({"smooth_enabled": True}, show_algorithm=False)

    assert "smooth_algo" not in widget.get_config()


def test_golden_sample_widget_outputs_legacy_key(qapp):
    widget = GoldenSampleWidget({"golden_sample_checked": True})

    assert widget.is_checked() is True
    assert widget.display_mode() == "deviation"
    assert widget.get_config() == {
        "golden_sample_checked": True,
        "golden_sample_display_mode": "deviation",
    }


def test_golden_sample_widget_preserves_envelope_mode_and_enabled_state(qapp):
    widget = GoldenSampleWidget(
        {
            "golden_sample_checked": True,
            "golden_sample_display_mode": "envelope",
        }
    )

    assert widget.display_mode_combo.isEnabled() is True
    assert widget.display_mode() == "envelope"
    assert widget.limit_value_semantics() == "offset"
    assert "黄金样本上下框线" in widget.display_mode_combo.currentText()
    assert "偏差曲线模式" in widget.display_mode_combo.toolTip()
    assert "带符号偏移量" in widget.display_mode_combo.toolTip()
    assert "上框线" in widget.display_mode_combo.toolTip()
    assert "下框线" in widget.display_mode_combo.toolTip()
    assert "下框线 = 黄金样本曲线 + 下限值" in widget.display_mode_combo.toolTip()
    widget.enabled_checkbox.setChecked(False)
    assert widget.display_mode_combo.isEnabled() is False
    assert widget.limit_value_semantics() == "bounds"
    assert widget.get_config() == {
        "golden_sample_checked": False,
        "golden_sample_display_mode": "envelope",
    }


def test_golden_sample_widget_invalid_display_mode_falls_back_to_deviation(qapp):
    widget = GoldenSampleWidget({"golden_sample_display_mode": "unsupported"})

    assert widget.display_mode() == "deviation"


def test_harmonic_selector_filters_to_range(qapp):
    widget = HarmonicSelectorWidget({"selected_labels": [1, 2, "3", 40]}, start_order=2, end_order=35)

    assert widget.selected_labels() == [2, 3]
    assert widget.get_config() == {"selected_labels": [2, 3], "all_checked": False}


def test_harmonic_selector_all_checked_selects_range(qapp):
    widget = HarmonicSelectorWidget({"selected_labels": [10], "all_checked": True}, start_order=10, end_order=12)

    assert widget.selected_labels() == [10, 11, 12]
    assert widget.all_checked() is True
    assert widget.scroll_area.isEnabled() is False


def test_harmonic_selector_toggle_label_updates_selection(qapp):
    widget = HarmonicSelectorWidget({"selected_labels": [2]}, start_order=2, end_order=3)
    first_label = widget.box_layout.itemAt(0).widget()

    first_label.mousePressEvent(None)

    assert widget.selected_labels() == []
    assert first_label.text().startswith("  ")


def _filler_widget(min_height=120):
    widget = QWidget()
    widget.setMinimumHeight(min_height)
    layout = QVBoxLayout(widget)
    layout.addWidget(ChannelSelectorWidget({"analysis_channel": 0}, [0]))
    return widget


def test_semantic_dialog_registers_only_added_groups(qapp):
    dialog = SemanticAnalysisConfigDialogBase()
    dialog.add_semantic_section("input", widget=_filler_widget())
    dialog.add_semantic_section("compute", widget=_filler_widget())
    dialog.add_semantic_section("judgment", title="判定参数", widget=_filler_widget())

    assert dialog.semantic_group_keys() == ["input", "compute", "judgment"]
    assert set(dialog._semantic_nav_buttons) == {"input", "compute", "judgment"}
    assert dialog.current_semantic_group_key() == "input"
    assert dialog._semantic_nav_buttons["input"].isChecked() is True


def test_semantic_dialog_scrollbars_are_available_when_content_overflows(qapp):
    dialog = SemanticAnalysisConfigDialogBase()

    assert dialog.section_scroll_area.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert dialog.section_scroll_area.verticalScrollBarPolicy() == Qt.ScrollBarAsNeeded


def test_semantic_dialog_rejects_duplicate_group_keys(qapp):
    dialog = SemanticAnalysisConfigDialogBase()
    dialog.add_semantic_section("input", widget=_filler_widget())

    with pytest.raises(ValueError):
        dialog.add_semantic_section("input", widget=_filler_widget())


def test_semantic_dialog_nav_click_updates_active_group(qapp):
    dialog = SemanticAnalysisConfigDialogBase()
    dialog.add_semantic_section("input", widget=_filler_widget())
    dialog.add_semantic_section("compute", widget=_filler_widget())

    dialog.scroll_to_semantic_section("compute")

    assert dialog.current_semantic_group_key() == "compute"
    assert dialog._semantic_nav_buttons["compute"].isChecked() is True
    assert dialog._semantic_nav_buttons["input"].isChecked() is False


def test_semantic_dialog_sections_are_collapsible_and_expanded_by_default(qapp):
    dialog = SemanticAnalysisConfigDialogBase()
    dialog.add_semantic_section("compute", widget=_filler_widget())

    assert dialog.is_semantic_section_collapsed("compute") is False
    assert dialog._semantic_section_contents["compute"].isHidden() is False

    dialog.toggle_semantic_section("compute")

    assert dialog.is_semantic_section_collapsed("compute") is True
    assert dialog._semantic_section_contents["compute"].isHidden() is True
    assert dialog._semantic_section_indicators["compute"].text() == ">"

    dialog.set_semantic_section_collapsed("compute", False)

    assert dialog.is_semantic_section_collapsed("compute") is False
    assert dialog._semantic_section_contents["compute"].isHidden() is False
    assert dialog._semantic_section_indicators["compute"].text() == "v"


def test_semantic_dialog_scroll_sync_updates_active_group(qapp):
    dialog = SemanticAnalysisConfigDialogBase()
    dialog.add_semantic_section("input", widget=_filler_widget(200))
    dialog.add_semantic_section("compute", widget=_filler_widget(200))
    dialog.add_semantic_section("judgment", widget=_filler_widget(200))
    dialog.resize(500, 260)
    dialog.show()
    qapp.processEvents()

    judgment_y = dialog._semantic_sections["judgment"].y()
    dialog.section_scroll_area.verticalScrollBar().setValue(judgment_y)
    qapp.processEvents()

    assert dialog.current_semantic_group_key() == "judgment"
    assert dialog._semantic_nav_buttons["judgment"].isChecked() is True


def test_semantic_dialog_footer_buttons_call_callbacks(qapp):
    dialog = SemanticAnalysisConfigDialogBase()
    calls = []
    dialog.set_semantic_button_callbacks(
        default_callback=lambda: calls.append("default"),
        restore_callback=lambda: calls.append("restore"),
        ok_callback=lambda: calls.append("ok"),
        cancel_callback=lambda: calls.append("cancel"),
    )

    dialog.semantic_default_btn.click()
    dialog.semantic_restore_btn.click()
    dialog.semantic_cancel_btn.click()
    dialog.semantic_ok_btn.click()

    assert calls == ["default", "restore", "cancel", "ok"]
