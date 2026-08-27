import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import (
    QApplication,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyleOptionSpinBox,
    QWidget,
    QVBoxLayout,
)

from consts.harmonic_detection_consts import (
    HARMONIC_DETECTION_METHOD_FOURIER,
    HARMONIC_DETECTION_METHOD_KEY,
    HARMONIC_DETECTION_METHOD_SYNCHRONOUS,
)
from ui.ui_analysis_config.common_widgets import (
    AnalysisConfigDialogBase,
    AnalysisChannelSpinBoxWidget,
    ChannelSelectorWidget,
    GoldenSampleWidget,
    HarmonicDetectionMethodSelectorWidget,
    HarmonicSelectorWidget,
    OctaveSmoothingSelectorWidget,
    SemanticAnalysisConfigDialogBase,
    TimeSmoothingWidget,
    WeightingSelectorWidget,
)


class _ScreenStub:
    def __init__(self, width, height):
        self._available_geometry = QRect(0, 0, width, height)

    def availableGeometry(self):
        return self._available_geometry


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
        ({"analysis_channel": "1.5"}, [0, 1], 0),
        ({"analysis_channel": 128}, [0, 128], 128),
        ({}, [], 0),
        (None, None, 0),
    ],
)
def test_channel_selector_falls_back_safely(qapp, cfg, available_channels, expected):
    widget = ChannelSelectorWidget(cfg, available_channels)

    assert widget.current_channel() == expected


def test_analysis_channel_spinbox_uses_one_based_display_and_zero_based_config(qapp):
    widget = AnalysisChannelSpinBoxWidget(
        {"analysis_channel": 2},
        available_channels=[0],
    )

    assert widget.spin_box.minimum() == 1
    assert widget.spin_box.maximum() == 128
    assert widget.spin_box.value() == 3
    assert widget.current_channel() == 2
    assert widget.get_config() == {"analysis_channel": 2}


def test_analysis_channel_spinbox_allows_channel_absent_from_hardware_list(qapp):
    widget = AnalysisChannelSpinBoxWidget({}, available_channels=[0, 2])

    assert widget.spin_box.lineEdit().isReadOnly() is False
    widget.spin_box.setValue(128)

    assert widget.get_config() == {"analysis_channel": 127}


@pytest.mark.parametrize(
    ("cfg", "available_channels", "expected_display"),
    [
        ({"analysis_channel": 2}, [0, 2, 7], 3),
        ({"analysis_channel": 5}, [0, 2, 7], 1),
        ({"analysis_channel": "bad"}, [2, 7], 3),
        ({}, [2, 7], 3),
        ({"analysis_channel": 2}, [-1, 0, 2, 2, 128], 3),
        ({"analysis_channel": 2}, ["2", 7], 8),
        ({"analysis_channel": 7}, [-1, 128], 1),
    ],
)
def test_restricted_analysis_channel_spinbox_restores_only_allowed_values(
    qapp, cfg, available_channels, expected_display
):
    widget = AnalysisChannelSpinBoxWidget(
        cfg,
        available_channels=available_channels,
        restrict_to_available_channels=True,
    )

    assert widget.spin_box.lineEdit().isReadOnly() is True
    assert widget.spin_box.value() == expected_display
    assert widget.get_config() == {"analysis_channel": expected_display - 1}


def test_restricted_analysis_channel_spinbox_cycles_selected_channels(qapp):
    widget = AnalysisChannelSpinBoxWidget(
        {"analysis_channel": 0},
        available_channels=[7, 0, 2, 2],
        restrict_to_available_channels=True,
    )

    widget.spin_box.stepBy(1)
    assert widget.spin_box.value() == 3
    widget.spin_box.stepBy(1)
    assert widget.spin_box.value() == 8
    widget.spin_box.stepBy(1)
    assert widget.spin_box.value() == 1
    widget.spin_box.stepBy(-1)
    assert widget.spin_box.value() == 8
    widget.spin_box.stepBy(2)
    assert widget.spin_box.value() == 3
    assert widget.get_config() == {"analysis_channel": 2}


def test_restricted_analysis_channel_spinbox_single_channel_stays_fixed(qapp):
    widget = AnalysisChannelSpinBoxWidget(
        {"analysis_channel": 7},
        available_channels=[7],
        restrict_to_available_channels=True,
    )

    widget.spin_box.stepBy(1)
    widget.spin_box.stepBy(-3)

    assert widget.spin_box.value() == 8
    assert widget.get_config() == {"analysis_channel": 7}


def test_restricted_analysis_channel_spinbox_arrow_keys_wrap_at_numeric_bounds(qapp):
    widget = AnalysisChannelSpinBoxWidget(
        {"analysis_channel": 0},
        available_channels=[0, 2, 127],
        restrict_to_available_channels=True,
    )

    QTest.keyClick(widget.spin_box, Qt.Key_Down)
    assert widget.spin_box.value() == 128
    QTest.keyClick(widget.spin_box, Qt.Key_Up)
    assert widget.spin_box.value() == 1


def test_restricted_analysis_channel_spinbox_visible_up_button_wraps_at_maximum(qapp):
    widget = AnalysisChannelSpinBoxWidget(
        {"analysis_channel": 127},
        available_channels=[0, 2, 127],
        restrict_to_available_channels=True,
    )
    widget.show()
    qapp.processEvents()
    option = QStyleOptionSpinBox()
    widget.spin_box.initStyleOption(option)
    up_button = widget.spin_box.style().subControlRect(
        QStyle.CC_SpinBox,
        option,
        QStyle.SC_SpinBoxUp,
        widget.spin_box,
    )

    QTest.mouseClick(widget.spin_box, Qt.LeftButton, pos=up_button.center())

    assert widget.spin_box.value() == 1


def test_analysis_channel_spinbox_keeps_strict_fractional_string_fallback(qapp):
    widget = AnalysisChannelSpinBoxWidget(
        {"analysis_channel": "1.5"},
        available_channels=[0, 1],
    )

    assert widget.spin_box.value() == 1
    assert widget.get_config() == {"analysis_channel": 0}


def test_legacy_selector_and_strict_spinbox_preserve_distinct_overflow_contracts(qapp):
    with pytest.raises(OverflowError):
        ChannelSelectorWidget(
            {"analysis_channel": float("inf")},
            available_channels=[0],
        )

    widget = AnalysisChannelSpinBoxWidget(
        {"analysis_channel": float("inf")},
        available_channels=[0],
    )
    assert widget.get_config() == {"analysis_channel": 0}


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


def test_analysis_dialog_keeps_preferred_size_when_screen_has_room(qapp, monkeypatch):
    monkeypatch.setattr(QApplication, "primaryScreen", lambda: _ScreenStub(1920, 1040))
    dialog = AnalysisConfigDialogBase()

    dialog.apply_vertical_golden_dialog_size()

    assert (dialog.width(), dialog.height()) == (630, 700)
    assert (dialog.minimumWidth(), dialog.minimumHeight()) == (560, 480)


def test_analysis_dialog_fits_within_low_screen_available_height(qapp, monkeypatch):
    monkeypatch.setattr(QApplication, "primaryScreen", lambda: _ScreenStub(1366, 728))
    dialog = AnalysisConfigDialogBase()

    dialog.apply_vertical_golden_dialog_size()

    assert (dialog.width(), dialog.height()) == (630, 696)
    assert (dialog.minimumWidth(), dialog.minimumHeight()) == (560, 480)


def test_analysis_dialog_minimum_size_does_not_exceed_very_small_screen(qapp, monkeypatch):
    monkeypatch.setattr(QApplication, "primaryScreen", lambda: _ScreenStub(520, 440))
    dialog = AnalysisConfigDialogBase()

    dialog.apply_vertical_golden_dialog_size()

    assert (dialog.width(), dialog.height()) == (488, 408)
    assert (dialog.minimumWidth(), dialog.minimumHeight()) == (488, 408)


def test_semantic_dialog_refresh_respects_nested_group_size_hint(qapp):
    dialog = SemanticAnalysisConfigDialogBase()
    nested = QWidget()
    layout = QVBoxLayout(nested)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(_filler_widget(80))
    layout.addWidget(_filler_widget(80))
    dialog.add_semantic_section("compute", widget=nested)
    dialog.show()
    qapp.processEvents()

    dialog._refresh_section_container_minimum_height()

    assert dialog._semantic_sections["compute"].height() >= (
        dialog._semantic_sections["compute"].layout().sizeHint().height()
    )
    dialog.close()


def test_semantic_dialog_refresh_protected_maximum_height(qapp):
    dialog = SemanticAnalysisConfigDialogBase()
    capped = QWidget()
    capped_layout = QVBoxLayout(capped)
    capped_layout.addWidget(_filler_widget(80))
    capped_layout.addWidget(_filler_widget(80))
    capped.setMaximumHeight(64)
    dialog.add_semantic_section("compute", widget=capped)
    dialog.show()
    qapp.processEvents()

    dialog._refresh_section_container_minimum_height()

    assert capped.maximumHeight() == 64
    assert capped.minimumHeight() <= capped.maximumHeight()
    dialog.close()


def test_semantic_dialog_refresh_protected_fixed_height(qapp):
    dialog = SemanticAnalysisConfigDialogBase()
    fixed = _filler_widget(80)
    fixed.setFixedHeight(64)
    dialog.add_semantic_section("compute", widget=fixed)
    dialog.show()
    qapp.processEvents()

    dialog._refresh_section_container_minimum_height()

    assert fixed.height() == 64
    assert fixed.minimumHeight() == 64
    assert fixed.maximumHeight() == 64
    dialog.close()


def test_semantic_dialog_refresh_protected_fixed_vertical_policy(qapp):
    dialog = SemanticAnalysisConfigDialogBase()
    fixed_policy = _filler_widget(80)
    fixed_policy.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    dialog.add_semantic_section("compute", widget=fixed_policy)
    dialog.show()
    qapp.processEvents()

    dialog._refresh_section_container_minimum_height()

    assert fixed_policy.sizePolicy().verticalPolicy() == QSizePolicy.Fixed
    dialog.close()


def test_semantic_dialog_refresh_protected_visible_container_height(qapp):
    dialog = SemanticAnalysisConfigDialogBase()
    dialog.add_semantic_section("input", widget=_filler_widget(90))
    dialog.add_semantic_section("compute", widget=_filler_widget(110))
    dialog.add_semantic_section("display", widget=_filler_widget(100))
    dialog.show()
    qapp.processEvents()

    dialog._refresh_section_container_minimum_height()

    visible_sections = [section for section in dialog._semantic_sections.values() if section.isVisible()]
    expected = sum(section.sizeHint().height() for section in visible_sections)
    expected += dialog.section_layout.spacing() * max(0, len(visible_sections) - 1)
    assert dialog.section_container.minimumHeight() >= expected - 2
    dialog.close()


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


def test_semantic_dialog_enter_does_not_trigger_buttons(qapp):
    dialog = SemanticAnalysisConfigDialogBase()
    input_widget = QLineEdit()
    dialog.add_semantic_section("input", widget=input_widget)
    calls = []
    dialog.set_semantic_button_callbacks(
        default_callback=lambda: calls.append("default"),
        restore_callback=lambda: calls.append("restore"),
        ok_callback=lambda: calls.append("ok"),
        cancel_callback=lambda: calls.append("cancel"),
    )
    dialog.show()
    input_widget.setFocus()
    qapp.processEvents()

    QTest.keyClick(input_widget, Qt.Key_Return)
    qapp.processEvents()

    assert calls == []
    assert all(
        not button.autoDefault() and not button.isDefault()
        for button in dialog.findChildren(QPushButton)
    )

    dialog.semantic_default_btn.click()

    assert calls == ["default"]
    dialog.close()
