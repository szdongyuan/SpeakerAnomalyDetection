import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from ui.ui_analysis_config.common_widgets import (
    ChannelSelectorWidget,
    GoldenSampleWidget,
    HarmonicSelectorWidget,
    OctaveSmoothingSelectorWidget,
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
    assert widget.get_config() == {"golden_sample_checked": True}


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
