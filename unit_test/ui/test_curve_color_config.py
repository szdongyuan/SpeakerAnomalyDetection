import os
import sys
from colorsys import rgb_to_hsv
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QWidget
from pyqtgraph import PlotWidget

from consts.acoustic_analysis.curve_style_consts import (
    DEFAULT_CURVE_COLORS,
    LOWER_LIMIT_COLOR,
    MAIN_CURVE_COLOR,
    PRESET_CURVE_COLORS,
    UPPER_LIMIT_COLOR,
)
from ui.curve_style import (
    build_curve_color_config,
    resolve_curve_colors,
)
from ui.graph_widget import LimitPlotUtils
from ui.ui_analysis_config.common_widgets import SemanticAnalysisConfigDialogBase
from ui.ui_analysis_config.curve_color_config_widget import CurveColorConfigWidget, PresetColorDialog
from ui.ui_analysis_config.threshold_config_widget import ThresholdConfigWidget


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _pen_color(plot_item):
    return plot_item.opts["pen"].color().name().upper()


def _plot_data_snapshot(plot_widget):
    snapshot = []
    for item in plot_widget.listDataItems():
        x_data, y_data = item.getData()
        snapshot.append((np.asarray(x_data).tolist(), np.asarray(y_data).tolist()))
    return snapshot


def _rgb_channels(color):
    return tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))


def _linear_channel(channel):
    channel /= 255
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def _perceptual_lightness(color):
    red, green, blue = map(_linear_channel, _rgb_channels(color))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    if luminance > (6 / 29) ** 3:
        return 116 * luminance ** (1 / 3) - 16
    return (29 / 3) ** 3 * luminance


def _hue(color):
    red, green, blue = (channel / 255 for channel in _rgb_channels(color))
    hue, _, _ = rgb_to_hsv(red, green, blue)
    return hue * 360


def _hue_span(colors):
    hues = sorted(_hue(color) for color in colors)
    gaps = [right - left for left, right in zip(hues, hues[1:])]
    gaps.append(hues[0] + 360 - hues[-1])
    return 360 - max(gaps)


def test_curve_color_config_normalizes_values_and_preserves_display_fields():
    config = {
        "display": {
            MAIN_CURVE_COLOR: "#123abc",
            UPPER_LIMIT_COLOR: "invalid",
            "grid_visible": False,
        }
    }

    colors = resolve_curve_colors(config)
    saved = build_curve_color_config(config, colors)

    assert colors[MAIN_CURVE_COLOR] == "#123ABC"
    assert colors[UPPER_LIMIT_COLOR] == DEFAULT_CURVE_COLORS[UPPER_LIMIT_COLOR]
    assert colors[LOWER_LIMIT_COLOR] == DEFAULT_CURVE_COLORS[LOWER_LIMIT_COLOR]
    assert saved["display"]["grid_visible"] is False


def test_curve_color_widget_updates_preview_and_nested_config(qapp):
    widget = CurveColorConfigWidget({"display": {"grid_visible": True}})
    emitted = []
    expanded_states = []
    widget.colors_changed.connect(emitted.append)
    widget.expanded_changed.connect(expanded_states.append)

    assert widget.is_expanded() is False
    assert widget.expand_button.arrowType() == Qt.RightArrow
    collapsed_height = widget.sizeHint().height()
    widget.set_expanded(True)
    expanded_height = widget.sizeHint().height()
    widget.set_color(MAIN_CURVE_COLOR, "#2563eb")
    config = widget.get_config()

    assert widget.is_expanded() is True
    assert widget.expand_button.arrowType() == Qt.DownArrow
    assert expanded_states == [True]
    assert expanded_height > collapsed_height
    content_labels = widget.content_widget.findChildren(QLabel)
    assert {label.text() for label in content_labels} == {
        "主曲线颜色",
        "上限颜色",
        "下限颜色",
        "示意预览",
    }
    assert all("color: #475467" in label.styleSheet() for label in content_labels)
    assert all("font-size: 14px" in label.styleSheet() for label in content_labels)
    assert all("font-weight: 400" in label.styleSheet() for label in content_labels)
    assert emitted[-1][MAIN_CURVE_COLOR] == "#2563EB"
    assert config["display"][MAIN_CURVE_COLOR] == "#2563EB"
    assert config["display"]["grid_visible"] is True
    assert widget.preview_widget.minimumHeight() == 130


def test_preset_palette_is_expanded_and_marks_current_color(qapp):
    current_color = "#86EFAC"
    dialog = PresetColorDialog(current_color)
    preset_values = [color for name, color in PRESET_CURVE_COLORS]
    selected_button = dialog._color_buttons[current_color]

    assert len(PRESET_CURVE_COLORS) == 64
    assert len(preset_values) == len(set(preset_values))
    assert set(DEFAULT_CURVE_COLORS.values()).issubset(preset_values)
    assert "#86EFAC" in preset_values
    assert "#93C5FD" in preset_values
    assert "#FCA5A5" in preset_values
    assert "#FCD34D" in preset_values
    assert "#D4D4D4" in preset_values
    assert selected_button.property("selectedColor") is True
    assert selected_button.text() == "✓"
    assert selected_button.size().width() == 40
    assert selected_button.size().height() == 28
    assert dialog.palette_layout.columnCount() == 8
    assert dialog.palette_layout.rowCount() == 8
    assert "4px solid #FFD400" in selected_button.styleSheet()
    assert "color: #111827" in selected_button.styleSheet()


def test_preset_palette_uses_consistent_hue_rows_and_perceptual_lightness():
    expected_rows = (
        ("墨绿", "深绿", "森林绿", "翠绿", "绿色", "亮绿", "浅绿", "雾绿"),
        ("深青", "暗青", "蓝绿", "青色", "湖青", "亮青", "浅青", "雾青"),
        ("深蓝", "藏蓝", "浓蓝", "蓝色", "亮蓝", "天蓝", "浅蓝", "雾蓝"),
        ("深靛", "暗靛", "靛蓝", "蓝紫", "亮靛", "浅靛", "淡靛", "雾靛"),
        ("紫色", "深洋红", "暗洋红", "洋红", "亮洋红", "浅洋红", "粉紫", "雾紫"),
        ("深红", "暗红", "浓红", "红色", "亮红", "珊瑚红", "浅红", "雾红"),
        ("深棕", "棕色", "棕黄", "琥珀", "橙黄", "金黄", "浅黄", "雾黄"),
        ("纯黑", "黑色", "炭灰", "深灰", "石墨灰", "中灰", "浅灰", "雾灰"),
    )
    rows = tuple(
        PRESET_CURVE_COLORS[start:start + 8]
        for start in range(0, len(PRESET_CURVE_COLORS), 8)
    )
    actual_rows = tuple(tuple(name for name, color in row) for row in rows)

    assert actual_rows == expected_rows
    for row in rows:
        lightness = [_perceptual_lightness(color) for name, color in row]
        lightness_steps = [
            right - left
            for left, right in zip(lightness, lightness[1:])
        ]
        assert min(lightness_steps) >= 3
    for row in rows[:-1]:
        assert _hue_span([color for name, color in row]) <= 30


def test_preset_palette_applies_selection_only_after_confirmation(qapp):
    dialog = PresetColorDialog("#86EFAC")
    dialog.show()
    qapp.processEvents()

    dialog._select_color("#93C5FD")

    button_gap = dialog.confirm_button.x() - (
        dialog.cancel_button.x() + dialog.cancel_button.width()
    )
    assert dialog.footer_layout.spacing() == 16
    assert button_gap == 16
    assert dialog.result() == 0
    assert dialog.selected_color == "#93C5FD"
    assert dialog._color_buttons["#86EFAC"].property("selectedColor") is False
    assert dialog._color_buttons["#93C5FD"].property("selectedColor") is True
    assert dialog.confirm_button.isDefault() is True
    dialog.confirm_button.click()
    assert dialog.result() == QDialog.Accepted

    cancelled_dialog = PresetColorDialog("#86EFAC")
    cancelled_dialog._select_color("#93C5FD")
    cancelled_dialog.cancel_button.click()
    assert cancelled_dialog.result() == QDialog.Rejected


def test_threshold_preview_uses_bound_upper_and_lower_colors(qapp):
    limit_data = (
        np.array([100.0, 1000.0]),
        np.array([10.0, 12.0]),
        np.array([2.0, 3.0]),
    )
    color_widget = CurveColorConfigWidget(
        {
            "display": {
                UPPER_LIMIT_COLOR: "#DC2626",
                LOWER_LIMIT_COLOR: "#2563EB",
            }
        }
    )
    threshold_widget = ThresholdConfigWidget(load_config={"limit_data": limit_data}, model_type="FR")

    threshold_widget.bind_curve_color_widget(color_widget)
    items = threshold_widget.limit_graph.listDataItems()

    assert [_pen_color(item) for item in items] == ["#DC2626", "#2563EB"]
    assert all(item.opts["pen"].style() == Qt.DashLine for item in items)
    assert threshold_widget.get_config()["display"][LOWER_LIMIT_COLOR] == "#2563EB"


def test_threshold_manual_preview_survives_curve_color_binding_and_changes(qapp):
    stale_csv_limit_data = (
        np.array([100.0, 1000.0]),
        np.array([80.0, 90.0]),
        np.array([10.0, 20.0]),
    )
    expected_manual_preview = [([1.0, 3.0], [2.0, 4.0])]
    threshold_widget = ThresholdConfigWidget(
        load_config={
            "limit_checked": True,
            "limit_mode": "manual",
            "limit_data": stale_csv_limit_data,
            "manual_upper_enabled": True,
            "manual_lower_enabled": False,
            "manual_upper_segments": [
                {"start_x": 1.0, "start_y": 2.0, "end_x": 3.0, "end_y": 4.0}
            ],
            "manual_lower_segments": [],
        },
        model_type="FR",
        allow_manual_limits=True,
    )
    color_widget = CurveColorConfigWidget()

    assert _plot_data_snapshot(threshold_widget.limit_graph) == expected_manual_preview

    threshold_widget.bind_curve_color_widget(color_widget)
    assert _plot_data_snapshot(threshold_widget.limit_graph) == expected_manual_preview

    color_widget.set_color(UPPER_LIMIT_COLOR, "#16A34A")
    assert _plot_data_snapshot(threshold_widget.limit_graph) == expected_manual_preview


def test_shared_dialog_base_appends_display_and_binds_threshold(qapp):
    dialog = SemanticAnalysisConfigDialogBase()
    existing_display_widget = QWidget(dialog)
    threshold_widget = ThresholdConfigWidget(load_config={}, model_type="FR")
    dialog.add_semantic_section("display", widget=existing_display_widget)

    dialog.add_threshold_curve_sections(threshold_widget, {})

    display_layout = dialog._semantic_section_contents["display"].layout()
    assert dialog.semantic_group_keys() == ["display", "judgment"]
    assert display_layout.count() == 2
    assert threshold_widget._curve_color_widget is dialog.curve_color_widget


def test_limit_plot_uses_three_independent_colors(qapp):
    plot_widget = PlotWidget()
    colors = {
        MAIN_CURVE_COLOR: "#111827",
        UPPER_LIMIT_COLOR: "#EA580C",
        LOWER_LIMIT_COLOR: "#0891B2",
    }

    LimitPlotUtils.setup_limit_plot(
        plot_widget,
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        np.array([1.0, 2.0]),
        np.array([5.0, 5.0]),
        np.array([1.0, 1.0]),
        curve_colors=colors,
    )

    items = plot_widget.listDataItems()
    assert [_pen_color(item) for item in items] == ["#111827", "#EA580C", "#0891B2"]
    assert items[1].opts["pen"].style() == Qt.DashLine
    assert items[2].opts["pen"].style() == Qt.DashLine
