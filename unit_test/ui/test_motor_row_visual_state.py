"""Real Qt row state, rendering and bounded style-operation regressions."""
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPalette
from PyQt5.QtWidgets import QStyle, QStyleOptionButton

from ui.sequence.motor_result_panel import MotorResultPanel


def conditions():
    return [
        {"key": key, "group_name": group, "condition_name": key,
         "analysis_list": {"display_sequence": ["SPL"],
                           "SPL": {"type": "SPL", "channels": [0]}}}
        for key, group in (("a", "A"), ("b", "A"), ("c", "B"))
    ]


@pytest.fixture
def panel(qt_app):
    widget = MotorResultPanel(condition_configs=conditions(), queue_catalog={})
    widget.setAttribute(Qt.WA_DontShowOnScreen)
    widget.resize(600, 700)
    widget.show()
    qt_app.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    qt_app.processEvents()


def state(panel, key):
    return panel.rows[key]["button"].property("visualState")


@contextmanager
def style_calls(panel):
    """Wrap real methods; count only initialized row buttons/result labels."""
    widgets = [widget for row in panel.rows.values()
               for widget in (row["button"], row["labels"]["result"])]
    with ExitStack() as stack:
        calls = {name: [stack.enter_context(patch.object(widget, name, wraps=getattr(widget, name)))
                        for widget in widgets]
                 for name in ("setStyleSheet", "setProperty", "update")}
        styles = {id(widget.style()): widget.style() for widget in widgets}.values()
        for name in ("polish", "unpolish"):
            calls[name] = [stack.enter_context(patch.object(style, name, wraps=getattr(style, name)))
                           for style in styles]
        calls["all_rows"] = stack.enter_context(
            patch.object(panel, "_refresh_row_styles", wraps=panel._refresh_row_styles))
        yield calls, widgets


def operation_count(calls, widgets, name):
    if name in ("polish", "unpolish"):
        return sum(call.args[0] in widgets for mock in calls[name] for call in mock.call_args_list)
    return sum(mock.call_count for mock in calls[name])


def test_initial_and_view_recording_priority(panel):
    assert panel.selected_key == "a"
    assert panel.viewed_key == ""
    assert all(state(panel, key) == "normal" for key in panel.rows)
    assert all(row["labels"]["result"].property("resultTone") == "pending"
               for row in panel.rows.values())
    panel.select_condition("a")
    assert state(panel, "a") == "viewed"
    panel.set_condition_result("b", "采集中", "running")
    assert panel.viewed_key == ""
    assert state(panel, "a") == "normal"
    assert state(panel, "b") == "recording"
    panel.select_condition("b")
    assert state(panel, "b") == "recording"
    panel.set_condition_result("b", "OK", "ok")
    assert state(panel, "b") == "viewed"
    assert panel.rows["b"]["labels"]["result"].property("resultTone") == "ok"


def test_only_changed_widgets_are_polished_and_business_updates_continue(panel):
    signals = []
    panel.condition_selected.connect(signals.append)
    panel.select_condition("a", show_detail=True)
    with style_calls(panel) as (calls, widgets):
        panel.select_condition("b", show_detail=True)
        assert operation_count(calls, widgets, "setProperty") == 2
        assert operation_count(calls, widgets, "polish") == 2
        calls["all_rows"].assert_not_called()
    with style_calls(panel) as (calls, widgets):
        panel.select_condition("b", show_detail=True)
        assert panel.detail_frame.isHidden()
        panel.set_condition_result("b", "数据保存中", "running")
        assert operation_count(calls, widgets, "setStyleSheet") == 0
        assert operation_count(calls, widgets, "setProperty") == 1
        assert operation_count(calls, widgets, "polish") == 1
        calls["all_rows"].assert_not_called()
    label = panel.rows["b"]["labels"]["result"]
    assert label.width() >= label.fontMetrics().horizontalAdvance("数据保存中") + 12
    with style_calls(panel) as (calls, widgets):
        panel.set_condition_result("b", "数据保存中", "running")
        panel.set_condition_channel_results("b", [{"raw_channel": 0, "result": "OK", "SPL": "OK"}])
        for name in ("setStyleSheet", "setProperty", "polish", "unpolish", "update"):
            assert operation_count(calls, widgets, name) == 0, name
        calls["all_rows"].assert_not_called()
    assert panel.rows["b"]["completed_channels"] == 1
    assert panel.rows["b"]["labels"]["progress"].text() == "通道判定：1/1"
    assert panel.channel_detail_labels[0]["SPL"].text() == "OK"
    assert signals == ["a", "b", "b"]


def test_ordinary_result_only_synchronizes_target_row(panel):
    panel.select_condition("a")
    with patch.object(panel, "_refresh_row_style", wraps=panel._refresh_row_style) as refresh:
        panel.set_condition_result("b", "分析中", "running")
    refresh.assert_called_once_with("b")
    assert state(panel, "a") == "viewed"


@pytest.mark.parametrize("stage", ["准备采集", "采集中"])
def test_cross_port_auto_selection_and_hidden_recording(panel, stage):
    panel.select_condition("a", show_detail=True)
    signals = []
    panel.condition_selected.connect(signals.append)
    with style_calls(panel) as (calls, widgets):
        panel.set_condition_result("c", stage, "running")
        calls["all_rows"].assert_not_called()
        assert operation_count(calls, widgets, "setProperty") == (3 if stage == "采集中" else 2)
        assert operation_count(calls, widgets, "polish") == (3 if stage == "采集中" else 2)
    assert signals == ["c", "c"]  # synchronous port signal then automatic selection
    assert panel.selected_key == "c"
    assert panel.viewed_key == ""
    assert panel._detail_owner_key == "c"
    assert state(panel, "a") == "normal"
    assert state(panel, "c") == ("recording" if stage == "采集中" else "normal")
    panel.current_port_combo.setCurrentIndex(0)
    assert panel.rows["c"]["button"].isHidden()
    assert state(panel, "c") == ("recording" if stage == "采集中" else "normal")


def test_reset_rebuild_preserve_results_and_empty_configuration(panel):
    panel.select_condition("a")
    panel.set_condition_result("b", "NG", "ng")
    old_button = panel.rows["a"]["button"]
    assert panel.refresh_condition_configs(conditions(), queue_catalog={}, preserve_results=True)
    assert panel.rows["a"]["button"] is old_button
    assert state(panel, "a") == "viewed"
    assert panel.rows["b"]["labels"]["result"].property("resultTone") == "ng"
    panel.reset()
    assert panel.viewed_key == ""
    assert all(state(panel, key) == "normal" for key in panel.rows)
    assert all(row["labels"]["result"].property("resultTone") == "pending"
               for row in panel.rows.values())
    panel.select_condition("b", show_detail=True)
    panel.set_condition_configs(conditions(), queue_catalog={})
    assert panel.rows["a"]["button"] is not old_button
    assert all(state(panel, key) == "normal" for key in panel.rows)
    panel.set_condition_configs([], queue_catalog={})
    panel.reset()
    assert panel.rows == {}
    assert panel.selected_key == panel.viewed_key == ""
    assert not panel.set_condition_result("missing", "OK")


@pytest.mark.parametrize("previous", ["分析失败", "结果不完整", "OK", "NG"])
@pytest.mark.parametrize("placeholder", ["待判定", "未标记"])
def test_placeholders_preserve_analysis_outcomes(panel, previous, placeholder):
    if previous in ("OK", "NG"):
        panel.set_condition_channel_results("a", [{"raw_channel": 0, "result": previous}])
    panel.set_condition_result("a", previous)
    panel.set_condition_result("a", placeholder)
    assert panel.rows["a"]["result"] == previous
    expected = "ok" if previous == "OK" else "ng"
    assert panel.rows["a"]["tone"] == expected
    assert panel.rows["a"]["labels"]["result"].property("resultTone") == expected


@pytest.mark.parametrize("visual,bg,border,hover_bg,hover_border", [
    ("normal", "#f4f8fc", "#b8c8da", "#edf4fc", "#6fa8dc"),
    ("viewed", "#e1efff", "#1877c9", "#d7e9fc", "#1269b2"),
    ("recording", "#eaf2fb", "#2f80c9", "#e3eef9", "#286eae"),
])
def test_actual_button_and_hover_rendering(panel, qt_app, tmp_path, visual, bg, border, hover_bg, hover_border):
    if visual == "viewed":
        panel.select_condition("a")
    elif visual == "recording":
        panel.set_condition_result("a", "采集中", "running")
    button = panel.rows["a"]["button"]
    qt_app.processEvents()
    image = button.grab().toImage()
    assert image.pixelColor(image.width() // 2, 3).name() == bg
    assert image.pixelColor(image.width() // 2, 0).name() == border
    assert image.save(str(tmp_path / f"{visual}.png"))
    # Exercise the actual Qt QSS hover selector through the native style engine.
    option = QStyleOptionButton()
    button.initStyleOption(option)
    option.state |= QStyle.State_MouseOver
    hover = QImage(button.size(), QImage.Format_ARGB32)
    hover.fill(Qt.transparent)
    painter = QPainter(hover)
    button.style().drawControl(QStyle.CE_PushButton, option, painter, button)
    painter.end()
    assert hover.pixelColor(hover.width() // 2, 3).name() == hover_bg
    assert hover.pixelColor(hover.width() // 2, 0).name() == hover_border
    assert hover.save(str(tmp_path / f"{visual}-hover.png"))


@pytest.mark.parametrize("tone,color", [("ok", "#16864b"), ("ng", "#d94343"),
                                        ("running", "#2f6fb4"), ("pending", "#64748b"),
                                        ("unknown", "#64748b")])
def test_result_palette_text_pixels_and_label_isolation(panel, qt_app, tmp_path, tone, color):
    panel.set_condition_result("a", "TEST", tone)
    qt_app.processEvents()
    row = panel.rows["a"]
    label = row["labels"]["result"]
    assert row["tone"] == tone
    assert label.property("resultTone") == ("pending" if tone == "unknown" else tone)
    assert label.palette().color(QPalette.WindowText).name() == color
    assert label.font().bold()
    assert label.font().pixelSize() == 13
    for name in ("name", "progress"):
        assert row["labels"][name].palette().color(QPalette.WindowText).name() == "#1f2937"
    image = label.grab().toImage()
    assert image.save(str(tmp_path / f"result-{tone}.png"))
    expected = QColor(color).getRgb()[:3]
    # Subpixel font antialiasing can blend every glyph pixel with its background.
    closest = min(max(abs(a - b) for a, b in zip(image.pixelColor(x, y).getRgb()[:3], expected))
                  for y in range(image.height()) for x in range(image.width()))
    assert closest <= 35, closest
