"""Confirmed-factor full-scale SPL maths and panel display boundaries."""

import math
import os
import sys
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from unittest import mock

import pytest
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QFontDatabase
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QLabel, QWidget

from ui import calibration_window as calibration


@pytest.fixture
def panel_factory(ui_qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(calibration.LogManager, "set_log_handler", lambda *_: mock.Mock())
    monkeypatch.setattr(calibration, "load_mic_channel_v2pa_factors", lambda *_: {0: .0019})
    monkeypatch.setattr(calibration.InputCalibration, "_get_recording_bridge",
                        mock.Mock(side_effect=AssertionError("No recording in display tests")))
    panels = []

    def make(*, device=None, channels=(0,), factors=None, vk=False, dialog=False, **kwargs):
        if vk:
            from base.vkinging_stores import VECalibrationStore
            from unit_test.base.vkinging_fakes import device_info
            device = device_info(physical_channels=[0, 1])
            store = VECalibrationStore(tmp_path / f"calibration-{len(panels)}.json")
            for channel, factor in (factors or {}).items():
                store.save_factor(device, channel, v2pa_factor=factor,
                                  calibrated_at="2026-11-02T00:00:00+00:00")
            kwargs["ve_calibration_store"] = store
        elif device is None:
            device = dict(index=7, name="Test Mic", hostapi=3, max_input_channels=2)
        if factors is not None and not vk:
            monkeypatch.setattr(calibration, "load_mic_channel_v2pa_factors",
                                lambda *_: dict(factors))
        cls = calibration.CalibrationWindow if dialog else calibration.InputCalibration
        window = cls(deepcopy(device), channels, **kwargs)
        panel = window.input_cal_wnd if dialog else window
        panels.append((window, panel))
        return window

    yield make
    for window, panel in panels:
        panel.close_recording()
        window.close()


@pytest.mark.parametrize("factor,peak,expected", [
    (.0019, 1., 36.54), (10., 2.5, 118.93), (10., 5., 124.95), (10., 10., 130.97),
])
def test_full_scale_spl_known_values(factor, peak, expected):
    assert round(calibration.InputCalibration._full_scale_spl_from_factor(factor, peak), 2) == expected


def test_full_scale_spl_doubling_factor_or_peak():
    calculate = calibration.InputCalibration._full_scale_spl_from_factor
    baseline = calculate(3., 2.5)
    assert calculate(6., 2.5) - baseline == pytest.approx(6.020599913279624)
    assert calculate(3., 5.) - baseline == pytest.approx(6.020599913279624)


@pytest.mark.parametrize("value", [math.nextafter(0., 1.), sys.float_info.max])
def test_full_scale_spl_finite_extremes_without_product_overflow(value):
    result = calibration.InputCalibration._full_scale_spl_from_factor(value, value)
    assert math.isfinite(result)
    assert result == pytest.approx(40 * math.log10(value) + 90.96910013008056)


@pytest.mark.parametrize("invalid", [None, "2", object(), True, False, 0, -1,
                                    math.nan, math.inf, -math.inf, 10 ** 1000])
@pytest.mark.parametrize("argument", ["factor", "peak"])
def test_full_scale_spl_rejects_invalid_values(invalid, argument):
    kwargs = {"factor": 1., "peak": 1., argument: invalid}
    assert calibration.InputCalibration._full_scale_spl_from_factor(**kwargs) is None


def test_full_scale_spl_uses_confirmed_precision_and_soundcard_peak_one(panel_factory):
    widget = panel_factory(test_full_scale_peak_volts=10.)
    assert widget.v2pa_factor_lineedit.text() == "0.001900"
    assert widget.full_scale_spl_lineedit.text() == "36.54"
    assert widget.full_scale_spl_lineedit.isReadOnly()
    assert widget.full_scale_spl_label.text() == "量程上限"
    assert widget.full_scale_spl_unit_label.text() == "dB SPL"
    tooltip = "按峰值为 1 的正弦波 RMS 换算，不代表设备实际无失真测量上限。"
    for control in (widget.full_scale_spl_label, widget.full_scale_spl_lineedit,
                    widget.full_scale_spl_unit_label):
        assert control.toolTip() == tooltip


def test_loaded_factor_precision_is_not_taken_from_rounded_editor(panel_factory):
    widget = panel_factory(factors={0: .000000019})
    assert widget.v2pa_factor_lineedit.text() == "0.000000"
    assert widget.full_scale_spl_lineedit.text() == "-63.46"


@pytest.mark.parametrize("peak,expected", [(10., "130.97"), (5., "124.95"), (2.5, "118.93"),
                                          (1., "110.97"), (.5, "104.95"), (.1, "90.97"),
                                          (.02, "76.99")])
def test_vk_uses_only_valid_supported_test_range(panel_factory, peak, expected):
    widget = panel_factory(vk=True, factors={0: 10.}, test_full_scale_peak_volts=peak)
    assert widget.full_scale_spl_lineedit.text() == expected
    tooltip = (f"按当前测试量程 ±{peak:g} V 的正弦波 RMS 换算，"
               "不代表设备实际无失真测量上限。")
    for control in (widget.full_scale_spl_label, widget.full_scale_spl_lineedit,
                    widget.full_scale_spl_unit_label):
        assert control.toolTip() == tooltip


@pytest.mark.parametrize("peak", [None, True, False, "2.5", 0, -2.5, 3., math.nan,
                                  math.inf, object(), 10 ** 1000])
def test_vk_unavailable_range_does_not_block_coefficient_editing(panel_factory, peak):
    widget = panel_factory(vk=True, factors={0: 10.}, test_full_scale_peak_volts=peak)
    assert widget.full_scale_spl_lineedit.text() == "—"
    assert widget.test_full_scale_peak_volts is None
    assert widget.calibration_available
    assert not widget.v2pa_factor_lineedit.isReadOnly()
    tooltip = "当前测试量程不可用，无法换算量程上限；不影响输入校准。"
    for control in (widget.full_scale_spl_label, widget.full_scale_spl_lineedit,
                    widget.full_scale_spl_unit_label):
        assert control.toolTip() == tooltip


def test_dialog_hands_test_range_to_child(panel_factory):
    dialog = panel_factory(vk=True, dialog=True, factors={0: 10.}, test_full_scale_peak_volts=2.5)
    assert dialog.test_full_scale_peak_volts == 2.5
    assert dialog.input_cal_wnd.test_full_scale_peak_volts == 2.5
    assert dialog.input_cal_wnd.full_scale_spl_lineedit.text() == "118.93"


@pytest.mark.parametrize("vk", [False, True], ids=["soundcard", "vk"])
@pytest.mark.parametrize("dialog", [False, True], ids=["panel", "dialog"])
@pytest.mark.parametrize("gesture", ["return", "keypad", "panel", "group", "label", "margin", "blank-enter"])
def test_coefficient_confirmation_refreshes_saved_result_once(
        panel_factory, monkeypatch, ui_qapp, vk, dialog, gesture):
    window = panel_factory(vk=vk, dialog=dialog, factors={0: 1.25},
                           test_full_scale_peak_volts=2.5)
    panel = window.input_cal_wnd if dialog else window
    save = mock.Mock(wraps=panel.ve_calibration_store.save_factor) if vk else mock.Mock()
    if vk:
        monkeypatch.setattr(panel.ve_calibration_store, "save_factor", save)
    else:
        monkeypatch.setattr(calibration, "save_mic_channel_factor", save)
    changes = []
    panel.calibration_state_changed.connect(changes.append)
    actions = []
    if dialog:
        for button in (window.cal_btn, window.reset_btn,
                       next(button for button in window.findChildren(calibration.QPushButton)
                            if "退" in button.text())):
            button.clicked.connect(lambda: actions.append(True))
    window.show()
    window.activateWindow()
    ui_qapp.processEvents()
    line = panel.v2pa_factor_lineedit
    line.setFocus()
    line.selectAll()
    QTest.keyClicks(line, "2.5000004")
    original = "100.87" if vk else "92.91"
    expected = "106.89" if vk else "98.93"
    assert panel.full_scale_spl_lineedit.text() == original
    if gesture in ("return", "keypad"):
        QTest.keyClick(line, Qt.Key_Return if gesture == "return" else Qt.Key_Enter)
    else:
        target = {"panel": panel, "group": line.parentWidget(),
                  "label": panel.full_scale_spl_label}.get(gesture, window)
        QTest.mouseClick(target, Qt.LeftButton, pos=QPoint(6, 6))
        if gesture == "blank-enter":
            QTest.keyClick(line, Qt.Key_Return)
    ui_qapp.processEvents()
    assert save.call_count == 1
    assert panel.saved_v2pa_factors[0] == 2.5
    assert line.text() == "2.500000"
    assert panel.full_scale_spl_lineedit.text() == expected
    assert changes == [True]
    assert panel.current_channel == 0
    assert actions == []
    QTest.keyClick(line, Qt.Key_Return)
    QTest.mouseClick(window, Qt.LeftButton, pos=QPoint(6, 6))
    assert save.call_count == 1
    assert changes == [True]


@pytest.mark.parametrize("vk", [False, True], ids=["soundcard", "vk"])
@pytest.mark.parametrize("gesture", ["return", "keypad", "margin", "label"])
@pytest.mark.parametrize("failure", ["invalid", "save", "busy", "context"])
def test_failed_confirmation_keeps_saved_result_and_rolls_back(
        panel_factory, monkeypatch, ui_qapp, vk, gesture, failure):
    from unit_test.ui.test_vkinging_calibration import CalibrationBridgeFake

    bridge = CalibrationBridgeFake(None)
    window = panel_factory(vk=vk, dialog=True, factors={0: 1.25},
                           recording_bridge=bridge, test_full_scale_peak_volts=2.5)
    panel = window.input_cal_wnd
    panel.calibration_popup = mock.Mock()
    error = calibration.VEStoreIOError("denied") if vk else calibration.MicCalibrationIOError("denied")
    save = mock.Mock(side_effect=error if failure == "save" else None)
    if vk:
        monkeypatch.setattr(panel.ve_calibration_store, "save_factor", save)
    else:
        monkeypatch.setattr(calibration, "save_mic_channel_factor", save)
    changes = []
    panel.calibration_state_changed.connect(changes.append)
    window.show()
    window.activateWindow()
    ui_qapp.processEvents()
    line = panel.v2pa_factor_lineedit
    line.setFocus()
    line.selectAll()
    QTest.keyClicks(line, "invalid" if failure == "invalid" else "2.5")
    if failure == "busy":
        bridge.service.busy = True
    elif failure == "context":
        panel.input_device["machine_id" if vk else "name"] = "Changed"
    if gesture in ("return", "keypad"):
        QTest.keyClick(line, Qt.Key_Return if gesture == "return" else Qt.Key_Enter)
    else:
        QTest.mouseClick(window if gesture == "margin" else panel.full_scale_spl_label,
                         Qt.LeftButton, pos=QPoint(6, 6))
    assert save.call_count == int(failure == "save")
    assert panel.saved_v2pa_factors == {0: 1.25}
    assert line.text() == "1.250000"
    assert panel.full_scale_spl_lineedit.text() == ("100.87" if vk else "92.91")
    assert changes == []
    assert panel.calibration_popup.call_count == 1
    QTest.keyClick(line, Qt.Key_Return)
    assert panel.calibration_popup.call_count == 1
    assert save.call_count == int(failure == "save")


def _assert_visible_layout_bounds(dialog):
    # Check actual rendered geometry, including the panel, groups and buttons.
    for parent in [dialog, *dialog.findChildren(QWidget)]:
        children = [child for child in parent.findChildren(
            QWidget, options=Qt.FindDirectChildrenOnly) if child.isVisibleTo(dialog)]
        for child in children:
            assert parent.contentsRect().contains(child.geometry()), (
                type(parent).__name__, type(child).__name__, child.geometry(),
                parent.contentsRect())
        for left, right in combinations(children, 2):
            assert not left.geometry().intersects(right.geometry()), (
                type(left).__name__, left.geometry(), type(right).__name__, right.geometry())


@pytest.fixture(scope="session")
def layout_font(ui_qapp):
    # Windows' offscreen plugin can expose no system fonts and paint blank text.
    # Keep the production font alive for the QApplication/process lifetime.
    if not QFontDatabase().families() and sys.platform == "win32":
        font_path = Path(os.environ["WINDIR"]) / "Fonts" / "simsun.ttc"
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        assert font_id != -1, f"Cannot load layout font: {font_path}"
    assert QFontDatabase().families(), "Layout checks require a renderable font"


@pytest.mark.parametrize("backend", ["soundcard", "vk"])
def test_full_scale_dialog_layout(panel_factory, ui_qapp, layout_font, backend):
    from unit_test.ui.test_vkinging_calibration import CalibrationBridgeFake

    bridge = CalibrationBridgeFake(None)
    dialog = panel_factory(
        vk=backend == "vk", dialog=True,
        factors={0: 10. if backend == "vk" else .0019},
        test_full_scale_peak_volts=2.5, recording_bridge=bridge)
    panel = dialog.input_cal_wnd
    result_box = panel.full_scale_spl_label.parentWidget()
    coefficient_label = next(label for label in result_box.findChildren(QLabel)
                             if label.text() == "校准系数（Pa/V）：")
    full_scale_row = (panel.full_scale_spl_label, panel.full_scale_spl_lineedit,
                      panel.full_scale_spl_unit_label)
    expected_tooltip = (
        "按当前测试量程 ±2.5 V 的正弦波 RMS 换算，不代表设备实际无失真测量上限。"
        if backend == "vk" else
        "按峰值为 1 的正弦波 RMS 换算，不代表设备实际无失真测量上限。")
    reminders = [label for label in panel.findChildren(QLabel)
                 if label.text() == "更换麦克风后请清除该通道旧校准并重新校准"]
    assert len(reminders) == int(backend == "vk")
    dialog.show()
    ui_qapp.processEvents()
    initial_size = dialog.size()
    for size_name, size in (("normal", initial_size), ("minimum", dialog.minimumSize())):
        dialog.resize(size)
        ui_qapp.processEvents()
        assert dialog.size() == size
        assert panel.full_scale_spl_label.text() == "量程上限"
        assert panel.full_scale_spl_unit_label.text() == "dB SPL"
        assert panel.full_scale_spl_lineedit.text() == ("118.93" if backend == "vk" else "36.54")
        assert panel.full_scale_spl_lineedit.isReadOnly()
        for control in full_scale_row:
            assert control.isVisibleTo(dialog)
            assert control.toolTip() == expected_tooltip
        for upper in (coefficient_label, panel.v2pa_factor_lineedit):
            for lower in full_scale_row:
                assert upper.geometry().bottom() < lower.geometry().top()
        _assert_visible_layout_bounds(dialog)
        # Chinese labels and the wrapped VK reminder must fit at the real size.
        for label in panel.findChildren(QLabel):
            if label.textFormat() == Qt.RichText or "<span" in label.text():
                continue  # The existing recording countdown uses HTML.
            if label.wordWrap():
                assert label.height() >= label.heightForWidth(label.width()), label.text()
            else:
                assert label.width() >= label.fontMetrics().horizontalAdvance(label.text()), label.text()
                assert label.height() >= label.fontMetrics().height(), label.text()
        for line in (panel.v2pa_factor_lineedit, panel.full_scale_spl_lineedit):
            assert line.contentsRect().width() >= line.fontMetrics().horizontalAdvance(line.text())
            assert line.contentsRect().height() >= line.fontMetrics().height()
        screenshot_dir = os.environ.get("CALIBRATION_SPL_SCREENSHOT_DIR")
        if screenshot_dir:
            target = Path(screenshot_dir)
            target.mkdir(parents=True, exist_ok=True)
            scale = os.environ.get("QT_SCALE_FACTOR", "1")
            assert dialog.grab().save(str(target / f"{backend}-{size_name}-scale-{scale}.png"))
    assert bridge.sessions == []


@pytest.mark.parametrize("channels,factors", [((), {0: .0019}), ((0,), {})])
def test_no_coefficient_or_no_selected_channel_has_no_result(panel_factory, channels, factors):
    widget = panel_factory(channels=channels, factors=factors)
    assert widget.full_scale_spl_lineedit.text() == "—"


def test_no_input_device_has_no_result(panel_factory):
    # The factory installs the log/storage/recording isolation used here too.
    widget = calibration.InputCalibration(None, [0])
    try:
        assert widget.full_scale_spl_lineedit.text() == "—"
    finally:
        widget.close_recording()
        widget.close()


def test_invalid_channel_selection_and_unavailable_state_clear_result(panel_factory):
    widget = panel_factory()
    widget.channel_combo_box.setCurrentIndex(-1)
    assert widget.full_scale_spl_lineedit.text() == "—"
    widget._select_channel(0)
    assert widget.full_scale_spl_lineedit.text() == "36.54"
    widget._set_calibration_unavailable("Unavailable")
    assert widget.full_scale_spl_lineedit.text() == "—"


def test_poll_keeps_confirmed_result_and_draft_without_editor_refresh(panel_factory, ui_qapp):
    widget = panel_factory()
    widget.show()
    widget.activateWindow()
    ui_qapp.processEvents()
    widget.v2pa_factor_lineedit.setFocus()
    widget.v2pa_factor_lineedit.selectAll()
    QTest.keyClicks(widget.v2pa_factor_lineedit, "7.25")
    with mock.patch.object(widget, "_refresh_channel_display",
                           side_effect=AssertionError("Polling reset draft")), \
            mock.patch.object(widget.coefficient_editor, "show_value",
                              side_effect=AssertionError("Polling replaced confirmed value")):
        QTest.qWait(150)
        assert widget.v2pa_factor_lineedit.text() == "7.25"
        assert widget.full_scale_spl_lineedit.text() == "36.54"
        widget.coefficient_editor.discard_pending()


@pytest.mark.parametrize("change", ["identity", "unavailable", "selected_channels", "physical_channels",
                                   "device_mapping", "calibration_available"])
def test_poll_clears_stale_soundcard_context(panel_factory, change):
    widget = panel_factory()
    if change == "identity":
        widget.input_device["name"] = "Changed"
    elif change == "unavailable":
        widget.input_device["available"] = False
    elif change == "selected_channels":
        widget.input_channels = []
    elif change == "physical_channels":
        widget.input_device["max_input_channels"] = 0
    elif change == "device_mapping":
        widget.input_device = None
    else:
        widget.calibration_available = False
    QTest.qWait(150)
    assert widget.full_scale_spl_lineedit.text() == "—"


def test_busy_and_capture_keep_confirmed_result_but_close_clears_it(panel_factory):
    from unit_test.ui.test_vkinging_calibration import CalibrationBridgeFake
    bridge = CalibrationBridgeFake(None)
    widget = panel_factory(recording_bridge=bridge)
    bridge.service.busy = True
    widget._sync_coefficient_editability()
    assert widget.full_scale_spl_lineedit.text() == "36.54"
    widget._begin_capture(0)
    widget.v2pa_factor_lineedit.clear()
    QTest.qWait(150)
    assert widget.v2pa_factor_lineedit.text() == ""
    assert widget.full_scale_spl_lineedit.text() == "36.54"
    widget.close_recording()
    assert widget.full_scale_spl_lineedit.text() == "—"


def test_close_clears_result_when_calibration_becomes_unavailable_before_poll(panel_factory):
    widget = panel_factory()
    widget.calibration_available = False
    widget.close_recording()
    assert widget.full_scale_spl_lineedit.text() == "—"


@pytest.mark.parametrize("change", ["invalidated", "identity", "physical_channels", "malformed_context"])
def test_vk_status_and_live_context_required_even_with_cached_factor(panel_factory, change):
    widget = panel_factory(vk=True, factors={0: 10.}, test_full_scale_peak_volts=2.5)
    if change == "invalidated":
        widget._ve_calibration_records[0]["status"] = "invalidated"
    elif change == "identity":
        widget.input_device["machine_id"] = "Changed"
    elif change == "physical_channels":
        widget.input_device["physical_channels"] = [1]
    else:
        widget.input_device["input_config"]["input_mode"] = None
    QTest.qWait(150)
    assert widget.saved_v2pa_factors == {0: 10.}
    assert widget.full_scale_spl_lineedit.text() == "—"
