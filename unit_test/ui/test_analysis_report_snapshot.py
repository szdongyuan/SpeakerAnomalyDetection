import logging
from types import SimpleNamespace

import pytest
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QColor, QImage

from ui.sequence.analysis_report_snapshot import (
    _report_plot_widgets,
    build_analysis_report_items,
    export_plot_widget_png,
)


class _StubTitleLabel:
    def __init__(self, text="稳态平均响度：33.1 sone"):
        self.text = text
        self.opts = {"color": "k", "size": "14px"}
        self.visible = True

    def isVisible(self):
        return self.visible


class _StubLayout:
    def __init__(self):
        self.activations = 0

    def activate(self):
        self.activations += 1


class _StubPlotItem:
    def __init__(self, title="稳态平均响度：33.1 sone"):
        self.titleLabel = _StubTitleLabel(title)
        self.layout = _StubLayout()
        self.title_calls = []

    def setTitle(self, title=None, **options):
        self.title_calls.append((title, options))
        self.titleLabel.visible = title is not None
        if title is not None:
            self.titleLabel.text = title
            self.titleLabel.opts.update(options)


class _StubPlotWidget:
    def __init__(self, title="稳态平均响度：33.1 sone"):
        self.plot_item = _StubPlotItem(title)
        self.current_size = QSize(640, 480)

    def getPlotItem(self):
        return self.plot_item

    def size(self):
        return QSize(self.current_size)

    def resize(self, *args):
        self.current_size = QSize(args[0]) if len(args) == 1 else QSize(*args)

    def ensurePolished(self):
        return None


def test_build_analysis_report_items_keeps_item_without_threshold():
    instance = SimpleNamespace(
        _sequence_analysis_key="响度 (LOUD) 1",
        title_name="响度 (LOUD) 1--通道1",
        _product_report_analysis_state="completed",
    )

    items = build_analysis_report_items(
        [instance],
        {
            "响度 (LOUD) 1": {
                "type": "LOUD",
                "limit_checked": False,
            }
        },
        {},
    )

    assert len(items) == 1
    assert items[0]["name"] == "响度 (LOUD) 1"
    assert items[0]["status"] == "未启用判定"
    assert items[0]["state"] == "completed"
    assert items[0]["images"] == []


def test_build_analysis_report_items_adds_structured_spl_measurement_and_limits():
    instance = SimpleNamespace(
        _sequence_analysis_key="声压级",
        title_name="声压级--通道1",
        _product_report_analysis_state="completed",
        result={
            "overall_spl": 72.345,
            "signal_spl": [70.0, 72.345],
        },
        _get_spl_unit=lambda: "dB(A)",
    )

    items = build_analysis_report_items(
        [instance],
        {
            "声压级": {
                "type": "SPL",
                "limit_checked": True,
                "limit_data": [
                    [0.0, 1.0],
                    [80.0, 82.0],
                    [60.0, 61.0],
                ],
            }
        },
        {"声压级--通道1": (True, 0.0)},
    )

    assert items[0]["measurement"] == "曲线判定"
    assert items[0]["lower_limit"] == "见分析图"
    assert items[0]["upper_limit"] == "见分析图"
    assert items[0]["unit"] == "dB(A)"
    assert items[0]["status"] == "OK"


def test_curve_report_keeps_constant_limit_values_without_scalar_measurement():
    instance = SimpleNamespace(
        _sequence_analysis_key="声压级",
        title_name="声压级--通道1",
        _product_report_analysis_state="completed",
        result={
            "overall_spl": 72.345,
            "signal_spl": [70.0, 72.345],
        },
        _get_spl_unit=lambda: "dB(A)",
    )

    items = build_analysis_report_items(
        [instance],
        {
            "声压级": {
                "type": "SPL",
                "limit_checked": True,
                "constant_lower_enabled": True,
                "constant_lower_value": 60.0,
                "constant_upper_enabled": True,
                "constant_upper_value": 80.0,
            }
        },
        {"声压级--通道1": (True, 0.0)},
    )

    assert items[0]["measurement"] == "曲线判定"
    assert items[0]["lower_limit"] == "60"
    assert items[0]["upper_limit"] == "80"
    assert items[0]["unit"] == "dB(A)"


def test_fba_curve_report_does_not_present_overall_level_as_judged_value():
    instance = SimpleNamespace(
        _sequence_analysis_key="频段分析",
        title_name="频段分析--通道1",
        _product_report_analysis_state="completed",
        result={
            "overall_weighted_db": 75.0,
            "upper_limits": [80.0, 82.0],
            "lower_limits": [60.0, 61.0],
            "weighting": "A",
        },
    )

    items = build_analysis_report_items(
        [instance],
        {
            "频段分析": {
                "type": "FBA",
                "limit_checked": True,
            }
        },
        {"频段分析--通道1": (False, 1.5)},
    )

    assert items[0]["measurement"] == "曲线判定"
    assert items[0]["lower_limit"] == "见分析图"
    assert items[0]["upper_limit"] == "见分析图"
    assert items[0]["unit"] == "dB(A)"
    assert items[0]["status"] == "NG"
    assert items[0]["deviation"] == "1.5"


def test_build_analysis_report_items_adds_loudness_scalar_measurement():
    instance = SimpleNamespace(
        _sequence_analysis_key="响度",
        title_name="响度--通道1",
        _product_report_analysis_state="completed",
        result={
            "summary": {
                "steady_state_average_sone": 4.25,
            }
        },
    )

    items = build_analysis_report_items(
        [instance],
        {
            "响度": {
                "type": "LOUD",
                "limit_checked": True,
                "limit_metric": "steady_state_average",
                "curve_limit_unit": "sone",
                "scalar_lower_enabled": True,
                "scalar_lower_value": 3.0,
                "scalar_upper_enabled": True,
                "scalar_upper_value": 6.0,
            }
        },
        {"响度--通道1": (True, 0.0)},
    )

    assert items[0]["measurement"] == "4.25"
    assert items[0]["lower_limit"] == "3"
    assert items[0]["upper_limit"] == "6"
    assert items[0]["unit"] == "sone"


def test_loudness_curve_report_uses_curve_limits_not_inactive_scalar_limits():
    instance = SimpleNamespace(
        _sequence_analysis_key="响度",
        title_name="响度--通道1",
        _product_report_analysis_state="completed",
        result={"loudness_sone": [2.0, 4.0]},
    )

    items = build_analysis_report_items(
        [instance],
        {
            "响度": {
                "type": "LOUD",
                "limit_checked": True,
                "limit_metric": "curve_y",
                "curve_limit_unit": "sone",
                "curve_lower_enabled": True,
                "curve_lower_value": 1.0,
                "curve_upper_enabled": True,
                "curve_upper_value": 5.0,
                "scalar_lower_enabled": True,
                "scalar_lower_value": 30.0,
                "scalar_upper_enabled": True,
                "scalar_upper_value": 40.0,
            }
        },
        {"响度--通道1": (True, 0.0)},
    )

    assert items[0]["measurement"] == "曲线判定"
    assert items[0]["lower_limit"] == "1"
    assert items[0]["upper_limit"] == "5"


def test_report_plot_widgets_only_uses_explicit_business_plot_provider(
    monkeypatch,
):
    class _PlotWidget:
        pass

    business_plot = _PlotWidget()
    hidden_plot = _PlotWidget()
    instance = SimpleNamespace(
        get_report_plot_widgets=lambda: [business_plot],
        findChildren=lambda _widget_type: [hidden_plot],
    )
    monkeypatch.setattr(
        "ui.sequence.analysis_report_snapshot.pg.PlotWidget",
        _PlotWidget,
    )

    assert _report_plot_widgets(instance) == [business_plot]


def test_report_plot_widgets_does_not_fall_back_to_child_plots(monkeypatch):
    class _PlotWidget:
        pass

    hidden_plot = _PlotWidget()
    instance = SimpleNamespace(
        analysis_plot=hidden_plot,
        findChildren=lambda _widget_type: [hidden_plot],
    )
    monkeypatch.setattr(
        "ui.sequence.analysis_report_snapshot.pg.PlotWidget",
        _PlotWidget,
    )

    assert _report_plot_widgets(instance) == []


def test_export_plot_widget_png_hides_title_and_restores_plot(monkeypatch):
    class _Exporter:
        def __init__(self, _plot_item):
            self.params = {}

        def parameters(self):
            return self.params

        def export(self, *, toBytes=False):
            assert toBytes is True
            assert self.params["width"] == 1000
            image = QImage(320, 180, QImage.Format_RGB32)
            image.fill(QColor("white"))
            return image

    monkeypatch.setattr(
        "ui.sequence.analysis_report_snapshot.pg.exporters.ImageExporter",
        _Exporter,
    )
    plot_widget = _StubPlotWidget()

    png_data = export_plot_widget_png(plot_widget)

    assert png_data.startswith(b"\x89PNG")
    assert plot_widget.current_size == QSize(640, 480)
    assert plot_widget.plot_item.title_calls[0] == (None, {})
    assert plot_widget.plot_item.title_calls[-1] == (
        "稳态平均响度：33.1 sone",
        {"color": "k", "size": "14px"},
    )
    assert plot_widget.plot_item.layout.activations >= 2


def test_export_plot_widget_png_restores_plot_after_export_failure(monkeypatch):
    plot_widget = _StubPlotWidget("失败后仍需恢复")
    original_size = QSize(plot_widget.size())

    class _FailingExporter:
        def __init__(self, _plot_item):
            self.params = {}

        def parameters(self):
            return self.params

        def export(self, *, toBytes=False):
            assert toBytes is True
            raise RuntimeError("snapshot failed")

    monkeypatch.setattr(
        "ui.sequence.analysis_report_snapshot.pg.exporters.ImageExporter",
        _FailingExporter,
    )

    with pytest.raises(RuntimeError, match="snapshot failed"):
        export_plot_widget_png(plot_widget)

    assert plot_widget.size() == original_size
    assert plot_widget.plot_item.titleLabel.isVisible()
    assert plot_widget.plot_item.titleLabel.text == "失败后仍需恢复"
    assert plot_widget.plot_item.layout.activations >= 2


def test_build_analysis_report_items_isolates_single_plot_failure(
    monkeypatch,
):
    failed_plot = _StubPlotWidget("失败图")
    successful_plot = _StubPlotWidget("成功图")
    instance = SimpleNamespace(
        _sequence_analysis_key="分析项",
        title_name="分析项--通道1",
        _product_report_analysis_state="completed",
        get_report_plot_widgets=lambda: [failed_plot, successful_plot],
    )

    def fake_export(plot_widget):
        if plot_widget is failed_plot:
            raise RuntimeError("capture failed")
        return b"\x89PNG\r\n\x1a\n"

    monkeypatch.setattr(
        "ui.sequence.analysis_report_snapshot.pg.PlotWidget",
        _StubPlotWidget,
    )
    monkeypatch.setattr(
        "ui.sequence.analysis_report_snapshot.export_plot_widget_png",
        fake_export,
    )

    items = build_analysis_report_items(
        [instance],
        {"分析项": {"type": "SPEC", "limit_checked": False}},
        {},
    )

    assert len(items[0]["image_errors"]) == 1
    assert "失败图: capture failed" in items[0]["image_errors"][0]
    assert items[0]["images"] == [
        {
            "caption": "成功图",
            "png_data": b"\x89PNG\r\n\x1a\n",
        }
    ]
def test_export_plot_widget_png_logs_only_slow_snapshot(
    monkeypatch,
    caplog,
):
    class _Exporter:
        def __init__(self, _plot_item):
            self.params = {}

        def parameters(self):
            return self.params

        def export(self, *, toBytes=False):
            assert toBytes is True
            image = QImage(320, 180, QImage.Format_RGB32)
            image.fill(QColor("white"))
            return image

    plot_widget = _StubPlotWidget("慢截图")
    monkeypatch.setattr(
        "ui.sequence.analysis_report_snapshot.pg.exporters.ImageExporter",
        _Exporter,
    )
    monkeypatch.setattr(
        "ui.sequence.analysis_report_snapshot._REPORT_PLOT_SLOW_SECONDS",
        0.0,
    )
    caplog.set_level(
        logging.WARNING,
        logger="ui.sequence.analysis_report_snapshot",
    )

    export_plot_widget_png(plot_widget)

    assert "product_pdf_plot_snapshot_slow" in caplog.text
    assert "width=1000" in caplog.text
