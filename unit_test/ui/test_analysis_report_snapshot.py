import logging
from types import SimpleNamespace

import pytest
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QColor, QImage

from base.analysis_process_protocol import (
    AnalysisArtifactResult,
    AnalysisInstanceResult,
    AnalysisTaskResult,
)

from ui.sequence.analysis_report_snapshot import (
    _report_plot_widgets,
    build_analysis_report_items,
    build_analysis_report_items_from_task_result,
    export_plot_widget_png,
)
from ui.sequence.analysis_channel_preflight import AnalysisChannelSkip, preflight_analysis_channels


def test_process_result_report_snapshot_reads_saved_channel_image(tmp_path):
    image_path = tmp_path / "CH1.png"
    image_path.write_bytes(b"png-data")
    instance = AnalysisInstanceResult(
        task_id="task-1",
        config_key="声压级",
        runtime_key="声压级--通道1",
        analysis_type="SPL",
        raw_channel=0,
        source_wav_column=0,
        execution_status="分析完成",
        contributes_to_final=True,
        judgement="OK",
        metrics={
            "overall_spl": 72.5,
            "overall_lower_limit": 60.0,
            "overall_upper_limit": 80.0,
            "unit": "dB",
        },
        display_payload={},
        artifacts=(
            AnalysisArtifactResult("图片", "已保存", path=str(image_path)),
        ),
    )
    result = AnalysisTaskResult(
        task_id="task-1",
        condition_key="0.3",
        wav_path=str((tmp_path / "record.wav").resolve()),
        source="自动分析",
        execution_status="分析完成",
        judgement_status="已判定",
        final_judgement="OK",
        instance_results=(instance,),
    )

    items = build_analysis_report_items_from_task_result(
        result,
        {
            "声压级": {
                "type": "SPL",
                "limit_checked": True,
                "limit_metric": "overall_spl",
            }
        },
    )

    assert items[0]["status"] == "OK"
    assert items[0]["measurement"] == "72.5"
    assert items[0]["lower_limit"] == "60"
    assert items[0]["upper_limit"] == "80"
    assert items[0]["images"][0]["png_data"] == b"png-data"


def test_process_result_report_snapshot_records_artifact_save_failure(tmp_path):
    instance = AnalysisInstanceResult(
        task_id="task-2",
        config_key="声压级",
        runtime_key="声压级--通道1",
        analysis_type="SPL",
        raw_channel=0,
        source_wav_column=0,
        execution_status="分析完成",
        contributes_to_final=True,
        judgement="OK",
        metrics={"overall_spl": 72.5, "unit": "dB"},
        display_payload={},
        artifacts=(
            AnalysisArtifactResult(
                "CSV:实时声压级",
                "保存失败",
                error_message="磁盘空间不足",
            ),
        ),
    )
    result = AnalysisTaskResult(
        task_id="task-2",
        condition_key="0.3",
        wav_path=str((tmp_path / "record.wav").resolve()),
        source="自动分析",
        execution_status="分析完成",
        judgement_status="已判定",
        final_judgement="OK",
        instance_results=(instance,),
    )

    items = build_analysis_report_items_from_task_result(
        result,
        {"声压级": {"type": "SPL", "limit_checked": False}},
    )

    assert items[0]["status"] == "OK"
    assert "CSV:实时声压级保存失败：磁盘空间不足" in items[0]["error"]


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


@pytest.mark.parametrize("available", [[0, 2], [0]])
def test_multichannel_report_keeps_independent_results_and_missing_channel(available):
    config = {
        "display_sequence": ["SPL 1"],
        "SPL 1": {"type": "SPL", "analysis_channels": [0, 2], "limit_checked": True},
    }
    preflight = preflight_analysis_channels(config, active_input_channels=available)
    instances = [
        SimpleNamespace(
            _sequence_analysis_key="SPL 1",
            _sequence_runtime_key=f"SPL 1--通道{channel + 1}",
            _sequence_multi_channel_expansion=True,
            title_name=f"SPL 1--通道{channel + 1}",
            _product_report_analysis_state="completed",
        )
        for channel in available
    ]

    items = build_analysis_report_items(
        instances, config,
        {"SPL 1--通道1": (True, 0.0), "SPL 1--通道3": (False, 1.0)},
        {skip.item_key: skip for skip in preflight.skipped},
    )

    assert [item["name"] for item in items] == ["SPL 1--通道1", "SPL 1--通道3"]
    assert items[0]["status"] == "OK"
    if len(available) == 2:
        assert items[1]["status"] == "NG"
    else:
        assert items[1]["state"] == "skipped"
        assert "status" not in items[1]


def test_build_analysis_report_items_adds_each_preflight_skip_once_without_values():
    executed = SimpleNamespace(
        _sequence_analysis_key="valid-spl",
        title_name="valid-spl--通道1",
        _product_report_analysis_state="completed",
        result={"overall_spl": 72.0},
        _get_spl_unit=lambda: "dB SPL",
    )
    skip = AnalysisChannelSkip(
        item_key="missing-spec",
        item_type="Spec",
        requested_channel=2,
        available_channels=(0, 1),
        reason="请求通道 In3 不存在；可用通道：In1、In2",
    )

    items = build_analysis_report_items(
        [executed],
        {
            "display_sequence": ["valid-spl", "missing-spec"],
            "valid-spl": {"type": "SPL", "limit_checked": False},
            "missing-spec": {"type": "Spec", "limit_checked": False},
        },
        {},
        {"missing-spec": skip},
    )

    assert [item["name"] for item in items] == ["valid-spl", "missing-spec"]
    assert items[0]["state"] == "completed"
    skipped = items[1]
    assert skipped == {
        "name": "missing-spec",
        "item_key": "missing-spec",
        "runtime_key": "missing-spec",
        "channel_key": "",
        "type": "Spec",
        "state": "skipped",
        "reason": skip.reason,
        "error": "",
        "image_errors": [],
        "images": [],
    }
    assert "measurement" not in skipped
    assert "status" not in skipped


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


@pytest.mark.parametrize(("overall", "expected_measurement"), [(72.345, "72.345"), (None, "-")])
def test_overall_spl_report_uses_scalar_limits_and_never_substitutes_curve_peak(overall, expected_measurement):
    instance = SimpleNamespace(
        _sequence_analysis_key="SPL", title_name="SPL--通道1",
        result={"overall_spl": overall, "signal_spl": [130.0]},
    )
    item = build_analysis_report_items([instance], {"SPL": {
        "type": "SPL", "limit_checked": True, "limit_metric": "overall_spl",
        "scalar_upper_enabled": True, "scalar_upper_value": 80.0,
        "scalar_lower_enabled": True, "scalar_lower_value": 60.0,
        "constant_upper_enabled": True, "constant_upper_value": 10.0,
    }}, {"SPL--通道1": (True, 0.0)})[0]

    assert item["measurement"] == expected_measurement
    assert item["upper_limit"] == "80"
    assert item["lower_limit"] == "60"


@pytest.mark.parametrize(
    ("config", "lower", "upper"),
    [
        ({}, "-", "100"),
        ({"scalar_upper_enabled": False, "scalar_lower_enabled": True,
          "scalar_lower_value": 60.0}, "60", "-"),
        ({"limit_checked": False}, "-", "-"),
    ],
)
def test_spl_overall_report_never_uses_inactive_curve_limits(config, lower, upper):
    instance = SimpleNamespace(
        _sequence_analysis_key="SPL", title_name="SPL",
        result={"overall_spl": 72.0},
    )
    item = build_analysis_report_items([instance], {"SPL": {
        "type": "SPL", "limit_checked": True, "limit_metric": "overall_spl",
        "limit_mode": "csv", "limit_data": [[0.0], [10.0], [5.0]],
        **config,
    }}, {"SPL": (True, 0.0)})[0]

    assert item["lower_limit"] == lower
    assert item["upper_limit"] == upper


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
