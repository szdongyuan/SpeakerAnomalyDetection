import base64
from dataclasses import replace

import pyqtgraph as pg
import pytest

from base.analysis_process_protocol import AnalysisInstanceResult
from PyQt5.QtCore import QBuffer, QByteArray, QIODevice
from PyQt5.QtGui import QColor, QImage
from PyQt5.QtWidgets import QFrame, QLabel
from ui.analysis_multichannel_result_window import (
    AnalysisMultichannelResultWindow,
)


def _result(raw_channel, judgement):
    return AnalysisInstanceResult(
        task_id="manual-task",
        config_key="声压级",
        runtime_key=f"声压级--通道{raw_channel + 1}",
        analysis_type="SPL",
        raw_channel=raw_channel,
        source_wav_column=raw_channel,
        execution_status="分析完成",
        contributes_to_final=True,
        judgement=judgement,
        metrics={"overall_spl": 72.0 + raw_channel},
        display_payload={
            "kind": "curve",
            "x": [0.0, 0.01],
            "y": [72.0, 73.0],
            "lower": [60.0, 60.0],
            "upper": [80.0, 80.0],
            "x_label": "Time (s)",
            "y_label": "SPL (dB)",
        },
    )


def test_one_item_window_uses_dropdown_to_switch_channels(ui_qapp):
    window = AnalysisMultichannelResultWindow(
        "声压级",
        [_result(2, "NG"), _result(0, "OK")],
        channel_labels={"CH1": "前", "CH3": "左"},
    )

    assert window.windowTitle() == "声压级"
    assert window.channel_combo.count() == 2
    assert window.channel_combo.itemData(0) == 0
    assert window.channel_combo.itemData(1) == 2
    assert window.channel_combo.itemText(0) == "CH1(前)"
    assert window.channel_combo.itemText(1) == "CH3(左)"
    assert window.channel_combo.width() == 88
    assert window.channel_combo.height() == 28
    assert "QComboBox#analysisResultChannelCombo::down-arrow" in window.styleSheet()
    assert "ui/ui_analysis_config/assets/combobox_down_arrow.svg" in (
        window.styleSheet()
    )
    assert "border-left: 1px solid" in window.styleSheet()
    assert window.objectName() == "analysisResultWindow"
    assert window.layout().getContentsMargins() == (10, 10, 10, 10)
    selector_panel = window.findChild(QFrame, "analysisResultChannelPanel")
    assert selector_panel is not None
    assert window.layout().indexOf(window._content) == 0
    assert window.layout().indexOf(selector_panel) == 1
    assert selector_panel.width() == 88
    assert window._pages[0].backgroundBrush().color().name() == "#fbfcfe"
    window.close()


def test_spec_image_scales_inside_small_window_without_clipping(ui_qapp):
    image = QImage(1200, 800, QImage.Format_RGB32)
    image.fill(QColor("#4CAF50"))
    encoded = QByteArray()
    buffer = QBuffer(encoded)
    assert buffer.open(QIODevice.WriteOnly)
    assert image.save(buffer, "PNG")
    buffer.close()
    result = AnalysisInstanceResult(
        task_id="manual-task",
        config_key="频谱分析 (Spec) 1",
        runtime_key="频谱分析 (Spec) 1--通道1",
        analysis_type="Spec",
        raw_channel=0,
        source_wav_column=0,
        execution_status="分析完成",
        contributes_to_final=False,
        judgement=None,
        metrics={},
        display_payload={
            "kind": "image",
            "png_base64": base64.b64encode(bytes(encoded)).decode("ascii"),
        },
    )
    window = AnalysisMultichannelResultWindow("频谱分析 (Spec) 1", [result])
    window.resize(760, 520)
    window.show()
    ui_qapp.processEvents()

    label = window._pages[0]
    assert isinstance(label, QLabel)
    assert label.pixmap() is not None
    assert label.pixmap().width() <= label.contentsRect().width()
    assert label.pixmap().height() <= label.contentsRect().height()
    window.close()


def test_channel_without_position_label_keeps_physical_channel_name(ui_qapp):
    window = AnalysisMultichannelResultWindow(
        "声压级",
        [_result(0, "OK")],
        channel_labels={},
    )

    assert window.channel_combo.itemText(0) == "CH1"
    window.close()


def test_result_window_title_includes_manual_analysis_source(ui_qapp):
    window = AnalysisMultichannelResultWindow(
        "声压级",
        [_result(0, "OK")],
        source_label="A口 / 0.1",
    )

    assert window.windowTitle() == "声压级 — A口 / 0.1"
    window.close()


def test_fba_result_window_draws_discrete_bars_with_band_labels(ui_qapp):
    result = AnalysisInstanceResult(
        task_id="manual-task",
        config_key="频段能量 (FBA) 1",
        runtime_key="频段能量 (FBA) 1--通道1",
        analysis_type="FBA",
        raw_channel=0,
        source_wav_column=0,
        execution_status="分析完成",
        contributes_to_final=True,
        judgement="NG",
        metrics={"weighting": "A"},
        display_payload={
            "kind": "bar",
            "x": [100.0, 1000.0, 10_000.0],
            "y": [51.0, 62.0, 48.0],
            "labels": ["100", "1k", "10k"],
            "lower": [40.0, 40.0, 40.0],
            "upper": [70.0, 70.0, 70.0],
            "out_mask": [False, False, True],
            "x_label": "Frequency Band",
            "y_label": "Band SPL [dB(A)]",
        },
    )

    window = AnalysisMultichannelResultWindow(
        "频段能量 (FBA) 1",
        [result],
    )

    plot_item = window._pages[0].getPlotItem()
    assert any(isinstance(item, pg.BarGraphItem) for item in plot_item.items)
    assert plot_item.getAxis("bottom")._tickLevels[0] == [
        (0, "100"),
        (1, "1k"),
        (2, "10k"),
    ]
    assert not plot_item.getAxis("top").isVisible()
    window.close()


@pytest.mark.parametrize("segment_count", [0, 1, 10, 100])
def test_manual_spl_keeps_whole_curve_without_segment_marks(ui_qapp, segment_count):
    results = []
    duration = max(1, segment_count)
    for channel in (0, 2):
        result = _result(channel, "OK")
        payload = result.display_payload.to_dict()
        payload.update(
            x=[0, duration / 2, duration], y=[70, 72, 71], lower=[], upper=[],
            overall_spl=43.21 + channel, unit="dBA", y_label="SPL (dBA)",
            recording_time_range=[0, duration],
            segment_boundaries=list(range(1, segment_count)),
            segment_annotations=[
                {"start": i, "end": i + 1, "label": f"{i}～{i + 1} s",
                 "overall_spl": 43.20 + channel, "unit": "dBA",
                 "execution_status": "分析完成"}
                for i in range(segment_count)
            ],
        )
        results.append(replace(result, display_payload=payload))
    window = AnalysisMultichannelResultWindow("声压级", results)
    window.resize(760, 520)
    window.show()
    for index, page in enumerate(window._pages):
        window.channel_combo.setCurrentIndex(index)
        ui_qapp.processEvents()
        plot = page.getPlotItem()
        assert not plot.getAxis("top").isVisible()
        assert not any(isinstance(item, pg.InfiniteLine) for item in plot.items)
        assert plot.titleLabel.text == f"总体声压级：{43.21 + results[index].raw_channel:.2f} dBA"
        assert plot.getAxis("left").labelText == "SPL (dBA)"
        assert plot.vb.viewRange()[0] == [0, duration]
        assert len(plot.listDataItems()) == 1
        page.setXRange(0.25, 0.75, padding=0)
        assert plot.vb.viewRange()[0] == [0.25, 0.75]
    window.close()
