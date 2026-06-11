from datetime import datetime
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base import pdf_result_exporter
from base.pdf_result_exporter import (
    build_pdf_header_rows,
    build_pdf_report_items,
    calculate_overall_status,
    export_analysis_to_pdf,
    resolve_pdf_output_path,
    summarize_result_payload,
)


def test_resolve_pdf_output_path_uses_audio_basename_and_hhmmss():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = resolve_pdf_output_path(
            {"save_dir": tmpdir},
            audio_path="D:/records/demo.wav",
            now_dt=datetime(2026, 6, 5, 14, 23, 8),
        )
        assert path == os.path.join(tmpdir, "demo_142308.pdf")


def test_resolve_pdf_output_path_sanitizes_basename_without_suffixes():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = resolve_pdf_output_path(
            {"save_dir": tmpdir},
            audio_path="D:/records/a:b?c.wav",
            now_dt=datetime(2026, 6, 5, 9, 1, 2),
        )
        assert os.path.basename(path) == "a_b_c_090102.pdf"
        assert "_2.pdf" not in path


def test_resolve_pdf_output_path_falls_back_when_audio_path_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = resolve_pdf_output_path(
            {"save_dir": tmpdir},
            audio_path=None,
            now_dt=datetime(2026, 6, 5, 9, 1, 2),
        )
        assert path == os.path.join(tmpdir, "analysis_result_090102.pdf")


def test_resolve_pdf_output_path_requires_save_dir():
    with pytest.raises(ValueError):
        resolve_pdf_output_path(
            {},
            audio_path="D:/records/demo.wav",
            now_dt=datetime(2026, 6, 5, 9, 1, 2),
        )


def test_build_pdf_report_items_places_judged_items_first_preserving_order():
    selected = ["SPL1", "Spec1", "FR1"]
    analysis_items_data = {
        "SPL1": {"type": "SPL", "result": {"signal_spl": [1, 2, 3]}},
        "Spec1": {"type": "Spec", "result": None},
        "FR1": {"type": "FR", "result": {"fr": [3, 4], "frequency_list": [100, 200]}},
    }
    result_dict = {"FR1": (False, 2.5), "SPL1": (True, 0.1)}
    items = build_pdf_report_items(selected, analysis_items_data, {}, result_dict)
    assert [x["name"] for x in items] == ["SPL1", "FR1", "Spec1"]


def test_build_pdf_report_items_preserves_tables_and_excludes_duplicate_summary_fields():
    table = {
        "title": "Mel 分析表格",
        "headers": ["整体SPL dB(A)", "热点时间(s)"],
        "rows": [["-8.13", "1.365"]],
    }
    items = build_pdf_report_items(
        ["Mel1"],
        {
            "Mel1": {
                "type": "Mel",
                "result": {
                    "overall_spl_dba": -8.13,
                    "hotspot": {"time_s": 1.365, "freq_hz": 2423.0},
                    "mel_db_a": [[1.0, 2.0]],
                    "times_s": [0.0, 1.0],
                },
                "tables": [table],
                "pdf_summary_exclude_fields": ["overall_spl_dba", "hotspot"],
            }
        },
        {},
        {},
    )

    keys = [key for key, _value in items[0]["result_rows"]]
    assert items[0]["tables"] == [table]
    assert "整体声压级 dB(A)" not in keys
    assert "热点" not in keys
    assert "A计权Mel频谱" in keys
    assert "时间轴" in keys


def test_summarize_result_payload_summarizes_long_arrays():
    rows = summarize_result_payload({"curve": list(range(100)), "score": 0.95})
    text = "\n".join(value for _key, value in rows)
    assert "100" in text
    assert "score" in "\n".join(key for key, _value in rows)


def test_summarize_result_payload_summarizes_large_nested_numeric_lists():
    matrix = [list(range(row * 10, row * 10 + 10)) for row in range(20)]
    rows = summarize_result_payload({"matrix": matrix})
    value = dict(rows)["matrix"]
    assert len(value) <= 120
    assert "shape=20x10" in value
    assert "200" in value
    assert "min=0" in value
    assert "max=199" in value
    assert "[0, 1, 2, 3, 4" not in value


def test_summarize_result_payload_labels_spl_time_domain_fields():
    rows = summarize_result_payload(
        {
            "signal_duration": [0.0, 0.1],
            "recorded_signal": [0.01, -0.01],
            "signal_spl": [91.0, 92.0],
        }
    )
    keys = [key for key, _value in rows]
    assert "时间轴" in keys
    assert "录音信号" in keys
    assert "声压级" in keys


def test_summarize_result_payload_labels_distortion_fields():
    rows = summarize_result_payload(
        {
            "freq_value": [100.0, 200.0],
            "harmonic": [[1.0, 2.0], [3.0, 4.0]],
            "thd": [3.0, 4.0],
        }
    )
    keys = [key for key, _value in rows]
    assert "频率点" in keys
    assert "谐波" in keys
    assert "失真/响度值" in keys


def test_summarize_result_payload_suppresses_duplicate_thd_raw():
    rows = summarize_result_payload(
        {
            "thd": [1.0, 2.0],
            "thd_raw": [1.0, 2.0],
        }
    )
    keys = [key for key, _value in rows]
    assert "失真/响度值" in keys
    assert "原始失真/响度值" not in keys


def test_summarize_result_payload_keeps_different_thd_raw():
    rows = summarize_result_payload(
        {
            "thd": [1.0, 2.0],
            "thd_raw": [3.0, 4.0],
        }
    )
    keys = [key for key, _value in rows]
    assert "失真/响度值" in keys
    assert "原始失真/响度值" in keys


def test_summarize_result_payload_suppresses_duplicate_splf_raw():
    rows = summarize_result_payload(
        {
            "frequency_list": [100.0, 200.0],
            "spl_db": [70.0, 71.0],
            "spl_db_raw": [70.0, 71.0],
        }
    )
    keys = [key for key, _value in rows]
    assert "频率点" in keys
    assert "声压级" in keys
    assert "原始声压级" not in keys
    assert "spl_db_raw" not in keys


def test_summarize_result_payload_keeps_different_splf_raw():
    rows = summarize_result_payload(
        {
            "spl_db": [1.0, 2.0],
            "spl_db_raw": [3.0, 4.0],
        }
    )
    keys = [key for key, _value in rows]
    assert "声压级" in keys
    assert "原始声压级" in keys


def test_summarize_result_payload_keeps_numeric_looking_text_raw_when_text_differs():
    rows = summarize_result_payload(
        {
            "spl_db": "01",
            "spl_db_raw": "1",
        }
    )
    keys = [key for key, _value in rows]
    assert "声压级" in keys
    assert "原始声压级" in keys


def test_summarize_result_payload_suppresses_exact_equal_text_raw():
    rows = summarize_result_payload(
        {
            "spl_db": "01",
            "spl_db_raw": "01",
        }
    )
    keys = [key for key, _value in rows]
    assert "声压级" in keys
    assert "原始声压级" not in keys


def test_summarize_result_payload_keeps_numeric_looking_text_sequence_raw_when_text_differs():
    rows = summarize_result_payload(
        {
            "spl_db": ["01"],
            "spl_db_raw": ["1"],
        }
    )
    keys = [key for key, _value in rows]
    assert "声压级" in keys
    assert "原始声压级" in keys


def test_summarize_result_payload_suppresses_duplicate_fr_raw():
    rows = summarize_result_payload(
        {
            "fr": [1.0, 2.0],
            "fr_raw": [1.0, 2.0],
        }
    )
    keys = [key for key, _value in rows]
    assert "频响" in keys
    assert "原始频响" not in keys
    assert "fr_raw" not in keys


def test_summarize_result_payload_labels_fr_and_keeps_unknown_fields():
    rows = summarize_result_payload(
        {
            "fr": [1.0, 2.0],
            "fr_raw": [1.5, 2.5],
            "custom_metric": 12,
        }
    )
    keys = [key for key, _value in rows]
    assert "频响" in keys
    assert "原始频响" in keys
    assert "custom_metric" in keys


def test_summarize_result_payload_labels_frequency_band_fields():
    rows = summarize_result_payload(
        {
            "band_centers": [20.0, 1000.0, 20000.0],
            "band_levels_db": [60.0, 70.0, 80.0],
            "band_levels_weighted_db": [50.0, 65.0, 78.0],
            "baseline_band_levels_weighted_db": [45.0, 60.0, 70.0],
            "delta_band_levels_weighted_db": [5.0, 5.0, 8.0],
            "plot_band_levels_weighted_db": [5.0, 5.0, 8.0],
            "overall_db": 95.4212,
            "overall_weighted_db": 86.5292,
            "weighting": "A",
            "baseline_display_mode": "delta",
            "dominant_tones": [{"interval_label": "Low", "frequency_hz": 315.0}],
            "exceeded_bands": [],
        }
    )
    keys = [key for key, _value in rows]
    assert "频段中心频率" in keys
    assert "各频段声压级" in keys
    assert "加权各频段声压级" in keys
    assert "背景基线加权频段声压级" in keys
    assert "频段声压级与背景差值" in keys
    assert "当前显示频段声压级" in keys
    assert "总声压级" in keys
    assert "计权总声压级" in keys
    assert "计权方式" in keys
    assert "背景显示方式" in keys
    assert "主音识别结果" in keys
    assert "超限频段" in keys


def test_summarize_result_payload_labels_fft_fields():
    rows = summarize_result_payload(
        {
            "frequency_bins": [100.0, 1000.0],
            "fft_db": [30.0, 50.0],
            "baseline_db": [20.0, 45.0],
            "delta_db": [10.0, 5.0],
            "plot_db": [10.0, 5.0],
            "display_mode": "delta",
            "baseline_smooth_third_octave": True,
            "n_fft": 4096,
            "window": "hann",
            "overlap_ratio": 0.5,
            "x_axis_scale": "log",
            "dominant_tones": [{"interval_label": "Overall", "frequency_hz": 1000.0}],
        }
    )
    keys = [key for key, _value in rows]
    assert "FFT 频率点" in keys
    assert "FFT 频谱" in keys
    assert "背景噪声基线" in keys
    assert "FFT 与背景差值" in keys
    assert "当前显示曲线" in keys
    assert "背景显示方式" in keys
    assert "背景基线 1/3 倍频程平滑" in keys
    assert "FFT 点数" in keys
    assert "窗函数" in keys
    assert "重叠率" in keys
    assert "横轴类型" in keys
    assert "主音识别结果" in keys


def test_summarize_result_payload_labels_mel_and_modulation_fields():
    rows = summarize_result_payload(
        {
            "mel_db_a": [[1.0, 2.0]],
            "times_s": [0.0, 1.0],
            "mel_axis": [100.0, 200.0],
            "mel_freq_edges_hz": [80.0, 160.0, 320.0],
            "overall_spl_dba": -8.13,
            "hotspot": {"time_s": 1.365, "freq_hz": 2423.0},
            "params": {"n_mels": 128},
            "mod_depth_matrix": [[0.0, 12.0]],
            "signal_freq_axis_hz": [1200.0],
            "mod_freq_axis_hz": [75.0],
            "main_tone_results": [{"target_signal_freq_hz": 1200.0}],
            "global_hotspots": [],
            "mechanical_references": [{"freq_hz": 75.0}],
            "mechanical_mod_freqs_hz": [75.0],
            "stft_params": {"n_fft": 2048},
            "threshold_percent": 10.0,
            "min_modulation_depth_percent": 1.0,
        }
    )
    keys = [key for key, _value in rows]
    assert "A计权Mel频谱" in keys
    assert "时间轴" in keys
    assert "Mel频率轴" in keys
    assert "Mel频带边界" in keys
    assert "整体声压级 dB(A)" in keys
    assert "热点" in keys
    assert "参数" in keys
    assert "调制深度矩阵" in keys
    assert "信号频率轴" in keys
    assert "调制频率轴" in keys
    assert "主音分析结果" in keys
    assert "全局热点" in keys
    assert "机械参考频率" in keys
    assert "机械调制频率" in keys
    assert "STFT参数" in keys
    assert "阈值(%)" in keys
    assert "最小调制深度(%)" in keys


def test_summarize_result_payload_labels_pr_fields_and_status_values():
    rows = summarize_result_payload(
        {
            "max_pr_db": 4.63635,
            "max_pr_frequency": 1096.44,
            "max_pr_frequency_hz": 1096.44,
            "max_exceed_db": 0,
            "decision_status": "not_applicable",
            "tone_count": 0,
        }
    )
    row_dict = dict(rows)

    assert "最大突出比(dB)" in row_dict
    assert "最大突出比频率(Hz)" in row_dict
    assert "最大超限(dB)" in row_dict
    assert "判定状态" in row_dict
    assert "主音数量" in row_dict
    assert row_dict["判定状态"] == "不适用"
    assert "not_applicable" not in "\n".join(row_dict.values())


def test_calculate_overall_status_uses_selected_judged_items_only():
    assert calculate_overall_status(
        ["SPL1", "FR1"], {"SPL1": (True, 0.1), "FR1": (True, 0.2)}
    ) == "OK"
    assert calculate_overall_status(
        ["SPL1", "FR1"], {"SPL1": (True, 0.1), "FR1": (False, 2.5)}
    ) == "NG"
    assert calculate_overall_status(["Spec1"], {"SPL1": (False, 2.5)}) == "-"


def test_build_pdf_header_rows_includes_overall_status():
    rows = build_pdf_header_rows(
        audio_name="demo.wav",
        audio_path="D:/records/demo.wav",
        sn="SN001",
        product_model="MODEL-A",
        date_text="2026/6/5 14:23:08",
        overall_status="NG",
    )
    assert ("总体结果", "NG") in rows


def test_build_pdf_header_rows_preserves_unjudged_overall_status():
    rows = build_pdf_header_rows(
        audio_name="demo.wav",
        audio_path=None,
        sn=None,
        product_model=None,
        date_text="2026/6/5 14:23:08",
        overall_status="-",
    )
    assert ("总体结果", "-") in rows
    assert ("S/N", "-") not in rows
    assert ("产品型号", "-") not in rows


def test_summary_table_omits_deviation_result_summary_column(monkeypatch):
    rendered_rows = []

    def fake_draw(self, rows):
        rendered_rows.extend(rows)

    monkeypatch.setattr(pdf_result_exporter._PdfPainter, "_draw_three_col_table", fake_draw, raising=False)
    layout = pdf_result_exporter._PdfPainter.__new__(pdf_result_exporter._PdfPainter)

    layout.summary_table(
        [
            {
                "name": "SPL1",
                "type": "SPL",
                "status": "OK",
                "deviation": "0",
                "result_rows": [("结果", "1")],
            },
            {
                "name": "SPLF1",
                "type": "SPLF",
                "status": "NG",
                "deviation": "91.09",
                "result_rows": [("频率点", "28 点")],
            },
        ]
    )

    assert rendered_rows == [
        ("分析项", "类型", "判定"),
        ("SPL1", "SPL", "OK"),
        ("SPLF1", "SPLF", "NG"),
    ]
    assert "偏差/结果摘要" not in "\n".join(" ".join(row) for row in rendered_rows)


def test_render_pdf_starts_details_on_new_page_after_summary(monkeypatch):
    calls = []

    class FakePainter:
        def __init__(self, writer):
            calls.append(("painter", writer))

        def isActive(self):
            return True

        def end(self):
            calls.append(("end",))

    class FakeLayout:
        def __init__(self, writer):
            calls.append(("layout", writer))

        def bind(self, painter):
            calls.append(("bind", painter))

        def text(self, text, **kwargs):
            calls.append(("text", text))

        def rows(self, rows, **kwargs):
            calls.append(("rows", rows))

        def spacer(self, height):
            calls.append(("spacer", height))

        def summary_table(self, items):
            calls.append(("summary_table", items))

        def new_page(self):
            calls.append(("new_page",))

        def section_title(self, title):
            calls.append(("section_title", title))

        def _status_color(self, value):
            return value

        def table(self, headers, rows, title=None):
            calls.append(("table", headers, rows, title))

        def image(self, image_path, title=None):
            calls.append(("image", image_path, title))

    import PyQt5.QtGui

    fake_writer = object()
    monkeypatch.setattr(pdf_result_exporter, "_ensure_qt_application", lambda: None)
    monkeypatch.setattr(pdf_result_exporter, "_configure_pdf_writer", lambda file_path: fake_writer)
    monkeypatch.setattr(pdf_result_exporter, "_PdfPainter", FakeLayout)
    monkeypatch.setattr(PyQt5.QtGui, "QPainter", FakePainter)

    report_items = [
        {
            "name": "SPL1",
            "type": "SPL",
            "status": "OK",
            "deviation": "0",
            "result_rows": [("score", "1.23")],
            "tables": [
                {
                    "title": "Mel 分析表格",
                    "headers": ["整体SPL dB(A)", "热点时间(s)"],
                    "rows": [["-8.13", "1.365"]],
                }
            ],
            "images": [{"title": "plot", "path": "plot.png"}],
        }
    ]
    pdf_result_exporter._render_pdf("out.pdf", header_rows=[("音频文件", "demo.wav")], report_items=report_items)

    summary_index = calls.index(("summary_table", report_items))
    new_page_index = calls.index(("new_page",))
    first_section_index = calls.index(("section_title", "SPL1  SPL"))
    assert calls[0] == ("layout", fake_writer)
    assert calls[1] == ("painter", fake_writer)
    assert calls[2][0] == "bind"
    assert calls[3:6] == [
        ("text", "分析结果报告"),
        ("rows", [("音频文件", "demo.wav")]),
        ("spacer", 12),
    ]
    assert calls[6] == ("text", "结果汇总")
    assert summary_index < new_page_index < first_section_index
    assert ("text", "判定: OK") in calls
    assert not any(call[0] == "text" and "偏差" in call[1] for call in calls)
    table_index = calls.index(("table", ["整体SPL dB(A)", "热点时间(s)"], [["-8.13", "1.365"]], "Mel 分析表格"))
    image_index = calls.index(("image", "plot.png", "plot"))
    assert first_section_index < table_index < image_index


def test_render_pdf_does_not_add_blank_page_without_report_items(monkeypatch):
    calls = []

    class FakePainter:
        def __init__(self, writer):
            pass

        def isActive(self):
            return True

        def end(self):
            calls.append(("end",))

    class FakeLayout:
        def __init__(self, writer):
            pass

        def bind(self, painter):
            pass

        def text(self, text, **kwargs):
            calls.append(("text", text))

        def rows(self, rows, **kwargs):
            calls.append(("rows", rows))

        def spacer(self, height):
            calls.append(("spacer", height))

        def summary_table(self, items):
            calls.append(("summary_table", items))

        def new_page(self):
            calls.append(("new_page",))

        def section_title(self, title):
            calls.append(("section_title", title))

    import PyQt5.QtGui

    monkeypatch.setattr(pdf_result_exporter, "_ensure_qt_application", lambda: None)
    monkeypatch.setattr(pdf_result_exporter, "_configure_pdf_writer", lambda file_path: object())
    monkeypatch.setattr(pdf_result_exporter, "_PdfPainter", FakeLayout)
    monkeypatch.setattr(PyQt5.QtGui, "QPainter", FakePainter)

    pdf_result_exporter._render_pdf("out.pdf", header_rows=[("音频文件", "demo.wav")], report_items=[])

    assert ("summary_table", []) in calls
    assert ("new_page",) not in calls
    assert not any(call[0] == "section_title" for call in calls)


def test_export_analysis_to_pdf_creates_non_empty_pdf(tmp_path):
    cfg = {"enabled": True, "save_dir": str(tmp_path), "save_items": ["SPL1"]}
    ret = export_analysis_to_pdf(
        cfg,
        audio_path="D:/records/demo.wav",
        sn="SN001",
        product_model="MODEL-A",
        date_text="2026/6/5 14:23:08",
        analysis_items_data={"SPL1": {"type": "SPL", "result": {"score": 1.23}}},
        analysis_config={"SPL1": {"type": "SPL"}},
        analysis_result_dict={"SPL1": (True, 0.0)},
        image_exporters={},
        now_dt=datetime(2026, 6, 5, 14, 23, 8),
    )
    assert ret.ok
    assert os.path.exists(ret.file_path)
    assert os.path.getsize(ret.file_path) > 100


def test_export_analysis_to_pdf_renders_wide_table_without_float_drawline_error(tmp_path):
    cfg = {"enabled": True, "save_dir": str(tmp_path), "save_items": ["Mel1"]}
    ret = export_analysis_to_pdf(
        cfg,
        audio_path="D:/records/demo.wav",
        sn="SN001",
        product_model="MODEL-A",
        date_text="2026/6/5 14:23:08",
        analysis_items_data={
            "Mel1": {
                "type": "Mel",
                "result": {"mel_db_a": [[1.0, 2.0]]},
                "tables": [
                    {
                        "title": "分析表格",
                        "headers": ["整体SPL dB(A)", "热点时间(s)", "热点频率(kHz)", "热点Mel", "热点dB(A)"],
                        "rows": [["-8.13", "1.365", "2.423", "1685.3", "-9.02"]],
                    }
                ],
            }
        },
        analysis_config={"Mel1": {"type": "Mel"}},
        analysis_result_dict={},
        image_exporters={},
        now_dt=datetime(2026, 6, 5, 14, 23, 8),
    )

    assert ret.ok, ret.message
    assert os.path.exists(ret.file_path)
    assert os.path.getsize(ret.file_path) > 100


def test_export_analysis_to_pdf_invokes_image_exporters_embeds_image_and_cleans_temp(tmp_path):
    from PyQt5.QtGui import QColor, QImage

    seen_dirs = []

    def image_exporter(output_dir):
        seen_dirs.append(output_dir)
        image_path = os.path.join(output_dir, "plot.png")
        image = QImage(32, 24, QImage.Format_RGB32)
        image.fill(QColor("green"))
        assert image.save(image_path)
        return [{"title": "Plot", "path": image_path}]

    cfg = {"enabled": True, "save_dir": str(tmp_path), "save_items": ["SPL1"]}
    ret = export_analysis_to_pdf(
        cfg,
        audio_path="D:/records/demo.wav",
        sn="SN001",
        product_model="MODEL-A",
        date_text="2026/6/5 14:23:08",
        analysis_items_data={"SPL1": {"type": "SPL", "result": {"score": 1.23}}},
        analysis_config={"SPL1": {"type": "SPL"}},
        analysis_result_dict={"SPL1": (True, 0.0)},
        image_exporters={"SPL1": image_exporter},
        now_dt=datetime(2026, 6, 5, 14, 23, 8),
    )
    assert ret.ok
    assert seen_dirs
    assert os.path.exists(ret.file_path)
    assert os.path.getsize(ret.file_path) > 100
    with open(ret.file_path, "rb") as f:
        pdf_bytes = f.read()
    assert b"/Image" in pdf_bytes or b"/Subtype /Image" in pdf_bytes
    assert not os.path.exists(seen_dirs[0])


def test_export_analysis_to_pdf_isolates_same_named_images_until_render(tmp_path, monkeypatch):
    render_images = {}

    def fake_render(_file_path, *, header_rows, report_items):
        assert header_rows
        for item in report_items:
            images = item.get("images") or []
            assert len(images) == 1
            image_path = images[0]["path"]
            assert os.path.exists(image_path)
            render_images[item["name"]] = (image_path, open(image_path, "rb").read())

    def image_exporter(payload):
        def _export(output_dir):
            image_path = os.path.join(output_dir, "plot.png")
            with open(image_path, "wb") as f:
                f.write(payload)
            return [{"title": "Plot", "path": image_path}]

        return _export

    monkeypatch.setattr(pdf_result_exporter, "_render_pdf", fake_render)
    cfg = {"enabled": True, "save_dir": str(tmp_path), "save_items": ["SPL1", "FR1"]}
    ret = export_analysis_to_pdf(
        cfg,
        audio_path="D:/records/demo.wav",
        sn="SN001",
        product_model="MODEL-A",
        date_text="2026/6/5 14:23:08",
        analysis_items_data={
            "SPL1": {"type": "SPL", "result": {"score": 1.23}},
            "FR1": {"type": "FR", "result": {"score": 2.34}},
        },
        analysis_config={"SPL1": {"type": "SPL"}, "FR1": {"type": "FR"}},
        analysis_result_dict={"SPL1": (True, 0.0), "FR1": (True, 0.0)},
        image_exporters={
            "SPL1": image_exporter(b"first image"),
            "FR1": image_exporter(b"second image"),
        },
        now_dt=datetime(2026, 6, 5, 14, 23, 8),
    )
    assert ret.ok
    assert render_images["SPL1"][0] != render_images["FR1"][0]
    assert render_images["SPL1"][1] == b"first image"
    assert render_images["FR1"][1] == b"second image"


def test_export_analysis_to_pdf_fails_without_save_dir(tmp_path):
    ret = export_analysis_to_pdf(
        {"enabled": True, "save_items": ["SPL1"]},
        audio_path="D:/records/demo.wav",
        sn="SN001",
        product_model="MODEL-A",
        date_text="2026/6/5 14:23:08",
        analysis_items_data={"SPL1": {"type": "SPL", "result": {"score": 1.23}}},
        analysis_config={"SPL1": {"type": "SPL"}},
        analysis_result_dict={"SPL1": (True, 0.0)},
        image_exporters={},
        now_dt=datetime(2026, 6, 5, 14, 23, 8),
    )
    assert not ret.ok
    assert "保存目录" in ret.message
