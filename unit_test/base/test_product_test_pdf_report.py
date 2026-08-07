import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from PyQt5.QtCore import QByteArray, QBuffer, QIODevice
from PyQt5.QtGui import QColor, QImage, QPainter
from PyQt5.QtWidgets import QApplication

from base.product_test_pdf_report import (
    _ReportPainter,
    _configure_pdf_writer,
    _local_mac_address_text,
    _report_display_value,
    export_product_test_pdf,
    product_report_signature,
    resolve_product_pdf_output_path,
)


def make_report_data():
    return {
        "group_id": "cycle-1",
        "product_model": "S004-1",
        "barcode": "SN123456",
        "created_at": "2026-08-06T10:30:20",
        "test_time": "2026-08-06 10:30:20",
        "overall_result": "NG",
        "conditions": [
            {
                "key": "01",
                "name": "6000 rpm",
                "result": "OK",
                "recorded_path": "D:/audio/S004-1_SN123456_6000rpm.wav",
                "record_time": "2026-08-06 10:30:20",
                "sample_rate": 48000,
                "analysis_results": [
                    {"name": "声压级", "status": "OK", "deviation": "0.2"},
                ],
            },
            {
                "key": "02",
                "name": "7000 rpm",
                "result": "NG",
                "recorded_path": "D:/audio/S004-1_SN123456_7000rpm.wav",
                "record_time": "2026-08-06 10:30:25",
                "sample_rate": 48000,
                "analysis_results": [
                    {"name": "频段能量", "status": "NG", "deviation": "1.5"},
                ],
            },
        ],
    }


def make_png_bytes(color="orange"):
    image = QImage(320, 180, QImage.Format_RGB32)
    image.fill(QColor(color))
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.WriteOnly)
    assert image.save(buffer, "PNG")
    buffer.close()
    return bytes(data)


def test_report_display_value_translates_internal_states_without_changing_results():
    assert _report_display_value("not_labeled") == "未标记"
    assert _report_display_value("completed") == "分析完成"
    assert _report_display_value("failed") == "分析失败"
    assert _report_display_value("pending") == "等待分析"
    assert _report_display_value("OK") == "OK"
    assert _report_display_value("NG") == "NG"


def test_section_title_keeps_table_header_and_first_row_on_same_page(tmp_path):
    app = QApplication.instance() or QApplication([])
    writer = _configure_pdf_writer(str(tmp_path / "keep-section.pdf"))
    painter = QPainter(writer)
    try:
        layout = _ReportPainter(writer, painter)
        headers = ["项目", "内容"]
        rows = [
            ["结果", "OK"],
            ["录音时间", "2026-08-07 11:00:00"],
        ]
        width_ratios = [0.22, 0.78]
        column_widths = [
            layout.content_width * ratio
            for ratio in width_ratios
        ]
        table_start_height = layout._table_row_height(
            headers,
            column_widths,
            is_header=True,
        ) + layout._table_row_height(
            rows[0],
            column_widths,
            is_header=False,
        )
        title_height = layout._text_height(
            "工况详情 - 7000 rpm",
            size=11,
            bold=True,
            min_height=30,
        )
        layout.y = layout.content_bottom - title_height - 5

        layout.section_title(
            "工况详情 - 7000 rpm",
            keep_with_next_height=table_start_height,
        )
        page_after_title = layout.page_number
        layout.table(headers, rows, width_ratios=width_ratios)

        assert page_after_title == 2
        assert layout.page_number == page_after_title
        layout.finish()
    finally:
        painter.end()
        app.processEvents()


def test_local_mac_address_text_uses_fixed_width_hex(monkeypatch):
    monkeypatch.setattr(
        "base.product_test_pdf_report.uuid.getnode",
        lambda: 0x001122AABBCC,
    )

    assert _local_mac_address_text() == "001122AABBCC"


def test_resolve_product_pdf_output_path_uses_product_identity(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "base.product_test_pdf_report._local_mac_address_text",
        lambda: "001122AABBCC",
    )
    output_path = resolve_product_pdf_output_path(
        {"enabled": True, "save_dir": str(tmp_path)},
        make_report_data(),
        now_dt=datetime(2026, 8, 6, 11, 0, 0),
    )

    assert output_path == os.path.join(
        str(tmp_path),
        "S004-1_SN123456_20260806-103020_001122AABBCC.pdf",
    )


def test_product_report_signature_changes_when_condition_result_changes():
    report_data = make_report_data()
    first_signature = product_report_signature(report_data)

    report_data["conditions"][1]["result"] = "OK"

    assert product_report_signature(report_data) != first_signature


def test_product_report_signature_changes_when_analysis_image_changes():
    report_data = make_report_data()
    report_data["conditions"][0]["analysis_items"] = [
        {
            "name": "声压级",
            "status": "未启用判定",
            "deviation": "-",
            "images": [{"caption": "声压级曲线", "png_data": make_png_bytes("orange")}],
        }
    ]
    first_signature = product_report_signature(report_data)

    report_data["conditions"][0]["analysis_items"][0]["images"][0][
        "png_data"
    ] = make_png_bytes("blue")

    assert product_report_signature(report_data) != first_signature


def test_product_report_signature_changes_when_measurement_changes():
    report_data = make_report_data()
    report_data["conditions"][0]["analysis_items"] = [
        {
            "name": "声压级",
            "measurement": "72.3",
            "lower_limit": "60",
            "upper_limit": "80",
            "unit": "dB(A)",
            "status": "OK",
            "deviation": "0",
            "images": [],
        }
    ]
    first_signature = product_report_signature(report_data)

    report_data["conditions"][0]["analysis_items"][0]["measurement"] = "73.1"

    assert product_report_signature(report_data) != first_signature


def test_export_product_test_pdf_generates_pdf(tmp_path):
    QApplication.instance() or QApplication([])
    result = export_product_test_pdf(
        {"enabled": True, "save_dir": str(tmp_path)},
        make_report_data(),
        now_dt=datetime(2026, 8, 6, 11, 0, 0),
    )

    assert result.ok, result.message
    assert result.file_path is not None
    assert os.path.isfile(result.file_path)
    assert os.path.getsize(result.file_path) > 1000
    with open(result.file_path, "rb") as stream:
        assert stream.read(4) == b"%PDF"


def test_export_product_test_pdf_renders_in_background_thread(tmp_path):
    app = QApplication.instance() or QApplication([])
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            export_product_test_pdf,
            {"enabled": True, "save_dir": str(tmp_path)},
            make_report_data(),
            now_dt=datetime(2026, 8, 6, 11, 0, 0),
        )
        result = future.result(timeout=10)

    assert result.ok, result.message
    assert result.file_path is not None
    assert os.path.isfile(result.file_path)
    app.processEvents()


def test_export_product_test_pdf_embeds_analysis_image(tmp_path):
    QApplication.instance() or QApplication([])
    report_data = make_report_data()
    report_data["conditions"][0]["analysis_items"] = [
        {
            "name": "声压级",
            "status": "未启用判定",
            "deviation": "-",
            "images": [{"caption": "声压级曲线", "png_data": make_png_bytes()}],
        }
    ]

    result = export_product_test_pdf(
        {"enabled": True, "save_dir": str(tmp_path)},
        report_data,
        now_dt=datetime(2026, 8, 6, 11, 0, 0),
    )

    assert result.ok, result.message
    assert result.file_path is not None
    assert os.path.getsize(result.file_path) > 2000


def test_export_product_test_pdf_skips_when_disabled(tmp_path):
    result = export_product_test_pdf(
        {"enabled": False, "save_dir": str(tmp_path)},
        make_report_data(),
    )

    assert result.ok
    assert result.file_path is None
    assert list(tmp_path.iterdir()) == []
