from __future__ import annotations

import hashlib
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from consts.running_consts import PRODUCT_TEST_REPORT_DIR


_INVALID_FILENAME_CHARS_RE = re.compile(r'[<>:"/\\|?*]')
_QT_APP = None
_REPORT_DISPLAY_VALUES = {
    "not_labeled": "未标记",
    "not labeled": "未标记",
    "completed": "分析完成",
    "failed": "分析失败",
    "pending": "等待分析",
}


@dataclass(frozen=True)
class ProductPdfExportResult:
    ok: bool
    message: str
    file_path: str | None = None


def _report_display_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "-"
    return _REPORT_DISPLAY_VALUES.get(text.lower(), text)


def _sanitize_filename_part(value: Any, fallback: str) -> str:
    text = _INVALID_FILENAME_CHARS_RE.sub("_", str(value or "").strip())
    text = text.strip(". ")
    return text or fallback


def _parse_report_datetime(value: Any, now_dt: datetime) -> datetime:
    text = str(value or "").strip()
    if not text:
        return now_dt
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return now_dt


def _local_mac_address_text() -> str:
    return f"{uuid.getnode() & 0xFFFFFFFFFFFF:012X}"


def resolve_product_pdf_output_path(
    report_config: dict[str, Any],
    report_data: dict[str, Any],
    *,
    now_dt: datetime | None = None,
) -> str:
    now_dt = now_dt or datetime.now()
    save_dir = str(report_config.get("save_dir", "") or "").strip()
    effective_dir = os.path.abspath(os.path.normpath(save_dir or PRODUCT_TEST_REPORT_DIR))
    product_model = _sanitize_filename_part(
        report_data.get("product_model"),
        "unknown_model",
    )
    barcode = str(report_data.get("barcode") or "").strip()
    test_datetime = _parse_report_datetime(report_data.get("created_at"), now_dt)
    mac_address = _local_mac_address_text()
    file_name_parts = [product_model]
    if barcode:
        file_name_parts.append(
            _sanitize_filename_part(barcode, "unknown_product")
        )
    file_name_parts.extend(
        [
            test_datetime.strftime("%Y%m%d-%H%M%S"),
            mac_address,
        ]
    )
    file_name = "_".join(file_name_parts) + ".pdf"
    os.makedirs(effective_dir, exist_ok=True)
    return os.path.join(effective_dir, file_name)


def product_report_signature(report_data: dict[str, Any]) -> tuple[Any, ...]:
    condition_signatures = []
    for condition in report_data.get("conditions", []) or []:
        analysis_signatures = []
        for analysis in condition.get("analysis_results", []) or []:
            analysis_signatures.append(
                (
                    str(analysis.get("name", "")),
                    str(analysis.get("status", "")),
                    str(analysis.get("deviation", "")),
                )
            )
        analysis_item_signatures = []
        for analysis in condition.get("analysis_items", []) or []:
            image_signatures = []
            for image in analysis.get("images", []) or []:
                png_data = image.get("png_data", b"")
                if isinstance(png_data, (bytes, bytearray)):
                    image_digest = hashlib.sha256(bytes(png_data)).hexdigest()
                else:
                    image_digest = str(png_data or "")
                image_signatures.append(
                    (str(image.get("caption", "")), image_digest)
                )
            analysis_item_signatures.append(
                (
                    str(analysis.get("name", "")),
                    str(analysis.get("type", "")),
                    str(analysis.get("state", "")),
                    str(analysis.get("status", "")),
                    str(analysis.get("deviation", "")),
                    str(analysis.get("measurement", "")),
                    str(analysis.get("lower_limit", "")),
                    str(analysis.get("upper_limit", "")),
                    str(analysis.get("unit", "")),
                    str(analysis.get("error", "")),
                    tuple(str(value) for value in analysis.get("image_errors", []) or []),
                    tuple(image_signatures),
                )
            )
        condition_signatures.append(
            (
                str(condition.get("key", "")),
                str(condition.get("result", "")),
                str(condition.get("recorded_path", "")),
                str(condition.get("analysis_state", "")),
                tuple(analysis_signatures),
                tuple(analysis_item_signatures),
            )
        )
    return (
        str(report_data.get("group_id", "")),
        str(report_data.get("overall_result", "")),
        tuple(condition_signatures),
    )


def _ensure_qt_application() -> None:
    global _QT_APP
    from PyQt5.QtWidgets import QApplication

    if QApplication.instance() is None:
        _QT_APP = QApplication([])


class _ReportPainter:
    def __init__(self, writer: Any, painter: Any):
        from PyQt5.QtCore import QRectF, Qt
        from PyQt5.QtGui import QColor, QFont, QFontMetrics, QPen

        self.writer = writer
        self.painter = painter
        self.QRectF = QRectF
        self.Qt = Qt
        self.QColor = QColor
        self.QFont = QFont
        self.QFontMetrics = QFontMetrics
        self.QPen = QPen
        self.margin = 44
        self.footer_height = 24
        self.page_width = int(writer.width())
        self.page_height = int(writer.height())
        self.content_width = self.page_width - (self.margin * 2)
        self.content_bottom = self.page_height - self.margin - self.footer_height
        self.y = self.margin
        self.page_number = 1

    def _font(self, size: int, bold: bool = False):
        font = self.QFont("Microsoft YaHei", size)
        font.setBold(bool(bold))
        return font

    def _draw_footer(self) -> None:
        font = self._font(8)
        self.painter.setFont(font)
        self.painter.setPen(self.QPen(self.QColor(110, 110, 110)))
        rect = self.QRectF(
            self.margin,
            self.page_height - self.margin - 4,
            self.content_width,
            18,
        )
        self.painter.drawText(
            rect,
            int(self.Qt.AlignHCenter | self.Qt.AlignVCenter),
            f"第 {self.page_number} 页",
        )

    def new_page(self) -> None:
        self._draw_footer()
        self.writer.newPage()
        self.page_number += 1
        self.y = self.margin

    def finish(self) -> None:
        self._draw_footer()

    def ensure_space(self, height: float) -> bool:
        if self.y + height <= self.content_bottom:
            return False
        self.new_page()
        return True

    def spacer(self, height: float) -> None:
        self.ensure_space(height)
        self.y += height

    def _text_height(
        self,
        value: Any,
        *,
        size: int,
        bold: bool,
        min_height: int,
    ) -> int:
        font = self._font(size, bold)
        metrics = self.QFontMetrics(font)
        flags = int(self.Qt.TextWordWrap | self.Qt.TextWrapAnywhere)
        measured = metrics.boundingRect(
            0,
            0,
            int(self.content_width),
            10000,
            flags,
            str(value or ""),
        )
        return max(min_height, measured.height() + 6)

    def text(
        self,
        value: Any,
        *,
        size: int = 10,
        bold: bool = False,
        color: Any = None,
        min_height: int = 22,
        align: int | None = None,
    ) -> None:
        text = str(value or "")
        font = self._font(size, bold)
        flags = int(self.Qt.TextWordWrap | self.Qt.TextWrapAnywhere)
        height = self._text_height(
            text,
            size=size,
            bold=bold,
            min_height=min_height,
        )
        self.ensure_space(height)
        self.painter.setFont(font)
        self.painter.setPen(self.QPen(color or self.QColor(35, 35, 35)))
        rect = self.QRectF(self.margin, self.y, self.content_width, height)
        draw_flags = flags | int(align or (self.Qt.AlignLeft | self.Qt.AlignVCenter))
        self.painter.drawText(rect, draw_flags, text)
        self.y += height

    def section_title(
        self,
        title: str,
        *,
        keep_with_next_height: float = 0,
    ) -> None:
        height = self._text_height(
            title,
            size=11,
            bold=True,
            min_height=30,
        )
        self.ensure_space(height + 5 + max(0, keep_with_next_height))
        self.painter.fillRect(
            self.QRectF(self.margin, self.y, self.content_width, height),
            self.QColor(230, 238, 248),
        )
        self.text(title, size=11, bold=True, min_height=height)
        self.y += 5

    def status_color(self, value: Any):
        normalized = str(value or "").strip().upper()
        if normalized == "OK":
            return self.QColor(0, 125, 70)
        if normalized == "NG":
            return self.QColor(190, 45, 45)
        return self.QColor(60, 60, 60)

    def image(
        self,
        png_data: Any,
        *,
        heading: str = "",
        summary: str = "",
    ) -> bool:
        from PyQt5.QtGui import QImage

        if not isinstance(png_data, (bytes, bytearray)) or not png_data:
            return False
        image = QImage.fromData(bytes(png_data), "PNG")
        if image.isNull() or image.width() <= 0 or image.height() <= 0:
            return False

        heading_height = (
            self._text_height(
                heading,
                size=10,
                bold=True,
                min_height=24,
            )
            if heading
            else 0
        )
        summary_height = (
            self._text_height(
                summary,
                size=9,
                bold=False,
                min_height=22,
            )
            if summary
            else 0
        )
        text_spacing = 5 if heading or summary else 0
        page_image_height = max(
            1,
            self.content_bottom
            - self.margin
            - heading_height
            - summary_height
            - text_spacing
            - 10,
        )
        scale = min(
            self.content_width / float(image.width()),
            page_image_height / float(image.height()),
        )
        draw_width = image.width() * scale
        draw_height = image.height() * scale
        required_height = (
            heading_height
            + summary_height
            + text_spacing
            + draw_height
            + 10
        )
        if self.y + required_height > self.content_bottom:
            self.new_page()

        if heading:
            self.text(heading, size=10, bold=True, min_height=heading_height)
        if summary:
            self.text(
                summary,
                size=9,
                color=self.QColor(80, 80, 80),
                min_height=summary_height,
            )
        if text_spacing:
            self.spacer(text_spacing)

        x = self.margin + (self.content_width - draw_width) / 2.0
        self.painter.drawImage(
            self.QRectF(x, self.y, draw_width, draw_height),
            image,
        )
        self.y += draw_height
        self.spacer(10)
        return True

    def table(
        self,
        headers: list[Any],
        rows: list[list[Any]],
        *,
        width_ratios: list[float] | None = None,
        continuation_title: str | None = None,
    ) -> None:
        header_values = [str(value or "") for value in headers]
        if not header_values:
            return
        if not width_ratios or len(width_ratios) != len(header_values):
            width_ratios = [1.0] * len(header_values)
        ratio_total = sum(width_ratios) or 1.0
        column_widths = [
            self.content_width * (ratio / ratio_total)
            for ratio in width_ratios
        ]
        normalized_rows = []
        for row in rows or []:
            values = [str(value if value is not None else "") for value in (row or [])]
            if len(values) < len(header_values):
                values.extend([""] * (len(header_values) - len(values)))
            normalized_rows.append(values[: len(header_values)])

        header_height = self._table_row_height(
            header_values,
            column_widths,
            is_header=True,
        )
        first_row_height = (
            self._table_row_height(
                normalized_rows[0],
                column_widths,
                is_header=False,
            )
            if normalized_rows
            else 0
        )
        if self.y + header_height + first_row_height > self.content_bottom:
            self.new_page()
            if continuation_title:
                self.text(f"{continuation_title}（续）", size=10, bold=True)
        self._table_row(header_values, column_widths, is_header=True)
        for values in normalized_rows:
            row_height = self._table_row_height(values, column_widths, is_header=False)
            if self.y + row_height > self.content_bottom:
                self.new_page()
                if continuation_title:
                    self.text(f"{continuation_title}（续）", size=10, bold=True)
                self._table_row(header_values, column_widths, is_header=True)
            self._table_row(values, column_widths, is_header=False)
        self.y += 7

    def _table_row_height(
        self,
        values: list[str],
        column_widths: list[float],
        *,
        is_header: bool,
    ) -> int:
        font = self._font(9, is_header)
        metrics = self.QFontMetrics(font)
        flags = int(self.Qt.TextWordWrap | self.Qt.TextWrapAnywhere)
        heights = []
        for value, width in zip(values, column_widths):
            rect = metrics.boundingRect(
                0,
                0,
                max(1, int(width - 12)),
                10000,
                flags,
                value,
            )
            heights.append(rect.height() + 10)
        return max(28, max(heights, default=28))

    def _table_row(
        self,
        values: list[str],
        column_widths: list[float],
        *,
        is_header: bool,
    ) -> None:
        height = self._table_row_height(values, column_widths, is_header=is_header)
        self.ensure_space(height)
        x = float(self.margin)
        y = float(self.y)
        if is_header:
            self.painter.fillRect(
                self.QRectF(x, y, self.content_width, height),
                self.QColor(244, 247, 250),
            )
        self.painter.setPen(self.QPen(self.QColor(185, 190, 196)))
        self.painter.drawRect(self.QRectF(x, y, self.content_width, height))
        font = self._font(9, is_header)
        self.painter.setFont(font)
        flags = int(
            self.Qt.AlignLeft
            | self.Qt.AlignVCenter
            | self.Qt.TextWordWrap
            | self.Qt.TextWrapAnywhere
        )
        for index, (value, width) in enumerate(zip(values, column_widths)):
            if index:
                self.painter.drawLine(int(round(x)), int(y), int(round(x)), int(y + height))
            self.painter.setPen(
                self.QPen(
                    self.status_color(value)
                    if not is_header and str(value).strip().upper() in ("OK", "NG")
                    else self.QColor(45, 45, 45)
                )
            )
            self.painter.drawText(
                self.QRectF(x + 6, y + 2, width - 12, height - 4),
                flags,
                value,
            )
            x += width
        self.y += height


def _configure_pdf_writer(file_path: str):
    from PyQt5.QtGui import QPageSize, QPagedPaintDevice, QPdfWriter

    writer = QPdfWriter(file_path)
    writer.setResolution(96)
    writer.setTitle("产品测试报告")
    try:
        writer.setPageSize(QPageSize(QPageSize.A4))
    except Exception:
        writer.setPageSize(QPagedPaintDevice.A4)
    return writer


def _render_product_pdf(file_path: str, report_data: dict[str, Any]) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QPainter

    _ensure_qt_application()
    writer = _configure_pdf_writer(file_path)
    painter = QPainter(writer)
    if not painter.isActive():
        raise RuntimeError("PDF painter 初始化失败")

    layout = _ReportPainter(writer, painter)
    try:
        layout.text(
            "产品测试报告",
            size=18,
            bold=True,
            min_height=42,
            align=int(Qt.AlignHCenter | Qt.AlignVCenter),
        )
        layout.spacer(6)
        overall_result = _report_display_value(report_data.get("overall_result"))
        product_rows = [
            ["产品型号", report_data.get("product_model") or "-"],
            ["产品条码", report_data.get("barcode") or "-"],
            ["测试时间", report_data.get("test_time") or "-"],
            ["汇总结果", overall_result],
            ["报告生成时间", report_data.get("export_time") or "-"],
        ]
        layout.section_title("产品信息")
        layout.table(["项目", "内容"], product_rows, width_ratios=[0.22, 0.78])

        conditions = list(report_data.get("conditions", []) or [])
        summary_rows = [
            [
                condition.get("name") or "-",
                os.path.basename(str(condition.get("recorded_path") or "")) or "-",
                _report_display_value(condition.get("result")),
            ]
            for condition in conditions
        ]
        layout.section_title("工况汇总")
        layout.table(
            ["工况", "音频文件", "结果"],
            summary_rows,
            width_ratios=[0.22, 0.56, 0.22],
        )

        for condition in conditions:
            condition_title = f"工况详情 - {condition.get('name') or '-'}"
            condition_rows = [
                ["结果", _report_display_value(condition.get("result"))],
                ["录音时间", condition.get("record_time") or "-"],
                ["采样率", condition.get("sample_rate") or "-"],
                ["音频文件", condition.get("recorded_path") or "-"],
            ]
            condition_headers = ["项目", "内容"]
            condition_width_ratios = [0.22, 0.78]
            condition_column_widths = [
                layout.content_width * ratio
                for ratio in condition_width_ratios
            ]
            condition_table_start_height = layout._table_row_height(
                condition_headers,
                condition_column_widths,
                is_header=True,
            ) + layout._table_row_height(
                condition_rows[0],
                condition_column_widths,
                is_header=False,
            )
            layout.section_title(
                condition_title,
                keep_with_next_height=condition_table_start_height,
            )
            layout.table(
                condition_headers,
                condition_rows,
                width_ratios=condition_width_ratios,
                continuation_title=condition_title,
            )
            analysis_items = list(condition.get("analysis_items", []) or [])
            analysis_row_items = analysis_items or list(
                condition.get("analysis_results", []) or []
            )
            analysis_rows = [
                [
                    item.get("name") or "-",
                    item.get("measurement") or "-",
                    item.get("lower_limit") or "-",
                    item.get("upper_limit") or "-",
                    item.get("unit") or "-",
                    _report_display_value(item.get("status")),
                    item.get("deviation") or "-",
                ]
                for item in analysis_row_items
            ]
            if analysis_rows:
                layout.table(
                    [
                        "分析项目",
                        "测量值",
                        "下限",
                        "上限",
                        "单位",
                        "判定",
                        "偏差",
                    ],
                    analysis_rows,
                    width_ratios=[0.26, 0.13, 0.11, 0.11, 0.11, 0.13, 0.15],
                    continuation_title=condition_title,
                )
            else:
                layout.text("未配置分析项", size=9, min_height=24)

            for item in analysis_items:
                item_name = str(item.get("name") or "分析项")
                error = str(item.get("error") or "").strip()
                if error:
                    layout.text(
                        f"{item_name}：{error}",
                        size=9,
                        color=layout.QColor(190, 45, 45),
                    )

                images = list(item.get("images", []) or [])
                if images:
                    for index, image in enumerate(images):
                        caption = str(image.get("caption") or "").strip()
                        summary = caption if caption and caption != item_name else ""
                        heading = f"分析图 - {item_name}"
                        if index:
                            heading += "（续）"
                        if not layout.image(
                            image.get("png_data"),
                            heading=heading,
                            summary=summary,
                        ):
                            layout.text("分析图数据不可用", size=9)
                else:
                    layout.text(f"{item_name}：无可用分析图", size=9)

                for image_error in item.get("image_errors", []) or []:
                    layout.text(
                        f"分析图导出失败：{image_error}",
                        size=9,
                        color=layout.QColor(190, 45, 45),
                    )
            layout.spacer(8)
        layout.finish()
    finally:
        painter.end()


def export_product_test_pdf(
    report_config: dict[str, Any],
    report_data: dict[str, Any],
    *,
    now_dt: datetime | None = None,
) -> ProductPdfExportResult:
    if not isinstance(report_config, dict) or not report_config.get("enabled", False):
        return ProductPdfExportResult(ok=True, message="PDF报告未启用")
    if not isinstance(report_data, dict) or not report_data.get("conditions"):
        return ProductPdfExportResult(ok=False, message="没有可导出的产品工况结果")

    try:
        output_path = resolve_product_pdf_output_path(
            report_config,
            report_data,
            now_dt=now_dt,
        )
    except (OSError, ValueError) as error:
        return ProductPdfExportResult(
            ok=False,
            message=f"PDF报告保存目录不可用：{error}",
        )

    now_dt = now_dt or datetime.now()
    rendered_data = dict(report_data)
    rendered_data["export_time"] = now_dt.strftime("%Y-%m-%d %H:%M:%S")
    temp_path = output_path + ".tmp.pdf"
    try:
        _render_product_pdf(temp_path, rendered_data)
        os.replace(temp_path, output_path)
        return ProductPdfExportResult(
            ok=True,
            message=f"PDF报告已生成：{output_path}",
            file_path=output_path,
        )
    except PermissionError:
        return ProductPdfExportResult(
            ok=False,
            message=f"PDF报告文件被占用，请关闭后重试：{output_path}",
            file_path=output_path,
        )
    except Exception as error:
        return ProductPdfExportResult(
            ok=False,
            message=f"生成PDF报告失败：{error}",
            file_path=output_path,
        )
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
