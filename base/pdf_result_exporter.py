from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import numpy as np

from consts.pdf_result_exporter_consts import (
    INVALID_FILENAME_CHARS_RE,
    MAX_RENDERED_VALUE_LENGTH,
    RESULT_FIELD_LABELS,
    RESULT_VALUE_LABELS,
)

_QT_APP = None


@dataclass(frozen=True)
class PdfExportResult:
    ok: bool
    message: str
    file_path: str | None = None


def _sanitize_filename_base(name: Any, *, fallback: str = "analysis_result") -> str:
    text = "" if name is None else str(name).strip()
    text = INVALID_FILENAME_CHARS_RE.sub("_", text)
    text = text.strip(". ")
    return text or fallback


def resolve_pdf_output_path(
    pdf_cfg: dict[str, Any],
    *,
    audio_path: str | None = None,
    now_dt: datetime | None = None,
) -> str:
    save_dir = pdf_cfg.get("save_dir") if isinstance(pdf_cfg, dict) else None
    if not save_dir:
        raise ValueError("保存目录不能为空")

    now_dt = now_dt or datetime.now()
    if audio_path:
        audio_base = os.path.splitext(os.path.basename(str(audio_path)))[0]
    else:
        audio_base = "analysis_result"
    filename_base = _sanitize_filename_base(audio_base)
    filename = f"{filename_base}_{now_dt.strftime('%H%M%S')}.pdf"

    os.makedirs(str(save_dir), exist_ok=True)
    return os.path.join(str(save_dir), filename)


def export_plot_widget_image(plot_widget_or_item: Any, output_dir: str, filename_base: str) -> str:
    import pyqtgraph as pg
    import pyqtgraph.exporters

    os.makedirs(str(output_dir), exist_ok=True)
    safe_filename = _sanitize_filename_base(filename_base, fallback="plot") + ".png"
    image_path = os.path.abspath(os.path.join(str(output_dir), safe_filename))

    if isinstance(plot_widget_or_item, pg.PlotWidget):
        export_item = plot_widget_or_item.getPlotItem()
    elif isinstance(plot_widget_or_item, pg.PlotItem):
        export_item = plot_widget_or_item
    elif hasattr(plot_widget_or_item, "getPlotItem"):
        export_item = plot_widget_or_item.getPlotItem()
    else:
        export_item = plot_widget_or_item

    exporter = pyqtgraph.exporters.ImageExporter(export_item)
    params = exporter.parameters()
    if params is not None and "width" in params:
        params["width"] = 1200
    exporter.export(image_path)
    return image_path


def _is_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray, dict)):
        return False
    return hasattr(value, "__iter__")


def _sequence_values(value: Any) -> list[Any]:
    try:
        if hasattr(value, "tolist"):
            value = value.tolist()
        return list(value)
    except Exception:
        return []


def _format_scalar(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return RESULT_VALUE_LABELS.get(value, value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _cap_rendered_value(text: str) -> str:
    if len(text) <= MAX_RENDERED_VALUE_LENGTH:
        return text
    return text[: MAX_RENDERED_VALUE_LENGTH - 3] + "..."


def _sequence_shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return tuple(int(part) for part in shape)
        except Exception:
            pass

    values = _sequence_values(value)
    if not values:
        return (0,)
    child_shapes = [_sequence_shape(item) for item in values if _is_sequence(item)]
    if len(child_shapes) == len(values) and child_shapes:
        first_shape = child_shapes[0]
        if all(child_shape == first_shape for child_shape in child_shapes):
            return (len(values),) + first_shape
    return (len(values),)


def _flatten_numeric_sequence(value: Any) -> tuple[list[float], int]:
    if _is_sequence(value):
        numeric_values: list[float] = []
        leaf_count = 0
        for item in _sequence_values(value):
            child_numeric_values, child_leaf_count = _flatten_numeric_sequence(item)
            numeric_values.extend(child_numeric_values)
            leaf_count += child_leaf_count
        return numeric_values, leaf_count
    try:
        return [float(value)], 1
    except Exception:
        return [], 1


def _format_sequence_shape(shape: tuple[int, ...]) -> str:
    return "x".join(str(part) for part in shape)


def _summarize_sequence(value: Any) -> str:
    values = _sequence_values(value)
    count = len(values)
    if count == 0:
        return "0 项"

    numeric_values, leaf_count = _flatten_numeric_sequence(value)
    if len(numeric_values) == leaf_count:
        if count <= 8 and leaf_count <= 8:
            return _cap_rendered_value(", ".join(_format_scalar(v) for v in values))
        shape = _sequence_shape(value)
        shape_text = f", shape={_format_sequence_shape(shape)}" if len(shape) > 1 else ""
        return _cap_rendered_value(
            f"{leaf_count} 点{shape_text}, min={min(numeric_values):.6g}, max={max(numeric_values):.6g}"
        )
    if count <= 8:
        return _cap_rendered_value(", ".join(_format_scalar(v) for v in values))
    return _cap_rendered_value(f"{count} 项, first={_format_scalar(values[0])}, last={_format_scalar(values[-1])}")


def _result_field_label(key: Any) -> str:
    key_text = str(key)
    return RESULT_FIELD_LABELS.get(key_text, key_text)


def _is_numeric_payload(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, np.ndarray):
        return np.issubdtype(value.dtype, np.number) and not np.issubdtype(value.dtype, np.bool_)
    if isinstance(value, (int, float, np.number)):
        return True
    if _is_sequence(value):
        values = _sequence_values(value)
        return all(_is_numeric_payload(item) for item in values)
    return False


def _result_values_equivalent(base_value: Any, raw_value: Any) -> bool:
    if _is_numeric_payload(base_value) and _is_numeric_payload(raw_value):
        try:
            base_array = np.asarray(base_value, dtype=float)
            raw_array = np.asarray(raw_value, dtype=float)
            if base_array.shape != raw_array.shape:
                return False
            return bool(np.allclose(base_array, raw_array, rtol=1e-6, atol=1e-9, equal_nan=True))
        except Exception:
            return False

    try:
        return bool(base_value == raw_value)
    except Exception:
        return False


def summarize_result_payload(result: Any, *, exclude_keys: set[str] | None = None) -> list[tuple[str, str]]:
    if result is None:
        return []
    excluded = {str(key) for key in (exclude_keys or set())}
    if isinstance(result, dict):
        rows: list[tuple[str, str]] = []
        for key, value in result.items():
            key_text = str(key)
            if key_text in excluded:
                continue
            if key_text.endswith("_raw"):
                base_key = key_text[: -len("_raw")]
                if base_key in result and _result_values_equivalent(result[base_key], value):
                    continue
            display_key = _result_field_label(key_text)
            if isinstance(value, dict):
                scalar_parts = [
                    f"{k}={_cap_rendered_value(_format_scalar(v))}"
                    for k, v in value.items()
                    if not isinstance(v, dict) and not _is_sequence(v)
                ]
                rows.append((display_key, _cap_rendered_value(", ".join(scalar_parts)) if scalar_parts else f"{len(value)} 项"))
            elif _is_sequence(value):
                rows.append((display_key, _summarize_sequence(value)))
            else:
                rows.append((display_key, _cap_rendered_value(_format_scalar(value))))
        return rows
    if _is_sequence(result):
        return [("结果", _summarize_sequence(result))]
    return [("结果", _cap_rendered_value(_format_scalar(result)))]


def format_ok_ng(result_tuple: Any) -> tuple[str, str]:
    if not (isinstance(result_tuple, tuple) and len(result_tuple) >= 1):
        return "-", ""
    ok_value = result_tuple[0]
    if ok_value is None:
        status = "-"
    else:
        status = "OK" if bool(ok_value) else "NG"
    deviation = ""
    if len(result_tuple) >= 2 and result_tuple[1] is not None:
        deviation = _format_scalar(result_tuple[1])
    return status, deviation


def build_pdf_report_items(
    save_items: list[str],
    analysis_items_data: dict[str, dict[str, Any]],
    analysis_config: dict[str, Any],
    analysis_result_dict: dict[str, tuple[bool, float]],
) -> list[dict[str, Any]]:
    report_items: list[dict[str, Any]] = []
    for item_name in save_items or []:
        item_data = analysis_items_data.get(item_name)
        if not isinstance(item_data, dict):
            item_data = {}
        cfg = analysis_config.get(item_name) if isinstance(analysis_config, dict) else None
        item_type = item_data.get("type")
        if item_type is None and isinstance(cfg, dict):
            item_type = cfg.get("type")
        status, deviation = format_ok_ng((analysis_result_dict or {}).get(item_name))
        summary_exclude_fields = set(item_data.get("pdf_summary_exclude_fields") or [])
        result_rows = summarize_result_payload(item_data.get("result"), exclude_keys=summary_exclude_fields)
        detail_rows = [
            row
            for key, value in item_data.items()
            if key not in {"type", "result", "tables", "pdf_summary_exclude_fields"}
            for row in summarize_result_payload({key: value})
        ]
        report_items.append(
            {
                "name": item_name,
                "type": item_type or "",
                "status": status,
                "deviation": deviation,
                "result_rows": result_rows + detail_rows,
                "tables": item_data.get("tables") or [],
                "has_judgment": item_name in (analysis_result_dict or {}),
            }
        )

    judged = [item for item in report_items if item["has_judgment"]]
    unjudged = [item for item in report_items if not item["has_judgment"]]
    return judged + unjudged


def calculate_overall_status(
    save_items: list[str],
    analysis_result_dict: dict[str, tuple[bool, float]],
) -> str:
    judged_statuses: list[bool] = []
    for item_name in save_items or []:
        result_tuple = (analysis_result_dict or {}).get(item_name)
        if isinstance(result_tuple, tuple) and len(result_tuple) >= 1 and result_tuple[0] is not None:
            judged_statuses.append(bool(result_tuple[0]))
    if not judged_statuses:
        return "-"
    return "OK" if all(judged_statuses) else "NG"


def build_pdf_header_rows(
    *,
    audio_name: str | None,
    audio_path: str | None,
    sn: str | None,
    product_model: str | None,
    date_text: str | None,
    overall_status: str,
) -> list[tuple[str, str]]:
    rows = [
        ("音频文件", _format_scalar(audio_name)),
        ("音频路径", _format_scalar(audio_path)),
        ("S/N", _format_scalar(sn)),
        ("产品型号", _format_scalar(product_model)),
        ("导出时间", _format_scalar(date_text)),
        ("总体结果", _format_scalar(overall_status)),
    ]
    return [(key, value) for key, value in rows if key == "总体结果" or value != "-"]


class _PdfPainter:
    def __init__(self, writer: Any):
        from PyQt5.QtCore import QRectF, Qt
        from PyQt5.QtGui import QColor, QFont, QPen

        self.writer = writer
        self.painter = None
        self.QRectF = QRectF
        self.Qt = Qt
        self.QColor = QColor
        self.QFont = QFont
        self.QPen = QPen
        self.margin = 48
        self.page_width = int(writer.width())
        self.page_height = int(writer.height())
        self.content_width = self.page_width - (self.margin * 2)
        self.y = self.margin

    def bind(self, painter: Any) -> None:
        self.painter = painter

    def _ensure_space(self, height: int) -> None:
        if self.y + height > self.page_height - self.margin:
            self.writer.newPage()
            self.y = self.margin

    def new_page(self) -> None:
        self.writer.newPage()
        self.y = self.margin

    def text(self, text: str, *, size: int = 10, bold: bool = False, color: Any = None, height: int = 22) -> None:
        painter = self.painter
        if painter is None:
            return
        self._ensure_space(height)
        font = self.QFont("Microsoft YaHei", size)
        font.setBold(bold)
        painter.setFont(font)
        painter.setPen(self.QPen(color or self.QColor(30, 30, 30)))
        rect = self.QRectF(self.margin, self.y, self.content_width, height)
        painter.drawText(rect, int(self.Qt.AlignLeft | self.Qt.AlignVCenter), str(text))
        self.y += height

    def wrapped_text(self, text: str, *, size: int = 9, height: int = 44) -> None:
        painter = self.painter
        if painter is None:
            return
        self._ensure_space(height)
        font = self.QFont("Microsoft YaHei", size)
        painter.setFont(font)
        painter.setPen(self.QPen(self.QColor(50, 50, 50)))
        rect = self.QRectF(self.margin, self.y, self.content_width, height)
        painter.drawText(rect, int(self.Qt.AlignLeft | self.Qt.AlignTop | self.Qt.TextWordWrap), str(text))
        self.y += height

    def spacer(self, height: int) -> None:
        self._ensure_space(height)
        self.y += height

    def section_title(self, title: str) -> None:
        painter = self.painter
        if painter is None:
            return
        height = 28
        self._ensure_space(height + 4)
        painter.fillRect(self.QRectF(self.margin, self.y, self.content_width, height), self.QColor(232, 239, 247))
        self.text(title, size=11, bold=True, height=height)
        self.spacer(4)

    def rows(self, rows: list[tuple[str, str]], *, key_width: int = 120) -> None:
        painter = self.painter
        if painter is None:
            return
        row_height = 24
        for key, value in rows:
            self._ensure_space(row_height)
            y = self.y
            painter.setPen(self.QPen(self.QColor(190, 190, 190)))
            painter.drawRect(self.QRectF(self.margin, y, self.content_width, row_height))
            painter.drawLine(
                self.margin + key_width,
                y,
                self.margin + key_width,
                y + row_height,
            )
            font = self.QFont("Microsoft YaHei", 9)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(self.QPen(self.QColor(45, 45, 45)))
            painter.drawText(
                self.QRectF(self.margin + 6, y, key_width - 12, row_height),
                int(self.Qt.AlignLeft | self.Qt.AlignVCenter),
                str(key),
            )
            font.setBold(False)
            painter.setFont(font)
            painter.drawText(
                self.QRectF(self.margin + key_width + 6, y, self.content_width - key_width - 12, row_height),
                int(self.Qt.AlignLeft | self.Qt.AlignVCenter),
                str(value),
            )
            self.y += row_height

    def summary_table(self, items: list[dict[str, Any]]) -> None:
        rows = [("分析项", "类型", "判定")]
        for item in items:
            rows.append((item.get("name", ""), item.get("type", ""), item.get("status", "-")))
        self._draw_three_col_table(rows)

    def _draw_three_col_table(self, rows: list[tuple[str, str, str]]) -> None:
        painter = self.painter
        if painter is None:
            return
        widths = [260, 120, self.content_width - 380]
        row_height = 24
        for row_index, row in enumerate(rows):
            self._ensure_space(row_height)
            x = self.margin
            y = self.y
            if row_index == 0:
                painter.fillRect(self.QRectF(x, y, self.content_width, row_height), self.QColor(245, 245, 245))
            painter.setPen(self.QPen(self.QColor(190, 190, 190)))
            painter.drawRect(self.QRectF(x, y, self.content_width, row_height))
            font = self.QFont("Microsoft YaHei", 9)
            font.setBold(row_index == 0)
            painter.setFont(font)
            for col_index, value in enumerate(row):
                width = widths[col_index]
                if col_index:
                    painter.drawLine(x, y, x, y + row_height)
                painter.setPen(self.QPen(self._status_color(value) if col_index == 2 else self.QColor(45, 45, 45)))
                painter.drawText(
                    self.QRectF(x + 5, y, width - 10, row_height),
                    int(self.Qt.AlignLeft | self.Qt.AlignVCenter),
                    str(value),
                )
                x += width
            self.y += row_height

    def _status_color(self, value: Any) -> Any:
        if value == "OK":
            return self.QColor(0, 128, 70)
        if value == "NG":
            return self.QColor(190, 40, 40)
        return self.QColor(45, 45, 45)

    def table(self, headers: list[Any], rows: list[list[Any]], title: str | None = None) -> None:
        painter = self.painter
        if painter is None:
            return
        headers = [str(item) for item in (headers or [])]
        if not headers:
            return
        normalized_rows = []
        for row in rows or []:
            row_values = [str(value) for value in (row or [])]
            if len(row_values) < len(headers):
                row_values.extend([""] * (len(headers) - len(row_values)))
            normalized_rows.append(row_values[: len(headers)])

        if title:
            self.text(str(title), size=9, bold=True, height=20)

        col_count = max(1, len(headers))
        col_width = self.content_width / col_count
        row_height = 24

        def draw_row(values: list[str], *, header: bool = False) -> None:
            self._ensure_space(row_height)
            y = self.y
            x = self.margin
            if header:
                painter.fillRect(self.QRectF(self.margin, y, self.content_width, row_height), self.QColor(245, 245, 245))
            painter.setPen(self.QPen(self.QColor(190, 190, 190)))
            painter.drawRect(self.QRectF(self.margin, y, self.content_width, row_height))
            font = self.QFont("Microsoft YaHei", 8)
            font.setBold(header)
            painter.setFont(font)
            painter.setPen(self.QPen(self.QColor(45, 45, 45)))
            for col_index, value in enumerate(values):
                if col_index:
                    line_x = int(round(x))
                    painter.drawLine(line_x, y, line_x, y + row_height)
                painter.drawText(
                    self.QRectF(x + 4, y, col_width - 8, row_height),
                    int(self.Qt.AlignLeft | self.Qt.AlignVCenter),
                    str(value),
                )
                x += col_width
            self.y += row_height

        draw_row(headers, header=True)
        for row in normalized_rows:
            draw_row(row)
        self.spacer(6)

    def image(self, image_path: str, title: str | None = None) -> None:
        from PyQt5.QtGui import QImage

        painter = self.painter
        if painter is None:
            return
        image = QImage(str(image_path))
        if image.isNull():
            return
        if title:
            self.text(title, size=9, bold=True, height=20)
        max_width = self.content_width
        scaled_height = max(1, int(image.height() * (max_width / max(1, image.width()))))
        draw_height = min(scaled_height, self.page_height - (self.margin * 2))
        self._ensure_space(draw_height + 8)
        target = self.QRectF(self.margin, self.y, max_width, draw_height)
        painter.drawImage(target, image)
        self.y += draw_height + 8


def _configure_pdf_writer(file_path: str) -> Any:
    from PyQt5.QtGui import QPageSize, QPagedPaintDevice, QPdfWriter

    writer = QPdfWriter(file_path)
    writer.setResolution(96)
    try:
        writer.setPageSize(QPageSize(QPageSize.A4))
    except Exception:
        writer.setPageSize(QPagedPaintDevice.A4)
    return writer


def _ensure_qt_application() -> None:
    global _QT_APP
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        _QT_APP = QApplication([])


def _render_pdf(
    file_path: str,
    *,
    header_rows: list[tuple[str, str]],
    report_items: list[dict[str, Any]],
) -> None:
    from PyQt5.QtGui import QPainter

    _ensure_qt_application()
    writer = _configure_pdf_writer(file_path)
    layout = _PdfPainter(writer)
    painter = QPainter(writer)
    if not painter.isActive():
        raise RuntimeError("PDF painter 初始化失败")
    try:
        layout.bind(painter)
        layout.text("分析结果报告", size=18, bold=True, height=38)
        layout.rows(header_rows)
        layout.spacer(12)
        layout.text("结果汇总", size=12, bold=True, height=26)
        layout.summary_table(report_items)
        if report_items:
            layout.new_page()
        else:
            layout.spacer(14)

        for item in report_items:
            section_name = f"{item.get('name', '')}  {item.get('type', '')}".strip()
            layout.section_title(section_name)
            status = item.get("status", "-")
            status_line = f"判定: {status}"
            layout.text(status_line, size=10, bold=True, color=layout._status_color(status), height=22)
            rows = item.get("result_rows") or []
            if rows:
                layout.rows([(str(k), str(v)) for k, v in rows])
            else:
                layout.text("无文本结果", size=9, height=20)
            for table_info in item.get("tables") or []:
                if isinstance(table_info, dict):
                    layout.table(
                        list(table_info.get("headers") or []),
                        list(table_info.get("rows") or []),
                        table_info.get("title"),
                    )
            for image_info in item.get("images") or []:
                if isinstance(image_info, dict):
                    layout.image(str(image_info.get("path") or ""), image_info.get("title"))
            layout.spacer(10)
    finally:
        painter.end()


def _collect_item_images(
    report_items: list[dict[str, Any]],
    image_exporters: dict[str, Callable[[str], list[dict[str, Any]]]] | None,
    temp_dir: str,
) -> None:
    exporters = image_exporters or {}
    for index, item in enumerate(report_items):
        item_name = item.get("name")
        exporter = exporters.get(item_name)
        item["images"] = []
        if exporter is None:
            continue
        item_dir_name = f"{index:03d}_{_sanitize_filename_base(item_name, fallback='item')}"
        item_temp_dir = os.path.join(temp_dir, item_dir_name)
        os.makedirs(item_temp_dir, exist_ok=True)
        try:
            exported_images = exporter(item_temp_dir) or []
        except Exception as exc:
            item["result_rows"].append(("图像导出", f"失败: {exc}"))
            continue
        for image_info in exported_images:
            if not isinstance(image_info, dict):
                continue
            image_path = image_info.get("path")
            if image_path and os.path.exists(str(image_path)):
                item["images"].append({"title": image_info.get("title") or "", "path": str(image_path)})


def export_analysis_to_pdf(
    pdf_cfg: dict[str, Any],
    *,
    audio_path: str | None,
    sn: str | None,
    product_model: str | None,
    date_text: str | None,
    analysis_items_data: dict[str, dict[str, Any]],
    analysis_config: dict[str, Any],
    analysis_result_dict: dict[str, tuple[bool, float]],
    image_exporters: dict[str, Callable[[str], list[dict[str, Any]]]] | None,
    now_dt: datetime | None = None,
) -> PdfExportResult:
    if not isinstance(pdf_cfg, dict) or not pdf_cfg.get("enabled", True):
        return PdfExportResult(ok=True, message="PDF导出未启用")

    save_items = pdf_cfg.get("save_items") or []
    if not isinstance(save_items, list) or len(save_items) == 0:
        return PdfExportResult(ok=False, message="未选择需要保存的分析项")

    try:
        file_path = resolve_pdf_output_path(pdf_cfg, audio_path=audio_path, now_dt=now_dt)
    except Exception as exc:
        return PdfExportResult(ok=False, message=f"保存目录不可达或无权限: {exc}")

    try:
        report_items = build_pdf_report_items(save_items, analysis_items_data, analysis_config, analysis_result_dict)
        overall_status = calculate_overall_status(save_items, analysis_result_dict)
        audio_name = os.path.basename(str(audio_path)) if audio_path else "analysis_result"
        export_time = date_text or (now_dt or datetime.now()).strftime("%Y/%m/%d %H:%M:%S")
        header_rows = build_pdf_header_rows(
            audio_name=audio_name,
            audio_path=audio_path,
            sn=sn,
            product_model=product_model,
            date_text=export_time,
            overall_status=overall_status,
        )
        with tempfile.TemporaryDirectory(prefix="pdf_result_export_") as temp_dir:
            _collect_item_images(report_items, image_exporters, temp_dir)
            _render_pdf(file_path, header_rows=header_rows, report_items=report_items)
        return PdfExportResult(ok=True, message=f"PDF导出成功: {file_path}", file_path=file_path)
    except PermissionError:
        return PdfExportResult(ok=False, message=f"PDF文件被占用，请关闭后重试: {file_path}", file_path=file_path)
    except Exception as exc:
        return PdfExportResult(ok=False, message=f"保存PDF失败: {exc}", file_path=file_path)
