"""Measure complete report pages with the same Qt layout used for PDF drawing."""

from html import escape, unescape
import re

from PyQt5 import sip
from PyQt5.QtCore import QMarginsF, QSizeF
from PyQt5.QtGui import QFont, QPageLayout, QPageSize, QTextDocument


PDF_RESOLUTION = 96
PDF_FOOTER_HEIGHT = 14.0


def report_page_layout():
    return QPageLayout(
        QPageSize(QPageSize.A4), QPageLayout.Portrait,
        QMarginsF(12.0, 12.0, 12.0, 12.0), QPageLayout.Millimeter,
    )


class MeasuredReportLayout:
    def __init__(self, stylesheet, font_family, cancel_requested=None):
        self.stylesheet = stylesheet
        self.font_family = font_family
        self.cancel_requested = cancel_requested
        rect = report_page_layout().paintRectPixels(PDF_RESOLUTION)
        self.content_size = QSizeF(rect.width(), rect.height() - PDF_FOOTER_HEIGHT)
        self.pages = []
        self.current = ""
        self._document = None

    def check_cancelled(self):
        if callable(self.cancel_requested) and self.cancel_requested():
            raise InterruptedError("PDF 报告导出已取消")

    def document(self, body):
        self.check_cancelled()
        if self._document is None:
            document = QTextDocument()
            self._document = document
            # Pixel fonts keep measurement at PDF DPI without exposing Qt's
            # internally owned document-layout object to Python's lifetime tracking.
            font = QFont(self.font_family)
            font.setPixelSize(round(9 * PDF_RESOLUTION / 72))
            document.setDefaultFont(font)
            document.setDocumentMargin(4)
            stylesheet = re.sub(
                r"(font-size\s*:\s*)([\d.]+)pt\b",
                lambda match: f"{match[1]}{round(float(match[2]) * PDF_RESOLUTION / 72)}px",
                self.stylesheet,
            )
            document.setDefaultStyleSheet(stylesheet)
        self._document.setHtml(_fix_table_columns(body))
        self._document.setTextWidth(self.content_size.width())
        return self._document

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        self.close()

    def close(self):
        # Reuse one document per phase, then destroy it in its export thread.
        if self._document is not None:
            sip.delete(self._document)
            self._document = None

    def fits(self, body):
        document = self.document(body)
        return document.size().height() <= self.content_size.height()

    def new_page(self):
        if self.current:
            self.pages.append(self.current)
            self.current = ""

    def append_block(self, body, *, start_new_page=False):
        if start_new_page:
            self.new_page()
        if not self.fits(self.current + body):
            self.new_page()
        if not self.fits(self.current + body):
            raise ValueError("报告单个内容块超过一页，请缩短标题或说明内容后重新导出")
        self.current += body

    def append_table(self, title, headers, rows, *, lead="", table_class="data-table"):
        """Choose the longest prefix that fits; never split an individual row."""
        remaining = list(rows)
        continuation = 1
        while remaining:
            self.check_cancelled()
            suffix = f" - 续页 {continuation}" if continuation > 1 else ""
            prefix = (
                lead + "<div class='section'>"
                f"<h3>{escape(title + suffix)}</h3>"
                f"<table class='{table_class}' width='100%'><thead><tr>"
                + headers + "</tr></thead><tbody>"
            )
            closing = "</tbody></table></div>"
            accepted = []
            for row in remaining:
                candidate = self.current + prefix + "".join(accepted) + row + closing
                if not self.fits(candidate):
                    break
                accepted.append(row)
            if not accepted:
                if self.current:
                    self.new_page()
                    continue
                raise ValueError(
                    f"报告“{title}”的表头和单行内容超过一页，请缩短过长字段后重新导出"
                )
            self.current += prefix + "".join(accepted) + closing
            remaining = remaining[len(accepted):]
            if remaining:
                self.new_page()
                continuation += 1
                lead = ""

    def finish(self):
        self.new_page()

    def to_html(self):
        body = "<div class='page-break'></div>".join(self.pages)
        return (
            '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
            f"<style>{self.stylesheet}\n.page-break {{ page-break-before: always; }}"
            f"</style></head><body>{body}</body></html>"
        )


def _fix_table_columns(body):
    """Keep each table's column widths constant across all its continuation pages."""
    def replace_header(match):
        header = match.group(0)
        cells = re.findall(r"<th>(.*?)</th>", header, flags=re.S)
        weights = []
        for cell in cells:
            text = unescape(cell)
            if text.startswith("CH") or text == "说明":
                weights.append(1.5 if text.startswith("CH") else 8.0)
            elif text in {"样本", "序号"}:
                weights.append(0.8)
            elif text == "型号":
                weights.append(1.4)
            elif text == "轮次":
                weights.append(1.1)
            elif text == "总体判定":
                weights.append(1.6)
            elif text == "判定标准":
                weights.append(2.0)
            else:
                weights.append(1.3)
        total = sum(weights)
        for weight in weights:
            header = header.replace("<th>", f"<th width='{100 * weight / total:.4f}%'>", 1)
        return header

    return re.sub(r"<thead>.*?</thead>", replace_header, body, flags=re.S)
