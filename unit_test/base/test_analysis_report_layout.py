import re
from threading import get_ident

import pytest

from base.analysis_report import _report_stylesheet, _render_html_pdf, prepare_analysis_report_runtime
from base.analysis_report_layout import MeasuredReportLayout


def make_layout(cancel_requested=None):
    return MeasuredReportLayout(_report_stylesheet(), prepare_analysis_report_runtime(), cancel_requested)


def _run_export_thread(work, *, collect_garbage=False):
    import gc
    import time
    from PyQt5.QtCore import QThread
    from PyQt5.QtWidgets import QApplication

    errors = []

    class ExportThread(QThread):
        def run(self):
            try:
                work()
            except Exception as error:
                errors.append(error)

    thread = ExportThread()
    thread.start()
    while thread.isRunning():
        QApplication.instance().processEvents()
        if collect_garbage:
            gc.collect()
        time.sleep(.01)
    thread.wait()
    if errors:
        raise errors[0]


@pytest.mark.parametrize("fail", [False, True])
def test_document_is_reused_and_released_in_export_thread(fail, monkeypatch):
    from PyQt5 import sip

    font_family = prepare_analysis_report_runtime()
    destroyed_in = []
    delete = sip.delete

    def tracked_delete(document):
        destroyed_in.append(get_ident())
        delete(document)

    monkeypatch.setattr(sip, "delete", tracked_delete)

    def measure():
        layout = MeasuredReportLayout(_report_stylesheet(), font_family)
        try:
            with layout:
                document = layout.document("<p>报告内容</p>")
                assert not sip.isdeleted(document)
                assert document.size().height() > 0
                next_document = layout.document("<p>下一页</p>")
                assert next_document is document
                if fail:
                    raise ValueError("drawing failed")
        except ValueError as error:
            assert fail and str(error) == "drawing failed"
        assert sip.isdeleted(document)
        assert destroyed_in == [get_ident()]

    _run_export_thread(measure)


def test_many_chart_pages_export_while_ui_collects_garbage(tmp_path):
    import base64
    from PyQt5.QtCore import QByteArray, QBuffer, QIODevice
    from PyQt5.QtGui import QColor, QImage

    font_family = prepare_analysis_report_runtime()
    image = QImage(32, 32, QImage.Format_RGB32)
    image.fill(QColor("#4e8df5"))
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.WriteOnly)
    assert image.save(buffer, "PNG")
    buffer.close()
    encoded = base64.b64encode(bytes(data)).decode("ascii")
    target = tmp_path / "many-chart-pages.pdf"

    def export():
        layout = MeasuredReportLayout(_report_stylesheet(), font_family)
        layout.pages = [
            f"<h3>图表 {i}</h3><img src='data:image/png;base64,{encoded}' width='600' height='400'>"
            for i in range(40)
        ]
        for _ in range(3):
            assert all(layout.fits(page) for page in layout.pages)
        _render_html_pdf(str(target), layout)

    _run_export_thread(export, collect_garbage=True)

    fitz = pytest.importorskip("fitz")
    with fitz.open(target) as pdf:
        assert len(pdf) == 40
        assert all(page.get_images() for page in pdf)


def table_rows(page):
    return re.findall(r"<tbody>(.*?)</tbody>", page, re.S)


@pytest.mark.parametrize("channels", [1, 5])
@pytest.mark.parametrize("count", [1, 25, 64])
def test_measured_pages_keep_all_rows_and_fill_available_space(tmp_path, channels, count):
    layout = make_layout()
    headers = "<th>样本</th><th>输入电压</th><th>输出负载（A）</th>"
    headers += "".join(f"<th>CH{i}（位置{i}）</th>" for i in range(1, channels + 1))
    rows = [
        f"<tr><td>S{index:03d}</td><td>230Vac/50Hz</td><td>{'/' if index % 3 == 0 else '0.3'}</td>"
        + "".join("<td>39.67<br>OK<br>限值 0.00～100.00</td>" for _ in range(channels))
        + "</tr>"
        for index in range(count)
    ]
    layout.append_table("A口 / 档位1 / SPL", headers, rows, lead="<h1>报告</h1>")
    layout.finish()
    all_rows = []
    for index, page in enumerate(layout.pages):
        assert layout.fits(page)
        assert headers in page and "A口 / 档位1 / SPL" in page
        page_rows = re.findall(r"<tr>.*?</tr>", table_rows(page)[0], re.S)
        all_rows.extend(page_rows)
        if index + 1 < len(layout.pages):
            next_row = re.findall(r"<tr>.*?</tr>", table_rows(layout.pages[index + 1])[0], re.S)[0]
            with_next = page.replace("</tbody>", next_row + "</tbody>")
            assert not layout.fits(with_next), "A full next row still fits before the page break"
    assert all_rows == rows

    fitz = pytest.importorskip("fitz")
    output = tmp_path / "measured.pdf"
    _render_html_pdf(str(output), layout)
    with fitz.open(output) as pdf:
        assert len(pdf) == len(layout.pages)
        seen = []
        for page in pdf:
            text = page.get_text()
            seen.extend(re.findall(r"S\d{3}", text))
            footer_top = min(word[1] for word in page.get_text("words") if word[4] == "/" and word[1] > page.rect.height * .9)
            assert max(drawing["rect"].y1 for drawing in page.get_drawings()) < footer_top
        assert seen == [f"S{index:03d}" for index in range(count)]


def test_long_row_is_whole_and_oversized_row_is_reported():
    layout = make_layout()
    headers = "<th>序号</th><th>说明</th>"
    rows = [f"<tr><td>{i}</td><td>{'长说明<br>' * 18}</td></tr>" for i in range(4)]
    layout.append_table("结果说明", headers, rows)
    layout.finish()
    assert len(layout.pages) > 1
    assert "".join(table_rows(p)[0] for p in layout.pages) == "".join(rows)
    assert all(layout.fits(page) for page in layout.pages)
    oversized = make_layout()
    with pytest.raises(ValueError, match="单行内容超过一页"):
        oversized.append_table("结果说明", headers, ["<tr><td>1</td><td>" + "过长说明<br>" * 200 + "</td></tr>"])


def test_pagination_honors_cancellation_during_measurement():
    checks = 0

    def cancelled():
        nonlocal checks
        checks += 1
        return checks > 8

    layout = make_layout(cancelled)
    with pytest.raises(InterruptedError):
        layout.append_table("SPL", "<th>样本</th>", [f"<tr><td>{i}</td></tr>" for i in range(100)])
    assert checks == 9
