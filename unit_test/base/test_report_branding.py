"""Verify company branding in rendered PDFs and asset failure handling."""

from dataclasses import replace
from pathlib import Path

import pytest
from PyQt5.QtGui import QColor, QImage

from base import report_branding
from base.analysis_report import export_analysis_report_pdf
from base.analysis_report_source import AnalysisItemIdentity
from unit_test.base.test_analysis_report import _candidate


@pytest.fixture
def logo_path(tmp_path, monkeypatch, qt_app):
    path = tmp_path / "公司-LOGO.png"
    image = QImage(360, 120, QImage.Format_RGB32)
    image.fill(QColor("#205080"))
    assert image.save(str(path), "PNG")
    monkeypatch.setattr(report_branding, "COMPANY_LOGO_PATH", path)
    return path


def export_analysis_sample(directory, report_content="values_only", *, include_missing=False):
    candidate = _candidate(directory)
    candidates = [
        replace(candidate, sample=f"S{i}", wav_path=str(directory / f"record-{i}.wav"))
        for i in range(35 if report_content == "values_only" else 5)
    ]
    if include_missing:
        candidates[0] = replace(candidates[0], analysis_items=())
    for item in candidates:
        Path(item.wav_path).write_bytes(b"RIFF-test")
    return export_analysis_report_pdf(
        str(directory / "analysis.pdf"), candidates,
        [AnalysisItemIdentity("custom_spl", "SPL")], report_content=report_content,
    )


@pytest.mark.parametrize("report_content,include_missing", [
    ("values_only", False), ("values_and_charts", False), ("values_only", True),
])
def test_pdf_logo_repeats_without_overlapping_page_content(
    tmp_path, qt_app, logo_path, report_content, include_missing,
):
    fitz = pytest.importorskip("fitz")
    source_bytes = report_branding.COMPANY_LOGO_PATH.read_bytes()
    result = export_analysis_sample(tmp_path, report_content, include_missing=include_missing)
    assert result.ok, result.message
    original = QImage.fromData(source_bytes, "PNG")
    logo_dimensions = (original.width(), original.height())
    with fitz.open(result.file_path) as document:
        assert len(document) > 1
        images = document[0].get_image_info()
        assert len(images) == 1
        image = images[0]
        assert (image["width"], image["height"]) == logo_dimensions
        x0, y0, x1, y1 = image["bbox"]
        assert (x1 - x0) * 25.4 / 72 == pytest.approx(30, abs=0.2)
        # Display dimensions round to whole pixels at 96 DPI (half a pixel is 0.375 pt).
        expected_height = (x1 - x0) * logo_dimensions[1] / logo_dimensions[0]
        assert y1 - y0 == pytest.approx(expected_height, abs=0.4)
        title = "声学测试分析报告"
        title_rect = document[0].search_for(title)[0]
        assert x1 < title_rect.x0
        assert abs((y0 + y1) / 2 - (title_rect.y0 + title_rect.y1) / 2) < 8
        for page in document:
            page_images = page.get_image_info()
            logos = [info for info in page_images
                     if (info["width"], info["height"]) == logo_dimensions]
            assert len(logos) == 1
            assert logos[0]["bbox"] == pytest.approx(image["bbox"], abs=0.1)
            logo_pixels = page.get_pixmap(
                clip=fitz.Rect(logos[0]["bbox"]), matrix=fitz.Matrix(2, 2),
            )
            assert min(logo_pixels.samples) < 128
            body_words = [
                word for word in page.get_text("words")
                if page.number or not fitz.Rect(word[:4]).intersects(title_rect)
            ]
            # Keep at least 3 mm of breathing room below the logo on every page.
            assert min(word[1] for word in body_words) - y1 >= 3 * 72 / 25.4
            if page.number:
                assert not page.search_for(title)
            assert all(info["bbox"][1] > y1 for info in page_images if info not in logos)
        assert all(page.get_text().strip() for page in document)
        if include_missing:
            assert any(page.search_for("缺失明细") for page in document)
    assert report_branding.COMPANY_LOGO_PATH.read_bytes() == source_bytes


@pytest.mark.parametrize("asset_state", ["corrupt", "unreadable"])
def test_export_reports_logo_asset_failure(tmp_path, monkeypatch, logo_path, asset_state):
    if asset_state == "corrupt":
        logo_path.write_bytes(b"invalid PNG")
    else:
        read_bytes = Path.read_bytes

        def deny_logo_read(path):
            if path == logo_path:
                raise PermissionError("test logo access denied")
            return read_bytes(path)

        monkeypatch.setattr(Path, "read_bytes", deny_logo_read)
    result = export_analysis_sample(tmp_path)
    assert not result.ok
    assert "公司 Logo" in result.message
    assert not list(tmp_path.glob("*.pdf"))


@pytest.mark.parametrize("report_content,include_missing", [
    ("values_only", False), ("values_and_charts", False), ("values_only", True),
])
@pytest.mark.parametrize("operation", ["delete", "rename"])
def test_missing_logo_exports_complete_multipage_report(
    tmp_path, logo_path, report_content, include_missing, operation,
):
    fitz = pytest.importorskip("fitz")
    if operation == "delete":
        logo_path.unlink()
    else:
        logo_path.rename(logo_path.with_name("公司-LOGO-renamed.png"))
    result = export_analysis_sample(tmp_path, report_content, include_missing=include_missing)
    assert result.ok, result.message
    assert not any("Logo" in warning or "LOGO" in warning for warning in result.warnings)
    with fitz.open(result.file_path) as document:
        assert len(document) > 1
        texts = [page.get_text() for page in document]
        assert "声学测试分析报告" in texts[0]
        assert all("声学测试分析报告" not in text for text in texts[1:])
        count = 35 if report_content == "values_only" else 5
        words = {word[4] for page in document for word in page.get_text("words")}
        assert {f"S{i}" for i in range(count)} <= words
        for page, text in zip(document, texts):
            assert text.strip()
            assert "页" in text
            assert all((image["width"], image["height"]) != (360, 120)
                       for image in page.get_image_info())
        if report_content == "values_only":
            assert all(not page.get_image_info() for page in document)
        else:
            assert any(page.get_image_info() for page in document)
        if include_missing:
            assert result.warnings
            assert any("缺失明细" in text for text in texts)
        else:
            assert not result.warnings


def test_replacing_logo_changes_next_pdf_without_changing_source(tmp_path, logo_path):
    fitz = pytest.importorskip("fitz")
    for width, height, color in [(360, 120, "#205080"), (240, 100, "#803020")]:
        replacement = QImage(width, height, QImage.Format_RGB32)
        replacement.fill(QColor(color))
        assert replacement.save(str(logo_path), "PNG")
        before = logo_path.read_bytes()
        result = export_analysis_sample(tmp_path)
        assert result.ok, result.message
        with fitz.open(result.file_path) as document:
            for page in document:
                images = page.get_image_info()
                assert len(images) == 1
                image = images[0]
                assert (image["width"], image["height"]) == (width, height)
                x0, y0, x1, y1 = image["bbox"]
                assert y1 - y0 == pytest.approx((x1 - x0) * height / width, abs=0.4)
        assert logo_path.read_bytes() == before
