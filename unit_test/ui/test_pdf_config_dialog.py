import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ui.ui_analysis_config.pdf_config_dialog import PdfConfigWindow


class FakeConfigManager:
    def __init__(self, cfg):
        self.config = cfg
        self.saved = None

    def load_config(self):
        return self.config

    def save_default_config(self, type_name, config_data):
        self.saved = (type_name, config_data)
        return True


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_pdf_config_available_items_exclude_exporters_and_unimplemented(qapp):
    cfg = {
        "SPL1": {"type": "SPL"},
        "SPLF1": {"type": "SPLF"},
        "FR1": {"type": "FR"},
        "RSC1": {"type": "RSC"},
        "HD1": {"type": "HD"},
        "RB1": {"type": "RB"},
        "PRB1": {"type": "PRB"},
        "LP1": {"type": "LP"},
        "FBA1": {"type": "FBA"},
        "FFT1": {"type": "FFT"},
        "Spec1": {"type": "Spec"},
        "AI1": {"type": "AI"},
        "Excel1": {"type": "Excel"},
        "PDF1": {"type": "PDF"},
        "PD1": {"type": "PD"},
        "PM1": {"type": "PM"},
        "ED1": {"type": "ED"},
    }
    window = PdfConfigWindow(FakeConfigManager(cfg), "PDF1")

    assert window._get_available_analysis_items() == [
        "FBA1",
        "FFT1",
        "FR1",
        "HD1",
        "LP1",
        "PRB1",
        "RB1",
        "RSC1",
        "SPL1",
        "SPLF1",
        "Spec1",
    ]


def test_pdf_config_requires_save_dir(qapp):
    window = PdfConfigWindow(FakeConfigManager({"PDF1": {"type": "PDF"}}), "PDF1")

    ok, msg = window._validate_save_dir_text("", create=False)

    assert not ok
    assert "保存目录" in msg


def test_pdf_config_accepts_creatable_save_dir(qapp, tmp_path):
    window = PdfConfigWindow(FakeConfigManager({"PDF1": {"type": "PDF"}}), "PDF1")
    target = tmp_path / "new_pdf_dir"

    ok, msg = window._validate_save_dir_text(str(target), create=True)

    assert ok, msg
    assert target.is_dir()


def test_pdf_config_rejects_unreachable_save_dir(qapp, monkeypatch):
    window = PdfConfigWindow(FakeConfigManager({"PDF1": {"type": "PDF"}}), "PDF1")

    def boom(_path, exist_ok=False):
        raise PermissionError("denied")

    monkeypatch.setattr("os.makedirs", boom)
    ok, msg = window._validate_save_dir_text("D:/blocked/pdf", create=True)

    assert not ok
    assert "保存目录" in msg


def test_pdf_config_rejects_unwritable_save_dir(qapp, tmp_path, monkeypatch):
    window = PdfConfigWindow(FakeConfigManager({"PDF1": {"type": "PDF"}}), "PDF1")
    target = tmp_path / "pdf_dir"
    target.mkdir()

    def deny_probe(*_args, **_kwargs):
        raise PermissionError("no write")

    monkeypatch.setattr("builtins.open", deny_probe)
    ok, msg = window._validate_save_dir_text(str(target), create=True)

    assert not ok
    assert "写入" in msg or "权限" in msg


def test_pdf_config_refuses_accept_when_save_dir_invalid(qapp, monkeypatch):
    from ui.ui_analysis_config import pdf_config_dialog as module

    window = PdfConfigWindow(FakeConfigManager({"PDF1": {"type": "PDF"}}), "PDF1")
    window.save_dir_edit.setText("D:/blocked/pdf")
    monkeypatch.setattr(window, "_validate_save_dir_text", lambda _text, create: (False, "保存目录不可用"))
    monkeypatch.setattr(module.MessageBox, "warning", lambda *args, **kwargs: None)

    window.on_click_ok_btn()

    assert window.result() == 0


def test_pdf_config_refuses_accept_without_selected_items(qapp, tmp_path, monkeypatch):
    from ui.ui_analysis_config import pdf_config_dialog as module

    window = PdfConfigWindow(
        FakeConfigManager({"PDF1": {"type": "PDF"}, "SPL1": {"type": "SPL"}}),
        "PDF1",
    )
    window.save_dir_edit.setText(str(tmp_path))
    window.on_clear_all()
    monkeypatch.setattr(module.MessageBox, "warning", lambda *args, **kwargs: None)

    window.on_click_ok_btn()

    assert window.result() == 0


def test_pdf_config_accepts_with_valid_dir_and_selected_item(qapp, tmp_path):
    window = PdfConfigWindow(
        FakeConfigManager({"PDF1": {"type": "PDF"}, "SPL1": {"type": "SPL"}}),
        "PDF1",
    )
    window.save_dir_edit.setText(str(tmp_path))
    window._item_checkbox_by_name["SPL1"].setChecked(True)

    config = window.on_click_ok_btn()

    assert window.result() == 1
    assert config == {"enabled": True, "save_dir": str(tmp_path), "save_items": ["SPL1"]}
