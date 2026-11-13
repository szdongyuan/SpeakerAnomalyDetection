from unittest.mock import Mock

import pytest

from unit_test.ui.test_workbench_input_ops import host


@pytest.fixture
def barcode_host(host, monkeypatch):
    monkeypatch.setattr(host, '_warn_invalid_barcode', Mock())
    return host


@pytest.mark.parametrize("source", [
    "hid", "serial", "wedge_enter", "wedge_debounce",
    "wedge_global_enter", "wedge_global_debounce",
])
@pytest.mark.parametrize("barcode", ["ABC/123/XY", "/ABC123", "ABC123/", "/"])
def test_scan_preserves_slashes_and_allows_recording(barcode_host, source, barcode):
    barcode_host._commit_barcode(barcode, source=source)

    assert barcode_host.lineedit_s_or_n.text() == barcode
    assert barcode_host._validate_current_barcode_before_recording()
    barcode_host._warn_invalid_barcode.assert_not_called()


@pytest.mark.parametrize("character", list('\\:*?"<>|') + [" ", "\t", "\r", "\n", "\x00", "\x7f"])
def test_other_invalid_characters_remain_blocked(barcode_host, character):
    barcode = f"ABC/{character}123"
    barcode_host._commit_barcode(barcode, source="serial")

    assert barcode_host.lineedit_s_or_n.text() == ""
    barcode_host._warn_invalid_barcode.assert_called_once_with(barcode, [character])

    barcode_host.lineedit_s_or_n.setText(barcode)
    assert not barcode_host._validate_current_barcode_before_recording()


def test_manual_slash_barcode_allows_recording(barcode_host):
    barcode_host.lineedit_s_or_n.setText("ABC/123/XY")

    assert barcode_host._validate_current_barcode_before_recording()
    assert barcode_host.lineedit_s_or_n.text() == "ABC/123/XY"
    barcode_host._warn_invalid_barcode.assert_not_called()
