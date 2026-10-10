"""Statistics retirement must leave existing mode and mark actions intact."""

import builtins
from unittest.mock import Mock

import pytest
from PyQt5.QtWidgets import QLineEdit, QMessageBox

from ui.sequence.sequencement_count_board import SequenceCountBoard


@pytest.mark.parametrize("existing", [False, True])
def test_modes_do_not_access_statistics(qt_app, tmp_path, monkeypatch, existing):
    from ui.sequence import sequencement_count_board as module

    mark_path = tmp_path / "ui/ui_config/mark_result.json"
    test_path = tmp_path / "log/test_result_log/old.dat"
    if existing:
        for path in (mark_path, test_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"legacy statistics must remain untouched")
    monkeypatch.setattr(module, "DEFAULT_DIR", tmp_path.as_posix() + "/")
    original_open = builtins.open
    accessed = []

    def guarded_open(file, *args, **kwargs):
        name = str(file).replace("\\", "/")
        if name.endswith("/mark_result.json") or "/test_result_log/" in name:
            accessed.append(name)
            raise AssertionError("statistics I/O is retired")
        return original_open(file, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, "open", guarded_open)
        board = SequenceCountBoard({})
        states = []
        board.register_mode_change_callback(states.append)
        board.on_test_btn_clicked()
        board.on_mark_btn_clicked()
        assert [state["mode"] for state in states] == ["test", "mark"]
        assert board.findChildren(QLineEdit) == []
        assert not hasattr(board, "reset_btn")
        assert not accessed
        board.deleteLater()
    for path in (mark_path, test_path):
        if existing:
            assert path.read_bytes() == b"legacy statistics must remain untouched"
        else:
            assert not path.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == (["log", "ui"] if existing else [])


@pytest.mark.parametrize("preserve_mode", [False, True])
def test_unavailable_mode_preserves_existing_fallback_contract(qt_app, monkeypatch, preserve_mode):
    board = SequenceCountBoard({})
    board.on_test_btn_clicked()
    states = []
    board.register_mode_change_callback(states.append)
    board.set_test_available(False, "invalid configuration", preserve_mode=preserve_mode)
    assert board.mode == ("test" if preserve_mode else "mark")
    assert states[-1]["test_available"] is False
    assert not board.test_btn.isEnabled()
    prompt = Mock()
    monkeypatch.setattr(QMessageBox, "information", prompt)
    board.on_test_btn_clicked()
    prompt.assert_called_once_with(board, "提示", "invalid configuration")
    assert board.mode == "mark"
    board.deleteLater()


def test_allowed_mode_keeps_warning_and_mark_button_signals(qt_app, monkeypatch):
    board = SequenceCountBoard({})
    prompt = Mock()
    monkeypatch.setattr(QMessageBox, "information", prompt)
    board.set_test_available(True, "no judging rules")
    board.test_btn.click()
    assert board.mode == "test"
    prompt.assert_called_once_with(board, "测试结果说明", "no judging rules")
    board.mark_btn.click()
    assert board.mode == "mark"
    marked = []
    board.ok_btn.clicked.connect(lambda: marked.append("OK"))
    board.ng_btn.clicked.connect(lambda: marked.append("NG"))
    board.ok_btn.click()
    board.ng_btn.click()
    assert marked == ["OK", "NG"]
    board.deleteLater()
