import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin


class _Logger:
    def info(self, _message):
        pass

    def warning(self, _message):
        pass

    def error(self, _message):
        pass


class _ExportHost(SequenceWidgetAnalysisOpsMixin):
    def __init__(self):
        self.analysis_config = {
            "结果导出 (Excel) 1": {
                "type": "Excel",
                "fast_mode": True,
                "save_items": [],
            }
        }
        self.recorded_signal_info = {"file_path": "D:/audio/test.wav"}
        self.recorded_path = None
        self._excel_exported_record_id = None
        self._excel_export_cache = {
            "record_id": "D:/audio/test.wav",
            "sn": "SN001",
            "analysis_items_data": {},
            "analysis_result_dict": {},
        }
        self.default_logger = _Logger()

    def _schedule_excel_spool_build(self, _configs):
        raise AssertionError("failed exports must not schedule an Excel build")


class _MessageBox:
    Warning = 1
    AcceptRole = 2
    RejectRole = 3
    instances = []
    clicked_roles = []

    def __init__(self, _parent):
        self.title = ""
        self.text = ""
        self.buttons = []
        self._clicked_button = None
        self.__class__.instances.append(self)

    def setIcon(self, _icon):
        pass

    def setWindowTitle(self, title):
        self.title = title

    def setText(self, text):
        self.text = text

    def addButton(self, text, role):
        button = SimpleNamespace(text=text, role=role)
        self.buttons.append(button)
        return button

    def setDefaultButton(self, _button):
        pass

    def exec_(self):
        role = self.__class__.clicked_roles.pop(0) if self.__class__.clicked_roles else None
        self._clicked_button = next(
            (button for button in self.buttons if button.role == role),
            None,
        )

    def clickedButton(self):
        return self._clicked_button


def _install_message_box(monkeypatch, clicked_roles=()):
    _MessageBox.instances = []
    _MessageBox.clicked_roles = list(clicked_roles)
    monkeypatch.setattr(
        "ui.sequence.sequence_widget_analysis_ops.QMessageBox",
        _MessageBox,
    )


def test_empty_save_items_shows_configuration_guidance_without_retry(monkeypatch):
    _install_message_box(monkeypatch)
    monkeypatch.setattr(
        "ui.sequence.sequence_widget_analysis_ops.export_analysis_to_csv_spool",
        lambda *_args, **_kwargs: SimpleNamespace(
            ok=False,
            message="未选择需要保存的分析项",
        ),
    )

    _ExportHost()._maybe_export_excel_results()

    dialog = _MessageBox.instances[0]
    assert dialog.title == "Excel导出配置不完整"
    assert "未选择需要保存的分析项" in dialog.text
    assert "至少勾选一个分析项" in dialog.text
    assert "文件被占用" not in dialog.text
    assert [button.text for button in dialog.buttons] == ["知道了"]


def test_file_error_shows_real_reason_and_allows_retry(monkeypatch):
    _install_message_box(
        monkeypatch,
        clicked_roles=(_MessageBox.AcceptRole, _MessageBox.RejectRole),
    )
    calls = []

    def fail_export(*_args, **_kwargs):
        calls.append(True)
        return SimpleNamespace(
            ok=False,
            message="Excel文件被占用，请关闭后重试: D:/results.xlsx",
        )

    monkeypatch.setattr(
        "ui.sequence.sequence_widget_analysis_ops.export_analysis_to_csv_spool",
        fail_export,
    )

    _ExportHost()._maybe_export_excel_results()

    assert len(calls) == 2
    assert len(_MessageBox.instances) == 2
    first_dialog = _MessageBox.instances[0]
    assert first_dialog.title == "Excel结果导出失败"
    assert "D:/results.xlsx" in first_dialog.text
    assert [button.text for button in first_dialog.buttons] == ["重试", "忽略"]
