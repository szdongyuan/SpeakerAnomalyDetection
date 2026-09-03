from types import SimpleNamespace

from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin


class _Host(SequenceWidgetAnalysisOpsMixin):
    def __init__(self, *, auto_analysis=True, channels=(0, 1, 2, 3)):
        self.analysis_config = {
            "auto_analysis": auto_analysis,
            "display_sequence": ["声压级 (SPL) 1", "频谱分析 (Spec) 1"],
            "声压级 (SPL) 1": {
                "type": "SPL",
                "analysis_channels": [0, 1, 2, 3, 4],
            },
            "频谱分析 (Spec) 1": {
                "type": "Spec",
                "analysis_channels": [0, 1, 2, 3, 4],
            },
        }
        self._configured_input_channels = tuple(channels)
        self.sequence_config = [{"seq1": {}}]
        self.mic = {"max_input_channels": 5}
        self._active_product_condition_key = ""
        self.default_logger = SimpleNamespace(warning=lambda _message: None)


def test_recording_preflight_blocks_missing_automatic_analysis_channel(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        "ui.sequence.sequence_widget_analysis_ops.QMessageBox.warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )
    host = _Host()

    assert host.checked_work_status_message() is True
    assert warnings == [
        (
            "分析通道配置不匹配",
            "当前录音通道：CH1、CH2、CH3、CH4\n"
            "以下自动分析项配置了不存在的通道：\n"
            "声压级 (SPL) 1：CH5\n"
            "频谱分析 (Spec) 1：CH5\n\n"
            "请修改分析通道或录音通道配置后再开始录制。",
        )
    ]


def test_recording_preflight_allows_valid_automatic_analysis_channels(monkeypatch):
    warning = []
    monkeypatch.setattr(
        "ui.sequence.sequence_widget_analysis_ops.QMessageBox.warning",
        lambda *_args: warning.append(True),
    )
    host = _Host(channels=(0, 1, 2, 3, 4))

    assert host.checked_work_status_message() is False
    assert warning == []


def test_recording_preflight_does_not_block_record_only_mode(monkeypatch):
    warning = []
    monkeypatch.setattr(
        "ui.sequence.sequence_widget_analysis_ops.QMessageBox.warning",
        lambda *_args: warning.append(True),
    )
    host = _Host(auto_analysis=False)

    assert host.checked_work_status_message() is False
    assert warning == []
