"""Choose which existing files to export for the selected recordings."""

from PyQt5.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from ui.config_dialog_base import ConfigDialogBase


class ArchiveAudioPackageDialog(ConfigDialogBase):
    def __init__(self, file_counts, recording_count, parent=None, *, missing_wav_count=0):
        super().__init__(parent)
        self.setWindowTitle("选择打包内容")
        self.resize(460, 280)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(12)
        summary = f"已选{recording_count}条录音"
        if missing_wav_count:
            summary += f"，{missing_wav_count}个 WAV 未找到"
        summary_label = QLabel(summary)
        summary_label.setWordWrap(True)
        layout.addWidget(summary_label)
        self.checkboxes = {}
        for kind, label, description in (
            ("wav", "录音 WAV", "保存的原始录音文件"),
            ("raw_csv", "录音采样数据 CSV", "对应“WAV＋CSV”保存的逐点采样值"),
            ("images", "分析图片", "分析产生的图像"),
            ("analysis_csv", "分析结果数据 CSV", "分析产生的数值或曲线，例如声压级、频谱"),
        ):
            count = file_counts[kind]
            checkbox = QCheckBox(f"{label}（{count}个）")
            checkbox.setEnabled(count > 0)
            checkbox.setChecked(kind == "wav" and count > 0)
            checkbox.setToolTip(description)
            self.checkboxes[kind] = checkbox
            layout.addWidget(checkbox)
        layout.addStretch()
        layout.addWidget(QLabel("附带完整音频数据库"))
        buttons = QHBoxLayout()
        buttons.addStretch()
        self.cancel_button = QPushButton("取消")
        self.continue_button = QPushButton("继续")
        self.continue_button.setDefault(True)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.continue_button)
        layout.addLayout(buttons)
        self.cancel_button.clicked.connect(self.reject)
        self.continue_button.clicked.connect(self.accept)
        for checkbox in self.checkboxes.values():
            checkbox.toggled.connect(self._update_continue_button)
        self._update_continue_button()

    def selected_kinds(self):
        return {kind for kind, checkbox in self.checkboxes.items() if checkbox.isChecked()}

    def _update_continue_button(self):
        self.continue_button.setEnabled(bool(self.selected_kinds()))
