"""Explicit confirmation of a complete recording deletion."""

from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from ui.config_dialog_base import ConfigDialogBase


class ArchiveAudioDeleteDialog(ConfigDialogBase):
    def __init__(self, file_counts, recording_count, parent=None):
        super().__init__(parent)
        missing_files = not any(file_counts.values())
        self.setWindowTitle("从列表移除" if missing_files else "确认删除录音")
        width, height = (470, 145) if missing_files else (510, 330)
        self.resize(width, height)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(14)
        if missing_files:
            text = (
                "找不到这个音频文件，是否从列表中移除？" if recording_count == 1
                else f"找不到所选的 {recording_count} 个音频文件，是否从列表中移除？"
            )
        else:
            text = f"将删除所选 {recording_count} 条录音及其配套文件："
        title = QLabel(text)
        title.setWordWrap(True)
        layout.addWidget(title)
        if not missing_files:
            for kind, label in (
                ("wav", "录音 WAV"),
                ("raw_csv", "录音采样数据（CSV）"),
                ("images", "分析图片"),
                ("analysis_csv", "分析结果数据（CSV）"),
            ):
                layout.addWidget(QLabel(f"{label}：{file_counts[kind]} 个文件"))
            layout.addWidget(QLabel(f"同时移除 {recording_count} 条对应数据库记录。"))
            warning = QLabel("此操作不可恢复。")
            font = warning.font()
            font.setBold(True)
            warning.setFont(font)
            layout.addWidget(warning)
        layout.addStretch()
        buttons = QHBoxLayout()
        buttons.addStretch()
        self.cancel_button = QPushButton("取消")
        self.delete_button = QPushButton("从列表移除" if missing_files else "确认删除")
        self.cancel_button.setDefault(True)
        self.delete_button.setAutoDefault(False)
        self.cancel_button.setFocus()
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.delete_button)
        layout.addLayout(buttons)
        self.cancel_button.clicked.connect(self.reject)
        self.delete_button.clicked.connect(self.accept)
