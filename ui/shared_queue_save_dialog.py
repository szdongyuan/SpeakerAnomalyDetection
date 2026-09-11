"""Explicit confirmation for shared or incompletely checked queue writes."""

import os

from PyQt5.QtWidgets import QDialogButtonBox, QLabel, QPlainTextEdit, QVBoxLayout

from ui.config_dialog_base import ConfigDialogBase


class SharedQueueSaveDialog(ConfigDialogBase):
    def __init__(self, target_path, result, parent=None):
        super().__init__(parent)
        self.setWindowTitle("修改共享测试队列")
        layout = QVBoxLayout(self)
        name = os.path.splitext(os.path.basename(target_path))[0]
        if result.issues:
            introduction = "无法完整检查队列引用，保存可能影响未列出的工况"
        else:
            introduction = f"测试队列“{name}”被以下工况共同使用："
        label = QLabel(introduction)
        label.setWordWrap(True)
        layout.addWidget(label)
        rows = []
        product_identities = {}
        for reference in result.references:
            for names in reference.display_names:
                product_identities.setdefault(names.product_name, set()).add(reference.product_identity)
        for reference in result.references:
            variants = []
            for names in reference.display_names:
                text = f"{names.product_name}／{names.group_name}／{names.condition_name}"
                if len(product_identities[names.product_name]) > 1:
                    identity = os.path.basename(reference.product_path) if reference.product_path else "未保存产品"
                    text += f"（{identity}）"
                source = "已保存" if names.source == "saved" else "草稿"
                if len(reference.sources) == 1:
                    source = "仅" + source
                variants.append(f"{text}（{source}）")
            rows.append(
                f"{'; '.join(variants)} [分组 {reference.group_index}，工况 {reference.condition_index}]"
            )
        if result.issues:
            rows.append("\n读取失败的配置：")
            for issue in result.issues:
                rows.append(f"{issue.product_name} — {issue.path or '未保存草稿'}：{issue.message}")
        self.details = QPlainTextEdit()
        self.details.setStyleSheet("font-size: 16px;")
        self.details.setReadOnly(True)
        self.details.setPlainText("\n".join(rows))
        layout.addWidget(self.details)
        explanation = QLabel(
            "保存后，此队列的修改（包括采样率和量程）将同时影响以上所有工况。"
            "若只想修改其中一个工况，请取消后将队列另存为新队列，并为该工况选择新队列。"
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        buttons = QDialogButtonBox()
        self.save_button = buttons.addButton(
            "继续保存（可能影响未列出的工况）" if result.issues else "保存并影响以上工况",
            QDialogButtonBox.AcceptRole,
        )
        self.cancel_button = buttons.addButton("取消", QDialogButtonBox.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.cancel_button.setDefault(True)
        layout.addWidget(buttons)
        self.resize(740, 520)
