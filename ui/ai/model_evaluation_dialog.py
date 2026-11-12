"""Evaluation layout only: no audio loading, inference, or fabricated metrics."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from base.ai_model_catalog import ModelRecord
from ui.ai.model_dialog_base import AiModelDialogBase, model_button, model_table


class AiModelEvaluationDialog(AiModelDialogBase):
    def __init__(self, record: ModelRecord, parent=None):
        super().__init__("模型评估", parent)
        self.record = record
        self.resize(820, 650)
        self.setMinimumSize(660, 500)
        self.add_heading("模型评估")
        content = QWidget(self)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)
        self.model_label = QLabel(f"当前模型：{record.model_name}", content)
        self.model_label.setTextFormat(Qt.PlainText)
        self.model_label.setWordWrap(True)
        layout.addWidget(self.model_label)
        samples = QHBoxLayout()
        self.select_audio_button = model_button("选择评估音频", content)
        self.select_audio_button.setEnabled(False)
        self.select_audio_button.setToolTip("评估音频选择功能尚未接入。")
        samples.addWidget(self.select_audio_button)
        self.sample_count_label = QLabel("已选 0 条音频", content)
        self.sample_count_label.setObjectName("aiModelMuted")
        samples.addWidget(self.sample_count_label)
        samples.addStretch()
        layout.addLayout(samples)
        counts = QLabel("正常 0 条    异常 0 条    未标注 0 条", content)
        counts.setObjectName("aiModelMuted")
        layout.addWidget(counts)
        metrics = QHBoxLayout()
        metrics.setSpacing(24)
        self.metric_labels = {}
        for key, title, denominator in (
            ("miss", "异常漏检率", "漏检 / 异常样本"),
            ("false_alarm", "正常误报率", "误报 / 正常样本"),
            ("accuracy", "总体准确率", "正确 / 全部样本"),
        ):
            column = QVBoxLayout()
            column.setSpacing(5)
            name_label = QLabel(title, content)
            name_label.setObjectName("aiModelMuted")
            value = QLabel("—", content)
            value.setObjectName("aiModelMetric")
            value.setAccessibleName(title)
            self.metric_labels[key] = value
            caption = QLabel(denominator, content)
            caption.setObjectName("aiModelMuted")
            column.addWidget(name_label)
            column.addWidget(value)
            column.addWidget(caption)
            metrics.addLayout(column, 1)
        layout.addLayout(metrics)
        layout.addWidget(QLabel("逐条结果", content))
        self.result_table = model_table(["音频名称", "人工标注", "模型判断", "结果"], content)
        layout.addWidget(self.result_table, 1)
        self.root_layout.addWidget(content, 1)
        footer = self.add_footer()
        footer.addStretch()
        close_button = model_button("关闭", self)
        close_button.clicked.connect(self.reject)
        footer.addWidget(close_button)
        self.start_button = model_button("开始评估", self, primary=True)
        self.start_button.setEnabled(False)
        self.start_button.setToolTip("模型评估功能尚未接入。")
        footer.addWidget(self.start_button)
