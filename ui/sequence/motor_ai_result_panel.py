from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget, QSizePolicy

from consts import ui_style_const
from ui.sequence.motor_panel_common import (
    MotorSectionCard,
    create_badge_label,
)


class MotorAiResultPanel(QWidget):
    _TONE_STYLES = {
        "pending": ui_style_const.motor_status_badge_pending_style,
        "running": ui_style_const.motor_status_badge_running_style,
        "ok": ui_style_const.motor_status_badge_ok_style,
        "ng": ui_style_const.motor_status_badge_ng_style,
    }
    _TITLE_TONE_STYLES = {
        "pending": ui_style_const.motor_inline_title_label_pending_style,
        "running": ui_style_const.motor_inline_title_label_running_style,
        "ok": ui_style_const.motor_inline_title_label_ok_style,
        "ng": ui_style_const.motor_inline_title_label_ng_style,
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._direction_title_width = 88
        self._direction_row_spacing = 10
        self.stage_value = create_badge_label()
        self.forward_value = create_badge_label()
        self.forward_title_label = QLabel("正转结果")
        self.forward_ok_score_label = QLabel("OK分：--")
        self.forward_ng_score_label = QLabel("NG分：--")
        self.reverse_value = create_badge_label()
        self.reverse_title_label = QLabel("反转结果")
        self.reverse_ok_score_label = QLabel("OK分：--")
        self.reverse_ng_score_label = QLabel("NG分：--")
        self.final_value = create_badge_label()

        self._init_ui()
        self.reset()

    def _init_ui(self):
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        card = MotorSectionCard("AI评判结果")
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        card.content_layout.setContentsMargins(14, 14, 14, 14)
        card.content_layout.setSpacing(12)

        card.content_layout.addWidget(self._create_direction_result_section(
            self.forward_title_label,
            self.forward_value,
            self.forward_ok_score_label,
            self.forward_ng_score_label,
        ))
        card.content_layout.addWidget(self._create_divider())
        card.content_layout.addWidget(self._create_direction_result_section(
            self.reverse_title_label,
            self.reverse_value,
            self.reverse_ok_score_label,
            self.reverse_ng_score_label,
        ))
        card.content_layout.addWidget(self._create_divider())
        card.content_layout.addWidget(self._create_final_result_section())

        layout.addWidget(card)
        self.setLayout(layout)
        self._apply_result_visual_hierarchy()

    def _create_final_result_section(self):
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title = QLabel("最终结果")
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title.setStyleSheet(ui_style_const.motor_final_result_title_style)
        layout.addWidget(title)
        self.final_value.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout.addWidget(self.final_value)
        widget.setLayout(layout)
        return widget

    def _create_direction_result_section(self, title_label: QLabel, value_widget, ok_score_label: QLabel, ng_score_label: QLabel):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(6)

        result_row = QHBoxLayout()
        result_row.setContentsMargins(0, 0, 0, 0)
        result_row.setSpacing(self._direction_row_spacing)

        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(ui_style_const.motor_inline_title_label_style)
        result_row.addWidget(title_label, stretch=0)
        result_row.addWidget(value_widget, stretch=1)
        layout.addLayout(result_row)

        score_row = QHBoxLayout()
        score_row.setContentsMargins(self._direction_title_width + 4, 0, 0, 0)
        score_row.setSpacing(6)
        score_title_label = QLabel("评分：")
        score_title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        score_title_label.setStyleSheet(ui_style_const.motor_score_title_style)
        ok_score_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        ok_score_label.setStyleSheet(ui_style_const.motor_score_label_style)
        ng_score_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        ng_score_label.setStyleSheet(ui_style_const.motor_score_label_style)
        score_row.addWidget(score_title_label, stretch=0)
        score_row.addWidget(ok_score_label, stretch=0)
        score_row.addWidget(ng_score_label, stretch=0)
        score_row.addStretch()
        layout.addLayout(score_row)

        widget.setLayout(layout)
        return widget

    @staticmethod
    def _create_divider():
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Plain)
        divider.setStyleSheet(ui_style_const.motor_result_divider_style)
        divider.setFixedHeight(1)
        return divider

    def _apply_result_visual_hierarchy(self):
        self.stage_value.setAlignment(Qt.AlignCenter)
        self.stage_value.setMinimumHeight(44)
        self.stage_value.setMinimumWidth(120)
        self.forward_value.setAlignment(Qt.AlignCenter)
        self.reverse_value.setAlignment(Qt.AlignCenter)
        self.final_value.setAlignment(Qt.AlignCenter)
        self.forward_value.setMinimumHeight(50)
        self.reverse_value.setMinimumHeight(50)
        self.final_value.setMinimumHeight(72)
        self.forward_title_label.setFixedWidth(self._direction_title_width)
        self.reverse_title_label.setFixedWidth(self._direction_title_width)
        self.forward_ok_score_label.setMinimumWidth(84)
        self.forward_ng_score_label.setMinimumWidth(84)
        self.reverse_ok_score_label.setMinimumWidth(84)
        self.reverse_ng_score_label.setMinimumWidth(84)

    def reset(self):
        self.set_current_stage("待开始", tone="pending")
        self.set_forward_result("待检测")
        self.set_forward_scores(None, None)
        self.set_reverse_result("待检测")
        self.set_reverse_scores(None, None)
        self.set_final_result("待判定")

    def set_current_stage(self, stage_text: str, tone: str = "pending"):
        self._set_badge_state(self.stage_value, self._normalize_text(stage_text, "待开始"), tone)

    def set_forward_result(self, result_text: str, tone: str = None):
        tone = tone or self._guess_tone_from_result(result_text)
        self._set_badge_state(self.forward_value, self._normalize_text(result_text, "待检测"), tone)
        self._set_direction_title_state(self.forward_title_label, tone)

    def set_reverse_result(self, result_text: str, tone: str = None):
        tone = tone or self._guess_tone_from_result(result_text)
        self._set_badge_state(self.reverse_value, self._normalize_text(result_text, "待检测"), tone)
        self._set_direction_title_state(self.reverse_title_label, tone)

    def set_final_result(self, result_text: str, tone: str = None):
        tone = tone or self._guess_tone_from_result(result_text)
        self._set_badge_state(self.final_value, self._normalize_text(result_text, "待判定"), tone)

    def set_forward_scores(self, ok_score=None, ng_score=None):
        self.forward_ok_score_label.setText(self._format_score_text("OK分", ok_score))
        self.forward_ng_score_label.setText(self._format_score_text("NG分", ng_score))

    def set_reverse_scores(self, ok_score=None, ng_score=None):
        self.reverse_ok_score_label.setText(self._format_score_text("OK分", ok_score))
        self.reverse_ng_score_label.setText(self._format_score_text("NG分", ng_score))

    def _set_badge_state(self, label, text: str, tone: str):
        label.setText(str(text or "--"))
        base_style = ui_style_const.motor_status_badge_large_style if label is self.final_value else (
            ui_style_const.motor_status_badge_compact_style
            if label in (self.forward_value, self.reverse_value)
            else ui_style_const.motor_status_badge_style
        )
        label.setStyleSheet(
            base_style + self._TONE_STYLES.get(tone, self._TONE_STYLES["pending"])
        )

    def _set_direction_title_state(self, label: QLabel, tone: str):
        label.setStyleSheet(
            ui_style_const.motor_inline_title_label_style
            + self._TITLE_TONE_STYLES.get(tone, self._TITLE_TONE_STYLES["pending"])
        )

    @staticmethod
    def _normalize_text(text: str, fallback: str = "--") -> str:
        value = str(text or "").strip()
        return value if value else fallback

    @staticmethod
    def _format_score_text(label: str, score) -> str:
        if score in (None, ""):
            return f"{label}：--"
        try:
            return f"{label}：{float(score):.2f}%"
        except Exception:
            return f"{label}：{score}"

    @staticmethod
    def _guess_tone_from_result(text: str) -> str:
        normalized = str(text or "").strip().upper()
        if normalized == "OK":
            return "ok"
        if normalized == "NG":
            return "ng"
        if "中" in str(text or ""):
            return "running"
        return "pending"
