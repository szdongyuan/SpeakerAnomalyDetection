"""
展示看板窗口 (DisplayWindow)

用于在第二显示器上全屏展示测试结果汇总信息，包括：
- 当前产品 SN / 型号 / OK-NG 判定
- 每日统计（总数、OK、NG、通过率）
- 最近测试记录滚动列表
"""

import json
import os
from collections import deque
from datetime import datetime

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QColor, QPainter, QIcon
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QFrame,
    QApplication,
    QSizePolicy,
    QScrollArea,
)

from consts.running_consts import DEFAULT_DIR

_CONFIG_PATH = os.path.join(DEFAULT_DIR, "ui/ui_config/display_config.json")

_FONT_FAMILY = "Microsoft YaHei"

_COLOR_TITLE_BG = "#1565C0"
_COLOR_OK = "#4CAF50"
_COLOR_NG = "#F44336"
_COLOR_IDLE = "#9E9E9E"
_COLOR_PANEL_BG = "#FAFAFA"
_COLOR_PANEL_BORDER = "#E0E0E0"
_COLOR_TEXT_DARK = "#212121"
_COLOR_TEXT_LIGHT = "#FFFFFF"
_COLOR_STAT_VALUE = "#1565C0"
_COLOR_PASS_RATE_GOOD = "#4CAF50"
_COLOR_PASS_RATE_BAD = "#FF9800"

_HISTORY_MAX = 50


def load_display_config() -> dict:
    if os.path.exists(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"enabled": False}


def save_display_config(config: dict):
    os.makedirs(os.path.dirname(_CONFIG_PATH), exist_ok=True)
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


class _SectionFrame(QFrame):
    """带标题的面板容器。"""

    def __init__(self, title: str, title_en: str = "", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            f"_SectionFrame {{ background: {_COLOR_PANEL_BG}; "
            f"border: 2px solid {_COLOR_PANEL_BORDER}; border-radius: 8px; }}"
        )

        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(16, 8, 16, 12)
        self._outer.setSpacing(6)

        header = QLabel()
        header_text = f"<b>{title}</b>"
        if title_en:
            header_text += f"<br><span style='font-size:14px; color:#757575;'>{title_en}</span>"
        header.setText(header_text)
        header.setFont(_make_font(18, bold=True))
        header.setStyleSheet(f"color: {_COLOR_TITLE_BG}; border: none; background: transparent;")
        self._outer.addWidget(header)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFixedHeight(2)
        sep.setStyleSheet(f"background: {_COLOR_PANEL_BORDER}; border: none;")
        self._outer.addWidget(sep)

        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(0, 4, 0, 0)
        self.content_layout.setSpacing(6)
        self._outer.addLayout(self.content_layout)


def _make_font(size: int, bold: bool = False) -> QFont:
    f = QFont(_FONT_FAMILY, size)
    f.setBold(bold)
    return f


def _make_label(text: str = "", font_size: int = 20, bold: bool = False,
                color: str = _COLOR_TEXT_DARK, align=Qt.AlignLeft) -> QLabel:
    lbl = QLabel(text)
    lbl.setFont(_make_font(font_size, bold))
    lbl.setAlignment(align)
    lbl.setStyleSheet(f"color: {color}; border: none; background: transparent;")
    lbl.setWordWrap(True)
    return lbl


class _StatRow(QWidget):
    """统计行：标签 + 数值。"""

    def __init__(self, label_text: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent; border: none;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)
        self.label = _make_label(label_text, 22)
        self.value = _make_label("—", 32, bold=True, color=_COLOR_STAT_VALUE, align=Qt.AlignRight)
        self.value.setMinimumWidth(120)
        layout.addWidget(self.label, stretch=1)
        layout.addWidget(self.value, stretch=0)


class DisplayWindow(QWidget):
    """全屏展示看板，用于第二显示器实时展示测试结果。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("谛听异音检测 — 展示看板")
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self.setMinimumSize(960, 600)

        self._history = deque(maxlen=_HISTORY_MAX)

        self._build_ui()
        self._set_idle_state()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_title_bar())

        body = QWidget()
        body.setStyleSheet("background: #ECEFF1;")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(20, 16, 20, 16)
        body_layout.setSpacing(16)

        top_row = QHBoxLayout()
        top_row.setSpacing(16)
        top_row.addWidget(self._build_test_info_section(), stretch=3)
        top_row.addWidget(self._build_result_section(), stretch=4)
        body_layout.addLayout(top_row, stretch=4)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(16)
        bottom_row.addWidget(self._build_statistics_section(), stretch=3)
        bottom_row.addWidget(self._build_history_section(), stretch=4)
        body_layout.addLayout(bottom_row, stretch=5)

        root.addWidget(body, stretch=1)

    def _build_title_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(56)
        bar.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0, "
            f"stop:0 {_COLOR_TITLE_BG}, stop:1 #1976D2);"
        )
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(24, 0, 24, 0)
        title = _make_label("谛听异音检测系统", 28, bold=True, color=_COLOR_TEXT_LIGHT, align=Qt.AlignVCenter)
        self._datetime_label = _make_label("", 18, color="#BBDEFB", align=Qt.AlignVCenter | Qt.AlignRight)
        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(self._datetime_label)
        self._update_datetime()
        return bar

    def _build_test_info_section(self) -> _SectionFrame:
        section = _SectionFrame("测试信息", "Test Information")
        grid = QGridLayout()
        grid.setSpacing(10)

        self._sn_label = _make_label("—", 26, bold=True)
        self._model_label = _make_label("型号1", 26, bold=True)

        grid.addWidget(_make_label("SN：", 22, color="#616161"), 0, 0)
        grid.addWidget(self._sn_label, 0, 1)
        grid.addWidget(_make_label("产品型号：", 22, color="#616161"), 1, 0)
        grid.addWidget(self._model_label, 1, 1)
        grid.setColumnStretch(1, 1)

        section.content_layout.addLayout(grid)
        section.content_layout.addStretch()
        return section

    def _build_result_section(self) -> _SectionFrame:
        section = _SectionFrame("判定结果", "Test Result")
        self._result_label = QLabel("待测试")
        self._result_label.setAlignment(Qt.AlignCenter)
        self._result_label.setFont(_make_font(48, bold=False))
        self._result_label.setMinimumHeight(140)
        self._result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._set_result_style("#E0E0E0", "#9E9E9E")
        section.content_layout.addWidget(self._result_label, stretch=1)
        return section

    def _build_statistics_section(self) -> _SectionFrame:
        section = _SectionFrame("每日统计", "Daily Statistics")
        self._stat_total = _StatRow("总数 / Total Quantity：")
        self._stat_ok = _StatRow("OK数 / OK Quantity：")
        self._stat_ng = _StatRow("NG数 / NG Quantity：")
        self._stat_rate = _StatRow("通过率 / Pass Rate：")
        for w in (self._stat_total, self._stat_ok, self._stat_ng, self._stat_rate):
            section.content_layout.addWidget(w)
        section.content_layout.addStretch()
        return section

    def _build_history_section(self) -> _SectionFrame:
        section = _SectionFrame("最近测试记录", "Recent Test History")

        # Table header
        header_frame = QFrame()
        header_frame.setFixedHeight(36)
        header_frame.setStyleSheet(
            f"QFrame {{ background: {_COLOR_TITLE_BG}; border-radius: 4px; border: none; }}"
        )
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(12, 0, 12, 0)
        header_layout.setSpacing(0)
        for text, stretch in [("#", 1), ("时间", 3), ("SN", 5), ("型号", 3), ("结果", 2)]:
            lbl = _make_label(text, 14, bold=True, color=_COLOR_TEXT_LIGHT)
            lbl.setAlignment(Qt.AlignVCenter)
            header_layout.addWidget(lbl, stretch=stretch)
        section.content_layout.addWidget(header_frame)

        # Scrollable rows
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollBar:vertical { width: 10px; background: #ECEFF1; border: none; }"
            "QScrollBar::handle:vertical { background: #90A4AE; border-radius: 5px; min-height: 30px; }"
            "QScrollBar::handle:vertical:hover { background: #78909C; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
        )
        self._history_container = QWidget()
        self._history_container.setStyleSheet("background: transparent;")
        self._history_layout = QVBoxLayout(self._history_container)
        self._history_layout.setContentsMargins(0, 0, 0, 0)
        self._history_layout.setSpacing(2)
        self._history_layout.addStretch()
        scroll.setWidget(self._history_container)
        self._history_scroll = scroll
        section.content_layout.addWidget(scroll)
        return section

    def _set_result_style(self, bg_color: str, text_color: str = _COLOR_TEXT_LIGHT):
        self._result_label.setStyleSheet(
            f"background: {bg_color}; color: {text_color}; "
            f"border-radius: 16px; border: none; padding: 8px;"
        )

    def _set_idle_state(self):
        self._result_label.setText("待测试")
        self._result_label.setFont(_make_font(48, bold=False))
        self._set_result_style("#E0E0E0", "#9E9E9E")

    def _update_datetime(self):
        now = datetime.now().strftime("%Y/%m/%d  %H:%M")
        self._datetime_label.setText(now)

    def update_display(self, data: dict):
        """
        接收测试结果数据，刷新展示看板。

        Parameters
        ----------
        data : dict
            {
                "sn": str,
                "product_model": str,
                "overall_result": "OK" | "NG",
                "statistics": {"total": int, "ok": int, "ng": int, "pass_rate": str},
            }
        """
        self._update_datetime()

        sn = data.get("sn", "—") or "—"
        self._sn_label.setText(sn)

        product_model = data.get("product_model", "")
        if product_model:
            self._model_label.setText(product_model)

        result = data.get("overall_result", "—")
        if result == "OK":
            self._result_label.setFont(_make_font(100, bold=True))
            self._result_label.setText("OK")
            self._set_result_style(_COLOR_OK)
        elif result == "NG":
            self._result_label.setFont(_make_font(100, bold=True))
            self._result_label.setText("NG")
            self._set_result_style(_COLOR_NG)
        else:
            self._set_idle_state()

        stats = data.get("statistics") or {}
        total = stats.get("total", 0)
        ok = stats.get("ok", 0)
        ng = stats.get("ng", 0)
        pass_rate = stats.get("pass_rate", "—")
        self._stat_total.value.setText(str(total))
        self._stat_ok.value.setText(str(ok))
        self._stat_ok.value.setStyleSheet(f"color: {_COLOR_OK}; border: none; background: transparent;")
        self._stat_ng.value.setText(str(ng))
        self._stat_ng.value.setStyleSheet(f"color: {_COLOR_NG}; border: none; background: transparent;")
        self._stat_rate.value.setText(str(pass_rate))
        try:
            rate_num = float(str(pass_rate).replace("%", ""))
            rate_color = _COLOR_PASS_RATE_GOOD if rate_num >= 90 else _COLOR_PASS_RATE_BAD
        except (ValueError, TypeError):
            rate_color = _COLOR_STAT_VALUE
        self._stat_rate.value.setStyleSheet(
            f"color: {rate_color}; border: none; background: transparent;"
        )

        if result in ("OK", "NG"):
            model = self._model_label.text() or "—"
            self._append_history(sn, model, result)

    def _append_history(self, sn: str, model: str, result: str):
        now_str = datetime.now().strftime("%H:%M:%S")
        self._history.appendleft({
            "time": now_str,
            "sn": sn,
            "model": model,
            "result": result,
        })
        self._rebuild_history_ui()

    def _rebuild_history_ui(self):
        while self._history_layout.count() > 0:
            child = self._history_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        for idx, record in enumerate(self._history):
            row = self._create_history_row(idx, record)
            self._history_layout.addWidget(row)
        self._history_layout.addStretch()

        # Auto scroll to top (latest record)
        self._history_scroll.verticalScrollBar().setValue(0)

    def _create_history_row(self, idx: int, record: dict) -> QFrame:
        is_latest = (idx == 0)
        result = record.get("result", "—")
        is_ok = result == "OK"

        row = QFrame()
        bg = "#E8F5E9" if is_ok else "#FFEBEE"
        border_color = _COLOR_OK if is_ok else _COLOR_NG
        border_w = "2px" if is_latest else "1px"
        row.setStyleSheet(
            f"QFrame {{ background: {bg}; "
            f"border: {border_w} solid {border_color if is_latest else _COLOR_PANEL_BORDER}; "
            f"border-radius: 4px; }}"
        )
        row.setFixedHeight(40)

        layout = QHBoxLayout(row)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(0)

        seq_lbl = _make_label(str(idx + 1), 14, color="#757575")
        seq_lbl.setAlignment(Qt.AlignVCenter)
        layout.addWidget(seq_lbl, stretch=1)

        time_lbl = _make_label(record.get("time", ""), 14, color="#424242")
        time_lbl.setAlignment(Qt.AlignVCenter)
        layout.addWidget(time_lbl, stretch=3)

        sn_lbl = _make_label(record.get("sn", "—"), 14, bold=is_latest, color="#212121")
        sn_lbl.setAlignment(Qt.AlignVCenter)
        layout.addWidget(sn_lbl, stretch=5)

        model_lbl = _make_label(record.get("model", "—"), 14, color="#616161")
        model_lbl.setAlignment(Qt.AlignVCenter)
        layout.addWidget(model_lbl, stretch=3)

        badge = QLabel(result)
        badge.setFont(_make_font(12, bold=True))
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedSize(48, 24)
        badge_bg = _COLOR_OK if is_ok else _COLOR_NG
        badge.setStyleSheet(
            f"color: {_COLOR_TEXT_LIGHT}; background: {badge_bg}; "
            f"border-radius: 3px; border: none;"
        )
        result_wrapper = QHBoxLayout()
        result_wrapper.setContentsMargins(0, 0, 0, 0)
        result_wrapper.addWidget(badge)
        result_wrapper.addStretch()
        result_widget = QWidget()
        result_widget.setStyleSheet("background: transparent; border: none;")
        result_widget.setLayout(result_wrapper)
        layout.addWidget(result_widget, stretch=2)

        return row

    def move_to_secondary_screen(self):
        """检测多屏幕环境，将窗口移至第二显示器并最大化。"""
        app = QApplication.instance()
        screens = app.screens() if app else []
        if len(screens) >= 2:
            secondary = screens[1]
            geo = secondary.geometry()
            self.setGeometry(geo)
            self.showFullScreen()
        else:
            self.showMaximized()

    def closeEvent(self, event):
        self._notify_closed()
        event.accept()

    def _notify_closed(self):
        """Uncheck the menu action when window is closed by the user."""
        app = QApplication.instance()
        if app is None:
            return
        for widget in app.topLevelWidgets():
            action = getattr(widget, "function_action_display_board", None)
            if action is not None:
                action.setChecked(False)
                break
