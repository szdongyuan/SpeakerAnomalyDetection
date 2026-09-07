"""One analysis-item window with a channel selector."""

from __future__ import annotations

import base64

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from consts import ui_style_const
from consts.acoustic_analysis.curve_style_consts import (
    LOWER_LIMIT_COLOR,
    MAIN_CURVE_COLOR,
    UPPER_LIMIT_COLOR,
)
from consts.running_consts import DEFAULT_DIR
from ui.curve_style import resolve_curve_colors


_CHANNEL_SELECTOR_WIDTH = 88
_CHANNEL_SELECTOR_ARROW_ICON = (
    DEFAULT_DIR + "ui/ui_analysis_config/assets/combobox_down_arrow.svg"
)


class _ResponsivePixmapLabel(QLabel):
    """Keep the complete result image visible as its page is resized."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source_pixmap = QPixmap()
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(False)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(1, 1)

    def set_source_pixmap(self, pixmap):
        self._source_pixmap = QPixmap(pixmap)
        self._refresh_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self):
        if self._source_pixmap.isNull():
            return
        target_size = self.contentsRect().size()
        if target_size.isEmpty():
            return
        self.setPixmap(
            self._source_pixmap.scaled(
                target_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )


class AnalysisMultichannelResultWindow(QWidget):
    """Display one configured analysis item; switch channels without rerunning."""

    def __init__(
        self,
        config_key,
        instance_results,
        config=None,
        channel_labels=None,
        source_label="",
        parent=None,
    ):
        super().__init__(parent)
        self._config_key = str(config_key or "分析结果")
        self._source_label = str(source_label or "").strip()
        self._results = tuple(
            sorted(instance_results, key=lambda item: int(item.raw_channel))
        )
        self._config = dict(config or {})
        self._channel_labels = {
            str(channel): str(label).strip()
            for channel, label in dict(channel_labels or {}).items()
            if str(label).strip()
        }
        self._channel_combo = QComboBox(self)
        self._content = QStackedWidget(self)
        self._pages = []
        title = self._config_key
        if self._source_label:
            title = f"{title} — {self._source_label}"
        self.setWindowTitle(title)
        self.setMinimumSize(760, 520)
        self._build_ui()

    @property
    def channel_combo(self):
        return self._channel_combo

    def _build_ui(self):
        self.setObjectName("analysisResultWindow")
        self._content.setObjectName("analysisResultContent")
        self._channel_combo.setObjectName("analysisResultChannelCombo")
        self.setStyleSheet(
            f"""
            QWidget#analysisResultWindow {{
                background-color: {ui_style_const.COLOR_PAGE_BG};
            }}
            QFrame#analysisResultChannelPanel {{
                background-color: transparent;
                border: none;
            }}
            QComboBox#analysisResultChannelCombo {{
                background-color: {ui_style_const.COLOR_CARD_BG};
                color: {ui_style_const.COLOR_TEXT};
                border: 1px solid {ui_style_const.COLOR_BORDER_STRONG};
                border-radius: 4px;
                padding: 3px 1px 3px 5px;
                font-family: {ui_style_const.UI_FONT_FAMILY};
                font-size: 12px;
            }}
            QComboBox#analysisResultChannelCombo::drop-down {{
                border: none;
                border-left: 1px solid {ui_style_const.COLOR_BORDER};
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
                width: 20px;
                subcontrol-origin: padding;
                subcontrol-position: top right;
                background-color: {ui_style_const.COLOR_CONTROL_HOVER};
            }}
            QComboBox#analysisResultChannelCombo::drop-down:hover {{
                background-color: {ui_style_const.COLOR_CONTROL_PRESSED};
            }}
            QComboBox#analysisResultChannelCombo::down-arrow {{
                image: url("{_CHANNEL_SELECTOR_ARROW_ICON}");
                width: 12px;
                height: 8px;
            }}
            QComboBox#analysisResultChannelCombo QAbstractItemView {{
                background-color: {ui_style_const.COLOR_CARD_BG};
                color: {ui_style_const.COLOR_TEXT};
                selection-background-color: {ui_style_const.COLOR_PRIMARY};
                selection-color: #FFFFFF;
            }}
            QStackedWidget#analysisResultContent {{
                background-color: #FBFCFE;
                border: 1px solid {ui_style_const.COLOR_BORDER};
                border-radius: 5px;
            }}
            """
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(self._content, stretch=1)
        selector = QFrame(self)
        selector.setObjectName("analysisResultChannelPanel")
        selector.setFixedWidth(_CHANNEL_SELECTOR_WIDTH)
        selector_layout = QVBoxLayout(selector)
        selector_layout.setContentsMargins(0, 0, 0, 0)
        selector_layout.setSpacing(0)
        self._channel_combo.setFixedSize(_CHANNEL_SELECTOR_WIDTH, 28)
        selector_layout.addWidget(self._channel_combo, alignment=Qt.AlignTop)
        selector_layout.addStretch(1)
        layout.addWidget(selector)
        for result in self._results:
            page = self._build_result_page(result)
            self._pages.append(page)
            self._content.addWidget(page)
            channel_name = f"CH{int(result.raw_channel) + 1}"
            position_label = self._channel_labels.get(channel_name, "")
            display_name = (
                f"{channel_name}({position_label})"
                if position_label
                else channel_name
            )
            self._channel_combo.addItem(display_name, int(result.raw_channel))
        self._channel_combo.currentIndexChanged.connect(
            self._content.setCurrentIndex
        )

    def _build_result_page(self, result):
        if result.execution_status != "分析完成":
            return self._message_page(
                "分析失败\n" + str(result.error_message or "未返回错误信息")
            )
        payload = result.display_payload.to_dict()
        kind = payload.get("kind")
        if kind == "curve":
            return self._curve_page(payload)
        if kind == "bar":
            return self._bar_page(payload)
        if kind == "image":
            return self._image_page(payload)
        if kind == "values":
            return self._values_page(payload)
        return self._message_page("分析完成，但没有可显示的结果。")

    def _curve_page(self, payload):
        widget = self._new_plot_widget()
        colors = resolve_curve_colors(self._config)
        x_values = np.asarray(payload.get("x") or [], dtype=np.float64)
        y_values = np.asarray(payload.get("y") or [], dtype=np.float64)
        widget.plot(
            x_values,
            y_values,
            pen=pg.mkPen(colors[MAIN_CURVE_COLOR], width=2),
        )
        for key, color_key in (
            ("lower", LOWER_LIMIT_COLOR),
            ("upper", UPPER_LIMIT_COLOR),
        ):
            values = np.asarray(payload.get(key) or [], dtype=np.float64)
            if values.size == x_values.size and values.size:
                widget.plot(
                    x_values,
                    values,
                    pen=pg.mkPen(
                        colors[color_key],
                        width=2,
                        style=Qt.DashLine,
                    ),
                )
        self._finish_plot_widget(widget, payload)
        widget.setLogMode(x=bool(payload.get("log_x")), y=False)
        overall = payload.get("overall_spl")
        if overall is not None:
            try:
                widget.setTitle(
                    f"总体声压级：{float(overall):.2f} dB",
                    color=ui_style_const.COLOR_TEXT_MUTED,
                )
            except (TypeError, ValueError):
                pass
        return widget

    def _bar_page(self, payload):
        widget = self._new_plot_widget()
        colors = resolve_curve_colors(self._config)
        y_values = np.asarray(payload.get("y") or [], dtype=np.float64)
        positions = np.arange(y_values.size, dtype=np.float64)
        finite_mask = np.isfinite(y_values)
        plot_values = np.where(finite_mask, y_values, 0.0)
        brushes = [
            pg.mkBrush(colors[MAIN_CURVE_COLOR])
            if is_finite
            else pg.mkBrush("#BDBDBD")
            for is_finite in finite_mask
        ]
        out_mask = np.asarray(payload.get("out_mask") or [], dtype=bool)
        if out_mask.size == y_values.size:
            for index in np.flatnonzero(out_mask & finite_mask):
                brushes[int(index)] = pg.mkBrush("#F44336")
        widget.addItem(
            pg.BarGraphItem(
                x=positions,
                height=plot_values,
                width=0.7,
                brushes=brushes,
                pen=pg.mkPen("#FFFFFF", width=0.5),
            )
        )
        for key, color_key in (
            ("lower", LOWER_LIMIT_COLOR),
            ("upper", UPPER_LIMIT_COLOR),
        ):
            values = np.asarray(payload.get(key) or [], dtype=np.float64)
            if values.size == y_values.size and values.size:
                widget.plot(
                    positions,
                    values,
                    pen=pg.mkPen(
                        colors[color_key],
                        width=2,
                        style=Qt.DashLine,
                    ),
                )
        labels = [str(label) for label in payload.get("labels") or ()]
        if len(labels) == y_values.size:
            step = max(1, int(np.ceil(len(labels) / 16.0)))
            indices = list(range(0, len(labels), step))
            if indices[-1] != len(labels) - 1:
                indices.append(len(labels) - 1)
            widget.getAxis("bottom").setTicks(
                [[(index, labels[index]) for index in indices]]
            )
        self._finish_plot_widget(widget, payload)
        return widget

    @staticmethod
    def _new_plot_widget():
        widget = pg.PlotWidget()
        widget.setBackground("#FBFCFE")
        widget.setStyleSheet("border: none;")
        return widget

    @staticmethod
    def _finish_plot_widget(widget, payload):
        widget.setLabel(
            "bottom",
            str(payload.get("x_label") or ""),
            color=ui_style_const.COLOR_TEXT_MUTED,
        )
        widget.setLabel(
            "left",
            str(payload.get("y_label") or ""),
            color=ui_style_const.COLOR_TEXT_MUTED,
        )
        plot_item = widget.getPlotItem()
        for axis_name in ("bottom", "left"):
            axis = plot_item.getAxis(axis_name)
            axis.setPen(pg.mkPen(ui_style_const.COLOR_BORDER_STRONG))
            axis.setTextPen(pg.mkPen(ui_style_const.COLOR_TEXT_MUTED))
        widget.showGrid(x=True, y=True, alpha=0.22)

    def _image_page(self, payload):
        label = _ResponsivePixmapLabel()
        encoded = str(payload.get("png_base64") or "")
        pixmap = QPixmap()
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError):
            raw = b""
        if raw and pixmap.loadFromData(raw, "PNG"):
            label.set_source_pixmap(pixmap)
        else:
            label.setText("结果图片无法显示。")
        return label

    def _values_page(self, payload):
        page = QWidget()
        form = QFormLayout(page)
        labels = (
            ("最终判定", payload.get("result")),
            ("评分模型", payload.get("model_name")),
            ("模型输出值", payload.get("model_output_value")),
            ("判定阈值", payload.get("decision_threshold")),
            ("OK Score", payload.get("ok_score")),
            ("NG Score", payload.get("ng_score")),
        )
        for name, value in labels:
            if value is not None and value != "":
                form.addRow(name, QLabel(str(value)))
        return page

    @staticmethod
    def _message_page(message):
        label = QLabel(str(message or ""))
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        return label
