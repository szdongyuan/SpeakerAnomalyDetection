"""Shared widgets for analysis configuration dialogs."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QDialog,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from consts.acoustic_analysis.common_consts import (
    DEFAULT_GOLDEN_SAMPLE_DISPLAY_MODE,
    GOLDEN_SAMPLE_CHECKED_KEY,
    GOLDEN_SAMPLE_DISPLAY_DEVIATION,
    GOLDEN_SAMPLE_DISPLAY_ENVELOPE,
    GOLDEN_SAMPLE_DISPLAY_MODE_KEY,
    LIMIT_VALUE_SEMANTICS_BOUNDS,
    LIMIT_VALUE_SEMANTICS_OFFSET,
)
from consts.acoustic_analysis.curve_style_consts import PLOT_VIEW_DIALOG_WIDTH
from consts.harmonic_detection_consts import (
    HARMONIC_DETECTION_METHOD_KEY,
    HARMONIC_DETECTION_METHOD_LABELS,
    normalize_harmonic_detection_method,
)
from ui.custom_ui_widget.popuputils import PopupUtils
from ui.custom_ui_widget.widgets import (
    CheckBox,
    ComboBox,
    DoubleSpinBox,
    Label,
    MessageBox,
    PushButton,
    RadioButton,
    SpinBox,
)
from ui.plot_view import build_plot_view_config
from ui.ui_analysis_config.curve_color_config_widget import CurveColorConfigWidget
from ui.ui_analysis_config.plot_view_config_widget import PlotViewConfigWidget
from ui.ui_analysis_config.config_normalization import (
    OCTAVE_SMOOTHING_OPTIONS,
    WEIGHTING_OPTIONS,
    normalize_analysis_channel,
    normalize_octave_smoothing,
    normalize_time_smoothing,
    normalize_weighting,
    weighting_to_display_label,
)


OCTAVE_SMOOTHING_LABELS = {
    0: "不平滑",
    1: "1/1 Oct",
    3: "1/3 Oct",
    6: "1/6 Oct",
    12: "1/12 Oct",
    24: "1/24 Oct",
    48: "1/48 Oct",
}

SEMANTIC_GROUP_TITLES = {
    "input": "输入参数",
    "compute": "计算参数",
    "preprocess": "预处理参数",
    "detection": "检测参数",
    "judgment": "判定参数",
    "reference": "基准参数",
    "display": "显示参数",
    "output": "输出参数",
}

VERTICAL_GOLDEN_DIALOG_WIDTH = 630
VERTICAL_GOLDEN_DIALOG_HEIGHT = 840


class AnalysisConfigDialogBase(QDialog):
    """Base dialog helpers for analysis configuration windows."""

    DEFAULT_DIALOG_WIDTH = VERTICAL_GOLDEN_DIALOG_WIDTH
    DEFAULT_DIALOG_HEIGHT = VERTICAL_GOLDEN_DIALOG_HEIGHT

    def __init__(self, *args, disable_close_button: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_standard_window_flags(disable_close_button=disable_close_button)

    def apply_standard_window_flags(self, disable_close_button: bool = True) -> None:
        if disable_close_button:
            self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)

    def create_standard_button_layout(self, default_callback, ok_callback) -> QHBoxLayout:
        btn_layout = QHBoxLayout()
        default_btn = PushButton(" 设为默认 ")
        default_btn.clicked.connect(default_callback)
        ok_btn = PushButton(" 确  认 ")
        ok_btn.clicked.connect(ok_callback)
        btn_layout.addWidget(default_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        return btn_layout

    def show_default_save_popup(self, success_flag: bool) -> None:
        PopupUtils().save_popup(self, success_flag=success_flag)

    def apply_vertical_golden_dialog_size(
        self,
        width: int = DEFAULT_DIALOG_WIDTH,
        height: int = DEFAULT_DIALOG_HEIGHT,
    ) -> None:
        self.setMinimumSize(width, height)
        self.resize(width, height)


class SemanticAnalysisConfigDialogBase(AnalysisConfigDialogBase):
    """Base dialog with semantic navigation and a single scrollable form page."""

    DEFAULT_DIALOG_WIDTH = VERTICAL_GOLDEN_DIALOG_WIDTH
    DEFAULT_DIALOG_HEIGHT = VERTICAL_GOLDEN_DIALOG_HEIGHT

    def __init__(
        self,
        *args,
        nav_width: int = 150,
        disable_close_button: bool = True,
        **kwargs,
    ):
        super().__init__(*args, disable_close_button=disable_close_button, **kwargs)
        self._semantic_nav_buttons: dict[str, PushButton] = {}
        self._semantic_sections: dict[str, QWidget] = {}
        self._semantic_section_contents: dict[str, QWidget] = {}
        self._semantic_section_indicators: dict[str, Label] = {}
        self._semantic_section_collapsed: dict[str, bool] = {}
        self._active_semantic_group_key: str | None = None
        self._default_callback: Callable[[], Any] | None = None
        self._restore_callback: Callable[[], Any] | None = None
        self._ok_callback: Callable[[], Any] | None = None
        self._cancel_callback: Callable[[], Any] | None = None
        self._syncing_scroll = False
        self.curve_color_widget = None
        self.plot_view_config_widget = None
        self.setObjectName("semanticAnalysisConfigDialog")
        self.setStyleSheet(self._semantic_dialog_stylesheet())

        self._root_layout = QVBoxLayout(self)
        self._root_layout.setContentsMargins(12, 12, 12, 12)
        self._root_layout.setSpacing(10)

        self._content_layout = QHBoxLayout()
        self._content_layout.setSpacing(12)

        self.nav_widget = QWidget(self)
        self.nav_widget.setObjectName("semanticNav")
        self.nav_widget.setFixedWidth(nav_width)
        self.nav_layout = QVBoxLayout(self.nav_widget)
        self.nav_layout.setContentsMargins(10, 12, 10, 12)
        self.nav_layout.setSpacing(6)
        self.nav_title_label = Label("参数分组")
        self.nav_title_label.setObjectName("semanticNavTitle")
        self.nav_layout.addWidget(self.nav_title_label)
        self.nav_layout.addStretch(1)

        self.section_scroll_area = QScrollArea(self)
        self.section_scroll_area.setObjectName("semanticSectionScrollArea")
        self.section_scroll_area.setWidgetResizable(True)
        self.section_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.section_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.section_container = QWidget(self.section_scroll_area)
        self.section_container.setObjectName("semanticSectionContainer")
        self.section_container.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        self.section_layout = QVBoxLayout(self.section_container)
        self.section_layout.setContentsMargins(0, 0, 0, 0)
        self.section_layout.setSpacing(22)
        self.section_layout.addStretch(1)
        self.section_scroll_area.setWidget(self.section_container)
        self.section_scroll_area.verticalScrollBar().valueChanged.connect(self._sync_active_section_from_scroll)

        self._content_layout.addWidget(self.nav_widget)
        self._content_layout.addWidget(self.section_scroll_area, 1)
        self._root_layout.addLayout(self._content_layout, 1)
        self._root_layout.addLayout(self._create_semantic_footer_layout())

    def apply_semantic_dialog_size(
        self,
        width: int = DEFAULT_DIALOG_WIDTH,
        height: int = DEFAULT_DIALOG_HEIGHT,
    ) -> None:
        self.apply_vertical_golden_dialog_size(width, height)

    def _semantic_dialog_stylesheet(self) -> str:
        return """
        QDialog#semanticAnalysisConfigDialog {
            background: #f5f7fa;
        }
        QWidget#semanticNav {
            background: #f2f5f9;
            border: 1px solid #d9e0ea;
            border-radius: 8px;
        }
        Label#semanticNavTitle {
            color: #667085;
            font-size: 13px;
            font-weight: 600;
            padding: 0 4px 6px 4px;
        }
        PushButton#semanticNavButton {
            min-height: 32px;
            text-align: left;
            padding: 5px 10px;
            border: 1px solid transparent;
            border-radius: 7px;
            background: transparent;
            color: #344054;
        }
        PushButton#semanticNavButton:hover {
            background: #edf2f7;
            border-color: #d9e0ea;
        }
        PushButton#semanticNavButton[active="true"] {
            background: #e8f0ff;
            border-color: #bad0ff;
            color: #123d93;
            font-weight: 600;
        }
        QScrollArea#semanticSectionScrollArea {
            background: #f8fafc;
            border: 1px solid #d9e0ea;
            border-radius: 8px;
        }
        QWidget#semanticSectionContainer {
            background: #f8fafc;
        }
        QWidget#semanticSectionCard {
            background: #ffffff;
            border: 1px solid #d9e0ea;
            border-radius: 8px;
        }
        QWidget#semanticSectionHeader {
            background: #fbfcfe;
            border-bottom: 1px solid #d9e0ea;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
        }
        Label#semanticSectionTitle {
            color: #1f2937;
            font-size: 16px;
            font-weight: 600;
        }
        Label#semanticSectionDescription {
            color: #667085;
            font-size: 12px;
        }
        Label#semanticSectionIndicator {
            color: #667085;
            font-size: 16px;
            font-weight: 600;
        }
        QWidget#semanticSectionContent {
            background: #ffffff;
            border-bottom-left-radius: 8px;
            border-bottom-right-radius: 8px;
        }
        PushButton {
            min-height: 30px;
            border: 1px solid #b9c4d2;
            border-radius: 6px;
            background: #ffffff;
            padding: 0 12px;
            color: #344054;
        }
        PushButton:hover {
            border-color: #8fa4c0;
            background: #f8fafc;
        }
        PushButton#semanticPrimaryButton {
            background: #2563eb;
            border-color: #1d4ed8;
            color: #ffffff;
        }
        PushButton#semanticPrimaryButton:hover {
            background: #1d4ed8;
            border-color: #1e40af;
        }
        QGroupBox {
            background: #f8fafc;
            border: 1px solid #d9e0ea;
            border-radius: 6px;
            margin-top: 12px;
            padding: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
            color: #344054;
        }
        QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {
            min-height: 28px;
            border: 1px solid #b9c4d2;
            border-radius: 6px;
            background: #ffffff;
            padding: 2px 8px;
            color: #1f2937;
        }
        QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {
            border-color: #8fa4c0;
        }
        """

    def _create_semantic_footer_layout(self) -> QHBoxLayout:
        footer_layout = QHBoxLayout()
        self.semantic_default_btn = PushButton(" 设为默认 ")
        self.semantic_restore_btn = PushButton(" 恢复默认 ")
        self.semantic_cancel_btn = PushButton(" 取  消 ")
        self.semantic_ok_btn = PushButton(" 确  认 ")
        self.semantic_ok_btn.setObjectName("semanticPrimaryButton")

        self.semantic_default_btn.clicked.connect(self._on_default_clicked)
        self.semantic_restore_btn.clicked.connect(self._on_restore_clicked)
        self.semantic_cancel_btn.clicked.connect(self._on_cancel_clicked)
        self.semantic_ok_btn.clicked.connect(self._on_ok_clicked)

        footer_layout.addWidget(self.semantic_default_btn)
        footer_layout.addWidget(self.semantic_restore_btn)
        footer_layout.addStretch()
        footer_layout.addWidget(self.semantic_cancel_btn)
        footer_layout.addWidget(self.semantic_ok_btn)
        return footer_layout

    def set_semantic_button_callbacks(
        self,
        *,
        default_callback: Callable[[], Any] | None = None,
        restore_callback: Callable[[], Any] | None = None,
        ok_callback: Callable[[], Any] | None = None,
        cancel_callback: Callable[[], Any] | None = None,
    ) -> None:
        self._default_callback = default_callback
        self._restore_callback = restore_callback
        self._ok_callback = ok_callback
        self._cancel_callback = cancel_callback

    def add_semantic_section(
        self,
        group_key: str,
        *,
        title: str | None = None,
        description: str | None = None,
        widget: QWidget | None = None,
        layout: QVBoxLayout | QHBoxLayout | None = None,
    ) -> QVBoxLayout:
        if group_key in self._semantic_sections:
            raise ValueError(f"Semantic section already exists: {group_key}")

        section_title = title or SEMANTIC_GROUP_TITLES.get(group_key, str(group_key))
        nav_button = PushButton(section_title)
        nav_button.setObjectName("semanticNavButton")
        nav_button.setCheckable(True)
        nav_button.setFlat(True)
        nav_button.setCursor(Qt.PointingHandCursor)
        nav_button.clicked.connect(partial(self.scroll_to_semantic_section, group_key))
        self._semantic_nav_buttons[group_key] = nav_button
        self.nav_layout.insertWidget(max(self.nav_layout.count() - 1, 0), nav_button)

        section_widget = QWidget(self.section_container)
        section_widget.setObjectName("semanticSectionCard")
        section_widget.setProperty("semanticGroupKey", group_key)
        section_widget.setAttribute(Qt.WA_StyledBackground, True)
        section_widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        section_layout = QVBoxLayout(section_widget)
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.setSpacing(0)

        header_widget = QWidget(section_widget)
        header_widget.setObjectName("semanticSectionHeader")
        header_widget.setAttribute(Qt.WA_StyledBackground, True)
        header_widget.setCursor(Qt.PointingHandCursor)
        header_widget.mousePressEvent = partial(self._on_section_header_clicked, group_key)
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(14, 10, 14, 10)
        header_layout.setSpacing(4)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(8)
        title_label = Label(section_title)
        title_label.setObjectName("semanticSectionTitle")
        title_label.setAlignment(Qt.AlignLeft)
        indicator_label = Label("v")
        indicator_label.setObjectName("semanticSectionIndicator")
        indicator_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        title_row.addWidget(title_label)
        title_row.addStretch()
        title_row.addWidget(indicator_label)
        header_layout.addLayout(title_row)

        if description:
            description_label = Label(description)
            description_label.setObjectName("semanticSectionDescription")
            description_label.setAlignment(Qt.AlignLeft)
            header_layout.addWidget(description_label)
        section_layout.addWidget(header_widget)

        content_widget = QWidget(section_widget)
        content_widget.setObjectName("semanticSectionContent")
        content_widget.setAttribute(Qt.WA_StyledBackground, True)
        content_widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(14, 12, 14, 14)
        content_layout.setSpacing(10)
        if widget is not None:
            widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
            content_layout.addWidget(widget)
        if layout is not None:
            content_layout.addLayout(layout)
        section_layout.addWidget(content_widget)

        self._semantic_sections[group_key] = section_widget
        self._semantic_section_contents[group_key] = content_widget
        self._semantic_section_indicators[group_key] = indicator_label
        self._semantic_section_collapsed[group_key] = False
        self.section_layout.insertWidget(max(self.section_layout.count() - 1, 0), section_widget)
        self._refresh_section_container_minimum_height()

        if self._active_semantic_group_key is None:
            self._set_active_semantic_group(group_key)
        return content_layout

    def _refresh_section_container_minimum_height(self) -> None:
        self.section_layout.activate()
        self.section_container.setMinimumWidth(0)
        self.section_container.setMinimumHeight(max(0, self.section_container.sizeHint().height()))

    def semantic_group_keys(self) -> list[str]:
        return list(self._semantic_sections.keys())

    def add_or_append_semantic_widget(self, group_key, widget, title=None):
        """Add a semantic section or append a widget to an existing section."""
        if group_key not in self._semantic_sections:
            return self.add_semantic_section(group_key, title=title, widget=widget)
        content_layout = self._semantic_section_contents[group_key].layout()
        widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        content_layout.addWidget(widget)
        self._refresh_section_container_minimum_height()
        return content_layout

    def enable_curve_color_config(self, load_config, threshold_widget=None):
        """Mount curve colors in the display section and optionally bind thresholds."""
        self.curve_color_widget = CurveColorConfigWidget(load_config, self)
        self.curve_color_widget.expanded_changed.connect(self._on_display_config_expanded)
        self.add_or_append_semantic_widget("display", self.curve_color_widget)
        if threshold_widget is not None:
            threshold_widget.bind_curve_color_widget(self.curve_color_widget)
        return self.curve_color_widget

    def _on_display_config_expanded(self, _expanded):
        self._refresh_section_container_minimum_height()

    def enable_plot_view_config(
        self,
        load_config,
        x_unit="",
        y_unit="",
        allow_x=True,
        allow_y=True,
        positive_x=False,
    ):
        """Mount optional plot-view range controls in the display section."""
        self.plot_view_config_widget = PlotViewConfigWidget(
            load_config,
            x_unit,
            y_unit,
            allow_x,
            allow_y,
            positive_x,
            self,
        )
        self.plot_view_config_widget.expanded_changed.connect(self._on_display_config_expanded)
        self.add_or_append_semantic_widget("display", self.plot_view_config_widget)
        if self.width() < PLOT_VIEW_DIALOG_WIDTH:
            self.resize(PLOT_VIEW_DIALOG_WIDTH, self.height())
        return self.plot_view_config_widget

    def merge_plot_view_config(self, config):
        """Merge plot-view values without overwriting curve colors."""
        if self.plot_view_config_widget is None or not self.plot_view_config_widget.should_save():
            return config
        source_config = dict(self.load_config) if isinstance(self.load_config, dict) else {}
        if isinstance(config.get("display"), dict):
            source_config["display"] = config["display"]
        config.update(
            build_plot_view_config(
                source_config,
                self.plot_view_config_widget.plot_view_config(),
            )
        )
        return config

    def validate_plot_view_config(self):
        """Validate enabled custom ranges and show a focused UI warning."""
        if self.plot_view_config_widget is None:
            return True
        error_message = self.plot_view_config_widget.validation_error()
        if error_message is None:
            return True
        MessageBox.warning(self, "设置警告", error_message)
        return False

    def add_threshold_curve_sections(self, threshold_widget, load_config):
        """Opt in to shared curve colors and add the threshold section."""
        self.enable_curve_color_config(load_config, threshold_widget)
        return self.add_semantic_section("judgment", widget=threshold_widget)

    def current_semantic_group_key(self) -> str | None:
        return self._active_semantic_group_key

    def clear_semantic_sections(self) -> None:
        for button in self._semantic_nav_buttons.values():
            self.nav_layout.removeWidget(button)
            button.deleteLater()
        for section in self._semantic_sections.values():
            self.section_layout.removeWidget(section)
            section.deleteLater()
        self._semantic_nav_buttons.clear()
        self._semantic_sections.clear()
        self._semantic_section_contents.clear()
        self._semantic_section_indicators.clear()
        self._semantic_section_collapsed.clear()
        self._active_semantic_group_key = None
        self.curve_color_widget = None
        self.plot_view_config_widget = None
        self._refresh_section_container_minimum_height()

    def is_semantic_section_collapsed(self, group_key: str) -> bool:
        return bool(self._semantic_section_collapsed.get(group_key, False))

    def set_semantic_section_collapsed(self, group_key: str, collapsed: bool) -> None:
        content = self._semantic_section_contents.get(group_key)
        indicator = self._semantic_section_indicators.get(group_key)
        if content is None:
            return
        collapsed = bool(collapsed)
        content.setVisible(not collapsed)
        self._semantic_section_collapsed[group_key] = collapsed
        if indicator is not None:
            indicator.setText(">" if collapsed else "v")
        self._refresh_section_container_minimum_height()

    def toggle_semantic_section(self, group_key: str) -> None:
        self.set_semantic_section_collapsed(group_key, not self.is_semantic_section_collapsed(group_key))

    def _on_section_header_clicked(self, group_key: str, event) -> None:
        self.toggle_semantic_section(group_key)
        if event is not None:
            event.accept()

    def scroll_to_semantic_section(self, group_key: str) -> None:
        section = self._semantic_sections.get(group_key)
        if section is None:
            return
        self._set_active_semantic_group(group_key)
        self.section_scroll_area.ensureWidgetVisible(section, 0, 0)

    def _sync_active_section_from_scroll(self) -> None:
        if self._syncing_scroll or not self._semantic_sections:
            return
        scroll_value = self.section_scroll_area.verticalScrollBar().value()
        current_key = self._active_semantic_group_key
        for group_key, section in self._semantic_sections.items():
            if section.y() <= scroll_value + 12:
                current_key = group_key
            else:
                break
        if current_key is not None:
            self._set_active_semantic_group(current_key)

    def _set_active_semantic_group(self, group_key: str) -> None:
        if group_key not in self._semantic_sections:
            return
        self._syncing_scroll = True
        try:
            self._active_semantic_group_key = group_key
            for key, button in self._semantic_nav_buttons.items():
                button.setChecked(key == group_key)
                button.setProperty("active", key == group_key)
                button.style().unpolish(button)
                button.style().polish(button)
        finally:
            self._syncing_scroll = False

    def _on_default_clicked(self) -> None:
        if self._default_callback is not None:
            self._default_callback()

    def _on_restore_clicked(self) -> None:
        if self._restore_callback is not None:
            self._restore_callback()

    def _on_cancel_clicked(self) -> None:
        if self._cancel_callback is not None:
            self._cancel_callback()
        else:
            self.reject()

    def _on_ok_clicked(self) -> None:
        if self._ok_callback is not None:
            self._ok_callback()
        else:
            self.accept()


class ChannelSelectorWidget(QWidget):
    """Channel selector that preserves the legacy analysis_channel key."""

    def __init__(self, cfg: dict[str, Any] | None = None, available_channels=None, parent=None):
        super().__init__(parent)
        self.available_channels = self._normalize_available_channels(available_channels)
        self.combo_box = ComboBox(self)
        for ch in self.available_channels:
            self.combo_box.addItem(f"In{int(ch) + 1}", int(ch))

        selected = normalize_analysis_channel(cfg, self.available_channels)
        selected_idx = self.combo_box.findData(selected)
        self.combo_box.setCurrentIndex(selected_idx if selected_idx >= 0 else 0)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(Label("通道:"))
        layout.addWidget(self.combo_box)
        layout.addStretch()

    @staticmethod
    def _normalize_available_channels(available_channels) -> list[int]:
        channels = []
        for ch in available_channels or []:
            try:
                channels.append(int(ch))
            except (TypeError, ValueError):
                continue
        channels = sorted(set(channels))
        return channels or [0]

    def current_channel(self) -> int:
        return int(self.combo_box.currentData())

    def get_config(self) -> dict[str, int]:
        return {"analysis_channel": self.current_channel()}


class WeightingSelectorWidget(QWidget):
    """Weighting selector that displays Z as Z(None) but saves canonical values."""

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        allowed_options: tuple[str, ...] | list[str] = WEIGHTING_OPTIONS,
        default: str = "Z",
        parent=None,
    ):
        super().__init__(parent)
        self.allowed_options = tuple(normalize_weighting(option) for option in allowed_options)
        if not self.allowed_options:
            self.allowed_options = ("Z",)
        normalized_default = normalize_weighting(default)
        if normalized_default not in self.allowed_options:
            normalized_default = self.allowed_options[0]

        self.combo_box = ComboBox(self)
        for option in self.allowed_options:
            self.combo_box.addItem(weighting_to_display_label(option), option)

        selected = normalize_weighting((cfg or {}).get("weighting"), default=normalized_default)
        if selected not in self.allowed_options:
            selected = normalized_default
        selected_idx = self.combo_box.findData(selected)
        self.combo_box.setCurrentIndex(selected_idx if selected_idx >= 0 else 0)
        self.combo_box.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(Label("计权方式:"))
        layout.addWidget(self.combo_box)
        layout.addStretch()

    def current_weighting(self) -> str:
        return normalize_weighting(self.combo_box.currentData())

    def get_config(self) -> dict[str, str]:
        return {"weighting": self.current_weighting()}


class HarmonicDetectionMethodSelectorWidget(QWidget):
    """Standard HD/RB detection method selector."""

    def __init__(self, cfg: dict[str, Any] | None = None, parent=None):
        super().__init__(parent)
        self.combo_box = ComboBox(self)
        for method, label in HARMONIC_DETECTION_METHOD_LABELS.items():
            self.combo_box.addItem(label, method)

        selected = normalize_harmonic_detection_method((cfg or {}).get(HARMONIC_DETECTION_METHOD_KEY))
        selected_idx = self.combo_box.findData(selected)
        self.combo_box.setCurrentIndex(selected_idx if selected_idx >= 0 else 0)
        self.combo_box.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(Label("检测算法:"))
        layout.addWidget(self.combo_box)
        layout.addStretch()

    def current_method(self) -> str:
        return normalize_harmonic_detection_method(self.combo_box.currentData())

    def get_config(self) -> dict[str, str]:
        return {HARMONIC_DETECTION_METHOD_KEY: self.current_method()}


class OctaveSmoothingSelectorWidget(QWidget):
    """Frequency-domain octave smoothing selector."""

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        allowed_options: tuple[int, ...] | list[int] = OCTAVE_SMOOTHING_OPTIONS,
        default: int = 0,
        legacy_true_default: int = 6,
        parent=None,
    ):
        super().__init__(parent)
        self.allowed_options = tuple(option for option in allowed_options if option in OCTAVE_SMOOTHING_OPTIONS)
        if not self.allowed_options:
            self.allowed_options = (0,)
        if default not in self.allowed_options:
            default = self.allowed_options[0]

        self.combo_box = ComboBox(self)
        for option in self.allowed_options:
            self.combo_box.addItem(OCTAVE_SMOOTHING_LABELS[option], option)

        selected = normalize_octave_smoothing(cfg, default=default, legacy_true_default=legacy_true_default)
        if selected not in self.allowed_options:
            selected = default
        selected_idx = self.combo_box.findData(selected)
        self.combo_box.setCurrentIndex(selected_idx if selected_idx >= 0 else 0)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(Label("平滑"))
        layout.addWidget(self.combo_box)
        layout.addStretch()

    def current_octave_smoothing(self) -> int:
        return int(self.combo_box.currentData())

    def get_config(self) -> dict[str, int]:
        return {"octave_smoothing": self.current_octave_smoothing()}


class TimeSmoothingWidget(QWidget):
    """Time or point smoothing control for event-detection style dialogs."""

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        defaults: dict[str, Any] | None = None,
        show_algorithm: bool = True,
        min_points: int = 1,
        parent=None,
    ):
        super().__init__(parent)
        self.show_algorithm = show_algorithm
        self.min_points = int(min_points)
        self.control_widgets: list[QWidget] = []
        smoothing = normalize_time_smoothing(cfg, defaults)

        self.enabled_checkbox = CheckBox("平滑")
        self.enabled_checkbox.setChecked(smoothing["enabled"])
        self.enabled_checkbox.stateChanged.connect(self._sync_control_enabled)

        self.unit_combo = ComboBox(self)
        self.unit_combo.addItem("时间(秒)", "time")
        self.unit_combo.addItem("格点数", "points")
        unit_idx = self.unit_combo.findData(smoothing["unit"])
        self.unit_combo.setCurrentIndex(unit_idx if unit_idx >= 0 else 0)
        self.unit_combo.currentIndexChanged.connect(self._update_unit_visibility)
        self.control_widgets.append(self.unit_combo)

        self.time_spin = DoubleSpinBox(self)
        self.time_spin.setRange(0.00, 999.00)
        self.time_spin.setDecimals(4)
        self.time_spin.setSingleStep(0.01)
        self.time_spin.setValue(float(smoothing["time_sec"]))
        self.time_spin.setMaximumWidth(120)
        self.control_widgets.append(self.time_spin)

        self.points_spin = SpinBox(self)
        self.points_spin.setRange(self.min_points, 99999)
        self.points_spin.setValue(max(self.min_points, int(smoothing["points"])))
        self.points_spin.setMaximumWidth(120)
        self.unit_combo.setMaximumWidth(120)
        self.control_widgets.append(self.points_spin)

        main_row = QHBoxLayout()
        main_row.addWidget(self.enabled_checkbox)

        unit_row = QHBoxLayout()
        unit_row.addWidget(Label("单位:"))
        unit_row.addWidget(self.unit_combo)
        unit_row.addWidget(self.time_spin)
        unit_row.addWidget(self.points_spin)
        unit_row.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(main_row)
        layout.addLayout(unit_row)

        self.algo_group = None
        if show_algorithm:
            self.algo_group = QButtonGroup(self)
            algo_row = QHBoxLayout()
            algo_row.addWidget(Label("平滑算法:"))
            algo_buttons = (
                (RadioButton("平均"), 1),
                (RadioButton("Golay"), 2),
                (RadioButton("Gaussian"), 3),
            )
            for button, value in algo_buttons:
                self.algo_group.addButton(button, value)
                algo_row.addWidget(button)
                self.control_widgets.append(button)
                if int(smoothing["algo"]) == value:
                    button.setChecked(True)
            if self.algo_group.checkedId() < 0:
                self.algo_group.button(1).setChecked(True)
            layout.addLayout(algo_row)

        self._update_unit_visibility()
        self._sync_control_enabled()

    def _update_unit_visibility(self) -> None:
        is_time = self.unit_combo.currentData() == "time"
        self.time_spin.setVisible(is_time)
        self.points_spin.setVisible(not is_time)

    def set_smoothing_enabled(self, enabled: bool) -> None:
        self.enabled_checkbox.setChecked(bool(enabled))
        self._sync_control_enabled()

    def _sync_control_enabled(self, *args) -> None:
        enabled = self.enabled_checkbox.isChecked()
        for widget in self.control_widgets:
            widget.setEnabled(enabled)

    def get_config(self) -> dict[str, Any]:
        config = {
            "smooth_enabled": self.enabled_checkbox.isChecked(),
            "smooth_unit": str(self.unit_combo.currentData()),
            "smooth_time_sec": float(self.time_spin.value()),
            "smooth_points": int(self.points_spin.value()),
        }
        if self.show_algorithm:
            config["smooth_algo"] = int(self.algo_group.checkedId() or 1)
        return config


class GoldenSampleWidget(QWidget):
    """Golden-sample enable switch and display-mode selector."""

    DISPLAY_MODES = {
        GOLDEN_SAMPLE_DISPLAY_DEVIATION: "偏差曲线（测试 - 黄金）",
        GOLDEN_SAMPLE_DISPLAY_ENVELOPE: "测试曲线 + 黄金样本上下框线",
    }

    def __init__(self, cfg: dict[str, Any] | None = None, parent=None):
        super().__init__(parent)
        config = cfg or {}

        self.enabled_checkbox = CheckBox("使用黄金样本", self)
        self.enabled_checkbox.setChecked(bool(config.get(GOLDEN_SAMPLE_CHECKED_KEY, False)))
        self.enabled_checkbox.stateChanged.connect(self._sync_display_mode_enabled)

        self.display_mode_combo = ComboBox(self)
        for value, label in self.DISPLAY_MODES.items():
            self.display_mode_combo.addItem(label, value)
        self.display_mode_combo.setToolTip(
            "偏差曲线模式：上下限为偏差曲线的最终范围；"
            "黄金样本上下框线模式：上下限均为相对黄金样本曲线的带符号偏移量；"
            "上框线 = 黄金样本曲线 + 上限值；"
            "下框线 = 黄金样本曲线 + 下限值。"
        )
        saved_mode = str(
            config.get(
                GOLDEN_SAMPLE_DISPLAY_MODE_KEY,
                DEFAULT_GOLDEN_SAMPLE_DISPLAY_MODE,
            )
            or DEFAULT_GOLDEN_SAMPLE_DISPLAY_MODE
        ).lower()
        mode_index = self.display_mode_combo.findData(saved_mode)
        self.display_mode_combo.setCurrentIndex(mode_index if mode_index >= 0 else 0)

        mode_row = QHBoxLayout()
        mode_row.addWidget(Label("图形显示方式：", self))
        mode_row.addWidget(self.display_mode_combo, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.enabled_checkbox)
        layout.addLayout(mode_row)
        self._sync_display_mode_enabled()

    def _sync_display_mode_enabled(self, *args) -> None:
        self.display_mode_combo.setEnabled(self.enabled_checkbox.isChecked())

    def is_checked(self) -> bool:
        return self.enabled_checkbox.isChecked()

    def display_mode(self) -> str:
        mode = str(self.display_mode_combo.currentData() or DEFAULT_GOLDEN_SAMPLE_DISPLAY_MODE)
        return mode if mode in self.DISPLAY_MODES else DEFAULT_GOLDEN_SAMPLE_DISPLAY_MODE

    def limit_value_semantics(self) -> str:
        if self.is_checked() and self.display_mode() == GOLDEN_SAMPLE_DISPLAY_ENVELOPE:
            return LIMIT_VALUE_SEMANTICS_OFFSET
        return LIMIT_VALUE_SEMANTICS_BOUNDS

    def get_config(self) -> dict[str, Any]:
        return {
            GOLDEN_SAMPLE_CHECKED_KEY: self.is_checked(),
            GOLDEN_SAMPLE_DISPLAY_MODE_KEY: self.display_mode(),
        }


class HarmonicSelectorWidget(QWidget):
    """Scrollable harmonic order selector with optional all-selected state."""

    selected_labels_changed = pyqtSignal()

    def __init__(self, cfg: dict[str, Any] | None = None, start_order: int = 2, end_order: int = 35, parent=None):
        super().__init__(parent)
        self.start_order = int(start_order)
        self.end_order = int(end_order)
        if self.end_order < self.start_order:
            self.start_order, self.end_order = self.end_order, self.start_order

        self._selected_labels = self._load_selected_labels(cfg)
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setFixedSize(120, 150)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        box_container = QWidget()
        self.box_layout = QVBoxLayout()
        for order in self._all_orders():
            label = Label("  " + str(order))
            label.setMinimumWidth(90)
            label.setMinimumHeight(25)
            label.setAlignment(Qt.AlignLeft)
            label.mousePressEvent = partial(self._on_label_click, label)
            if order in self._selected_labels:
                label.setText("\u2713" + str(order))
            self.box_layout.addWidget(label)
        box_container.setLayout(self.box_layout)
        self.scroll_area.setWidget(box_container)

        self.select_all_check = CheckBox("全选")
        self.select_all_check.setChecked(bool((cfg or {}).get("all_checked", False)))
        self.select_all_check.stateChanged.connect(self._on_select_all_changed)
        if self.select_all_check.isChecked():
            self._selected_labels = self._all_orders()
            self._sync_labels()
            self.scroll_area.setDisabled(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.scroll_area)
        layout.addStretch()
        layout.addWidget(self.select_all_check)

    def _all_orders(self) -> list[int]:
        return list(range(self.start_order, self.end_order + 1))

    def _load_selected_labels(self, cfg: dict[str, Any] | None) -> list[int]:
        selected = []
        for value in (cfg or {}).get("selected_labels", []):
            try:
                order = int(value)
            except (TypeError, ValueError):
                continue
            if self.start_order <= order <= self.end_order:
                selected.append(order)
        return sorted(set(selected))

    def _on_select_all_changed(self, state) -> None:
        if state == Qt.Checked:
            self.scroll_area.setDisabled(True)
            self._selected_labels = self._all_orders()
        else:
            self.scroll_area.setDisabled(False)
            self._selected_labels = []
        self._sync_labels()
        self.selected_labels_changed.emit()

    def _on_label_click(self, label, event) -> None:
        order = int("".join(filter(str.isdigit, label.text())))
        if order in self._selected_labels:
            self._selected_labels.remove(order)
        else:
            self._selected_labels.append(order)
            self._selected_labels.sort()
        self._sync_labels()
        self.selected_labels_changed.emit()

    def _sync_labels(self) -> None:
        selected = set(self._selected_labels)
        for i in range(self.box_layout.count()):
            label = self.box_layout.itemAt(i).widget()
            order = int("".join(filter(str.isdigit, label.text())))
            label.setText(("\u2713" if order in selected else "  ") + str(order))

    def selected_labels(self) -> list[int]:
        return list(self._selected_labels)

    def all_checked(self) -> bool:
        return self.select_all_check.isChecked()

    def get_config(self) -> dict[str, Any]:
        return {
            "selected_labels": self.selected_labels(),
            "all_checked": self.all_checked(),
        }


class AnalysisTimeRangeWidget(QWidget):
    """Reusable analysis time range selector.

    Config keys produced:
      - ``analysis_time_range_enabled``  (bool)
      - ``analysis_start_time_sec``      (float, seconds)
      - ``analysis_end_time_sec``        (float, seconds; 0 = until end)
    """

    def __init__(self, cfg: dict | None = None, parent=None, show_checkbox: bool = False):
        super().__init__(parent)
        config = cfg or {}
        self.enabled_checkbox = CheckBox("限制分析时间范围")
        self.enabled_checkbox.setChecked(bool(config.get("analysis_time_range_enabled", False)))
        self.enabled_checkbox.stateChanged.connect(self._update_enabled)

        self.start_spin = DoubleSpinBox(self)
        self.start_spin.setRange(0.0, 999999.0)
        self.start_spin.setDecimals(4)
        self.start_spin.setSingleStep(0.1)
        self.start_spin.setValue(float(config.get("analysis_start_time_sec", 0.0) or 0.0))
        self.start_spin.setMaximumWidth(120)

        self.end_spin = DoubleSpinBox(self)
        self.end_spin.setRange(0.0, 999999.0)
        self.end_spin.setDecimals(4)
        self.end_spin.setSingleStep(0.1)
        self.end_spin.setValue(float(config.get("analysis_end_time_sec", 0.0) or 0.0))
        self.end_spin.setMaximumWidth(120)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if show_checkbox:
            layout.addWidget(self.enabled_checkbox)
        else:
            self.enabled_checkbox.setVisible(False)
        value_row = QHBoxLayout()
        value_row.addWidget(Label("起始(s):"))
        value_row.addWidget(self.start_spin)
        value_row.addWidget(Label("结束(s):"))
        value_row.addWidget(self.end_spin)
        value_row.addStretch()
        layout.addLayout(value_row)
        self._update_enabled()

    def _update_enabled(self, *args) -> None:
        enabled = self.enabled_checkbox.isChecked()
        self.start_spin.setEnabled(enabled)
        self.end_spin.setEnabled(enabled)

    def set_range_enabled(self, enabled: bool) -> None:
        self.enabled_checkbox.setChecked(bool(enabled))
        self._update_enabled()

    def get_config(self) -> dict:
        return {
            "analysis_time_range_enabled": self.enabled_checkbox.isChecked(),
            "analysis_start_time_sec": float(self.start_spin.value()),
            "analysis_end_time_sec": float(self.end_spin.value()),
        }
