"""Shared widgets for analysis configuration dialogs."""

from __future__ import annotations

from functools import partial
from typing import Any

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QButtonGroup,
    QDialog,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.custom_ui_widget.popuputils import PopupUtils
from ui.custom_ui_widget.widgets import CheckBox, ComboBox, DoubleSpinBox, Label, PushButton, RadioButton, SpinBox
from ui.ui_analysis_config.config_normalization import (
    OCTAVE_SMOOTHING_OPTIONS,
    WEIGHTING_OPTIONS,
    normalize_analysis_channel,
    normalize_octave_smoothing,
    normalize_time_smoothing,
    normalize_weighting,
    weighting_to_display_label,
)
from ui.ui_src import ui_resources


OCTAVE_SMOOTHING_LABELS = {
    0: "不平滑",
    1: "1/1 Oct",
    3: "1/3 Oct",
    6: "1/6 Oct",
    12: "1/12 Oct",
    24: "1/24 Oct",
    48: "1/48 Oct",
}


class AnalysisConfigDialogBase(QDialog):
    """Base dialog helpers for analysis configuration windows."""

    def __init__(self, *args, disable_close_button: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_standard_window_flags(disable_close_button=disable_close_button)

    def apply_standard_window_flags(self, disable_close_button: bool = True) -> None:
        if disable_close_button:
            self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))

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
        smoothing = normalize_time_smoothing(cfg, defaults)

        self.enabled_checkbox = CheckBox("平滑")
        self.enabled_checkbox.setChecked(smoothing["enabled"])

        self.unit_combo = ComboBox(self)
        self.unit_combo.addItem("时间(秒)", "time")
        self.unit_combo.addItem("格点数", "points")
        unit_idx = self.unit_combo.findData(smoothing["unit"])
        self.unit_combo.setCurrentIndex(unit_idx if unit_idx >= 0 else 0)
        self.unit_combo.currentIndexChanged.connect(self._update_unit_visibility)

        self.time_spin = DoubleSpinBox(self)
        self.time_spin.setRange(0.00, 999.00)
        self.time_spin.setDecimals(4)
        self.time_spin.setSingleStep(0.01)
        self.time_spin.setValue(float(smoothing["time_sec"]))

        self.points_spin = SpinBox(self)
        self.points_spin.setRange(self.min_points, 99999)
        self.points_spin.setValue(max(self.min_points, int(smoothing["points"])))

        main_row = QHBoxLayout()
        main_row.addWidget(self.enabled_checkbox)
        main_row.addStretch()
        main_row.addWidget(Label("单位:"))
        main_row.addWidget(self.unit_combo)
        main_row.addWidget(self.time_spin)
        main_row.addWidget(self.points_spin)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(main_row)

        self.algo_group = None
        if show_algorithm:
            self.algo_group = QButtonGroup(self)
            algo_row = QHBoxLayout()
            algo_row.addStretch()
            algo_row.addWidget(Label("平滑算法:"))
            algo_buttons = (
                (RadioButton("平均"), 1),
                (RadioButton("Golay"), 2),
                (RadioButton("Gaussian"), 3),
            )
            for button, value in algo_buttons:
                self.algo_group.addButton(button, value)
                algo_row.addWidget(button)
                if int(smoothing["algo"]) == value:
                    button.setChecked(True)
            if self.algo_group.checkedId() < 0:
                self.algo_group.button(1).setChecked(True)
            layout.addLayout(algo_row)

        self._update_unit_visibility()

    def _update_unit_visibility(self) -> None:
        is_time = self.unit_combo.currentData() == "time"
        self.time_spin.setVisible(is_time)
        self.points_spin.setVisible(not is_time)

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


class GoldenSampleWidget(CheckBox):
    """Golden sample checkbox that preserves golden_sample_checked."""

    def __init__(self, cfg: dict[str, Any] | None = None, parent=None):
        super().__init__("使用黄金样本", parent)
        self.setChecked(bool((cfg or {}).get("golden_sample_checked", False)))

    def is_checked(self) -> bool:
        return self.isChecked()

    def get_config(self) -> dict[str, bool]:
        return {"golden_sample_checked": self.is_checked()}


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
