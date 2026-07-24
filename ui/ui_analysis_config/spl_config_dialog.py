"""
SPL (Sound Pressure Level) 分析配置对话框
"""

import re
from typing import List, Optional

from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from ui.custom_ui_widget.popuputils import PopupUtils
from ui.custom_ui_widget.widgets import CheckBox, ComboBox, DoubleSpinBox, GroupBox, Label, RadioButton, SpinBox
from ui.ui_analysis_config.common_widgets import (
    AnalysisTimeRangeWidget,
    ChannelSelectorWidget,
    GoldenSampleWidget,
    OctaveSmoothingSelectorWidget,
    SemanticAnalysisConfigDialogBase,
    TimeSmoothingWidget,
    WeightingSelectorWidget,
)
from ui.ui_analysis_config.threshold_config_widget import ThresholdConfigWidget


class ConfigSubsectionWidget(GroupBox):
    """Small bordered subsection for related analysis configuration controls."""

    def __init__(self, title: str, content_widget: QWidget, checkable: bool = False, checked: bool = True, parent=None):
        super().__init__(title, parent)
        self.content_widget = content_widget
        self.setCheckable(checkable)
        if checkable:
            self.setChecked(bool(checked))
            self.toggled.connect(self._sync_content_enabled)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(content_widget)
        self._sync_content_enabled()

    def _sync_content_enabled(self, *args) -> None:
        self.content_widget.setEnabled(self.isChecked() if self.isCheckable() else True)


class SplWindowLengthWidget(QWidget):
    """SPL calculation window length control preserving the legacy fixed window."""

    def __init__(self, cfg: dict | None = None, parent=None):
        super().__init__(parent)
        config = cfg or {}
        self.unit_combo = ComboBox(self)
        self.unit_combo.addItem("时间(秒)", "time")
        self.unit_combo.addItem("格点数", "points")
        self.unit_combo.setMaximumWidth(120)
        unit = str(config.get("spl_window_unit", "points") or "points").lower()
        unit_idx = self.unit_combo.findData(unit if unit in ("time", "points") else "points")
        self.unit_combo.setCurrentIndex(unit_idx if unit_idx >= 0 else 1)
        self.unit_combo.currentIndexChanged.connect(self._update_unit_visibility)

        self.time_spin = DoubleSpinBox(self)
        self.time_spin.setRange(0.0001, 999.0000)
        self.time_spin.setDecimals(4)
        self.time_spin.setSingleStep(0.001)
        self.time_spin.setValue(float(config.get("spl_window_time_sec", 0.0272) or 0.0272))
        self.time_spin.setMaximumWidth(120)

        self.points_spin = SpinBox(self)
        self.points_spin.setRange(1, 999999)
        self.points_spin.setValue(int(config.get("spl_window_points", 1201) or 1201))
        self.points_spin.setMaximumWidth(120)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        value_row = QHBoxLayout()
        value_row.addWidget(Label("单位:"))
        value_row.addWidget(self.unit_combo)
        value_row.addWidget(self.time_spin)
        value_row.addWidget(self.points_spin)
        value_row.addStretch()
        layout.addLayout(value_row)
        self._update_unit_visibility()

    def _update_unit_visibility(self) -> None:
        is_time = self.unit_combo.currentData() == "time"
        self.time_spin.setVisible(is_time)
        self.points_spin.setVisible(not is_time)

    def get_config(self) -> dict:
        return {
            "spl_window_unit": str(self.unit_combo.currentData()),
            "spl_window_time_sec": float(self.time_spin.value()),
            "spl_window_points": int(self.points_spin.value()),
        }


class SplConfigWindow(SemanticAnalysisConfigDialogBase):
    """SPL 分析配置对话框"""

    def __init__(self, config_manager, model_type, available_channels: Optional[List[int]] = None):
        super().__init__(disable_close_button=True)
        self.config_manager = config_manager
        self.config_key = model_type
        self.load_config = self.config_manager.load_config().get(self.config_key, {})
        self.model_type = "".join(re.findall(r"[A-Za-z]", str(model_type))) or "SPL"
        self.show_channel_selector = available_channels is not None
        self.available_channels = available_channels
        self.init_ui()

    def init_ui(self):
        self.apply_semantic_dialog_size()
        self.set_semantic_button_callbacks(
            default_callback=self.on_default_btn_clicked,
            restore_callback=self.on_restore_default_btn_clicked,
            ok_callback=self.on_click_ok_btn,
        )
        self._build_semantic_sections()

    def _build_semantic_sections(self):
        self.spl_window_widget = None
        self.time_smoothing_widget = None
        self.analysis_time_range_widget = None
        if self.model_type != "SPLF":
            self.spl_window_widget = SplWindowLengthWidget(self.load_config, self)
            self.time_smoothing_widget = TimeSmoothingWidget(
                self.load_config,
                defaults={
                    "enabled": bool(self.load_config.get("smooth_checked", True)),
                    "unit": "points",
                    "time_sec": 0.0250,
                    "points": 1102,
                    "algo": 2,
                },
                parent=self,
            )
            self.time_smoothing_widget.enabled_checkbox.setVisible(False)
            self.analysis_time_range_widget = AnalysisTimeRangeWidget(self.load_config, self)

        # SPLF: calculation mode (fundamental-only vs total RMS SPL)
        self.splf_mode_group_box = None
        if self.model_type == "SPLF":
            self.splf_mode_group_box = GroupBox("SPLF 计算方式")
            self.radio_fundamental = RadioButton("仅基频")
            self.radio_total = RadioButton("总SPL")

            mode = str(self.load_config.get("splf_calc_mode", "fundamental") or "fundamental").lower()
            if mode == "total":
                self.radio_total.setChecked(True)
            else:
                self.radio_fundamental.setChecked(True)

            mode_layout = QVBoxLayout()
            mode_layout.addWidget(self.radio_fundamental)
            mode_layout.addWidget(self.radio_total)
            self.splf_mode_group_box.setLayout(mode_layout)

            self.smoothing_selector = OctaveSmoothingSelectorWidget(self.load_config, parent=self)

        # SPLF only: golden sample checkbox (placed above threshold widget)
        self.golden_chk_box = None
        if self.model_type == "SPLF":
            self.golden_chk_box = GoldenSampleWidget(self.load_config, self)

        self.threshold_widget = ThresholdConfigWidget(
            parent=self,
            load_config=self.load_config,
            model_type=self.model_type,
            allow_manual_limits=True,
            limit_value_semantics_provider=(
                self.golden_chk_box.limit_value_semantics
                if self.golden_chk_box is not None
                else None
            ),
        )

        self.weighting_selector = WeightingSelectorWidget(self.load_config, parent=self)
        self.show_overall_spl_box = None
        if self.model_type != "SPLF":
            self.show_overall_spl_box = CheckBox("显示总体声压级", self)
            self.show_overall_spl_box.setChecked(bool(self.load_config.get("show_overall_spl", False)))

        if self.show_channel_selector:
            self.channel_selector = ChannelSelectorWidget(self.load_config, self.available_channels, self)
            self.add_semantic_section("input", widget=self.channel_selector)

        if self.model_type != "SPLF":
            preprocess_widget = QWidget(self)
            preprocess_layout = QVBoxLayout(preprocess_widget)
            preprocess_layout.setContentsMargins(0, 0, 0, 0)
            preprocess_layout.setSpacing(10)

            preprocess_layout.addWidget(ConfigSubsectionWidget("SPL计算窗长", self.spl_window_widget, parent=self))

            self.smoothing_section = ConfigSubsectionWidget(
                "平滑",
                self.time_smoothing_widget,
                checkable=True,
                checked=self.time_smoothing_widget.enabled_checkbox.isChecked(),
                parent=self,
            )
            self.smoothing_section.toggled.connect(self.time_smoothing_widget.set_smoothing_enabled)
            preprocess_layout.addWidget(self.smoothing_section)

            self.analysis_time_range_section = ConfigSubsectionWidget(
                "限制分析时间范围",
                self.analysis_time_range_widget,
                checkable=True,
                checked=self.analysis_time_range_widget.enabled_checkbox.isChecked(),
                parent=self,
            )
            self.analysis_time_range_section.toggled.connect(self.analysis_time_range_widget.set_range_enabled)
            preprocess_layout.addWidget(self.analysis_time_range_section)
            self.add_semantic_section("preprocess", widget=preprocess_widget)

        compute_widget = QWidget(self)
        compute_layout = QVBoxLayout(compute_widget)
        compute_layout.setContentsMargins(0, 0, 0, 0)
        compute_layout.setSpacing(8)
        compute_layout.addWidget(self.weighting_selector)
        if self.show_overall_spl_box is not None:
            compute_layout.addWidget(self.show_overall_spl_box)
        if self.splf_mode_group_box is not None:
            compute_layout.addWidget(self.splf_mode_group_box)
            compute_layout.addWidget(self.smoothing_selector)
        self.add_semantic_section("compute", widget=compute_widget)

        if self.golden_chk_box is not None:
            self.add_semantic_section("reference", widget=self.golden_chk_box)
        self.add_threshold_curve_sections(self.threshold_widget, self.load_config)
        if self.model_type == "SPLF":
            self.enable_plot_view_config(self.load_config, "Hz", "dB", True, True, True)
        else:
            self.enable_plot_view_config(self.load_config, "s", "dB", True, True, False)

    def create_btn(self):
        return self.create_standard_button_layout(self.on_default_btn_clicked, self.on_click_ok_btn)

    def get_default_config(self):
        """获取配置数据"""
        config = {}
        if self.model_type != "SPLF":
            if self.spl_window_widget is not None:
                config.update(self.spl_window_widget.get_config())
            if self.time_smoothing_widget is not None:
                smoothing_config = self.time_smoothing_widget.get_config()
                config.update(smoothing_config)
                config["smooth_checked"] = bool(smoothing_config["smooth_enabled"])
            if self.analysis_time_range_widget is not None:
                config.update(self.analysis_time_range_widget.get_config())
            if self.show_overall_spl_box is not None:
                config["show_overall_spl"] = self.show_overall_spl_box.isChecked()
        if self.model_type == "SPLF":
            calc_mode = "fundamental"
            if hasattr(self, "radio_total") and self.radio_total.isChecked():
                calc_mode = "total"
            config["splf_calc_mode"] = calc_mode
            config.update(self.smoothing_selector.get_config())
            if self.golden_chk_box is not None:
                config.update(self.golden_chk_box.get_config())
        config.update(self.threshold_widget.get_config())
        config.update(self.weighting_selector.get_config())
        if self.show_channel_selector and hasattr(self, "channel_selector"):
            config.update(self.channel_selector.get_config())
        else:
            config["analysis_channel"] = int(self.load_config.get("analysis_channel", 0) or 0)
        return self.merge_plot_view_config(config)

    def on_default_btn_clicked(self):
        if not self.validate_plot_view_config():
            return
        if not self.threshold_widget.validate():
            return
        config_data = self.get_default_config()
        save_flag = self.config_manager.save_default_config(self.model_type, config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_restore_default_btn_clicked(self):
        self.load_config = self.config_manager.load_config().get(self.config_key, {})
        self.clear_semantic_sections()
        self._build_semantic_sections()

    def on_click_ok_btn(self):
        if not self.validate_plot_view_config():
            return
        if not self.threshold_widget.validate():
            return
        config_data = self.get_default_config()
        self.accept()
        return config_data
