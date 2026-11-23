"""Configuration for sharpness and roughness using the existing semantic controls."""
from copy import deepcopy

from PyQt5.QtWidgets import QFormLayout, QVBoxLayout, QWidget

from base.analysis_config_validation import sharpness_execution_issue
from consts.acoustic_analysis.common_consts import (
    SOUND_QUALITY_ANALYSES, SHARPNESS_METHOD, SHARPNESS_OUTPUT_TIME_RESOLUTION_MS,
)
from ui.custom_ui_widget.popuputils import PopupUtils
from ui.custom_ui_widget.widgets import CheckBox, ComboBox, DoubleSpinBox, MessageBox
from ui.ui_analysis_config.common_widgets import (
    AnalysisTimeRangeWidget, ChannelSelectorWidget, MultiChannelSelectorWidget,
    SemanticAnalysisConfigDialogBase,
)
from ui.ui_analysis_config.manual_limit_segments import ManualLimitValidationError, validate_constant_limit_config
from ui.ui_analysis_config.threshold_config_widget import ThresholdConfigWidget


class SoundQualityConfigWindow(SemanticAnalysisConfigDialogBase):
    """One editor for the two algorithms; no dependency on their numerical modules."""

    @staticmethod
    def merge_with_defaults(kind, config):
        defaults = {
            "enabled": True, "analysis_channel": 0, "advanced": {},
            "method": SHARPNESS_METHOD if kind == "SHRP" else "daniel_weber",
            "limit_checked": False, "limit_metric": "curve_y", "limit_mode": "manual",
            "manual_input_mode": "constant", "constant_upper_enabled": True,
            "constant_upper_value": 3.0, "constant_lower_enabled": False,
            "constant_lower_value": 0.0, "scalar_upper_enabled": True,
            "scalar_upper_value": 3.0, "scalar_lower_enabled": False, "scalar_lower_value": 0.0,
        }
        defaults.update(deepcopy(config or {}))
        return defaults

    def __init__(self, config_manager, model_type, analysis_type, available_channels=None,
                 allow_multiple_channels=False):
        super().__init__(disable_close_button=True)
        self.config_manager, self.model_type = config_manager, model_type
        self.analysis_type = analysis_type
        self.spec = SOUND_QUALITY_ANALYSES[analysis_type]
        self.available_channels, self.allow_multiple_channels = available_channels, allow_multiple_channels
        self.load_config = self.merge_with_defaults(analysis_type, config_manager.load_config().get(model_type, {}))
        self.setWindowTitle(self.spec["label"] + "分析配置")
        self.apply_semantic_dialog_size()
        self.set_semantic_button_callbacks(default_callback=self.on_default_btn_clicked,
                                          restore_callback=self.on_restore_default_btn_clicked,
                                          ok_callback=self.on_click_ok_btn)
        self._build_sections()

    @staticmethod
    def _spin(value, minimum=0.0, maximum=1000000.0, decimals=4):
        spin = DoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(minimum, maximum)
        spin.setValue(float(value))
        spin.setMaximumWidth(150)
        return spin

    def _build_sections(self):
        config = self.load_config
        if self.available_channels is not None:
            selector = MultiChannelSelectorWidget if self.allow_multiple_channels else ChannelSelectorWidget
            self.channel_selector = selector(config, self.available_channels, self)
            self.add_semantic_section("input", title="输入参数", widget=self.channel_selector)
        compute = QWidget()
        form = QFormLayout(compute)
        advanced = config.get("advanced") or {}
        self.time_range = AnalysisTimeRangeWidget(advanced, self, show_checkbox=True)
        form.addRow(self.time_range)
        if self.analysis_type == "ROUGH":
            self.method_combo = ComboBox()
            self.method_combo.addItem("Daniel–Weber", "daniel_weber")
            if config["method"] != "daniel_weber":
                self.method_combo.addItem("不支持的模式：" + str(config["method"]), config["method"])
                self.method_combo.setCurrentIndex(1)
            form.addRow("计算模式", self.method_combo)
            self.overlap_spin = self._spin(config.get("overlap", 0.5), 0.0, 0.9999)
            self.overlap_spin.setToolTip("步长 = 200 ms × 系数；0 使用不重叠的 200 ms 帧。0.5 对应 100 ms 步长。")
            form.addRow("帧步长系数", self.overlap_spin)
        self.add_semantic_section("compute", title="算法参数", widget=compute)
        judgment = QWidget()
        layout = QVBoxLayout(judgment)
        self.limit_enabled = CheckBox("启用阈值判定")
        self.limit_enabled.setChecked(config["limit_checked"])
        layout.addWidget(self.limit_enabled)
        self.metric_combo = ComboBox()
        self.metric_combo.addItem("时间曲线", "curve_y")
        for key, label in self.spec["metrics"]:
            self.metric_combo.addItem(label, key)
        self.metric_combo.setCurrentIndex(max(0, self.metric_combo.findData(config["limit_metric"])))
        layout.addWidget(self.metric_combo)
        self.threshold_widget = ThresholdConfigWidget(
            self, config, model_type=self.analysis_type, allow_manual_limits=True,
            allow_constant_limits=True, constant_limit_unit=self.spec["unit"],
        )
        self.threshold_widget.limit_checkbox.hide()
        layout.addWidget(self.threshold_widget)
        self.scalar_widget = QWidget()
        scalar_layout = QFormLayout(self.scalar_widget)
        self.scalar_controls = {}
        for side, label in (("upper", "上限"), ("lower", "下限")):
            enabled = CheckBox(label + " (" + self.spec["unit"] + ")")
            enabled.setChecked(config[f"scalar_{side}_enabled"])
            spin = self._spin(config[f"scalar_{side}_value"])
            enabled.toggled.connect(spin.setEnabled)
            spin.setEnabled(enabled.isChecked())
            scalar_layout.addRow(enabled, spin)
            self.scalar_controls[side] = enabled, spin
        layout.addWidget(self.scalar_widget)
        self.enable_curve_color_config(config, self.threshold_widget)
        self.enable_plot_view_config(config, x_unit="s", y_unit=self.spec["unit"])
        self.add_semantic_section("judgment", title="判定阈值", widget=judgment)
        self.metric_combo.currentIndexChanged.connect(self._update_limits)
        self.limit_enabled.toggled.connect(self._update_limits)
        self._update_limits()

    def _update_limits(self, *_):
        curve = self.metric_combo.currentData() == "curve_y"
        enabled = self.limit_enabled.isChecked()
        self.metric_combo.setEnabled(enabled)
        self.threshold_widget.setVisible(curve)
        self.threshold_widget.limit_checkbox.setChecked(enabled and curve)
        self.scalar_widget.setVisible(not curve)
        self.scalar_widget.setEnabled(enabled)
        self._refresh_section_container_minimum_height()

    def get_default_config(self):
        config = deepcopy(self.load_config)
        config.update(self.threshold_widget.get_config())
        config.update(self.curve_color_widget.get_config())
        config.update(limit_checked=self.limit_enabled.isChecked(), limit_metric=self.metric_combo.currentData(),
                      method=SHARPNESS_METHOD if self.analysis_type == "SHRP" else self.method_combo.currentData())
        advanced = config.setdefault("advanced", {})
        advanced.update(self.time_range.get_config())
        if self.analysis_type == "SHRP":
            advanced["output_time_resolution_ms"] = SHARPNESS_OUTPUT_TIME_RESOLUTION_MS
            for key in ("stationary_frame_duration_s", "stationary_hop_duration_s", "stationary_overlap_percent",
                        "frame_duration_s", "hop_duration_s"):
                advanced.pop(key, None)
        else:
            config["overlap"] = self.overlap_spin.value()
        for side, (enabled, spin) in self.scalar_controls.items():
            config[f"scalar_{side}_enabled"] = enabled.isChecked()
            config[f"scalar_{side}_value"] = spin.value()
        config.update(self.channel_selector.get_config() if self.available_channels is not None
                      else ChannelSelectorWidget.normalized_config(self.load_config))
        return self.merge_plot_view_config(config)

    def validate_config(self):
        if not self.validate_plot_view_config():
            return False
        config = self.get_default_config()
        advanced = config["advanced"]
        start, end = advanced["analysis_start_time_sec"], advanced["analysis_end_time_sec"]
        if advanced["analysis_time_range_enabled"] and end and end <= start:
            MessageBox.warning(self, "设置警告", "分析结束时间必须大于起始时间，或使用 0 表示音频末尾。")
            return False
        if self.analysis_type == "ROUGH" and advanced.get("background_noise_processing_enabled"):
            MessageBox.warning(self, "设置警告", "粗糙度暂不支持背景参考降噪，请移除该配置。")
            return False
        if self.analysis_type == "ROUGH" and config["method"] != "daniel_weber":
            MessageBox.warning(self, "设置警告", "请选择受支持的计算模式。")
            return False
        if not config["limit_checked"]:
            return True
        if config["limit_metric"] == "curve_y":
            return self.threshold_widget.validate()
        try:
            validate_constant_limit_config({key.replace("scalar_", "constant_"): value
                                            for key, value in config.items() if key.startswith("scalar_")})
        except ManualLimitValidationError as error:
            MessageBox.warning(self, "设置警告", str(error))
            return False
        return True

    def _confirm_legacy_conversion(self):
        if self.analysis_type != "SHRP" or sharpness_execution_issue(self.load_config) is None:
            return True
        return MessageBox.question(
            self, "更新尖锐度配置", "旧配置将改为时变方式（2 ms），原判定阈值需复核。是否保存？",
            MessageBox.Yes | MessageBox.No, MessageBox.No,
        ) == MessageBox.Yes

    def on_default_btn_clicked(self):
        if self.validate_config() and self._confirm_legacy_conversion():
            saved = self.config_manager.save_default_config(self.model_type, self.get_default_config())
            PopupUtils().save_popup(self, success_flag=saved)

    def on_restore_default_btn_clicked(self):
        self.load_config = self.merge_with_defaults(self.analysis_type, self.config_manager.load_config().get(self.model_type, {}))
        self.clear_semantic_sections()
        self._build_sections()

    def on_click_ok_btn(self):
        if self.validate_config() and self._confirm_legacy_conversion():
            self.accept_config(self.get_default_config())
