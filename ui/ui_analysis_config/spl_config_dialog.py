"""SPL (Sound Pressure Level) analysis configuration dialog."""

import re
from typing import List, Optional

from PyQt5.QtWidgets import QVBoxLayout, QWidget

from ui.custom_ui_widget.popuputils import PopupUtils
from ui.custom_ui_widget.widgets import CheckBox, GroupBox, RadioButton
from ui.ui_analysis_config.common_widgets import (
    ChannelSelectorWidget,
    GoldenSampleWidget,
    OctaveSmoothingSelectorWidget,
    SemanticAnalysisConfigDialogBase,
    WeightingSelectorWidget,
)
from ui.ui_analysis_config.threshold_config_widget import ThresholdConfigWidget


class SplConfigWindow(SemanticAnalysisConfigDialogBase):
    """SPL and SPLF analysis configuration window."""

    def __init__(
        self,
        config_manager,
        model_type,
        available_channels: Optional[List[int]] = None,
    ):
        super().__init__(disable_close_button=True)
        self.config_manager = config_manager
        self.config_key = model_type
        self.model_type = "".join(
            re.findall(r"[A-Za-z]", str(model_type))
        ) or "SPL"
        self.show_channel_selector = available_channels is not None
        self.available_channels = available_channels
        self.load_config = self.config_manager.load_config().get(
            self.config_key,
            {},
        )
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(f"{self.model_type} 分析配置")
        self.apply_semantic_dialog_size()
        self.set_semantic_button_callbacks(
            default_callback=self.on_default_btn_clicked,
            restore_callback=self.on_restore_default_btn_clicked,
            ok_callback=self.on_click_ok_btn,
        )
        self._build_semantic_sections()

    def _build_semantic_sections(self):
        if self.show_channel_selector:
            self.channel_selector = ChannelSelectorWidget(
                self.load_config,
                self.available_channels,
                self,
            )
            self.add_semantic_section(
                "input",
                widget=self.channel_selector,
            )

        compute_widget = QWidget(self)
        compute_layout = QVBoxLayout(compute_widget)
        compute_layout.setContentsMargins(0, 0, 0, 0)
        compute_layout.setSpacing(12)

        self.weighting_selector = WeightingSelectorWidget(
            self.load_config,
            parent=self,
        )
        compute_layout.addWidget(self.weighting_selector)

        self.smooth_checkbox = None
        self.splf_mode_group = None
        self.smoothing_selector = None
        self.golden_sample_widget = None
        if self.model_type == "SPLF":
            self.splf_mode_group = GroupBox("SPLF 计算方式", self)
            mode_layout = QVBoxLayout(self.splf_mode_group)
            self.radio_fundamental = RadioButton("仅基频", self)
            self.radio_total = RadioButton("总 SPL", self)
            if (
                str(
                    self.load_config.get(
                        "splf_calc_mode",
                        "fundamental",
                    )
                    or "fundamental"
                ).lower()
                == "total"
            ):
                self.radio_total.setChecked(True)
            else:
                self.radio_fundamental.setChecked(True)
            mode_layout.addWidget(self.radio_fundamental)
            mode_layout.addWidget(self.radio_total)
            compute_layout.addWidget(self.splf_mode_group)

            self.smoothing_selector = OctaveSmoothingSelectorWidget(
                self.load_config,
                parent=self,
            )
            compute_layout.addWidget(self.smoothing_selector)
            self.golden_sample_widget = GoldenSampleWidget(
                self.load_config,
                self,
            )
        else:
            self.smooth_checkbox = CheckBox("平滑", self)
            self.smooth_checkbox.setChecked(
                bool(self.load_config.get("smooth_checked", False))
            )
            compute_layout.addWidget(self.smooth_checkbox)

        self.add_semantic_section("compute", widget=compute_widget)

        if self.golden_sample_widget is not None:
            self.add_semantic_section(
                "reference",
                widget=self.golden_sample_widget,
            )

        self.threshold_widget = ThresholdConfigWidget(
            parent=self,
            load_config=self.load_config,
            model_type=self.model_type,
            limit_value_semantics_provider=(
                self.golden_sample_widget.limit_value_semantics
                if self.golden_sample_widget is not None
                else None
            ),
        )

        self.enable_plot_view_config(
            self.load_config,
            "Hz" if self.model_type == "SPLF" else "s",
            "dB",
            True,
            True,
            self.model_type == "SPLF",
        )
        self.add_threshold_curve_sections(
            self.threshold_widget,
            self.load_config,
        )

    def get_default_config(self):
        config = {}
        if self.model_type == "SPLF":
            config["splf_calc_mode"] = (
                "total"
                if self.radio_total.isChecked()
                else "fundamental"
            )
            config.update(self.smoothing_selector.get_config())
            config.update(self.golden_sample_widget.get_config())
        else:
            config["smooth_checked"] = self.smooth_checkbox.isChecked()

        config.update(self.weighting_selector.get_config())
        config.update(self.threshold_widget.get_config())
        if self.show_channel_selector:
            config.update(self.channel_selector.get_config())
        else:
            config["analysis_channel"] = int(
                self.load_config.get("analysis_channel", 0) or 0
            )
        return self.merge_plot_view_config(config)

    def _validate_config(self):
        if not self.validate_plot_view_config():
            return False
        return self.threshold_widget.validate()

    def on_default_btn_clicked(self):
        if not self._validate_config():
            return
        save_flag = self.config_manager.save_default_config(
            self.model_type,
            self.get_default_config(),
        )
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_restore_default_btn_clicked(self):
        self.load_config = self.config_manager.load_config().get(
            self.config_key,
            {},
        )
        self.clear_semantic_sections()
        self._build_semantic_sections()

    def on_click_ok_btn(self):
        if not self._validate_config():
            return
        self.accept()
        return self.get_default_config()
