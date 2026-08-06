"""SPL (Sound Pressure Level) analysis configuration dialog."""

import re
from typing import List, Optional

from PyQt5.QtWidgets import QVBoxLayout, QWidget

from ui.custom_ui_widget.popuputils import PopupUtils
from ui.custom_ui_widget.widgets import CheckBox, GroupBox, RadioButton
from ui.ui_analysis_config.common_widgets import (
    AnalysisTimeRangeWidget,
    ChannelSelectorWidget,
    GoldenSampleWidget,
    OctaveSmoothingSelectorWidget,
    SemanticAnalysisConfigDialogBase,
    WeightingSelectorWidget,
)
from ui.ui_analysis_config.threshold_config_widget import ThresholdConfigWidget


class SplConfigWindow(SemanticAnalysisConfigDialogBase):
    """SPL and SPLF analysis configuration window."""

    DEFAULT_CONFIG = {
        "analysis_channel": 0,
        "analysis_time_range_enabled": False,
        "analysis_start_time_sec": 0.0,
        "analysis_end_time_sec": 0.0,
        "weighting": "Z",
        "smooth_checked": False,
        "limit_checked": False,
        "limit_data": None,
        "limit_mode": "csv",
        "manual_input_mode": "constant",
        "constant_upper_enabled": True,
        "constant_lower_enabled": False,
        "constant_upper_value": 100.0,
        "constant_lower_value": 0.0,
        "manual_upper_enabled": True,
        "manual_lower_enabled": False,
        "manual_upper_segments": [],
        "manual_lower_segments": [],
    }
    _MODERN_THRESHOLD_KEYS = {
        "limit_data",
        "limit_mode",
        "manual_input_mode",
        "constant_upper_enabled",
        "constant_lower_enabled",
        "constant_upper_value",
        "constant_lower_value",
        "manual_upper_enabled",
        "manual_lower_enabled",
        "manual_upper_segments",
        "manual_lower_segments",
    }
    _LEGACY_THRESHOLD_KEYS = {
        "self_defined",
        "import_config",
        "upper_limit",
        "lower_limit",
        "config_dir",
    }

    @classmethod
    def _merge_config_defaults(cls, raw_config=None):
        raw = dict(raw_config or {})
        config = {**cls.DEFAULT_CONFIG, **raw}
        if (
            "manual_input_mode" not in raw
            and str(raw.get("limit_mode", "csv") or "csv").lower()
            == "manual"
            and (
                raw.get("manual_upper_segments")
                or raw.get("manual_lower_segments")
            )
        ):
            config["manual_input_mode"] = "segments"
        return config

    @classmethod
    def new_item_default_config(cls, raw_config=None):
        """Merge code defaults and discard unsupported legacy-only limits."""
        raw = dict(raw_config or {})
        has_modern_threshold = any(
            key in raw for key in cls._MODERN_THRESHOLD_KEYS
        )
        clean = {
            key: value
            for key, value in raw.items()
            if key not in cls._LEGACY_THRESHOLD_KEYS
        }
        if not has_modern_threshold:
            clean.pop("limit_checked", None)
        return cls._merge_config_defaults(clean)

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
        saved_config = self.config_manager.load_config().get(
            self.config_key,
            {},
        )
        self.load_config = self._merge_config_defaults(saved_config)
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

        self.analysis_time_range_widget = None
        if self.model_type != "SPLF":
            self.analysis_time_range_widget = AnalysisTimeRangeWidget(
                self.load_config,
                self,
                show_checkbox=True,
            )
            self.add_semantic_section(
                "preprocess",
                widget=self.analysis_time_range_widget,
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

        self.show_overall_spl_box = None
        if self.model_type != "SPLF":
            self.show_overall_spl_box = CheckBox(
                "显示总体声压级",
                self,
            )
            self.show_overall_spl_box.setChecked(
                bool(
                    self.load_config.get(
                        "show_overall_spl",
                        False,
                    )
                )
            )

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
            compute_layout.addWidget(self.show_overall_spl_box)

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
            allow_manual_limits=True,
            allow_constant_limits=True,
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
            config["show_overall_spl"] = (
                self.show_overall_spl_box.isChecked()
            )
            config.update(self.analysis_time_range_widget.get_config())

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
        saved_config = self.config_manager.load_config().get(
            self.config_key,
            {},
        )
        self.load_config = self._merge_config_defaults(saved_config)
        self.clear_semantic_sections()
        self._build_semantic_sections()

    def on_click_ok_btn(self):
        if not self._validate_config():
            return
        self.accept()
        return self.get_default_config()
