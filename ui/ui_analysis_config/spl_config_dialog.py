"""SPL (Sound Pressure Level) analysis configuration dialog."""

import re
from typing import List, Optional

from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from base.pre_processing.spl_runtime_config import (
    resolve_directional_additional_correction_db,
    resolve_free_field_distance_correction_db,
)
from ui.custom_ui_widget.popuputils import PopupUtils
from ui.custom_ui_widget.widgets import (
    CheckBox,
    DoubleSpinBox,
    GroupBox,
    Label,
    RadioButton,
)
from ui.ui_analysis_config.common_widgets import (
    AnalysisChannelSpinBoxWidget,
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
        "free_field_distance_enabled": False,
        "measurement_distance_m": 0.05,
        "target_distance_m": 1.0,
        "directional_correction_enabled": False,
        "directional_additional_correction_db": 0.0,
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
        restrict_analysis_channel: bool = False,
    ):
        super().__init__(disable_close_button=True)
        self.config_manager = config_manager
        self.config_key = model_type
        self.model_type = "".join(
            re.findall(r"[A-Za-z]", str(model_type))
        ) or "SPL"
        self.show_channel_selector = available_channels is not None
        self.available_channels = available_channels
        self.restrict_analysis_channel = restrict_analysis_channel
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
            if self.model_type == "SPL":
                self.channel_selector = AnalysisChannelSpinBoxWidget(
                    self.load_config,
                    self.available_channels,
                    self,
                    restrict_to_available_channels=(
                        self.restrict_analysis_channel
                    ),
                )
            else:
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
        self.free_field_distance_box = None
        self.free_field_distance_parameters_widget = None
        self.measurement_distance_spin = None
        self.target_distance_spin = None
        self.directional_correction_widget = None
        self.directional_correction_box = None
        self.directional_additional_correction_spin = None
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
            self.free_field_distance_box = CheckBox(
                "启用球面扩散距离推算",
                self,
            )
            self.free_field_distance_box.setChecked(
                bool(
                    self.load_config.get(
                        "free_field_distance_enabled",
                        False,
                    )
                )
            )

            self.free_field_distance_parameters_widget = QWidget(
                compute_widget
            )
            parameters_layout = QVBoxLayout(
                self.free_field_distance_parameters_widget
            )
            parameters_layout.setContentsMargins(24, 0, 0, 0)
            parameters_layout.setSpacing(8)

            distance_row = QHBoxLayout()
            self.measurement_distance_spin = self._create_distance_spin(
                self.load_config.get("measurement_distance_m", 0.05)
            )
            self.target_distance_spin = self._create_distance_spin(
                self.load_config.get("target_distance_m", 1.0)
            )
            distance_row.addWidget(Label("测量距离："))
            distance_row.addWidget(self.measurement_distance_spin)
            distance_row.addWidget(Label("目标距离："))
            distance_row.addWidget(self.target_distance_spin)
            distance_row.addStretch(1)
            parameters_layout.addLayout(distance_row)

            self.directional_correction_widget = QWidget(compute_widget)
            correction_row = QHBoxLayout(
                self.directional_correction_widget
            )
            correction_row.setContentsMargins(0, 0, 0, 0)
            self.directional_correction_box = CheckBox(
                "方向修正",
                self.directional_correction_widget,
            )
            self.directional_correction_box.setChecked(
                bool(
                    self.load_config.get(
                        "directional_correction_enabled",
                        False,
                    )
                )
            )
            self.directional_additional_correction_spin = (
                self._create_correction_spin(
                    self.load_config.get(
                        "directional_additional_correction_db",
                        0.0,
                    )
                )
            )
            correction_row.addWidget(self.directional_correction_box)
            correction_row.addWidget(
                self.directional_additional_correction_spin
            )
            correction_row.addStretch(1)

            for spin in (
                self.measurement_distance_spin,
                self.target_distance_spin,
                self.directional_additional_correction_spin,
            ):
                spin.valueChanged.connect(
                    self._update_spl_correction_tooltip
                )
            self.directional_correction_box.stateChanged.connect(
                self._sync_directional_correction_controls
            )
            self.free_field_distance_box.stateChanged.connect(
                self._sync_free_field_distance_controls
            )
            self._sync_directional_correction_controls()
            self._sync_free_field_distance_controls()

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
            compute_layout.addWidget(self.free_field_distance_box)
            compute_layout.addWidget(
                self.free_field_distance_parameters_widget
            )
            compute_layout.addWidget(self.directional_correction_widget)

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
            allow_csv_limit_offsets=True,
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
            config.update(
                {
                    "free_field_distance_enabled": (
                        self.free_field_distance_box.isChecked()
                    ),
                    "measurement_distance_m": float(
                        self.measurement_distance_spin.value()
                    ),
                    "target_distance_m": float(
                        self.target_distance_spin.value()
                    ),
                    "directional_correction_enabled": (
                        self.directional_correction_box.isChecked()
                    ),
                    "directional_additional_correction_db": float(
                        self.directional_additional_correction_spin.value()
                    ),
                }
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

    def _create_distance_spin(self, value):
        spin = DoubleSpinBox(self)
        spin.setRange(0.0001, 1000000.0)
        spin.setDecimals(4)
        spin.setSingleStep(0.1)
        spin.setSuffix(" m")
        spin.setValue(float(value))
        spin.setMinimumWidth(140)
        spin.setMaximumWidth(180)
        return spin

    def _create_correction_spin(self, value):
        spin = DoubleSpinBox(self)
        spin.setRange(-200.0, 200.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.1)
        spin.setSuffix(" dB")
        spin.setValue(float(value))
        spin.setMinimumWidth(140)
        spin.setMaximumWidth(180)
        return spin

    def _sync_free_field_distance_controls(self, *args):
        enabled = self.free_field_distance_box.isChecked()
        self.free_field_distance_parameters_widget.setVisible(enabled)
        self._update_spl_correction_tooltip()
        self.free_field_distance_parameters_widget.updateGeometry()
        parent_widget = self.free_field_distance_parameters_widget.parentWidget()
        parent_layout = parent_widget.layout()
        if parent_layout is not None:
            parent_layout.invalidate()
            parent_layout.activate()
        self._refresh_section_container_minimum_height()

    def _sync_directional_correction_controls(self, *args):
        enabled = self.directional_correction_box.isChecked()
        self.directional_additional_correction_spin.setVisible(enabled)
        self.directional_additional_correction_spin.setEnabled(enabled)
        self._update_spl_correction_tooltip()

    def _update_spl_correction_tooltip(self, *args):
        correction_config = {
            "free_field_distance_enabled": (
                self.free_field_distance_box.isChecked()
            ),
            "measurement_distance_m": self.measurement_distance_spin.value(),
            "target_distance_m": self.target_distance_spin.value(),
            "directional_correction_enabled": (
                self.directional_correction_box.isChecked()
            ),
            "directional_additional_correction_db": (
                self.directional_additional_correction_spin.value()
            ),
        }
        distance_correction_db = (
            resolve_free_field_distance_correction_db(correction_config)
        )
        directional_correction_db = (
            resolve_directional_additional_correction_db(correction_config)
        )
        total_correction_db = (
            distance_correction_db + directional_correction_db
        )
        status_text = f"当前总修正量：{total_correction_db:+.2f} dB"
        if self.free_field_distance_box.isChecked():
            distance_text = f"球面扩散：{distance_correction_db:+.2f} dB"
        else:
            distance_text = "球面扩散：未启用"
        if self.directional_correction_box.isChecked():
            directional_text = (
                f"方向修正：{directional_correction_db:+.2f} dB"
            )
        else:
            directional_text = "方向修正：未启用"
        tooltip = (
            f"{status_text}\n"
            f"{distance_text}\n"
            f"{directional_text}（正值提高，负值降低）"
        )
        for widget in (
            self.free_field_distance_box,
            self.free_field_distance_parameters_widget,
            self.measurement_distance_spin,
            self.target_distance_spin,
            self.directional_correction_widget,
            self.directional_correction_box,
            self.directional_additional_correction_spin,
        ):
            widget.setToolTip(tooltip)

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
