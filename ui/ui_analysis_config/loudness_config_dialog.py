from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout, QWidget

from base.core_algorithm.sound_quality.psychoacoustic_constants import (
    LOUDNESS_DEFAULT_STATIONARY_FRAME_DURATION_S,
    LOUDNESS_DEFAULT_STATIONARY_HOP_DURATION_S,
    LOUDNESS_DEFAULT_STATIONARY_OVERLAP_PERCENT,
)

_DEFAULT_FRAME_S = LOUDNESS_DEFAULT_STATIONARY_FRAME_DURATION_S
_DEFAULT_HOP_S = LOUDNESS_DEFAULT_STATIONARY_HOP_DURATION_S
_DEFAULT_OVERLAP = LOUDNESS_DEFAULT_STATIONARY_OVERLAP_PERCENT
from ui.custom_ui_widget.popuputils import PopupUtils
from ui.custom_ui_widget.widgets import CheckBox, ComboBox, DoubleSpinBox, GroupBox, Label, PushButton
from ui.ui_src import ui_resources


class LoudnessConfigPanel(QWidget):
    """Reusable editor for the LOUD item config."""

    METHOD_VALUE_TO_LABEL = {
        "per_segment": "分段响度",
        "time_varying_iso532_1": "时变响度",
    }
    METHOD_LABEL_TO_VALUE = {label: value for value, label in METHOD_VALUE_TO_LABEL.items()}
    SPECIFIC_PROFILE_VALUE_TO_LABEL = {
        "steady_average": "稳态平均",
        "max_loudness": "最大响度时刻",
    }
    SPECIFIC_PROFILE_LABEL_TO_VALUE = {
        label: value for value, label in SPECIFIC_PROFILE_VALUE_TO_LABEL.items()
    }
    EXCEEDANCE_MODE_VALUE_TO_LABEL = {
        "threshold": "固定限值 T",
        "ref_line": "参考曲线 (SSTS)",
    }
    EXCEEDANCE_MODE_LABEL_TO_VALUE = {
        label: value for value, label in EXCEEDANCE_MODE_VALUE_TO_LABEL.items()
    }
    EXCEEDANCE_REF_VALUE_TO_LABEL = {
        "ref1": "Ref 1",
        "ref2": "Ref 2",
        "ref3": "Ref 3",
        "ref4": "Ref 4",
    }
    EXCEEDANCE_REF_LABEL_TO_VALUE = {
        label: value for value, label in EXCEEDANCE_REF_VALUE_TO_LABEL.items()
    }

    def __init__(self, load_config=None, title_prefix="", comparison_only=False, parent=None):
        super().__init__(parent)
        self.load_config = load_config or {}
        self.title_prefix = title_prefix
        self.comparison_only = bool(comparison_only)
        self.init_ui()

    def _group_title(self, title):
        return f"{self.title_prefix}{title}" if self.title_prefix else title

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._create_algorithm_group())
        layout.addWidget(self._create_output_group())
        layout.addWidget(self._create_limit_group())
        layout.addStretch()
        self.setLayout(layout)

    def _create_algorithm_group(self):
        group = GroupBox(self._group_title("算法参数"))
        layout = QVBoxLayout()
        advanced_cfg = self.load_config.get("advanced", {}) or {}

        method_layout = QHBoxLayout()
        method_layout.addWidget(Label("计算模式:"))
        self.method_combo = ComboBox()
        self.method_combo.addItems(list(self.METHOD_LABEL_TO_VALUE.keys()))
        method_value = str(self.load_config.get("method", "per_segment") or "per_segment")
        self.method_combo.setCurrentText(self.METHOD_VALUE_TO_LABEL.get(method_value, "分段响度"))
        self.method_combo.setToolTip(
            "响度按 ISO 532-1 / Zwicker 方法计算。\n"
            "分段响度：适合校准音、稳定噪声、电机稳定运行趋势。\n"
            "时变响度：适合 chirp、step、启动/停止、冲击、调制等时间动态分析。"
        )
        self.method_combo.setMinimumWidth(220)
        self.method_combo.setMaximumWidth(300)
        method_layout.addWidget(self.method_combo)
        method_layout.addStretch(1)
        layout.addLayout(method_layout)

        self.stationary_controls_widget = QWidget(self)
        self.stationary_controls_widget.setVisible(False)

        group.setLayout(layout)
        return group

    def _create_output_group(self):
        group = GroupBox(self._group_title("显示设置"))
        group.setMinimumHeight(290)
        layout = QHBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 22, 10, 14)

        display_cfg = self.load_config.get("display", {}) or {}
        advanced_cfg = self.load_config.get("advanced", {}) or {}

        metric_group = GroupBox("指标")
        metric_group.setMinimumSize(300, 250)
        metric_layout = QVBoxLayout()
        metric_layout.setSpacing(10)
        metric_layout.setContentsMargins(12, 20, 12, 12)
        if self.comparison_only:
            summary_metrics = display_cfg.get("summary_metrics", []) or []
            self.show_mean_box = CheckBox("平均响度（跟随曲线纵轴单位）")
            self.show_mean_box.setToolTip("勾选后在图标题显示平均响度；纵轴为 sone 时显示 sone，纵轴为 phon 时显示 phon。")
            self.show_mean_box.setChecked(
                any(key in summary_metrics for key in ("mean_loudness", "mean_sone", "mean_phon"))
            )
            metric_layout.addWidget(self.show_mean_box)
        else:
            summary_metrics = display_cfg.get("summary_metrics", []) or []
            self.show_specific_exceedance_box = CheckBox("特征响度超限总量")
            self.show_specific_exceedance_box.setToolTip(
                "对时间平均后的特征响度曲线 N'(z)，超出限值的部分沿 Bark 频段累加。\n"
                "判定方式可选：固定限值 T（所有频段同一阈值），或参考曲线 Ref1-Ref4（随频段变化的 SSTS 限值曲线）。"
            )
            self.show_specific_exceedance_box.setChecked(
                "specific_loudness_summed_exceedance" in summary_metrics
                or "specific_loudness_exceedance" in summary_metrics
            )
            self.show_steady_avg_box = CheckBox("稳态平均响度")
            self.show_steady_avg_box.setToolTip("响度曲线的时间平均值，单位跟随纵轴（sone 或 phon）。")
            self.show_steady_avg_box.setChecked(
                "steady_state_average_sone" in summary_metrics
                or "steady_state_average_phon" in summary_metrics
                or "steady_state_average_loudness" in summary_metrics
                or "mean_loudness" in summary_metrics
                or "mean_sone" in summary_metrics
                or "mean_phon" in summary_metrics
            )
            self.show_max_transient_box = CheckBox("最大瞬态响度")
            self.show_max_transient_box.setToolTip("响度曲线的最大值，单位跟随纵轴（sone 或 phon）。")
            self.show_max_transient_box.setChecked(
                "max_transient_sone" in summary_metrics
                or "max_transient_phon" in summary_metrics
                or "max_transient_loudness" in summary_metrics
                or "nmax_sone" in summary_metrics
            )
            self.show_specific_sum_box = CheckBox("特征响度总贡献")
            self.show_specific_sum_box.setToolTip("对当前特征响度曲线 N'(z) 沿 Bark 轴积分，单位 sone。")
            self.show_specific_sum_box.setChecked("specific_loudness_sum_sone" in summary_metrics)
            self.specific_exceedance_threshold_widget = QWidget(self)
            exceedance_layout = QVBoxLayout()
            exceedance_layout.setContentsMargins(18, 0, 0, 0)
            exceedance_layout.setSpacing(6)

            mode_row = QHBoxLayout()
            mode_row.setContentsMargins(0, 0, 0, 0)
            mode_row.addWidget(Label("判定方式:"))
            self.specific_exceedance_mode_combo = ComboBox()
            self.specific_exceedance_mode_combo.addItems(
                list(self.EXCEEDANCE_MODE_LABEL_TO_VALUE.keys())
            )
            saved_ref_line = str(
                advanced_cfg.get("specific_loudness_exceedance_ref_line", "") or ""
            ).lower()
            saved_mode = "ref_line" if saved_ref_line in self.EXCEEDANCE_REF_VALUE_TO_LABEL else "threshold"
            self.specific_exceedance_mode_combo.setCurrentText(
                self.EXCEEDANCE_MODE_VALUE_TO_LABEL.get(saved_mode, "固定限值 T")
            )
            self.specific_exceedance_mode_combo.setMinimumWidth(140)
            self.specific_exceedance_mode_combo.setMaximumWidth(180)
            self.specific_exceedance_mode_combo.setMinimumHeight(30)
            mode_row.addWidget(self.specific_exceedance_mode_combo)
            mode_row.addStretch(1)
            exceedance_layout.addLayout(mode_row)

            self.specific_exceedance_t_widget = QWidget(self)
            threshold_layout = QHBoxLayout()
            threshold_layout.setContentsMargins(0, 0, 0, 0)
            threshold_layout.addWidget(Label("限值 T:"))
            self.specific_exceedance_threshold_spin = DoubleSpinBox()
            self.specific_exceedance_threshold_spin.setRange(0.0, 1000.0)
            self.specific_exceedance_threshold_spin.setDecimals(2)
            self.specific_exceedance_threshold_spin.setSingleStep(0.01)
            self.specific_exceedance_threshold_spin.setValue(
                float(advanced_cfg.get("specific_loudness_exceedance_threshold_sone_per_bark", 0.0) or 0.0)
            )
            self.specific_exceedance_threshold_spin.setMinimumWidth(95)
            self.specific_exceedance_threshold_spin.setMaximumWidth(115)
            self.specific_exceedance_threshold_spin.setMinimumHeight(30)
            threshold_layout.addWidget(self.specific_exceedance_threshold_spin)
            threshold_layout.addWidget(Label("sone/Bark"))
            threshold_layout.addStretch(1)
            self.specific_exceedance_t_widget.setLayout(threshold_layout)
            exceedance_layout.addWidget(self.specific_exceedance_t_widget)

            self.specific_exceedance_ref_widget = QWidget(self)
            ref_layout = QHBoxLayout()
            ref_layout.setContentsMargins(0, 0, 0, 0)
            ref_layout.addWidget(Label("参考曲线:"))
            self.specific_exceedance_ref_combo = ComboBox()
            self.specific_exceedance_ref_combo.addItems(
                list(self.EXCEEDANCE_REF_LABEL_TO_VALUE.keys())
            )
            self.specific_exceedance_ref_combo.setCurrentText(
                self.EXCEEDANCE_REF_VALUE_TO_LABEL.get(saved_ref_line, "Ref 1")
            )
            self.specific_exceedance_ref_combo.setToolTip(
                "SSTS 频率相关限值曲线，随 Bark 频段变化。Ref1-Ref4 为不同严格程度的预设曲线。"
            )
            self.specific_exceedance_ref_combo.setMinimumWidth(95)
            self.specific_exceedance_ref_combo.setMaximumWidth(115)
            self.specific_exceedance_ref_combo.setMinimumHeight(30)
            ref_layout.addWidget(self.specific_exceedance_ref_combo)
            ref_layout.addStretch(1)
            self.specific_exceedance_ref_widget.setLayout(ref_layout)
            exceedance_layout.addWidget(self.specific_exceedance_ref_widget)

            self.specific_exceedance_threshold_widget.setLayout(exceedance_layout)
            self.show_specific_exceedance_box.toggled.connect(self.specific_exceedance_threshold_widget.setVisible)
            self.specific_exceedance_threshold_widget.setVisible(self.show_specific_exceedance_box.isChecked())
            self.specific_exceedance_mode_combo.currentTextChanged.connect(
                self._refresh_exceedance_mode
            )
            self._refresh_exceedance_mode()
            metric_layout.addWidget(self.show_steady_avg_box)
            metric_layout.addWidget(self.show_max_transient_box)
            metric_layout.addWidget(self.show_specific_sum_box)
            metric_layout.addWidget(self.show_specific_exceedance_box)
            metric_layout.addWidget(self.specific_exceedance_threshold_widget)
        metric_group.setLayout(metric_layout)

        graph_group = GroupBox("图形")
        graph_group.setMinimumSize(300, 250)
        graph_layout = QVBoxLayout()
        graph_layout.setSpacing(10)
        graph_layout.setContentsMargins(12, 20, 12, 12)
        self.show_curve_box = CheckBox("响度时间曲线（N-t）")
        self.show_curve_box.setToolTip("N-t：显示响度 N 随时间 t 变化的曲线。")
        self.show_curve_box.setChecked("loudness_time" in (display_cfg.get("curves", []) or []))
        self.curve_y_unit_widget = QWidget(self)
        curve_y_unit_layout = QHBoxLayout()
        curve_y_unit_layout.setContentsMargins(0, 0, 0, 0)
        curve_y_unit_layout.addWidget(Label("纵轴:"))
        self.curve_y_unit_combo = ComboBox()
        self.curve_y_unit_combo.addItems(["sone", "phon"])
        curve_y_unit = str(advanced_cfg.get("curve_y_unit", "sone") or "sone").lower()
        self.curve_y_unit_combo.setCurrentText("phon" if curve_y_unit == "phon" else "sone")
        self.curve_y_unit_combo.setToolTip("响度时间曲线的纵轴单位。sone 为响度 N，phon 为响度级 LN。")
        self.show_curve_box.toggled.connect(self.curve_y_unit_widget.setVisible)
        self.curve_y_unit_combo.setMinimumWidth(120)
        self.curve_y_unit_combo.setMaximumWidth(180)
        curve_y_unit_layout.addWidget(self.curve_y_unit_combo)
        curve_y_unit_layout.addStretch(1)
        self.curve_y_unit_widget.setLayout(curve_y_unit_layout)
        self.curve_y_unit_widget.setVisible(self.show_curve_box.isChecked())
        self.show_specific_box = CheckBox("特征响度分布图")
        self.show_specific_box.setToolTip(
            "特征响度（Specific Loudness, N'）：响度沿 Bark 临界频带的分布，"
            "单位 sone/Bark；以热力图形式展示，主要用于研发分析。"
        )
        self.show_specific_box.setChecked(bool(advanced_cfg.get("show_specific_loudness_heatmap", False)))
        self.show_specific_profile_box = CheckBox("特征响度曲线 N'(z)")
        self.show_specific_profile_box.setToolTip("显示一条 Bark 轴特征响度曲线，单位 sone/Bark。")
        self.show_specific_profile_box.setChecked("specific_loudness_profile" in (display_cfg.get("curves", []) or []))
        self.specific_profile_mode_widget = QWidget(self)
        profile_mode_layout = QHBoxLayout()
        profile_mode_layout.setContentsMargins(0, 0, 0, 0)
        profile_mode_layout.addWidget(Label("曲线口径:"))
        self.specific_profile_mode_combo = ComboBox()
        self.specific_profile_mode_combo.addItems(list(self.SPECIFIC_PROFILE_LABEL_TO_VALUE.keys()))
        profile_mode = str(advanced_cfg.get("specific_loudness_profile_mode", "steady_average") or "steady_average")
        self.specific_profile_mode_combo.setCurrentText(
            self.SPECIFIC_PROFILE_VALUE_TO_LABEL.get(profile_mode, "稳态平均")
        )
        self.specific_profile_mode_combo.setToolTip("稳态平均：对稳态段 N'(z,t) 求时间平均；最大响度时刻：取总响度最大帧的 N'(z)。")
        self.specific_profile_mode_combo.setMinimumWidth(145)
        self.specific_profile_mode_combo.setMaximumWidth(180)
        self.specific_profile_mode_combo.setMinimumHeight(30)
        profile_mode_layout.addWidget(self.specific_profile_mode_combo)
        profile_mode_layout.addStretch(1)
        self.specific_profile_mode_widget.setLayout(profile_mode_layout)
        self.show_specific_profile_box.toggled.connect(self.specific_profile_mode_widget.setVisible)
        self.specific_profile_mode_widget.setVisible(self.show_specific_profile_box.isChecked())
        self.specific_colormap_widget = QWidget(self)
        specific_colormap_layout = QHBoxLayout()
        specific_colormap_layout.setContentsMargins(0, 0, 0, 0)
        specific_colormap_layout.addWidget(Label("色图:"))
        self.specific_colormap_combo = ComboBox()
        self.specific_colormap_combo.addItems(["viridis", "plasma", "magma", "inferno"])
        self.specific_colormap_combo.setCurrentText(advanced_cfg.get("specific_loudness_colormap", "viridis"))
        self.specific_colormap_combo.setToolTip("特征响度分布图使用的颜色映射。")
        self.show_specific_box.toggled.connect(self.specific_colormap_widget.setVisible)
        self.specific_colormap_combo.setMinimumWidth(140)
        self.specific_colormap_combo.setMaximumWidth(200)
        specific_colormap_layout.addWidget(self.specific_colormap_combo)
        specific_colormap_layout.addStretch(1)
        self.specific_colormap_widget.setLayout(specific_colormap_layout)
        self.specific_colormap_widget.setVisible(self.show_specific_box.isChecked())
        graph_layout.addWidget(self.show_curve_box)
        graph_layout.addWidget(self.curve_y_unit_widget)
        graph_layout.addWidget(self.show_specific_profile_box)
        graph_layout.addWidget(self.specific_profile_mode_widget)
        graph_layout.addWidget(self.show_specific_box)
        graph_layout.addWidget(self.specific_colormap_widget)
        graph_group.setLayout(graph_layout)

        layout.addWidget(metric_group)
        layout.addWidget(graph_group, 1)

        group.setLayout(layout)
        return group

    LIMIT_METRIC_OPTIONS = {
        "curve_y": "响度曲线逐点判定",
        "steady_state_average": "稳态平均响度",
        "max_transient": "最大瞬态响度",
    }
    LIMIT_METRIC_LABEL_TO_VALUE = {label: value for value, label in LIMIT_METRIC_OPTIONS.items()}

    def _create_limit_group(self):
        group = GroupBox(self._group_title("判定阈值"))
        group.setMinimumHeight(220)
        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(12, 24, 12, 14)

        upper_enabled_key = "curve_upper_enabled"
        upper_value_key = "curve_upper_value"
        lower_enabled_key = "curve_lower_enabled"
        lower_value_key = "curve_lower_value"

        self.limit_checked_box = CheckBox("启用响度 OK/NG 判定")
        self.limit_checked_box.setToolTip("启用后，根据选定的判定指标对响度结果进行 OK/NG 判定。")
        self.limit_checked_box.setChecked(bool(self.load_config.get("limit_checked", False)))
        layout.addWidget(self.limit_checked_box)

        metric_layout = QHBoxLayout()
        metric_layout.addWidget(Label("判定指标:"))
        self.limit_metric_combo = ComboBox()
        self.limit_metric_combo.addItems(list(self.LIMIT_METRIC_OPTIONS.values()))
        limit_metric_value = str(self.load_config.get("limit_metric", "curve_y") or "curve_y")
        self.limit_metric_combo.setCurrentText(
            self.LIMIT_METRIC_OPTIONS.get(limit_metric_value, "响度曲线逐点判定")
        )
        self.limit_metric_combo.setToolTip(
            "响度曲线逐点判定：响度曲线每个时间点都不能超限。\n"
            "稳态平均响度：时间平均响度不能超限。\n"
            "最大瞬态响度：响度最大值不能超限。"
        )
        self.limit_metric_combo.setMinimumWidth(220)
        self.limit_metric_combo.setMaximumWidth(320)
        self.limit_metric_combo.setMinimumHeight(30)
        metric_layout.addWidget(self.limit_metric_combo)
        metric_layout.addStretch(1)
        layout.addLayout(metric_layout)

        upper_layout = QHBoxLayout()
        self.upper_enabled_box = CheckBox("上限")
        self.upper_enabled_box.setChecked(
            bool(
                self.load_config.get(
                    upper_enabled_key,
                    self.load_config.get("mean_upper_enabled", self.load_config.get("nmax_upper_enabled", True)),
                )
            )
        )
        self.upper_spin = DoubleSpinBox()
        self.upper_spin.setRange(0.0, 10000.0)
        self.upper_spin.setDecimals(3)
        self.upper_spin.setValue(
            float(
                self.load_config.get(
                    upper_value_key,
                    self.load_config.get("mean_upper_sone", self.load_config.get("nmax_upper_sone", 20.0)),
                )
            )
        )
        self.upper_spin.setSuffix(f" {self._curve_limit_unit()}")
        self.upper_spin.setMinimumWidth(160)
        self.upper_spin.setMaximumWidth(220)
        upper_layout.addWidget(self.upper_enabled_box)
        upper_layout.addWidget(self.upper_spin)
        upper_layout.addStretch(1)
        layout.addLayout(upper_layout)

        lower_layout = QHBoxLayout()
        self.lower_enabled_box = CheckBox("下限")
        self.lower_enabled_box.setChecked(
            bool(
                self.load_config.get(
                    lower_enabled_key,
                    self.load_config.get("mean_lower_enabled", self.load_config.get("nmax_lower_enabled", False)),
                )
            )
        )
        self.lower_spin = DoubleSpinBox()
        self.lower_spin.setRange(0.0, 10000.0)
        self.lower_spin.setDecimals(3)
        self.lower_spin.setValue(
            float(
                self.load_config.get(
                    lower_value_key,
                    self.load_config.get("mean_lower_sone", self.load_config.get("nmax_lower_sone", 0.0)),
                )
            )
        )
        self.lower_spin.setSuffix(f" {self._curve_limit_unit()}")
        self.lower_spin.setMinimumWidth(160)
        self.lower_spin.setMaximumWidth(220)
        lower_layout.addWidget(self.lower_enabled_box)
        lower_layout.addWidget(self.lower_spin)
        lower_layout.addStretch(1)
        layout.addLayout(lower_layout)

        self.curve_y_unit_combo.currentTextChanged.connect(self._refresh_limit_unit_suffix)
        group.setLayout(layout)
        return group

    def _curve_limit_unit(self):
        return "phon" if self.curve_y_unit_combo.currentText() == "phon" else "sone"

    def _refresh_limit_unit_suffix(self):
        suffix = f" {self._curve_limit_unit()}"
        self.upper_spin.setSuffix(suffix)
        self.lower_spin.setSuffix(suffix)

    def _refresh_exceedance_mode(self):
        mode = self.EXCEEDANCE_MODE_LABEL_TO_VALUE.get(
            self.specific_exceedance_mode_combo.currentText(), "threshold"
        )
        is_ref = mode == "ref_line"
        self.specific_exceedance_t_widget.setVisible(not is_ref)
        self.specific_exceedance_ref_widget.setVisible(is_ref)

    def _exceedance_ref_line_value(self, advanced_cfg):
        if self.comparison_only:
            return str(advanced_cfg.get("specific_loudness_exceedance_ref_line", "") or "")
        mode = self.EXCEEDANCE_MODE_LABEL_TO_VALUE.get(
            self.specific_exceedance_mode_combo.currentText(), "threshold"
        )
        if mode != "ref_line":
            return ""
        return self.EXCEEDANCE_REF_LABEL_TO_VALUE.get(
            self.specific_exceedance_ref_combo.currentText(), "ref1"
        )

    @staticmethod
    def _positive_float_or_default(value, default):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return float(default)
        return numeric if numeric > 0.0 else float(default)

    def _stationary_overlap_percent_from_config(self, advanced_cfg):
        configured_overlap = advanced_cfg.get("stationary_overlap_percent")
        if configured_overlap is not None:
            try:
                return self._clamp_float(float(configured_overlap), 0.0, 90.0)
            except (TypeError, ValueError):
                pass
        if "stationary_frame_duration_s" not in advanced_cfg and "stationary_hop_duration_s" not in advanced_cfg:
            return LOUDNESS_DEFAULT_STATIONARY_OVERLAP_PERCENT

        frame_s = self._positive_float_or_default(
            advanced_cfg.get("stationary_frame_duration_s", _DEFAULT_FRAME_S),
            _DEFAULT_FRAME_S,
        )
        hop_s = self._positive_float_or_default(
            advanced_cfg.get("stationary_hop_duration_s", _DEFAULT_HOP_S),
            _DEFAULT_HOP_S,
        )
        overlap_percent = (1.0 - min(hop_s, frame_s) / frame_s) * 100.0
        return self._clamp_float(overlap_percent, 0.0, 90.0)

    def _stationary_hop_duration_s(self):
        overlap_ratio = _DEFAULT_OVERLAP / 100.0
        return max(_DEFAULT_FRAME_S * (1.0 - overlap_ratio), 0.001)

    @staticmethod
    def _clamp_float(value, lower, upper):
        return max(lower, min(upper, float(value)))

    def get_default_config(self):
        if self.comparison_only:
            summary_metrics = []
            if self.show_mean_box.isChecked():
                summary_metrics.append("mean_loudness")
        else:
            summary_metrics = []
            if self.show_steady_avg_box.isChecked():
                summary_metrics.append("steady_state_average_loudness")
            if self.show_max_transient_box.isChecked():
                summary_metrics.append("max_transient_loudness")
            if self.show_specific_sum_box.isChecked():
                summary_metrics.append("specific_loudness_sum_sone")
            if self.show_specific_exceedance_box.isChecked():
                summary_metrics.append("specific_loudness_summed_exceedance")

        curves = ["loudness_time"] if self.show_curve_box.isChecked() else []
        if not self.comparison_only and self.show_specific_profile_box.isChecked():
            curves.append("specific_loudness_profile")
        heatmaps = ["specific_loudness"] if self.show_specific_box.isChecked() else []
        advanced_cfg = self.load_config.get("advanced", {}) or {}

        return {
            "enabled": True,
            "field_type": "free",
            "method": self.METHOD_LABEL_TO_VALUE.get(self.method_combo.currentText(), "per_segment"),
            "display": {
                "summary_metrics": summary_metrics,
                "curves": curves,
                "heatmaps": heatmaps,
            },
            "save": {
                "summary": False,
                "curve": False,
                "specific_loudness": False,
            },
            "advanced": {
                "show_specific_loudness_heatmap": self.show_specific_box.isChecked(),
                "save_specific_loudness_npz": False,
                "specific_loudness_colormap": self.specific_colormap_combo.currentText(),
                "curve_y_unit": self.curve_y_unit_combo.currentText(),
                "specific_loudness_profile_mode": (
                    self.SPECIFIC_PROFILE_LABEL_TO_VALUE.get(
                        self.specific_profile_mode_combo.currentText(), "steady_average"
                    )
                    if not self.comparison_only
                    else str(advanced_cfg.get("specific_loudness_profile_mode", "steady_average") or "steady_average")
                ),
                "specific_loudness_exceedance_threshold_sone_per_bark": (
                    self.specific_exceedance_threshold_spin.value()
                    if not self.comparison_only
                    else float(advanced_cfg.get("specific_loudness_exceedance_threshold_sone_per_bark", 0.0) or 0.0)
                ),
                "specific_loudness_exceedance_ref_line": self._exceedance_ref_line_value(advanced_cfg),
                "output_time_resolution_ms": 2.0,
                "stationary_frame_duration_s": _DEFAULT_FRAME_S,
                "stationary_overlap_percent": _DEFAULT_OVERLAP,
                "stationary_hop_duration_s": self._stationary_hop_duration_s(),
                "curve_y_axis_zero_based": bool(advanced_cfg.get("curve_y_axis_zero_based", True)),
            },
            "limit_checked": self.limit_checked_box.isChecked(),
            "limit_metric": self.LIMIT_METRIC_LABEL_TO_VALUE.get(
                self.limit_metric_combo.currentText(), "curve_y"
            ),
            "curve_limit_unit": self._curve_limit_unit(),
            "curve_upper_enabled": self.upper_enabled_box.isChecked(),
            "curve_upper_value": self.upper_spin.value(),
            "curve_lower_enabled": self.lower_enabled_box.isChecked(),
            "curve_lower_value": self.lower_spin.value(),
            "nmax_upper_enabled": False,
            "nmax_upper_sone": 0.0,
            "nmax_lower_enabled": False,
            "nmax_lower_sone": 0.0,
            "mean_upper_enabled": False,
            "mean_upper_sone": 0.0,
            "mean_lower_enabled": False,
            "mean_lower_sone": 0.0,
        }


class LoudnessConfigWindow(QDialog):
    """Configuration dialog for ISO 532-1 Zwicker loudness analysis."""

    def __init__(self, config_manager, model_type):
        super().__init__()
        self.config_manager = config_manager
        self.model_type = model_type
        self.load_config = self.config_manager.load_config().get(model_type, {})
        self.panel = LoudnessConfigPanel(self.load_config, comparison_only=False, parent=self)
        self.init_ui()

    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))
        self.setWindowTitle("响度分析配置")
        self.setMinimumSize(680, 700)

        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.addWidget(self.panel)
        layout.addLayout(self._create_btn_layout())
        self.setLayout(layout)
        self.resize(680, 700)

    def _create_btn_layout(self):
        btn_layout = QHBoxLayout()
        default_btn = PushButton(" 设为默认 ")
        default_btn.clicked.connect(self.on_default_btn_clicked)
        ok_btn = PushButton(" 确  认 ")
        ok_btn.clicked.connect(self.on_click_ok_btn)
        btn_layout.addWidget(default_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        return btn_layout

    def get_default_config(self):
        return self.panel.get_default_config()

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        save_flag = self.config_manager.save_default_config("LOUD", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        self._accepted_config = self.get_default_config()
        self.accept()
        return self._accepted_config
