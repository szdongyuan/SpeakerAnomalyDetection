from typing import List, Optional

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from base.core_algorithm.sound_quality.noise_reduction import (
    DEFAULT_SPECTRAL_SUBTRACTION_ALPHA,
    DEFAULT_SPECTRAL_SUBTRACTION_FLOOR,
    DEFAULT_SPECTRAL_SUBTRACTION_FREQ_SMOOTHING_BINS,
    DEFAULT_SPECTRAL_SUBTRACTION_GAIN_SMOOTHING,
    DEFAULT_SPECTRAL_SUBTRACTION_HOP_SIZE,
    DEFAULT_SPECTRAL_SUBTRACTION_MIN_GAIN_DB,
    DEFAULT_SPECTRAL_SUBTRACTION_N_FFT,
)
from base.core_algorithm.sound_quality.psychoacoustic_constants import (
    LOUDNESS_DEFAULT_OUTPUT_TIME_RESOLUTION_MS,
    LOUDNESS_DEFAULT_STATIONARY_FRAME_DURATION_S,
    LOUDNESS_DEFAULT_STATIONARY_HOP_DURATION_S,
    LOUDNESS_DEFAULT_STATIONARY_OVERLAP_PERCENT,
)
from consts.running_consts import DEFAULT_DIR

_DEFAULT_FRAME_S = LOUDNESS_DEFAULT_STATIONARY_FRAME_DURATION_S
_DEFAULT_HOP_S = LOUDNESS_DEFAULT_STATIONARY_HOP_DURATION_S
_DEFAULT_OVERLAP = LOUDNESS_DEFAULT_STATIONARY_OVERLAP_PERCENT
_DEFAULT_OUTPUT_TIME_RESOLUTION_MS = LOUDNESS_DEFAULT_OUTPUT_TIME_RESOLUTION_MS
from ui.custom_ui_widget.popuputils import PopupUtils
from ui.custom_ui_widget.widgets import CheckBox, ComboBox, DoubleSpinBox, GroupBox, Label, LineEdit
from ui.ui_analysis_config.common_widgets import (
    AnalysisTimeRangeWidget,
    ChannelSelectorWidget,
    SemanticAnalysisConfigDialogBase,
)
from ui.ui_src import ui_resources


class LoudnessConfigPanel(QWidget):
    """Reusable editor for the LOUD item config."""

    METHOD_VALUE_TO_LABEL = {
        "time_varying_iso532_1": "时变响度",
        "per_segment": "分段响度",
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
        self.algorithm_group = self._create_algorithm_group()
        self.output_group = self._create_output_group()
        self.limit_group = self._create_limit_group()

        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.algorithm_group)
        layout.addWidget(self.output_group)
        layout.addWidget(self.limit_group)
        layout.addStretch()
        self.setLayout(layout)

    def _create_algorithm_group(self):
        group = QWidget()
        layout = QVBoxLayout()
        advanced_cfg = self.load_config.get("advanced", {}) or {}

        method_layout = QHBoxLayout()
        method_layout.addWidget(Label("计算模式:"))
        self.method_combo = ComboBox()
        self.method_combo.addItems(list(self.METHOD_LABEL_TO_VALUE.keys()))
        method_value = str(self.load_config.get("method", "time_varying_iso532_1") or "time_varying_iso532_1")
        self.method_combo.setCurrentText(self.METHOD_VALUE_TO_LABEL.get(method_value, "时变响度"))
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

        self.analysis_time_range_widget = AnalysisTimeRangeWidget(
            self.load_config.get("advanced", {}), self, show_checkbox=True
        )
        layout.addWidget(self.analysis_time_range_widget)

        noise_layout = QVBoxLayout()
        noise_layout.setSpacing(8)
        self.background_noise_enabled_box = CheckBox("启用背景噪音处理")
        self.background_noise_enabled_box.setToolTip(
            "启用后先用背景噪声音频做保守谱减，再计算响度。"
        )
        self.background_noise_enabled_box.setChecked(
            bool(advanced_cfg.get("background_noise_processing_enabled", False))
        )
        noise_layout.addWidget(self.background_noise_enabled_box)

        self.background_noise_file_widget = QWidget(self)
        file_layout = QHBoxLayout(self.background_noise_file_widget)
        file_layout.setContentsMargins(18, 0, 0, 0)
        file_layout.addWidget(Label("背景音频:"))
        self.background_noise_path_edit = LineEdit()
        self.background_noise_path_edit.setReadOnly(True)
        self.background_noise_path_edit.setText(
            str(advanced_cfg.get("background_noise_file_path", "") or "")
        )
        icon = QIcon(":/ui/icon/folder-s.png")
        action = self.background_noise_path_edit.addAction(icon, LineEdit.TrailingPosition)
        action.setToolTip("选择背景噪声音频")
        action.triggered.connect(self._on_background_noise_file_clicked)
        file_layout.addWidget(self.background_noise_path_edit, 1)
        noise_layout.addWidget(self.background_noise_file_widget)
        self.background_noise_enabled_box.toggled.connect(
            self.background_noise_file_widget.setVisible
        )
        self.background_noise_file_widget.setVisible(
            self.background_noise_enabled_box.isChecked()
        )
        layout.addLayout(noise_layout)

        group.setLayout(layout)
        return group

    def _on_background_noise_file_clicked(self):
        from PyQt5.QtWidgets import QFileDialog

        start_dir = DEFAULT_DIR
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择背景噪声音频文件",
            start_dir,
            "Audio Files (*.wav *.WAV);;All Files (*)",
        )
        if path:
            self.background_noise_path_edit.setText(path)

    def _create_output_group(self):
        group = QWidget()
        group.setMinimumHeight(500)
        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(12, 18, 12, 14)

        display_cfg = self.load_config.get("display", {}) or {}
        advanced_cfg = self.load_config.get("advanced", {}) or {}

        metric_group = GroupBox("指标")
        metric_group.setMinimumHeight(220)
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
        graph_group.setMinimumHeight(230)
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
        layout.addWidget(graph_group)

        group.setLayout(layout)
        return group

    def _create_limit_group(self):
        group = QWidget()
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
        for value, label in (
            ("curve_y", "响度曲线逐点判定"),
            ("steady_state_average", "稳态平均响度"),
            ("max_transient", "最大瞬态响度"),
        ):
            self.limit_metric_combo.addItem(label, value)
        limit_metric_value = str(self.load_config.get("limit_metric", "curve_y") or "curve_y")
        limit_metric_index = self.limit_metric_combo.findData(limit_metric_value)
        if limit_metric_index < 0:
            limit_metric_index = self.limit_metric_combo.findData("curve_y")
        self.limit_metric_combo.setCurrentIndex(limit_metric_index if limit_metric_index >= 0 else 0)
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
            "method": self.METHOD_LABEL_TO_VALUE.get(self.method_combo.currentText(), "time_varying_iso532_1"),
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
                "output_time_resolution_ms": self._positive_float_or_default(
                    advanced_cfg.get("output_time_resolution_ms", _DEFAULT_OUTPUT_TIME_RESOLUTION_MS),
                    _DEFAULT_OUTPUT_TIME_RESOLUTION_MS,
                ),
                "stationary_frame_duration_s": _DEFAULT_FRAME_S,
                "stationary_overlap_percent": _DEFAULT_OVERLAP,
                "stationary_hop_duration_s": self._stationary_hop_duration_s(),
                "curve_y_axis_zero_based": bool(advanced_cfg.get("curve_y_axis_zero_based", True)),
                **self.analysis_time_range_widget.get_config(),
                "background_noise_processing_enabled": self.background_noise_enabled_box.isChecked(),
                "background_noise_file_path": self.background_noise_path_edit.text().strip(),
                "background_noise_algorithm": "spectral_subtraction_audio",
                "background_noise_n_fft": int(
                    advanced_cfg.get("background_noise_n_fft", DEFAULT_SPECTRAL_SUBTRACTION_N_FFT)
                    or DEFAULT_SPECTRAL_SUBTRACTION_N_FFT
                ),
                "background_noise_hop_size": int(
                    advanced_cfg.get("background_noise_hop_size", DEFAULT_SPECTRAL_SUBTRACTION_HOP_SIZE)
                    or DEFAULT_SPECTRAL_SUBTRACTION_HOP_SIZE
                ),
                "background_noise_oversubtraction_factor": float(
                    advanced_cfg.get(
                        "background_noise_oversubtraction_factor",
                        DEFAULT_SPECTRAL_SUBTRACTION_ALPHA,
                    )
                    or DEFAULT_SPECTRAL_SUBTRACTION_ALPHA
                ),
                "background_noise_spectral_floor": float(
                    advanced_cfg.get("background_noise_spectral_floor", DEFAULT_SPECTRAL_SUBTRACTION_FLOOR)
                    or DEFAULT_SPECTRAL_SUBTRACTION_FLOOR
                ),
                "background_noise_min_gain_db": float(
                    advanced_cfg.get("background_noise_min_gain_db", DEFAULT_SPECTRAL_SUBTRACTION_MIN_GAIN_DB)
                    or DEFAULT_SPECTRAL_SUBTRACTION_MIN_GAIN_DB
                ),
                "background_noise_frequency_smoothing_bins": int(
                    advanced_cfg.get(
                        "background_noise_frequency_smoothing_bins",
                        DEFAULT_SPECTRAL_SUBTRACTION_FREQ_SMOOTHING_BINS,
                    )
                    or DEFAULT_SPECTRAL_SUBTRACTION_FREQ_SMOOTHING_BINS
                ),
                "background_noise_gain_time_smoothing": float(
                    advanced_cfg.get(
                        "background_noise_gain_time_smoothing",
                        DEFAULT_SPECTRAL_SUBTRACTION_GAIN_SMOOTHING,
                    )
                    or DEFAULT_SPECTRAL_SUBTRACTION_GAIN_SMOOTHING
                ),
            },
            "limit_checked": self.limit_checked_box.isChecked(),
            "limit_metric": str(self.limit_metric_combo.currentData() or "curve_y"),
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


class LoudnessConfigWindow(SemanticAnalysisConfigDialogBase):
    """Configuration dialog for ISO 532-1 Zwicker loudness analysis."""

    def __init__(self, config_manager, model_type, available_channels: Optional[List[int]] = None):
        super().__init__(disable_close_button=True)
        self.config_manager = config_manager
        self.model_type = model_type
        self.load_config = self.config_manager.load_config().get(model_type, {})
        self.show_channel_selector = available_channels is not None
        self.available_channels = available_channels
        self.panel = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("响度分析配置")
        self.apply_semantic_dialog_size()
        self.set_semantic_button_callbacks(
            default_callback=self.on_default_btn_clicked,
            restore_callback=self.on_restore_default_btn_clicked,
            ok_callback=self.on_click_ok_btn,
        )
        self._build_semantic_sections()

    def _build_semantic_sections(self):
        if self.show_channel_selector:
            self.channel_selector = ChannelSelectorWidget(self.load_config, self.available_channels, self)
            self.add_semantic_section("input", title="输入参数", widget=self.channel_selector)

        self.panel = LoudnessConfigPanel(self.load_config, comparison_only=False, parent=self)
        self.add_semantic_section("compute", title="算法参数", widget=self.panel.algorithm_group)
        self.add_semantic_section("display", title="显示设置", widget=self.panel.output_group)
        self.add_semantic_section("judgment", title="判定阈值", widget=self.panel.limit_group)

    def get_default_config(self):
        config = self.panel.get_default_config()
        if self.show_channel_selector and hasattr(self, "channel_selector"):
            config.update(self.channel_selector.get_config())
        else:
            config["analysis_channel"] = int(self.load_config.get("analysis_channel", 0) or 0)
        return config

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        save_flag = self.config_manager.save_default_config(self.model_type, config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_restore_default_btn_clicked(self):
        self.load_config = self.config_manager.load_config().get(self.model_type, {})
        self.clear_semantic_sections()
        self._build_semantic_sections()

    def on_click_ok_btn(self):
        self._accepted_config = self.get_default_config()
        self.accept()
        return self._accepted_config
