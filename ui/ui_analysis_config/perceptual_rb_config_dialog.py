"""
PRB (Perceptual Rub & Buzz) 分析配置对话框

PRB 使用固定谐波范围 (2阶-35阶) 结合 SoundCheck/Listen (SC) 心理声学模型，
计算感知失真响度。
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout

from ui.custom_ui_widget.popuputils import PopupUtils
from ui.ui_analysis_config.common_widgets import GoldenSampleWidget, SemanticAnalysisConfigDialogBase
from ui.ui_analysis_config.threshold_config_widget import ThresholdConfigWidget
from ui.custom_ui_widget.widgets import GroupBox, Label, ComboBox


class PerceptualRbConfigWindow(SemanticAnalysisConfigDialogBase):
    """
    Perceptual Rub & Buzz (PRB) 分析配置对话框

    PRB 使用固定谐波范围 (2阶-35阶) 和 SC 模型计算。
    该对话框允许用户选择输出指标以及配置阈值曲线。
    """

    def __init__(self, config_manager, model_type):
        super().__init__(disable_close_button=True)
        self.config_manager = config_manager
        self.model_type = model_type
        self.load_config = self.config_manager.load_config().get(model_type, {}) if self.config_manager else {}
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
        group = GroupBox("PRB")
        group.setObjectName("prb_group_box")
        group_layout = QVBoxLayout()

        self.sc_metric_desc = Label("选择输出结果：")
        self.sc_metric_desc.setAlignment(Qt.AlignLeft)
        self.sc_metric_combo = ComboBox()
        self.sc_metric_combo.addItem("感知失真指数", "totalnl_x_ehs")
        # self.sc_metric_combo.addItem("感知失真响度", "totalnl")  # 功能尚未完善，暂时禁用

        saved_masking = self.load_config.get("masking_config", {})
        if not isinstance(saved_masking, dict):
            saved_masking = {}
        saved_metric = str(saved_masking.get("sc_metric", "totalnl_x_ehs")).strip().lower()
        if saved_metric == "totalnl_phons":
            saved_metric = "totalnl"
        # 由于"感知失真响度"选项已禁用，强制使用默认选项
        if saved_metric not in {"totalnl_x_ehs"}:
            saved_metric = "totalnl_x_ehs"
        idx_metric = self.sc_metric_combo.findData(saved_metric)
        if idx_metric >= 0:
            self.sc_metric_combo.setCurrentIndex(idx_metric)

        group_layout.addWidget(self.sc_metric_desc)
        group_layout.addWidget(self.sc_metric_combo)
        group.setLayout(group_layout)

        self.golden_chk_box = GoldenSampleWidget(self.load_config, self)

        # 阈值配置组件
        self.threshold_widget = ThresholdConfigWidget(
            parent=self,
            load_config=self.load_config,
            model_type=self.model_type,
            allow_manual_limits=True,
        )

        self.add_semantic_section("compute", widget=group)
        self.add_semantic_section("reference", widget=self.golden_chk_box)
        self.add_threshold_curve_sections(self.threshold_widget, self.load_config)
        self.enable_plot_view_config(self.load_config, "Hz", "phon", True, True, True)

    def get_default_config(self):
        """获取配置数据"""
        metric = self.sc_metric_combo.currentData()
        if metric == "totalnl_phons":
            metric = "totalnl"
        if metric not in {"totalnl_x_ehs", "totalnl"}:
            metric = "totalnl_x_ehs"

        masking_config = {}
        saved_masking = self.load_config.get("masking_config", {})
        if isinstance(saved_masking, dict):
            masking_config.update(saved_masking)
        masking_config["sc_metric"] = metric

        config = {"prb_method": "sc", "masking_config": masking_config}
        config.update(self.golden_chk_box.get_config())
        config.update(self.threshold_widget.get_config())
        return self.merge_plot_view_config(config)

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        if not self.validate_plot_view_config():
            return
        if not self.threshold_widget.validate():
            return
        save_flag = self.config_manager.save_default_config("PRB", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_restore_default_btn_clicked(self):
        self.load_config = self.config_manager.load_config().get(self.model_type, {}) if self.config_manager else {}
        self.clear_semantic_sections()
        self._build_semantic_sections()

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        if not self.validate_plot_view_config():
            return
        if not self.threshold_widget.validate():
            return
        self.accept()
        return config_data
