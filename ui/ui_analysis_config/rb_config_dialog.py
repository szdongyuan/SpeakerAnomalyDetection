"""
RB (Rub & Buzz) 分析配置对话框

Rub & Buzz 使用高阶谐波失真 (10阶-35阶) 来检测扬声器的摩擦和蜂鸣问题。
"""

from PyQt5.QtWidgets import QVBoxLayout

from ui.custom_ui_widget.popuputils import PopupUtils
from ui.custom_ui_widget.widgets import GroupBox, MessageBox
from ui.ui_analysis_config.common_widgets import (
    GoldenSampleWidget,
    HarmonicDetectionMethodSelectorWidget,
    HarmonicSelectorWidget,
    SemanticAnalysisConfigDialogBase,
)
from ui.ui_analysis_config.threshold_config_widget import ThresholdConfigWidget


class RbConfigWindow(SemanticAnalysisConfigDialogBase):
    """
    Rub & Buzz 分析配置对话框

    允许选择 10阶-35阶 谐波进行分析，并支持阈值曲线配置。
    """

    def __init__(self, config_manager, model_type):
        super().__init__(disable_close_button=True)
        self.config_manager = config_manager
        self.model_type = model_type
        self.load_config = self.config_manager.load_config().get(model_type, {})
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
        harmonic_group_box = GroupBox("Rub & Buzz")
        harmonic_group_box.setObjectName("harmonic_group_box")
        harmonic_slider_layout = QVBoxLayout()
        harmonic_slider_layout.setSpacing(12)
        self.detection_method_selector = HarmonicDetectionMethodSelectorWidget(self.load_config, parent=self)
        self.harmonic_selector = HarmonicSelectorWidget(self.load_config, start_order=10, end_order=35, parent=self)
        harmonic_slider_layout.addWidget(self.detection_method_selector)
        harmonic_slider_layout.addWidget(self.harmonic_selector)
        harmonic_group_box.setLayout(harmonic_slider_layout)

        self.golden_chk_box = GoldenSampleWidget(self.load_config, self)

        # 阈值配置组件
        self.threshold_widget = ThresholdConfigWidget(
            parent=self,
            load_config=self.load_config,
            model_type=self.model_type,
            allow_manual_limits=True,
        )

        self.add_semantic_section("detection", widget=harmonic_group_box)
        self.add_semantic_section("reference", widget=self.golden_chk_box)
        self.add_threshold_curve_sections(self.threshold_widget, self.load_config)
        self.enable_plot_view_config(self.load_config, "Hz", "%", True, True, True)

    def create_btn(self):
        return self.create_standard_button_layout(self.on_default_btn_clicked, self.on_click_ok_btn)

    def get_default_config(self):
        """获取配置数据"""
        config = {}
        config.update(self.detection_method_selector.get_config())
        config.update(self.harmonic_selector.get_config())
        config.update(self.golden_chk_box.get_config())
        config.update(self.threshold_widget.get_config())
        return self.merge_plot_view_config(config)

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        if not self.validate_plot_view_config():
            return
        if not self.threshold_widget.validate():
            return
        save_flag = self.config_manager.save_default_config("RB", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_restore_default_btn_clicked(self):
        self.load_config = self.config_manager.load_config().get(self.model_type, {})
        self.clear_semantic_sections()
        self._build_semantic_sections()

    def on_click_ok_btn(self):
        if not self.harmonic_selector.selected_labels():
            MessageBox.warning(self, "设置警告", "请选择Rub & Buzz阶数")
        else:
            config_data = self.get_default_config()
            if not self.validate_plot_view_config():
                return
            if not self.threshold_widget.validate():
                return
            self.accept()
            return config_data
