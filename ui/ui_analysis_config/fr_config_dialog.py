"""
FR (Frequency Response) 分析配置对话框
"""

from ui.custom_ui_widget.popuputils import PopupUtils
from ui.ui_analysis_config.common_widgets import (
    GoldenSampleWidget,
    OctaveSmoothingSelectorWidget,
    SemanticAnalysisConfigDialogBase,
)
from ui.ui_analysis_config.threshold_config_widget import ThresholdConfigWidget


class FrConfigWindow(SemanticAnalysisConfigDialogBase):
    """FR 频率响应分析配置对话框"""

    def __init__(self, config_manager, model_type):
        super().__init__(disable_close_button=True)
        self.config_manager = config_manager
        self.config_key = model_type
        self.model_type = model_type
        self.load_config = self.config_manager.load_config().get(self.config_key, {})
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
        self.smoothing_selector = OctaveSmoothingSelectorWidget(self.load_config, parent=self)
        self.golden_chk_box = GoldenSampleWidget(self.load_config, self)

        # 使用通用阈值配置组件
        self.threshold_widget = ThresholdConfigWidget(
            parent=self,
            load_config=self.load_config,
            model_type=self.model_type,
            allow_manual_limits=True,
            limit_value_semantics_provider=self.golden_chk_box.limit_value_semantics,
        )

        self.add_semantic_section("compute", widget=self.smoothing_selector)
        self.add_semantic_section("reference", widget=self.golden_chk_box)
        self.add_threshold_curve_sections(self.threshold_widget, self.load_config)
        self.enable_plot_view_config(self.load_config, "Hz", "dB", True, True, True)

    def create_btn(self):
        return self.create_standard_button_layout(self.on_default_btn_clicked, self.on_click_ok_btn)

    def get_default_config(self):
        """获取配置数据"""
        config = {}
        config.update(self.smoothing_selector.get_config())
        config.update(self.golden_chk_box.get_config())
        config.update(self.threshold_widget.get_config())
        return self.merge_plot_view_config(config)

    def on_default_btn_clicked(self):
        if not self.validate_plot_view_config():
            return
        if not self.golden_chk_box.validate():
            return
        if not self.threshold_widget.validate():
            return
        config_data = self.get_default_config()
        save_flag = self.config_manager.save_default_config("FR", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_restore_default_btn_clicked(self):
        self.load_config = self.config_manager.load_config().get(self.config_key, {})
        self.clear_semantic_sections()
        self._build_semantic_sections()

    def on_click_ok_btn(self):
        if not self.validate_plot_view_config():
            return
        if not self.golden_chk_box.validate():
            return
        if not self.threshold_widget.validate():
            return
        config_data = self.get_default_config()
        self.accept()
        return config_data
