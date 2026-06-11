"""
HD (Harmonic Distortion / THD) 分析配置对话框
"""

from PyQt5.QtWidgets import QVBoxLayout

from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.popuputils import PopupUtils
from ui.ui_analysis_config.common_widgets import AnalysisConfigDialogBase, GoldenSampleWidget, HarmonicSelectorWidget
from ui.ui_analysis_config.threshold_config_widget import ThresholdConfigWidget
from ui.custom_ui_widget.widgets import GroupBox, MessageBox
from ui.ui_src import ui_resources


class HdConfigWindow(AnalysisConfigDialogBase):
    """谐波失真 (THD) 分析配置对话框"""

    def __init__(self, config_manager, model_type):
        super().__init__(disable_close_button=True)
        self.config_manager = config_manager
        self.model_type = model_type
        self.load_config = self.config_manager.load_config().get(model_type, {})
        self.init_ui()

    def init_ui(self):
        self.setObjectName("HdConfigWindow")
        self.setMinimumSize(320, 480)
        self.resize(380, 620)

        layout = QVBoxLayout()

        # 谐波选择组
        harmonic_group_box = GroupBox("谐波失真")
        harmonic_slider_layout = QVBoxLayout()
        harmonic_slider_layout.setSpacing(12)
        self.harmonic_selector = HarmonicSelectorWidget(self.load_config, start_order=2, end_order=35, parent=self)
        harmonic_slider_layout.addWidget(self.harmonic_selector)
        harmonic_group_box.setLayout(harmonic_slider_layout)

        self.golden_chk_box = GoldenSampleWidget(self.load_config, self)

        # 阈值配置组件
        self.threshold_widget = ThresholdConfigWidget(
            parent=self,
            load_config=self.load_config,
            model_type=self.model_type,
        )

        btn_layout = self.create_btn()

        layout.addWidget(harmonic_group_box)
        layout.addWidget(self.golden_chk_box)
        layout.addWidget(self.threshold_widget)
        layout.addStretch()
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def create_btn(self):
        return self.create_standard_button_layout(self.on_default_btn_clicked, self.on_click_ok_btn)

    def get_default_config(self):
        """获取配置数据"""
        config = {}
        config.update(self.harmonic_selector.get_config())
        config.update(self.golden_chk_box.get_config())
        config.update(self.threshold_widget.get_config())
        return config

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        if not self.threshold_widget.validate():
            return
        save_flag = self.config_manager.save_default_config("HD", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        if not self.harmonic_selector.selected_labels():
            MessageBox.warning(self, "设置警告", "请选择谐波失真阶数")
        else:
            config_data = self.get_default_config()
            if not self.threshold_widget.validate():
                return
            self.accept()
            return config_data
