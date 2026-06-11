"""
FR (Frequency Response) 分析配置对话框
"""

from PyQt5.QtWidgets import QVBoxLayout

from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.popuputils import PopupUtils
from ui.ui_analysis_config.common_widgets import (
    AnalysisConfigDialogBase,
    GoldenSampleWidget,
    OctaveSmoothingSelectorWidget,
)
from ui.ui_analysis_config.threshold_config_widget import ThresholdConfigWidget
from ui.ui_src import ui_resources


class FrConfigWindow(AnalysisConfigDialogBase):
    """FR 频率响应分析配置对话框"""

    def __init__(self, config_manager, model_type):
        super().__init__(disable_close_button=True)
        self.config_manager = config_manager
        self.model_type = model_type
        self.load_config = self.config_manager.load_config().get(model_type, {})
        self.init_ui()

    def init_ui(self):
        # 默认高度偏小会把阈值绘图区域压缩得很矮
        self.setMinimumSize(380, 420)
        self.resize(380, 450)

        layout = QVBoxLayout()

        self.smoothing_selector = OctaveSmoothingSelectorWidget(self.load_config, parent=self)
        self.golden_chk_box = GoldenSampleWidget(self.load_config, self)

        # 使用通用阈值配置组件
        self.threshold_widget = ThresholdConfigWidget(
            parent=self,
            load_config=self.load_config,
            model_type=self.model_type,
        )

        btn_layout = self.create_btn()

        layout.addWidget(self.smoothing_selector)
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
        config.update(self.smoothing_selector.get_config())
        config.update(self.golden_chk_box.get_config())
        config.update(self.threshold_widget.get_config())
        return config

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        if not self.threshold_widget.validate():
            return
        save_flag = self.config_manager.save_default_config("FR", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        if not self.threshold_widget.validate():
            return
        self.accept()
        return config_data
