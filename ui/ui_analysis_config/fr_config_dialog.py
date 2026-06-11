"""
FR (Frequency Response) 分析配置对话框
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout

from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.popuputils import PopupUtils
from ui.ui_analysis_config.config_normalization import normalize_octave_smoothing
from ui.ui_analysis_config.threshold_config_widget import ThresholdConfigWidget
from ui.custom_ui_widget.widgets import CheckBox, ComboBox, Label, PushButton
from ui.ui_src import ui_resources


class FrConfigWindow(QDialog):
    """FR 频率响应分析配置对话框"""

    OCTAVE_SMOOTHING_LABELS = {
        "不平滑": 0,
        "1/1 Oct": 1,
        "1/3 Oct": 3,
        "1/6 Oct": 6,
        "1/12 Oct": 12,
        "1/24 Oct": 24,
        "1/48 Oct": 48,
    }

    def __init__(self, config_manager, model_type):
        super().__init__()
        self.config_manager = config_manager
        self.model_type = model_type
        self.load_config = self.config_manager.load_config().get(model_type, {})
        self.init_ui()

    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))
        # 默认高度偏小会把阈值绘图区域压缩得很矮
        self.setMinimumSize(380, 420)
        self.resize(380, 450)

        layout = QVBoxLayout()

        # Octave smoothing
        self.smooth_combo_box = ComboBox()
        self.smooth_combo_box.addItems(list(self.OCTAVE_SMOOTHING_LABELS.keys()))

        selected_oct = normalize_octave_smoothing(self.load_config, default=0)
        selected_label = next(
            (k for k, v in self.OCTAVE_SMOOTHING_LABELS.items() if v == selected_oct),
            "不平滑",
        )
        self.smooth_combo_box.setCurrentText(selected_label)

        # Golden sample checkbox (placed above threshold widget)
        self.golden_chk_box = CheckBox("使用黄金样本")
        self.golden_chk_box.setChecked(self.load_config.get("golden_sample_checked", False))

        # 使用通用阈值配置组件
        self.threshold_widget = ThresholdConfigWidget(
            parent=self,
            load_config=self.load_config,
            model_type=self.model_type,
        )

        btn_layout = self.create_btn()

        layout.addWidget(Label("平滑"))
        layout.addWidget(self.smooth_combo_box)
        layout.addWidget(self.golden_chk_box)
        layout.addWidget(self.threshold_widget)
        layout.addStretch()
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def create_btn(self):
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
        """获取配置数据"""
        smooth_label = self.smooth_combo_box.currentText()
        config = {
            "octave_smoothing": int(self.OCTAVE_SMOOTHING_LABELS.get(smooth_label, 0)),
            "golden_sample_checked": self.golden_chk_box.isChecked(),
        }
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
