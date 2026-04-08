"""
RB (Rub & Buzz) 分析配置对话框

Rub & Buzz 使用高阶谐波失真 (10阶-35阶) 来检测扬声器的摩擦和蜂鸣问题。
"""

from functools import partial

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QVBoxLayout,
    QMessageBox,
    QPushButton,
    QLabel,
    QSizePolicy,
    QScrollArea,
    QWidget,
)

from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.popuputils import PopupUtils
from ui.ui_analysis_config.threshold_config_widget import ThresholdConfigWidget


class RbConfigWindow(QDialog):
    """
    Rub & Buzz 分析配置对话框

    允许选择 10阶-35阶 谐波进行分析，并支持阈值曲线配置。
    """

    selected_labels_changed = pyqtSignal()

    def __init__(self, config_manager, model_type):
        super().__init__()
        self.config_manager = config_manager
        self.model_type = model_type
        self.load_config = self.config_manager.load_config().get(model_type, {})
        # Filter harmonics to valid range (10-35)
        loaded_labels = self.load_config.get("selected_labels", [])
        self.selected_labels = [h for h in loaded_labels if 10 <= h <= 35]
        self.init_ui()

    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setMinimumSize(320, 480)
        self.resize(380, 620)

        layout = QVBoxLayout()

        # 谐波选择组
        harmonic_group_box = QGroupBox("Rub & Buzz")
        harmonic_group_box.setObjectName("harmonic_group_box")
        harmonic_group_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        harmonic_slider_layout = self.create_harmonic_slider_layout()
        harmonic_slider_layout.setSpacing(12)
        self.select_all_check = QCheckBox("全选")
        self.select_all_check.setChecked(self.load_config.get("all_checked", False))
        self.select_all_check.stateChanged.connect(self.on_select_all_changed)
        harmonic_slider_layout.addStretch()
        harmonic_slider_layout.addWidget(self.select_all_check)
        harmonic_group_box.setLayout(harmonic_slider_layout)

        # Golden sample checkbox (placed above threshold widget)
        self.golden_chk_box = QCheckBox("使用黄金样本")
        self.golden_chk_box.setChecked(self.load_config.get("golden_sample_checked", False))

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

        self.setStyleSheet(
            ui_style_const.qgroupbox_style
            + ui_style_const.qcheckbox_style
            + ui_style_const.qpushbutton_style
            + ui_style_const.qlabel_style
            + ui_style_const.qlineedit_style
            + ui_style_const.qradiobutton_style
            + ui_style_const.qdoublespinbox_style
        )

    def create_harmonic_slider_layout(self):
        """创建谐波选择布局，范围 10-35"""
        harmonic_slider_layout = QVBoxLayout()
        self.scroll_area = QScrollArea()
        self.scroll_area.setFixedSize(120, 150)
        box_container = QWidget()
        self.box_layout = QVBoxLayout()
        # Rub & Buzz: harmonics 10-35 (26 harmonics total)
        for i in range(10, 36):
            label = QLabel("  " + str(i))
            label.setMinimumWidth(90)
            label.setMinimumHeight(25)
            label.setAlignment(Qt.AlignLeft)
            label.setStyleSheet("QLabel:focus { outline: none; }")
            label.setAutoFillBackground(True)
            label.mousePressEvent = partial(self.on_label_click, label)
            label.enterEvent = partial(self.on_label_enter, label)
            label.leaveEvent = partial(self.on_label_leave, label)
            if i in self.selected_labels:
                label.setText("\u2713" + label.text().strip())
            self.box_layout.addWidget(label)
        if self.load_config.get("all_checked"):
            self.scroll_area.setDisabled(True)
        box_container.setLayout(self.box_layout)
        self.scroll_area.setWidget(box_container)
        harmonic_slider_layout.addWidget(self.scroll_area)
        harmonic_slider_layout.addStretch()
        return harmonic_slider_layout

    def on_select_all_changed(self, state):
        self.get_default_config()
        if state == Qt.Checked:
            self.scroll_area.setDisabled(True)
            self.scroll_area.setStyleSheet("color: rgb(162, 162, 162);")
            # Select all rub&buzz harmonics (10-35)
            self.selected_labels = list(range(10, 36))
            for i in range(self.box_layout.count()):
                label = self.box_layout.itemAt(i).widget()
                text = label.text().strip()
                if not text.startswith("\u2713"):
                    label.setText("\u2713" + text)
        else:
            self.scroll_area.setDisabled(False)
            self.scroll_area.setStyleSheet("color: rgb(0, 0, 0);")
            self.selected_labels = []
            for i in range(self.box_layout.count()):
                label = self.box_layout.itemAt(i).widget()
                text = label.text().strip()
                if text.startswith("\u2713"):
                    label.setText("  " + text[1:])
        self.selected_labels_changed.emit()

    def on_label_click(self, label, event):
        checked_box = "\u2713"
        cleaned_label = "".join(filter(str.isdigit, label.text()))
        label_value = int(cleaned_label)
        if label_value in self.selected_labels:
            self.selected_labels.remove(label_value)
            self.selected_labels.sort()
            label.setText("  " + label.text().replace(checked_box, "").strip())
        else:
            self.selected_labels.append(label_value)
            label.setText(checked_box + label.text().strip())
        self.selected_labels_changed.emit()

    def on_label_enter(self, label, event):
        label.setStyleSheet("background-color: #5099ccff;")

    def on_label_leave(self, label, event):
        label.setStyleSheet("background-color: transparent;")

    def create_btn(self):
        btn_layout = QHBoxLayout()
        default_btn = QPushButton(" 设为默认 ")
        default_btn.clicked.connect(self.on_default_btn_clicked)
        ok_btn = QPushButton(" 确  认 ")
        ok_btn.clicked.connect(self.on_click_ok_btn)
        btn_layout.addWidget(default_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        return btn_layout

    def get_default_config(self):
        """获取配置数据"""
        config = {
            "selected_labels": self.selected_labels,
            "all_checked": self.select_all_check.isChecked(),
            "golden_sample_checked": self.golden_chk_box.isChecked(),
        }
        config.update(self.threshold_widget.get_config())
        return config

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        if not self.threshold_widget.validate():
            return
        save_flag = self.config_manager.save_default_config("RB", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        if not self.selected_labels:
            QMessageBox.warning(self, "设置警告", "请选择Rub & Buzz阶数")
        else:
            config_data = self.get_default_config()
            if not self.threshold_widget.validate():
                return
            self.accept()
            return config_data
