from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog, QGroupBox, QHBoxLayout, QVBoxLayout, QPushButton
from PyQt5.QtWidgets import QLabel, QMessageBox, QComboBox, QSizePolicy, QCheckBox, QWidget

from base.training_model_management import TrainingModelManagement
from consts import ui_style_const, error_code
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.popuputils import PopupUtils


class AIConfigWindow(QDialog):
    def __init__(self, config_manager, model_type, signal_len=None):
        super().__init__()
        self.signal_len = signal_len
        self.config_manager = config_manager
        self.model_list = self.load_model_name_from_db()
        self.load_config = self.config_manager.load_config().get(model_type, {})
        self.init_ui()

    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setMinimumSize(350, 350)
        self.resize(420, 360)
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        model_box = self.create_model_layout()
        advanced_box = self.create_advanced_settings_box()
        btn_layout = self.create_btn()
        layout.addWidget(model_box)
        layout.addWidget(advanced_box)
        layout.addStretch()
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.setStyleSheet(
            ui_style_const.qgroupbox_style
            + ui_style_const.qpushbutton_style
            + ui_style_const.qlabel_style
            + ui_style_const.qcombobox_style
            + ui_style_const.qcheckbox_style
            + """
            QGroupBox#aiModelBox, QGroupBox#aiAdvancedBox {
                margin-top: 8px;
            }
            QGroupBox#aiModelBox::title, QGroupBox#aiAdvancedBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 0px;
                padding: 0px 2px 0px 0px;
            }
            """
        )

    def cheack_model_list(self):
        if self.analyse_model_combo_box.count() == 0:
            QMessageBox.warning(self, "设置警告", "没有可用的AI模型选型,请检查配置!")

    def create_model_layout(self):
        model_box = QGroupBox("模型")
        model_box.setObjectName("aiModelBox")
        model_box.setMinimumHeight(118)
        analyse_model_label = QLabel("分析模型:")
        self.analyse_model_combo_box = QComboBox(self)
        self.analyse_model_combo_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.analyse_model_combo_box.setFixedHeight(30)
        for model_name in self.model_list:
            self.analyse_model_combo_box.addItem(model_name)
        self.analyse_model_combo_box.setCurrentText(self.load_config.get("analyse_model_name"))
        self.analyse_model_combo_box.currentTextChanged.connect(self.get_default_config)
        QTimer.singleShot(5, self.cheack_model_list)
        analyse_model_combo_layout = QHBoxLayout()
        analyse_model_combo_layout.setContentsMargins(12, 18, 12, 12)
        analyse_model_combo_layout.addWidget(analyse_model_label)
        analyse_model_combo_layout.addWidget(self.analyse_model_combo_box)
        analyse_model_combo_layout.setSpacing(10)
        model_box.setLayout(analyse_model_combo_layout)
        return model_box

    def create_advanced_settings_box(self):
        advanced_box = QGroupBox("高级设置")
        advanced_box.setObjectName("aiAdvancedBox")
        box_layout = QVBoxLayout()
        box_layout.setContentsMargins(12, 18, 12, 12)
        box_layout.setSpacing(10)

        self.channel_model_switch_box = QCheckBox("按通道数切换模型")
        channel_switch_config = self.get_channel_model_switch_config()
        self.channel_model_switch_box.setChecked(bool(channel_switch_config.get("enabled", False)))
        self.channel_model_switch_box.toggled.connect(self.update_channel_model_switch_state)
        box_layout.addWidget(self.channel_model_switch_box)

        self.channel_model_detail_widget = QWidget(self)
        self.channel_model_detail_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        detail_layout = QVBoxLayout()
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(10)

        single_layout = QHBoxLayout()
        single_label = QLabel("单通道模型:")
        self.single_channel_model_combo_box = QComboBox(self)
        self.single_channel_model_combo_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.single_channel_model_combo_box.setFixedHeight(30)
        self.populate_optional_model_combo(
            self.single_channel_model_combo_box,
            channel_switch_config.get("single_channel_model_name", ""),
        )
        single_layout.addWidget(single_label)
        single_layout.addWidget(self.single_channel_model_combo_box)
        detail_layout.addLayout(single_layout)

        multi_layout = QHBoxLayout()
        multi_label = QLabel("多通道模型:")
        self.multi_channel_model_combo_box = QComboBox(self)
        self.multi_channel_model_combo_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.multi_channel_model_combo_box.setFixedHeight(30)
        self.populate_optional_model_combo(
            self.multi_channel_model_combo_box,
            channel_switch_config.get("multi_channel_model_name", ""),
        )
        multi_layout.addWidget(multi_label)
        multi_layout.addWidget(self.multi_channel_model_combo_box)
        detail_layout.addLayout(multi_layout)

        self.channel_model_detail_widget.setLayout(detail_layout)
        box_layout.addWidget(self.channel_model_detail_widget)
        self.channel_model_detail_animation = QPropertyAnimation(self.channel_model_detail_widget, b"maximumHeight", self)
        self.channel_model_detail_animation.setDuration(140)
        self.channel_model_detail_animation.setEasingCurve(QEasingCurve.InOutCubic)
        self.channel_model_detail_animation.finished.connect(self.on_channel_model_detail_animation_finished)

        advanced_box.setLayout(box_layout)
        self.sync_channel_model_detail_visibility(self.channel_model_switch_box.isChecked(), animate=False)
        return advanced_box

    def get_channel_model_switch_config(self):
        channel_switch_config = self.load_config.get("channel_model_switch", {})
        return channel_switch_config if isinstance(channel_switch_config, dict) else {}

    def populate_optional_model_combo(self, combo_box, selected_model_name):
        if hasattr(combo_box, "setPlaceholderText"):
            combo_box.setPlaceholderText("选择模型")
        for model_name in self.model_list:
            combo_box.addItem(model_name, model_name)
        target_model_name = str(selected_model_name or "").strip()
        target_index = combo_box.findData(target_model_name)
        combo_box.setCurrentIndex(target_index if target_index >= 0 else -1)

    def update_channel_model_switch_state(self, enabled):
        is_enabled = bool(enabled)
        self.single_channel_model_combo_box.setEnabled(is_enabled)
        self.multi_channel_model_combo_box.setEnabled(is_enabled)
        self.sync_channel_model_detail_visibility(is_enabled, animate=True)

    def sync_channel_model_detail_visibility(self, is_visible, animate):
        detail_height = self.channel_model_detail_widget.sizeHint().height()
        self.channel_model_detail_animation.stop()

        if not animate:
            self.channel_model_detail_widget.setVisible(bool(is_visible))
            self.channel_model_detail_widget.setMaximumHeight(detail_height if is_visible else 0)
            return

        if is_visible:
            self.channel_model_detail_widget.setVisible(True)
            start_height = max(self.channel_model_detail_widget.maximumHeight(), 0)
            self.channel_model_detail_animation.setStartValue(start_height)
            self.channel_model_detail_animation.setEndValue(detail_height)
        else:
            start_height = self.channel_model_detail_widget.height() or detail_height
            self.channel_model_detail_animation.setStartValue(start_height)
            self.channel_model_detail_animation.setEndValue(0)
        self.channel_model_detail_animation.start()

    def on_channel_model_detail_animation_finished(self):
        if not self.channel_model_switch_box.isChecked():
            self.channel_model_detail_widget.setVisible(False)

    def load_model_name_from_db(self):
        model_list = []
        query_code, query_result = TrainingModelManagement().get_all_model_name_from_db()
        if query_code == error_code.OK:
            for idx, name in enumerate(query_result):
                query_result_idx = query_result[idx]
                input_dim = int(query_result_idx[1].split(" ")[0])
                if not self.signal_len:
                    model_list.append(query_result_idx[0])
                else:
                    if input_dim == self.signal_len:
                        model_list.append(query_result_idx[0])
        return model_list

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
        default_config = {
            "analyse_model_name": self.analyse_model_combo_box.currentText(),
            "channel_model_switch": {
                "enabled": self.channel_model_switch_box.isChecked(),
                "single_channel_model_name": self.single_channel_model_combo_box.currentData() or "",
                "multi_channel_model_name": self.multi_channel_model_combo_box.currentData() or "",
            },
        }
        return default_config

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        save_flag = self.config_manager.save_default_config("AI", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        self.accept()
        return config_data
