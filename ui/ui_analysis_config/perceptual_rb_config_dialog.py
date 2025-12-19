from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QComboBox, QDialog, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.popuputils import PopupUtils


class PerceptualRbConfigWindow(QDialog):
    """
    Configuration dialog for Perceptual Rub & Buzz analysis (PRB).

    PRB is now computed with a fixed harmonic range (2nd-35th) to avoid inconsistent results.
    This dialog only controls which loudness model is used:
      - "sc": Listen/SoundCheck simplified perceptual model (PEAQ-SC paper path)
      - "iso226 and iso 532": ISO-based loudness path (mosqito / Zwicker loudness)
    """

    def __init__(self, config_manager, model_type):
        super().__init__()
        self.config_manager = config_manager
        self.model_type = model_type
        self.load_config = self.config_manager.load_config().get(model_type, {}) if self.config_manager else {}
        self.init_ui()

    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setMinimumSize(360, 220)
        self.resize(420, 240)

        root_layout = QVBoxLayout()

        group = QGroupBox("PRB")
        group.setObjectName("prb_group_box")
        group_layout = QVBoxLayout()

        desc = QLabel("选择 PRB 的响度模型：")
        desc.setAlignment(Qt.AlignLeft)

        self.method_combo = QComboBox()
        # Display text follows the user's requested naming; stored value is the normalized key used in code.
        self.method_combo.addItem("sc", "sc")
        self.method_combo.addItem("iso226 and iso 532", "iso")

        saved = str(self.load_config.get("prb_method", "iso")).strip().lower()
        if saved in {"sc", "soundcheck", "peaq_sc", "peaq-sc"}:
            idx = self.method_combo.findData("sc")
        else:
            idx = self.method_combo.findData("iso")
        if idx >= 0:
            self.method_combo.setCurrentIndex(idx)

        group_layout.addWidget(desc)
        group_layout.addWidget(self.method_combo)
        group.setLayout(group_layout)

        btn_layout = QHBoxLayout()
        default_btn = QPushButton(" 设为默认 ")
        default_btn.clicked.connect(self.on_default_btn_clicked)
        ok_btn = QPushButton(" 确  认 ")
        ok_btn.clicked.connect(self.on_click_ok_btn)
        btn_layout.addWidget(default_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)

        root_layout.addWidget(group)
        root_layout.addStretch()
        root_layout.addLayout(btn_layout)
        self.setLayout(root_layout)

        self.setStyleSheet(
            ui_style_const.qgroupbox_style
            + ui_style_const.qpushbutton_style
            + ui_style_const.qlabel_style
            + ui_style_const.qcombobox_style
        )

    def get_default_config(self):
        method = self.method_combo.currentData()
        if method not in {"sc", "iso"}:
            method = "iso"
        return {"prb_method": method}

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        save_flag = self.config_manager.save_default_config("PRB", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        self.accept()
        return config_data

