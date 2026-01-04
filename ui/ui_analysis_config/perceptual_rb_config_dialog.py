from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.popuputils import PopupUtils


class PerceptualRbConfigWindow(QDialog):
    """
    Configuration dialog for Perceptual Rub & Buzz analysis (PRB).

    PRB is computed with a fixed harmonic range (2nd-35th) using the SoundCheck/Listen (SC) model.
    This dialog allows users to select the output metric: PRB Index (TotalNL×EHS) or PRB Loudness (TotalNL).
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
        self.setMinimumSize(320, 160)
        self.resize(360, 180)

        root_layout = QVBoxLayout()

        group = QGroupBox("PRB")
        group.setObjectName("prb_group_box")
        group_layout = QVBoxLayout()

        # 输出结果选择
        self.sc_metric_desc = QLabel("选择输出结果：")
        self.sc_metric_desc.setAlignment(Qt.AlignLeft)
        self.sc_metric_combo = QComboBox()
        self.sc_metric_combo.addItem("感知失真指数", "totalnl_x_ehs")
        # self.sc_metric_combo.addItem("感知失真响度", "totalnl")  # 功能尚未完善，暂时禁用

        saved_masking = self.load_config.get("masking_config", {})
        if not isinstance(saved_masking, dict):
            saved_masking = {}
        saved_metric = str(saved_masking.get("sc_metric", "totalnl_x_ehs")).strip().lower()
        if saved_metric == "totalnl_phons":
            saved_metric = "totalnl"
        # Map old "ehs" option to default
        # 由于"感知失真响度"选项已禁用，强制使用默认选项
        if saved_metric not in {"totalnl_x_ehs"}:
            saved_metric = "totalnl_x_ehs"
        idx_metric = self.sc_metric_combo.findData(saved_metric)
        if idx_metric >= 0:
            self.sc_metric_combo.setCurrentIndex(idx_metric)

        group_layout.addWidget(self.sc_metric_desc)
        group_layout.addWidget(self.sc_metric_combo)
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
        return {"prb_method": "sc", "masking_config": masking_config}

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        save_flag = self.config_manager.save_default_config("PRB", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        self.accept()
        return config_data
