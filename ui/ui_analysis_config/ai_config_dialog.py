import os
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog, QGroupBox, QHBoxLayout, QVBoxLayout, QPushButton
from PyQt5.QtWidgets import QLabel, QMessageBox, QComboBox, QSizePolicy
from typing import List, Optional

from base.load_config import load_config
from base.model_runtime_validation import resolve_effective_signal_length
from base.training_model_management import TrainingModelManagement
from consts import ui_style_const, error_code
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.popuputils import PopupUtils


class AIConfigWindow(QDialog):
    def __init__(self, config_manager, model_type, signal_len=None, available_channels: Optional[List[int]] = None):
        super().__init__()
        self.signal_len = signal_len
        self.config_manager = config_manager
        self.model_list = self.load_model_name_from_db()
        self.load_config = self.config_manager.load_config().get(model_type, {})
        self.available_channels = self._normalize_available_channels(available_channels)
        self.init_ui()

    @staticmethod
    def _normalize_available_channels(available_channels):
        channels = []
        try:
            channels = sorted({int(ch) for ch in (available_channels or [])})
        except Exception:
            channels = []
        if not channels:
            channels = [0]
        return channels

    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setMinimumSize(350, 350)
        self.resize(350, 350)
        layout = QVBoxLayout()
        layout.addLayout(self.create_channel_layout())
        model_box = self.create_model_layout()
        btn_layout = self.create_btn()
        layout.addWidget(model_box)
        layout.addStretch()
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.setStyleSheet(
            ui_style_const.qgroupbox_style
            + ui_style_const.qpushbutton_style
            + ui_style_const.qlabel_style
            + ui_style_const.qcombobox_style
        )

    def create_channel_layout(self):
        channel_label = QLabel("通道:")
        self.channel_combo_box = QComboBox(self)
        for ch in self.available_channels:
            self.channel_combo_box.addItem(f"In{int(ch) + 1}", int(ch))

        saved_channel = self.load_config.get("analysis_channel", None)
        if saved_channel is None or int(saved_channel) not in self.available_channels:
            saved_channel = int(self.available_channels[0])
        idx = self.channel_combo_box.findData(int(saved_channel))
        self.channel_combo_box.setCurrentIndex(idx if idx >= 0 else 0)

        channel_layout = QHBoxLayout()
        channel_layout.addWidget(channel_label)
        channel_layout.addWidget(self.channel_combo_box)
        channel_layout.setSpacing(10)
        return channel_layout

    def cheack_model_list(self):
        if self.analyse_model_combo_box.count() == 0:
            QMessageBox.warning(self, "设置警告", "没有可用的AI模型选型,请检查配置!")

    def create_model_layout(self):
        model_box = QGroupBox("模型")
        model_box.setMinimumSize(150, 150)
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
        analyse_model_combo_layout.addWidget(analyse_model_label)
        analyse_model_combo_layout.addWidget(self.analyse_model_combo_box)
        analyse_model_combo_layout.setSpacing(10)
        model_box.setLayout(analyse_model_combo_layout)
        return model_box

    def load_model_name_from_db(self):
        model_list = []
        query_code, query_result = TrainingModelManagement().get_all_model_info_from_db()
        if query_code == error_code.OK:
            for model_info in query_result:
                model_name = model_info[0]
                input_dim = model_info[1]
                config_path = model_info[5] if len(model_info) > 5 else ""
                if self._model_matches_signal_len(input_dim, config_path):
                    model_list.append(model_name)
        return model_list

    @staticmethod
    def _parse_input_length(input_dim):
        try:
            return int(float(str(input_dim or "").split("x")[0].strip()))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _resolve_model_config_path(config_path):
        normalized = str(config_path or "").strip()
        if not normalized:
            return ""
        if os.path.isabs(normalized):
            return normalized
        return os.path.join(DEFAULT_DIR, normalized)

    def _load_model_preprocess_config(self, config_path):
        normalized = self._resolve_model_config_path(config_path)
        if not normalized or not os.path.exists(normalized):
            return {}
        try:
            result = load_config(config_path=normalized, module_name="preprocess")
            return result if isinstance(result, dict) else {}
        except Exception:
            return {}

    def _model_matches_signal_len(self, input_dim, config_path):
        if not self.signal_len:
            return True

        input_length = self._parse_input_length(input_dim)
        if input_length is None:
            return False

        try:
            raw_length = int(float(self.signal_len))
        except (TypeError, ValueError):
            return input_length == self.signal_len

        preprocess_config = self._load_model_preprocess_config(config_path)
        effective_length = resolve_effective_signal_length(raw_length, preprocess_config)
        return input_length == effective_length

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
            "analysis_channel": int(self.channel_combo_box.currentData()),
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
