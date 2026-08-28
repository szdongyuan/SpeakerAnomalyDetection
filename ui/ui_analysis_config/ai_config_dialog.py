import os
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QHBoxLayout, QSizePolicy
from typing import List, Optional

from base.load_config import load_config
from base.model_runtime_validation import resolve_effective_signal_length
from base.training_model_management import TrainingModelManagement
from consts import error_code
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.popuputils import PopupUtils
from ui.custom_ui_widget.widgets import ComboBox, GroupBox, Label, MessageBox
from ui.ui_analysis_config.common_widgets import (
    ChannelSelectorWidget,
    MultiChannelSelectorWidget,
    SemanticAnalysisConfigDialogBase,
)


class AIConfigWindow(SemanticAnalysisConfigDialogBase):
    def __init__(
        self, config_manager, model_type, signal_len=None,
        available_channels: Optional[List[int]] = None,
        allow_multiple_channels: bool = False,
    ):
        super().__init__(disable_close_button=True)
        self.signal_len = signal_len
        self.config_manager = config_manager
        self.config_key = model_type
        self.model_list = self.load_model_name_from_db()
        self.load_config = self.config_manager.load_config().get(self.config_key, {})
        self.show_channel_selector = available_channels is not None
        self.allow_multiple_channels = allow_multiple_channels
        self.available_channels = available_channels
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("AI 分析配置")
        self.apply_semantic_dialog_size()
        self.set_semantic_button_callbacks(
            default_callback=self.on_default_btn_clicked,
            restore_callback=self.on_restore_default_btn_clicked,
            ok_callback=self.on_click_ok_btn,
        )
        self._build_semantic_sections()

    def _build_semantic_sections(self):
        if self.show_channel_selector:
            selector_type = (
                MultiChannelSelectorWidget
                if self.allow_multiple_channels else ChannelSelectorWidget
            )
            self.channel_selector = selector_type(
                self.load_config,
                self.available_channels,
                self,
            )
            self.add_semantic_section("input", widget=self.channel_selector)
        self.add_semantic_section("compute", widget=self.create_model_layout())

    def cheack_model_list(self):
        if self.analyse_model_combo_box.count() == 0:
            MessageBox.warning(self, "设置警告", "没有可用的AI模型选型,请检查配置!")

    def create_model_layout(self):
        model_box = GroupBox("模型")
        model_box.setMinimumSize(150, 150)
        analyse_model_label = Label("分析模型:")
        self.analyse_model_combo_box = ComboBox(self)
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
        return self.create_standard_button_layout(
            self.on_default_btn_clicked,
            self.on_click_ok_btn,
        )

    def get_default_config(self):
        default_config = {
            "analyse_model_name": self.analyse_model_combo_box.currentText(),
        }
        if self.show_channel_selector:
            default_config.update(self.channel_selector.get_config())
        else:
            default_config.update(
                ChannelSelectorWidget.normalized_config(self.load_config)
            )
        return default_config

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        save_flag = self.config_manager.save_default_config("AI", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_restore_default_btn_clicked(self):
        self.load_config = self.config_manager.load_config().get(
            self.config_key,
            {},
        )
        self.clear_semantic_sections()
        self._build_semantic_sections()

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        self.accept()
        return config_data
