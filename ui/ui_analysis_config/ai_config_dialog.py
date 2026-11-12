"""LYL AI configuration interaction adapted to an injected model catalog."""

from copy import deepcopy

from PyQt5.QtWidgets import QVBoxLayout, QWidget

from base.ai_model_catalog import ModelCatalogError
from ui.custom_ui_widget.widgets import ComboBox, Label
from ui.ui_analysis_config.common_widgets import (
    ChannelSelectorWidget,
    SemanticAnalysisConfigDialogBase,
)


class AIConfigWindow(SemanticAnalysisConfigDialogBase):
    DEFAULT_CONFIG = {"analyse_model_name": "", "analysis_channel": 0}

    def __init__(self, config_manager, model_type, *, model_catalog, available_channels=None):
        super().__init__(disable_close_button=True)
        self.config_manager = config_manager
        self.config_key = model_type
        self.model_catalog = model_catalog
        self.available_channels = available_channels
        self.load_config = deepcopy(config_manager.config.get(model_type, {}))
        self.setWindowTitle("AI 分析配置")
        self.apply_semantic_dialog_size()
        self.set_semantic_button_callbacks(
            default_callback=self.on_default_btn_clicked,
            restore_callback=self.on_restore_default_btn_clicked,
            ok_callback=self.on_click_ok_btn,
        )
        self._build_semantic_sections()

    def _build_semantic_sections(self):
        self.channel_selector = ChannelSelectorWidget(
            self.load_config, self.available_channels, self,
        )
        self.add_semantic_section("input", widget=self.channel_selector)
        content = QWidget(self)
        layout = QVBoxLayout(content)
        layout.addWidget(Label("分析模型："))
        self.analyse_model_combo_box = ComboBox(content)
        self.analyse_model_combo_box.addItem("未选择模型", "")
        layout.addWidget(self.analyse_model_combo_box)
        self.model_status_label = Label(content)
        self.model_status_label.setObjectName("semanticSectionDescription")
        self.model_status_label.setWordWrap(True)
        layout.addWidget(self.model_status_label)
        self.add_semantic_section("compute", widget=content)
        self._load_model_choices()

    def _load_model_choices(self):
        try:
            choices = self.model_catalog.list_models()
        except ModelCatalogError as error:
            choices = ()
            self.model_status_label.setText(str(error))
        else:
            self.model_status_label.clear()
        self.model_status_label.setVisible(bool(self.model_status_label.text()))
        saved = self.load_config.get("analyse_model_name") or ""
        combo = self.analyse_model_combo_box
        for choice in choices:
            title = choice.name if choice.available else f"{choice.name}（{choice.reason}）"
            combo.addItem(title, choice.name)
            combo.model().item(combo.count() - 1).setEnabled(choice.available)
        index = combo.findData(saved)
        if index < 0:
            combo.addItem(f"{saved}（模型不可用，保留原配置）", saved)
            index = combo.count() - 1
            combo.model().item(index).setEnabled(False)
        combo.setCurrentIndex(index)

    def get_default_config(self):
        return {
            "analyse_model_name": self.analyse_model_combo_box.currentData(),
            "analysis_channel": self.channel_selector.current_channel(),
        }

    def on_default_btn_clicked(self):
        self.show_default_save_popup(
            self.config_manager.save_default_config("AI", self.get_default_config())
        )

    def on_restore_default_btn_clicked(self):
        self.load_config = deepcopy(self.config_manager.load_config().get(self.config_key, {}))
        self.clear_semantic_sections()
        self._build_semantic_sections()

    def on_click_ok_btn(self):
        config = self.get_default_config()
        self.accept()
        return config
