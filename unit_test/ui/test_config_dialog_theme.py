import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.acquisition_config_window import BaseConfigWindow
from ui.archive_audio_data_dialog import ArchiveAudioDataDialog
from ui.config_dialog_base import ConfigDialogBase
from ui.config_dialog_theme import (
    CONFIG_DIALOG_BASE_STYLESHEET,
    build_config_dialog_stylesheet,
)
from ui.product_test_program_config_dialog import (
    ProductTestProgramConfigDialog,
)
from ui.serial_discrete_input_config_dialog import (
    SerialDiscreteInputConfigDialog,
)
from ui.tcp_config_dialog import TcpConfigDialog
from ui.custom_ui_widget.audio_data_manage_dialog import FilterAudioDialog
from ui.ui_analysis_config.ai_config_dialog import AIConfigWindow
from ui.ui_analysis_config.common_widgets import (
    ChannelSelectorWidget,
    SemanticAnalysisConfigDialogBase,
)
from ui.ui_analysis_config.spec_config_dialog import SpecConfigWindow
from ui.ui_analysis_config.spl_config_dialog import SplConfigWindow
from ui.ui_analysis_config.excel_config_dialog import ExcelConfigWindow
from ui.ui_analysis_config.fba_config_dialog import FbaConfigWindow
from ui.ui_analysis_config.fft_config_dialog import FftConfigWindow
from ui.ui_analysis_config.lp_config_dialog import LPConfigWindow
from ui.ui_analysis_config.reference_spectrum_config_dialog import (
    ReferenceSpectrumConfigWindow,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.closeAllWindows()
    app.quit()


def test_base_stylesheet_reuses_existing_main_window_tokens():
    assert ui_style_const.COLOR_PAGE_BG in CONFIG_DIALOG_BASE_STYLESHEET
    assert ui_style_const.COLOR_PRIMARY in CONFIG_DIALOG_BASE_STYLESHEET
    assert ui_style_const.COLOR_BORDER in CONFIG_DIALOG_BASE_STYLESHEET
    assert ui_style_const.COLOR_TEXT in CONFIG_DIALOG_BASE_STYLESHEET
    assert "# QTreeView" not in CONFIG_DIALOG_BASE_STYLESHEET


def test_feature_stylesheet_is_appended_after_shared_rules():
    feature_stylesheet = "QPushButton#primaryButton { font-weight: bold; }"

    stylesheet = build_config_dialog_stylesheet(feature_stylesheet)

    assert stylesheet.startswith(CONFIG_DIALOG_BASE_STYLESHEET)
    assert stylesheet.endswith(feature_stylesheet)


@pytest.mark.parametrize(
    "dialog_class",
    [
        BaseConfigWindow,
        ProductTestProgramConfigDialog,
        SerialDiscreteInputConfigDialog,
        TcpConfigDialog,
        AIConfigWindow,
        ArchiveAudioDataDialog,
        FilterAudioDialog,
        SplConfigWindow,
        SpecConfigWindow,
    ],
)
def test_configuration_entrypoints_use_shared_base(dialog_class):
    assert issubclass(dialog_class, ConfigDialogBase)


def test_dialog_theme_does_not_replace_parent_main_window_style(qapp):
    main_window = QMainWindow()
    main_stylesheet = "QPushButton { background-color: #112233; }"
    main_window.setStyleSheet(main_stylesheet)
    main_button = QPushButton("main", main_window)

    dialog = ConfigDialogBase(main_window)
    dialog_button = QPushButton("dialog", dialog)
    dialog_button.setObjectName("featureButton")
    dialog.apply_config_dialog_theme(
        "QPushButton#featureButton { background-color: #334455; }"
    )

    main_window.show()
    dialog.show()
    qapp.processEvents()

    assert main_window.styleSheet() == main_stylesheet
    assert CONFIG_DIALOG_BASE_STYLESHEET in dialog.styleSheet()
    assert main_button.palette().button().color().name() == "#112233"
    assert (
        "QPushButton#featureButton { background-color: #334455; }"
        in dialog.styleSheet()
    )

    dialog.close()
    main_window.close()
    qapp.processEvents()


def test_semantic_dialog_uses_shared_base_and_main_window_tokens(qapp):
    dialog = SemanticAnalysisConfigDialogBase()

    stylesheet = dialog.styleSheet()

    assert CONFIG_DIALOG_BASE_STYLESHEET in stylesheet
    assert ui_style_const.COLOR_PAGE_BG in stylesheet
    assert ui_style_const.COLOR_PRIMARY in stylesheet
    assert ui_style_const.COLOR_TEXT in stylesheet
    assert "#2563eb" not in stylesheet.lower()
    assert "font-size: 15px" not in stylesheet

    dialog.deleteLater()
    qapp.processEvents()


def test_semantic_cards_receive_final_palette_after_creation(qapp):
    dialog = SemanticAnalysisConfigDialogBase()
    dialog.add_semantic_section("input", widget=QWidget(dialog))
    dialog.show()
    qapp.processEvents()

    card = dialog._semantic_sections["input"]
    content = dialog._semantic_section_contents["input"]

    assert (
        dialog.section_container.palette()
        .color(dialog.section_container.backgroundRole())
        .name()
        == ui_style_const.COLOR_FIELD_DISABLED_BG.lower()
    )
    assert (
        card.palette().color(card.backgroundRole()).name()
        == ui_style_const.COLOR_CARD_BG.lower()
    )
    assert (
        content.palette().color(content.backgroundRole()).name()
        == ui_style_const.COLOR_CARD_BG.lower()
    )

    dialog.close()
    qapp.processEvents()


def test_semantic_dialog_preserves_existing_main_font_scale(qapp):
    dialog = SemanticAnalysisConfigDialogBase()
    content = QWidget(dialog)
    layout = QVBoxLayout(content)
    combo_box = QComboBox(content)
    check_box = QCheckBox("check", content)
    layout.addWidget(combo_box)
    layout.addWidget(check_box)
    dialog.add_semantic_section("input", widget=content)
    dialog.show()
    qapp.processEvents()

    assert combo_box.font().family() == "SimSun"
    assert combo_box.font().pixelSize() == 18
    assert check_box.font().family() == "SimSun"
    assert check_box.font().pixelSize() == 20
    assert dialog.semantic_default_btn.font().pixelSize() == 20

    dialog.close()
    qapp.processEvents()


def test_semantic_combobox_uses_complete_shared_frame(qapp):
    dialog = SemanticAnalysisConfigDialogBase()
    selector = ChannelSelectorWidget(available_channels=[0, 1])
    dialog.add_semantic_section("input", widget=selector)
    dialog.show()
    qapp.processEvents()

    stylesheet = dialog.styleSheet()
    combo_box = selector.combo_box
    image = combo_box.grab().toImage()
    bottom_center = image.pixelColor(
        image.width() // 2,
        image.height() - 1,
    ).name()

    assert "QComboBox::drop-down" in stylesheet
    assert "QComboBox::down-arrow" in stylesheet
    assert (
        DEFAULT_DIR
        + "ui/ui_analysis_config/assets/combobox_down_arrow.svg"
        in stylesheet
    )
    assert bottom_center == ui_style_const.COLOR_BORDER_STRONG.lower()

    dialog.close()
    qapp.processEvents()


@pytest.mark.parametrize(
    "dialog_class",
    [
        SplConfigWindow,
        SpecConfigWindow,
        FftConfigWindow,
        FbaConfigWindow,
        ReferenceSpectrumConfigWindow,
        AIConfigWindow,
        LPConfigWindow,
        ExcelConfigWindow,
    ],
)
def test_all_analysis_entrypoints_use_semantic_layout(dialog_class):
    assert issubclass(dialog_class, SemanticAnalysisConfigDialogBase)


class _ConfigManager:
    def __init__(self, config):
        self.config = config

    def load_config(self):
        return self.config

    def save_default_config(self, model_type, config):
        self.config[model_type] = config
        return True


def test_migrated_analysis_dialogs_keep_existing_config_contracts(
    qapp,
    monkeypatch,
):
    monkeypatch.setattr(
        AIConfigWindow,
        "load_model_name_from_db",
        lambda self: ["demo-model"],
    )
    manager = _ConfigManager(
        {
            "AI 1": {
                "analysis_channel": 1,
                "analyse_model_name": "demo-model",
            },
            "LP 1": {
                "analysis_channel": 1,
                "trigger_threshold": 10,
                "hysterests_threshold": 3,
                "min_check_duration": 10,
                "max_check_duration": 100,
                "loose_particle_num": 1,
                "cutoff_freq": 12000,
            },
            "RSC 1": {},
            "Excel 1": {
                "file_base": "analysis_results",
                "save_items": ["LP 1"],
            },
        }
    )
    dialogs = [
        AIConfigWindow(manager, "AI 1", available_channels=[0, 1]),
        LPConfigWindow(manager, "LP 1", available_channels=[0, 1]),
        ReferenceSpectrumConfigWindow(
            manager,
            "RSC 1",
            available_channels=[0, 1],
        ),
        ExcelConfigWindow(manager, "Excel 1"),
    ]

    assert [dialog.semantic_group_keys() for dialog in dialogs] == [
        ["input", "compute"],
        ["input", "detection"],
        ["reference", "compute", "judgment", "display"],
        ["output"],
    ]
    assert dialogs[0].get_default_config() == {
        "analyse_model_name": "demo-model",
        "analysis_channel": 1,
    }
    assert dialogs[1].get_default_config()["analysis_channel"] == 1
    assert dialogs[3].get_default_config()["file_base"] == "analysis_results"

    for dialog in dialogs:
        dialog.close()
    qapp.processEvents()
