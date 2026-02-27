import os
import re
import sys
import json
import copy
from datetime import datetime

from PyQt5.QtCore import Qt, QModelIndex, QSize
from PyQt5.QtGui import QIcon, QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import (
    QDialog,
    QLabel,
    QListView,
    QVBoxLayout,
    QCheckBox,
    QHBoxLayout,
    QPushButton,
    QTreeView,
)
from PyQt5.QtWidgets import QApplication, QMenu, QAction, QFileDialog, QMessageBox
from time import time

from base.data_struct.data_deal_struct import DataDealStruct
from base.data_struct.sequence_data import SequenceData
from base.load_config import ConfigManager, LoadUiConfig
from base.log_manager import LogManager
from base.soundcard_calibration_manager import get_mic_v2pa_factor
from base.stimulus_resolver import (
    set_data_struct_stimulus_signal as _safe_set_data_struct_stimulus_signal,
)
from base.soundcard_audio_processor import SoundcardAudioProcessor
from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.acquisition_config_window import (
    RecordConfigWindow,
    PlayRecordConfigWindow,
    ImportAudioConfigWindow,
    ImportStimulusAudioConfigWindow,
)

from ui.ui_analysis_config.ai_config_dialog import AIConfigWindow
from ui.ui_analysis_config.fr_config_dialog import FrConfigWindow
from ui.ui_analysis_config.hd_config_dialog import HdConfigWindow
from ui.ui_analysis_config.lp_config_dialog import LPConfigWindow
from ui.ui_analysis_config.pattern_match_config_dialog import PatternMatchConfigWindow
from ui.ui_analysis_config.pd_config_dialog import PDConfigWindow
from ui.ui_analysis_config.perceptual_rb_config_dialog import PerceptualRbConfigWindow
from ui.ui_analysis_config.pipeline_pd_pm_config import PipelinePdPmConfigWindow
from ui.ui_analysis_config.rb_config_dialog import RbConfigWindow
from ui.ui_analysis_config.spec_config_dialog import SpecConfigWindow
from ui.ui_analysis_config.spl_config_dialog import SplConfigWindow
from ui.signal_analysis_window import get_class_mapping
from ui.ui_analysis_config.excel_config_dialog import ExcelConfigWindow


class AnalysisModelSelect(QDialog):

    def __init__(self, using_config_path, mic=None, speaker=None):
        super().__init__()
        # When main window has no active config selected ("无配置"), using_config_path can be None.
        # Fall back to the built-in default sequence config so the test-queue window can still open.
        if not using_config_path:
            using_config_path = DEFAULT_DIR + "ui/ui_config/none_path.json"
        self.using_config_path = using_config_path
        # When user selects a target path via “新建”, confirm should save to that path
        # without touching main window's using_config_path registry.
        self._new_target_path_selected = False

        self.analysis_list = QTreeView()
        self.analysis_list.setSelectionMode(QTreeView.SingleSelection)
        self.default_logger = LogManager.set_log_handler("core")
        self.select_list = OptionList(
            self.default_logger, using_config_path, mic=mic, speaker=speaker
        )
        self.analysis_list.setEditTriggers(QTreeView.NoEditTriggers)
        self.select_list.setEditTriggers(QTreeView.NoEditTriggers)

        self.drag_drop_function()
        self.init_ui()

    def _get_using_config_display_name(self) -> str:
        """
        Prefer the registry key that maps to current using_config_path.
        Fallback to filename (without extension) when not found.
        """
        try:
            using_path = (self.using_config_path or "").replace("\\", "/")
            registry = LoadUiConfig._load_sequence_config_registry() or {}
            if using_path:
                for k, v in registry.items():
                    if k == "using_config_path":
                        continue
                    if isinstance(v, str) and v.replace("\\", "/") == using_path:
                        return str(k)
            if using_path:
                base = os.path.splitext(os.path.basename(using_path))[0]
                return base or using_path
        except Exception:
            pass
        return "无配置"

    def _update_current_config_label(self):
        if not hasattr(self, "current_config_label") or self.current_config_label is None:
            return
        name = self._get_using_config_display_name()
        self.current_config_label.setText(f"当前配置：{name}")

    def init_ui(self):
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setWindowTitle("测试队列")

        self.current_config_label = QLabel()
        self.current_config_label.setText("当前配置：")
        self.current_config_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        current_config_layout = QHBoxLayout()
        current_config_layout.addWidget(self.current_config_label)
        current_config_layout.addStretch()

        analysis_list_layout = self.create_analysis_list_layout()
        select_list_layout = self.create_select_list_layout()
        btn_layout = self.create_btn_layout()
        move_btn_layout = self.move_item_btn_layout()

        add_analysis_btn = QPushButton()
        add_analysis_btn.setDisabled(True)
        add_analysis_btn.setToolTip("添加分析")
        add_analysis_btn.setStyleSheet(ui_style_const.toolbar_button_style)
        add_analysis_btn.setIcon(
            QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/right_arrow.svg")
        )
        add_analysis_btn.setFixedSize(50, 50)
        add_analysis_btn.setIconSize(QSize(50, 50))
        add_analysis_btn.clicked.connect(self.add_analysis_btn_clicked)

        analysis_layout = QHBoxLayout()
        analysis_layout.addLayout(analysis_list_layout)
        analysis_layout.addWidget(add_analysis_btn)
        analysis_layout.addLayout(select_list_layout)
        analysis_layout.addLayout(move_btn_layout)

        layout = QVBoxLayout()
        layout.addLayout(current_config_layout)
        layout.addLayout(analysis_layout)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

        self.setStyleSheet(
            ui_style_const.qcombobox_style
            + ui_style_const.qpushbutton_style
            + ui_style_const.qlabel_style
            + ui_style_const.qcheckbox_style
            + ui_style_const.qlistview_style
            + ui_style_const.qtreeview_style
        )
        self._update_current_config_label()
        self.resize(740, 540)

    def add_analysis_btn_clicked(self):
        if self.analysis_list.currentIndex().row() != -1:
            index = self.analysis_list.currentIndex()
            if not index.parent().isValid():
                QMessageBox.information(self, "提示", "请选择要添加的子项")
                return
            text = index.data()
            self.select_list.set_new_analysis_config(text)
            self.select_list.data_struct.add_stft_or_fft_count(text)

    def drag_drop_function(self):
        self.analysis_list.setDragEnabled(True)
        self.analysis_list.setAcceptDrops(False)
        self.analysis_list.setDragDropMode(QTreeView.DragOnly)
        self.analysis_list.setDefaultDropAction(Qt.CopyAction)

        self.select_list.setDragEnabled(True)
        self.select_list.setDragDropMode(QListView.DragDrop)
        self.select_list.setDefaultDropAction(Qt.MoveAction)
        self.select_list.setDropIndicatorShown(True)
        self.select_list.setDragDropOverwriteMode(False)
        self.select_list.setMovement(QListView.Snap)
        self.select_list.setFlow(QListView.TopToBottom)

    def up_btn_clicked(self):
        self.select_list.itemmove("up")

    def down_btn_clicked(self):
        self.select_list.itemmove("down")

    def top_btn_clicked(self):
        self.select_list.itemmove("top")

    def bottom_btn_clicked(self):
        self.select_list.itemmove("bottom")

    def create_analysis_list_layout(self):
        analysis_label = QLabel("测试项目")

        self.analysis_model = AnalysisModel()
        sound_item = QStandardItem("音频设置")
        sound_items = ["播放与录制", "录制音频", "导入音频", "导入激励与音频"]
        for item in sound_items:
            list_item = QStandardItem(item.lstrip())
            list_item.setData(item, Qt.DisplayRole)
            sound_item.appendRow(list_item)
        self.analysis_model.appendRow(sound_item)

        analysis_item_item = QStandardItem("音频分析")
        analysis_items = [
            "声压级 (SPL) ",
            "声压级-频率 (SPLF) ",
            "频谱分析 (Spec) ",
            "频响 (FR) ",
            "谐波失真 (HD) ",
            "高阶谐波失真 (RB) ",
            "感知失真 (PRB) ",
            "松散颗粒 (LP) ",
            "峰值检测 (PD) ",
            "模式匹配(PM)",
            "AI 分析 ",
            "事件检测 (ED) ",
            "结果导出 (Excel) ",
        ]
        for item in analysis_items:
            list_item = QStandardItem(item.lstrip())
            list_item.setData(item, Qt.DisplayRole)
            analysis_item_item.appendRow(list_item)
        self.analysis_model.appendRow(analysis_item_item)
        self.analysis_list.setModel(self.analysis_model)
        self.analysis_list.header().hide()
        index = self.analysis_model.index(0, 0)
        self.analysis_list.setCurrentIndex(index)

        layout = QVBoxLayout()
        layout.addWidget(analysis_label)
        layout.addWidget(self.analysis_list)

        return layout

    def move_item_btn_layout(self):
        up_btn = QPushButton()
        down_btn = QPushButton()
        top_btn = QPushButton()
        bottom_btn = QPushButton()
        up_btn.setFixedSize(30, 30)
        down_btn.setFixedSize(30, 30)
        top_btn.setFixedSize(30, 30)
        bottom_btn.setFixedSize(30, 30)

        up_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/up.svg"))
        up_btn.setIconSize(QSize(30, 30))
        down_btn.setIcon(
            QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/down.svg")
        )
        down_btn.setIconSize(QSize(30, 30))
        top_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/top.svg"))
        top_btn.setIconSize(QSize(30, 30))
        bottom_btn.setIcon(
            QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/bottom.svg")
        )
        bottom_btn.setIconSize(QSize(30, 30))

        up_btn.clicked.connect(self.up_btn_clicked)
        down_btn.clicked.connect(self.down_btn_clicked)
        top_btn.clicked.connect(self.top_btn_clicked)
        bottom_btn.clicked.connect(self.bottom_btn_clicked)

        clear_btn = QPushButton()
        clear_btn.setToolTip("清空")
        clear_btn.setFixedSize(30, 30)
        clear_btn.setIcon(
            QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/clear_icon.png")
        )
        clear_btn.setIconSize(QSize(26, 26))
        clear_btn.clicked.connect(self.select_list.clear_option_list)

        layout = QVBoxLayout()
        layout.addWidget(top_btn)
        layout.addWidget(up_btn)
        layout.addWidget(down_btn)
        layout.addWidget(bottom_btn)
        layout.addStretch()
        layout.addWidget(clear_btn)
        layout.setContentsMargins(0, 30, 0, 0)

        return layout

    def create_select_list_layout(self):
        select_analysis_label = QLabel("测试序列")
        self.auto_analysis_box = QCheckBox("自动分析")
        if self.select_list.config:
            self.auto_analysis_box.setChecked(self.select_list.config[0].auto_analysis)
        self.auto_analysis_box.setLayoutDirection(Qt.RightToLeft)

        analysis_title_layout = QHBoxLayout()
        analysis_title_layout.addWidget(select_analysis_label)
        analysis_title_layout.addStretch()
        analysis_title_layout.addWidget(self.auto_analysis_box)

        layout = QVBoxLayout()
        layout.addLayout(analysis_title_layout)
        layout.addWidget(self.select_list)

        return layout

    def create_btn_layout(self):
        new_btn = QPushButton("新建")
        new_btn.clicked.connect(self.new_btn_clicked)
        new_btn.setMinimumWidth(100)

        record_golden_btn = QPushButton("录制黄金样本")
        record_golden_btn.clicked.connect(self.record_golden_sample_btn_clicked)
        record_golden_btn.setMinimumWidth(140)

        load_btn = QPushButton("导入")
        load_btn.clicked.connect(self.load_btn_clicked)
        save_btn = QPushButton("另存为")
        save_btn.clicked.connect(self.save_btn_clicked)
        ok_btn = QPushButton("保存")
        ok_btn.clicked.connect(self.ok_btn_clicked)
        ok_btn.setDefault(True)
        load_btn.setMinimumWidth(100)
        save_btn.setMinimumWidth(100)
        ok_btn.setMinimumWidth(100)

        layout = QHBoxLayout()
        layout.addWidget(record_golden_btn)
        layout.addStretch()
        layout.addWidget(new_btn)
        layout.addWidget(load_btn)
        layout.addWidget(save_btn)
        layout.addWidget(ok_btn)
        layout.setSpacing(20)

        return layout

    def record_golden_sample_btn_clicked(self):
        """
        Record a golden sample (baseline) for the currently selected sequence config:
        - Play configured stimulus and record once (blocking)
        - Run analysis ONLY for items with golden_sample_checked=True
        - Save analysis results into a JSON file
        - Store that JSON path into analysis_list['golden_sample_result_path'] (persisted on confirm/save)
        """
        if not self.select_list.config:
            QMessageBox.warning(self, "提示", "请先配置测试序列")
            return

        seq = self.select_list.config[0]

        detail = getattr(seq, "detail", None) or {}
        data_struct = self.select_list.data_struct

        # 录音前预检查：未勾选“使用黄金样本”的分析项则直接提示，避免白录一遍
        analysis_cfg = getattr(seq, "analysis_list", {}) or {}
        item_sort_list = analysis_cfg.get("display_sequence", [])
        if not item_sort_list:
            QMessageBox.warning(self, "提示", "当前序列没有可分析项目")
            return
        has_any_golden_checked = False
        for key in item_sort_list:
            key_config = analysis_cfg.get(key)
            if not isinstance(key_config, dict):
                continue
            if not key_config.get("golden_sample_checked", False):
                continue
            item_type = key_config.get("type")
            if item_type not in {"SPLF", "FR", "HD", "RB", "PRB"}:
                continue
            has_any_golden_checked = True
            break
        if not has_any_golden_checked:
            QMessageBox.warning(self, "提示", "没有勾选任何“使用黄金样本”的分析项")
            return

        try:
            # Ensure stimulus data is loaded/generated into DataDealStruct
            self.set_data_struct_stimulus_signal(
                data_struct,
                detail,
                using_config_path=self.using_config_path,
                logger=self.default_logger,
            )
        except Exception as e:
            QMessageBox.warning(self, "提示", f"加载激励失败: {str(e)[:200]}")
            return

        try:
            stimulus_dict, recorded_dict = (
                LoadUiConfig.get_rec_and_play_dict_base_sequence_dict(data_struct)
            )
        except Exception as e:
            QMessageBox.warning(self, "提示", f"生成播放/录制参数失败: {str(e)[:200]}")
            return

        # Prepare default save locations
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        golden_dir = os.path.join(DEFAULT_DIR, "audio_data", "stored_sample", "golden")
        try:
            os.makedirs(golden_dir, exist_ok=True)
        except Exception:
            pass

        recorded_wav_path = os.path.join(golden_dir, f"golden_record_{ts}.wav").replace(
            "\\", "/"
        )

        # Blocking play+record (also saves wav)
        try:
            sap = SoundcardAudioProcessor()
            record_code, aligned_data = sap.sd_play_rec(
                recorded_dict, stimulus_dict, recorded_wav_path
            )
            if record_code != 0 or aligned_data is None:
                QMessageBox.warning(self, "提示", "录制黄金样本失败")
                return
            data_struct.store_wave_data = aligned_data
        except Exception as e:
            QMessageBox.warning(self, "提示", f"录制黄金样本失败: {str(e)[:200]}")
            return

        # Collect only checked items
        items_out = {}
        class_mapping = get_class_mapping()
        for key in item_sort_list:
            key_config = analysis_cfg.get(key)
            if not isinstance(key_config, dict):
                continue
            if not key_config.get("golden_sample_checked", False):
                continue
            item_type = key_config.get("type")
            if item_type not in {"SPLF", "FR", "HD", "RB", "PRB"}:
                continue

            cls_map = class_mapping.get(item_type)
            if cls_map is None:
                continue

            try:
                params = copy.deepcopy(key_config)
                # When generating baseline, always disable golden/threshold influence
                params["golden_sample_checked"] = False
                params.pop("golden_sample_result_path", None)
                params["limit_checked"] = False
                params["limit_data"] = None

                instance = cls_map(key)
                instance.analysis_config = params
                instance.v2pa_factor = get_mic_v2pa_factor()

                result = None
                if hasattr(instance, "calculate_spl"):
                    result = instance.calculate_spl()
                elif hasattr(instance, "calculate_fr"):
                    result = instance.calculate_fr()
                elif hasattr(instance, "calculate_thd"):
                    result = instance.calculate_thd()
                    result.pop("harmonic", None)
                else:
                    continue

                items_out[key] = {"type": item_type, "result": result}
            except Exception as e:
                self.default_logger.error(
                    f"Golden sample analysis failed for {key}: {e}"
                )
                continue

        default_json_path = os.path.join(
            golden_dir, f"golden_baseline_{ts}.json"
        ).replace("\\", "/")
        json_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存黄金样本分析结果",
            default_json_path,
            filter="JSON Files (*.json)",
        )
        if not json_path:
            return

        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "stimulus_info": getattr(data_struct, "stimulus_info", None),
            "sample_rate": getattr(data_struct, "sample_rate", None),
            "recorded_wav_path": recorded_wav_path,
            "items": items_out,
        }

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.warning(self, "提示", f"保存黄金样本文件失败: {str(e)[:200]}")
            return

        # Store path into sequence analysis config; persisted when user clicks 保存/确定
        analysis_cfg["golden_sample_result_path"] = json_path.replace("\\", "/")

    def load_btn_clicked(self):
        default_dir = os.path.normpath(
            os.path.join(DEFAULT_DIR, "ui", "ui_config", "analysis_sequence_config")
        )
        try:
            os.makedirs(default_dir, exist_ok=True)
        except Exception:
            pass
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "导入配置文件",
            default_dir,
            filter="JSON Files (*.json)",
        )
        if file_path:
            try:
                file_path = file_path.replace("\\", "/")
                self.select_list.load_model_config(file_path)
                # Make "当前配置" reflect the real save target for 保存/确定.
                self.using_config_path = file_path
                self._new_target_path_selected = False
                self._update_current_config_label()
                LoadUiConfig.append_sequence_config_registry_entry(file_path)
            except Exception as e:
                self.default_logger.error(
                    f"Unable to parse JSON data in {file_path}. {e}"
                )

    def new_btn_clicked(self):
        default_dir = os.path.normpath(
            os.path.join(DEFAULT_DIR, "ui", "ui_config", "analysis_sequence_config")
        )
        try:
            os.makedirs(default_dir, exist_ok=True)
        except Exception:
            pass
        # Use a non-native dialog so we can control button text ("确认").
        dialog = QFileDialog(self, "新建配置文件")
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setNameFilter("JSON Files (*.json)")
        dialog.setDefaultSuffix("json")
        dialog.setDirectory(default_dir)
        dialog.setLabelText(QFileDialog.Accept, "确认")
        dialog.setLabelText(QFileDialog.Reject, "取消")

        if dialog.exec_() != QDialog.Accepted:
            return
        selected = dialog.selectedFiles()
        file_path = selected[0] if selected else ""
        if not file_path:
            return
        # If user didn't type extension, default to .json
        if not os.path.splitext(file_path)[1]:
            file_path = file_path + ".json"
        file_path = file_path.replace("\\", "/")

        # Do NOT update main window registry here; only affect this dialog's save target.
        self.using_config_path = file_path
        self._new_target_path_selected = True
        self._update_current_config_label()

        # Start from empty config
        self.select_list.clear_option_list()

    def format_config_data(self, config_data):
        for item in config_data:
            item.auto_analysis = self.auto_analysis_box.isChecked()
        save_config = [x.config_info for x in config_data]
        return save_config

    def save_btn_clicked(self):
        save_config = self.format_config_data(self.select_list.config)
        if not save_config:
            QMessageBox.warning(self, "警告", "没有配置测试内容")
            return

        default_dir = os.path.normpath(
            os.path.join(DEFAULT_DIR, "ui", "ui_config", "analysis_sequence_config")
        )
        try:
            os.makedirs(default_dir, exist_ok=True)
        except Exception:
            pass
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "另存为",
            default_dir,
            filter="JSON Files (*.json)",
        )
        if file_path:
            if not LoadUiConfig.save_sequence_config_to_json(save_config, file_path):
                QMessageBox.warning(self, "警告", "保存配置文件失败")
                self.close()
                return
            # Append the saved config path into registry json (filename as key)
            LoadUiConfig.append_sequence_config_registry_entry(file_path)

    def ok_btn_clicked(self):
        save_config = self.format_config_data(self.select_list.config)

        if not save_config:
            QMessageBox.warning(self, "警告", "没有配置测试内容")
            return

        if not self._new_target_path_selected:
            # If registry contains only using_config_path (no saved/imported entries),
            # add the built-in default config mapping on confirm.
            registry = LoadUiConfig._load_sequence_config_registry()
            other_keys = [
                k for k in (registry or {}).keys() if k != "using_config_path"
            ]

            if len(other_keys) == 0:
                if self.select_list.config:
                    LoadUiConfig.ensure_sequence_config_registry_field(
                        "默认配置",
                        DEFAULT_DIR + "ui/ui_config/sequence_config.json",
                    )
                    LoadUiConfig.update_using_config_path(
                        DEFAULT_DIR + "ui/ui_config/sequence_config.json"
                    )
                    self.using_config_path = (
                        DEFAULT_DIR + "ui/ui_config/sequence_config.json"
                    )
        if not LoadUiConfig.save_sequence_config_to_json(
            save_config, self.using_config_path
        ):
            QMessageBox.warning(self, "警告", "保存配置文件失败")
            self.close()
            return
        # If the target path was chosen via “新建”, register it for future selection,
        # but do NOT switch main window's current using_config_path.
        if self._new_target_path_selected:
            LoadUiConfig.append_sequence_config_registry_entry(self.using_config_path)

        # No forced mode switch / model sync here.
        # Main window will refresh the active config after this dialog closes.
        self.close()

    @staticmethod
    def set_data_struct_stimulus_signal(
        data_struct, detail, using_config_path: str = None, logger=None
    ):
        return _safe_set_data_struct_stimulus_signal(
            data_struct,
            detail,
            using_config_path=using_config_path,
            logger=logger,
        )

    # NOTE: removed update_test_file_current_model (no current model field in test log anymore)


class OptionList(QListView):

    def __init__(self, logger, using_config_path, mic=None, speaker=None):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.select_analysis_model = QStandardItemModel()
        self.setModel(self.select_analysis_model)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        self.select_analysis_model.dataChanged.connect(self.is_edit_model_item)

        self.default_logger = logger
        self.mic = mic
        self.speaker = speaker
        self.row_num = None
        self.darpflag = None
        self.sound_item_type = None
        self.start_row_number = None
        self.old_name = None
        self.press_time = None
        self.prev_select_ai = None
        self.is_edit_item = True
        self.index_num = None
        self.all_ai_item = []
        self.config = list()
        self.drop_is_accept = True
        self.signal_len = 0
        self.load_model_config(using_config_path)

        self.mousePressEvent = self.mousepressevent
        self.mouseReleaseEvent = self.mousereleaseevent
        self.dragEnterEvent = self.dragenterevent
        self.dragMoveEvent = self.dragmoveevent
        self.dropEvent = self.dropevent

    def itemmove(self, index):
        if self.index_num is None or not index:
            return
        item_index = self.model().index(self.index_num, 0)
        text = self.model().itemFromIndex(item_index).text()
        new_item = QStandardItem(text)
        if self.config[0].default_ai == text:
            new_item.setIcon(
                QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/star.png")
            )
        else:
            new_item.setIcon(
                QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/blank_icon.png")
            )
        if index == "top":
            if self.index_num == 0:
                return
            self.update_at_itemmove(
                1,
                new_item,
                self.index_num + 1,
                self.config[0].display_sequence,
                self.index_num,
                1,
            )
            self.index_num = 1
        elif index == "bottom" and self.index_num != 0:
            self.update_at_itemmove(
                self.model().rowCount(),
                new_item,
                self.index_num,
                self.config[0].display_sequence,
                self.index_num,
                self.model().rowCount() - 1,
            )
            self.index_num = self.model().rowCount() - 1
        elif index == "up" and self.index_num != 1 and self.index_num != 0:
            self.update_at_itemmove(
                self.index_num - 1,
                new_item,
                self.index_num + 1,
                self.config[0].display_sequence,
                self.index_num,
                self.index_num - 1,
            )
            self.index_num -= 1
        elif (
            index == "down"
            and self.index_num != self.model().rowCount() - 1
            and self.index_num != 0
        ):
            self.update_at_itemmove(
                self.index_num + 2,
                new_item,
                self.index_num,
                self.config[0].display_sequence,
                self.index_num,
                self.index_num + 1,
            )
            self.index_num += 1
        self.setCurrentIndex(self.model().index(self.index_num, 0))

    def update_at_itemmove(
        self, insert_index, new_item, pop_index, list: list, old_item_num, new_item_num
    ):
        self.model().insertRow(insert_index, new_item)
        self.model().removeRow(pop_index)
        self.swap_list_index(list, old_item_num, new_item_num)
        self.update_select_ai(old_item_num, new_item_num, False)

    def show_context_menu(self, pos):
        index = self.indexAt(pos)
        if index.isValid():
            menu = QMenu(self)
            menu.setStyleSheet(ui_style_const.main_window_menubar_style)
            open_action = QAction("打开", self)
            open_action.triggered.connect(lambda: self.show_dialog(index.data()))
            delete_action = QAction("删除", self)
            delete_action.triggered.connect(lambda: self.delete_item(index))
            rename_action = QAction("重命名", self)
            rename_action.triggered.connect(lambda: self.rename_item(index))

            self.old_name = index.data()
            self.disabled_rename_action(index, rename_action)

            menu.addAction(open_action)
            menu.addAction(delete_action)
            menu.addAction(rename_action)
            menu.exec_(self.mapToGlobal(pos))

    def disabled_rename_action(self, index, action):
        if index.row() == 0:
            action.setEnabled(False)
        else:
            action.setEnabled(True)

    # NOTE: removed “设为评判模型”逻辑（default_ai 不再作为测试模式依赖源）

    def store_ai_item(self, ai_list: list, name):
        if not name or name in ai_list:
            return
        ai_list.append(name)

    def check_item_isai(self, name):
        if not name:
            return None
        if name in self.all_ai_item:
            return True
        else:
            return False

    def show_dialog(self, name):
        model = QDialog(self)
        if name == self.config[0].name:
            if "播放与录制" in self.config[0].name:
                model = PlayRecordConfigWindow(
                    self.config[0].detail, mic=self.mic, speaker=self.speaker
                )
            elif "录制音频" in self.config[0].name:
                model = RecordConfigWindow(self.config[0].detail, mic=self.mic)
            elif "导入音频" in self.config[0].name:
                model = ImportAudioConfigWindow(self.config[0].detail, mic=self.mic)
            elif "导入激励与音频" in self.config[0].name:
                model = ImportStimulusAudioConfigWindow(
                    self.config[0].detail, mic=self.mic, speaker=self.speaker
                )
            result = model.exec()
            if result is not None:
                self.config[0].detail = result
                if "播放与录制" in name or "导入激励与音频" in name:
                    self.signal_len = int(
                        result["stimulus_info"]["total_time"]
                        * result["stimulus_info"]["sample_rate"]
                    )
                elif "录制音频" in name:
                    self.signal_len = int(result["total_time"] * result["sample_rate"])

        elif name in self.config[0].display_sequence:
            prev_config_file = DEFAULT_DIR + "ui/ui_config/sequence_config.json"
            model_type = None
            config_manager = None
            if name in self.config[0].analysis_list:
                config_manager = ConfigManager(prev_config_file)
            type = self.config[0].analysis_list.get(name)["type"]
            if self.config[0].analysis_list.get(name):
                config_manager.config = self.config[0].analysis_list
                model_type = name
            model = self.create_config_dialog(
                model, config_manager, model_type, type, self.signal_len
            )
            model.setWindowTitle(name)
            if model.exec_() == QDialog.Accepted:
                config_data = model.on_click_ok_btn()
                self.add_config(name, config_data)

    def load_stimulus_config(self):
        default_config_file = DEFAULT_DIR + "ui/ui_config/default_stimulus.json"
        code, data = LoadUiConfig.load_data_from_json(default_config_file)
        if code == 0:
            return True, data
        else:
            self.default_logger.error(f"load default stimulus config error {data}")
            return False, {}

    def create_config_dialog(
        self, model: QDialog, config_manager: ConfigManager, name, type, signal_len
    ):
        if type == "SPL":
            model = SplConfigWindow(config_manager, name)
        elif type == "SPLF":
            model = SplConfigWindow(config_manager, name)
        elif type == "FR":
            model = FrConfigWindow(config_manager, name)
        elif type == "HD":
            model = HdConfigWindow(config_manager, name)
        elif type == "RB":
            model = RbConfigWindow(config_manager, name)
        elif type == "PRB":
            model = PerceptualRbConfigWindow(config_manager, name)
        elif type == "AI":
            model = AIConfigWindow(config_manager, name, signal_len)
        elif type == "Spec":
            model = SpecConfigWindow(config_manager, name)
        elif type == "LP":
            model = LPConfigWindow(config_manager, name)
        elif type == "PD":
            model = PDConfigWindow(config_manager, name)
        elif type == "ED":
            model = PipelinePdPmConfigWindow(config_manager, name)
        elif type == "PM":
            model = PatternMatchConfigWindow(config_manager, name)
        elif type == "Excel":
            model = ExcelConfigWindow(config_manager, name)
        return model

    def init_config_info(self, config_file):
        code, config_info = LoadUiConfig.load_data_from_json(config_file)
        if code != 0:
            self.default_logger.error(
                f"Failed to load the default config file. {config_info}"
            )
            return
        if config_info:
            for i in config_info:
                key, value = next(iter(i.items()))
                sequence_config = SequenceData(key)
                sequence_config.name = value.get("acq", {}).get("name", None)
                sequence_config.mode = value.get("acq", {}).get("mode", None)
                self.sound_item_type = sequence_config.name.lstrip()
                sequence_config.detail = value.get("acq", {}).get("detail", {})

                i_analysis_list = value.get("analysis_list", {})
                # default_ai is deprecated for business logic (no longer used as test-mode dependency)
                i_analysis_list.pop("default_ai", None)
                sequence_config.default_ai = None
                sequence_config.display_sequence = i_analysis_list.pop(
                    "display_sequence", []
                )
                sequence_config.auto_analysis = i_analysis_list.pop(
                    "auto_analysis", False
                )

                sequence_config.analysis_list.update(i_analysis_list)
                self.config.append(sequence_config)
                if sequence_config.mode != "IMPORT_AUDIO":
                    self.signal_len = sequence_config.detail.get(
                        "total_time", 4.0
                    ) * sequence_config.detail.get("sample_rate", 44100)
                else:
                    self.signal_len = 0

    def clear_option_list(self):
        self.config = list()
        self.data_struct.clear_fft_and_stft_flag()
        self.model().clear()
        self.prev_select_ai = None
        self.all_ai_item = []
        self.sound_item_type = None
        self.drop_is_accept = True

    def load_model_config(self, config_path):
        if not config_path or not isinstance(config_path, (str, bytes, os.PathLike)):
            # Keep the dialog usable even when no config is currently selected in main window.
            self.default_logger.warning(
                f"Invalid config_path for OptionList.load_model_config: {config_path!r}"
            )
            self.clear_option_list()
            return
        if os.path.exists(config_path):
            self.clear_option_list()
            self.init_config_info(config_path)
        else:
            self.default_logger.warning(f"Config file does not exist: {config_path!r}")
            self.clear_option_list()
            return
        for config in self.config:
            if config.mode:
                mode_item = QStandardItem(config.name)
                mode_item.setIcon(
                    QIcon(
                        DEFAULT_DIR + "ui/ui_pic/select_analysis_model/blank_icon.png"
                    )
                )
                self.model().appendRow(mode_item)
            else:
                continue
            for key, value in self.config[0].analysis_list.items():
                if (
                    key != "auto_analysis"
                    and key != "default_ai"
                    and key != "display_sequence"
                    and key != "golden_sample_result_path"
                ):
                    if "AI" == value.get("type"):
                        self.store_ai_item(self.all_ai_item, key)
            model_item_list = self.config[0].display_sequence
            for item_name in model_item_list:
                self.data_struct.add_stft_or_fft_count(
                    self.config[0].analysis_list[item_name]["type"]
                )
                if item_name == self.config[0].default_ai:
                    list_item = QStandardItem(item_name)
                    list_item.setIcon(
                        QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/star.png")
                    )
                    self.model().appendRow(list_item)
                    last_row = self.model().rowCount() - 1
                    self.prev_select_ai = self.model().index(last_row, 0)
                else:
                    list_item = QStandardItem(item_name)
                    list_item.setIcon(
                        QIcon(
                            DEFAULT_DIR
                            + "ui/ui_pic/select_analysis_model/blank_icon.png"
                        )
                    )
                    self.model().appendRow(list_item)

    def add_config(self, class_name, config_data):
        if class_name in self.config[0].analysis_list:
            self.config[0].analysis_list[class_name].update(config_data)
        else:
            self.config[0].analysis_list[class_name] = config_data

    def delete_item(self, index):
        if index.data().lstrip() == self.sound_item_type:
            self.clear_option_list()
            return
        if self.config[0].default_ai == index.data():
            self.config[0].display_sequence.remove(self.config[0].default_ai)
            self.delete_item_config(self.config[0].default_ai)
            self.config[0].default_ai = None
            self.prev_select_ai = None
        else:
            self.config[0].display_sequence.remove(index.data())
            self.data_struct.minus_stft_or_fft_count(
                self.config[0].analysis_list[index.data()]["type"]
            )
            self.delete_item_config(index.data())
            self.update_default_ai_index_at_delete_item(index)
        model = self.model()
        model.removeRow(index.row())

    def update_default_ai_index_at_delete_item(self, index):
        if self.prev_select_ai is None:
            return
        if index.row() < self.prev_select_ai.row():
            self.prev_select_ai = self.model().index(self.prev_select_ai.row() - 1, 0)

    def delete_item_config(self, name):
        if not name:
            return
        # Remove deleted items from any Excel export selection list to avoid stale references
        try:
            for _, cfg in self.config[0].analysis_list.items():
                if not isinstance(cfg, dict):
                    continue
                if cfg.get("type") != "Excel":
                    continue
                save_items = cfg.get("save_items")
                if isinstance(save_items, list) and name in save_items:
                    cfg["save_items"] = [x for x in save_items if x != name]
        except Exception:
            pass
        if name in self.config[0].analysis_list:
            del self.config[0].analysis_list[name]

    def rename_item(self, index):
        self.is_update_config = True
        self.is_select_ai = self.config[0].default_ai == self.model().data(index)
        self.edit(index)

    def update_model_list(
        self,
        config: dict,
        new_item: QStandardItem,
        old_index,
        new_index,
        step_index: bool,
    ):
        if not new_item or old_index == 0:
            self.setCurrentIndex(self.model().index(0, 0))
            return
        else:
            if new_index == 0:
                self.model().insertRow(1, new_item)
                new_index = 1
                self.start_row_number = 1
            else:
                self.model().insertRow(new_index, new_item)
            if step_index:
                self.model().removeRow(old_index)
                self.swap_list_index(
                    config["display_sequence"], old_index, new_index - 1
                )
                self.start_row_number = new_index - 1
            else:
                self.model().removeRow(old_index + 1)
                self.swap_list_index(config["display_sequence"], old_index, new_index)
                if new_index != 0:
                    self.start_row_number = new_index
        self.setCurrentIndex(self.model().index(self.start_row_number, 0))
        self.update_select_ai(old_index, new_index, True)

    def update_select_ai(self, old_index, new_index, step_index: bool):
        if (
            old_index == new_index
            or old_index == -1
            or new_index == -1
            or not self.prev_select_ai
        ):
            return

        select_ai_row = self.prev_select_ai.row()
        if select_ai_row < old_index and select_ai_row >= new_index:
            select_ai_row = select_ai_row + 1
            self.prev_select_ai = self.model().index(select_ai_row, 0)
        elif select_ai_row > old_index and select_ai_row <= new_index:
            select_ai_row = select_ai_row - 1
            self.prev_select_ai = self.model().index(select_ai_row, 0)
        elif select_ai_row == old_index:
            if step_index:
                if new_index > select_ai_row:
                    self.prev_select_ai = self.model().index(new_index - 1, 0)
                elif new_index < select_ai_row:
                    self.prev_select_ai = self.model().index(new_index, 0)
            else:
                self.prev_select_ai = self.model().index(new_index, 0)

    def set_model_data(self, index: QModelIndex, name):
        self.is_edit_item = False
        self.model().setData(index, name)

    def update_config_data(self, old_name, new_name, list):
        if not new_name in list:
            if old_name in self.config[0].analysis_list:
                value = self.config[0].analysis_list.pop(old_name)
                self.config[0].analysis_list[new_name] = value
            index = list.index(old_name)
            list[index] = new_name
        if old_name in self.all_ai_item:
            if not new_name in self.all_ai_item:
                ai_index = self.all_ai_item.index(old_name)
                self.all_ai_item[ai_index] = new_name
        # Keep Excel export item's selection in sync when other items are renamed
        try:
            for _, cfg in self.config[0].analysis_list.items():
                if not isinstance(cfg, dict):
                    continue
                if cfg.get("type") != "Excel":
                    continue
                save_items = cfg.get("save_items")
                if isinstance(save_items, list) and old_name in save_items:
                    cfg["save_items"] = [
                        new_name if x == old_name else x for x in save_items
                    ]
        except Exception:
            # Never block rename flow
            pass

    def is_edit_model_item(self, topLeft, bottomRight, roles):
        if Qt.EditRole in roles:
            for row in range(topLeft.row(), bottomRight.row() + 1):
                index = self.model().index(row, topLeft.column())
            self.on_data_changed(index, self.is_edit_item)

    def on_data_changed(self, index: QModelIndex, is_edit_item):
        if is_edit_item is False:
            self.is_edit_item = True
            return
        new_name = self.model().data(index)
        really_new_name = new_name.replace(" ", "")
        if new_name != self.old_name and really_new_name:
            if new_name in self.config[0].display_sequence:
                QMessageBox.warning(self, "警告", "项目名称重复，请重新输入！")
                self.set_model_data(index, self.old_name)
                return
            if self.is_select_ai:
                if new_name != self.old_name:
                    self.update_config_data(
                        self.old_name, new_name, self.config[0].display_sequence
                    )
                self.old_name = new_name
                self.config[0].default_ai = new_name
                self.is_select_ai = False
            else:
                if self.is_update_config:
                    self.update_config_data(
                        self.old_name, new_name, self.config[0].display_sequence
                    )
                    self.is_update_config = False
                self.old_name = new_name
        else:
            if new_name == self.old_name:
                return
            self.set_model_data(index, self.old_name)

    def swap_list_index(self, list: list, old_index, new_index):
        if old_index == new_index or not list:
            return
        old_index -= 1
        new_index -= 1
        old_name = list[old_index]
        list.pop(old_index)
        list.insert(new_index, old_name)

    def mousepressevent(self, e):
        index = self.indexAt(e.pos())
        if Qt.LeftButton == e.button():
            if index.isValid():
                self.darpflag = True
                self.start_index = index
                self.start_row_number = index.row()
        if Qt.RightButton == e.button():
            self.setCurrentIndex(self.indexAt(e.pos()))
            self.index_num = index.row()
        e.accept()

    def mousereleaseevent(self, e):
        if Qt.LeftButton != e.button():
            return
        t1 = time()
        if self.press_time != None:
            time_area = t1 - self.press_time
            if time_area > 0.3:
                self.row_num = None
        self.press_time = t1
        index = self.indexAt(e.pos())
        self.setCurrentIndex(index)
        row_number = index.row()
        if row_number == -1:
            self.index_num = None
        if self.darpflag:
            text = self.start_index.data()
            new_item = QStandardItem(text)
            if text == self.config[0].default_ai:
                new_item.setIcon(
                    QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/star.png")
                )
            else:
                new_item.setIcon(
                    QIcon(
                        DEFAULT_DIR + "ui/ui_pic/select_analysis_model/blank_icon.png"
                    )
                )
            if row_number == -1:
                self.update_model_list(
                    self.config[0].analysis_list,
                    new_item,
                    self.start_row_number,
                    self.model().rowCount(),
                    True,
                )
            else:
                if row_number > self.start_row_number:
                    self.update_model_list(
                        self.config[0].analysis_list,
                        new_item,
                        self.start_row_number,
                        row_number,
                        True,
                    )
                elif row_number < self.start_row_number:
                    self.update_model_list(
                        self.config[0].analysis_list,
                        new_item,
                        self.start_row_number,
                        row_number,
                        False,
                    )
            # Update the starting item name and index number, and end the drag-and-drop state
            self.start_item_name = new_item.text()
            self.index_num = self.start_row_number
            self.darpflag = False
        if self.row_num == row_number & row_number != -1:
            name_str = self.model().itemFromIndex(index).text()
            self.show_dialog(name_str)
            self.row_num = None
        else:
            self.row_num = row_number
        e.accept()

    def dragenterevent(self, event):
        if event.mimeData().hasText():
            text = event.mimeData().text()
            if text in ["播放与录制", "录制音频", "导入音频", "导入激励与音频"]:
                if self.sound_item_type:
                    self.drop_is_accept = False
            elif self.sound_item_type in ["录制音频", "导入音频"]:
                if text in [
                    "声压级-频率 (SPLF) ",
                    "频响 (FR) ",
                    "谐波失真 (HD) ",
                    "高阶谐波失真 (RB) ",
                    "感知失真 (PRB) ",
                ]:
                    self.drop_is_accept = False
            elif not self.sound_item_type:
                self.drop_is_accept = False
            event.accept()
        else:
            event.ignore()

    def dragmoveevent(self, event):
        if event.mimeData().hasText():
            event.accept()
        else:
            event.ignore()

    def dropevent(self, event):
        if event.mimeData().hasText():
            text = event.mimeData().text().lstrip()
            if self.drop_is_accept is False:
                if text in ["播放与录制", "录制音频", "导入音频", "导入激励与音频"]:
                    QMessageBox.warning(self, "警告", "已选择测试模式")
                elif not self.config:
                    QMessageBox.warning(self, "警告", "请选择测试模式")
                elif text in [
                    "声压级-频率 (SPLF) ",
                    "频响 (FR) ",
                    "谐波失真 (HD) ",
                    "高阶谐波失真 (RB) ",
                    "感知失真 (PRB) ",
                ]:
                    QMessageBox.warning(self, "警告", "当前模式不支持此功能")
                self.drop_is_accept = True
                return
            elif text in ["播放与录制", "录制音频", "导入音频", "导入激励与音频"]:
                self.set_sound_item(text)
                self.sound_item_type = text
            else:
                self.set_new_analysis_config(text)
                self.data_struct.add_stft_or_fft_count(text)
            event.accept()
        else:
            event.ignore()

    def set_sound_item(self, item_text):
        list_item = QStandardItem(item_text)
        list_item.setIcon(
            QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/blank_icon.png")
        )
        if self.config:
            seq_name_list = list()
            for key, value in self.config.items():
                seq_name_list.append(key)
            count = len(seq_name_list)

            while True:
                seq_name = "seq" + str(count)
                if seq_name not in seq_name_list:
                    break
                count += 1
        else:
            seq_name = "seq1"
        seq_item = SequenceData(seq_name)
        seq_item.name = item_text
        if item_text == "播放与录制":
            flag, config = self.load_stimulus_config()
            if flag:
                seq_item.mode = "PLAY_AND_RECORD"
                seq_item.detail = config
                self.signal_len = seq_item.detail.get(
                    "total_time", 4.0
                ) * seq_item.detail.get("sample_rate", 44100)
            else:
                QMessageBox.warning(self, "提示", "窗口配置错误，请检查配置!")
                return
        elif item_text == "录制音频":
            seq_item.mode = "RECORD_ONLY"
            seq_item.detail = {"total_time": 4.0, "sample_rate": 44100}
            self.signal_len = seq_item.detail.get(
                "total_time", 4.0
            ) * seq_item.detail.get("sample_rate", 44100)
        elif item_text == "导入音频":
            seq_item.mode = "IMPORT_AUDIO"
            seq_item.detail = {"sample_rate": 44100}
            self.signal_len = 0
        elif item_text == "导入激励与音频":
            flag, config = self.load_stimulus_config()
            if flag:
                seq_item.mode = "IMPORT_STIMULUS_AUDIO"
                seq_item.detail = config
                self.signal_len = seq_item.detail.get(
                    "total_time", 4.0
                ) * seq_item.detail.get("sample_rate", 44100)
            else:
                QMessageBox.warning(self, "提示", "窗口配置错误，请检查配置!")
                return
        self.config.append(seq_item)

        self.model().insertRow(0, list_item)

    def set_new_analysis_config(self, item_text):
        count = 1
        item_exist = self.model().findItems(item_text + f"{count}")
        while item_exist:
            count += 1
            item_exist = self.model().findItems(item_text + f"{count}")
        list_item = QStandardItem(item_text + f"{count}")
        list_item.setIcon(
            QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/blank_icon.png")
        )
        self.model().insertRow(self.model().rowCount(), list_item)
        list_item_text = list_item.text()
        if "AI" in item_text:
            self.store_ai_item(self.all_ai_item, list_item_text)
        self.config[0].display_sequence.append(list_item_text)
        self.get_item_default_config(item_text, list_item_text)

    def get_item_default_config(self, item_text, list_item_text):
        if not item_text or not list_item_text:
            return
        type = "".join(re.findall(r"[A-Za-z]", item_text))
        default_config_file = DEFAULT_DIR + "ui/ui_config/analysis_default_config.json"
        code, data = LoadUiConfig.load_data_from_json(default_config_file)
        if code != 0:
            self.default_logger.error(f"Failed to load the default config file. {data}")
            return

        default_of_type = data.get(type, {})
        self.config[0].analysis_list[list_item_text] = default_of_type
        self.config[0].analysis_list[list_item_text]["type"] = type


class AnalysisModel(QStandardItemModel):

    def mimeTypes(self):
        return ["text/plain"]

    def mimeData(self, indexes):
        mime_data = super().mimeData(indexes)
        texts = [index.data(Qt.DisplayRole) for index in indexes if index.isValid()]
        mime_data.setText("\n".join(texts))
        return mime_data

    def flags(self, index):
        default_flags = super().flags(index)

        if not index.isValid():
            return default_flags

        if not index.parent().isValid():
            return default_flags & ~Qt.ItemIsDragEnabled

        return default_flags | Qt.ItemIsDragEnabled


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnalysisModelSelect()
    window.show()
    sys.exit(app.exec_())
