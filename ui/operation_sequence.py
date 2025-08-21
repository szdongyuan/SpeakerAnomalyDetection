import os
import re
import sys
from datetime import datetime

from PyQt5.QtCore import Qt, QModelIndex, QSize
from PyQt5.QtGui import QIcon, QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QDialog, QLabel, QListView, QVBoxLayout, QCheckBox, QHBoxLayout, QPushButton, QTreeView
from PyQt5.QtWidgets import QApplication, QMenu, QAction, QFileDialog, QMessageBox
from time import time

from base.data_struct.data_deal_struct import DataDealStruct
from base.data_struct.sequence_data import SequenceData
from base.load_audio import load_audio_simple
from base.load_config import ConfigManager, LoadUiConfig
from base.log_manager import LogManager
from base.utils.custom_signals import sign
from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.acquisition_config_window import RecordConfigWindow, PlayRecordConfigWindow
from ui.analysis_config_window import SplConfigWindow, FrConfigWindow, HdConfigWindow, AIConfigWindow, SpecConfigWindow, \
    PatternMatchConfigWindow
from ui.analysis_config_window import LPConfigWindow
from ui.analysis_config_window import PDConfigWindow


class AnalysisModelSelect(QDialog):

    def __init__(self):
        super().__init__()

        self.analysis_list = QTreeView()
        self.analysis_list.setSelectionMode(QTreeView.SingleSelection)
        self.default_logger = LogManager.set_log_handler("core")
        self.select_list = OptionList(self.default_logger)
        self.analysis_list.setEditTriggers(QTreeView.NoEditTriggers)
        self.select_list.setEditTriggers(QTreeView.NoEditTriggers)

        self.drag_drop_function()
        self.init_ui()

    def init_ui(self):
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setWindowTitle("测试队列")

        analysis_list_layout = self.create_analysis_list_layout()
        select_list_layout = self.create_select_list_layout()
        btn_layout = self.create_btn_layout()
        move_btn_layout = self.move_item_btn_layout()

        add_analysis_btn = QPushButton()
        add_analysis_btn.setDisabled(True)
        add_analysis_btn.setToolTip("添加分析")
        add_analysis_btn.setStyleSheet(ui_style_const.toolbar_button_style)
        add_analysis_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/right_arrow.svg"))
        add_analysis_btn.setFixedSize(50, 50)
        add_analysis_btn.setIconSize(QSize(50, 50))
        add_analysis_btn.clicked.connect(self.add_analysis_btn_clicked)

        analysis_layout = QHBoxLayout()
        analysis_layout.addLayout(analysis_list_layout)
        analysis_layout.addWidget(add_analysis_btn)
        analysis_layout.addLayout(select_list_layout)
        analysis_layout.addLayout(move_btn_layout)

        layout = QVBoxLayout()
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
        sound_items = ["播放与录制", "录制音频"]
        for item in sound_items:
            list_item = QStandardItem(item.lstrip())
            list_item.setData(item, Qt.DisplayRole)
            sound_item.appendRow(list_item)
        self.analysis_model.appendRow(sound_item)

        analysis_item_item = QStandardItem("音频分析")
        analysis_items = [
            "声压级 (SPL) ",
            "频响 (FR) ",
            "谐波失真 (HD) ",
            "松散颗粒 (LP) ",
            "峰值检测 (PD) ",
            "模式匹配(PM)",
            "AI 分析 ",
            "频谱分析 (Spec) ",
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
        down_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/down.svg"))
        down_btn.setIconSize(QSize(30, 30))
        top_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/top.svg"))
        top_btn.setIconSize(QSize(30, 30))
        bottom_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/bottom.svg"))
        bottom_btn.setIconSize(QSize(30, 30))

        up_btn.clicked.connect(self.up_btn_clicked)
        down_btn.clicked.connect(self.down_btn_clicked)
        top_btn.clicked.connect(self.top_btn_clicked)
        bottom_btn.clicked.connect(self.bottom_btn_clicked)

        layout = QVBoxLayout()
        layout.addWidget(top_btn)
        layout.addWidget(up_btn)
        layout.addWidget(down_btn)
        layout.addWidget(bottom_btn)
        layout.addStretch()
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
        analysis_title_layout.addWidget(self.auto_analysis_box)

        layout = QVBoxLayout()
        layout.addLayout(analysis_title_layout)
        layout.addWidget(self.select_list)

        return layout

    def create_btn_layout(self):
        load_btn = QPushButton("导入")
        load_btn.clicked.connect(self.load_btn_clicked)
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self.save_btn_clicked)
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(self.ok_btn_clicked)
        ok_btn.setDefault(True)
        clear_btn = QPushButton("清空")
        clear_btn.clicked.connect(self.select_list.clear_option_list)
        load_btn.setMinimumWidth(100)
        save_btn.setMinimumWidth(100)
        ok_btn.setMinimumWidth(100)
        clear_btn.setMinimumWidth(100)

        layout = QHBoxLayout()
        layout.addStretch()
        layout.addWidget(clear_btn)
        layout.addWidget(load_btn)
        layout.addWidget(save_btn)
        layout.addWidget(ok_btn)
        layout.setSpacing(20)

        return layout

    def load_btn_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "导入配置文件",
            DEFAULT_DIR + "ui/ui_config/analysis_sequence_config",
            filter="JSON Files (*.json);;All Files (*)",
        )
        if file_path:
            try:
                self.select_list.load_model_config(file_path)
            except Exception as e:
                self.default_logger.error(f"Unable to parse JSON data in {file_path}. {e}")

    def format_config_data(self, config_data):
        for item in config_data:
            item.auto_analysis = self.auto_analysis_box.isChecked()
        save_config = [x.config_info for x in config_data]
        return save_config

    def save_btn_clicked(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存配置文件",
            DEFAULT_DIR + "ui/ui_config/analysis_sequence_config",
            filter="JSON Files (*.json);;All Files (*)",
        )
        if file_path:
            save_config = self.format_config_data(self.select_list.config)
            if not LoadUiConfig.save_sequence_config_to_json(save_config, file_path):
                QMessageBox.warning(self, "警告", "保存配置文件失败")
                self.close()
                return

    def ok_btn_clicked(self):
        save_config = self.format_config_data(self.select_list.config)
        json_file_path = DEFAULT_DIR + "ui/ui_config/sequence_config.json"
        if not LoadUiConfig.save_sequence_config_to_json(save_config, json_file_path):
            QMessageBox.warning(self, "警告", "保存配置文件失败")
            self.close()
            return
        if self.select_list.config:
            data_struct = DataDealStruct()
            detail = self.select_list.config[0].detail
            if self.select_list.config[0].mode == "PLAY_AND_RECORD":
                self.set_data_struct_stimulus_signal(data_struct, detail)
            else:
                data_struct.sample_rate = detail["sample_rate"]
        self.update_test_file_current_model()
        sign.update_mode_display_sign.emit(0)
        self.close()

    @staticmethod
    def set_data_struct_stimulus_signal(data_struct, detail):
        stimulus_info = detail.get("stimulus_info")
        path = os.path.join(DEFAULT_DIR, detail.get("stimulus_signal_path", ""))
        stimulus_signal, _ = load_audio_simple(path, stimulus_info["sample_rate"])
        data_struct.stimulus_info = stimulus_info
        data_struct.stimulus_data = stimulus_signal
        data_struct.sample_rate = stimulus_info["sample_rate"]

    def update_test_file_current_model(self):
        if self.select_list.config:
            default_ai_model = self.select_list.config[0].default_ai
        else:
            default_ai_model = None
        if default_ai_model:
            analyse_model_name = (
                self.select_list.config[0].analysis_list.get(default_ai_model, {}).get("analyse_model_name", None)
            )
            current_time = datetime.now().strftime("%Y-%m-%d")
            test_result_path = DEFAULT_DIR + f"log/test_result_log/{current_time}.dat"
            with open(test_result_path, "r") as r:
                lines = r.readlines()
                lines[4] = f"current_model: {analyse_model_name}\n"
            with open(test_result_path, "w") as w:
                w.writelines(lines)


class OptionList(QListView):

    def __init__(self, logger):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.select_analysis_model = QStandardItemModel()
        self.setModel(self.select_analysis_model)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        self.select_analysis_model.dataChanged.connect(self.is_edit_model_item)

        self.default_logger = logger
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
        self.load_model_config(DEFAULT_DIR + "ui/ui_config/sequence_config.json")

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
            new_item.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/star.png"))
        else:
            new_item.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/blank_icon.png"))
        if index == "top":
            if self.index_num == 0:
                return
            self.update_at_itemmove(1, new_item, self.index_num + 1, self.config[0].display_sequence, self.index_num, 1)
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
        elif index == "down" and self.index_num != self.model().rowCount() - 1 and self.index_num != 0:
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

    def update_at_itemmove(self, insert_index, new_item, pop_index, list: list, old_item_num, new_item_num):
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
            self.select_ai_action = QAction("设为评判模型", self)
            self.select_ai_action.triggered.connect(lambda: self.select_ai(index))

            self.old_name = index.data()
            self.rename_select_ai_action(index)
            self.disabled_rename_action(index, rename_action)

            menu.addAction(open_action)
            menu.addAction(self.select_ai_action)
            menu.addAction(delete_action)
            menu.addAction(rename_action)
            menu.exec_(self.mapToGlobal(pos))

    def disabled_rename_action(self, index, action):
        if index.row() == 0:
            action.setEnabled(False)
        else:
            action.setEnabled(True)

    def rename_select_ai_action(self, index: QModelIndex):
        if self.check_item_isai(index.data()):
            self.select_ai_action.setEnabled(True)
            if self.config[0].default_ai == index.data():
                self.select_ai_action.setText("取消设定")
            else:
                self.select_ai_action.setText("设为评判模型")
        else:
            self.select_ai_action.setEnabled(False)

    def select_ai(self, index):
        is_ai = self.check_item_isai(index.data())
        if is_ai is False:
            return
        item = self.model().itemFromIndex(index)
        if self.config[0].default_ai == index.data():
            item.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/blank_icon.png"))
            self.config[0].default_ai = None
            self.prev_select_ai = None
        else:
            if self.prev_select_ai is None:
                self.prev_select_ai = index
            else:
                prev_ai_item = self.model().itemFromIndex(self.prev_select_ai)
                prev_ai_item.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/blank_icon.png"))
                self.prev_select_ai = index
            self.config[0].default_ai = index.data()
            item.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/star.png"))

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
                model = PlayRecordConfigWindow(self.config[0].detail)
            elif "录制音频" in self.config[0].name:
                model = RecordConfigWindow(self.config[0].detail)
            result = model.exec()
            if result is not None:
                self.config[0].detail = result
                if "播放与录制" in name:
                    self.signal_len = int(
                        result["stimulus_info"]["total_time"] * result["stimulus_info"]["sample_rate"]
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
            model = self.create_config_dialog(model, config_manager, model_type, type, self.signal_len)
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

    def create_config_dialog(self, model: QDialog, config_manager: ConfigManager, name, type, signal_len):
        if type == "SPL":
            model = SplConfigWindow(config_manager, name)
        elif type == "FR":
            model = FrConfigWindow(config_manager, name)
        elif type == "HD":
            model = HdConfigWindow(config_manager, name)
        elif type == "AI":
            model = AIConfigWindow(config_manager, name, signal_len)
        elif type == "Spec":
            model = SpecConfigWindow(config_manager, name)
        elif type == "LP":
            model = LPConfigWindow(config_manager, name)
        elif type == "PD":
            model = PDConfigWindow(config_manager, name)
        elif type == "PM":
            model = PatternMatchConfigWindow(config_manager, name)
        return model

    def init_config_info(self, config_file):
        code, config_info = LoadUiConfig.load_data_from_json(config_file)
        if code != 0:
            self.default_logger.error(f"Failed to load the default config file. {config_info}")
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
                sequence_config.default_ai = i_analysis_list.pop("default_ai", None)
                sequence_config.display_sequence = i_analysis_list.pop("display_sequence", [])
                sequence_config.auto_analysis = i_analysis_list.pop("auto_analysis", False)

                sequence_config.analysis_list.update(i_analysis_list)
                self.config.append(sequence_config)

    def clear_option_list(self):
        self.config = list()
        self.data_struct.clear_fft_and_stft_flag()
        self.model().clear()
        self.prev_select_ai = None
        self.all_ai_item = []
        self.sound_item_type = None
        self.drop_is_accept = True

    def load_model_config(self, config_path):
        if os.path.exists(config_path):
            self.clear_option_list()
            self.init_config_info(config_path)
        else:
            return
        for config in self.config:
            if config.mode:
                mode_item = QStandardItem(config.name)
                mode_item.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/blank_icon.png"))
                self.model().appendRow(mode_item)
            else:
                continue
            for key, value in self.config[0].analysis_list.items():
                if key != "auto_analysis" and key != "default_ai" and key != "display_sequence":
                    if "AI" == value.get("type"):
                        self.store_ai_item(self.all_ai_item, key)
            model_item_list = self.config[0].display_sequence
            for item_name in model_item_list:
                self.data_struct.add_stft_or_fft_count(self.config[0].analysis_list[item_name]["type"])
                if item_name == self.config[0].default_ai:
                    list_item = QStandardItem(item_name)
                    list_item.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/star.png"))
                    self.model().appendRow(list_item)
                    last_row = self.model().rowCount() - 1
                    self.prev_select_ai = self.model().index(last_row, 0)
                else:
                    list_item = QStandardItem(item_name)
                    list_item.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/blank_icon.png"))
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
            self.data_struct.minus_stft_or_fft_count(self.config[0].analysis_list[index.data()]["type"])
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
        if name in self.config[0].analysis_list:
            del self.config[0].analysis_list[name]

    def rename_item(self, index):
        self.is_update_config = True
        self.is_select_ai = self.config[0].default_ai == self.model().data(index)
        self.edit(index)

    def update_model_list(self, config: dict, new_item: QStandardItem, old_index, new_index, step_index: bool):
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
                self.swap_list_index(config["display_sequence"], old_index, new_index - 1)
                self.start_row_number = new_index - 1
            else:
                self.model().removeRow(old_index + 1)
                self.swap_list_index(config["display_sequence"], old_index, new_index)
                if new_index != 0:
                    self.start_row_number = new_index
        self.setCurrentIndex(self.model().index(self.start_row_number, 0))
        self.update_select_ai(old_index, new_index, True)

    def update_select_ai(self, old_index, new_index, step_index: bool):
        if old_index == new_index or old_index == -1 or new_index == -1 or not self.prev_select_ai:
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
                    self.update_config_data(self.old_name, new_name, self.config[0].display_sequence)
                self.old_name = new_name
                self.config[0].default_ai = new_name
                self.is_select_ai = False
            else:
                if self.is_update_config:
                    self.update_config_data(self.old_name, new_name, self.config[0].display_sequence)
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
                new_item.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/star.png"))
            else:
                new_item.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/blank_icon.png"))
            if row_number == -1:
                self.update_model_list(
                    self.config[0].analysis_list, new_item, self.start_row_number, self.model().rowCount(), True
                )
            else:
                if row_number > self.start_row_number:
                    self.update_model_list(
                        self.config[0].analysis_list, new_item, self.start_row_number, row_number, True
                    )
                elif row_number < self.start_row_number:
                    self.update_model_list(
                        self.config[0].analysis_list, new_item, self.start_row_number, row_number, False
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
            if text in ["播放与录制", "录制音频"]:
                if self.sound_item_type:
                    self.drop_is_accept = False
            elif self.sound_item_type == "录制音频":
                if text in ["频响 (FR) ", "谐波失真 (HD) "]:
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
                if text in ["播放与录制", "录制音频"]:
                    QMessageBox.warning(self, "警告", "已选择测试模式")
                elif not self.config:
                    QMessageBox.warning(self, "警告", "请选择测试模式")
                elif text in ["频响 (FR) ", "谐波失真 (HD) "]:
                    QMessageBox.warning(self, "警告", "当前模式不支持此功能")
                self.drop_is_accept = True
                return
            elif text in ["播放与录制", "录制音频"]:
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
        list_item.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/blank_icon.png"))
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
            else:
                QMessageBox.warning(self, "提示", "窗口配置错误，请检查配置!")
                return
        elif item_text == "录制音频":
            seq_item.mode = "RECORD_ONLY"
            seq_item.detail = {"total_time": 4.0, "sample_rate": 44100}
        self.config.append(seq_item)

        self.model().insertRow(0, list_item)

    def set_new_analysis_config(self, item_text):
        count = 1
        item_exist = self.model().findItems(item_text + f"{count}")
        while item_exist:
            count += 1
            item_exist = self.model().findItems(item_text + f"{count}")
        list_item = QStandardItem(item_text + f"{count}")
        list_item.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/select_analysis_model/blank_icon.png"))
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
