import json
import os
import re
import sys

from PyQt5.QtCore import Qt, QModelIndex, QSize
from PyQt5.QtGui import QIcon, QStandardItemModel, QStandardItem, QPixmap
from PyQt5.QtWidgets import QDialog, QLabel, QListView, QVBoxLayout, QCheckBox, QHBoxLayout, QPushButton, QApplication, QMenu, \
     QAction, QSpacerItem, QSizePolicy, QFileDialog, QMessageBox
from time import time

from base.log_manager import LogManager
from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.analysis_config_window import SplConfigWindow, ConfigManager, FrConfigWindow, HdConfigWindow, AIConfigWindow


class AnalysisModelSellect(QDialog):

    def __init__(self):
        super().__init__()

        self.analysis_list = QListView()
        self.default_logger = LogManager.set_log_handler("core")
        self.sellect_list = OptionList(self.default_logger)
        self.analysis_list.setEditTriggers(QListView.NoEditTriggers)
        self.sellect_list.setEditTriggers(QListView.NoEditTriggers)

        self.drag_drop_function()
        self.init_ui()

    def init_ui(self):
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setWindowTitle("分析队列")

        analysis_list_layout = self.create_analysis_list_layout()
        sellect_list_layout = self.create_sellect_list_layout()
        btn_layout = self.create_btn_layout()
        move_btn_layout = self.move_item_btn_layout()  

        arrow_label = QLabel()
        arrow_label.setPixmap(QPixmap(DEFAULT_DIR + "ui/ui_pic/sellect_analysis_model/jiantou.svg"))
        arrow_label.setFixedSize(50, 50)
        arrow_label.setScaledContents(True)

        analysis_layout = QHBoxLayout()
        analysis_layout.addLayout(analysis_list_layout)
        analysis_layout.addWidget(arrow_label)
        analysis_layout.addLayout(sellect_list_layout)
        analysis_layout.addLayout(move_btn_layout)

        layout = QVBoxLayout()
        layout.addLayout(analysis_layout)
        layout.addLayout(btn_layout)
        

        self.setLayout(layout)

        self.setStyleSheet(ui_style_const.qcombobox_stytle +
                           ui_style_const.qpushbutton_stytle +
                           ui_style_const.qlabel_stytle + 
                           ui_style_const.qcheckbox_stytle +
                           ui_style_const.qlistview_stytle)
        
    def drag_drop_function(self):
        self.analysis_list.setDragEnabled(True)
        self.analysis_list.setAcceptDrops(False)
        self.analysis_list.setDragDropMode(QListView.DragOnly)
        self.analysis_list.setDefaultDropAction(Qt.CopyAction)

        self.sellect_list.setDragEnabled(True)
        self.sellect_list.setDragDropMode(QListView.DragDrop)
        self.sellect_list.setDefaultDropAction(Qt.MoveAction)
        self.sellect_list.setDropIndicatorShown(True)
        self.sellect_list.setDragDropOverwriteMode(False)
        self.sellect_list.setMovement(QListView.Snap)
        self.sellect_list.setFlow(QListView.TopToBottom)

    def up_btn_clicked(self):
        self.sellect_list.itemmove("up")

    def down_btn_clicked(self):
        self.sellect_list.itemmove("down")
    
    def top_btn_clicked(self):
        self.sellect_list.itemmove("top")

    def bottom_btn_clicked(self):
        self.sellect_list.itemmove("bottom")

    def create_analysis_list_layout(self):
        analysis_label = QLabel("可选分析")

        self.analysis_model = QStandardItemModel()
        items = ["声压级 (SPL) ", "频响 (FR) ", "谐波失真 (HD) ", "松散颗粒 (LP) ", "AI 分析 "]
        for item in items:
            list_item = QStandardItem(item)
            self.analysis_model.appendRow(list_item)
        self.analysis_list.setModel(self.analysis_model)

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

        up_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sellect_analysis_model/up.svg"))
        up_btn.setIconSize(QSize(30, 30))
        down_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sellect_analysis_model/down.svg"))
        down_btn.setIconSize(QSize(30, 30))
        top_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sellect_analysis_model/top.svg"))
        top_btn.setIconSize(QSize(30, 30))
        bottom_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sellect_analysis_model/bottom.svg"))
        bottom_btn.setIconSize(QSize(30, 30))

        up_btn.clicked.connect(self.up_btn_clicked)
        down_btn.clicked.connect(self.down_btn_clicked)
        top_btn.clicked.connect(self.top_btn_clicked)
        bottom_btn.clicked.connect(self.bottom_btn_clicked)

        space = QSpacerItem(30, 50, QSizePolicy.Minimum, QSizePolicy.Expanding)

        layout = QVBoxLayout()
        layout.addWidget(top_btn)
        layout.addWidget(up_btn)
        layout.addWidget(down_btn)
        layout.addWidget(bottom_btn)
        layout.addItem(space)
        layout.setContentsMargins(0, 30, 0, 0)

        return layout
    
    def create_sellect_list_layout(self):
        sellect_analysis_label = QLabel("分析")
        self.auto_analysis_box = QCheckBox("自动分析")
        self.auto_analysis_box.setChecked(self.sellect_list.config.get("auto_analysis", True))
        self.auto_analysis_box.setLayoutDirection(Qt.RightToLeft)

        analysis_title_layout = QHBoxLayout()
        analysis_title_layout.addWidget(sellect_analysis_label)
        analysis_title_layout.addWidget(self.auto_analysis_box)

        layout = QVBoxLayout()
        layout.addLayout(analysis_title_layout)
        layout.addWidget(self.sellect_list)

        return layout
    
    def create_btn_layout(self):
        load_btn = QPushButton("导入")
        load_btn.clicked.connect(self.load_btn_clicked)
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self.save_btn_clicked)
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(self.ok_btn_clicked)
        clear_btn = QPushButton("清空")
        clear_btn.clicked.connect(self.clear_btn_clicked)
        load_btn.setMinimumWidth(100)
        save_btn.setMinimumWidth(100)
        ok_btn.setMinimumWidth(100)
        clear_btn.setMinimumWidth(100)

        space = QSpacerItem(30, 50, QSizePolicy.Expanding, QSizePolicy.Minimum)

        layout = QHBoxLayout()
        layout.addItem(space)
        layout.addWidget(clear_btn)
        layout.addWidget(load_btn)
        layout.addWidget(save_btn)
        layout.addWidget(ok_btn)
        layout.setSpacing(20)

        return layout
    
    def clear_btn_clicked(self):
        self.sellect_list.config = {"display_sequence":[],
                                    "default_ai":None}
        self.sellect_list.model().clear()
        self.sellect_list.prev_sellect_ai = None
        self.sellect_list.all_ai_item = []

    def load_btn_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "导入配置文件",
                                                   DEFAULT_DIR + "ui/ui_config/analysis_sequence_config",
                                                   filter="JSON Files (*.json);;All Files (*)")
        if file_path:
            try:
                self.sellect_list.load_model_config(file_path)
            except Exception as e:
                self.default_logger.error(f"Unable to parse JSON data in {file_path}. {e}")

    def save_btn_clicked(self):
        file_path, _ = QFileDialog.getSaveFileName(self,
                                                   "保存配置文件",
                                                   DEFAULT_DIR + "ui/ui_config/analysis_sequence_config",
                                                   filter="JSON Files (*.json);;All Files (*)")
        if file_path:
            self.sellect_list.config["auto_analysis"] = self.auto_analysis_box.isChecked()
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    json.dump(self.sellect_list.config, file, indent=4)
                self.default_logger.info(f"The config file has been saved to {file_path}")
            except Exception as e:
                self.default_logger.error(f"The config file saved failed. {e}")

    def ok_btn_clicked(self):
        self.sellect_list.config["auto_analysis"] = self.auto_analysis_box.isChecked()
        self.save_analyse_config_to_json(self.sellect_list.config)
        self.close()

    def save_analyse_config_to_json(self, config_data):
        analyse_config_file = DEFAULT_DIR + "ui/ui_config/analysis_temp_config.json"
        try:
            with open(analyse_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4)
                self.default_logger.info(f"The config info for analysis has been saved to {analyse_config_file}.")
                return True
        except Exception as e:
            self.default_logger.error(f"The config info for analysis save failed. {e}")
            return False
        

class OptionList(QListView):
    def __init__(self, logger):
        super().__init__()
        self.sellect_analysis_model = QStandardItemModel()
        self.setModel(self.sellect_analysis_model)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        self.sellect_analysis_model.dataChanged.connect(self.is_edit_model_item)

        self.default_logger = logger
        self.row_num = None
        self.darpflag = None
        self.start_row_number = None
        self.old_name = None
        self.press_time = None
        self.prev_sellect_ai = None
        self.is_edit_item = True
        self.index_num = None
        self.all_ai_item = []
        self.config = {"display_sequence":[],
                       "default_ai":None}
        self.load_model_config(DEFAULT_DIR + "ui/ui_config/analysis_temp_config.json")

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

        if index == "top":
            self.updata_at_itemmove(0, new_item, 
                                    self.index_num + 1, 
                                    self.config["display_sequence"], self.index_num, 0)
            self.index_num = 0
        elif index == "bottom":
            self.updata_at_itemmove(self.model().rowCount(), new_item, 
                                    self.index_num, 
                                    self.config["display_sequence"], self.index_num, self.model().rowCount() - 1)
            self.index_num = self.model().rowCount() - 1     
        elif index == "up" and self.index_num != 0:
            self.updata_at_itemmove(self.index_num - 1, new_item, 
                                    self.index_num + 1, 
                                    self.config["display_sequence"], self.index_num, self.index_num - 1)
            self.index_num -= 1
        elif index == "down" and self.index_num != self.model().rowCount() - 1:
            self.updata_at_itemmove(self.index_num + 2, new_item, 
                                    self.index_num, 
                                    self.config["display_sequence"], self.index_num, self.index_num + 1)
            self.index_num += 1
        self.setCurrentIndex(self.model().index(self.index_num, 0))

    def updata_at_itemmove(self, insert_index, new_item, pop_index, list: list, old_item_num, new_item_num):
        self.model().insertRow(insert_index, new_item)
        self.model().removeRow(pop_index)
        self.swap_list_index(list,old_item_num, new_item_num)
        self.updata_sellect_ai(old_item_num, new_item_num, False)

    def show_context_menu(self, pos):
        index = self.indexAt(pos)
        if index.isValid():
            menu = QMenu(self)
            menu.setStyleSheet(ui_style_const.main_window_menubar_stytle)
            open_action = QAction("打开", self)
            open_action.triggered.connect(lambda: self.show_dialog(index.data()))
            delete_action = QAction("删除", self)
            delete_action.triggered.connect(lambda: self.delete_item(index))
            rename_action = QAction("重命名", self)
            rename_action.triggered.connect(lambda: self.rename_item(index))
            self.sellect_ai_action = QAction("设为评判模型", self)
            self.sellect_ai_action.triggered.connect(lambda: self.sellect_ai(index))

            self.old_name = index.data()
            self.rename_sellect_ai_action(index)
                
            menu.addAction(open_action)
            menu.addAction(self.sellect_ai_action)
            menu.addAction(delete_action)
            menu.addAction(rename_action)
            menu.exec_(self.mapToGlobal(pos))

    def rename_sellect_ai_action(self, index: QModelIndex):
        if self.check_item_isai(index.data()) or "\u2605" in index.data():
            self.sellect_ai_action.setEnabled(True)
            if "\u2605" in index.data():
                self.sellect_ai_action.setText("取消设定")
            else:
                self.sellect_ai_action.setText("设为评判模型")
        else:
            self.sellect_ai_action.setEnabled(False)

    def valid_char(self, text):
        valid_chars = text.replace("\xa0", "")
        return valid_chars

    def sellect_ai(self, index):
        is_ai = self.check_item_isai(index.data()) or "\u2605" in index.data()
        if is_ai is False:
            return
        if "\u2605" in index.data():
            new_name = index.data().replace("\u2605", "\xa0")
            self.set_model_data(index, new_name)
            self.config["default_ai"] = None
            self.prev_sellect_ai = None
        else:
            if self.prev_sellect_ai is None:
                self.prev_sellect_ai = index
            else:
                prev_new_name = self.prev_sellect_ai.data().replace("\u2605", "\xa0")
                self.set_model_data(self.prev_sellect_ai, prev_new_name)
                self.prev_sellect_ai = index
            self.config["default_ai"] =index.data()
            self.is_select_ai = True
            new_name = "\u2605" + self.valid_char(index.data())
            self.set_model_data(index, new_name)

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
        if "\u2605" in name:
            name = name.replace("\u2605", "\xa0")
        if not name in self.config["display_sequence"]:
            return
        prev_config_file = DEFAULT_DIR + "ui/ui_config/analysis_temp_config.json"
        model_type = None
        config_manager = None
        if name in self.config:
            config_manager = ConfigManager(prev_config_file)
        model = QDialog(self)
        type = self.config.get(name)["type"]
        if self.config.get(name):
            config_manager.config = self.config
            model_type = name
        model = self.create_config_dialog(model, config_manager, model_type, type)
        model.setWindowTitle(name)
        if model.exec_() == QDialog.Accepted:
            config_data = model.on_click_ok_btn()
            self.add_config(name, config_data)

    def create_config_dialog(self, model: QDialog, config_manager: ConfigManager, name, type):
        if type == "SPL":
            model = SplConfigWindow(config_manager, name)
        elif type == "FR":
            model = FrConfigWindow(config_manager, name)
        elif type == "HD":
            model = HdConfigWindow(config_manager, name)
        elif type == "AI":
            model = AIConfigWindow(config_manager, name)
        return model

    def load_config(self, config_file):
        try:
            with open(config_file, 'r') as f:
                default_config = json.load(f)
            return default_config
        except Exception as e:
            self.default_logger.error(f"Failed to load the default config file. {e}")
            return {}
        
    def clear_option_list(self):
        self.config = {"display_sequence":[],
                       "default_ai":None}
        self.model().clear()
        self.prev_sellect_ai = None
        self.all_ai_item = []
        
    def load_model_config(self, config_path):
        if os.path.exists(config_path):
            self.clear_option_list()
            self.config = self.load_config(config_path)
        for key, value in self.config.items():
            if key != "auto_analysis" and key != "default_ai" and key != "display_sequence":
                if "AI" == value.get("type"):
                    self.store_ai_item(self.all_ai_item, key)
        model_item_list = self.config.get("display_sequence", "")
        for item_name in model_item_list:
            if item_name == self.config.get("default_ai", ""):
                self.config["default_ai"] = item_name
                item_name = item_name.replace("\xa0", "")
                self.model().appendRow(QStandardItem("\u2605" + item_name))
                last_row = self.model().rowCount() - 1
                self.prev_sellect_ai = self.model().index(last_row, 0)
            else:
                self.model().appendRow(QStandardItem(item_name))

    def add_config(self, class_name, config_data):
        if class_name in self.config:
            self.config[class_name].update(config_data)
        else:
            self.config[class_name] = config_data

    def delete_item(self, index):
        if "\u2605" in index.data():
            self.config["display_sequence"].remove(self.config["default_ai"])
            self.delete_item_config(self.config["default_ai"])
            self.config["default_ai"] = None
            self.prev_sellect_ai = None
        else:
            self.config["display_sequence"].remove(index.data())
            self.delete_item_config(index.data())
        model = self.model() 
        model.removeRow(index.row())
        self.updata_sellect_ai(index.row() + 1, index.row(), False)

    def delete_item_config(self, name):
        if not name:
            return
        if name in self.config:
            del self.config[name]

    def rename_item(self, index):
        self.is_updata_config = True
        self.is_select_ai = "\u2605" in self.model().data(index)
        self.edit(index)

    def updata_model_list(self, config: dict, new_item: QStandardItem, old_index, new_index, step_index: bool):
        if not new_item:
            return
        self.model().insertRow(new_index, new_item)
        if step_index:
            self.model().removeRow(old_index)
            self.setCurrentIndex(self.model().index(new_index - 1, 0))
            self.swap_list_index(config["display_sequence"], old_index, new_index - 1)
            self.start_row_number = new_index - 1
        else:
            self.model().removeRow(old_index + 1)
            self.setCurrentIndex(self.model().index(new_index, 0))
            self.swap_list_index(config["display_sequence"], old_index, new_index)
            self.start_row_number = new_index
        self.updata_sellect_ai(old_index, new_index, True)

    def updata_sellect_ai(self, old_index, new_index, step_index: bool):
        if old_index == new_index or old_index == -1 or new_index == -1 or not self.prev_sellect_ai: 
            return
        
        sellect_ai_row = self.prev_sellect_ai.row()
        if sellect_ai_row < old_index and sellect_ai_row >= new_index:
            sellect_ai_row = sellect_ai_row + 1
            self.prev_sellect_ai = self.model().index(sellect_ai_row, 0)
        elif sellect_ai_row > old_index and sellect_ai_row <= new_index:
            sellect_ai_row = sellect_ai_row - 1
            self.prev_sellect_ai = self.model().index(sellect_ai_row, 0)
        elif sellect_ai_row == old_index:
            if step_index:  
                if new_index > old_index:
                    self.prev_sellect_ai = self.model().index(new_index - 1, 0)
                elif new_index < old_index:
                    self.prev_sellect_ai = self.model().index(new_index, 0)
            else:  
                self.prev_sellect_ai = self.model().index(new_index, 0)

    def set_model_data(self, index:QModelIndex, name):
        self.is_edit_item = False
        self.model().setData(index, name)

    def chack_item_name_have_space(self, name):
        if not name:
            return None
        if name.find("\xa0") == 0:
            return True
        else:
            return False

    def updata_config_file(self, old_name, new_name):
        if not old_name or not new_name:
            return
        if old_name in self.config:
            value = self.config.pop(old_name)
            if self.chack_item_name_have_space(new_name) is True:
                self.config[new_name] = value
            elif self.chack_item_name_have_space(new_name) is False:
                self.config["\xa0" + new_name] = value

    def updata_display_sequence_list(self, list, old_name, new_name):
        if not old_name or not new_name or old_name == new_name:
            return
        try:
            index = list.index(old_name)
        except ValueError:
            return
        if self.chack_item_name_have_space(new_name) is True:
            list[index] = new_name
        elif self.chack_item_name_have_space(new_name) is False:
            list[index] = "\xa0" + new_name

    def rename_item_space_equal(self, name, index):
        if not name or not index:
            return
        if self.chack_item_name_have_space(name) is True:
            self.set_model_data(index, name)
            return name
        elif self.chack_item_name_have_space(name) is False:
            self.set_model_data(index, "\xa0" + name)
            return "\xa0" + name

    def updata_config_data(self, old_name, new_name, list):
        if not new_name in list:
            self.updata_config_file(old_name, new_name)
            self.updata_display_sequence_list(list, old_name, new_name)
        self.updata_ai_list(new_name, old_name)
        
    def updata_ai_list(self, new_name, old_name):
        if not old_name in self.all_ai_item or old_name == new_name:
            return
        if not new_name in self.all_ai_item:
            index = self.all_ai_item.index(old_name)
            if self.chack_item_name_have_space(new_name) is True:
                self.all_ai_item[index] = new_name
            elif self.chack_item_name_have_space(new_name) is False:
                self.all_ai_item[index] = "\xa0" + new_name

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
            if new_name in self.config["display_sequence"]:
                QMessageBox.warning(self, "警告", "模型名称重复，请重新输入！")
                self.set_model_data(index, self.old_name)
                return
            if self.is_select_ai:
                if "\u2605" in new_name:
                    self.config["default_ai"] = new_name.replace("\u2605","\xa0")
                    if self.config["default_ai"] != self.old_name:
                        self.updata_config_data(self.old_name.replace("\u2605","\xa0"), self.config["default_ai"], self.config["display_sequence"])
                    self.old_name = new_name
                    self.set_model_data(index, new_name)
                else:
                    self.updata_config_data(self.old_name.replace("\u2605","\xa0"), new_name, self.config["display_sequence"])
                    self.config["default_ai"] = new_name.replace("\u2605","\xa0")
                    self.old_name = new_name
                    self.config["default_ai"] = new_name
                    self.set_model_data(index, "\u2605" + new_name)
                self.is_select_ai = False
            else:
                if self.is_updata_config:
                    self.updata_config_data(self.old_name, new_name, self.config["display_sequence"])
                    self.is_updata_config = False
                self.old_name = "\xa0" + new_name
                self.rename_item_space_equal(new_name, index)
        else:
            if new_name == self.old_name:
                return 
            self.set_model_data(index, self.old_name)

    def swap_list_index(self, list:list, old_index, new_index):
        if old_index == new_index or not list:
            return
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
            new_item = QStandardItem(self.start_index.data())
            if row_number == -1:
                self.updata_model_list(self.config, new_item, self.start_row_number, self.model().rowCount(), True)
            else:
                if row_number > self.start_row_number:
                    self.updata_model_list(self.config, new_item, self.start_row_number, row_number, True)
                else:
                    self.updata_model_list(self.config, new_item, self.start_row_number, row_number, 0)
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
        if event.mimeData().hasFormat('application/x-qabstractitemmodeldatalist'):
            item_model = QStandardItemModel()
            item_model.dropMimeData(event.mimeData(), Qt.MoveAction, 0, 0, QModelIndex())
            item_text = item_model.item(0).text()

            if "松散颗粒 (LP)" in item_text:
                event.ignore()
            else:
                event.accept()
        else:
            event.ignore()

    def dragmoveevent(self, event):
        if event.mimeData().hasFormat('application/x-qabstractitemmodeldatalist'):
            event.setDropAction(Qt.MoveAction)
            item_model = QStandardItemModel()
            item_model.dropMimeData(event.mimeData(), Qt.MoveAction, 0, 0, QModelIndex())
            item_text = item_model.item(0).text()

            if "松散颗粒 (LP)" in item_text:
                event.ignore()
            else:
                event.accept()
        else:
            event.ignore()

    def dropevent(self, event):
        if event.mimeData().hasFormat('application/x-qabstractitemmodeldatalist'):
            mime_data = event.mimeData()
            item_model = QStandardItemModel()
            if isinstance(item_model, QStandardItemModel):
                item_model.dropMimeData(mime_data, Qt.MoveAction, 0, 0, QModelIndex())

                for row in range(item_model.rowCount()):
                    item = item_model.item(row)
                    item_text = item.text()
                    count = 1                    
                    item_exist = self.model().findItems("\xa0" + item_text + f"{count}")
                    item_star_exist = self.model().findItems("\u2605" + item_text + f"{count}")
                    while  item_exist or item_star_exist:
                        count += 1 
                        item_exist = self.model().findItems("\xa0" + item_text + f"{count}")
                        item_star_exist = self.model().findItems("\u2605" + item_text + f"{count}")                                

                    list_item = QStandardItem("\xa0" + item_text + f"{count}")
                    self.model().insertRow(self.model().rowCount(), list_item)
                    list_item_text = list_item.text()
                    if "AI" in item_text:
                        self.store_ai_item(self.all_ai_item, list_item_text)
                    self.config["display_sequence"].append(list_item_text)
                    self.get_item_default_config(item_text, list_item_text)
                    event.accept()
            else:
                event.ignore()
        else:
            event.ignore()

    def get_item_default_config(self, item_text, list_item_text):
        if not item_text or not list_item_text:
            return
        type = ''.join(re.findall(r'[A-Za-z]', item_text))
        default_config_file = DEFAULT_DIR + "ui/ui_config/analysis_default_config.json"
        data = self.load_config(default_config_file)
        self.config[list_item_text] = data[type]
        self.config[list_item_text]["type"] = type

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnalysisModelSellect()
    window.show()
    sys.exit(app.exec_())
         