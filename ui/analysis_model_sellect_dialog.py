import json
import re
import sys

from PyQt5.QtCore import Qt, QModelIndex, QSize
from PyQt5.QtGui import QIcon, QStandardItemModel, QStandardItem, QPixmap
from PyQt5.QtWidgets import QDialog, QLabel, QListView, QVBoxLayout, QCheckBox, QHBoxLayout, QPushButton, QApplication
from PyQt5.QtWidgets import QMenu, QAction, QSpacerItem, QSizePolicy, QFileDialog, QMessageBox
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
        self.setWindowTitle("方法选择")

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

    def up_btn_clicked(self):
        self.sellect_list.itemmove("up")

    def down_btn_clicked(self):
        self.sellect_list.itemmove("down")
    
    def top_btn_clicked(self):
        self.sellect_list.itemmove("top")

    def bottom_btn_clicked(self):
        self.sellect_list.itemmove("bottom")
    
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
        load_btn.setMinimumWidth(100)
        save_btn.setMinimumWidth(100)
        ok_btn.setMinimumWidth(100)

        space = QSpacerItem(30, 50, QSizePolicy.Expanding, QSizePolicy.Minimum)

        layout = QHBoxLayout()
        layout.addItem(space)
        layout.addWidget(load_btn)
        layout.addWidget(save_btn)
        layout.addWidget(ok_btn)
        layout.setSpacing(20)

        return layout

    def load_btn_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "导入配置文件",
                                                   DEFAULT_DIR + "ui/ui_config/analysis_sequence_config",
                                                   filter="JSON Files (*.json);;All Files (*)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    self.sellect_list.config = data
                    self.append_row_to_select_list(data)
            except Exception as e:
                self.default_logger.error(f"Unable to parse JSON data in {file_path}. {e}")

    def append_row_to_select_list(self, data):
        sellect_analysis_model = self.sellect_list.sellect_analysis_model
        sellect_analysis_model.clear()
        for key, value in data.items():
            if key == "auto_analysis":
                self.auto_analysis_box.setChecked(value)
            else:
                list_item = QStandardItem(key)
                sellect_analysis_model.appendRow(list_item)
    
    def ok_btn_clicked(self):
        self.sellect_list.config["auto_analysis"] = self.auto_analysis_box.isChecked()
        self.save_analyse_config_to_json(self.sellect_list.config)
        self.close()

    def ok_btn_popup(self):
        ok_msg = QMessageBox(self)
        ok_msg.setIcon(QMessageBox.Warning)
        ok_msg.setText("请进行参数配置")
        ok_msg.setWindowTitle("设置警告")
        ok_msg.setStandardButtons(QMessageBox.Ok)
        ok_msg.exec_()

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

    def save_analyse_config_to_json(self, config_data):
        analyse_config_file = DEFAULT_DIR + "ui/ui_config/analysis_temp_config.json"
        try:
            with open(analyse_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4)
                self.default_logger.info(f"The config info for analysis has been saved to {self.config_file}.")
                return True
        except Exception as e:
            self.default_logger.error(f"The config info for analysis save failed. {e}")
            return False

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


class OptionList(QListView):
    def __init__(self, logger):
        super().__init__()
        self.sellect_analysis_model = QStandardItemModel()
        self.setModel(self.sellect_analysis_model)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        self.sellect_analysis_model.dataChanged.connect(self.on_data_changed)
        self.default_logger = logger
        self.row_num = None
        self.press_time = None  
        self.index_num = None
        self.darpflag = None
        self.start_row_number = None
        self.old_name = None
        self.config = {}

        self.dragEnterEvent = self.dragenterevent
        self.dragMoveEvent = self.dragmoveevent
        self.dropEvent = self.dropevent
        self.mousePressEvent = self.mousepressevent
        self.mouseReleaseEvent = self.mousereleaseevent 

    def show_dialog(self, name):
        config_file = DEFAULT_DIR + "ui/ui_config/analysis_default_config.json"
        config_manager = ConfigManager(config_file)
        model = QDialog(self)
        model.setWindowTitle(name)
        type = self.config[name]["type"]
        if type == "SPL":
            model = SplConfigWindow(config_manager)
        elif type == "FR":
            model = FrConfigWindow(config_manager)
        elif type == "HD":
            model = HdConfigWindow(config_manager)
        elif type == "AI":
            model = AIConfigWindow(config_manager)
        if model.exec_() == QDialog.Accepted:
            config_data = model.on_click_ok_btn()
            self.add_config(name, config_data)

    def add_config(self, class_name, config_data):
        if class_name in self.config:
            self.config[class_name].update(config_data)
        else:
            self.config[class_name] = config_data

    def load_default_config(self):
        default_config_file = DEFAULT_DIR + "ui/ui_config/analysis_default_config.json"
        try:
            with open(default_config_file, 'r') as f:
                default_config = json.load(f)
            return default_config
        except Exception as e:
            self.default_logger.error(f"Failed to load the default config file. {e}")
            return {}

    def show_context_menu(self, pos):
        index = self.indexAt(pos)
        if index.isValid():
            menu = QMenu(self)
            menu.setStyleSheet(ui_style_const.main_window_menubar_stytle)
            delete_action = QAction("删除", self)
            delete_action.triggered.connect(lambda: self.delete_item(index))
            rename_action = QAction("重命名", self)
            rename_action.triggered.connect(lambda: self.rename_item(index))
            menu.addAction(delete_action)
            menu.addAction(rename_action)
            menu.exec_(self.mapToGlobal(pos))

    def delete_item(self, index):
        model = self.model()
        index_name = model.data(index)
        self.delete_item_config(index_name)
        model.removeRow(index.row())

    def delete_item_config(self, name):
        if name in self.config:
            del self.config[name]

    def rename_item(self, index):
        self.old_name = self.model().data(index)
        self.edit(index)

    def on_data_changed(self, topLeft, bottomRight, roles):
        if Qt.EditRole in roles:
            for row in range(topLeft.row(), bottomRight.row() + 1):
                index = self.model().index(row, topLeft.column())
                new_name = self.model().data(index)
                if self.old_name in self.config:
                    value = self.config.pop(self.old_name)
                    self.config[new_name] = value
                self.old_name = None

    def itemmove(self, index):
        if self.index_num is None:
            return
        new_item = QStandardItem(self.start_item_name)
        if index == "top":
            self.model().insertRow(0, new_item)
            self.model().removeRow(self.index_num + 1)
            self.index_num = 0
        elif index == "bottom":
            self.model().insertRow(self.model().rowCount(), new_item)
            self.model().removeRow(self.index_num)
            self.index_num = self.model().rowCount() - 1
        elif index == "up" and self.index_num != 0:
            self.model().insertRow(self.index_num - 1, new_item)
            self.model().removeRow(self.index_num + 1)
            self.index_num -= 1
        elif index == "down" and self.index_num != self.model().rowCount() - 1:
            self.model().insertRow(self.index_num + 2, new_item) 
            self.model().removeRow(self.index_num)
            self.index_num += 1
        self.setCurrentIndex(self.model().index(self.index_num, 0))

    def mousepressevent(self, e):
        index = self.indexAt(e.pos())
        if index.isValid():
            self.darpflag = True
            self.start_row_number = index.row()
            self.start_item_name = self.model().itemFromIndex(index).text()
            self.index_num = self.start_row_number

    def mousereleaseevent(self, e):
        t1 = time()
        if self.press_time != None:
            time_area = t1 - self.press_time
            if time_area > 0.7:
                self.row_num = None
        self.press_time = t1
        index = self.indexAt(e.pos())
        row_number = index.row()
        if self.darpflag:       
            new_item = QStandardItem(self.start_item_name)
            if row_number == -1:
                self.model().insertRow(self.model().rowCount(), new_item)
                self.model().removeRow(self.start_row_number)
                self.setCurrentIndex(self.model().index(self.model().rowCount() - 1, 0))
            else:
                self.model().insertRow(row_number, new_item)
                if row_number > self.start_row_number:
                    self.model().removeRow(self.start_row_number)
                    self.setCurrentIndex(self.model().index(row_number - 1, 0))
                else:
                    self.model().removeRow(self.start_row_number + 1)
                    self.setCurrentIndex(self.model().index(row_number, 0))
                
            self.darpflag = False
        if self.start_row_number != row_number:
            self.row_num = None
        if self.row_num == row_number & row_number != -1:
            name_str = self.model().itemFromIndex(index).text()
            self.show_dialog(name_str)
            self.row_num = None
        else:
            self.row_num = row_number

    def dragenterevent(self, event):
        if event.mimeData().hasFormat('application/x-qabstractitemmodeldatalist'):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragmoveevent(self, event):
        if event.mimeData().hasFormat('application/x-qabstractitemmodeldatalist'):
            event.setDropAction(Qt.MoveAction)
            event.acceptProposedAction()
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
                    while self.sellect_analysis_model.findItems(item_text + f"{count}"):
                        count += 1                                 
                    list_item = QStandardItem(item_text + f"{count}")
                    self.sellect_analysis_model.insertRow(self.sellect_analysis_model.rowCount(), list_item)
                    list_item_text = list_item.text()
                    self.get_item_default_config(item_text, list_item_text)
                    event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()

    def get_item_default_config(self, item_text, list_item_text):
        type = ''.join(re.findall(r'[A-Za-z]', item_text))
        data = self.load_default_config()
        self.config[list_item_text] = data[type]
        self.config[list_item_text]["type"] = type


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnalysisModelSellect()
    window.show()
    sys.exit(app.exec_())