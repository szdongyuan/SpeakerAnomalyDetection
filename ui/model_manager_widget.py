import sys
from os import path, makedirs, remove, rename
from shutil import copy2

from PyQt5.QtCore import  Qt, QSize
from PyQt5.QtGui import  QStandardItemModel, QStandardItem, QIcon
from PyQt5.QtWidgets import QApplication, QFileDialog, QDialog, QGroupBox, QLabel, QLineEdit, QMessageBox
from PyQt5.QtWidgets import QVBoxLayout, QTableView, QHeaderView, QPushButton, QHBoxLayout

from base.training_model_management import TrainingModelManagement
from base.log_manager import LogManager
from consts import error_code, ui_style_const
from consts.running_consts import DEFAULT_DIR


class ModelInfoList(QDialog):
    def __init__(self, logger: LogManager):
        super().__init__()
        self.logger = logger
        self.model_info_list = MytableView(self.logger)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("模型列表")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setMinimumSize(QSize(530, 200))
        layout = QVBoxLayout()
        model_list_layout = self.create_model_info_list_layout()
        btn_layout = self.create_btn_layout()

        layout.addLayout(model_list_layout)
        layout.addSpacing(10)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        self.setStyleSheet(ui_style_const.qpushbutton_stytle +
                           ui_style_const.qtableview_stytle)

    def create_model_info_list_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(self.model_info_list)
        return layout
    
    def create_btn_layout(self):
        ok_btn = QPushButton("确定")
        set_new_model_btn = QPushButton("新建模型")
        register_model_btn = QPushButton("注册模型")
        del_model_btn = QPushButton("删除模型")

        del_model_btn.clicked.connect(self.model_info_list.del_model_info)
        register_model_btn.clicked.connect(self.model_info_list.register_model_info)
        set_new_model_btn.clicked.connect(self.model_info_list.set_new_model_info)

        ok_btn.clicked.connect(self.close)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(set_new_model_btn)
        btn_layout.addWidget(register_model_btn)
        btn_layout.addWidget(del_model_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        return btn_layout
        

class MytableView(QTableView):
    def __init__(self, logger: LogManager):
        super().__init__()
        self.logger = logger
        self.list_model = CustomStandardItemModel(0, 5, [0, 4])
        self.model_management = TrainingModelManagement()
        self.setModel(self.list_model)
        self.model_info = None
        self.is_edit_item = True
        self.sellect_model_row = None

        self.model().dataChanged.connect(self.is_edit_model_item)
        
        self.init_ui()

    def init_ui(self):
        self.load_model_info_from_db()

        self.setSelectionBehavior(QTableView.SelectItems)
        self.setSelectionMode(QTableView.ExtendedSelection)

        self.horizontalHeader().sectionClicked.connect(self.clear_sellect_model_row)
        self.verticalHeader().sectionClicked.connect(self.on_column_clicked)
        self.clicked.connect(self.on_item_clicked)
        "Set up a slot function for multi - selection."
        self.selectionModel().selectionChanged.connect(self.clear_sellect_model_row)
        self.setStyleSheet(ui_style_const.qlabel_stytle)

    def clear_sellect_model_row(self):
        self.sellect_model_row = None

    def on_column_clicked(self, column):
        self.sellect_model_row = column

    def on_item_clicked(self, index):
        self.sellect_model_row = index.row()

    def is_edit_model_item(self, topLeft, bottomRight, roles):
        if Qt.EditRole in roles:
            for row in range(topLeft.row(), bottomRight.row() + 1):
                index = self.model().index(row, topLeft.column())
            self.on_data_changed(index, self.is_edit_item)
        else:
            return
        
    def create_model_stytle(self):
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.Interactive)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.Interactive)
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.Interactive)

        self.setColumnWidth(1, 110)
        self.setColumnWidth(2, 90)
        self.setColumnWidth(3, 60)

    def check_model_name_is_special(self, model_name: str):
        special_char = ["\\", "/", ":", "*", "?", "\"", "<", ">", "|"]
        if model_name is not None:
            for i in special_char:
                if i in model_name:
                    return True

    def on_data_changed(self, index: QStandardItem, is_edit_item):
        if index.data() != "":
            if is_edit_item:
                old_name = None
                model_name = None
                model_description = None
                new_model_info = index.data()
                model_path = None
                if isinstance(self.model_info, tuple):
                    self.model_info = [list(row) for row in self.model_info]
                if index.column() == 0:
                    old_name = self.model_info[index.row()][index.column()]
                    model_name = new_model_info
                    model_data = {"old_name": old_name, "model_name": model_name, "model_path": self.model_info[index.row()][-1]}
                    model_path = DEFAULT_DIR + self.model_info[index.row()][-1]
                    if model_name is not None:
                        if self.check_model_name_is_special(model_name):
                            self.model().setData(index, self.model_info[index.row()][index.column()])
                            QMessageBox.warning(self, "警告", "模型名称不能包含特殊字符")
                            return
                        try:
                            new_model_path = model_path.replace(old_name, model_name)
                            rename(model_path, new_model_path)
                        except Exception as e:
                            QMessageBox.warning(self, "警告", "%s" % str(e)[:31])
                            self.logger.error("rename model [%s] error. [%s]" % (model_path, e))
                            self.model().setData(index, self.model_info[index.row()][index.column()])
                            self.is_edit_item = False
                            return
                else:
                    old_name = self.model_info[index.row()][0]
                    old_description = self.model_info[index.row()][index.column()]
                    model_description = new_model_info
                    model_data = {"old_name": old_name, "old_description" : old_description, "model_description": model_description}
                self.model_management.update_model_info_to_db(model_data)
                self.model().clear()
                self.load_model_info_from_db()
            else:
                self.is_edit_item = True
                return
        else:
            self.is_edit_item = False
            self.model().setData(index, self.model_info[index.row()][index.column()])
            return

    def del_model_in_model_info(self):
        if self.sellect_model_row is None:
            return
        model_name = self.model_info[self.sellect_model_row][0]
        code = self.model_management.delete_model_info_from_db(model_name)[0]
        if code == error_code.OK:
            self.logger.info("delete model info from db.")
        self.model_info.pop(self.sellect_model_row)
        self.list_model.removeRow(self.sellect_model_row)

    def load_model_info_from_db(self):
        self.model().setHorizontalHeaderLabels(["模型名称", "输入维度", "输出维度", "精度", "模型备注"])
        query_code, query_result = TrainingModelManagement().get_all_model_info_from_db()
        if query_code == error_code.OK:
            self.add_model_info_to_model(model_info=query_result)
            query_result = list(map(list, query_result))
            self.model_info = query_result
        self.create_model_stytle()
        
    def add_model_info_to_model(self, model_info):
        for idx, model_info in enumerate(model_info):
            model_info_items = []
            model_name, input_dim, output_dim, precision, model_description, model_path = model_info
            item_model_name = QStandardItem(model_name)
            item_model_name.setToolTip(model_name)
            item_input_dim = QStandardItem(str(input_dim))
            item_input_dim.setTextAlignment(Qt.AlignCenter)
            item_output_dim = QStandardItem(str(output_dim))
            item_output_dim.setTextAlignment(Qt.AlignCenter)
            item_precision = QStandardItem(str(precision))
            item_precision.setTextAlignment(Qt.AlignCenter)
            item_model_description = QStandardItem(model_description)
            item_model_description.setTextAlignment(Qt.AlignCenter)
            model_info_items.append(item_model_name)
            model_info_items.append(item_input_dim)
            model_info_items.append(item_output_dim)
            model_info_items.append(item_precision)
            model_info_items.append(item_model_description)
            self.model().appendRow(model_info_items)
            self.check_model_file(model_path)

    def check_model_file(self, model_path: str):
        model_path = DEFAULT_DIR + model_path
        if not path.exists(model_path):
            model_path = model_path.replace("consts/../", "")
            print(model_path)
            QMessageBox.warning(self, "警告", "模型文件不存在: %s" % model_path)

    def copy_file(self, source_path: str, model_name: str, model_type: str):
        if source_path and model_name and model_type:
            if self.check_model_name_is_special(model_name):
                QMessageBox.warning(self, "警告", "模型名称不能包含特殊字符")
                return
            if not path.isfile(source_path):
                self.logger.error("source file is empty")
                return error_code.INVALID_PATH
            target_dir = DEFAULT_DIR+ "models/"
            makedirs(target_dir, exist_ok=True)
            target_path = path.join(target_dir, model_name + "." + model_type)
            if path.exists(target_path):
                if path.samefile(source_path, target_path):
                    self.logger.info("source file is same as target file")
                    return error_code.OK
            try:
                copy2(source_path, target_path)
                self.logger.info("copy file success")
                if path.isfile(target_path):
                    return error_code.OK
                else:
                    QMessageBox.warning(self, "警告", "模型文件复制失败")
                    self.logger.error("copy file error")
                    return error_code.COPY_FILE_ERROR
            except Exception as e:
                QMessageBox.warning(self, "警告", "%s" % e)
                self.logger.error("copy file error: %s" % e)
        else:
            self.logger.error("model_name or source_path or model_type is empty")
            return error_code.COPY_FILE_ERROR

    def register_model_info_to_db(self, model_name: str, model_config: dict, model_type: str):
        result = any(model_name in row for row in self.model_info)
        if result:
            self.logger.error("model_name is exist")
            return True
        if model_name and model_config and model_type:
            code, code_str = self.model_management.register_new_model_info_to_db(model_name,
                                                                                input_dim=model_config.get("input_dim"), 
                                                                                output_dim=model_config.get("output_dim"),
                                                                                model_description=model_config.get("model_description", "No description"),
                                                                                model_type = model_type)
            if code == error_code.INVALID_INSERT:
                self.logger.error(code_str)
                return False
            elif code == error_code.INVALID_DATA_LOADING:
                self.logger.error(code_str)
                return False
            return True
        else:
            self.logger.error("model_name or model_config or model_type is empty")
            return False
        
    def get_model_config(self, model_name: str, action_type:str = None):
        dim_dict = {}
        if action_type == "new":
            dim_dict["input_left"] = self.model_info[self.sellect_model_row][1].split(" x ")[0]
            dim_dict["input_right"] = self.model_info[self.sellect_model_row][1].split(" x ")[1]
            dim_dict["output_dim"] = self.model_info[self.sellect_model_row][2]
            model_obj_data = SetModelConfig(model_name=model_name, dim=dim_dict)
            model_obj_data.model_input_dim_box.setEnabled(False)
            model_obj_data.model_output_dim_box.setEnabled(False)
            model_config = model_obj_data.exec()
        else:
            model_obj_data = SetModelConfig(model_name=model_name)
            model_config = model_obj_data.exec()
        return model_config
        
    def check_model_config(self, model_name: str = None, model_type: str = "keras", recursion_num: int = 0, action_type:str = None):
        if recursion_num > 2:
            self.logger.error("get model config error")
            return {}
        model_config = self.get_model_config(model_name = model_name, action_type = action_type)
        target_dir = DEFAULT_DIR + "models/"     
        makedirs(target_dir, exist_ok = True)
        if model_config.get("model_name", False):
            target_path = path.join(target_dir, model_config["model_name"] + "." + model_type)
            if path.exists(target_path):
                result = any(model_config["model_name"] in row for row in self.model_info)
                if result is False:
                    return model_config
                QMessageBox.warning(self, "警告", "模型已存在, 请请重新设置!")
                recursion_num = recursion_num + 1
                model_config = self.check_model_config(recursion_num = recursion_num, action_type = action_type)
            return model_config
    
    def updata_model_info(self, model_path: str, model_name: str = None, model_type: str = "keras", action_type: str = None):
        if not model_path:
            return
        model_config = self.check_model_config(model_name, model_type, action_type = action_type)
        is_success_register = False
        if model_config:
            code= self.copy_file(model_path, model_config["model_name"], model_type)
            if code == error_code.OK:
                is_success_register = self.register_model_info_to_db(model_config["model_name"], model_config, model_type)
                if is_success_register:
                    self.logger.info("set new model info success")
                    self.model().clear()
                    self.load_model_info_from_db()
                else:
                    QMessageBox.warning(self, "警告", "模型添加到数据库失败!")
                    self.logger.error("Add new model info to database error")
            else:
                self.logger.error("copy file error")
                return
        else:
            return

    def set_new_model_info(self):
        if self.sellect_model_row is None:
            QMessageBox.warning(self, "警告", "请选择模型")
            return
        item = self.model().item(self.sellect_model_row, 0)
        if item:
            model_path = DEFAULT_DIR + self.model_info[item.row()][-1]
            if path.isfile(model_path):
                model_type = model_path.split(".")[-1]
                self.updata_model_info(model_path = model_path, model_type = model_type, action_type="new")
                self.sellect_model_row = None

    def register_model_info(self):
        home_directory = path.expanduser("~")
        model_path = QFileDialog.getOpenFileName(self, 
                                                 "选择模型文件",
                                                 home_directory, 
                                                 "KERAS Files (*.keras);;"
                                                 "PKL Files (*.pkl);;"
                                                 "All Files (*)")[0]
        if path.isfile(model_path):
            model_name = path.basename(model_path)
            model_name = model_name.split(".")[0]
            model_type = model_path.split(".")[-1]
        else:
            return
        self.updata_model_info(model_path, model_name, model_type)
        self.sellect_model_row = None
        
    def del_model_info(self):
        if self.sellect_model_row is None:
            QMessageBox.warning(self, "警告", "请选择模型")
            return
        path_index = len(self.model_info[0]) - 1
        model_path = DEFAULT_DIR + self.model_info[self.sellect_model_row][path_index]
        if path.isfile(model_path):
            try:
                remove(model_path)
            except Exception as e:
                error_data = str(e).replace("consts/../", "")
                QMessageBox.warning(self, "警告", "%s" % error_data)
                self.logger.error(e)
        if not path.exists(model_path):
            self.del_model_in_model_info()
            self.sellect_model_row = None
            self.logger.info("delete model info success")


class SetModelConfig(QDialog):
    def __init__(self, parent=None, model_name: str = None, dim: dict = None):
        super().__init__(parent) 
        self.config = {"model_name": model_name, "output_dim": None,}
        self.model_name = model_name
        self.dim = dim
        self.input_dim_left: str = None
        self.input_dim_right: str = None
        self.clicked_ok_close = False

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("设置模型信息")
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        if self.dim:
            self.input_dim_left = self.dim["input_left"]
            self.input_dim_right = self.dim["input_right"]
            self.config["output_dim"] = self.dim["output_dim"]
        else:
            self.input_dim_left = "176400"
            self.input_dim_right = "1"
            self.config["output_dim"] = "1"
        model_name_box = self.create_model_name_box()
        self.model_input_dim_box = self.create_inpot_dim_box()
        self.model_output_dim_box = self.create_output_dim_box()
        model_description_box = self.create_model_description_box()
        btn_layout = self.create_btn_layout()

        layout = QVBoxLayout()
        layout.addWidget(model_name_box)
        layout.addWidget(self.model_input_dim_box)
        layout.addWidget(self.model_output_dim_box)
        layout.addWidget(model_description_box)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

        self.setStyleSheet(ui_style_const.qpushbutton_stytle + 
                           ui_style_const.qlineedit_stytle + 
                           ui_style_const.qgroupbox_stytle +
                           ui_style_const.qlabel_stytle)

    def create_model_name_box(self):
        model_name_box = QGroupBox()
        model_name_label = QLabel("模型名称:")
        model_name_edit = QLineEdit()
        model_name_edit.setText(self.model_name)
        model_name_edit.setPlaceholderText("请输入模型名称")
        model_name_edit.editingFinished.connect(self.on_model_name_edit_finished)

        model_name_layout = QHBoxLayout()
        model_name_layout.addWidget(model_name_label)
        model_name_layout.addStretch()
        model_name_layout.addWidget(model_name_edit)

        model_name_box.setLayout(model_name_layout)
        return model_name_box
    
    def create_inpot_dim_box(self):
        inpot_dim_box = QGroupBox()
        inpot_dim_label = QLabel("输入维度:")
        inpot_dim_edit_left = QLineEdit()
        label = QLabel("x")
        input_dim_edit_right = QLineEdit()

        inpot_dim_edit_left.setText(str(self.input_dim_left))
        input_dim_edit_right.setText(str(self.input_dim_right))

        inpot_dim_edit_left.editingFinished.connect(self.on_inpot_dim_edit_left_finished)
        input_dim_edit_right.editingFinished.connect(self.on_inpot_dim_edit_right_finished)

        inpot_dim_layout = QHBoxLayout()
        inpot_dim_layout.addWidget(inpot_dim_label)
        inpot_dim_layout.addWidget(inpot_dim_edit_left)
        inpot_dim_layout.addWidget(label)
        inpot_dim_layout.addWidget(input_dim_edit_right)
        inpot_dim_box.setLayout(inpot_dim_layout)
        return inpot_dim_box

    def create_output_dim_box(self):
        output_dim_box = QGroupBox()
        output_dim_label = QLabel("输出维度:")
        output_dim_edit = QLineEdit()
        output_dim_edit.setText(str(self.config["output_dim"]))
        output_dim_edit.editingFinished.connect(self.on_output_dim_edit_finished)

        output_dim_layout = QHBoxLayout()
        output_dim_layout.addWidget(output_dim_label)
        output_dim_layout.addStretch()
        output_dim_layout.addWidget(output_dim_edit)
        output_dim_box.setLayout(output_dim_layout)
        return output_dim_box
    
    def create_model_description_box(self):
        model_description_box = QGroupBox()
        model_description_label = QLabel("模型描述:")
        model_description_edit = QLineEdit()
        model_description_edit.setPlaceholderText("No description")
        model_description_edit.editingFinished.connect(self.on_model_description_edit_finished)
        model_description_layout = QHBoxLayout()
        model_description_layout.addWidget(model_description_label)
        model_description_layout.addStretch()
        model_description_layout.addWidget(model_description_edit)
        model_description_box.setLayout(model_description_layout)
        return model_description_box
    
    def create_btn_layout(self):
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(self.on_close)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        return btn_layout
    
    def on_model_name_edit_finished(self):
        self.config["model_name"] = self.sender().text()

    def on_inpot_dim_edit_left_finished(self):
        self.input_dim_left = self.sender().text()

    def on_inpot_dim_edit_right_finished(self):
        self.input_dim_right = self.sender().text()

    def on_output_dim_edit_finished(self):
        self.config["output_dim"] = self.sender().text()

    def on_model_description_edit_finished(self):
        self.config["model_description"] = self.sender().text()

    def on_close(self):
        if self.config.get("model_name", None) and self.input_dim_left and self.input_dim_right and self.config.get("output_dim", None):
            self.clicked_ok_close = True
            self.close()
        else:
            QMessageBox.warning(self, "警告", "请配置信息！")
            return
    
    def exec(self):
        super().exec()
        if self.clicked_ok_close:
            self.config["input_dim"] = self.input_dim_left + " x " + self.input_dim_right
            return self.config
        else:
            return {}


class CustomStandardItemModel(QStandardItemModel):
    def __init__(self, rows, columns, editable_column: list, parent=None):
        super().__init__(rows, columns, parent)
        self.editable_column = editable_column

    def flags(self, index):
        if index.isValid():
            if index.column() in self.editable_column:
                return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable
            else:
                return Qt.ItemIsEnabled | Qt.ItemIsSelectable
        return super().flags(index)
    
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ModelInfoList(LogManager.set_log_handler("core"))
    # window = SetModelConfig()
    window.show()
    a = app.exec()
    # a = window.exec()
    print(a)
