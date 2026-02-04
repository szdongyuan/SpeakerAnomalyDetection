from re import match
from shutil import copy2
import os
import sys

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QStandardItemModel, QIcon
from PyQt5.QtWidgets import QApplication, QFileDialog, QDialog, QGroupBox, QLabel, QLineEdit, QMessageBox
from PyQt5.QtWidgets import QVBoxLayout, QPushButton, QHBoxLayout, QComboBox, QSizePolicy

from base.file_ops import FileOps
from base.training_model_management import TrainingModelManagement
from base.log_manager import LogManager
from consts import error_code, ui_style_const
from consts.running_consts import DEFAULT_DIR
from machine_learning.model_builder import build_and_save_model_from_config
from ui.custom_ui_widget.custom_table_widget import DataManageDialog


def setdata(model, index, value, role=Qt.EditRole):
    if index.column() == 0:
        if not match(r"^[A-Za-z0-9_]*$", str(value)):
            QMessageBox.warning(None, "警告", "模型名称只能由大小写字母、数字和下划线组成!")
            return False
    return QStandardItemModel.setData(model, index, value, role)


class ModelInfoList(DataManageDialog):
    def __init__(self, logger: LogManager):
        super().__init__()
        self.logger = logger
        self.init_ui_layout(0, 6, [1, 5])
        self.model().setData = lambda index, value, role=Qt.EditRole: setdata(self.model(), index, value, role)
        self.set_view_checked_changed(self.on_row_checkbox_toggled)

        self.select_model_row = None
        self.model_management = TrainingModelManagement()
        self.model_info = list()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("模型管理")
        self.resize(QSize(650, 300))
        self.create_btn_layout()
        self.set_checkable_of_column([0])
        self.set_h_header(["", "模型名称", "输入维度", "输出维度", "精度", "模型备注"])
        self.verticalHeader().setVisible(False)
        self.layout().setContentsMargins(13, 20, 13, 13)
        self.set_column_alignment(3, Qt.AlignCenter)
        self.select_data_num.hide()
        self.load_model_info_from_db()

    def create_btn_layout(self):
        ok_btn = QPushButton(" 确  定 ")
        set_new_model_btn = QPushButton("新建模型")
        register_model_btn = QPushButton("注册模型")
        del_model_btn = QPushButton("删除模型")

        del_model_btn.clicked.connect(self.del_model_info)
        register_model_btn.clicked.connect(self.register_model_info)
        set_new_model_btn.clicked.connect(self.set_new_model_info)

        ok_btn.clicked.connect(self.close)

        self.bottom_layout.addWidget(set_new_model_btn)
        self.bottom_layout.addWidget(register_model_btn)
        self.bottom_layout.addWidget(del_model_btn)
        self.bottom_layout.addStretch()
        self.bottom_layout.addWidget(ok_btn)

    def del_model_info(self):
        if self.select_model_row is None:
            QMessageBox.warning(self, "警告", "请选择模型")
            return
        path_index = len(self.model_info[0]) - 1
        model_path = DEFAULT_DIR + self.model_info[self.select_model_row][path_index]
        if os.path.isfile(model_path):
            try:
                os.remove(model_path)
            except Exception as e:
                error_data = str(e).replace("consts/../", "")
                QMessageBox.warning(self, "警告", "%s" % error_data)
                self.logger.error(e)
        if not os.path.exists(model_path):
            self.del_model_in_model_info()
            self.select_model_row = None
            self.logger.info("delete model info success")

    def del_model_in_model_info(self):
        if self.select_model_row is None:
            return
        model_name = self.model_info[self.select_model_row][0]
        code = self.model_management.delete_model_info_from_db(model_name)[0]
        if code == error_code.OK:
            self.logger.info("delete model info from db.")
        self.model_info.pop(self.select_model_row)
        self.model().removeRow(self.select_model_row)

    def get_model_config(self, model_name: str, action_type: str = None):
        dim_dict = None
        is_new = (action_type == "new")

        enable_widgets = (action_type == "new" or action_type == "register")

        if not enable_widgets:
            if self.select_model_row is None:
                QMessageBox.warning(self, "警告", "请选择模型")
                return None

            dim_dict = {}
            config_path_from_db = self.model_info[self.select_model_row][-2]
            dim_dict["input_left"] = self.model_info[self.select_model_row][1].split(" x ")[0]
            dim_dict["input_right"] = self.model_info[self.select_model_row][1].split(" x ")[1]
            dim_dict["output_dim"] = self.model_info[self.select_model_row][2]
            dim_dict["config_path"] = os.path.basename(config_path_from_db).split(".yml")[0]

        model_obj_data = SetModelConfig(
            model_info=self.model_info,
            model_name=model_name,
            dim=dim_dict,
            is_new_model=is_new
        )

        model_obj_data.model_input_dim_box.setEnabled(enable_widgets)
        model_obj_data.model_output_dim_box.setEnabled(enable_widgets)
        model_obj_data.model_config_box.setEnabled(enable_widgets)

        model_config = model_obj_data.exec()

        if action_type == "register" and model_config:
            config_full_path = model_config.pop("config_path_full", None)
            if config_full_path:
                model_config["config_path"] = FileOps.get_relative_path(config_full_path, DEFAULT_DIR)
            else:
                QMessageBox.warning(self, "警告", "未获取到模型配置文件，注册失败。")
                return None

        return model_config

    def register_model_info(self):
        home_directory = os.path.expanduser("~")
        model_path = QFileDialog.getOpenFileName(
            self, "选择模型文件", home_directory, "KERAS Files (*.keras)"
        )[0]
        if os.path.isfile(model_path):
            model_name = os.path.basename(model_path)
            model_name = model_name.split(".")[0]
            model_type = model_path.split(".")[-1]
        else:
            return

        self.update_model_info(model_path, model_name, model_type, action_type="register")
        self.select_model_row = None

    def update_model_info(
            self, model_path: str, model_name: str = None, model_type: str = "keras", action_type: str = None
    ):
        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "警告", f"模型路径无效或不存在：{model_path}")
            return

        model_config = self.get_model_config(model_name=model_name, action_type=action_type)
        if not model_config:
            return

        new_model_name = model_config["model_name"]
        relative_config_path = model_config["config_path"]
        copy_code = self.copy_file(
            source_path=model_path,
            model_name=new_model_name,
            model_type=model_type
        )

        if copy_code != error_code.OK:
            self.logger.error(f"文件复制失败，无法注册 {new_model_name}")
            return

        relative_model_path = os.path.join("models", f"{new_model_name}.{model_type}").replace("\\", "/")

        is_success_register = self.register_model_info_to_db(
            model_name=new_model_name,
            model_config={
                **model_config,
                "model_path": relative_model_path,
                "config_path": relative_config_path
            },
            model_type=model_type
        )

        if is_success_register:
            QMessageBox.information(self, "成功", f"模型注册成功！")
            self.load_model_info_from_db()

    def register_model_info_to_db(self, model_name: str, model_config: dict, model_type: str):
        if not (model_name and model_config and model_type):
            self.logger.error("model_name or model_config or model_type is empty")
            return False

        config_path = model_config.get("config_path")
        input_dim = model_config.get("input_dim")
        output_dim = model_config.get("output_dim")
        model_description = model_config.get("model_description", "No description")
        model_path = model_config.get("model_path")

        code, code_str = self.model_management.register_new_model_info_to_db(
            model_name=model_name,
            config_path=config_path,
            input_dim=input_dim,
            output_dim=output_dim,
            model_description=model_description,
            model_type=model_type,
            model_path=model_path
        )
        if code == error_code.OK:
            return True
        else:
            self.logger.error(f"数据库插入失败: {code_str}")
            return False

    def set_new_model_info(self):
        model_config_dialog = SetModelConfig(
            model_info=self.model_info,
            is_new_model=True
        )
        model_config_result = model_config_dialog.exec()
        if not model_config_result:
            return

        model_config = model_config_result
        yml_config_path = model_config.get("yml_config_path")
        model_name = model_config.get("model_name")
        save_path = model_config.get("save_path")

        if not yml_config_path:
            QMessageBox.warning(self, "警告", "请先选择YML配置文件！")
            return
        try:
            new_model = build_and_save_model_from_config(
                config_path=yml_config_path,
                compile_model=True
            )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型构建失败：{str(e)}")
            self.logger.error(f"Model build failed: {e}")
            return
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            new_model.save(save_path)
            if not os.path.exists(save_path):
                raise FileNotFoundError(f"保存后文件不存在：{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型保存失败：{str(e)}")
            self.logger.error(f"Model save failed: {e}")
            return

        relative_model_path = FileOps.get_relative_path(save_path, DEFAULT_DIR)
        relative_config_path = FileOps.get_relative_path(yml_config_path, DEFAULT_DIR)

        db_config = {
            **model_config,
            "model_path": relative_model_path,
            "config_path": relative_config_path
        }
        is_success_register = self.register_model_info_to_db(
            model_name=model_name,
            model_config=db_config,
            model_type="keras"
        )
        if is_success_register:
            QMessageBox.information(self, "成功", "模型新建成功！")
            self.load_model_info_from_db()

    def copy_file(self, source_path: str, model_name: str, model_type: str):
        if source_path and model_name and model_type:
            if not os.path.isfile(source_path):
                self.logger.error("source file is empty")
                return error_code.INVALID_PATH
            target_dir = DEFAULT_DIR + "models/"
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, model_name + "." + model_type)
            if os.path.exists(target_path):
                if os.path.samefile(source_path, target_path):
                    return error_code.OK
                if self.override_model_file_part():
                    self.logger.info("source file is same as target file, Deleting the target file...")
                    os.remove(target_path)
                else:
                    return error_code.OK
            try:
                copy2(source_path, target_path)
                self.logger.info("copy file success")
                if os.path.isfile(target_path):
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

    def override_model_file_part(self):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("文件存在")
        msg_box.setText("目标文件已存在，是否覆盖？")

        msg_box.addButton("是", QMessageBox.YesRole)
        no_btn = msg_box.addButton("否", QMessageBox.NoRole)
        msg_box.exec_()

        if msg_box.clickedButton() == no_btn:
            self.logger.warning("user cancel override the file")
            return False
        else:
            return True

    def on_row_checkbox_toggled(self, item, is_checked):
        if is_checked:
            if self.select_model_row is not None:
                self.model().item(self.select_model_row, 0).setCheckState(Qt.Unchecked)
            self.select_model_row = item.row()
        else:
            self.select_model_row = None

    def load_model_info_from_db(self):
        query_code, query_result = TrainingModelManagement().get_all_model_info_from_db()
        if query_code == error_code.OK:
            self.setRowCount(0)
            self.add_model_info_to_model(model_info=query_result)
            query_result = list(map(list, query_result))
            self.model_info = query_result

    def add_model_info_to_model(self, model_info):
        for idx, model_info in enumerate(model_info):
            model_name, input_dim, output_dim, precision, model_description, _, model_path = model_info
            row_item_list = [None, model_name, input_dim, output_dim, precision, model_description]
            self.add_row_data(row_item_list)
            self.check_model_file(model_path)

    def check_model_file(self, model_path: str):
        model_path = DEFAULT_DIR + model_path
        if not os.path.exists(model_path):
            model_path = model_path.replace("consts/../", "")
            QMessageBox.warning(self, "警告", "模型文件不存在: %s" % model_path)


class SetModelConfig(QDialog):
    def __init__(self, model_info: list, model_name: str = None, dim: dict = None, is_new_model: bool = False):
        super().__init__()
        self.config = {
            "model_name": model_name,
            "output_dim": None,
            "save_path": "",
            "yml_config_path": None
        }
        self.model_name = model_name
        self.dim = dim
        self.input_dim_left: str = None
        self.input_dim_right: str = None
        self.clicked_ok_close = False
        self.model_info = model_info
        self.is_new_model = is_new_model

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("设置模型信息")
        self.setMinimumWidth(600)
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

        self.yml_config_box = self.create_yml_config_box() if self.is_new_model else None

        model_name_box = self.create_model_name_box()
        self.model_input_dim_box = self.create_input_dim_box()
        self.model_output_dim_box = self.create_output_dim_box()
        self.model_config_box = self.create_model_config_box()
        model_description_box = self.create_model_description_box()

        self.save_path_box = self.create_save_path_box() if self.is_new_model else None

        btn_layout = self.create_btn_layout()

        layout = QVBoxLayout()

        layout.addWidget(model_name_box)
        layout.addWidget(self.model_input_dim_box)
        layout.addWidget(self.model_output_dim_box)
        layout.addWidget(model_description_box)

        if self.is_new_model:
            self.model_config_box.hide()
            if self.save_path_box:
                layout.addWidget(self.save_path_box)
            if self.yml_config_box:
                layout.addWidget(self.yml_config_box)
        else:
            layout.addWidget(self.model_config_box)

        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.setStyleSheet(
            ui_style_const.qpushbutton_style
            + ui_style_const.qlineedit_style
            + ui_style_const.qgroupbox_style
            + ui_style_const.qlabel_style
            + ui_style_const.qcombobox_style
        )

    def check_model_name(self, model_name: str):
        input_str = model_name
        reg = r"^[A-Za-z0-9_]*$"
        if not match(reg, input_str):
            QMessageBox.warning(self, "警告", "模型名称只能由大小写字母、数字和下划线组成!")
            return False
        result = False
        if self.model_info:
            result = any(model_name == row[0] for row in self.model_info)
        if result:
            QMessageBox.warning(self, "警告", "模型名称已存在！")
            return False
        return True

    def create_model_name_box(self):
        model_name_box = QGroupBox("基本信息")
        model_name_label = QLabel("模型名称:")
        self.model_name_edit = QLineEdit()
        self.model_name_edit.setText(self.model_name)
        self.model_name_edit.setPlaceholderText("请输入模型名称")
        self.model_name_edit.editingFinished.connect(self.on_model_name_edit_finished)

        model_name_layout = QHBoxLayout()
        model_name_layout.addWidget(model_name_label)
        model_name_layout.addStretch()
        model_name_layout.addWidget(self.model_name_edit)

        model_name_box.setLayout(model_name_layout)
        return model_name_box

    def create_input_dim_box(self):
        input_dim_box = QGroupBox()
        input_dim_label = QLabel("输入维度:")
        input_dim_edit_left = QLineEdit()
        label = QLabel("x")
        input_dim_edit_right = QLineEdit()

        input_dim_edit_left.setText(str(self.input_dim_left))
        input_dim_edit_right.setText(str(self.input_dim_right))

        input_dim_edit_left.editingFinished.connect(self.on_input_dim_edit_left_finished)
        input_dim_edit_right.editingFinished.connect(self.on_input_dim_edit_right_finished)

        input_dim_layout = QHBoxLayout()
        input_dim_layout.addWidget(input_dim_label)
        input_dim_layout.addWidget(input_dim_edit_left)
        input_dim_layout.addWidget(label)
        input_dim_layout.addWidget(input_dim_edit_right)
        input_dim_box.setLayout(input_dim_layout)
        return input_dim_box

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

    def create_model_config_box(self):
        model_config_box = QGroupBox()
        model_config_label = QLabel("模型配置:")
        self.model_config_combobox = QComboBox()
        size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.model_config_combobox.setSizePolicy(size_policy)
        self.model_config_combobox.currentTextChanged.connect(self.on_combobox_clicked)
        yaml_files_data = self.get_yaml_files()
        for name, full_path in yaml_files_data:
            self.model_config_combobox.addItem(name, userData=full_path)
        if not self.is_new_model and self.dim:
            self.model_config_combobox.setCurrentText(self.dim["config_path"])

        layout = QHBoxLayout()
        layout.addWidget(model_config_label)
        layout.addStretch()
        layout.addWidget(self.model_config_combobox)
        model_config_box.setLayout(layout)

        return model_config_box

    def on_combobox_clicked(self):
        if not self.is_new_model:
            full_path = self.sender().currentData()
            self.config["config_path_full"] = full_path

    def update_save_path_from_model_name(self):
        if self.is_new_model and self.save_path_edit and self.config["model_name"]:
            save_dir = os.path.dirname(self.save_path_edit.text())
            if not os.path.isdir(save_dir):
                save_dir = os.path.join(DEFAULT_DIR, "models/")
                os.makedirs(save_dir, exist_ok=True)
            self.save_path_edit.setText(os.path.join(save_dir, f"{self.config['model_name']}.keras"))

    def create_model_description_box(self):
        model_description_box = QGroupBox()
        model_description_label = QLabel("模型备注:")
        model_description_edit = QLineEdit()
        model_description_edit.setPlaceholderText("No description")
        model_description_edit.editingFinished.connect(self.on_model_description_edit_finished)
        model_description_layout = QHBoxLayout()
        model_description_layout.addWidget(model_description_label)
        model_description_layout.addStretch()
        model_description_layout.addWidget(model_description_edit)
        model_description_box.setLayout(model_description_layout)
        return model_description_box

    def create_yml_config_box(self):
        yml_box = QGroupBox("配置文件")
        yml_label = QLabel("YML文件路径:")
        self.yml_path_edit = QLineEdit()
        self.yml_path_edit.setPlaceholderText("请选择一个YML配置文件")
        self.yml_path_edit.setReadOnly(True)

        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self.on_browse_yml)

        yml_layout = QHBoxLayout()
        yml_layout.addWidget(yml_label)
        yml_layout.addWidget(self.yml_path_edit, 1)
        yml_layout.addWidget(browse_btn)
        yml_box.setLayout(yml_layout)
        return yml_box

    def on_browse_yml(self):
        default_yml_dir = os.path.join(DEFAULT_DIR, "configs/ai_model_config/")
        os.makedirs(default_yml_dir, exist_ok=True)

        yml_config_path = QFileDialog.getOpenFileName(
            self, "选择模型YML配置文件", default_yml_dir,
            "YML Files (*.yml)"
        )[0]
        if not yml_config_path:
            return
        self.yml_path_edit.setText(yml_config_path)
        self.config["yml_config_path"] = yml_config_path

    def create_save_path_box(self):
        save_path_box = QGroupBox("模型保存路径")
        save_path_label = QLabel("保存路径:")
        self.save_path_edit = QLineEdit()
        self.save_path_edit.setPlaceholderText("请选择模型保存路径")
        default_save_dir = os.path.join(DEFAULT_DIR, "models/")
        os.makedirs(default_save_dir, exist_ok=True)
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self.on_browse_save_path)

        save_path_layout = QHBoxLayout()
        save_path_layout.addWidget(save_path_label)
        save_path_layout.addWidget(self.save_path_edit, 1)
        save_path_layout.addWidget(browse_btn)
        save_path_box.setLayout(save_path_layout)
        return save_path_box

    def on_browse_save_path(self):
        initial_dir = os.path.dirname(self.save_path_edit.text()) if self.save_path_edit.text() else DEFAULT_DIR
        save_dir = QFileDialog.getExistingDirectory(self, "选择保存目录", initial_dir)
        if save_dir:
            if self.config["model_name"]:
                full_path = os.path.join(save_dir, f"{self.config['model_name']}.keras")
            else:
                full_path = save_dir
            self.save_path_edit.setText(full_path)
            self.config["save_path"] = full_path

    def create_btn_layout(self):
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(self.on_close)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        return btn_layout

    def get_yaml_files(self):
        directory = DEFAULT_DIR + "/configs/ai_model_config"
        yaml_files_data = []
        for filename in os.listdir(directory):
            if filename.endswith(".yml"):
                file_name_only = os.path.basename(filename).split(".yml")[0]
                full_path = os.path.join(directory, filename)
                yaml_files_data.append((file_name_only, full_path))
        return yaml_files_data

    def on_model_name_edit_finished(self):
        self.config["model_name"] = self.sender().text()
        self.update_save_path_from_model_name()

    def on_input_dim_edit_left_finished(self):
        self.input_dim_left = self.sender().text()

    def on_input_dim_edit_right_finished(self):
        self.input_dim_right = self.sender().text()

    def on_output_dim_edit_finished(self):
        self.config["output_dim"] = self.sender().text()

    def on_model_description_edit_finished(self):
        self.config["model_description"] = self.sender().text()

    def on_close(self):
        if not self._validate_common_fields():
            return

        if self.is_new_model:
            if not self._validate_new_model_fields():
                return
        else:
            if not self._validate_register_model_fields():
                return

        self.clicked_ok_close = True
        self.close()

    def _validate_common_fields(self) -> bool:
        if not self.config.get("model_name"):
            QMessageBox.warning(self, "警告", "请输入模型名称！")
            return False

        if not (self.input_dim_left and self.input_dim_right and self.config.get("output_dim")):
            QMessageBox.warning(self, "警告", "请完善所有维度信息！")
            return False

        return True

    def _validate_new_model_fields(self) -> bool:
        self.config["yml_config_path"] = self.yml_path_edit.text()
        self.config["save_path"] = self.save_path_edit.text()

        if not self.config.get("yml_config_path"):
            QMessageBox.warning(self, "警告", "请选择一个YML配置文件!")
            return False

        if not self.config.get("save_path"):
            QMessageBox.warning(self, "警告", "请选择模型保存路径！")
            return False

        if not self.config["save_path"].endswith(".keras"):
            QMessageBox.warning(self, "警告", "模型保存路径必须以.keras结尾！")
            return False

        return True

    def _validate_register_model_fields(self) -> bool:
        current_config_path = self.model_config_combobox.currentData()

        if not current_config_path:
            QMessageBox.warning(self, "警告", "请选择一个模型配置！")
            return False

        self.config["config_path_full"] = current_config_path
        return True

    def exec(self) -> dict:
        super().exec()
        if self.clicked_ok_close:
            self.config["input_dim"] = self.input_dim_left + " x " + self.input_dim_right
            return self.config
        else:
            return {}

    def closeEvent(self, close_event):
        if self.clicked_ok_close:
            if not self.check_model_name(self.config["model_name"]):
                close_event.ignore()
                self.clicked_ok_close = False
            else:
                close_event.accept()
        else:
            close_event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModelInfoList(LogManager.set_log_handler("core"))
    # window = SetModelConfig()
    window.show()
    a = app.exec()
    # a = window.exec()
    print(a)
