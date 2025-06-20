import sys

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QHBoxLayout, QHeaderView, QPushButton, QTableView, QMessageBox

from base.log_manager import LogManager
from base.stimulus_signal_management import StimulusSignalManagement
from consts import error_code, ui_style_const
from consts.running_consts import DEFAULT_DIR


class LoadStimulusDialog(QDialog):

    def __init__(self, logger: LogManager):
        super().__init__()
        self.logger = logger
        self.stimulus_config_view = StimulusConfigView(self.logger)
        self.is_clicked_ok = False

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("加载刺激信号")
        self.setWindowIcon(QIcon(""))
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setMinimumSize(QSize(530, 200))
        self.resize(957, 300)
        self.setup_main_layout()

        self.setStyleSheet(ui_style_const.qpushbutton_stytle +
                           ui_style_const.qtableview_stytle +
                           ui_style_const.qlabel_stytle)

    def setup_main_layout(self):
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.stimulus_config_view)

        btn_layout = self.create_stimulus_btn_layout()
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

    def create_stimulus_btn_layout(self):
        del_config_btn = QPushButton("删除配置")
        cancel_btn = QPushButton("取消")
        ok_btn = QPushButton("确定")

        del_config_btn.clicked.connect(self.stimulus_config_view.del_config)
        ok_btn.clicked.connect(self.ok_btn_clicked)
        cancel_btn.clicked.connect(self.close)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(del_config_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(ok_btn)

        return btn_layout
    
    def ok_btn_clicked(self):
        if self.stimulus_config_view.get_select_stimulus_row() is None:
            QMessageBox.warning(self, "提示", "请选择要加载的刺激信号配置")
            return
        self.is_clicked_ok = True
        self.close()

    def exec(self):
        super().exec_()
        if self.is_clicked_ok:
            return self.stimulus_config_view.get_selected_stimulus_config()
        else:
            return None
    

class StimulusConfigView(QTableView):
    # Define stimulus signal types
    STIMULUS_DICT = {
        "chirp":"啁啾",
        "step": "步进",
        "noise": "噪音",
    }
    # Mapping for stimulus signal chinese names
    STIMULUS_DICT_2 = {
        "log": "对数",
        "linear": "线性",
        "mirror_log": "对数镜像",
        "mirror_linear": "线性镜像",
        "white_noise": "白噪音",
        "pink_noise": "粉噪音",
    }

    def __init__(self, logger: LogManager):
        super().__init__()
        self.logger = logger
        self.stimulus_config_model = CustomStandardItemModel(0, 9, [0])
        self.select_stimulus_row = None
        self.is_edit_item = True
        self.loaded_stimulus = self.load_stimulus_config_from_db()

        self.setModel(self.stimulus_config_model)
        self.insert_stimulus_config_to_table(self.loaded_stimulus, self.model())
        self.model().dataChanged.connect(self.is_edit_model_item)

        self.init_ui()

    def init_ui(self):
        self.setSelectionBehavior(QTableView.SelectItems)
        self.setSelectionMode(QTableView.ExtendedSelection)

        self.horizontalHeader().sectionClicked.connect(self.clear_select_stumulus_row)
        self.verticalHeader().sectionClicked.connect(self.on_column_clicked)
        self.clicked.connect(self.on_item_clicked)
        # Set up a slot function for multi - selection.
        self.selectionModel().selectionChanged.connect(self.clear_select_stumulus_row)
        self.create_model_stytle()
        self.setStyleSheet(ui_style_const.qlabel_stytle)

    def create_model_stytle(self):
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)

    def clear_select_stumulus_row(self):
        self.select_stimulus_row = None

    def on_column_clicked(self, column):
        self.select_stimulus_row = column

    def on_item_clicked(self, index):
        self.select_stimulus_row = index.row()

    def del_config(self):
        if self.select_stimulus_row is not None:
            stimulus_name = self.loaded_stimulus[self.select_stimulus_row].get("stimulus_name")
            code, msg = StimulusSignalManagement().delete_stimulus_info_from_db(stimulus_name)
            if "FOREIGN KEY" in msg:
                QMessageBox.warning(self, "提示", "请先删除该激励信号下的所有数据")
                return
            if code == error_code.OK:
                self.logger.info("delete stimulus config %s success" % stimulus_name)
            self.loaded_stimulus.pop(self.select_stimulus_row)
            self.model().removeRow(self.select_stimulus_row)

    def is_edit_model_item(self, topLeft, bottomRight, roles):
        if Qt.EditRole in roles:
            for row in range(topLeft.row(), bottomRight.row() + 1):
                index = self.model().index(row, topLeft.column())
            self.on_data_changed(index, self.is_edit_item)
        else:
            return
        
    def on_data_changed(self, index: QStandardItem, is_edit_item):
        if index.data() != "":
            if is_edit_item:
                stimulus_id = self.loaded_stimulus[index.row()].get("stimulus_id")
                new_name = index.data()
                update_info = {"stimulus_id": stimulus_id, "new_name": new_name}
                code, msg = StimulusSignalManagement().update_stimulus_info_to_db(update_info)
                self.loaded_stimulus[index.row()]["stimulus_name"] = new_name
                if code == error_code.OK:
                    self.logger.info(msg)
                else:
                    self.logger.error(msg)
            else:
                is_edit_item = True
                return
        else:
            is_edit_item = False
            self.model().setData(index, self.loaded_stimulus[index.row()].get("stimulus_name"))
            return

    def insert_stimulus_config_to_table(self, stimulus_info: dict, model: QStandardItemModel):
        self.model().setHorizontalHeaderLabels(["配置名称", "信号模式", "信号类型", "重复次数", "起始频率", "截止频率", "采样频率", "信号时长", "步进数量"])
        for idx, stimulus in enumerate(stimulus_info):
            config_list = self.set_table_row_data(stimulus)
            model.appendRow(config_list)

    def set_table_row_data(self, stimulus: dict):
        config_list = list()
        stimulus_name = QStandardItem(stimulus.get("stimulus_name"))
        config_method = self.set_stimulus_method(stimulus.get("stimulus_method"))
        config_type = self.set_stimulus_type(stimulus.get("stimulus_type"))
        config_repeat_times = QStandardItem(str(stimulus.get("repeat_times")))
        config_start_freq = QStandardItem(str(stimulus.get("start_freq")))
        config_stop_freq = QStandardItem(str(stimulus.get("stop_freq")))
        config_sample_rate = QStandardItem(str(stimulus.get("sample_rate")))
        config_total_time = QStandardItem(str(stimulus.get("total_time")))
        config_num_steps = QStandardItem(str(stimulus.get("num_steps")))
        config_list.append(stimulus_name)
        config_list.append(config_method)
        config_list.append(config_type)
        config_list.append(config_repeat_times)
        config_list.append(config_start_freq)
        config_list.append(config_stop_freq)
        config_list.append(config_sample_rate)
        config_list.append(config_total_time)
        config_list.append(config_num_steps)
        return config_list
    
    def set_stimulus_method(self, config_method):
        stimulus_method = self.STIMULUS_DICT.get(config_method)
        stimulus_method_item = QStandardItem(stimulus_method)
        return stimulus_method_item
    
    def set_stimulus_type(self, config_type):
        stimulus_type = self.STIMULUS_DICT_2.get(config_type)
        stimulus_type_item = QStandardItem(stimulus_type)
        return stimulus_type_item

    @staticmethod
    def load_stimulus_config_from_db():
        """
            Loads stimulus configuration data from the database and converts it into a specific data structure.

            This function calls the `query_all_stimulus_info` method of the `StimulusSignalManagement` class
            to retrieve all stimulus signal information from the database. If the query is successful, it
            transforms each stimulus signal into a dictionary and appends it to the `stimulus_list` for return.

            Returns:
                list: A list containing all stimulus configuration information. Each stimulus configuration is a dictionary
                    with the following key-value pairs:
                        - 'name': The name of the stimulus signal, formatted as "stimulus_{stimulus_method}_{index}".
                        - 'stimulus_method': The method of the stimulus signal.
                        - 'stimulus_type': The type of the stimulus signal.
                        - 'start_freq': The starting frequency of the stimulus signal.
                        - 'stop_freq': The ending frequency of the stimulus signal.
                        - 'total_time': The total duration of the stimulus signal.
                        - 'repeat_times': The number of repetitions of the stimulus signal.
                        - 'sample_rate': The sampling rate of the stimulus signal.
                        - 'num_steps': The number of steps in the stimulus signal.
                        - 'stimulus_name': The name of the configuration for the stimulus signal.
                        - 'is_default': Indicates whether it is the default stimulus signal.
        """
        stimulus_list = []
        # Query the database to retrieve all stimulus signal information
        query_code, query_data = StimulusSignalManagement().query_all_stimulus_info()
        # If the query is successful, process the query results
        if query_code == error_code.OK:
            for idx, info in enumerate(query_data):
                query_data_idx = query_data[idx]
                # Convert each stimulus signal information into a dictionary
                stimulus = {
                    'stimulus_id': query_data_idx[0],
                    'stimulus_method': query_data_idx[1],
                    'stimulus_type': query_data_idx[2],
                    'start_freq': query_data_idx[4],
                    'stop_freq': query_data_idx[5],
                    'total_time': query_data_idx[7],
                    'repeat_times': query_data_idx[3],
                    'sample_rate': query_data_idx[6],
                    'num_steps': query_data_idx[8],
                    'is_default': query_data_idx[9],
                    'stimulus_name': query_data_idx[10]
                }
                # Append the converted stimulus signal information to the list
                stimulus_list.append(stimulus)
        return stimulus_list
    
    def get_selected_stimulus_config(self):
        if self.select_stimulus_row is None:
            return None
        self.loaded_stimulus[self.select_stimulus_row].pop("stimulus_name")
        self.loaded_stimulus[self.select_stimulus_row].pop("is_default")
        self.loaded_stimulus[self.select_stimulus_row].pop("stimulus_id")
        return self.loaded_stimulus[self.select_stimulus_row]
    
    def get_select_stimulus_row(self):
        return self.select_stimulus_row
    

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
    
    def setData(self, index, value, role=Qt.EditRole):
        return super().setData(index, value, role)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = LoadStimulusDialog()
    a = dialog.exec()
    print(a)
    # sys.exit(app.exec_())