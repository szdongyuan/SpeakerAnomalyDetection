import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItem
from PyQt5.QtWidgets import QApplication, QPushButton, QMessageBox

from base.log_manager import LogManager
from base.stimulus_signal_management import StimulusSignalManagement
from consts import error_code
from ui.custom_ui_widget.custom_table_widget import DataManageDialog


class LoadStimulusDialog(DataManageDialog):
    # Define stimulus signal types
    STIMULUS_DICT = {
        "chirp": "啁啾",
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
        self.is_clicked_ok = False
        self.select_stimulus_row = None
        self.is_edit_item = True

        self.loaded_stimulus = self.load_stimulus_config_from_db()

        self.init_ui_layout(0, 10, [1])

        self.set_view_checked_changed(self.on_row_checkbox_toggled)
        self.model().dataChanged.connect(self.is_edit_model_item)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("加载激励信号")
        self.set_checkable_of_column([0])
        self.set_h_header(
            [
                "",
                "配置名称",
                "信号模式",
                "信号类型",
                "重复次数",
                "起始频率",
                "截止频率",
                "采样频率",
                "信号时长",
                "步进数量",
            ]
        )
        self.verticalHeader().setVisible(False)
        self.insert_stimulus_config_to_table(self.loaded_stimulus)
        self.set_column_alignment(4, Qt.AlignCenter)
        self.set_column_alignment(9, Qt.AlignCenter)
        self.select_data_num.hide()

        self.set_bottom_layout()
        self.layout().setContentsMargins(13, 20, 13, 13)

    def set_bottom_layout(self):
        del_config_btn = QPushButton("删除配置")
        cancel_btn = QPushButton(" 取  消 ")
        ok_btn = QPushButton(" 确  定 ")

        del_config_btn.clicked.connect(self.del_config)
        ok_btn.clicked.connect(self.ok_btn_clicked)
        cancel_btn.clicked.connect(self.close)

        self.bottom_layout.addWidget(del_config_btn)
        self.bottom_layout.addStretch()
        self.bottom_layout.addWidget(cancel_btn)
        self.bottom_layout.addWidget(ok_btn)

    def on_row_checkbox_toggled(self, item, is_checked):
        if is_checked:
            if self.select_stimulus_row is not None:
                self.model().item(self.select_stimulus_row, 0).setCheckState(Qt.Unchecked)
            self.select_stimulus_row = item.row()
        else:
            self.select_stimulus_row = None

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
        query_code, query_data = StimulusSignalManagement().query_all_stimulus_info()
        if query_code == error_code.OK:
            for idx, info in enumerate(query_data):
                query_data_idx = query_data[idx]
                stimulus = {
                    "stimulus_id": query_data_idx[0],
                    "stimulus_method": query_data_idx[1],
                    "stimulus_type": query_data_idx[2],
                    "start_freq": query_data_idx[4],
                    "stop_freq": query_data_idx[5],
                    "total_time": query_data_idx[7],
                    "repeat_times": query_data_idx[3],
                    "sample_rate": query_data_idx[6],
                    "num_steps": query_data_idx[8],
                    "is_default": query_data_idx[9],
                    "stimulus_name": query_data_idx[10],
                }
                stimulus_list.append(stimulus)
        return stimulus_list

    def insert_stimulus_config_to_table(self, stimulus_info: list[dict]):
        self.setRowCount(0)
        for stimulus in stimulus_info:
            config_list = [
                None,
                stimulus["stimulus_name"],
                stimulus["stimulus_method"],
                stimulus["stimulus_type"],
                stimulus["repeat_times"],
                stimulus["start_freq"],
                stimulus["stop_freq"],
                stimulus["sample_rate"],
                stimulus["total_time"],
                stimulus["num_steps"],
            ]
            self.add_row_data(config_list)

    def get_selected_stimulus_config(self):
        if self.select_stimulus_row is None:
            return None
        self.loaded_stimulus[self.select_stimulus_row].pop("stimulus_name")
        self.loaded_stimulus[self.select_stimulus_row].pop("is_default")
        self.loaded_stimulus[self.select_stimulus_row].pop("stimulus_id")
        return self.loaded_stimulus[self.select_stimulus_row]

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

    def ok_btn_clicked(self):
        if self.select_stimulus_row is None:
            QMessageBox.warning(self, "提示", "请选择要加载的激励信号配置")
            return
        self.is_clicked_ok = True
        self.close()

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
            self.select_stimulus_row = None

    def exec(self):
        super().exec_()
        if self.is_clicked_ok:
            return self.get_selected_stimulus_config()
        else:
            return None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = LoadStimulusDialog(LogManager.set_log_handler("core"))
    a = dialog.exec()
    print(a)
    # sys.exit(app.exec_())
