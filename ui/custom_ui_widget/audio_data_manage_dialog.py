"""
========================================================================================================================
= Description: Audio data display
= Author: DY Gao Qiang
= Date: 2025/08/11
= Version: 0.25.07.01
= Function Description:
    1. Audio data display
        Load all audio record information from the database
        Display detailed information of the audio file in the table view, including:
        File name
        Product model
        Audio tag (OK/NG)
        Sampling rate
        Recording time
        Type of excitation signal
    2. Data filtering function
        Provide multi-dimensional filtering options
            Filter by product model
            Filter by recording date
            Filter by sampling rate (44100Hz, 48000Hz)
            Filter by label (OK, NG, unmarked)
        Support combined filtering conditions
        Provide the "Display All" function to reset the filter conditions

= History:
    2025/08/11: Set up the project.
========================================================================================================================
"""

import copy
from re import fullmatch

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox, QCheckBox, QGroupBox, QComboBox, QDialog, QVBoxLayout, QHBoxLayout, QPushButton

from base.log_manager import LogManager
from base.recording_management import RecordingManager
from consts import error_code, ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.custom_table_widget import DataManageDialog


class audioDataManageDialog(DataManageDialog):

    def __init__(self, logger: LogManager, hide_select_not_label = False):
        super(audioDataManageDialog, self).__init__()

        self.logger = logger
        self.recording_manager = RecordingManager()
        self.all_audio_data = list()
        self.product_model_set = set()
        self.record_date_set = set()
        self.filter_audio_data = list()
        self.is_filter_flag = False
        self.all_select_flag = False
        self.select_wave_data = dict()
        self.stimulus_name = dict()
        self.filter_config = dict()
        self.packaging_progress = None
        self.is_hide_select_not_label = hide_select_not_label

        self.all_selected_checkbox = QCheckBox("全选")

        self.init_ui_layout(0, 7, [])

        self.set_view_checked_changed(self.on_row_checkbox_toggled)

        self.init_base_ui()

    def init_base_ui(self):
        self.set_checkable_of_column([0])
        self.set_h_header(["", "文件名称", "产品型号", "音频标签", "采样率", "录音时间", "激励信号"])
        self.verticalHeader().setVisible(False)

        self.set_top_layout()

        self.load_all_audio_data()
        self.load_audio_data_to_view()
        self.set_select_wave_num_text(0)

    def set_top_layout(self):
        filter_btn = QPushButton(" 筛  选 ")

        self.all_selected_checkbox.clicked.connect(self.on_all_selected_changed)
        filter_btn.clicked.connect(self.on_click_filter_btn)

        self.top_layout.addSpacing(4)
        self.top_layout.addWidget(self.all_selected_checkbox)
        self.top_layout.addStretch()
        self.top_layout.addWidget(filter_btn)

    def on_all_selected_changed(self):
        if self.all_select_flag is True:
            self.set_all_checkboxes_checked([0], False)
            self.all_select_flag = False
        elif self.all_select_flag is False:
            self.set_all_checkboxes_checked([0], True)
            self.all_select_flag = True

    def on_click_filter_btn(self):
        filter_config = copy.deepcopy(self.filter_config)
        dlg = FilterAudioDialog(self.product_model_set, self.record_date_set, filter_config)
        dlg.hide_select_not_label_check_box(self.is_hide_select_not_label)
        flag, filter_config = dlg.exec()
        if flag == 1:
            self.filter_config = filter_config
            self.filter_audio_data_at_filter_config(self.filter_config)
            self.is_filter_flag = True
            if len(self.all_audio_data) == len(self.filter_audio_data) and not self.is_filter_flag:
                return
            self.load_audio_data_to_view()
            self.all_selected_checkbox.setChecked(False)
            self.all_select_flag = False
            self.set_select_wave_num_text(0)
        elif flag == 2:
            self.show_all_wave()

    def show_all_wave(self):
        if self.is_filter_flag is False or len(self.all_audio_data) == len(self.filter_audio_data):
            return
        self.is_filter_flag = False
        self.filter_audio_data.clear()
        self.load_audio_data_to_view()
        self.filter_config = dict()
        self.all_selected_checkbox.setChecked(False)
        self.all_select_flag = False
        self.set_select_wave_num_text(0)

    def update_filter_sets_after_deletion(self):
        if not self.all_audio_data:
            self.record_date_set.clear()
            self.filter_audio_data.clear()
            self.product_model_set.clear()
            self.filter_config.clear()
        elif len(self.filter_config) == 1:
            if "select_record_date" in self.filter_config:
                self.record_date_set.remove(self.filter_config["select_record_date"])
                self.filter_config.pop("select_record_date")
            if "select_product_model" in self.filter_config:
                self.product_model_set.remove(self.filter_config["select_product_model"])
                self.filter_config.pop("select_product_model")

    def load_all_audio_data(self):
        query_code, result = self.recording_manager.get_record_audio_data()
        if query_code == error_code.OK:
            self.all_audio_data = result
            self.logger.info("finish loading audio data")
        else:
            self.logger.error(result)

        code, result = self.recording_manager.query_stimulus_name_and_id()

        if code == error_code.OK:
            self.stimulus_name = result
            self.logger.info("finish loading stimulus data")
        else:
            self.logger.error(result)

    def load_audio_data_to_view(self):
        self.select_wave_data.clear()
        if self.is_filter_flag is True:
            self.load_audio_data_to_model(self.filter_audio_data, self.stimulus_name)
        elif self.is_filter_flag is False:
            result = self.load_audio_data_to_model(self.all_audio_data, self.stimulus_name)
            self.product_model_set, self.record_date_set = result

    def load_audio_data_to_model(self, audio_data, stimulus_name):
        self.setRowCount(0)
        if not audio_data:
            return set(), set()

        product_model_set = set()
        record_date_set = set()
        for row, item in enumerate(audio_data):
            product_model_set.add(item[2])
            record_date_set.add(item[4])

            file_name = item[1].split("/")[-1]
            row_stimulus_name = stimulus_name.get(item[7], "无激励信号")
            row_data_list = [None, file_name, item[2], item[5], item[3], item[4], row_stimulus_name]
            self.add_row_data(row_data_list)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        return product_model_set, record_date_set

    def filter_audio_data_at_filter_config(self, filter_config: dict):
        result = self.all_audio_data[:]
        for key, value in filter_config.items():
            if key in ["select_sample_rate", "select_labels"] and value:
                result = [i for i in result if any(v in i for v in value)]
            elif key not in ["select_sample_rate", "select_labels"]:
                result = [i for i in result if value in i]
        self.filter_audio_data = result

    def set_select_wave_num_text(self, select_num):
        if self.is_filter_flag is True:
            totle_num = len(self.filter_audio_data)
        elif self.is_filter_flag is False:
            totle_num = len(self.all_audio_data)
        text = "已选择 %s 个文件, 共 %s 个文件"
        self.set_select_data_num_text(text, (select_num, totle_num))

    def on_row_checkbox_toggled(self, item, is_checked):
        if self.is_filter_flag is True:
            audio_data = self.filter_audio_data
        elif self.is_filter_flag is False:
            audio_data = self.all_audio_data

        row = item.row()
        if is_checked is True:
            self.select_wave_data[str(row)] = audio_data[row]
        elif is_checked is False:
            self.select_wave_data.pop(str(row))

        if self.all_select_flag is True:
            self.all_selected_checkbox.setChecked(False)
            self.all_select_flag = False
        elif self.all_select_flag is False and len(self.select_wave_data) == len(audio_data):
            self.all_selected_checkbox.setChecked(True)
            self.all_select_flag = True

        select_num = len(self.select_wave_data)
        self.set_select_wave_num_text(select_num)

    def delete_audio_data_with_id(self, id_list: list):
        if not id_list:
            return
        self.all_audio_data = [item for item in self.all_audio_data if item[0] not in id_list]

        if self.is_filter_flag is True:
            self.filter_audio_data = [item for item in self.filter_audio_data if item[0] not in id_list]

    def update_packaging_progress(self, progress, total):
        self.packaging_progress.setValue(progress)


class FilterAudioDialog(QDialog):

    def __init__(self, product_model_set, record_date_set, filter_config: dict = None, parent=None):
        super(FilterAudioDialog, self).__init__(parent)

        self.product_model_set = product_model_set
        self.record_date_set = record_date_set
        self.select_sample_rate = list()
        self.select_labels = list()
        self.select_record_date = None
        self.select_product_model = None
        self.filter_config = filter_config
        self.filter_sample_rate_num = 0
        self.filter_label_num = 0
        self.sample_rate_filter_layout = QHBoxLayout()

        self.is_clicked_ok = False

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("筛选")
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        self.resize(300, 300)

        date_filter_groupbox = self.set_date_filter_groupbox()
        sample_rate_filter_groupbox = self.set_sample_rate_filter_groupbox()
        label_filter_groupbox = self.set_label_filter_groupbox()
        product_model_filter_groupbox = self.create_product_model_filter_groupbox()
        btn_layout = self.create_btn_layout()

        layout = QVBoxLayout()
        layout.addWidget(product_model_filter_groupbox)
        layout.addWidget(date_filter_groupbox)
        layout.addWidget(sample_rate_filter_groupbox)
        layout.addWidget(label_filter_groupbox)
        layout.addSpacing(10)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

        self.setStyleSheet(
            ui_style_const.qcheckbox_style
            + ui_style_const.qcombobox_style
            + ui_style_const.qlabel_style
            + ui_style_const.qpushbutton_style
            + ui_style_const.qgroupbox_style
        )

    def set_date_filter_groupbox(self):
        date_groupbox = QGroupBox("日期")
        data_filter_combobox = QComboBox()
        data_filter_combobox.setEditable(True)
        data_filter_combobox.addItem("ALL")
        data_filter_combobox.setMinimumWidth(100)
        if self.record_date_set:
            data_filter_combobox.addItems(sorted(self.record_date_set))
        data_filter_combobox.lineEdit().selectAll()
        data_filter_combobox.textActivated.connect(self.update_date_filter_config)

        if self.filter_config.get("select_record_date"):
            data_filter_combobox.setCurrentText(self.filter_config.get("select_record_date"))
            self.select_record_date = self.filter_config.get("select_record_date")

        data_filter_layout = QHBoxLayout()
        data_filter_layout.addWidget(data_filter_combobox, alignment=Qt.AlignLeft)
        date_groupbox.setLayout(data_filter_layout)

        return date_groupbox

    def set_sample_rate_filter_groupbox(self):
        sample_rate_groupbox = QGroupBox("采样率")
        select_44100_check_box = QCheckBox("44100")
        select_48000_check_box = QCheckBox("48000")

        select_44100_check_box.toggled.connect(self.on_select_44100_check_box_toggled)
        select_48000_check_box.toggled.connect(self.on_select_48000_check_box_toggled)

        if self.filter_config.get("select_sample_rate"):
            select_44100_check_box.setChecked(44100 in self.filter_config.get("select_sample_rate"))
            select_48000_check_box.setChecked(48000 in self.filter_config.get("select_sample_rate"))
        if not self.filter_config:
            select_44100_check_box.setChecked(True)
            select_48000_check_box.setChecked(True)

        self.sample_rate_filter_layout.addWidget(select_44100_check_box, 1)
        self.sample_rate_filter_layout.addWidget(select_48000_check_box, 1)
        
        sample_rate_groupbox.setLayout(self.sample_rate_filter_layout)

        return sample_rate_groupbox

    def set_label_filter_groupbox(self):
        label_filter_groupbox = QGroupBox("标签")
        self.select_ok_check_box = QCheckBox("OK")
        self.select_ng_check_box = QCheckBox("NG")
        self.select_not_label_check_box = QCheckBox("未标记")

        self.select_ng_check_box.toggled.connect(self.select_ng_check_box_toggled)
        self.select_ok_check_box.toggled.connect(self.select_ok_check_box_toggled)
        self.select_not_label_check_box.toggled.connect(self.select_not_label_check_box_toggled)

        if self.filter_config.get("select_labels"):
            self.select_ok_check_box.setChecked("OK" in self.filter_config.get("select_labels"))
            self.select_ng_check_box.setChecked("NG" in self.filter_config.get("select_labels"))
            self.select_not_label_check_box.setChecked("not_labeled" in self.filter_config.get("select_labels"))
        elif not self.filter_config:
            self.select_ok_check_box.setChecked(True)
            self.select_ng_check_box.setChecked(True)
            self.select_not_label_check_box.setChecked(True)

        label_filter_layout = QHBoxLayout()
        label_filter_layout.addWidget(self.select_ok_check_box, 1)
        label_filter_layout.addWidget(self.select_ng_check_box, 1)
        label_filter_layout.addWidget(self.select_not_label_check_box, 1)

        label_filter_groupbox.setLayout(label_filter_layout)

        return label_filter_groupbox
    
    def hide_select_not_label_check_box(self, is_hide: bool):
        if is_hide:
            self.select_not_label_check_box.hide()
            self.select_not_label_check_box.setChecked(False)
        else:
            self.sample_rate_filter_layout.addSpacing(10)
            self.sample_rate_filter_layout.addStretch(1)
            self.select_not_label_check_box.show()

    def create_product_model_filter_groupbox(self):
        product_model_filter_groupbox = QGroupBox("产品型号")
        product_model_combobox = QComboBox()
        product_model_combobox.setEditable(True)
        product_model_combobox.addItem("ALL")
        product_model_combobox.setMinimumWidth(100)
        if self.product_model_set:
            product_model_combobox.addItems(self.product_model_set)

        product_model_combobox.currentTextChanged.connect(self.update_product_model_filter)

        if self.filter_config.get("select_product_model"):
            product_model_combobox.setCurrentText(self.filter_config.get("select_product_model"))

        product_model_layout = QHBoxLayout()
        product_model_layout.addWidget(product_model_combobox, alignment=Qt.AlignLeft)
        product_model_filter_groupbox.setLayout(product_model_layout)

        return product_model_filter_groupbox

    def create_btn_layout(self):
        ok_btn = QPushButton(" 确  认 ")
        ok_btn.clicked.connect(self.on_click_ok_btn)
        cancel_btn = QPushButton(" 取  消 ")
        cancel_btn.clicked.connect(self.on_click_cancel_btn)
        a = QPushButton()
        a.setVisible(False)
        a.setDefault(True)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(a)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(ok_btn)

        return btn_layout

    def update_date_filter_config(self):
        select_date = self.sender().currentText()
        if self.check_date_style(select_date):
            if select_date == "ALL":
                if self.select_record_date:
                    self.select_record_date = None
            else:
                self.select_record_date = select_date
        else:
            if self.select_record_date == None:
                self.sender().setCurrentText("ALL")
            else:
                self.sender().setCurrentText(self.select_record_date)

    def on_select_44100_check_box_toggled(self):
        sender = self.sender()
        if sender.isChecked():
            if 44100 not in self.select_sample_rate:
                self.filter_sample_rate_num += 1
                self.select_sample_rate.append(44100)
        else:
            if self.filter_sample_rate_num == 1:
                sender.setChecked(True)
                QMessageBox.warning(self, "提示", "请至少选择一个采样率")
            else:
                self.select_sample_rate.remove(44100)
                self.filter_sample_rate_num -= 1

    def on_select_48000_check_box_toggled(self):
        sender = self.sender()
        if sender.isChecked():
            if 48000 not in self.select_sample_rate:
                self.filter_sample_rate_num += 1
                self.select_sample_rate.append(48000)
        else:
            if self.filter_sample_rate_num == 1:
                sender.setChecked(True)
                QMessageBox.warning(self, "提示", "请至少选择一个采样率")
            else:
                self.select_sample_rate.remove(48000)
                self.filter_sample_rate_num -= 1

    def select_ng_check_box_toggled(self):
        sender = self.sender()
        if sender.isChecked():
            if "NG" not in self.select_labels:
                self.filter_label_num += 1
                self.select_labels.append("NG")
        else:
            if self.filter_label_num == 1:
                sender.setChecked(True)
                QMessageBox.warning(self, "提示", "请至少选择一个标签")
            else:
                self.select_labels.remove("NG")
                self.filter_label_num -= 1

    def select_ok_check_box_toggled(self):
        sender = self.sender()
        if sender.isChecked():
            if "OK" not in self.select_labels:
                self.filter_label_num += 1
                self.select_labels.append("OK")
        else:
            if self.filter_label_num == 1:
                sender.setChecked(True)
                QMessageBox.warning(self, "提示", "请至少选择一个标签")
            else:
                self.select_labels.remove("OK")
                self.filter_label_num -= 1

    def select_not_label_check_box_toggled(self):
        sender = self.sender()
        if sender.isChecked():
            if "not_labeled" not in self.select_labels:
                self.filter_label_num += 1
                self.select_labels.append("not_labeled")
        else:
            if self.filter_label_num == 1:
                sender.setChecked(True)
                QMessageBox.warning(self, "提示", "请至少选择一个标签")
            else:
                self.select_labels.remove("not_labeled")
                self.filter_label_num -= 1

    def update_product_model_filter(self):
        sender = self.sender()
        if sender.currentText() == "ALL":
            if self.select_product_model:
                self.select_product_model = None
        else:
            self.select_product_model = sender.currentText()

    def check_date_style(self, input_str):
        if input_str == "ALL":
            return True
        reg = r"\d{4}-\d{2}-\d{2}"
        if not fullmatch(reg, input_str):
            QMessageBox.warning(self, "警告", "时间格式错误，请检查输入!")
            return False
        return True

    def on_click_cancel_btn(self):
        self.is_clicked_ok = 0
        self.close()

    def on_click_ok_btn(self):
        self.filter_config = dict()

        ok_flag = False

        if self.select_sample_rate:
            self.filter_config["select_sample_rate"] = self.select_sample_rate
        if self.select_labels:
            self.filter_config["select_labels"] = self.select_labels
        if self.select_record_date:
            self.filter_config["select_record_date"] = self.select_record_date
        if self.select_product_model:
            self.filter_config["select_product_model"] = self.select_product_model

        if (
            self.filter_sample_rate_num == 2
            and self.filter_label_num == 3
            and not self.select_product_model
            and not self.select_record_date
        ):
            ok_flag = True

        if ok_flag:
            self.is_clicked_ok = 2
        else:
            self.is_clicked_ok = 1
        self.close()

    def exec(self):
        super().exec()
        if self.is_clicked_ok:
            return self.is_clicked_ok, self.filter_config
        else:
            result = dict()
            return self.is_clicked_ok, result
