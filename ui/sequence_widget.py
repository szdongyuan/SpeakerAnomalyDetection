import json
import os
import re
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon, QPainter, QColor, QFont
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFrame, QCheckBox, QMessageBox
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QStackedWidget

from base.barcode_scanning_processor import BarcodeScanner
from base.data_struct.data_deal_struct import DataDealStruct
from base.file_ops import FileOps
from base.utils.custom_signals import sign
from base.load_config import LoadUiConfig
from base.log_manager import LogManager
from base.play_and_record import record_without_play, get_recorded_info
from base.recording_management import RecordingManager
from base.save_data import save_recorded_data_to_json, ensure_test_result_file
from base.soundcard_calibration_manager import get_mic_deviation_value
from base.tcp_service import TcpServer, check_tcp_msg_format
from base.temp_tcp_client import TempTcpClient
from consts import ui_style_const, error_code
from consts.action_code import RequestTypeEnum
from consts.running_consts import DEFAULT_DIR
from ui.operation_sequence import AnalysisModelSelect
from ui.signal_analysis_window import get_class_mapping


class SequenceWindow(QWidget):
    tcp_server = None

    def __init__(self):
        """Initializes the class instance, setting up the user interface and necessary parameters."""
        super().__init__()
        self.data_struct = DataDealStruct()
        self.collect_or_analyse_layout = QHBoxLayout()
        self.recorded_path = None
        self.refresh_stimulus_flag = None
        self.add_or_update_wave_flag = True

        self.deviation_value = get_mic_deviation_value()
        self.sequence_config = list()
        self.analysis_config = dict()
        self.spec_list = list()
        self.get_sequence_config_from_json()
        self.init_data_struct_stimulus_config()
        self.init_fft_and_stft_flag()
        self.analysis_window = []
        self.default_ed = None
        self.default_ed_result = None
        self.sequence_layout = QVBoxLayout()
        self.player_btn = QPushButton()
        self.replayer_btn = QPushButton()
        self.data_btn = QPushButton()
        self.player_status_flag = False
        self.scanner_barcode_thread = None
        self.barcode_scanner = BarcodeScanner()
        self.vendor_id = None
        self.product_id = None
        self.recorded_signal_info = {}
        self.ip_format = True
        self.port_format = True
        self.clicked_player_flag = False
        self.tcp_ip = None
        self.tcp_port = None
        self.tcp_ip, self.tcp_port = LoadUiConfig.get_tcp_config()
        self.mode = None
        self.current_recorded_count = None

        self.default_logger = LogManager.set_log_handler("core")
        self.init_ui()

    def init_ui(self):
        """
        Initializes the user interface of the SequenceWindow.

        This method sets up the window icon, minimum height, and creates the main layout
        by adding toolbar and waveform layouts. It also connects button click events to
        their respective handlers and applies style sheets to the widgets.
        """
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setMinimumHeight(700)
        layout = self.create_layout()
        waveform_layout = self.create_waveform_layout()

        self.sequence_layout.addLayout(layout)
        self.sequence_layout.addLayout(waveform_layout)
        self.sequence_layout.setAlignment(Qt.AlignCenter)
        self.sequence_layout.setContentsMargins(0, 0, 0, 0)

        self.setLayout(self.sequence_layout)

        sign.run_test_sign.connect(self.start_this_play, Qt.AutoConnection)
        # sign.get_result_file_sign.connect(self.get_result_file, Qt.AutoConnection)
        # sign.set_result_file_sign.connect(self.set_result_file, Qt.AutoConnection)
        sign.update_mode_display_sign.connect(self.get_sequence_config_from_json, Qt.AutoConnection)
        sign.test_insert_data_into_db_sign.connect(self.update_recorded_label_in_test_mode, Qt.AutoConnection)
        # sign.update_mode_display_sign.connect(self.update_mode_display, Qt.AutoConnection)
        self.update_mode_display(0)
        self.setStyleSheet(
            ui_style_const.qcombobox_style
            + ui_style_const.qpushbutton_style
            + ui_style_const.qlineedit_style
            + ui_style_const.qframe_style
            + ui_style_const.qlabel_style
            + ui_style_const.qcheckbox_style
        )

    def init_data_struct_stimulus_config(self):
        if not self.sequence_config:
            return
        acq_config = self.sequence_config[0]["seq1"]["acq"]
        if acq_config["mode"] == "PLAY_AND_RECORD":
            AnalysisModelSelect.set_data_struct_stimulus_signal(self.data_struct, acq_config["detail"])
        else:
            self.data_struct.sample_rate = acq_config["detail"]["sample_rate"]

    def create_layout(self):
        """
        Create the toolbar layout.

        This method initializes and configures the toolbar layout for the application.
        It sets up button styles, adds labels and input fields, and sets layout parameters.
        The layout is used at the top of the interface to provide easy access to key functionalities.

        Returns:
            QHBoxLayout: The configured toolbar layout object.
        """
        self.player_btn.setFixedSize(100, 40)
        self.setFocusPolicy(Qt.NoFocus)
        self.player_btn.setToolTip("播放")
        self.player_btn.setStyleSheet(ui_style_const.toolbar_button_style)
        self.player_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/play.png"))
        self.player_btn.setIconSize(QSize(35, 35))
        self.player_btn.clicked.connect(lambda: self.on_clicked_player_btn())
        self.replayer_btn.setFixedSize(100, 40)
        self.replayer_btn.setToolTip("重播")
        self.replayer_btn.setDisabled(True)
        self.replayer_btn.setStyleSheet(ui_style_const.toolbar_button_style)
        self.replayer_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/replay.png"))
        self.replayer_btn.setIconSize(QSize(30, 30))
        self.replayer_btn.clicked.connect(lambda: self.judge_play_and_record())

        self.data_btn.setFixedSize(100, 40)
        self.data_btn.setToolTip("分析")
        self.data_btn.setEnabled(False)
        self.data_btn.setStyleSheet(ui_style_const.toolbar_button_style)
        self.data_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/data.png"))
        self.data_btn.setIconSize(QSize(35, 35))

        self.data_btn.clicked.connect(self.run)

        type_label = QLabel(" 型 号： ")
        last_recorded_info = LoadUiConfig().load_last_recorded_info(self.default_logger)
        if last_recorded_info:
            product_model = last_recorded_info.get("product_model", "S004-1")
        else:
            product_model = "S004-1"
        type_label.setFixedHeight(40)
        self.lineedit_type = QLineEdit(product_model)
        self.lineedit_type.editingFinished.connect(lambda: self.lineedit_type_lose_focus(self.lineedit_type))
        self.lineedit_type.setFixedHeight(35)
        self.lineedit_type.setAlignment(Qt.AlignCenter)
        label_count = QLabel(" 计 数： ")
        label_count.setFixedHeight(40)

        result, _ = LoadUiConfig.load_recorded_num_from_json(self.default_logger)
        if result is None:
            self.current_recorded_count = 1
        else:
            self.current_recorded_count = result
        self.lineedit_count = QLineEdit(str(self.current_recorded_count))
        self.lineedit_count.setFixedHeight(35)
        self.lineedit_count.setAlignment(Qt.AlignCenter)
        self.lineedit_count.editingFinished.connect(lambda: self.lineedit_count_lose_focus(self.lineedit_count))
        self.lineedit_count.returnPressed.connect(lambda: self.validate_count(self.lineedit_count))

        self.scanner_tcp = QCheckBox(" TCP ", self)
        self.scanner_tcp.setChecked(False)
        self.scanner_tcp.stateChanged.connect(self.is_clicked_tcp)

        self.ip_label = QLabel(" IP:")
        self.ip_edit = QLineEdit(self)
        self.ip_edit.setPlaceholderText(f"{self.tcp_ip}")
        self.ip_edit.setDisabled(True)
        self.ip_edit.setFixedHeight(35)
        self.ip_edit.setAlignment(Qt.AlignCenter)

        self.port_label = QLabel(" Port:")
        self.port_edit = QLineEdit(self)
        self.port_edit.setPlaceholderText(f"{self.tcp_port}")
        self.port_edit.setDisabled(True)
        self.port_edit.setFixedHeight(35)
        self.port_edit.setFixedWidth(80)
        self.port_edit.setAlignment(Qt.AlignCenter)
        self.ip_edit.editingFinished.connect(self.validate_ip)
        self.port_edit.editingFinished.connect(self.validate_port)

        vertical_line_1 = QFrame()
        vertical_line_2 = QFrame()
        vertical_line_3 = QFrame()
        vertical_line_4 = QFrame()
        vertical_line_5 = QFrame()
        vertical_line_6 = QFrame()
        vertical_line_7 = QFrame()
        vertical_line_8 = QFrame()
        vertical_line_9 = QFrame()
        vertical_line_1.setFrameShape(QFrame.VLine)
        vertical_line_2.setFrameShape(QFrame.VLine)
        vertical_line_3.setFrameShape(QFrame.VLine)
        vertical_line_4.setFrameShape(QFrame.VLine)
        vertical_line_5.setFrameShape(QFrame.VLine)
        vertical_line_6.setFrameShape(QFrame.VLine)
        vertical_line_9.setFrameShape(QFrame.VLine)
        vertical_line_7.setFrameShape(QFrame.HLine)
        vertical_line_8.setFrameShape(QFrame.HLine)
        vertical_line_7.setFixedHeight(1)
        vertical_line_8.setFixedHeight(1)

        # Create and configure the toolbar layout
        tools_layout = QVBoxLayout()
        layout = QHBoxLayout()
        layout.addWidget(self.player_btn)
        layout.addWidget(vertical_line_1)
        layout.addWidget(self.replayer_btn)
        layout.addWidget(vertical_line_2)
        layout.addWidget(self.data_btn)
        layout.addWidget(vertical_line_3)
        layout.addWidget(type_label)
        layout.addWidget(self.lineedit_type)
        layout.addSpacing(10)
        layout.addWidget(vertical_line_4)
        layout.addWidget(label_count)
        layout.addWidget(self.lineedit_count)
        layout.addSpacing(10)
        layout.addWidget(vertical_line_5)
        layout.addSpacing(10)
        layout.addSpacing(10)
        layout.addWidget(vertical_line_9)
        layout.addSpacing(10)
        layout.addWidget(self.scanner_tcp)
        layout.addSpacing(10)
        layout.addWidget(self.ip_label)
        layout.addSpacing(10)
        layout.addWidget(self.ip_edit)
        layout.addSpacing(10)
        layout.addWidget(self.port_label)
        layout.addSpacing(10)
        layout.addWidget(self.port_edit)
        layout.addSpacing(10)
        layout.addWidget(vertical_line_6)
        layout.addStretch()
        layout.setContentsMargins(4, 0, 0, 0)
        tools_layout.addWidget(vertical_line_7)
        tools_layout.addLayout(layout)
        tools_layout.addWidget(vertical_line_8)

        tools_layout.setSpacing(0)

        return tools_layout

    def validate_ip(self):
        ip = self.ip_edit.text()
        pattern = r"^((25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)$"
        if not re.match(pattern, ip):
            self.ip_format = False
            QMessageBox.warning(self, "无效 IP", "每个段的值必须在 0 到 255 之间。")
            self.ip_edit.setFocus()
            self.scanner_tcp.setEnabled(False)
            return False
        self.ip_format = True
        self.update_scanner_tcp_state()

    def validate_port(self):
        port_text = self.port_edit.text()
        if not port_text.isdigit():
            self.port_format = False
            QMessageBox.warning(self, "无效端口", "端口号必须是数字")
            self.port_edit.setFocus()
            self.scanner_tcp.setEnabled(False)
            return False
        port = int(port_text)
        if not (0 < port < 65536):
            self.port_format = False
            QMessageBox.warning(self, "无效端口", "请输入 1 到 65535 之间的端口号")
            self.port_edit.setFocus()
            self.scanner_tcp.setEnabled(False)
            return False
        self.port_format = True
        self.update_scanner_tcp_state()

    def update_scanner_tcp_state(self):
        if self.ip_format and self.port_format:
            self.scanner_tcp.setEnabled(True)
            if self.ip_edit.text():
                self.tcp_ip = self.ip_edit.text()
            if self.port_edit.text():
                self.tcp_port = self.port_edit.text()
            LoadUiConfig.write_tcp_config(self.ip_edit.text(), self.port_edit.text(), self.default_logger)
            return True
        elif self.ip_format is False and self.port_format is True:
            QMessageBox.warning(self, "无效 IP", "ip 不对")
            self.scanner_tcp.setEnabled(False)
            return False
        elif self.ip_format is True and self.port_format is False:
            QMessageBox.warning(self, "无效 端口", "端口 不对")
            self.scanner_tcp.setEnabled(False)
            return False
        else:
            QMessageBox.warning(self, "无效", "端口和ip都不对")
            self.scanner_tcp.setEnabled(False)
            return False

    def create_waveform_layout(self):
        """
            Create waveform display layout

            This function is responsible for generating a horizontal layout to display the waveform and related button area.
            It first creates a horizontal layout object and a plot widget, then sets the background color and creates
        the button layout.
            Finally, it adds these components to the layout and sets the layout margins.

            Returns:
                QHBoxLayout: The configured wavefrom layout object.
        """
        layout = QHBoxLayout()
        self.line_graph_top = pg.PlotWidget()
        self.line_graph_top.setBackground("white")
        left_area = self.create_left_layout()
        self.line_graph_top.setLabel("left", "Amplitude(V)", **{"font-size": "20px"})
        self.line_graph_top.setLabel("bottom", "Time(s)", **{"font-size": "20px"})
        self.line_graph_top.showGrid(x=True, y=True)

        font_top = QFont()
        font_top.setPixelSize(20)

        self.text_top = pg.TextItem(text="Channel_1", color=(0, 0, 0), fill=(255, 255, 255, 127))  # 黑色  # 半透明背景
        self.text_top.setFont(font_top)
        self.text_top.setPos(0, 0)  # 设置左上角位置
        self.text_top.setParentItem(self.line_graph_top.getViewBox())  # 绑定到视图框
        self.text_top.setZValue(1000)

        self.line_graph_bottom = pg.PlotWidget()
        self.line_graph_bottom.setBackground("white")
        left_area = self.create_left_layout()
        self.line_graph_bottom.setLabel("left", "Amplitude(V)", **{"font-size": "20px"})
        self.line_graph_bottom.setLabel("bottom", "Time(s)", **{"font-size": "20px"})
        self.line_graph_bottom.showGrid(x=True, y=True)

        font_bottom = QFont()
        font_bottom.setPixelSize(20)

        self.text_bottom = pg.TextItem(
            text="Channel_2", color=(0, 0, 0), fill=(255, 255, 255, 127)  # 黑色  # 半透明背景
        )
        self.text_bottom.setFont(font_bottom)
        self.text_bottom.setPos(0, 0)  # 设置左上角位置
        self.text_bottom.setParentItem(self.line_graph_bottom.getViewBox())  # 绑定到视图框
        self.text_bottom.setZValue(1000)

        font_top = QFont()
        font_top.setPixelSize(20)
        b_axis = self.line_graph_top.getAxis("bottom")
        l_axis = self.line_graph_top.getAxis("left")
        b_axis.setTickFont(font_top)
        l_axis.setTickFont(font_top)
        b_axis.setTextPen("black")
        l_axis.setTextPen("black")

        font_bottom = QFont()
        font_bottom.setPixelSize(20)
        b_axis = self.line_graph_bottom.getAxis("bottom")
        l_axis = self.line_graph_bottom.getAxis("left")
        b_axis.setTickFont(font_bottom)
        l_axis.setTickFont(font_bottom)
        b_axis.setTextPen("black")
        l_axis.setTextPen("black")

        wave_layout = QVBoxLayout()
        wave_layout.addWidget(self.line_graph_top)
        wave_layout.addWidget(self.line_graph_bottom)

        layout.addLayout(left_area, stretch=1)
        layout.addSpacing(20)
        layout.addLayout(wave_layout, stretch=8)
        layout.setContentsMargins(40, 20, 40, 20)
        layout.setSpacing(30)
        return layout

    def create_left_layout(self):
        layout = QVBoxLayout()

        mode_label = QLabel("模式：")
        self.model_button = QHBoxLayout()
        self.model_button.setSpacing(0)
        self.test_btn = QLabel("测试")
        self.test_btn.setFixedSize(100, 35)
        self.test_btn.setFont(QFont("Arial", 14))
        self.model_button.addWidget(self.test_btn)
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(mode_label)
        mode_layout.addLayout(self.model_button)

        separator_line = QFrame()
        separator_line.setFrameShape(QFrame.HLine)
        separator_line.setFrameShadow(QFrame.Sunken)
        separator_line.setStyleSheet("color: gray;")
        separator_line.setFixedHeight(2)

        # --- page 1：test ---
        self.test_page = QWidget()
        test_layout = QVBoxLayout()

        total_layout = QHBoxLayout()
        self.total_label = QLabel("总数：")
        self.total_line_edit = QLineEdit()
        self.total_line_edit.setFixedHeight(35)
        self.total_line_edit.setFixedWidth(130)
        self.total_line_edit.setDisabled(True)
        self.total_line_edit.setAlignment(Qt.AlignCenter)
        total_layout.addWidget(self.total_label)
        total_layout.addWidget(self.total_line_edit)

        ok_layout = QHBoxLayout()
        self.ok_label = QLabel("OK数：")
        self.ok_line_edit = QLineEdit()
        self.ok_line_edit.setFixedHeight(35)
        self.ok_line_edit.setFixedWidth(130)
        self.ok_line_edit.setDisabled(True)
        self.ok_line_edit.setAlignment(Qt.AlignCenter)
        ok_layout.addWidget(self.ok_label)
        ok_layout.addWidget(self.ok_line_edit)

        ng_layout = QHBoxLayout()
        self.ng_label = QLabel("NG数：")
        self.ng_line_edit = QLineEdit()
        self.ng_line_edit.setFixedHeight(35)
        self.ng_line_edit.setFixedWidth(130)
        self.ng_line_edit.setDisabled(True)
        self.ng_line_edit.setAlignment(Qt.AlignCenter)
        ng_layout.addWidget(self.ng_label)
        ng_layout.addWidget(self.ng_line_edit)

        yield_layout = QHBoxLayout()
        self.yield_label = QLabel("良率：")
        self.yield_line_edit = QLineEdit()
        self.yield_line_edit.setFixedHeight(35)
        self.yield_line_edit.setFixedWidth(130)
        self.yield_line_edit.setDisabled(True)
        self.yield_line_edit.setAlignment(Qt.AlignCenter)
        yield_layout.addWidget(self.yield_label)
        yield_layout.addWidget(self.yield_line_edit)

        datatime_layout = QHBoxLayout()
        self.datatime_label = QLabel("日期：")
        self.datatime_line_edit = QLineEdit()
        self.datatime_line_edit.setFixedHeight(35)
        self.datatime_line_edit.setFixedWidth(130)
        self.datatime_line_edit.setDisabled(True)
        self.datatime_line_edit.setAlignment(Qt.AlignCenter)
        datatime_layout.addWidget(self.datatime_label)
        datatime_layout.addWidget(self.datatime_line_edit)

        reset_btn_layout = QHBoxLayout()
        reset_btn_layout.addStretch()
        self.reset_btn = QPushButton("重置统计")
        self.reset_btn.setStyleSheet(ui_style_const.qpushbutton_style)
        reset_btn_layout.addWidget(self.reset_btn)
        reset_btn_layout.addStretch()
        self.reset_btn.clicked.connect(self.reset_test_reord)
        test_layout.addLayout(total_layout, stretch=1)
        test_layout.addLayout(ok_layout, stretch=1)
        test_layout.addLayout(ng_layout, stretch=1)
        test_layout.addLayout(yield_layout, stretch=1)
        test_layout.addLayout(datatime_layout, stretch=1)
        test_layout.addLayout(reset_btn_layout, stretch=1)
        self.test_page.setLayout(test_layout)

        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(self.test_page)

        layout.addLayout(mode_layout)
        layout.addWidget(separator_line)
        layout.addWidget(self.stacked_widget)
        layout.addStretch()

        self.init_result_files()
        return layout

    def init_fft_and_stft_flag(self):
        model_item_list = self.analysis_config.get("display_sequence", "")
        for item_name in model_item_list:
            analysis_item = self.analysis_config.get(item_name, {})
            item_type = analysis_item.get("type", None)
            self.data_struct.add_stft_or_fft_count(item_type)

    def init_result_files(self):
        current_time = datetime.now().strftime("%Y-%m-%d")
        test_result_path = DEFAULT_DIR + f"log/test_result_log/{current_time}.dat"
        test_result_template = (
            f"total: 0\n" f"ok: 0\n" f"ng: 0\n" f"ok_percent: 0\n" f"current_model: xxx\n" f"datatime: {current_time}\n"
        )
        if not os.path.exists(test_result_path):
            os.makedirs(os.path.dirname(test_result_path), exist_ok=True)
            with open(test_result_path, "w") as f:
                f.write(test_result_template)

        mark_result_path = DEFAULT_DIR + "ui/ui_config/mark_result.json"
        mark_result_template = {"total": 0, "ok": 0, "ng": 0, "not_labels": 0, "datatime": current_time}
        if not os.path.exists(mark_result_path):
            os.makedirs(os.path.dirname(mark_result_path), exist_ok=True)
            with open(mark_result_path, "w") as f:
                json.dump(mark_result_template, f, indent=4)
        else:
            self.set_result_file(1, "init")

    def get_result_file(self, index):
        current_time = datetime.now().strftime("%Y-%m-%d")
        if index == 0:
            ensure_test_result_file(self.analysis_config)
            test_result_path = DEFAULT_DIR + f"log/test_result_log/{current_time}.dat"
            with open(test_result_path, "r") as f:
                lines = f.readlines()
                total = lines[0].split(":")[1].strip()
                ok = lines[1].split(":")[1].strip()
                ng = lines[2].split(":")[1].strip()
                ok_percent = lines[3].split(":")[1].strip()
                datatime = lines[5].split(":")[1].strip()
                self.total_line_edit.setText(total)
                self.ok_line_edit.setText(ok)
                self.ng_line_edit.setText(ng)
                self.yield_line_edit.setText(ok_percent)
                self.datatime_line_edit.setText(datatime)

    def set_result_file(self, index, params):
        current_time = datetime.now().strftime("%Y-%m-%d")
        if index == 0:
            ensure_test_result_file(self.analysis_config)
            test_result_path = DEFAULT_DIR + f"log/test_result_log/{current_time}.dat"
            with open(test_result_path, "r") as f:
                lines = f.readlines()
                total = int(lines[0].split(":")[1].strip())
                ok = int(lines[1].split(":")[1].strip())
                ng = int(lines[2].split(":")[1].strip())

                if params == "OK":
                    total += 1
                    ok += 1
                elif params == "NG":
                    total += 1
                    ng += 1
                lines[0] = f"total: {total}\n"
                lines[1] = f"ok: {ok}\n"
                lines[2] = f"ng: {ng}\n"
                ok_percent = round(ok / total * 100, 2) if total > 0 else 0
                lines[3] = f"ok_percent: {ok_percent}%\n"
            with open(test_result_path, "w") as f:
                f.writelines(lines)

    def update_mode_display(self, index):
        if index == 0:
            self.stacked_widget.setCurrentIndex(0)
            self.get_result_file(0)
            self.mode = "test"
        else:
            return

    def reset_test_reord(self):
        current_time = datetime.now().strftime("%Y-%m-%d")
        test_result_path = DEFAULT_DIR + f"log/test_result_log/{current_time}.dat"
        with open(test_result_path, "r") as f:
            lines = f.readlines()
            lines[0] = f"total: 0\n"
            lines[1] = f"ok: 0\n"
            lines[2] = f"ng: 0\n"
            lines[3] = f"ok_percent: 0\n"
        with open(test_result_path, "w") as f:
            f.writelines(lines)
        self.get_result_file(0)

    def lineedit_count_lose_focus(self, lineedit):
        self.current_recorded_count = int(lineedit.text())
        save_recorded_data_to_json(
            self.lineedit_type.text(),
            self.lineedit_count.text(),
        )
        lineedit.clearFocus()
        if lineedit.text() == "":
            result_count, _ = LoadUiConfig.load_recorded_num_from_json(self.default_logger)
            lineedit.setText(str(result_count))

    def lineedit_type_lose_focus(self, lineedit):
        save_recorded_data_to_json(
            self.lineedit_type.text(),
            self.lineedit_count.text(),
        )
        lineedit.clearFocus()
        if lineedit.text() == "":
            last_recorded_info = LoadUiConfig().load_last_recorded_info(self.default_logger)
            lineedit.setText(str(last_recorded_info.get("product_model", "S004-1")))

    def validate_count(self, lineedit):
        """
            Validates the count input from the user.

            This method checks if the user input in the lineedit is a valid number. If the input is not a number,
            it restores the previously recorded number. If the input is valid, it updates the recorded number and saves
        it to a file.

            Parameters:
            lineedit (QLineEdit): The QLineEdit object containing the user's count input.
        """
        count = lineedit.text()
        result_count, result_scanner_barcode = LoadUiConfig.load_recorded_num_from_json(self.default_logger)
        reg = r"^[0-9]*$"
        if not re.match(reg, count):
            lineedit.setText(str(result_count))
        if count == "":
            lineedit.setText(str(result_count))

    def is_clicked_tcp(self):
        if self.scanner_tcp.isChecked():
            self.ip_edit.setEnabled(True)
            self.port_edit.setEnabled(True)
            self.tcp_ip, self.tcp_port = LoadUiConfig.get_tcp_config()
            if hasattr(self, "tcp_server") and SequenceWindow.tcp_server:
                SequenceWindow.tcp_server.stop()
                SequenceWindow.tcp_server = None
            SequenceWindow.tcp_server = TcpServer(host=self.tcp_ip, port=self.tcp_port, callback=self.deal_package)
            SequenceWindow.tcp_server.start()
        else:
            self.ip_edit.setEnabled(False)
            self.port_edit.setEnabled(False)
            if hasattr(self, "tcp_server") and SequenceWindow.tcp_server:
                SequenceWindow.tcp_server.stop()
                SequenceWindow.tcp_server = None

    @staticmethod
    def deal_package(info):
        """
        info: {
                  "RequestType": "0-9999",
                  "RequestContent": {
                    "User": "Alice",
                    "Action": "ScanBarcode",
                    "label": "NG"
                  },
                  "IsSync": false,
                  "Timestamp": "2025-04-09T16:30:00"
              }
        """
        ok, data = check_tcp_msg_format(info)
        if not ok:
            return data
        request_type = int(data.get("RequestType"))
        request_content = data.get("RequestContent", {})
        is_sync = data.get("IsSync")
        timestamp = data.get("Timestamp")
        request_id = f"{request_type}@{timestamp}"
        if request_id == SequenceWindow.tcp_server.request_id:
            return "pass"
        else:
            SequenceWindow.tcp_server.request_id = request_id
        # allocating task
        if request_type == RequestTypeEnum.RUN_TEST.value:
            label = request_content.get("Label", "not_labeled")
            sign.run_test_sign.emit(label)
        return "ok"

    def scan_barcode(self, device):
        barcode = self.barcode_scanner.read_raw_data(device)
        if barcode:
            sign.signal_emitter.emit(barcode)

    def update_recorded_signal_info_to_db(self):
        new_file_path = FileOps.move_wav_to_dir(self.recorded_path, self.recorded_signal_info["labels"])
        old_file_path = self.recorded_signal_info["file_path"]
        self.recorded_signal_info["file_path"] = new_file_path.replace(DEFAULT_DIR, "")
        save_code, msg = RecordingManager().update_audio_label(self.recorded_signal_info, old_file_path)
        if save_code == error_code.OK:
            self.default_logger.info("Recorded signal successfully updated.")
        else:
            self.default_logger.error("Failed to update recorded signal.")

    def update_recorded_label_in_test_mode(self, label: str):
        self.default_ed_result = label
        if label == "OK":
            self.recorded_signal_info["labels"] = "OK"
        elif label == "NG":
            self.recorded_signal_info["labels"] = "NG"

    def test_insert_data_into_db(self):
        # self.update_recorded_signal_info_to_db()
        self.player_status_flag = False
        # self.replayer_btn.setDisabled(True)
        # self.data_btn.setEnabled(False)
        # self.default_ed_result = None
        self.default_ed = None

    def on_clicked_player_btn(self, label="not_labeled"):
        if self.default_ed_result:
            self.update_recorded_signal_info_to_db()
            self.set_result_file(0, self.default_ed_result)
            self.get_result_file(0)

        self.clicked_player_flag = True
        self.start_this_play(label)

    def start_this_play(self, label="not_labeled"):
        if self.clicked_player_flag is False:
            if SequenceWindow.tcp_server.client_address is None:
                QMessageBox.warning(self, "提示", "TCP链接异常")
                return

        self.judge_play_and_record(label)
        self.current_recorded_count += 1
        self.lineedit_count.setText(str(self.current_recorded_count))
        save_recorded_data_to_json(
            self.lineedit_type.text(),
            self.lineedit_count.text(),
        )

        if self.clicked_player_flag is True:
            self.clicked_player_flag = False
        elif self.clicked_player_flag is False:
            if self.scanner_tcp.isChecked():
                TempTcpClient(
                    SequenceWindow.tcp_server.client_address[0], SequenceWindow.tcp_server.client_address[1], "finish"
                )

    def checked_work_status_message(self):
        if not self.sequence_config:
            QMessageBox.warning(self, "提示", "未找到录音模式，请在功能-测试队列中配置")
            return

        if not self.mic:
            QMessageBox.warning(self, "提示", "未找到麦克风，请在硬件中设置")
            return

    def reset_work_pram(self, label):
        self.data_struct.clear_data()
        self.recorded_path, self.recorded_signal_info = get_recorded_info(
            self.lineedit_type.text(), self.lineedit_count.text(), label
        )
        acq_detail = self.sequence_config[0]["seq1"]["acq"]["detail"]
        total_time = float(acq_detail.get("total_time", 5.0))
        sample_rate = self.data_struct.sample_rate
        _, recorded_dict = LoadUiConfig.get_rec_and_play_dict_base_sequence_dict(self.data_struct, total_time)

        return recorded_dict, sample_rate

    def judge_play_and_record(self, label="not_labeled"):
        if self.checked_work_status_message():
            self.update_player_btn_is_paused()
            return
        if self.analysis_config["default_ed"] is None:
            self.update_player_btn_is_paused()
            QMessageBox.warning(self, "提示", "未设置默认评判项，请在测试队列中添加事件检测默认项！")
            return

        self.update_player_btn_is_playing()
        if self.player_status_flag:
            self.line_graph_top.clear()
            self.line_graph_bottom.clear()
        self.player_status_flag = True
        QApplication.processEvents()

        recorded_dict, sample_rate = self.reset_work_pram(label)

        if self.sequence_config[0]["seq1"]["acq"]["mode"] in ["RECORD_ONLY"]:
            record_without_play(recorded_dict, self.recorded_path, self.recorded_signal_info)
        self.plot_line_graph(self.data_struct.store_wave_data, sample_rate)

        self.data_btn.setEnabled(True)
        self.replayer_btn.setEnabled(True)

        if self.analysis_config["auto_analysis"]:
            self.run()
        self.update_player_btn_is_paused()

    def instance_analysis_class(self, key, type, params):
        """
        Instantiates and configures an analysis class based on the given type and parameters,
        and adds it to the analysis window list.

        Args:
            type (str): The type identifier of the analysis class, used to retrieve the corresponding class from the class mapping.
            params (dict): Configuration parameters for the analysis class, which will be passed to the instantiated class object.

        Returns:
            None: This function does not return a value but adds the instantiated class object to the self.analysis_window list.
        """
        class_mapping = get_class_mapping()
        if type in class_mapping.keys():
            cls_map = class_mapping.get(type)
            if cls_map:
                class_instance = cls_map(key)
                if self.analysis_config["default_ed"] == key:
                    self.default_ed = class_instance
                    class_instance.is_default_flag = True
                if type == "Spec":
                    self.spec_list.append(class_instance)
                class_instance.deviation_value = self.deviation_value
                class_instance.analysis_config = params
                self.analysis_window.append(class_instance)

    def run(self):
        """
        Executes the analysis tasks and displays the analysis windows.

        This method initializes the analysis windows based on the configuration and creates corresponding
        analysis instances according to the analysis types specified in the configuration. It then performs
        the respective calculations for each instance and displays the windows. The window positions are
        adjusted based on the screen size to ensure they do not overlap.
        """
        self.analysis_window = []
        width = int((self.screen().size().width() - 400) / 3)
        height = int((self.screen().size().height() - 400) / 3)
        if self.analysis_config:
            item_sort_list = self.analysis_config.get("display_sequence", [])
            for key in item_sort_list:
                key_config = self.analysis_config.get(key)
                if not isinstance(key_config, dict):
                    continue
                item_type = key_config.get("type")
                self.instance_analysis_class(key, item_type, key_config)
            for instance in self.analysis_window:
                if self.mode == "test":
                    if instance is self.default_ed:
                        continue
                if hasattr(instance, "calculate_spl"):
                    instance.calculate_spl()
                    instance.show()
                elif hasattr(instance, "calculate_spec"):
                    instance.calculate_spec()
                    instance.show()
                elif hasattr(instance, "calculate_peak_detection"):
                    instance.calculate_peak_detection()
                    instance.show()
                elif hasattr(instance, "calculate_loose_particle"):
                    instance.calculate_loose_particle()
                    instance.show()
                elif hasattr(instance, "calculate_pattern_match"):
                    instance.calculate_pattern_match()
                    instance.show()
                elif hasattr(instance, "calculate_pipeline_pd_pm"):
                    instance.calculate_pipeline_pd_pm()
                    instance.show()
                if instance in self.spec_list:
                    instance.setGeometry(width, height, 800, 500)
                else:
                    instance.setGeometry(width, height, 600, 500)
                instance.setMinimumSize(QSize(600, 500))
                width += 20
                height += 20
            self.default_ed.calculate_pipeline_pd_pm()
            self.default_ed.show()
            self.default_ed.setGeometry(width, height, 600, 500)
            self.test_insert_data_into_db()

    def get_sequence_config_from_json(self):
        """
        Retrieves the sequence configuration from a JSON file.

        This method attempts to load the sequence configuration from a JSON file by calling the `load_sequence_from_json()` method.
        If the loading is successful and the result is valid, it returns the configuration; otherwise, it returns an empty dictionary.

        Returns:
            dict: The sequence configuration if loading is successful and the result is valid; otherwise, an empty dictionary.
        """
        load_code, result = LoadUiConfig().load_sequence_config_from_json()
        if load_code == error_code.OK and result:
            self.sequence_config = result
            seq = self.sequence_config[0]["seq1"]
            self.analysis_config = seq.get("analysis_list", {})
        else:
            self.analysis_config = dict()

    def plot_line_graph(self, recorded_signal, sample_rate):
        """
        Plot a line graph of the recorded signal.

        Parameters:
        recorded_signal (list or numpy.array): The recorded signal data to be plotted.
        line_graph_top (matplotlib.axes.Axes): The Axes object used for plotting the line graph.
        sample_rate (int or float): The sample rate of the signal, used to calculate the duration of the signal.
        """
        self.line_graph_top.clear()
        signal_duration = np.linspace(0, len(recorded_signal[0]) / sample_rate, len(recorded_signal[0]))
        self.line_graph_top.plot(signal_duration, recorded_signal[0], pen="k")

        self.line_graph_bottom.clear()
        signal_duration = np.linspace(0, len(recorded_signal[1]) / sample_rate, len(recorded_signal[1]))
        self.line_graph_bottom.plot(signal_duration, recorded_signal[1], pen="k")

    def update_player_btn_is_playing(self):
        self.player_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/pause.png"))
        self.player_btn.setIconSize(QSize(35, 35))
        self.player_btn.setDisabled(True)

    def update_player_btn_is_paused(self):
        self.player_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/play.png"))
        self.player_btn.setIconSize(QSize(35, 35))
        self.player_btn.setDisabled(False)

    def paintEvent(self, event):
        painter = QPainter(self)
        width = self.width()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(208, 206, 202))
        painter.drawRect(1, 0, width - 2, 41)
        painter.end()
