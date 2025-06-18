import json
import os
import re
import shutil
import sys
import threading
from datetime import datetime

import librosa
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QSize, Qt, QObject, pyqtSignal
from PyQt5.QtGui import QIcon, QPainter, QColor, QFont
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFrame, QCheckBox, QMessageBox, \
    QStackedWidget
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from base.barcode_scanning_processor import BarcodeScanner
from base.data_struct.data_deal_struct import DataDealStruct
from base.utils.custom_signals import sign
from base.load_audio import load_audio_simple
from base.log_manager import LogManager
from base.recording_management import RecordingManager
from base.soundcard_audio_processor import SoundcardAudioProcessor
from base.tcp_service import TcpServer
from consts import ui_style_const, error_code, model_consts
from consts.action_code import RequestTypeEnum
from consts.running_consts import DEFAULT_DIR
from ui.signal_analysis_window import Spl, Distortion, AI, Frequency, Spectrogram, LooseParticle
from ui.login_window import get_mac_address


class SequenceWindow(QWidget):
    tcp_server = None

    def __init__(self):
        """Initializes the class instance, setting up the user interface and necessary parameters."""
        super().__init__()
        self.data_struct = DataDealStruct()
        self.collect_or_analyse_layout = QHBoxLayout()
        self.recorded_path = None  # Initialize the recorded path variable
        self.refresh_stimulus_flag = None  # Initialize the flag to indicate if stimulus needs refreshing
        # Retrieve stimulus information and signal from configuration
        self.data_struct.stimulus_info, self.data_struct.stimulus_data = self.get_stimulus_from_config()
        self.deviation_value = self.get_mic_deviation_value()  # Get the deviation value from the microphone
        self.analysis_config = self.get_sequence_config_from_json()
        self.init_fft_and_stft_flag()
        self.signal_info = {}  # Initialize an empty dictionary to store signal information
        self.analysis_window = []
        self.default_ai = None
        self.default_ai_result = None
        self.sequence_layout = QVBoxLayout()
        self.player_btn = QPushButton()
        self.replayer_btn = QPushButton()
        self.data_btn = QPushButton()
        self.player_status_flag = False
        self.scanner_barcode_thread = None
        self.barcode_scanner = BarcodeScanner()
        self.scanner_emitter = ScannerEmitter()
        self.vendor_id = None
        self.product_id = None
        self.recorded_signal_info = {}
        self.ip_format = True
        self.port_format = True
        self.tcp_ip = None
        self.tcp_port = None
        self.get_tcp_config()
        self.mode = None
        # Set up the default logger for logging messages
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

        self.ok_btn.clicked.connect(self.clicked_ok_or_ng)
        self.ng_btn.clicked.connect(self.clicked_ok_or_ng)
        sign.run_test_sign.connect(self.clicked_player_btn, Qt.AutoConnection)
        sign.get_result_file_sign.connect(self.get_result_file, Qt.AutoConnection)
        sign.set_result_file_sign.connect(self.set_result_file, Qt.AutoConnection)
        sign.test_insert_data_into_db_sign.connect(self.update_recorded_label_in_test_mode, Qt.AutoConnection)
        sign.update_mode_display_sign.connect(self.update_mode_display, Qt.AutoConnection)
        self.update_mode_display(0)
        self.setStyleSheet(ui_style_const.qcombobox_stytle +
                           ui_style_const.qpushbutton_stytle +
                           ui_style_const.qlineedit_stytle +
                           ui_style_const.qframe_stytle +
                           ui_style_const.qlabel_stytle +
                           ui_style_const.qcheckbox_stytle)

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
        self.player_btn.setToolTip("播放")
        self.player_btn.setStyleSheet(ui_style_const.toolbar_button_stytle)
        self.player_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/play.png"))
        self.player_btn.setIconSize(QSize(35, 35))
        self.player_btn.clicked.connect(self.clicked_player_btn)

        self.replayer_btn.setFixedSize(100, 40)
        self.replayer_btn.setToolTip("重播")
        self.replayer_btn.setDisabled(True)
        self.replayer_btn.setStyleSheet(ui_style_const.toolbar_button_stytle)
        self.replayer_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/replay.png"))
        self.replayer_btn.setIconSize(QSize(30, 30))
        self.replayer_btn.clicked.connect(self.clicked_player_btn)

        self.data_btn.setFixedSize(100, 40)
        self.data_btn.setToolTip("分析")
        self.data_btn.setEnabled(False)
        self.data_btn.setStyleSheet(ui_style_const.toolbar_button_stytle)
        self.data_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/data.png"))
        self.data_btn.setIconSize(QSize(35, 35))

        self.data_btn.clicked.connect(self.run)

        type_label = QLabel(" 型 号： ")
        data = self.load_last_recorded_info()
        if data:
            product_model = data.get("product_model", 'S004-1')
        else:
            product_model = "S004-1"
        type_label.setFixedHeight(40)
        self.lineedit_type = QLineEdit(product_model)
        self.lineedit_type.setFixedHeight(35)
        self.lineedit_type.setAlignment(Qt.AlignCenter)
        label_count = QLabel(" 计 数： ")
        label_count.setFixedHeight(40)

        result, _ = self.load_recorded_num_from_json()
        if result is None:
            current_recorded_count = 1
        else:
            current_recorded_count = result
        self.lineedit_count = QLineEdit(str(current_recorded_count))
        self.lineedit_count.setFixedHeight(35)
        self.lineedit_count.setAlignment(Qt.AlignCenter)
        self.lineedit_count.editingFinished.connect(lambda: self.lineedit_lose_focus(self.lineedit_count))
        self.lineedit_count.returnPressed.connect(lambda: self.validate_count(self.lineedit_count, True))

        self.barcode_scanner_box = QCheckBox("S/N:  ", self)
        self.barcode_scanner_box.setChecked(False)
        self.barcode_scanner_box.stateChanged.connect(self.clicked_scanner)

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

        self.lineedit_s_or_n = QLineEdit(self)
        self.lineedit_s_or_n.setDisabled(True)
        self.lineedit_s_or_n.setFixedHeight(35)
        self.lineedit_s_or_n.setAlignment(Qt.AlignCenter)
        self.lineedit_s_or_n.editingFinished.connect(lambda: self.validate_count(self.lineedit_s_or_n, False))
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
        layout.addWidget(self.barcode_scanner_box)
        layout.addWidget(self.lineedit_s_or_n)
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
        pattern = r'^((25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)$'
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
            self.write_tcp_config(self.ip_edit.text(), self.port_edit.text())
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
        self.line_graph = pg.PlotWidget()
        self.line_graph.setBackground('white')
        left_area = self.create_left_layout()
        self.line_graph.setLabel('left', 'Amplitude(V)')
        self.line_graph.setLabel('bottom', 'Time(s)')
        self.line_graph.showGrid(x=True, y=True)

        layout.addLayout(left_area, stretch=1)
        layout.addSpacing(20)
        layout.addWidget(self.line_graph, stretch=8)
        layout.setContentsMargins(40, 20, 40, 20)
        layout.setSpacing(30)
        return layout

    def create_left_layout(self):
        layout = QVBoxLayout()

        mode_label = QLabel("模式：")
        self.model_button = QHBoxLayout()
        self.model_button.setSpacing(0)
        self.test_btn = QPushButton("测试")
        self.test_btn.setFixedSize(100, 35)
        self.test_btn.setFont(QFont("Arial", 14))
        self.test_btn.clicked.connect(lambda: self.update_mode_display(0))
        self.mark_btn = QPushButton("标记")
        self.mark_btn.setFixedSize(100, 35)
        self.mark_btn.setFont(QFont("Arial", 14))
        self.mark_btn.clicked.connect(lambda: self.update_mode_display(1))
        self.model_button.addWidget(self.test_btn)
        self.model_button.addWidget(self.mark_btn)
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

        model_layout = QHBoxLayout()
        self.model_label = QLabel("当前模型：")
        self.model_line_edit = QLineEdit()
        self.model_line_edit.setFixedHeight(35)
        self.model_line_edit.setFixedWidth(130)
        self.model_line_edit.setDisabled(True)
        self.model_line_edit.setAlignment(Qt.AlignCenter)
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.model_line_edit)

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
        self.reset_btn.setStyleSheet(ui_style_const.qpushbutton_stytle)
        reset_btn_layout.addWidget(self.reset_btn)
        reset_btn_layout.addStretch()
        self.reset_btn.clicked.connect(self.reset_test_reord)
        test_layout.addLayout(total_layout, stretch=1)
        test_layout.addLayout(ok_layout, stretch=1)
        test_layout.addLayout(ng_layout, stretch=1)
        test_layout.addLayout(yield_layout, stretch=1)
        test_layout.addLayout(model_layout, stretch=1)
        test_layout.addLayout(datatime_layout, stretch=1)
        test_layout.addLayout(reset_btn_layout, stretch=1)
        self.test_page.setLayout(test_layout)

        # --- page 2：mark ---
        self.mark_page = QWidget()
        mark_layout = QVBoxLayout()

        mark_total_layout = QHBoxLayout()
        self.mark_total_label = QLabel("总数：")
        self.mark_total_edit = QLineEdit("0")
        self.mark_total_edit.setFixedHeight(35)
        self.mark_total_edit.setFixedWidth(130)
        self.mark_total_edit.setDisabled(True)
        self.mark_total_edit.setAlignment(Qt.AlignCenter)
        mark_total_layout.addWidget(self.mark_total_label)
        mark_total_layout.addWidget(self.mark_total_edit)

        mark_ok_layout = QHBoxLayout()
        self.mark_ok_label = QLabel("OK数：")
        self.mark_ok_edit = QLineEdit("0")
        self.mark_ok_edit.setFixedHeight(35)
        self.mark_ok_edit.setFixedWidth(130)
        self.mark_ok_edit.setDisabled(True)
        self.mark_ok_edit.setAlignment(Qt.AlignCenter)
        mark_ok_layout.addWidget(self.mark_ok_label)
        mark_ok_layout.addWidget(self.mark_ok_edit)

        mark_ng_layout = QHBoxLayout()
        self.mark_ng_label = QLabel("NG数：")
        self.mark_ng_edit = QLineEdit("0")
        self.mark_ng_edit.setFixedHeight(35)
        self.mark_ng_edit.setFixedWidth(130)
        self.mark_ng_edit.setDisabled(True)
        self.mark_ng_edit.setAlignment(Qt.AlignCenter)
        mark_ng_layout.addWidget(self.mark_ng_label)
        mark_ng_layout.addWidget(self.mark_ng_edit)

        ok_layout = QHBoxLayout()
        ok_layout.addStretch()
        self.ok_btn = QPushButton(" OK ")
        ok_layout.addWidget(self.ok_btn)
        ok_layout.addStretch()

        ng_layout = QHBoxLayout()
        ng_layout.addStretch()
        self.ng_btn = QPushButton(" NG ")
        ng_layout.addWidget(self.ng_btn)
        ng_layout.addStretch()
        self.ok_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/lvseyuan.png"))
        self.ok_btn.setStyleSheet(ui_style_const.sequence_qpushbutton_stytle)
        self.ok_btn.setFixedSize(180, 80)
        self.ok_btn.setIconSize(QSize(24, 24))
        self.ng_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/hongseyuan.png"))
        self.ng_btn.setStyleSheet(ui_style_const.sequence_qpushbutton_stytle)
        self.ng_btn.setFixedSize(180, 80)
        self.ng_btn.setIconSize(QSize(24, 24))

        mark_layout.addLayout(mark_total_layout, stretch=1)
        mark_layout.addLayout(mark_ok_layout, stretch=1)
        mark_layout.addLayout(mark_ng_layout, stretch=1)
        mark_layout.addLayout(ok_layout, stretch=2)
        mark_layout.addLayout(ng_layout, stretch=2)
        self.mark_page.setLayout(mark_layout)

        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(self.test_page)
        self.stacked_widget.addWidget(self.mark_page)

        layout.addLayout(mode_layout)
        layout.addWidget(separator_line)
        layout.addWidget(self.stacked_widget)
        layout.addStretch()

        self.init_result_files()
        return layout
    
    def init_fft_and_stft_flag(self):
        model_item_list = self.analysis_config.get("display_sequence", "")
        for item_name in model_item_list:
            self.data_struct.add_stft_or_fft_count(self.analysis_config[item_name]["type"]) 

    def init_result_files(self):
        current_time = datetime.now().strftime("%Y-%m-%d")
        test_result_path = DEFAULT_DIR + f"log/test_result_log/{current_time}.dat"
        test_result_template = (
            f"total: 0\n"
            f"ok: 0\n"
            f"ng: 0\n"
            f"ok_percent: 0\n"
            f"current_model: xxx\n"
            f"datatime: {current_time}\n"
        )
        if not os.path.exists(test_result_path):
            os.makedirs(os.path.dirname(test_result_path), exist_ok=True)
            with open(test_result_path, 'w') as f:
                f.write(test_result_template)

        mark_result_path = DEFAULT_DIR + "ui/ui_config/mark_result.json"
        mark_result_template = {
            "total": 0,
            "ok": 0,
            "ng": 0,
            "datatime": current_time
        }
        if not os.path.exists(mark_result_path):
            os.makedirs(os.path.dirname(mark_result_path), exist_ok=True)
            with open(mark_result_path, 'w') as f:
                json.dump(mark_result_template, f, indent=4)
        else:
            self.set_result_file(1, "init", None)

    @staticmethod
    def ensure_test_result_file():
        config_file_path = DEFAULT_DIR + "ui/ui_config/analysis_temp_config.json"
        with open(config_file_path, 'r') as f:
            default_config = json.load(f)
            default_ai_model = default_config["default_ai"]
            if default_ai_model:
                analyse_model_name = default_config.get(default_ai_model, None).get("analyse_model_name", None)
            else:
                analyse_model_name = "null"
        current_time = datetime.now().strftime("%Y-%m-%d")
        test_result_path = DEFAULT_DIR + f"log/test_result_log/{current_time}.dat"
        if not os.path.exists(test_result_path):
            os.makedirs(os.path.dirname(test_result_path), exist_ok=True)
            with open(test_result_path, 'w') as f:
                f.write(
                    f"total: 0\n"
                    f"ok: 0\n"
                    f"ng: 0\n"
                    f"ok_percent: 0\n"
                    f"current_model: {analyse_model_name}\n"
                    f"datatime: {current_time}\n"
                )

    def get_result_file(self, index):
        current_time = datetime.now().strftime("%Y-%m-%d")
        if index == 0:
            self.ensure_test_result_file()
            test_result_path = DEFAULT_DIR + f"log/test_result_log/{current_time}.dat"
            with open(test_result_path, 'r') as f:
                lines = f.readlines()
                total = lines[0].split(':')[1].strip()
                ok = lines[1].split(':')[1].strip()
                ng = lines[2].split(':')[1].strip()
                ok_percent = lines[3].split(':')[1].strip()
                current_model = lines[4].split(':')[1].strip()
                datatime = lines[5].split(':')[1].strip()
                self.total_line_edit.setText(total)
                self.ok_line_edit.setText(ok)
                self.ng_line_edit.setText(ng)
                self.yield_line_edit.setText(ok_percent)
                self.model_line_edit.setText(current_model)
                self.model_line_edit.setCursorPosition(0)
                self.datatime_line_edit.setText(datatime)
        if index == 1:
            mark_result_path = DEFAULT_DIR + "ui/ui_config/mark_result.json"
            with open(mark_result_path, 'r') as f:
                data = json.load(f)
                self.mark_total_edit.setText(str(data["total"]))
                self.mark_ok_edit.setText(str(data["ok"]))
                self.mark_ng_edit.setText(str(data["ng"]))

    def set_result_file(self, index, params, analyse_model_name):
        current_time = datetime.now().strftime("%Y-%m-%d")
        if index == 0:
            self.ensure_test_result_file()
            test_result_path = DEFAULT_DIR + f"log/test_result_log/{current_time}.dat"
            with open(test_result_path, 'r') as f:
                lines = f.readlines()
                total = int(lines[0].split(':')[1].strip())
                ok = int(lines[1].split(':')[1].strip())
                ng = int(lines[2].split(':')[1].strip())

                if params == "OK":
                    total +=1
                    ok +=1
                elif params == "NG":
                    total += 1
                    ng += 1
                lines[0] = f"total: {total}\n"
                lines[1] = f"ok: {ok}\n"
                lines[2] = f"ng: {ng}\n"
                ok_percent = round(ok / total * 100, 2) if total > 0 else 0
                lines[3] = f"ok_percent: {ok_percent}%\n"
                if analyse_model_name:
                    lines[4] = f"current_model: {analyse_model_name}\n"
            with open(test_result_path, 'w') as f:
                f.writelines(lines)
        if index == 1:
            mark_result_path = DEFAULT_DIR + "ui/ui_config/mark_result.json"
            with open(mark_result_path, 'r') as f:
                data = json.load(f)
            current_date = datetime.now().strftime("%Y-%m-%d")
            if params == "init":
                if data["datatime"] != current_date:
                    data["total"] = 0
                    data["ok"] = 0
                    data["ng"] = 0
                    data["datatime"] = current_date
            elif params == "OK":
                data["total"] += 1
                data["ok"] += 1
            elif params == "NG":
                data["total"] += 1
                data["ng"] += 1
            with open(mark_result_path, 'w') as f:
                json.dump(data, f, indent=4)

    def update_mode_display(self, index):
        if index == 0:
            self.stacked_widget.setCurrentIndex(0)
            self.get_result_file(0)
            self.mode = "test"
            config_file_path = DEFAULT_DIR + "ui/ui_config/analysis_temp_config.json"
            with open(config_file_path, 'r') as f:
                default_config = json.load(f)
                default_ai_model = default_config["default_ai"]
                if default_ai_model:
                    analyse_model_name = default_config.get(default_ai_model, None).get("analyse_model_name", None)
                    self.model_line_edit.setText(analyse_model_name)
                    self.model_line_edit.setCursorPosition(0)
                    self.test_btn.setStyleSheet("background-color: #007BFF; color: white; border: none;")
                    self.mark_btn.setStyleSheet("background-color: #E0E0E0; color: #666666; border: none;")
                    self.test_btn.setEnabled(False)
                    self.mark_btn.setEnabled(True)
                else:
                    self.update_mode_display(1)
        else:
            self.stacked_widget.setCurrentIndex(1)
            self.get_result_file(1)
            self.mode = "mark"
            self.test_btn.setStyleSheet("background-color: #E0E0E0; color: #666666; border: none;")
            self.mark_btn.setStyleSheet("background-color: #007BFF; color: white; border: none;")
            self.mark_btn.setEnabled(False)
            self.test_btn.setEnabled(True)

    def reset_test_reord(self):
        current_time = datetime.now().strftime("%Y-%m-%d")
        test_result_path = DEFAULT_DIR + f"log/test_result_log/{current_time}.dat"
        with open(test_result_path, 'r') as f:
            lines = f.readlines()
            lines[0] = f"total: 0\n"
            lines[1] = f"ok: 0\n"
            lines[2] = f"ng: 0\n"
            lines[3] = f"ok_percent: 0\n"
        with open(test_result_path, 'w') as f:
            f.writelines(lines)
        self.get_result_file(0)

    def lineedit_lose_focus(self, lineedit):
        lineedit.clearFocus()
        if lineedit.text() == "":
            result_count, _ = self.load_recorded_num_from_json()
            lineedit.setText(str(result_count))

    def validate_count(self, lineedit, is_s_or_n: bool):
        """
            Validates the count input from the user.

            This method checks if the user input in the lineedit is a valid number. If the input is not a number,
            it restores the previously recorded number. If the input is valid, it updates the recorded number and saves
        it to a file.

            Parameters:
            lineedit (QLineEdit): The QLineEdit object containing the user's count input.
        """
        # lineedit.clearFocus()
        s_or_n_count = lineedit.text()
        # Load the previously recorded number from a text file
        result_count, result_scanner_barcode = self.load_recorded_num_from_json()
        # Define a regular expression to match numbers
        reg = None
        if is_s_or_n:
            reg = r'^[0-9]*$'
        else:
            reg = r'^[0-9a-zA-Z]*$'
        # Check if the user input matches the regular expression
        if not re.match(reg, s_or_n_count):
            # If the input is not a number, restore the previously recorded number
            if is_s_or_n:
                lineedit.setText(str(result_count))
            else:
                lineedit.setText(str(result_scanner_barcode))
        elif s_or_n_count != "":
            # If the input is a number, Open the file and write the current recorded count and date
            if is_s_or_n:
                self.lineedit_s_or_n.setText("")
            self.save_recorded_num_to_json()
        if s_or_n_count == "":
            if is_s_or_n:
                lineedit.setText(str(result_count))
            else:
                lineedit.setText(str(result_scanner_barcode))

    def scanner_barcode_process(self):
        device = self.get_match_hid_device()
        if device:
            if self.scanner_barcode_thread is None or not self.scanner_barcode_thread.is_alive():
                self.scanner_barcode_thread = threading.Thread(target=self.scan_barcode,
                                                               args=(device,))
                self.scanner_emitter.signal_emitter.connect(self.on_barcode_received)
                self.scanner_barcode_thread.start()

    def write_tcp_config(self, ip, port):
        file_path = DEFAULT_DIR + "ui/ui_config/tcp_config.txt"
        if ip:
            self.tcp_ip = ip
        if port:
            self.tcp_port = port
        try:
            with open(file_path, 'w') as f:
                f.write(f"ip = {self.tcp_ip}\n")
                f.write(f"port = {self.tcp_port}\n")
            self.default_logger.info(f"write_tcp_config_success: {file_path}")
        except Exception as e:
            self.default_logger.error(f"write_tcp_config_error: {e}")

    def get_tcp_config(self):
        file_path = DEFAULT_DIR + "ui/ui_config/tcp_config.txt"
        with open(file_path, 'r') as f:
            config_data = f.readlines()
            ip = config_data[0].split('=')[1].strip()
            port_text = config_data[1].split('=')[1].strip()
            port = int(port_text)
            self.tcp_ip = ip
            self.tcp_port = port

    def is_clicked_tcp(self):
        if self.scanner_tcp.isChecked():
            self.ip_edit.setEnabled(True)
            self.port_edit.setEnabled(True)
            self.barcode_scanner_box.setEnabled(False)
            self.get_tcp_config()
            if hasattr(self, 'tcp_server') and SequenceWindow.tcp_server:
                SequenceWindow.tcp_server.stop()
                SequenceWindow.tcp_server = None
            SequenceWindow.tcp_server = TcpServer(host=self.tcp_ip, port=self.tcp_port, callback=self.deal_package)
            SequenceWindow.tcp_server.start()
        else:
            self.ip_edit.setEnabled(False)
            self.port_edit.setEnabled(False)
            self.barcode_scanner_box.setEnabled(True)
            if hasattr(self, 'tcp_server') and SequenceWindow.tcp_server:
                SequenceWindow.tcp_server.stop()
                SequenceWindow.tcp_server = None

    def generate_request_id(self, request_type,timestamp):
        """
        Args:
            request_type: int
            timestamp: str
        Returns:
           "102@2025-04-11T10:06:47"
        """
        return f"{request_type}@{timestamp}"

    def deal_package(self, info):
        """
        info: {
                  "RequestType": "0-9999",
                  "RequestContent": {
                    "User": "Alice",
                    "Action": "ScanBarcode"
                  },
                  "IsSync": false,
                  "Timestamp": "2025-04-09T16:30:00"
              }
        """
        ok, data = self.check_format(info)
        if not ok:
            return data
        request_type = int(data.get("RequestType"))
        is_sync = data.get("IsSync")
        timestamp = data.get("Timestamp")
        request_id = self.generate_request_id(request_type,timestamp)
        if request_id == SequenceWindow.tcp_server.request_id:
            return "pass"
        else:
            SequenceWindow.tcp_server.request_id = request_id
        # allocating task
        if request_type == RequestTypeEnum.RUN_TEST.value:
            sign.run_test_sign.emit()
        return "ok"

    def check_format(self, info):
        try:
            data = json.loads(info)
        except json.JSONDecodeError as e:
            return False, "error, json format error"
        req_type = int(data.get("RequestType"))
        is_sync = data.get("IsSync")
        timestamp = data.get("Timestamp")
        if req_type not in [rte.value for rte in RequestTypeEnum]:
            return False, "error, RequestType error"
        if not isinstance(is_sync, bool):
            return False, "error, IsSync type error"
        if not timestamp or not isinstance(timestamp, str):
            return False, "error, Timestamp type error "
        return True, data

    def scan_barcode(self, device):
        barcode = self.barcode_scanner.read_raw_data(device)
        if barcode:
            self.scanner_emitter.signal_emitter.emit(barcode)

    def on_barcode_received(self, barcode):
        if barcode:
            self.lineedit_s_or_n.setText(barcode)
            try:
                self.clicked_player_btn()
            except Exception as e:
                self.scanner_popup()
                self.default_logger.error(f"An error message occurred in the analysis window. {e}")
            self.scanner_emitter.signal_emitter.disconnect(self.on_barcode_received)
            if self.scanner_barcode_thread and self.scanner_barcode_thread.is_alive():
                self.scanner_barcode_thread.join()
            self.scanner_barcode_thread = None

    def clicked_scanner(self):
        if self.barcode_scanner_box.isChecked():
            self.scanner_tcp.setEnabled(False)
            self.lineedit_s_or_n.setEnabled(True)
            self.scanner_barcode_process()
        else:
            self.scanner_tcp.setEnabled(True)
            self.lineedit_s_or_n.clear()
            self.lineedit_s_or_n.setDisabled(True)
            self.barcode_scanner.stop_scanning()
            self.scanner_barcode_thread = None
            try:
                self.scanner_emitter.signal_emitter.disconnect(self.on_barcode_received)
            except Exception as e:
                self.default_logger.error(e)

    def get_match_hid_device(self):
        hid_params = self.load_scanner_hid_params()
        if hid_params:
            vendor_id, product_id = hid_params
            self.vendor_id = int(vendor_id, 16)
            self.product_id = int(product_id, 16)
            device = self.barcode_scanner.find_scanner(self.vendor_id, self.product_id)
            return device
        return None

    def load_scanner_hid_params(self):
        file_path = DEFAULT_DIR + "configs/scanner_barcode_config/scanner_hid_config.txt"
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                vendor_id = lines[1].strip()
                product_id = lines[3].strip()
                return vendor_id, product_id
        except Exception as e:
            self.default_logger.error(f"Failed to read the config params of the scanner hid. {e}")
            return None

    def scanner_popup(self):
        error_msg = QMessageBox(self)
        error_msg.setIcon(QMessageBox.Warning)
        error_msg.setText("分析报错，详情请查看日志！")
        error_msg.setWindowTitle("分析报错")
        error_msg.setStandardButtons(QMessageBox.Ok)
        error_msg.exec_()

    def clicked_ok_or_ng(self):
        """
            Handles the logic when the OK or NG button is clicked.

            This method performs several actions in response to a user clicking the OK or NG button:
            1. Saves the current recorded count to a text file.
            2. Updates the displayed recorded count in the UI.
            3. Inserts the recorded data into the database with a label based on which button was clicked (OK/NG).
            4. Resets the player status flag and updates the player icon accordingly.
            5. Clears the signal information and waveform graph.
            6. Disables the replay and data buttons to prevent further actions until the next recording.

            Parameters:
                self: The instance of the class containing this method.
        """
        if not self.player_status_flag:
            QMessageBox.warning(self, "警告", "请先录制声音！")
            return
        current_recorded_count = self.save_recorded_num_to_json("ok_ng")
        self.lineedit_count.setText(str(current_recorded_count))
        self.insert_data_into_db()
        self.mark_result()
        self.player_status_flag = False
        self.update_player_icon()
        self.signal_info.clear()
        # self.analyse_layout.signal_info = self.signal_info
        # self.analyse_layout.close()
        # self.clear_plg()
        self.lineedit_s_or_n.clear()
        self.line_graph.clear()
        self.replayer_btn.setDisabled(True)
        self.data_btn.setEnabled(False)
        self.default_ai_result = None
        self.default_ai = None
        self.clicked_scanner()

    def get_stimulus_from_config(self):
        """
            Retrieves stimulus information and signal from the configuration.

            This function attempts to load stimulus information from a JSON configuration file and then loads the audio
        signal based on the configuration.
            If the loading is successful and the configuration is valid, it parses and returns the stimulus information
        and the audio signal.
            If the loading fails or the configuration is invalid, it returns None.

            Returns:
                tuple: A tuple containing the stimulus information dictionary and the audio signal.
                    Returns (None, None) if the loading fails or the configuration is invalid.
        """
        load_code, result = self.load_stimulus_from_json()
        if load_code == error_code.OK and result:
            info = result.get("stimulus_info")
            path = DEFAULT_DIR + result.get("stimulus_signal_path")
            stimulus, _ = load_audio_simple(path, info["sample_rate"])
            return info, stimulus
        else:
            return None, None

    @staticmethod
    def load_stimulus_from_json():
        """
            Load stimulus configuration from a JSON file.

            This method attempts to load stimulus configuration from a predefined JSON file path and parse the
        configuration into a dictionary.
            If the JSON file does not exist, it returns an appropriate error code and message.

            Returns:
                tuple: A tuple containing the error code and configuration data or error message.
                    If the operation is successful, the error code is error_code.OK, and the configuration data is the
                 parsed dictionary.
                    If the operation fails, the error code is error_code.INVALID_DATA_LOADING, and the error message is a string.
        """
        json_file_path = DEFAULT_DIR + "ui/ui_config/stimulus.json"
        if not os.path.exists(json_file_path):
            return error_code.INVALID_DATA_LOADING, "This json file does not exist."
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
            return error_code.OK, data

    def save_recorded_num_to_json(self, start_position=None):
        """
            Save the recorded number to a text file.

            This function writes the current recorded number and the current date to a specified text file.
            If the file exists and the date matches, it updates the recorded number.
            If the file does not exist or the date does not match, it creates a new file and writes the initial recorded number.
        """
        dir_path = DEFAULT_DIR + 'ui/ui_config/'
        file_path = dir_path + "recorded_number.json"
        current_time = datetime.now().strftime("%Y-%m-%d")
        check_flag, count = self.check_datetime(current_time)
        if check_flag:
            current_recorded_count = int(count) + 1
        else:
            current_recorded_count = 2
        if self.lineedit_count.text() == "":
            self.lineedit_count.setText(str(count))
        if count != int(self.lineedit_count.text()):
            current_recorded_count = int(self.lineedit_count.text())
            if start_position == "ok_ng":
                current_recorded_count = current_recorded_count + 1
        data = {
            "product_model": self.lineedit_type.text(),
            "current_recorded_count": current_recorded_count,
            "scanner_barcode": self.lineedit_s_or_n.text(),
            "scanner_barcode_check": self.barcode_scanner_box.isChecked(),
            "datetime": current_time
        }
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        return current_recorded_count

    def load_last_recorded_info(self):
        """
            Load the recorded number from a text file.

            This method reads a recorded number and the last recorded date from a specified text file.
            If the file exists and the last recorded date matches the current date, it returns the recorded number;
            otherwise, it returns None.

            Returns:
                int or None: The recorded number if the file exists and the date matches; otherwise, None.
        """
        file_path = DEFAULT_DIR + "ui/ui_config/recorded_number.json"
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                return data
        except Exception as e:
            self.default_logger.error(f"Failed to read the info of recorded number: {e}")
            return None

    def load_recorded_num_from_json(self):
        """
            Load the recorded number from a text file.

            This method reads a recorded number and the last recorded date from a specified text file.
            If the file exists and the last recorded date matches the current date, it returns the recorded number;
            otherwise, it returns None.

            Returns:
                int or None: The recorded number if the file exists and the date matches; otherwise, None.
        """
        result = self.load_last_recorded_info()
        if result:
            last_datetime = result.get("datetime")
            recorded_count = result.get("current_recorded_count")
            scanner_barcode = result.get("scanner_barcode")
            if last_datetime == datetime.now().strftime("%Y-%m-%d"):
                return recorded_count, scanner_barcode
            else:
                return None, None
        else:
            return None, None

    def check_datetime(self, current_time):
        """
            Check the date and count information in the given file.

            This method first checks if the file exists. If it does, it opens the file and reads its content.
            It extracts the last count and date, then compares the date with the current time.
            If the date in the file matches the current time, it returns True and the last count value.
            If the dates do not match or the file is empty, it returns False and None.

            Args:
                param file_path: The path to the file storing the date and count information.
                param current_time: The current time, used to compare with the time in the file.
            Return:
                A tuple, where the first element is a boolean indicating whether the dates match;
                the second element is the last count value if the dates match, otherwise None.
        """
        result = self.load_last_recorded_info()
        if result:
            last_count = result.get("current_recorded_count")
            last_date = result.get("datetime")
            if last_date == current_time:
                return True, last_count
        return False, None
    
    def add_recorded_signal_info_to_db(self):
        move_recorded_path = self.move_wav_to_dir(self.recorded_signal_info["labels"])
        file_path = self.recorded_signal_info["file_path"]
        if move_recorded_path:
            file_path = move_recorded_path
        self.recorded_signal_info["file_path"] = file_path.replace(DEFAULT_DIR, "")
        save_code, msg = RecordingManager().save_signal_info_to_db(self.recorded_signal_info, self.data_struct.stimulus_info)
        if save_code == error_code.OK:
            self.default_logger.info("Recorded signal successfully insert.")
        else:
            self.default_logger.error("Failed insert recorded signal.")

    def insert_data_into_db(self):
        """
            Inserts recorded signal data into the database based on user input.

            This method determines which button (OK or NG) triggered the function call and sets the corresponding label
            in the recorded signal information. It then attempts to save this information to the database using the
            `RecordingManager` class. Depending on the success of the operation, it logs either a success or failure message.
        """
        button = self.sender()
        if button == self.ok_btn or self.default_ai_result:
            self.recorded_signal_info["labels"] = "OK"
        elif button == self.ng_btn:
            self.recorded_signal_info["labels"] = "NG"
        self.add_recorded_signal_info_to_db()

    def mark_result(self):
        button = self.sender()
        if button == self.ok_btn:
            self.set_result_file(1, "OK", None)
            self.get_result_file(1)
        elif button == self.ng_btn:
            self.set_result_file(1, "NG", None)
            self.get_result_file(1)

    def update_recorded_label_in_test_mode(self, label: str):
        if label == "OK":
            self.recorded_signal_info["labels"] = "OK"
        elif label == "NG":
            self.recorded_signal_info["labels"] = "NG"

    def test_insert_data_into_db(self):
        current_recorded_count = self.save_recorded_num_to_json("ok_ng")
        self.lineedit_count.setText(str(current_recorded_count))
        self.add_recorded_signal_info_to_db()
        self.player_status_flag = False
        self.update_player_icon()
        self.signal_info.clear()
        self.lineedit_s_or_n.clear()
        # self.line_graph.clear()
        self.replayer_btn.setDisabled(True)
        self.data_btn.setEnabled(False)
        self.default_ai_result = None
        self.default_ai = None
        self.clicked_scanner()

    def move_wav_to_dir(self, label):
        dir_paths = [model_consts.STORED_RECORDED_OK_PATH, model_consts.STORED_RECORDED_NG_PATH]
        for path in dir_paths:
            if not os.path.exists(path):
                os.makedirs(path)
        file_name = os.path.basename(self.recorded_path)
        target_path = ""
        if file_name:
            if label == 'OK':
                target_path = model_consts.STORED_RECORDED_OK_PATH + "/" + file_name
            elif label == 'NG':
                target_path = model_consts.STORED_RECORDED_NG_PATH + "/" + file_name
            shutil.move(self.recorded_path, target_path)
        return target_path

    def clicked_player_btn(self):
        """
            Handles the play button click event. This function performs the following operations:
            1. Clears the line graph based on the player status flag.
            2. Updates the play button state and icon.
            3. Retrieves the analysis configuration from the JSON file.
            4. If the stimulus signal needs to be refreshed, fetches the stimulus signal information from the configuration.
            5. Obtains the sample rate and generates dictionaries for the stimulus and recorded signals.
            6. Uses the soundcard audio processor to play the stimulus signal and record the response signal.
            7. If recording is successful, plots the recorded signal on the line graph and saves the signal information.
            8. Enables the data button and the replay button.
            9. If auto-analysis is configured, executes the analysis.
        """
        self.data_struct.clear_data()
        if self.player_status_flag:
            self.line_graph.clear()
        self.player_status_flag = True
        self.player_btn.setDisabled(True)
        self.update_player_icon()
        self.analysis_config = self.get_sequence_config_from_json()
        QApplication.processEvents()
        sample_rate = self.data_struct.stimulus_info["sample_rate"]
        stimulus_dict, recorded_dict = self.get_stimulus_recorded_dict(sample_rate)
        self.recorded_path, self.recorded_signal_info = self.get_recorded_info()
        sap = SoundcardAudioProcessor()
        record_code, self.data_struct.store_wave_data = sap.sd_play_rec(recorded_dict, stimulus_dict, self.recorded_path)
        if record_code == error_code.OK:
            self.plot_line_graph(self.data_struct.store_wave_data, self.line_graph, sample_rate)

        if self.data_struct.stft_flag != 0:
            self.data_struct.stft_result = librosa.stft(self.data_struct.store_wave_data,
                                                        n_fft=1024,
                                                        hop_length=16,
                                                        win_length=1024,
                                                        window="hann")
        if self.data_struct.fft_flag != 0:
            self.data_struct.fft_result = np.abs(np.fft.fft(self.data_struct.store_wave_data)[:self.data_struct.stimulus_info.get("sample_rate") // 2])

        self.data_btn.setEnabled(True)
        self.replayer_btn.setEnabled(True)
        if self.analysis_config["auto_analysis"]:
            self.run()

    @staticmethod
    def get_class_mapping():
        """
            Retrieves the class mapping dictionary.

            This method returns a dictionary where the keys are string identifiers and the values are the corresponding classes. 
            This mapping is typically used to dynamically retrieve the appropriate class based on an identifier.

            Returns:
                dict: A dictionary containing the class mapping, in the format {"identifier": class}.
        """
        class_mapping = {
            "SPL": Spl,
            "FR": Frequency,
            "HD": Distortion,
            "AI": AI,
            "Spec": Spectrogram,
            "LP": LooseParticle,
        }
        return class_mapping

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
        class_mapping = self.get_class_mapping()
        if type in class_mapping.keys():
            cls_map = class_mapping.get(type)
            if cls_map:
                class_instance = cls_map(key)
                if self.analysis_config["default_ai"] == key:
                    self.default_ai = class_instance
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
        width = int((self.screen().size().width() - 400) / 2)
        height = int((self.screen().size().height() - 400) / 2)
        if self.analysis_config:
            item_sort_list = self.analysis_config.get("display_sequence", [])
            for key in item_sort_list:
                key_config = self.analysis_config.get(key)
                if isinstance(key_config, dict):
                    self.instance_analysis_class(key, key_config["type"], key_config)
            for instance in self.analysis_window:
                if self.mode == "test":
                    if instance is self.default_ai:
                        continue
                if hasattr(instance, 'calculate_spl'):
                    instance.calculate_spl()
                    instance.show()
                elif hasattr(instance, 'calculate_fr'):
                    instance.calculate_fr()
                    instance.show()
                elif hasattr(instance, 'calculate_thd'):
                    instance.calculate_thd()
                    instance.show()
                elif hasattr(instance, 'calculate_ai_scores'):
                    instance.calculate_ai_scores(self.mode)
                    instance.show()
                elif hasattr(instance, 'calculate_spec'):
                    instance.calculate_spec()
                    instance.show()
                elif hasattr(instance, 'calculate_loose_particle'):
                    instance.calculate_loose_particle()
                    instance.show()
                instance.setGeometry(width, height, 600, 500)
                instance.setMinimumSize(QSize(600, 500))
                width += 20
                height += 20
            if self.mode == "test":
                    self.default_ai.calculate_ai_scores(self.mode)
                    self.default_ai.show()
                    self.default_ai.setGeometry(width, height, 600, 500)
                    self.test_insert_data_into_db()
            if self.default_ai:
                if self.default_ai.result == "OK":
                    for instance in self.analysis_window:
                        instance.close()
                    self.default_ai_result = True
                    self.clicked_ok_or_ng()

    def get_sequence_config_from_json(self):
        """
            Retrieves the sequence configuration from a JSON file.

            This method attempts to load the sequence configuration from a JSON file by calling the `load_sequence_from_json()` method.
            If the loading is successful and the result is valid, it returns the configuration; otherwise, it returns an empty dictionary.

            Returns:
                dict: The sequence configuration if loading is successful and the result is valid; otherwise, an empty dictionary.
        """
        load_code, result = self.load_sequence_from_json()
        if load_code == error_code.OK and result:
            return result
        else:
            return {}

    def load_sequence_from_json(self):
        """
            Loads analysis sequence configuration data from a specified JSON file.

            This function first checks if the JSON file exists. If not, it returns an error code and message.
            If the file exists, it attempts to read and parse the JSON file content, storing it in the class's
            `analysis_config` attribute. If any exception occurs during reading or parsing, it catches the
            exception and returns the corresponding error code and message.

            Returns:
                tuple: A tuple containing two elements:
                    - The first element is an error code indicating the result status of the operation.
                    - The second element is either an error message or the parsed JSON data.
        """
        json_file_path = DEFAULT_DIR + "ui/ui_config/analysis_temp_config.json"
        if not os.path.exists(json_file_path):
            return error_code.INVALID_DATA_LOADING, "This json file does not exist."
        try:
            with open(json_file_path, 'r') as json_file:
                self.analysis_config = json.load(json_file)
                return error_code.OK, self.analysis_config
        except Exception as e:
            err_msg = "Failed to load analysis sequence data from json.%s" % (str(e)[:50])
            return error_code.INVALID_DATA_LOADING, err_msg

    @staticmethod
    def get_mic_deviation_value():
        """
            Reads the microphone calibration deviation value from a specified file.

            This method is static because it does not depend on the instance state of the class and can operate independently.
            The deviation value is read from a file as it may vary based on environmental conditions and needs to be
        dynamically adjusted.

            Return:
                The microphone calibration deviation value. Returns 0.0 if reading the file fails.
        """
        file_path = DEFAULT_DIR + "ui/ui_config/mic_calibration.txt"
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                deviation_value = lines[1].strip()
                return float(deviation_value)
        except Exception as e:
            return 0.0

    def get_recorded_info(self):
        """
            Generate recorded information.

            This function generates a unique recording file name based on the current date, MAC address, product model,
        and product number.
            It also constructs the path for the recording file. Additionally, it creates a dictionary containing the
        recording file path and product information.

            Returns:
                tuple: A tuple containing the recording file path and a dictionary with recording information.
        """
        product_model = self.lineedit_type.text()
        recording_time = datetime.now().strftime("%Y-%m-%d")
        mac_address = get_mac_address()
        mac_address = mac_address.replace(":", "") if mac_address else None
        product_number = "{:03}".format(int(self.lineedit_count.text()))
        barcode = self.lineedit_s_or_n.text()
        recorded_name = product_model + "_" + recording_time + "_" + mac_address + "_" + product_number
        if barcode:
            recorded_name = recorded_name + "_BC" + barcode
        else:
            barcode = None
        recorded_name = recorded_name + '.wav'
        recorded_path = model_consts.STORED_RECORDED_PATH + "/" + recorded_name
        recorded_signal_info = {"file_path": recorded_path, "product_model": product_model,
                                "record_date": recording_time, "barcode": barcode
                                }
        return recorded_path, recorded_signal_info

    def get_stimulus_recorded_dict(self, sample_rate):
        """
            Generate dictionaries containing stimulus signal data and recording parameters.

            This function creates two dictionaries: one for the stimulus signal data and its related information,
            and another for the recording parameters. These dictionaries are used for subsequent signal processing and analysis.

            Args:
            - sample_rate (int): The sampling rate, indicating the number of samples collected per second.

            Returns:
            - stimulus_dict (dict): Dictionary containing the stimulus signal data and related information.
            - recorded_dict (dict): Dictionary containing the recording parameters.
        """
        # Define the prolongation time to calculate the extended frame count
        prolong = 3
        stimulus_dict = {"data": self.data_struct.stimulus_data,
                         "amplitude": self.data_struct.stimulus_info["amplitude"],
                         "sr": sample_rate
                         }
        recorded_dict = {"channels": 1,
                         "sr": sample_rate,
                         "num_frames": len(self.data_struct.stimulus_data) + int(prolong * sample_rate),
                         "prolong_frames": int(prolong * sample_rate)
                         }
        return stimulus_dict, recorded_dict

    @staticmethod
    def plot_line_graph(recorded_signal, line_graph, sample_rate):
        """
            Plot a line graph of the recorded signal.

            Parameters:
            recorded_signal (list or numpy.array): The recorded signal data to be plotted.
            line_graph (matplotlib.axes.Axes): The Axes object used for plotting the line graph.
            sample_rate (int or float): The sample rate of the signal, used to calculate the duration of the signal.
        """
        line_graph.clear()
        signal_duration = np.linspace(0, len(recorded_signal) / sample_rate, len(recorded_signal))
        line_graph.plot(signal_duration, recorded_signal)
        QApplication.processEvents()

    def update_player_icon(self):
        """
            Update the player button's icon and size based on the player status flag.

            If self.player_status_flag is True, it indicates that the player is in a paused state,
            and the button icon is set to a pause icon. If self.player_status_flag is False,
            it indicates that the player is in a playing state, and the button icon is set to a play icon,
            and the button is enabled.
        """
        if self.player_status_flag:
            self.player_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/pause.png"))
            self.player_btn.setIconSize(QSize(35, 35))
        else:
            self.player_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/play.png"))
            self.player_btn.setIconSize(QSize(35, 35))
            self.player_btn.setDisabled(False)

    def paintEvent(self, event):
        # Set the window Background-color
        painter = QPainter(self)
        width = self.width()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(208, 206, 202))
        painter.drawRect(1, 0, width - 2, 41)
        painter.end()


class ScannerEmitter(QObject):
    signal_emitter = pyqtSignal(str)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # stimulus_info = {'name': 'stimulus_chirps_1', 'use_custom_stimulus': True, 'stimulus_method': 'chirp',
    #                  'stimulus_type': 'log', 'start_freq': 1000, 'stop_freq': 80, 'total_time': 3.0, 'repeat_times': 1,
    #                  'num_steps': 1, 'amplitude_type': 'RMS', 'amplitude': 0.1, 'sample_rate': 44100}
    # stimulus_info = {'name': 'stimulus_chirps_1', 'use_custom_stimulus': True, 'stimulus_method': 'chirp', 'stimulus_type': 'log',
    #  'start_freq': 80, 'stop_freq': 1000, 'total_time': 3.0, 'repeat_times': 1, 'num_steps': 1, 'amplitude_type': 'RMS',
    #  'amplitude': 0.7, 'sample_rate': 44100}
    stimulus_info = {'name': 'stimulus_chirps_1', 'use_custom_stimulus': True, 'stimulus_method': 'chirp',
                     'stimulus_type': 'mirror_log', 'start_freq': 80, 'stop_freq': 1000, 'total_time': 3.0,
                     'repeat_times': 1,
                     'num_steps': 1, 'amplitude_type': 'RMS', 'amplitude': 0.1, 'sample_rate': 44100}

    # stimulus_signal, sr = librosa.load("../audio_data/stimulus/stimulus.wav", sr=44100)
    # stimulus_signal, sr = librosa.load("../audio_data/stimulus/stimulus111.wav", sr=44100)
    stimulus_signal, sr = librosa.load("../audio_data/stimulus/stimulus_mirror.wav", sr=44100)
    window = SequenceWindow()
    window.stimulus_info = stimulus_info
    window.stimulus_signal = stimulus_signal
    window.show()
    app.exec()
