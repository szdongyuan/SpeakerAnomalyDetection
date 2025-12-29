import json
import os
import re
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QSize, Qt, QTimer
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QMessageBox, QVBoxLayout, QWidget, QFileDialog

from base.data_struct.data_deal_struct import DataDealStruct
from base.file_ops import FileOps
from base.load_audio import load_audio_simple
from base.unified_hid_device_manager import UnifiedHardwareManager
from base.utils.custom_signals import sign
from base.load_config import LoadUiConfig
from base.log_manager import LogManager
from base.play_and_record import (
    play_last_stimulus_wave, record_without_play, get_recorded_info,
    stream_play_and_record, stream_record_without_play
)
from base.recording_management import RecordingManager
from base.save_data import save_recorded_data_to_json, save_audio_simple
from base.soundcard_calibration_manager import get_mic_v2pa_factor
from base.tcp_service import TcpServer, check_tcp_msg_format
from base.temp_tcp_client import TempTcpClient
from base.streaming_file_writer import StreamingWavWriter
from base.pre_processing.alignment_processing import AlignmentProcessing
from base.pre_processing.split_repeat_signal import SplitRepeatSignal
from consts import ui_style_const, error_code
from consts.action_code import RequestTypeEnum
from consts.running_consts import DEFAULT_DIR
from ui.operation_sequence import AnalysisModelSelect
from ui.sequence.sequence_tools_bar import SequenceToolsBar
from ui.signal_analysis_window import get_class_mapping
from ui.sequence.sequencement_count_board import SequenceCountBoard
from ui.tcp_config_dialog import TcpConfigDialog


class SequenceWindow(QWidget):
    tcp_server = None

    def __init__(self):
        """Initializes the class instance, setting up the user interface and necessary parameters."""
        super().__init__()
        self.data_struct = DataDealStruct()
        self.recorded_path = None
        self.refresh_stimulus_flag = None
        self.add_or_update_wave_flag = True
        self.count_board = None
        self.toolsbar = SequenceToolsBar()

        self.v2pa_factor = get_mic_v2pa_factor()
        self.sequence_config = list()
        self.analysis_config = dict()
        self.get_sequence_config_from_json()
        self.init_data_struct_stimulus_config()
        self.init_fft_and_stft_flag()
        self.signal_info = {}
        self.analysis_window = []
        self.default_ai = None
        self.default_ai_result = None

        self.init_result_files()
        self.count_board = SequenceCountBoard(self.analysis_config)
        self.player_status_flag = False
        self.recorded_signal_info = {}
        self.ip_format = True
        self.port_format = True
        self.clicked_player_flag = False
        self.tcp_flag = False
        self.tcp_ip, self.tcp_port = LoadUiConfig.get_tcp_config()
        self.mode = None
        self.current_recorded_count = None
        self.last_play_count = None  # Cache last play count for replay

        self.default_logger = LogManager.set_log_handler("core")

        # Streaming state variables
        self.streaming_buffer = []
        self.streaming_time_data = []
        self.streaming_plot_item = None  # Persistent plot item for zoom preservation
        self.streaming_wav_writer = None  # WAV file writer for incremental saving
        self.streaming_processor = None  # StreamingAudioProcessor instance
        self.streaming_stimulus_data = None  # Stimulus data for alignment (play+record mode)
        self.streaming_mode = None  # "play_record" or "record_only"
        self.use_streaming = True  # Set to True to use streaming, False for legacy blocking mode

        self.hw_manager = UnifiedHardwareManager()
        # Create QTimer in Qt main thread for queue polling
        self.streaming_poll_timer = QTimer(self)
        self.streaming_poll_timer.timeout.connect(self._poll_streaming_queue)

        self.set_member_connect()
        self.bind_hw_signals()
        self.init_lineedit_text()
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
        waveform_layout = self.create_waveform_layout()

        sequence_layout = QVBoxLayout()
        sequence_layout.addWidget(self.toolsbar)
        sequence_layout.addLayout(waveform_layout)
        sequence_layout.setAlignment(Qt.AlignCenter)
        sequence_layout.setContentsMargins(1, 0, 1, 0)

        self.setLayout(sequence_layout)

        sign.run_test_sign.connect(self.start_this_play, Qt.AutoConnection)
        sign.get_result_file_sign.connect(self.connect_get_result_file_sign, Qt.AutoConnection)
        sign.set_result_file_sign.connect(self.connect_set_result_file_sign, Qt.AutoConnection)
        sign.update_mode_display_sign.connect(self.get_sequence_config_from_json, Qt.AutoConnection)
        sign.test_insert_data_into_db_sign.connect(self.update_recorded_label_in_test_mode, Qt.AutoConnection)
        # sign.update_mode_display_sign.connect(self.update_mode_display, Qt.AutoConnection)
        sign.update_mode_display_sign.connect(self.count_board.on_test_btn_clicked, Qt.AutoConnection)
        # Streaming audio chunk signal for real-time waveform updates
        sign.stream_audio_chunk_signal.connect(self.on_audio_chunk_received, Qt.AutoConnection)
        # self.update_mode_display(0)
        self.count_board.on_test_btn_clicked
        self.setStyleSheet(
            ui_style_const.qcombobox_style
            + ui_style_const.qpushbutton_style
            + ui_style_const.qlineedit_style
            + ui_style_const.qframe_style
            + ui_style_const.qlabel_style
            + ui_style_const.qcheckbox_style
        )

    def update_v2pa_factor(self):
        self.v2pa_factor = get_mic_v2pa_factor()

    def connect_set_result_file_sign(self, index, label, model_name):
        if self.count_board.mode == "test":
            self.count_board.set_test_result_file(label, model_name)
        else:
            self.count_board.set_mark_result_file(label)

    def connect_get_result_file_sign(self):
        if self.count_board.mode == "test":
            self.count_board.set_test_text()
        else:
            self.count_board.set_mark_text()

    def set_member_connect(self):
        self.player_btn.clicked.connect(lambda: self.on_clicked_player_btn())
        self.replayer_btn.clicked.connect(lambda: self.judge_play_and_record(is_replay=True))
        self.data_btn.clicked.connect(self.run)
        self.lineedit_type.editingFinished.connect(lambda: self.lineedit_type_lose_focus(self.lineedit_type))
        self.lineedit_count.editingFinished.connect(lambda: self.lineedit_count_lose_focus(self.lineedit_count))
        self.lineedit_count.returnPressed.connect(lambda: self.validate_count(self.lineedit_count, True))
        self.barcode_scanner_box.stateChanged.connect(self.clicked_scanner)
        self.tcp_btn.clicked.connect(self.on_tcp_btn_clicked)
        self.count_board.ok_btn.clicked.connect(self.clicked_ok_or_ng)
        self.count_board.ng_btn.clicked.connect(self.clicked_ok_or_ng)

    def init_lineedit_text(self):
        last_recorded_info = LoadUiConfig().load_last_recorded_info(self.default_logger)
        if last_recorded_info:
            product_model = last_recorded_info.get("product_model", "S004-1")
        else:
            product_model = "S004-1"
        self.lineedit_type.setText(product_model)

        result, _ = LoadUiConfig.load_recorded_num_from_json(self.default_logger)
        if result is None:
            self.current_recorded_count = 1
        else:
            self.current_recorded_count = result

        self.lineedit_count.setText(str(self.current_recorded_count))

    @property
    def player_btn(self):
        return self.toolsbar.player_btn

    @property
    def replayer_btn(self):
        return self.toolsbar.replayer_btn

    @property
    def data_btn(self):
        return self.toolsbar.data_btn

    @property
    def lineedit_type(self):
        return self.toolsbar.lineedit_type

    @property
    def lineedit_count(self):
        return self.toolsbar.lineedit_count

    @property
    def lineedit_s_or_n(self):
        return self.toolsbar.lineedit_s_or_n

    @property
    def barcode_scanner_box(self):
        return self.toolsbar.barcode_scanner_box

    @property
    def tcp_btn(self):
        return self.toolsbar.tcp_btn

    def init_data_struct_stimulus_config(self):
        if not self.sequence_config:
            return
        acq_config = self.sequence_config[0]["seq1"]["acq"]
        if acq_config["mode"] == "PLAY_AND_RECORD":
            AnalysisModelSelect.set_data_struct_stimulus_signal(self.data_struct, acq_config["detail"])
        else:
            self.data_struct.sample_rate = acq_config["detail"]["sample_rate"]

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
        self.line_graph.setBackground("white")
        self.line_graph.setLabel("left", "Amplitude(V)", **{"font-size": "20px"})
        self.line_graph.setLabel("bottom", "Time(s)", **{"font-size": "20px"})
        self.line_graph.showGrid(x=True, y=True)

        font = QFont()
        font.setPixelSize(20)
        b_axis = self.line_graph.getAxis("bottom")
        l_axis = self.line_graph.getAxis("left")
        b_axis.setTickFont(font)
        l_axis.setTickFont(font)
        b_axis.setTextPen("black")
        l_axis.setTextPen("black")

        layout.addWidget(self.count_board, stretch=1)
        layout.addSpacing(20)
        layout.addWidget(self.line_graph, stretch=8)
        layout.setContentsMargins(40, 20, 40, 20)
        layout.setSpacing(30)
        return layout

    def init_fft_and_stft_flag(self):
        model_item_list = self.analysis_config.get("display_sequence", "")
        for item_name in model_item_list:
            self.data_struct.add_stft_or_fft_count(self.analysis_config[item_name]["type"])

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
            self.init_mark_result_file(mark_result_path, mark_result_template)

    def init_mark_result_file(self, mark_result_path, mark_result_template):
        with open(mark_result_path, "r") as f:
            data = json.load(f)
        current_date = datetime.now().strftime("%Y-%m-%d")
        if data["datatime"] != current_date:
            with open(mark_result_path, "w") as f:
                json.dump(mark_result_template, f, indent=4)

    def bind_hw_signals(self):
        """绑定 hardware manager 的信号, 避免重复连接"""
        try:
            self.hw_manager.sig_barcode.disconnect()
            self.hw_manager.sig_trigger.disconnect()
        except TypeError:
            pass
        self.hw_manager.sig_barcode.connect(self.on_barcode_received)
        self.hw_manager.sig_trigger.connect(self.on_sensor_triggered)

    def clicked_scanner(self):
        """Checkbox 状态改变时的回调"""
        if self.barcode_scanner_box.isChecked():
            if not self.hw_manager.ensure_config_loaded():
                self.barcode_scanner_box.setChecked(False)
                QMessageBox.warning(
                    self,
                    "硬件初始化失败",
                    "无法加载扫码枪/光电开关配置，请检查配置文件。"
                )
                return

            self.lineedit_s_or_n.setEnabled(True)
            if self.hw_manager.start():
                self.default_logger.info("硬件监听已启动")
            else:
                self.barcode_scanner_box.setChecked(False)
                QMessageBox.warning(
                    self,
                    "硬件参数缺失",
                    "HID 配置缺少有效的 VID/PID，已恢复到未勾选状态。"
                )
        else:
            self.lineedit_s_or_n.clear()
            self.lineedit_s_or_n.setDisabled(True)
            self.hw_manager.stop()
            self.default_logger.info("硬件监听已停止")

    def on_barcode_received(self, barcode):
        """处理扫码枪信号"""
        if not barcode:
            return

        self.lineedit_s_or_n.setText(barcode)

        if self.barcode_scanner_box.isChecked():
            if not self.player_status_flag:
                self.start_this_play("not_labeled")

    def on_sensor_triggered(self):
        """处理光电开关触发信号"""
        if not self.player_status_flag:
            self.default_logger.info("光电触发响应: 开始测试")
            self.start_this_play("not_labeled")
        else:
            self.default_logger.info("正在测试中，忽略光电触发")

    def closeEvent(self, event):
        """窗口关闭时释放硬件资源"""
        if hasattr(self, 'hw_manager'):
            self.hw_manager.stop()
        super().closeEvent(event)

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
        self.count_board.set_mark_text()

    def lineedit_count_lose_focus(self, lineedit):
        self.current_recorded_count = int(lineedit.text())
        save_recorded_data_to_json(
            self.lineedit_type.text(),
            self.lineedit_count.text(),
            self.lineedit_s_or_n.text(),
            self.barcode_scanner_box.isChecked(),
        )
        lineedit.clearFocus()
        if lineedit.text() == "":
            result_count, _ = LoadUiConfig.load_recorded_num_from_json(self.default_logger)
            lineedit.setText(str(result_count))

    def lineedit_type_lose_focus(self, lineedit):
        save_recorded_data_to_json(
            self.lineedit_type.text(),
            self.lineedit_count.text(),
            self.lineedit_s_or_n.text(),
            self.barcode_scanner_box.isChecked(),
        )
        lineedit.clearFocus()
        if lineedit.text() == "":
            last_recorded_info = LoadUiConfig().load_last_recorded_info(self.default_logger)
            lineedit.setText(str(last_recorded_info.get("product_model", "S004-1")))

    def validate_count(self, lineedit, is_s_or_n: bool):
        """
            Validates the count input from the user.

            This method checks if the user input in the lineedit is a valid number. If the input is not a number,
            it restores the previously recorded number. If the input is valid, it updates the recorded number and saves
        it to a file.

            Parameters:
            lineedit (QLineEdit): The QLineEdit object containing the user's count input.
        """
        s_or_n_count = lineedit.text()
        result_count, result_scanner_barcode = LoadUiConfig.load_recorded_num_from_json(self.default_logger)
        reg = None
        if is_s_or_n:
            reg = r"^[0-9]*$"
        else:
            reg = r"^[0-9a-zA-Z]*$"

        if not re.match(reg, s_or_n_count):
            if is_s_or_n:
                lineedit.setText(str(result_count))
            else:
                lineedit.setText(str(result_scanner_barcode))
        elif s_or_n_count != "":
            if is_s_or_n:
                self.lineedit_s_or_n.setText("")
        if s_or_n_count == "":
            if is_s_or_n:
                lineedit.setText(str(result_count))
            else:
                lineedit.setText(str(result_scanner_barcode))

    def swap_tcp_status(self):
        if self.tcp_flag:
            self.barcode_scanner_box.setEnabled(False)
            self.tcp_ip, self.tcp_port = LoadUiConfig.get_tcp_config()
            if hasattr(self, "tcp_server") and SequenceWindow.tcp_server:
                SequenceWindow.tcp_server.stop()
                SequenceWindow.tcp_server = None
            SequenceWindow.tcp_server = TcpServer(host=self.tcp_ip, port=self.tcp_port, callback=self.deal_package)
            SequenceWindow.tcp_server.start()
        else:
            self.barcode_scanner_box.setEnabled(True)
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

    def on_tcp_btn_clicked(self):
        tcp_config_dialog = TcpConfigDialog(self.tcp_flag, self.tcp_ip, self.tcp_port)
        result = tcp_config_dialog.exec()
        if result:
            self.tcp_flag, self.tcp_ip, self.tcp_port = result
            LoadUiConfig.write_tcp_config(self.tcp_ip, self.tcp_port, self.default_logger)
            self.swap_tcp_status()

    def clicked_ok_or_ng(self, manual=True):
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
            manual: If True (default), this is a manual user click; if False, this is an auto-triggered call.
                    Only manual calls will disable the replay and data buttons.
        """
        if not hasattr(self.data_struct, 'store_wave_data') or self.data_struct.store_wave_data is None or len(self.data_struct.store_wave_data) == 0:
            QMessageBox.warning(self, "警告", "请先录制声音！")
            return
        if self.sequence_config[0]["seq1"]["acq"]["mode"] == "IMPORT_AUDIO":
            QMessageBox.warning(self, "警告", "当前为导入音频模式，无需点击 OK/NG 按钮。")
            return

        self.update_audio_label_info()
        self.update_recorded_signal_info_to_db()

        self.mark_result()
        self.player_status_flag = False
        self.signal_info.clear()
        self.lineedit_s_or_n.clear()
        self.line_graph.clear()

        # Only disable buttons when manually clicking OK/NG
        # Auto-triggered calls should keep buttons enabled for user verification
        if manual:
            self.replayer_btn.setDisabled(True)
            self.data_btn.setEnabled(False)

        self.default_ai_result = None
        self.default_ai = None
        self.clicked_scanner()

    def update_recorded_signal_info_to_db(self):
        if self.recorded_signal_info["labels"] == "not_labeled":
            return
        new_file_path = FileOps.move_wav_to_dir(self.recorded_path, self.recorded_signal_info["labels"])
        old_file_path = self.recorded_signal_info["file_path"]
        self.recorded_signal_info["file_path"] = new_file_path.replace(DEFAULT_DIR, "")
        save_code, msg = RecordingManager().update_audio_label(self.recorded_signal_info, old_file_path)
        if save_code == error_code.OK:
            self.default_logger.info("Recorded signal successfully updated.")
        else:
            self.default_logger.error("Failed to update recorded signal.")

    def update_audio_label_info(self):
        button = self.sender()
        if button == self.count_board.ok_btn or self.default_ai_result:
            self.recorded_signal_info["labels"] = "OK"
        elif button == self.count_board.ng_btn:
            self.recorded_signal_info["labels"] = "NG"

    def mark_result(self):
        button = self.sender()
        if button == self.count_board.ok_btn or self.default_ai_result:
            self.count_board.set_mark_result_file("OK")
            self.count_board.set_mark_text()
        elif button == self.count_board.ng_btn:
            self.count_board.set_mark_result_file("NG")
            self.count_board.set_mark_text()

    def update_recorded_label_in_test_mode(self, label: str):
        if label == "OK":
            self.recorded_signal_info["labels"] = "OK"
        elif label == "NG":
            self.recorded_signal_info["labels"] = "NG"

    def test_insert_data_into_db(self):
        self.update_recorded_signal_info_to_db()
        self.player_status_flag = False
        self.signal_info.clear()
        self.lineedit_s_or_n.clear()
        self.replayer_btn.setDisabled(True)
        self.data_btn.setEnabled(False)
        self.default_ai_result = None
        self.default_ai = None
        self.clicked_scanner()

    def on_clicked_player_btn(self, label="not_labeled"):
        acq_mode = self.sequence_config[0]["seq1"]["acq"]["mode"]
        if acq_mode == "IMPORT_AUDIO":
            self.import_audio_and_analyze()
            return
        self.clicked_player_flag = True
        self.start_this_play(label)

    def import_audio_and_analyze(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择音频文件",
            DEFAULT_DIR + "audio_data/stored_data",
            "WAV Files (*.wav);;All Files (*)",
        )
        if not file_path:
            return
        acq_detail = self.sequence_config[0]["seq1"]["acq"]["detail"]
        sample_rate = acq_detail.get("sample_rate", 44100)
        y, _ = load_audio_simple(file_path, sample_rate)

        self.data_struct.store_wave_data = y
        self.data_struct.sample_rate = sample_rate

        self.line_graph.clear()
        self.plot_line_graph(self.data_struct.store_wave_data, self.line_graph, self.data_struct.sample_rate)

        self.data_btn.setEnabled(True)
        if self.analysis_config.get("auto_analysis"):
            self.run()

    def start_this_play(self, label="not_labeled"):
        if self.clicked_player_flag is False:
            if self.tcp_flag and SequenceWindow.tcp_server.client_address is None:
                QMessageBox.warning(self, "提示", "TCP链接异常")
                return

        # Increment count BEFORE recording (so display count = file count)
        self.current_recorded_count += 1
        self.lineedit_count.setText(str(self.current_recorded_count))

        # Cache this count for replay
        self.last_play_count = self.current_recorded_count

        save_recorded_data_to_json(
            self.lineedit_type.text(),
            self.lineedit_count.text(),
            self.lineedit_s_or_n.text(),
            self.barcode_scanner_box.isChecked(),
        )

        # Record with new count
        self.judge_play_and_record(label, is_replay=False)

        if self.clicked_player_flag is True:
            self.clicked_player_flag = False
        elif self.clicked_player_flag is False:
            if self.tcp_flag:
                TempTcpClient(
                    SequenceWindow.tcp_server.client_address[0], SequenceWindow.tcp_server.client_address[1], "finish"
                )

    def checked_work_status_message(self):
        if not self.sequence_config:
            QMessageBox.warning(self, "提示", "未找到录音模式，请在功能-测试队列中配置")
            return True

        if not self.mic:
            QMessageBox.warning(self, "提示", "未找到麦克风，请在硬件中设置")
            return True
        if not self.speaker:
            QMessageBox.warning(self, "提示", "未找到扬声器，请在硬件中设置")
            return True

        return False

    def reset_work_pram(self, label, count=None):
        self.data_struct.clear_data()

        # Use provided count if available (for replay), otherwise use lineedit value
        count_str = str(count) if count is not None else self.lineedit_count.text()

        self.recorded_path, self.recorded_signal_info = get_recorded_info(
            self.lineedit_type.text(), count_str, self.lineedit_s_or_n.text(), label
        )
        acq_detail = self.sequence_config[0]["seq1"]["acq"]["detail"]
        total_time = float(acq_detail.get("total_time", 5.0))
        sample_rate = self.data_struct.sample_rate
        stimulus_dict, recorded_dict = LoadUiConfig.get_rec_and_play_dict_base_sequence_dict(
            self.data_struct, total_time
        )

        # Add device information for streaming mode
        recorded_dict["device"] = self.mic
        recorded_dict["input_device"] = self.mic
        recorded_dict["output_device"] = self.speaker

        return stimulus_dict, recorded_dict, sample_rate

    def judge_play_and_record(self, label="not_labeled", is_replay=False):
        if self.checked_work_status_message():
            return

        self.line_graph.clear()
        # CRITICAL: Clean up any existing streaming resources before starting new recording
        # This prevents device conflicts and freezing when replay is clicked multiple times
        self._cleanup_streaming_resources()

        self.update_player_btn_is_playing()

        # Clear plot and reset streaming state for NEW recording
        if self.player_status_flag:
            self.line_graph.clear()

        self.streaming_buffer = []
        self.streaming_time_data = []
        self.streaming_plot_item = None  # Reset plot item for new recording

        self.player_status_flag = True

        # Disable replay and data buttons during recording/playback
        # They will be re-enabled in _on_streaming_complete()
        self.replayer_btn.setDisabled(True)
        self.data_btn.setDisabled(True)

        QApplication.processEvents()

        # For replay: use cached count to overwrite the same file
        # For play: use current lineedit count (already incremented in start_this_play)
        if is_replay:
            if self.last_play_count is None:
                QMessageBox.warning(self, "提示", "请先进行录音")
                return
            stimulus_dict, recorded_dict, sample_rate = self.reset_work_pram(label, count=self.last_play_count)
        else:
            stimulus_dict, recorded_dict, sample_rate = self.reset_work_pram(label)

        # Choose streaming or blocking(Not in use now) mode
        if self.use_streaming:
            # Use modern streaming approach with true real-time updates (non-blocking)
            if self.sequence_config[0]["seq1"]["acq"]["mode"] in ["PLAY_AND_RECORD"]:
                # Start streaming play+record (non-blocking)
                # Stream to TEMP file for safety - will be deleted after alignment+save succeeds
                temp_path = self.recorded_path.replace('.wav', '_temp.wav')
                self.streaming_wav_writer = StreamingWavWriter(temp_path, sample_rate)
                self.streaming_temp_path = temp_path

                self.streaming_processor, self.streaming_stimulus_data, _ = stream_play_and_record(
                    stimulus_dict, recorded_dict, self.recorded_path, self.recorded_signal_info
                )
                self.streaming_mode = "play_record"

            else:
                # Start streaming record-only (non-blocking)
                # Create WAV file writer for streaming saves (useful for long recordings)
                self.streaming_wav_writer = StreamingWavWriter(self.recorded_path, sample_rate)

                self.streaming_processor, _ = stream_record_without_play(
                    recorded_dict, self.recorded_path, self.recorded_signal_info
                )
                self.streaming_mode = "record_only"
                self.streaming_stimulus_data = None

            # Start polling timer to process queue and detect completion
            self.streaming_poll_timer.start(50)  # Poll every 50ms

            # Return immediately - completion will be handled by _on_streaming_complete()
            # Note: Don't enable buttons yet, that happens in _on_streaming_complete()
            return

        else:
            # Use legacy blocking approach
            if self.sequence_config[0]["seq1"]["acq"]["mode"] in ["PLAY_AND_RECORD"]:
                play_last_stimulus_wave(stimulus_dict, recorded_dict, self.recorded_path, self.recorded_signal_info)
            else:
                record_without_play(recorded_dict, self.recorded_path, self.recorded_signal_info)
            self.plot_line_graph(self.data_struct.store_wave_data, self.line_graph, sample_rate)

        self.player_status_flag = False  # Recording complete, allow hardware access
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
                if self.analysis_config["default_ai"] == key:
                    self.default_ai = class_instance
                class_instance.v2pa_factor = self.v2pa_factor
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
                if self.count_board.mode == "test":
                    if instance is self.default_ai:
                        continue
                if hasattr(instance, "calculate_spl"):
                    result = instance.calculate_spl()
                    if not result:
                        continue
                    instance.show()
                elif hasattr(instance, "calculate_fr"):
                    result = instance.calculate_fr()
                    if not result:
                        continue
                    instance.show()
                elif hasattr(instance, "calculate_thd"):
                    instance.calculate_thd()
                    instance.show()
                elif hasattr(instance, "calculate_ai_scores"):
                    instance.calculate_ai_scores(self.count_board.mode, self.analysis_config,
                                                 self.sequence_config[0]["seq1"]["acq"]["mode"])
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
                instance.setGeometry(width, height, 600, 500)
                instance.setMinimumSize(QSize(600, 500))
                width += 20
                height += 20
            if self.count_board.mode == "test":
                self.default_ai.calculate_ai_scores(self.count_board.mode, self.analysis_config)
                self.default_ai.show()
                self.default_ai.setGeometry(width, height, 600, 500)
                self.test_insert_data_into_db()
            elif self.default_ai:
                if self.default_ai.result == "OK":
                    for instance in self.analysis_window:
                        instance.close()
                    self.default_ai_result = True
                    self.clicked_ok_or_ng(manual=False)

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
            mode = seq["acq"]["mode"]
            if mode == "IMPORT_AUDIO":
                self.replayer_btn.setDisabled(True)
            if self.count_board:
                self.count_board.analysis_config = seq.get("analysis_list", {})
        else:
            self.sequence_config = []
            self.analysis_config = dict()

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
        # Use np.arange for correct sample time stamps (0, 1/fs, 2/fs, ..., (N-1)/fs)
        signal_duration = np.arange(len(recorded_signal)) / sample_rate
        line_graph.plot(signal_duration, recorded_signal, pen="k")

    def on_audio_chunk_received(self, chunk):
        """
        Handle streaming audio chunk for real-time waveform display.

        Updates plot incrementally while preserving zoom/pan state.
        Also writes chunk to file via connected wav_writer.

        Args:
            chunk (np.ndarray): Audio chunk received from streaming processor
        """
        # Append chunk to streaming buffer
        self.streaming_buffer.append(chunk)

        # Calculate accumulated audio data
        accumulated_audio = np.concatenate(self.streaming_buffer)

        # Calculate time axis for accumulated data
        # Use np.arange to ensure fixed time stamps for each sample point
        # This prevents time drift caused by linspace's varying interval (N/(N-1)/fs instead of 1/fs)
        sample_rate = self.data_struct.sample_rate
        time_axis = np.arange(len(accumulated_audio)) / sample_rate

        # Update plot - preserve zoom/pan by updating existing PlotDataItem
        if self.streaming_plot_item is None:
            # First chunk - create new plot item
            self.streaming_plot_item = self.line_graph.plot(time_axis, accumulated_audio, pen="k")
        else:
            # Subsequent chunks - update existing item (preserves zoom/pan) incrementally
            self.streaming_plot_item.setData(time_axis, accumulated_audio)    

        # Write chunk to file (if wav_writer is connected)
        if self.streaming_wav_writer:
            try:
                self.streaming_wav_writer.write_chunk(chunk)
            except Exception as e:
                self.default_logger.error(f"Error writing audio chunk to file: {e}")

    def _poll_streaming_queue(self):
        """
        Poll streaming queue and check for completion.

        Called by QTimer every 50ms from Qt main thread.
        Processes audio chunks from queue and checks if recording is finished.
        """
        if self.streaming_processor is None:
            return

        # Process all available chunks from queue (non-blocking)
        self.streaming_processor.process_queue()

        # Check if recording is complete
        if not self.streaming_processor.is_recording:
            self.streaming_poll_timer.stop()
            self._on_streaming_complete()

    def _on_streaming_complete(self):
        """
        Handle streaming completion: alignment, file save, and analysis.

        Called when streaming recording is finished. Performs:
        - Get recorded data from processor
        - Alignment (for play+record mode)
        - Finalize WAV file
        - Store data for analysis
        - Save to database
        - Enable buttons and optionally run analysis
        """
        try:
            # Get the complete recorded data
            recorded_data = self.streaming_processor.get_recorded_data()
            sample_rate = self.data_struct.sample_rate

            # VERIFICATION: Check if we captured the expected number of samples
            expected_samples = self.streaming_processor.target_samples
            actual_samples = len(recorded_data)
            if actual_samples != expected_samples:
                self.default_logger.warning(
                    f"Sample count mismatch! Expected: {expected_samples}, Got: {actual_samples}, "
                    f"Missing: {expected_samples - actual_samples} samples ({(expected_samples - actual_samples) / sample_rate * 1000:.1f}ms)"
                )
            else:
                self.default_logger.info(f"Recording complete: {actual_samples} samples captured (matches target)")

            # Perform alignment if in play+record mode
            if self.streaming_mode == "play_record" and self.streaming_stimulus_data is not None:
                # Finalize temp file first
                if self.streaming_wav_writer:
                    self.streaming_wav_writer.finalize()
                    self.streaming_wav_writer = None

                aligned_data = AlignmentProcessing.align_play_and_rec_data_using_gccphat(
                    self.streaming_stimulus_data, recorded_data
                )

                # Update plot with aligned data (refresh display)
                final_time_axis = np.linspace(0, len(aligned_data) / sample_rate, len(aligned_data))
                if self.streaming_plot_item:
                    self.streaming_plot_item.setData(final_time_axis, aligned_data)
                else:
                    self.streaming_plot_item = self.line_graph.plot(final_time_axis, aligned_data, pen="k")

                # Store aligned data
                self.data_struct.store_wave_data = aligned_data

                # Save aligned data to final file
                save_audio_simple(self.recorded_path, aligned_data, sample_rate)

                # Save to database
                self.recorded_signal_info["sample_rate"] = sample_rate
                save_code, save_msg = RecordingManager().save_signal_info_to_db(self.recorded_signal_info, self.data_struct.stimulus_info)
                if save_code == error_code.OK:
                    self.default_logger.info(f"Database save successful: {save_msg}")
                else:
                    self.default_logger.error(f"Database save failed: {save_msg}")

                # Delete temp file AFTER successful save (for data safety)
                try:
                    if hasattr(self, 'streaming_temp_path') and os.path.exists(self.streaming_temp_path):
                        os.remove(self.streaming_temp_path)
                        self.default_logger.info(f"Deleted temp file: {self.streaming_temp_path}")
                except Exception as e:
                    self.default_logger.warning(f"Failed to delete temp file: {e}")

            else:
                # Record-only mode - no alignment needed
                self.data_struct.store_wave_data = recorded_data

                # Finalize WAV file (for record-only, this is the final file)
                if self.streaming_wav_writer:
                    self.streaming_wav_writer.finalize()
                    self.streaming_wav_writer = None

                # Save to database
                self.recorded_signal_info["sample_rate"] = sample_rate
                save_code, save_msg = RecordingManager().save_signal_info_to_db(self.recorded_signal_info, None)
                if save_code == error_code.OK:
                    self.default_logger.info(f"Database save successful: {save_msg}")
                else:
                    self.default_logger.error(f"Database save failed: {save_msg}")

            # Handle repeat signal splitting if needed
            if self.streaming_mode == "play_record":
                repeat_times = self.data_struct.stimulus_info.get("repeat_times", 1)
                if repeat_times > 1:
                    kwargs = {"repeat_times": repeat_times}
                    self.data_struct.split_repeat_data = SplitRepeatSignal().split_repeat_signal(
                        self.data_struct.store_wave_data, sample_rate, **kwargs
                    )

            # Clean up streaming state
            self.streaming_processor = None
            self.streaming_stimulus_data = None
            self.streaming_mode = None
            self.player_status_flag = False  # Recording complete, allow hardware access

            # Enable buttons for replay and data analysis
            self.data_btn.setEnabled(True)
            self.replayer_btn.setEnabled(True)

            # Run auto-analysis if enabled
            if self.analysis_config.get("auto_analysis", False):
                self.run()

            # Update player button state
            self.update_player_btn_is_paused()

            self.default_logger.info("Streaming recording completed successfully")

        except Exception as e:
            self.default_logger.error(f"Error in streaming completion: {e}")
            # Clean up on error
            if self.streaming_wav_writer:
                self.streaming_wav_writer.finalize()
                self.streaming_wav_writer = None
            self.streaming_processor = None
            self.streaming_stimulus_data = None
            self.streaming_mode = None
            self.player_status_flag = False  # Clear flag even on error to prevent permanent blocking
            # Still enable buttons even on error
            self.data_btn.setEnabled(True)
            self.replayer_btn.setEnabled(True)
            self.update_player_btn_is_paused()

    def _cleanup_streaming_resources(self):
        """
        Clean up all streaming resources (timer, processor, wav_writer).
        
        Called before starting a new recording to prevent resource conflicts.
        Safe to call multiple times (idempotent).
        """
        # 1. Stop timer first (prevent callbacks during cleanup)
        try:
            if self.streaming_poll_timer.isActive():
                self.streaming_poll_timer.stop()
                self.default_logger.debug("Stopped streaming poll timer")
        except Exception as e:
            self.default_logger.error(f"Error stopping timer: {e}")
        
        # 2. Stop processor (stops audio streams - CRITICAL for preventing device conflicts)
        try:
            if self.streaming_processor is not None:
                self.streaming_processor.stop_streaming()
                self.streaming_processor = None
                self.default_logger.debug("Stopped streaming processor")
        except Exception as e:
            self.default_logger.error(f"Error stopping processor: {e}")
        
        # 3. Finalize wav writer
        try:
            if self.streaming_wav_writer is not None:
                self.streaming_wav_writer.finalize()
                self.streaming_wav_writer = None
                self.default_logger.debug("Finalized wav writer")
        except Exception as e:
            self.default_logger.error(f"Error finalizing wav writer: {e}")
        
        # 4. Clean up other state
        self.streaming_stimulus_data = None
        self.streaming_mode = None

    def update_player_btn_is_playing(self):
        self.player_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/pause.png"))
        self.player_btn.setIconSize(QSize(35, 35))
        self.player_btn.setDisabled(True)

    def update_player_btn_is_paused(self):
        self.player_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/play.png"))
        self.player_btn.setIconSize(QSize(35, 35))
        self.player_btn.setDisabled(False)
