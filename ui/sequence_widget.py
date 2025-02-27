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
from getmac import get_mac_address
from PyQt5.QtCore import QSize, Qt, QObject, pyqtSignal
from PyQt5.QtGui import QIcon, QPainter, QColor
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFrame, QCheckBox
from PyQt5.QtWidgets import QSpacerItem, QSizePolicy, QVBoxLayout, QWidget

from base.barcode_scanning_processor import BarcodeScanner
from base.load_audio import load_audio_simple
from base.log_manager import LogManager
from base.recording_management import RecordingManager
from base.soundcard_audio_processor import SoundcardAudioProcessor
from consts import ui_style_const, error_code, model_consts
from consts.running_consts import DEFAULT_DIR
from ui.signal_analysis_window import Spl, Distortion, AI, Frequency


class SequenceWindow(QWidget):

    def __init__(self):
        """Initializes the class instance, setting up the user interface and necessary parameters."""
        super().__init__()
        self.collect_or_analyse_layout = QHBoxLayout()
        self.recorded_path = None   # Initialize the recorded path variable
        self.refresh_stimulus_flag = None   # Initialize the flag to indicate if stimulus needs refreshing
        # Retrieve stimulus information and signal from configuration
        self.stimulus_info, self.stimulus_signal = self.get_stimulus_from_config()  
        self.deviation_value = self.get_mic_deviation_value()   # Get the deviation value from the microphone
        self.analysis_config = self.get_sequence_config_from_json()
        self.signal_info = {}   # Initialize an empty dictionary to store signal information
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
        toolbar_layout = self.create_toolbar_layout()
        waveform_layout = self.create_waveform_layout()

        self.sequence_layout.addLayout(toolbar_layout)
        self.sequence_layout.addLayout(waveform_layout)
        self.sequence_layout.setAlignment(Qt.AlignCenter)
        self.sequence_layout.setContentsMargins(0, 0, 0, 0)

        self.setLayout(self.sequence_layout)

        self.ok_btn.clicked.connect(self.clicked_ok_or_ng)
        self.ng_btn.clicked.connect(self.clicked_ok_or_ng)

        self.setStyleSheet(ui_style_const.qcombobox_stytle +
                           ui_style_const.qpushbutton_stytle +
                           ui_style_const.qlineedit_stytle +
                           ui_style_const.qframe_stytle + 
                           ui_style_const.qlabel_stytle + 
                           ui_style_const.qcheckbox_stytle)

    def create_toolbar_layout(self):
        """
            Create the toolbar layout.

            This method initializes and configures the toolbar layout for the application.
            It sets up button styles, adds labels and input fields, and sets layout parameters.
            The layout is used at the top of the interface to provide easy access to key functionalities.

            Returns:
                QHBoxLayout: The configured toolbar layout object.
        """
        self.player_btn.setFixedSize(100, 40)
        self.player_btn.setStyleSheet(ui_style_const.toolbar_button_stytle)
        self.player_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/play.png"))
        self.player_btn.setIconSize(QSize(35, 35))
        self.player_btn.clicked.connect(self.clicked_player_btn)

        self.replayer_btn.setFixedSize(100, 40)
        self.replayer_btn.setDisabled(True)
        self.replayer_btn.setStyleSheet(ui_style_const.toolbar_button_stytle)
        self.replayer_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/replay.png"))
        self.replayer_btn.setIconSize(QSize(30, 30))
        self.replayer_btn.clicked.connect(self.clicked_player_btn)

        self.data_btn.setFixedSize(100, 40)
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
        self.lineedit_type.setFixedHeight(40)
        self.lineedit_type.setAlignment(Qt.AlignCenter)
        label_count = QLabel(" 计 数： ")
        label_count.setFixedHeight(40)

        result, _ = self.load_recorded_num_from_json()
        if result is None:
            current_recorded_count = 1
        else:
            current_recorded_count = result
        self.lineedit_count = QLineEdit(str(current_recorded_count))
        self.lineedit_count.setFixedHeight(40)
        self.lineedit_count.setAlignment(Qt.AlignCenter)       
        self.lineedit_count.editingFinished.connect(lambda: self.lineedit_lose_focus(self.lineedit_count))
        self.lineedit_count.returnPressed.connect(lambda: self.validate_count(self.lineedit_count, True))

        self.barcode_scanner_box = QCheckBox("S/N:  ", self)
        self.barcode_scanner_box.setChecked(False)
        self.barcode_scanner_box.stateChanged.connect(self.clicked_scanner)

        self.lineedit_s_or_n = QLineEdit(self)
        self.lineedit_s_or_n.setDisabled(True)
        self.lineedit_s_or_n.setFixedHeight(40)
        self.lineedit_s_or_n.setAlignment(Qt.AlignCenter)
        self.lineedit_s_or_n.editingFinished.connect(lambda: self.validate_count(self.lineedit_s_or_n, False))

        vertical_line_1 = QFrame()
        vertical_line_2 = QFrame()
        vertical_line_3 = QFrame()
        vertical_line_4 = QFrame()
        vertical_line_5 = QFrame()
        vertical_line_6 = QFrame()
        vertical_line_1.setFrameShape(QFrame.VLine)
        vertical_line_2.setFrameShape(QFrame.VLine)
        vertical_line_3.setFrameShape(QFrame.VLine)
        vertical_line_4.setFrameShape(QFrame.VLine)
        vertical_line_5.setFrameShape(QFrame.VLine)
        vertical_line_6.setFrameShape(QFrame.VLine)
        h_spacer = QSpacerItem(30, 30, QSizePolicy.Expanding, QSizePolicy.Minimum)

        h_spacer_1 = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Minimum)
        h_spacer_2 = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Minimum)
        h_spacer_3 = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Minimum)
        h_spacer_4 = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Minimum)

        # Create and configure the toolbar layout
        toolbar_layout = QHBoxLayout()
        toolbar_layout.addWidget(self.player_btn)
        toolbar_layout.addWidget(vertical_line_1)
        toolbar_layout.addWidget(self.replayer_btn)
        toolbar_layout.addWidget(vertical_line_2)
        toolbar_layout.addWidget(self.data_btn)
        toolbar_layout.addWidget(vertical_line_3)
        toolbar_layout.addWidget(type_label)
        toolbar_layout.addWidget(self.lineedit_type)
        toolbar_layout.addItem(h_spacer_1)
        toolbar_layout.addWidget(vertical_line_4)
        toolbar_layout.addWidget(label_count)
        toolbar_layout.addWidget(self.lineedit_count)
        toolbar_layout.addItem(h_spacer_2)
        toolbar_layout.addWidget(vertical_line_5)
        toolbar_layout.addItem(h_spacer_3)
        toolbar_layout.addWidget(self.barcode_scanner_box)
        toolbar_layout.addWidget(self.lineedit_s_or_n)
        toolbar_layout.addItem(h_spacer_4)
        toolbar_layout.addWidget(vertical_line_6)
        toolbar_layout.addItem(h_spacer)

        toolbar_layout.setContentsMargins(4, 0, 0, 0)
        toolbar_layout.setSpacing(0)

        return toolbar_layout
    
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
        btn_area = self.create_waveform_btn_layout()
        self.line_graph.setLabel('left', 'Amplitude(V)')
        self.line_graph.setLabel('bottom', 'Time(s)')
        self.line_graph.showGrid(x=True, y=True)

        h_spacer_1 = QSpacerItem(70, 30, QSizePolicy.Minimum, QSizePolicy.Minimum)
        layout.addLayout(btn_area)
        layout.addItem(h_spacer_1)
        layout.addWidget(self.line_graph)
        layout.setContentsMargins(90, 20, 90, 30)

        return layout

    def create_waveform_btn_layout(self):
        """
            Create a button layout for waveform.

            This function generates a vertical layout containing two buttons: one labeled "OK" and the other labeled "NG".
            Each button is configured with an icon, stylesheet, fixed size, and icon size. The buttons are added to a QVBoxLayout,
            which is then returned.

            Returns:
                QVBoxLayout: A vertical layout containing the "OK" and "NG" buttons.
        """
        btn_layout = QVBoxLayout()
        self.ok_btn = QPushButton(" OK ")
        self.ng_btn = QPushButton(" NG ")
        self.ok_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/lvseyuan.png"))
        self.ok_btn.setStyleSheet(ui_style_const.sequence_qpushbutton_stytle)
        self.ok_btn.setFixedSize(200, 130)
        self.ok_btn.setIconSize(QSize(30, 30))
        self.ng_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/hongseyuan.png"))
        self.ng_btn.setStyleSheet(ui_style_const.sequence_qpushbutton_stytle)
        self.ng_btn.setFixedSize(200, 130)
        self.ng_btn.setIconSize(QSize(30, 30))
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.ng_btn)

        return btn_layout
    
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
            reg = r'^[0-9]*[a-z]*[A-Z]*$'
        # Check if the user input matches the regular expression
        if not re.match(reg,s_or_n_count):
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
        if self.barcode_scanner_box.isChecked():
            self.lineedit_s_or_n.setEnabled(True)
            device = self.get_match_hid_device()
            if device:
                if self.scanner_barcode_thread is None:
                    self.scanner_barcode_thread = threading.Thread(target=self.scan_barcode,
                                                                   args=(device,))
                    self.scanner_emitter.signal_emitter.connect(self.on_barcode_received)
                    self.scanner_barcode_thread.start()
        else:
            self.lineedit_s_or_n.setDisabled(True)
            if self.scanner_barcode_thread and self.scanner_barcode_thread.is_alive():
                self.barcode_scanner.stop_scanning()
                self.scanner_barcode_thread.join()
                self.scanner_emitter.signal_emitter.disconnect(self.on_barcode_received)
                self.scanner_barcode_thread = None

    def scan_barcode(self, device):
        barcode = self.barcode_scanner.read_raw_data(device)
        if barcode:
            self.scanner_emitter.signal_emitter.emit(barcode)

    def on_barcode_received(self, barcode):
        if barcode:
            self.lineedit_s_or_n.setText(barcode)
            self.clicked_player_btn()
            self.scanner_emitter.signal_emitter.disconnect(self.on_barcode_received)
            if self.scanner_barcode_thread and self.scanner_barcode_thread.is_alive():
                self.scanner_barcode_thread.join()
            self.scanner_barcode_thread = None

    def clicked_scanner(self):
        if self.barcode_scanner_box.isChecked():
            self.lineedit_s_or_n.setEnabled(True)
            self.scanner_barcode_process()
        else:
            self.lineedit_s_or_n.setDisabled(True)

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

    # def get_model_info(self, selected_model):
    #     """
    #         Retrieves model path and configuration path information based on the selected model.

    #         This function queries the model management system to obtain the storage path of the model and its configuration.
    #         If the query is successful, it returns the model path and configuration path. If the query fails, it logs
    #
    #         an error message and returns an error code and a failure message.

    #         Parameters:
    #         selected_model (str): The name of the model to query.

    #         Returns:
    #         tuple: A tuple containing the error code and the query result or a failure message.
    #             If the query is successful, the query result is a tuple containing the model path and configuration path.
    #             If the query fails, it returns an error code and a failure message.
    #     """
    #     query_code, query_result = TrainingModelManagement().get_model_path_from_db(selected_model)
    #     if query_code == error_code.OK:
    #         model_path, config_path = query_result[0]
    #         return error_code.OK, (model_path, config_path)
    #     else:
    #         self.default_logger.error(f"Failed to get the model {selected_model} information.")
    #         return error_code.INVALID_QUERY, "Failed to get the model information."

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
        current_recorded_count = self.save_recorded_num_to_json("ok_ng")
        self.lineedit_count.setText(str(current_recorded_count))
        self.insert_data_into_db()
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
        self.scanner_barcode_process()
        
    # def clear_plg(self):
    #     self.line_graph.clear()
    #     self.analyse_layout.signal_analyse_dialog.spl_wnd.waveform_plot.clear()
    #     self.analyse_layout.signal_analyse_dialog.spl_wnd.spl_plot.clear()
    #     self.analyse_layout.signal_analyse_dialog.frequency_wnd.fr_plot.clear()
    #     self.analyse_layout.signal_analyse_dialog.distortion_wnd.thd_plot.clear()
    #     self.analyse_layout.ai_analyse_score_lineedit.clear()
    #     QApplication.processEvents()

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
            info = result["stimulus_info"]
            path = result["stimulus_signal_path"]
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

    def save_recorded_num_to_json(self, start_position = None):
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
            self.recorded_signal_info["labels"] = 'NG'
        move_recorded_path = self.move_wav_to_dir(self.recorded_signal_info["labels"])
        file_path = self.recorded_signal_info["file_path"]
        if move_recorded_path:
            file_path = move_recorded_path
        self.recorded_signal_info["file_path"] = file_path.replace(DEFAULT_DIR, "")
        save_code, msg = RecordingManager().save_signal_info_to_db(self.recorded_signal_info, self.stimulus_info)
        if save_code == error_code.OK:
            self.default_logger.info("Recorded signal successfully insert.")
        else:
            self.default_logger.error("Failed insert recorded signal.")

    def move_wav_to_dir(self, label):
        dir_paths = [model_consts.STORED_RECORDED_OK_PATH, model_consts.STORED_RECORDED_NG_PATH]
        for path in dir_paths:
            if not os.path.exists(path):
                os.makedirs(path)
        file_name = os.path.basename(self.recorded_path)
        target_path = ''
        if file_name:
            if label == 'OK':
                target_path = model_consts.STORED_RECORDED_OK_PATH + "/" + file_name
            elif label == 'NG':
                target_path = model_consts.STORED_RECORDED_NG_PATH + "/" + file_name
            shutil.move(self.recorded_path, target_path)
        return target_path

    # def clicked_analyse_btn(self):
    #     selected_model = self.analyse.model_combo_box.currentText()
    #     code, result = self.get_model_info(selected_model)
    #     if code != error_code.OK or not os.path.exists(result[0]):
    #         if self.model_missing_popup():
    #             return
    #     else:
    #         self.save_analyse_model(selected_model)
    #         model_path, config_path = result
    #         kwargs = {"config_path": config_path}
    #         result_text = self.model_predict(model_path, **kwargs)
    #         self.analyse.ai_analyse_score_lineedit.setPlainText(result_text)

    # def model_missing_popup(self):
    #     model_missing_msg = QMessageBox(self)
    #     model_missing_msg.setIcon(QMessageBox.Critical)
    #     model_missing_msg.setText("模型不存在，请重新选择!")
    #     model_missing_msg.setWindowTitle("模型加载失败")
    #     model_missing_msg.setStandardButtons(QMessageBox.Ok)
    #     button = model_missing_msg.exec_()
    #     return button == QMessageBox.Ok

    # @staticmethod
    # def save_analyse_model(selected_model):
    #     file_path = DEFAULT_DIR + "ui/ui_config/analyse_model.txt"
    #     with open(file_path, 'w') as f:
    #         f.write(selected_model)

    # @staticmethod
    # def load_analyse_model():
    #     file_path = DEFAULT_DIR + "ui/ui_config/analyse_model.txt"
    #     if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
    #         return ""
    #     with open(file_path, 'r') as f:
    #         model_name = f.read().strip()
    #         return model_name

    # def model_predict(self, model_path, **kwargs):
    #     ret_str = predict(self.recorded_path, load_model_path=model_path, **kwargs)
    #     ret_dict = json.loads(ret_str)
    #     predict_result = ret_dict["result"]
    #     predict_label = predict_result[0][1]
    #     ok_scores = float(predict_result[0][2]) * 100
    #     ng_scores = 100 - ok_scores
    #     result_text = (
    #         f"评分：\n"
    #         f"OK Score: {ok_scores:.2f}%\n"
    #         f"NG Score: {ng_scores:.2f}%\n"
    #         f"评分结果: {predict_label}"
    #     )
    #     return result_text

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
        if self.player_status_flag:
            self.line_graph.clear()
        self.player_status_flag = True
        self.player_btn.setDisabled(True)
        self.update_player_icon()
        self.analysis_config = self.get_sequence_config_from_json()
        QApplication.processEvents()
        # self.analyse_layout.signal_analyse_dialog.distortion_wnd.refresh_stimulus_flag = self.refresh_stimulus_flag
        if self.refresh_stimulus_flag:
            self.stimulus_info, self.stimulus_signal = self.get_stimulus_from_config()
            self.refresh_stimulus_flag = False
        sample_rate = self.stimulus_info["sample_rate"]
        stimulus_dict, recorded_dict = self.get_stimulus_recorded_dict(sample_rate)
        self.recorded_path, self.recorded_signal_info = self.get_recorded_info()
        sap = SoundcardAudioProcessor()
        record_code, recorded_signal = sap.sd_play_rec(recorded_dict, stimulus_dict, self.recorded_path)
        if record_code == error_code.OK:
            self.plot_line_graph(recorded_signal, self.line_graph, sample_rate)
            self.signal_info = {"stimulus_signal": self.stimulus_signal,
                                "recorded_signal": recorded_signal,
                                "sample_rate": sample_rate,
                                "recorded_path": self.recorded_path}
            self.recorded_signal_info["sample_rate"] = sample_rate

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
                class_instance.signal_info = self.signal_info
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
            item_sort_list =  self.analysis_config.get("display_sequence", [])
            for key in item_sort_list:
                key_config = self.analysis_config.get(key)
                if isinstance(key_config, dict):
                    self.instance_analysis_class(key, key_config["type"], key_config)
            for instance in self.analysis_window:
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
                    instance.calculate_ai_scores()
                    instance.show()
                instance.setGeometry(width, height, 600, 500)
                instance.setMinimumSize(QSize(600, 500))
                width += 20
                height += 20
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

    # def load_model_name_from_db(self):
    #     model_list = []
    #     query_code, query_result = TrainingModelManagement().get_all_model_name_from_db()
    #     if query_code == error_code.OK:
    #         for idx, name in enumerate(query_result):
    #             query_result_idx = query_result[idx]
    #             input_dim = int(query_result_idx[1].split(' ')[0])
    #             if input_dim == len(self.stimulus_signal):
    #                 model_list.append(query_result_idx[0])
    #     return model_list

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
        stimulus_dict = {"data": self.stimulus_signal,
                         "amplitude": self.stimulus_info["amplitude"],
                         "sr": sample_rate
                         }
        recorded_dict = {"channels": 1,
                         "sr": sample_rate,
                         "num_frames": len(self.stimulus_signal) + int(prolong * sample_rate),
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
        painter.drawRect(1, 0, width - 2, 40)
        painter.end()


class ScannerEmitter(QObject):
    signal_emitter = pyqtSignal(str)


# class AnalyseWindow(QDialog):
#
#     def __init__(self):
#         super().__init__()
#         self.analyse_btn = QPushButton(" 分 析 ")
#         self.ai_analyse_score_lineedit = QTextEdit()
#         self.signal_info = None
#
#         self.init_ui()
#
#     def init_ui(self):
#         self.setStyleSheet(ui_style_const.qcombobox_stytle +
#                            ui_style_const.qlabel_stytle +
#                            ui_style_const.qpushbutton_stytle)
#         self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
#         self.setWindowTitle("AI 分析")
#         ai_analyse_layout = self.create_ai_analyse_layout()
#         self.setLayout(ai_analyse_layout)
#
#     def create_ai_analyse_layout(self):
#         ai_analyse_layout = QVBoxLayout()
#
#         ai_title_layout = QHBoxLayout()
#         h_title_space = QSpacerItem(30, 30, QSizePolicy.Expanding, QSizePolicy.Minimum)
#         ai_title_layout.addItem(h_title_space)
#
#         model_layout = QHBoxLayout()
#         model_label = QLabel(" 模 型: ")
#         model_label.setFixedSize(65, 30)
#         self.model_combo_box = QComboBox(self)
#         self.model_combo_box.setFixedHeight(25)
#         model_layout.addWidget(model_label)
#         model_layout.addWidget(self.model_combo_box)
#         model_layout.setSpacing(15)
#
#         analyse_btn_layout = QHBoxLayout()
#         self.analyse_btn.setFixedSize(100, 30)
#         h_analyse_btn_space_left = QSpacerItem(30, 30, QSizePolicy.Expanding, QSizePolicy.Minimum)
#         h_analyse_btn_space_right = QSpacerItem(30, 30, QSizePolicy.Expanding, QSizePolicy.Minimum)
#         analyse_btn_layout.addItem(h_analyse_btn_space_left)
#         analyse_btn_layout.addWidget(self.analyse_btn)
#         analyse_btn_layout.addItem(h_analyse_btn_space_right)
#
#         analyse_score_layout = QHBoxLayout()
#         self.ai_analyse_score_lineedit.setAlignment(Qt.AlignCenter)
#         self.ai_analyse_score_lineedit.setDisabled(True)
#         self.ai_analyse_score_lineedit.setMaximumWidth(600)
#         self.ai_analyse_score_lineedit.setStyleSheet("font-size: 23pt;")
#         analyse_score_layout.addWidget(self.ai_analyse_score_lineedit)
#         analyse_score_layout.setContentsMargins(20, 0, 20, 0)
#
#         v_ai_analyse_top_space = QSpacerItem(30, 50, QSizePolicy.Minimum, QSizePolicy.Minimum)
#         v_ai_analyse_center_space = QSpacerItem(30, 30, QSizePolicy.Minimum, QSizePolicy.Minimum)
#         v_ai_analyse_bottom_space = QSpacerItem(30, 30, QSizePolicy.Minimum, QSizePolicy.Minimum)
#
#         ai_analyse_layout.addLayout(ai_title_layout)
#         ai_analyse_layout.addLayout(model_layout)
#         ai_analyse_layout.addItem(v_ai_analyse_top_space)
#         ai_analyse_layout.addLayout(analyse_btn_layout)
#         ai_analyse_layout.addItem(v_ai_analyse_center_space)
#         ai_analyse_layout.addLayout(analyse_score_layout)
#         ai_analyse_layout.addItem(v_ai_analyse_bottom_space)
#
#         return ai_analyse_layout


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # stimulus_info = {'name': 'stimulus_chirps_1', 'use_custom_stimulus': True, 'stimulus_method': 'chirp',
    #                  'stimulus_type': 'log', 'start_freq': 1000, 'stop_freq': 80, 'total_time': 3.0, 'repeat_times': 1,
    #                  'num_steps': 1, 'amplitude_type': 'RMS', 'amplitude': 0.1, 'sample_rate': 44100}
    # stimulus_info = {'name': 'stimulus_chirps_1', 'use_custom_stimulus': True, 'stimulus_method': 'chirp', 'stimulus_type': 'log',
    #  'start_freq': 80, 'stop_freq': 1000, 'total_time': 3.0, 'repeat_times': 1, 'num_steps': 1, 'amplitude_type': 'RMS',
    #  'amplitude': 0.7, 'sample_rate': 44100}
    stimulus_info = {'name': 'stimulus_chirps_1', 'use_custom_stimulus': True, 'stimulus_method': 'chirp',
     'stimulus_type': 'mirror_log', 'start_freq': 80, 'stop_freq': 1000, 'total_time': 3.0, 'repeat_times': 1,
     'num_steps': 1, 'amplitude_type': 'RMS', 'amplitude': 0.1, 'sample_rate': 44100}

    # stimulus_signal, sr = librosa.load("../audio_data/stimulus/stimulus.wav", sr=44100)
    # stimulus_signal, sr = librosa.load("../audio_data/stimulus/stimulus111.wav", sr=44100)
    stimulus_signal, sr = librosa.load("../audio_data/stimulus/stimulus_mirror.wav", sr=44100)
    window = SequenceWindow()
    window.stimulus_info = stimulus_info
    window.stimulus_signal = stimulus_signal
    window.show()
    app.exec()