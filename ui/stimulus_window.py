import json
import os.path
import sys

import numpy as np
import pyqtgraph
import sounddevice as sd
from PyQt5.QtCore import Qt
from scipy.io import wavfile
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtWidgets import QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QMessageBox
from PyQt5.QtWidgets import QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QLineEdit
from PyQt5.QtWidgets import QSizePolicy, QSpinBox, QVBoxLayout, QAbstractSpinBox


from base.data_struct.data_deal_struct import DataDealStruct
from base.file_ops import FileOps
from base.load_audio import load_audio_simple, save_audio_simple
from base.log_manager import LogManager
from base.pre_processing.swept_sine_chirps import StimulusSignal
from base.soundcard_audio_processor import SoundcardAudioProcessor
from base.soundcard_calibration_manager import SoundcardCalibrationManager
from base.stimulus_signal_management import StimulusSignalManagement
from consts import error_code, model_consts, ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.load_stimulus_dialog import LoadStimulusDialog


class StimulusWindow(QDialog):
    # Define stimulus signal types
    STIMULUS_DICT = {
        "啁啾": {"name": "chirp", "sub_list": ["对数镜像", "线性镜像", "对数", "线性"]},
        "步进": {"name": "step", "sub_list": ["对数", "线性"]},
        "噪音": {"name": "noise", "sub_list": ["白噪音", "粉噪音"]},
    }
    # Mapping for stimulus signal code names
    STIMULUS_DICT_2 = {
        "对数": "log",
        "线性": "linear",
        "对数镜像": "mirror_log",
        "线性镜像": "mirror_linear",
        "白噪音": "white_noise",
        "粉噪音": "pink_noise",
    }

    def __init__(self):
        """Initialize stimulus window with default configurations"""
        super().__init__()
        # initialize stimulus signal type， and create variable to store stimulus signal data
        self.data_struct = DataDealStruct()
        # self.stimulus_info = {"name": "stimulus_chirps_1", "use_custom_stimulus": False}
        self.speaker = None
        self.load_wav_path = ""
        self.start_custom_check_status = self.data_struct.stimulus_info.get("use_custom_stimulus", False)
        self.load_stimulus_signal_path = None
        self.refresh_stimulus_info = False
        self.is_close_window = False
        self.default_logger = LogManager.set_log_handler("core")
        self.box_checked_enable_dict = {}
        self.box_checked_disable_list = []

        # create variable to set stimulus signal data
        self.stimulus_method_combo_box = QComboBox()
        self.stimulus_type_combo_box = QComboBox()
        self.start_freq_box = QSpinBox()
        self.stop_freq_box = QSpinBox()
        self.total_time_box = QDoubleSpinBox()
        self.repeat_box = QSpinBox()
        self.step_box = QSpinBox()
        self.voltage_combo_box = QComboBox()
        self.voltage_spin_box = QDoubleSpinBox()
        self.sample_rate_combo_box = QComboBox()
        # rcreate variable to control whether to use frequency or step
        self.frequency_group_box = self.create_frequency_group_box()
        self.step_group_box = self.create_step_group_box()
        self.init_ui()

        stimulus_info = self.get_stimulus_info_from_json()
        self.update_stimulus_ui_value(stimulus_info)
        self.old_stimulus_signal_length = self.data_struct.stimulus_info.get(
            "total_time"
        ) * self.data_struct.stimulus_info.get("sample_rate")

    def init_ui(self):
        # set window titlebar stytle
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setWindowTitle("激励信号")
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setFixedSize(540, 700)

        self.plot_stimulus = pyqtgraph.PlotWidget()
        self.plot_stimulus.setBackground("white")
        self.plot_stimulus.resize(400, 170)

        # create layout to strore custom button layout
        custom_stimulus_layout = QGridLayout()
        self.custom_chk_box = QCheckBox("自定义")
        self.custom_chk_box.stateChanged.connect(lambda: self.change_custom_chk_box(self.custom_chk_box.isChecked()))
        load_config_btn = QPushButton("导入配置")
        load_config_btn.clicked.connect(self.load_config_btn_clicked)
        save_config_btn = QPushButton("保存配置")
        save_config_btn.clicked.connect(self.save_config_btn_clicked)
        load_wav_btn = QPushButton("导入音频")
        load_wav_btn.clicked.connect(self.load_wav_btn_clicked)
        load_wav_btn.setDisabled(True)
        save_wav_btn = QPushButton("保存音频")
        save_wav_btn.clicked.connect(self.save_wav_btn_clicked)
        custom_stimulus_layout.addWidget(self.custom_chk_box, 0, 0)
        custom_stimulus_layout.addWidget(load_config_btn, 0, 1)
        custom_stimulus_layout.addWidget(save_config_btn, 1, 1)
        custom_stimulus_layout.addWidget(load_wav_btn, 0, 3)
        custom_stimulus_layout.addWidget(save_wav_btn, 1, 3)
        custom_stimulus_layout.setContentsMargins(0, 0, 10, 0)
        custom_stimulus_layout.setColumnStretch(2, 1)

        output_layout = QHBoxLayout()
        # voltage_group_box and sample_rate_group_box horizontal direction towards it
        voltage_group_box = self.create_voltage_group_box()
        sample_rate_group_box = self.create_sample_rate_group_box()
        output_layout.addWidget(voltage_group_box)
        output_layout.addWidget(sample_rate_group_box)

        stimulus_type_group_box = self.create_stimulus_type_group_box()
        time_group_box = self.create_time_group_box()
        # Disable step_group_box during initialization
        self.step_group_box.setDisabled(True)
        self.step_group_box.setStyleSheet("color: rgb(162, 162, 162);")
        function_btn_layout = self.create_function_btn_layout()

        layout = QVBoxLayout()
        layout.addWidget(self.plot_stimulus)
        layout.addStretch()
        layout.addLayout(custom_stimulus_layout)
        layout.addWidget(stimulus_type_group_box)
        layout.addWidget(self.frequency_group_box)
        layout.addWidget(time_group_box)
        layout.addWidget(self.step_group_box)
        layout.addStretch()
        layout.addLayout(output_layout)
        layout.addStretch()
        layout.addLayout(function_btn_layout)
        layout.setContentsMargins(25, 10, 25, 20)

        # Set custom_chk_box in different states, can use the function list
        self.box_checked_enable_dict = {
            "chirp": [
                load_config_btn,
                save_config_btn,
                stimulus_type_group_box,
                self.frequency_group_box,
                time_group_box,
            ],
            "noise": [load_config_btn, save_config_btn, stimulus_type_group_box, time_group_box],
            "step": [
                load_config_btn,
                save_config_btn,
                stimulus_type_group_box,
                self.frequency_group_box,
                time_group_box,
                self.step_group_box,
            ],
        }
        self.box_checked_disable_list = [load_wav_btn]

        self.setLayout(layout)

        self.setStyleSheet(
            ui_style_const.qcombobox_stytle
            + ui_style_const.qpushbutton_stytle
            + ui_style_const.qspinbox_stytle
            + ui_style_const.qdoublespinbox_stytle
            + ui_style_const.qlabel_stytle
            + ui_style_const.qcheckbox_stytle
            + ui_style_const.qgroupbox_stytle
        )

    def create_stimulus_type_group_box(self):
        """
        Create a QGroupBox for stimulus signal type selection.

        This method constructs a group box containing combo boxes for selecting different stimulus signal types.
        It configures layout components, establishes signal-slot connections for handling selection changes,
        and initializes default values from the stimulus dictionary.

        Returns:
            QGroupBox: Configured group box containing stimulus type selection components.
        """
        stimulus_type_group_box = QGroupBox("激励信号类型")
        self.stimulus_type_combo_box.currentTextChanged.connect(self.update_stimulus_info_from_stimulus_type_combo_box)
        self.stimulus_method_combo_box.currentTextChanged.connect(self.set_stimulus_type_connection)
        self.stimulus_method_combo_box.addItems(["啁啾", "步进", "噪音"])
        stimulus_type_layout = QHBoxLayout()
        stimulus_type_layout.addWidget(self.stimulus_method_combo_box)
        stimulus_type_layout.addWidget(self.stimulus_type_combo_box)
        stimulus_type_layout.setContentsMargins(10, 10, 10, 10)
        stimulus_type_layout.setSpacing(20)
        stimulus_type_group_box.setLayout(stimulus_type_layout)
        return stimulus_type_group_box

    def create_frequency_group_box(self):
        """
        Create a frequency range configuration group box

        Constructs a QGroupBox containing start/stop frequency spinboxes for user input.
        Features:
        - Two QDoubleSpinBox with Hz suffix
        - Value range: 10-24000 Hz
        - Default values: 80 Hz (start), 2000 Hz (stop)
        - Auto-triggers stimulus_changed signal on edit completion

        Returns:
            QGroupBox: Configured group box with frequency range widgets
        """
        frequency_group_box = QGroupBox("频率范围 (10 - 24000Hz)")
        start_freq_label = QLabel("起始频率：")
        self.start_freq_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.start_freq_box.setSuffix(" Hz")
        self.start_freq_box.setRange(10, 24000)
        self.start_freq_box.valueChanged.connect(
            lambda: self.update_stimulus_info_from_object(self.start_freq_box, "start_freq")
        )
        self.start_freq_box.setMinimumWidth(100)
        stop_freq_label = QLabel("截止频率：")
        self.stop_freq_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.stop_freq_box.setSuffix(" Hz")
        self.stop_freq_box.setRange(10, 24000)
        self.stop_freq_box.setMinimumWidth(100)
        self.stop_freq_box.valueChanged.connect(
            lambda: self.update_stimulus_info_from_object(self.stop_freq_box, "stop_freq")
        )
        frequency_layout = QHBoxLayout()
        frequency_layout.addWidget(start_freq_label)
        frequency_layout.addWidget(self.start_freq_box)
        frequency_layout.addWidget(stop_freq_label)
        frequency_layout.addWidget(self.stop_freq_box)
        frequency_group_box.setLayout(frequency_layout)
        frequency_layout.setContentsMargins(10, 10, 10, 10)
        frequency_layout.setSpacing(20)
        return frequency_group_box

    def create_time_group_box(self):
        """
        Create time parameters configuration group box
        Constructs a QGroupBox containing signal duration and repetition controls for configuring
        stimulus timing parameters. All value changes will trigger the stimulus_changed signal.

        Returns:
            QGroupBox: Container widget with horizontal layout of time configuration controls
        """
        time_group_box = QGroupBox()
        total_time_label = QLabel("信号时长：")
        self.total_time_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.total_time_box.setSuffix(" s")
        self.total_time_box.setDecimals(1)  # Allow one decimal place
        self.total_time_box.setRange(0.5, 60)  # Set range 0.5-60 seconds
        self.total_time_box.setSingleStep(0.5)
        self.total_time_box.setMinimumWidth(100)
        self.total_time_box.valueChanged.connect(
            lambda: self.update_stimulus_info_from_object(self.total_time_box, "total_time")
        )
        self.repeat_box.valueChanged.connect(
            lambda: self.update_stimulus_info_from_object(self.repeat_box, "repeat_times")
        )
        repeat_label = QLabel("信号重复：")
        self.repeat_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.repeat_box.setRange(1, 10)
        self.repeat_box.setSuffix(" 次")
        self.repeat_box.setMinimumWidth(100)
        time_layout = QHBoxLayout()
        time_layout.addWidget(total_time_label)
        time_layout.addWidget(self.total_time_box)
        time_layout.addWidget(repeat_label)
        time_layout.addWidget(self.repeat_box)
        time_layout.setContentsMargins(10, 10, 10, 10)
        time_layout.setSpacing(20)
        time_group_box.setLayout(time_layout)
        return time_group_box

    def create_step_group_box(self):
        """
        Create and configure the step setting group box
        Return:
            QGroupBox: A group box that contains a step number setting control with the following components:
            - QLabel Displays "Step quantity"
            - QSpinBox used to set the step value (range 1-100)
        Signal connection:
            The editingFinished signal of step_box is connected to the stimulus_changed method
        """
        step_group_box = QGroupBox()
        step_label = QLabel("步进数量")
        self.step_box.setFixedSize(100, 30)
        self.step_box.setRange(1, 100)
        self.step_box.valueChanged.connect(lambda: self.update_stimulus_info_from_object(self.step_box, "num_steps"))
        step_layout = QHBoxLayout()
        step_layout.addWidget(step_label)
        step_layout.addWidget(self.step_box)
        step_layout.setContentsMargins(10, 10, 10, 10)
        step_group_box.setLayout(step_layout)
        return step_group_box

    def create_voltage_group_box(self):
        """
            Creates a QGroupBox for setting the output voltage.

            This function creates a QGroupBox containing a combo box and a spin box for selecting the type of output
        voltage (RMS or Peak) and setting the voltage value.
            When the values in the combo box or spin box change, the `stimulus_changed` signal is triggered.

            Returns:
                QGroupBox: Returns a configured QGroupBox containing the controls for setting the output voltage.
        """
        voltage_group_box = QGroupBox("输出电压")
        self.voltage_combo_box.currentTextChanged.connect(self.update_stimulus_info_from_voltage_combo_box)
        self.voltage_combo_box.addItems(["RMS", "Peak"])
        self.voltage_spin_box.setSuffix(" V")
        # self.voltage_spin_box.setValue(self.load_voltage_from_txt())
        max_input_voltage = self.get_max_input_voltage()
        self.voltage_spin_box.setSingleStep(0.1)
        self.voltage_spin_box.setRange(0.1, max_input_voltage)
        self.voltage_spin_box.valueChanged.connect(
            lambda: self.update_stimulus_info_from_object(self.voltage_spin_box, "voltage")
        )
        self.voltage_spin_box.valueChanged.connect(self.update_amplitude)

        voltage_layout = QHBoxLayout()
        voltage_layout.addWidget(self.voltage_combo_box)
        voltage_layout.addWidget(self.voltage_spin_box)
        voltage_layout.setContentsMargins(10, 10, 6, 10)
        voltage_group_box.setLayout(voltage_layout)

        return voltage_group_box

    def create_sample_rate_group_box(self):
        """
        Creates a QGroupBox containing a sample rate selection combo box.

        This function generates a QGroupBox that includes a combo box for selecting the sample rate.
        The combo box options are 44100 and 48000. When the user changes the sample rate,
        it triggers the `stimulus_changed` signal.

        Returns:
            QGroupBox: A QGroupBox object containing the sample rate selection combo box.
        """
        sample_rate_group_box = QGroupBox("采样率")
        self.sample_rate_combo_box.currentTextChanged.connect(self.update_stimulus_info_from_sample_rate_combo_box)
        self.sample_rate_combo_box.addItems(["44100", "48000"])
        sample_rate_layout = QHBoxLayout()
        sample_rate_layout.addWidget(self.sample_rate_combo_box)
        sample_rate_layout.setContentsMargins(10, 10, 10, 10)
        sample_rate_group_box.setLayout(sample_rate_layout)
        return sample_rate_group_box

    def create_function_btn_layout(self):
        """
        Creates and returns a horizontal layout containing functional buttons.

        The layout includes three buttons: Play, Confirm, and Cancel. A horizontal spacer is used
        to ensure the buttons are properly distributed within the layout.

        Returns:
            QHBoxLayout: A horizontal layout object containing the functional buttons.
        """
        function_btn_layout = QHBoxLayout()
        default_config_btn = QPushButton(" 默认配置 ")
        default_config_btn.setStyleSheet("padding: 3px")
        default_config_btn.clicked.connect(self.default_config_btn_clicked)
        play_btn = QPushButton(" 试  播 ")
        play_btn.setStyleSheet("padding: 3px")
        play_btn.clicked.connect(self.play_btn_clicked)
        ok_btn = QPushButton(" 确  认 ")
        ok_btn.setStyleSheet("padding: 3px")
        ok_btn.clicked.connect(self.ok_btn_clicked)
        cancel_btn = QPushButton(" 取  消 ")
        cancel_btn.setStyleSheet("padding: 3px")
        cancel_btn.clicked.connect(self.cancel_btn_clicked)
        function_btn_layout.addWidget(default_config_btn)
        function_btn_layout.addStretch(20)
        function_btn_layout.addWidget(play_btn)
        function_btn_layout.addWidget(cancel_btn)
        function_btn_layout.addWidget(ok_btn)
        function_btn_layout.setSpacing(20)
        return function_btn_layout

    def update_stimulus_info_from_stimulus_type_combo_box(self):
        stimulus_type = self.stimulus_type_combo_box.currentText()
        self.data_struct.stimulus_info["stimulus_type"] = self.STIMULUS_DICT_2.get(stimulus_type)
        if (
            self.data_struct.stimulus_info.get("use_custom_stimulus", False)
            and 0 != self.stimulus_type_combo_box.count()
        ):
            self.create_signal_from_stimulus_info()
            self.graph_stimulus()

    def update_stimulus_info_from_object(self, object: QAbstractSpinBox, data_type: str):
        if data_type == "voltage" or data_type == "total_time":
            self.data_struct.stimulus_info[data_type] = float(object.value())
        else:
            self.data_struct.stimulus_info[data_type] = int(object.value())
        if self.data_struct.stimulus_info.get("use_custom_stimulus", False):
            self.create_signal_from_stimulus_info()
            self.graph_stimulus()

    def update_stimulus_info_from_voltage_combo_box(self):
        self.data_struct.stimulus_info["voltage_type"] = self.voltage_combo_box.currentText()

    def update_amplitude(self):
        self.data_struct.stimulus_info["amplitude"] = self.get_predict_amplitude(self.voltage_spin_box.value())

    def update_stimulus_info_from_sample_rate_combo_box(self):
        self.data_struct.stimulus_info["sample_rate"] = int(self.sample_rate_combo_box.currentText())
        if self.data_struct.stimulus_info.get("use_custom_stimulus", False):
            self.create_signal_from_stimulus_info()
            self.graph_stimulus()

    def change_custom_chk_box(self, custom_box_checked):
        """
            Updates the enabled/disabled state and style of related widgets based on the custom checkbox state.

            Parameters:
            custom_box_checked (bool): The checked state of the custom checkbox. If True, the checkbox is checked;
        if False, it is unchecked.
        """
        list_name = self.STIMULUS_DICT[self.stimulus_method_combo_box.currentText()]["name"]
        for widget in self.box_checked_enable_dict[list_name]:
            widget.setEnabled(custom_box_checked)
            if custom_box_checked:
                widget.setStyleSheet("color: rgb(0, 0, 0);")
            else:
                widget.setStyleSheet("color: rgb(162, 162, 162);")
        for widget in self.box_checked_disable_list:
            widget.setDisabled(custom_box_checked)
            if custom_box_checked:
                widget.setStyleSheet("color: rgb(162, 162, 162);")
            else:
                widget.setStyleSheet("color: rgb(0, 0, 0);")
        self.data_struct.stimulus_info["use_custom_stimulus"] = custom_box_checked
        if custom_box_checked:
            self.create_signal_from_stimulus_info()
        else:
            self.load_wav_path = self.load_stimulus_signal_path
            self.data_struct.stimulus_data, _ = load_audio_simple(
                self.load_wav_path, self.data_struct.stimulus_info["sample_rate"]
            )
        self.graph_stimulus()

    def set_stimulus_type_connection(self):
        """
            Sets the content of the stimulus type combo box and adjusts the state and style of related controls based on
         the selected stimulus method.

            This function first retrieves the currently selected stimulus method from `stimulus_method_combo_box`, then
        fetches the corresponding sublist from `STIMULUS_DICT`,and adds it to `stimulus_type_combo_box`. Next, it
        enables or disables related controls and sets their styles based on the selected stimulus method.
        """
        # Get the currently selected stimulus method
        stimulus_method = self.stimulus_method_combo_box.currentText()
        self.data_struct.stimulus_info["stimulus_method"] = self.STIMULUS_DICT[stimulus_method]["name"]
        self.stimulus_type_combo_box.clear()
        # Fetch the corresponding sublist from the dictionary and add it to the stimulus type combo box
        stimulus_item = self.STIMULUS_DICT.get(stimulus_method, {})
        self.stimulus_type_combo_box.addItems(stimulus_item.get("sub_list", []))
        # Adjust the state and style of related controls based on the selected stimulus method
        if stimulus_method == "啁啾":
            self.step_group_box.setDisabled(True)
            self.frequency_group_box.setEnabled(True)
            self.step_group_box.setStyleSheet("color: rgb(162, 162, 162);")
            self.frequency_group_box.setStyleSheet("color: rgb(0, 0, 0);")
        elif stimulus_method == "噪音":
            self.frequency_group_box.setDisabled(True)
            self.step_group_box.setDisabled(True)
            self.step_group_box.setStyleSheet("color: rgb(162, 162, 162);")
            self.frequency_group_box.setStyleSheet("color: rgb(162, 162, 162);")
        else:
            self.frequency_group_box.setEnabled(True)
            self.step_group_box.setEnabled(True)
            self.step_group_box.setStyleSheet("color: rgb(0, 0, 0);")
            self.frequency_group_box.setStyleSheet("color: rgb(0, 0, 0);")

    def get_predict_amplitude(self, target_voltage):
        """
        Retrieves the predicted amplitude based on the target voltage.

        This method uses the `SoundcardCalibrationManager` to calibrate the amplitude
        for the given target voltage. It reads calibration coefficients from a JSON file
        and returns the predicted amplitude if the calibration is successful. Otherwise, it returns 0.0.

        Parameters:
        target_voltage (float): The target voltage used for calibration.

        Returns:
        float: The predicted amplitude if calibration is successful; otherwise, 0.0.
        """
        code, result_amplitude = SoundcardCalibrationManager().calibrate_amplitude(
            target_voltage, json_file_name="calibration_coefficients.json"
        )
        if code == error_code.OK:
            predict_amplitude, max_voltage = result_amplitude
            return predict_amplitude
        else:
            return 0.0

    def get_max_input_voltage(self):
        """
        Retrieves the maximum input voltage from the calibration coefficients.
        """
        code, data = SoundcardCalibrationManager().load_data_from_json("calibration_coefficients.json")
        if code == error_code.OK:
            return data["max_voltage"]
        else:
            self.is_close_window = True
            return 0.0

    def create_signal_from_stimulus_info(self):
        """
        Generates a signal based on the stimulus information.

        This method selects the appropriate signal generation function based on the `stimulus_method` field
        in `stimulus_info` and calls the function to generate the signal. The generated signal is
        stored in `stimulus_signal`.

        Parameters:
            No explicit parameters, but relies on the following fields in the `stimulus_info` dictionary:
                - stimulus_method: A string specifying the signal generation method, with possible values "chirp",
             "step", or "noise".
                - Other fields: Depending on the `stimulus_method`, additional parameters may be required, which
            are passed to the respective signal generation function.

        Returns:
            No explicit return value, but updates the `stimulus_signal` attribute with the generated signal.
        """
        create_function_dict = {
            "chirp": StimulusSignal().generate_chirps,
            "step": StimulusSignal().generate_steps,
            "noise": StimulusSignal().generate_noise,
        }
        create_function = create_function_dict.get(self.data_struct.stimulus_info["stimulus_method"])
        self.data_struct.stimulus_data, _ = create_function(**self.data_struct.stimulus_info)

    def save_stimulus_to_json(self):
        """
        Saves the stimulus signal and its related information to a JSON file.

        This function performs the following steps:
         -Constructs the path for the JSON file.
         -Generates the name of the stimulus signal based on the stimulus information.
         -Saves the stimulus signal as a WAV file.
         -Saves the stimulus information and WAV file path to the JSON file.

        Parameters:
        self: The class instance containing stimulus information (stimulus_info) and stimulus signal (stimulus_signal).
        """
        # Construct the path for the JSON file
        # Generate the name of the stimulus signal based on the stimulus information
        stimulus_name = "_".join(str(value) for value in self.data_struct.stimulus_info.values())
        # Construct the path for the WAV file and save the stimulus signal as a WAV file
        stimulus_signal_path = model_consts.STORED_STIMULUS_PATH + "/" + stimulus_name + ".wav"
        wavfile.write(
            stimulus_signal_path,
            self.data_struct.stimulus_info["sample_rate"],
            self.data_struct.stimulus_data.astype("float32"),
        )
        stimulus_signal_path = FileOps.get_relative_path(stimulus_signal_path, DEFAULT_DIR)
        if self.load_stimulus_signal_path:
            self.load_stimulus_signal_path = FileOps.get_relative_path(self.load_stimulus_signal_path, DEFAULT_DIR)
        # Create a dictionary containing the stimulus information and WAV file path
        data = {
            "stimulus_info": self.data_struct.stimulus_info,
            "stimulus_signal_path": stimulus_signal_path,
            "load_stimulus_signal_path": self.load_stimulus_signal_path,
        }
        # Write the dictionary data to the JSON file and log the operation
        self.save_json_file(data)

    def save_json_file(self, data):
        json_file_path = DEFAULT_DIR + "ui/ui_config/stimulus.json"
        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=3)
            self.default_logger.info(f"stimulus saved to {json_file_path}.")

    def update_stimulus_ui_value(self, stimulus_info: dict):
        """
            Update the user interface values for stimulus parameters.

            This function updates the values of various UI controls based on the stimulus parameters stored in
        `stimulus_info`.
            Specifically, it updates the stimulus method, stimulus type, start frequency, stop frequency, total time,
        repeat times, number of steps, voltage type, voltage value, and sample rate.
        """
        for k, v in self.STIMULUS_DICT.items():
            if v["name"] == stimulus_info.get("stimulus_method"):
                self.stimulus_method_combo_box.setCurrentText(k)
                break
        for k, v in self.STIMULUS_DICT_2.items():
            if v == stimulus_info.get("stimulus_type"):
                self.stimulus_type_combo_box.setCurrentText(k)
                break
        self.start_freq_box.setValue(int(stimulus_info.get("start_freq", "80")))
        self.stop_freq_box.setValue(int(stimulus_info.get("stop_freq", "2000")))
        self.total_time_box.setValue(float(stimulus_info.get("total_time", "4")))
        self.repeat_box.setValue(int(stimulus_info.get("repeat_times", "1")))
        self.step_box.setValue(int(stimulus_info.get("num_steps", "3")))
        self.voltage_combo_box.setCurrentText(stimulus_info.get("voltage_type", "RMS"))
        self.voltage_spin_box.setValue(float(stimulus_info.get("voltage", "2.0")))
        self.sample_rate_combo_box.setCurrentText(str(stimulus_info.get("sample_rate", "44100")))
        if stimulus_info.get("use_custom_stimulus") == True:
            self.custom_chk_box.setChecked(stimulus_info.get("use_custom_stimulus", "False"))
        elif False == stimulus_info.get("use_custom_stimulus"):
            self.change_custom_chk_box(False)

    def graph_stimulus(self):
        """
        Plot the stimulus signal waveform.

        This function clears the current plot area, generates a time axis based on the sample rate and signal data,
        and then plots the stimulus signal waveform in the plot area. The graph displays the amplitude of the signal
        over time, with labeled axes indicating the units.

        Parameters:
            self: The instance of the class, containing stimulus signal information and the plot area.
        """
        self.plot_stimulus.clear()
        sample_rate = self.data_struct.stimulus_info["sample_rate"]
        if self.data_struct.stimulus_data is not None:
            signal_duration = (
                np.linspace(0, len(self.data_struct.stimulus_data) - 1, len(self.data_struct.stimulus_data))
                / sample_rate
            )
            self.plot_stimulus.plot(signal_duration, self.data_struct.stimulus_data, pen="b")
            self.plot_stimulus.setLabel("left", "Amplitude", **{"font-size": "20px"})
            self.plot_stimulus.setLabel("bottom", "Time (s)",  **{"font-size": "20px"})
            font = QFont()
            font.setPixelSize(20)
            b_axis = self.plot_stimulus.getAxis("bottom")
            l_axis = self.plot_stimulus.getAxis("left")
            b_axis.setTickFont(font)
            l_axis.setTickFont(font)
            b_axis.setTextPen("black")
            l_axis.setTextPen("black")
            
    def load_config_btn_clicked(self):
        """
        Handles the event when the load configuration button is clicked.

        This function is triggered when the user clicks the load configuration button. It opens a dialog
        to load stimulus configuration and updates the loaded configuration into the `stimulus_info` dictionary
        of the current object. Finally, it updates the user interface and emits a signal indicating that the
        stimulus configuration has changed.
        """
        dlg = LoadStimulusDialog(self.default_logger)
        loaded_stimulus = dlg.exec()
        if loaded_stimulus is None:
            return
        for stimulus_item in loaded_stimulus:
            self.data_struct.stimulus_info[stimulus_item] = loaded_stimulus[stimulus_item]
        self.update_stimulus_ui_value(self.data_struct.stimulus_info)

    def save_config_btn_clicked(self):
        """
        Handles the click event of the save configuration button.

        This function calls the `save_stimulus_info_to_db` method of the `StimulusSignalManagement` class
        to save the stimulus information from `stimulus_info` to the database. Based on the save result,
        it logs the corresponding message.
        """
        stimulus_name_dialog = SetConfigName()
        stimulus_name = stimulus_name_dialog.exec()
        if stimulus_name != None:
            self.data_struct.stimulus_info["stimulus_name"] = stimulus_name
        else:
            return
        save_code, msg = StimulusSignalManagement().save_stimulus_info_to_db(self.data_struct.stimulus_info)
        if save_code == error_code.OK:
            self.default_logger.info("Successfully saving stimulus info to database.")
            QMessageBox.information(self, "保存配置", "刺激信号保存成功.")
        elif save_code == error_code.INVALID_INSERT:
            self.default_logger.error("This stimulus signals info already exists.")
            QMessageBox.warning(self, "保存配置", "刺激信号已存在.")
        elif save_code == error_code.INVALID_SAVE:
            self.default_logger.error("Failed to save stimulus info to database.")
            QMessageBox.warning(self, "保存配置", "保存刺激信号信息到数据库失败.")
        elif save_code == error_code.INVALID_NAME:
            self.default_logger.error("Invalid stimulus name.")
            QMessageBox.warning(self, "保存配置", "配置名称已存在.")

    def load_wav_btn_clicked(self):
        """
        Handles the button click event for loading a WAV file.

        This function opens a file dialog to select a WAV file, loads the audio data upon selection,
        and stores the audio signal and its time information in the class attributes.
        It then calls the plotting function to display the audio waveform.
        """
        load_stimulus_signal_path = self.load_stimulus_signal_path
        load_wav_path = self.load_wav_path
        self.load_stimulus_signal_path = self.load_wav_path
        self.load_wav_path, _ = QFileDialog.getOpenFileName(
            self, "打开音频", DEFAULT_DIR + "audio_data/stimulus", "WAV Files (*.wav)"
        )
        if self.load_wav_path:
            self.data_struct.stimulus_data, _ = load_audio_simple(
                self.load_wav_path, self.data_struct.stimulus_info["sample_rate"]
            )
            self.graph_stimulus()
        else:
            self.load_stimulus_signal_path = load_stimulus_signal_path
            self.load_wav_path = load_wav_path

    def get_stimulus_info_from_json(self, default_config_flag: bool = False):
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
        load_code, result = self.load_stimulus_info_from_json(default_config_flag)
        if load_code == error_code.OK and result:
            info = result.get("stimulus_info")
            self.load_wav_path = result["stimulus_signal_path"]
            if result.get("load_stimulus_signal_path"):
                self.load_stimulus_signal_path = DEFAULT_DIR + result["load_stimulus_signal_path"]
            return info
        else:
            return {}

    @staticmethod
    def load_stimulus_info_from_json(default_config_flag: bool = False):
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
            json_file_path = DEFAULT_DIR + "ui/ui_config/default_stimulus.json"
        if default_config_flag:
            json_file_path = DEFAULT_DIR + "ui/ui_config/default_stimulus.json"
        if not os.path.exists(json_file_path):
            return error_code.INVALID_DATA_LOADING, "This json file does not exist."
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            return error_code.OK, data

    def default_config_btn_clicked(self):
        self.data_struct.stimulus_info.clear()
        self.data_struct.stimulus_info = self.get_stimulus_info_from_json(True)
        self.update_stimulus_ui_value(self.data_struct.stimulus_info)

    def save_wav_btn_clicked(self):
        """
        Handles the save audio button click event.

        This function is triggered when the user clicks the save audio button. It opens a file save dialog,
        allowing the user to choose the save path and file name. If a valid file name is selected, the current
        stimulus signal is saved as a WAV audio file.
        """
        file_name, _ = QFileDialog.getSaveFileName(
            self, "保存音频", DEFAULT_DIR + "audio_data/stimulus", "WAV Files (*.wav)"
        )
        if file_name:
            sr = self.data_struct.stimulus_info.get("sample_rate", 44100)
            save_audio_simple(file_name, self.data_struct.stimulus_data, sr)

    def play_btn_clicked(self):
        """
        Handles the play button click event to play the stimulus signal.

        This function retrieves the stimulus signal and related parameters from instance attributes,
        and uses an instance of the SoundcardAudioProcessor class to call the sd_play method
        for playing the signal. If the playback fails, an error log is recorded.
        """
        # Construct the stimulus parameter dictionary, including signal data, amplitude, and sample rate
        stimulus_param = {
            "data": self.data_struct.stimulus_data,
            "amplitude": self.data_struct.stimulus_info["amplitude"],
            "sr": self.data_struct.stimulus_info["sample_rate"],
        }
        # Create an instance of SoundcardAudioProcessor and play the stimulus signal
        sap = SoundcardAudioProcessor()
        play_code, msg = sap.sd_play(stimulus_param)
        # If playback fails, log the error
        if play_code != error_code.OK:
            self.default_logger.error(f"Failed to play the stimulus file. {msg}")

    def ok_btn_clicked(self):
        self.refresh_stimulus_info = True
        if not self.custom_chk_box.isChecked():
            if self.start_custom_check_status == True:
                self.set_ai_popup()
            # elif self.load_wav_path != self.load_stimulus_signal_path:
            elif not os.path.samefile(self.load_wav_path, self.load_stimulus_signal_path):
                self.set_ai_popup()
            self.load_stimulus_signal_path = self.load_wav_path
            if not self.load_wav_path:
                self.miss_popup()
                return
            data = self.load_stimulus_wav()
            data["stimulus_signal_path"] = FileOps.get_relative_path(data["stimulus_signal_path"], DEFAULT_DIR)
            data["load_stimulus_signal_path"] = FileOps.get_relative_path(
                data["load_stimulus_signal_path"], DEFAULT_DIR
            )
            self.save_json_file(data)
        else:
            self.save_stimulus_to_json()
            stimulus_signal_length = self.data_struct.stimulus_info.get(
                "sample_rate"
            ) * self.data_struct.stimulus_info.get("total_time")
            if self.start_custom_check_status == False:
                self.set_ai_popup()
            elif self.old_stimulus_signal_length != stimulus_signal_length:
                self.set_ai_popup()
        # self.save_voltage_to_txt()
        self.close()

    def load_stimulus_wav(self):
        data = {}
        if self.load_wav_path:
            data = {
                "stimulus_info": self.data_struct.stimulus_info,
                "stimulus_signal_path": self.load_wav_path,
                "load_stimulus_signal_path": self.load_stimulus_signal_path,
            }
        return data

    def miss_popup(self):
        error_msg = QMessageBox(self)
        error_msg.setIcon(QMessageBox.Warning)
        error_msg.setText("请先配置激励信号！")
        error_msg.setWindowTitle("未配置激励信号")
        error_msg.setStandardButtons(QMessageBox.Ok)
        button = error_msg.exec_()
        return button == QMessageBox.Ok

    def set_ai_popup(self):
        error_msg = QMessageBox(self)
        error_msg.setIcon(QMessageBox.Warning)
        error_msg.setText("激励信号变化, 请重新配置AI分析模型!")
        error_msg.setWindowTitle("配置AI模型")
        error_msg.setStandardButtons(QMessageBox.Ok)
        button = error_msg.exec_()
        return button == QMessageBox.Ok

    def cancel_btn_clicked(self):
        """
        Handles the cancel button click event.

        This method is triggered when the user clicks the cancel button. It performs the following actions:
        1. Sets `refresh_stimulus_info` to `False`, indicating that stimulus information does not need to be refreshed.
        2. Closes the current window or dialog.
        """
        self.refresh_stimulus_info = False
        self.close()

    def on_exec(self):
        """
        Executes the operation and returns whether the stimulus information needs to be refreshed.

        This method first calls the `exec()` method to perform the operation, then determines
        whether the stimulus information needs to be refreshed based on the value of the
        `refresh_stimulus_info` attribute.

        Returns:
            bool: Returns True if `refresh_stimulus_info` is True, indicating that the stimulus
                information needs to be refreshed; otherwise, returns False.
        """
        self.exec()
        if self.refresh_stimulus_info:
            return True
        return False

    def update_stimulus_info(self, dict_key, v, changed_flag=False):
        """
            Updates the stimulus information dictionary with the specified key-value pair and returns whether an update
        occurred.

            Parameters:
            - dict_key: The key in the dictionary to be updated.
            - v: The value to be assigned to the specified key.
            - changed_flag: A flag indicating whether a change has already been made. Defaults to False.

            Returns:
            - True if the value in the dictionary differs from the provided value, causing an update.
            - The value of changed_flag if no update occurs.
        """
        if self.data_struct.stimulus_info.get(dict_key) != v:
            self.data_struct.stimulus_info[dict_key] = v
            return True
        changed_flag = changed_flag or False
        return changed_flag


class SetConfigName(QDialog):

    def __init__(self, parent=None):
        super(SetConfigName, self).__init__(parent)

        self.stimulus_name = None
        self.clicked_ok_close = False

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("设置配置名称")
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))

        congfig_name_layout = self.config_name_layout()
        btn_layout = self.btn_layout()

        layout = QVBoxLayout()
        layout.addLayout(congfig_name_layout)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        self.setStyleSheet(
            ui_style_const.qpushbutton_stytle + ui_style_const.qlineedit_stytle + ui_style_const.qlabel_stytle
        )

    def config_name_layout(self):
        name_label = QLabel("配置名称:")
        name_edit = QLineEdit()
        name_edit.setPlaceholderText("请输入配置名称")
        name_layout = QHBoxLayout()
        name_layout.addWidget(name_label)
        name_layout.addWidget(name_edit)

        return name_layout

    def btn_layout(self):
        ok_btn = QPushButton("确定")
        cancel_btn = QPushButton("取消")
        ok_btn.clicked.connect(self.on_click_ok_btn)
        cancel_btn.clicked.connect(self.on_click_cancel_btn)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        return btn_layout

    def on_click_ok_btn(self):
        self.stimulus_name = self.findChild(QLineEdit).text()
        if not self.stimulus_name:
            QMessageBox.warning(self, "警告", "请输入配置名称")
            return
        self.clicked_ok_close = True
        self.close()

    def on_click_cancel_btn(self):
        self.clicked_ok_close = False
        self.close()

    def exec(self):
        super().exec()
        if self.clicked_ok_close:
            return self.stimulus_name
        else:
            return None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StimulusWindow()
    window.speaker = sd.default.device[1]
    # window = LoadStimulusConfig()
    window.show()
    result = window.on_exec()
    print("final result:", result)
