import json
import os.path
import sys

import numpy as np
import pyqtgraph
import soundcard
from PyQt5.QtCore import Qt
from scipy.io import wavfile
from PyQt5.QtGui import QStandardItem, QStandardItemModel, QIcon
from PyQt5.QtWidgets import QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog
from PyQt5.QtWidgets import QGridLayout, QGroupBox, QHBoxLayout, QLabel, QListView, QPushButton
from PyQt5.QtWidgets import QSpacerItem, QSizePolicy, QSpinBox, QVBoxLayout

from base.load_audio import load_audio_simple, save_audio_simple
from base.log_manager import LogManager
from base.pre_processing.swept_sine_chirps import StimulusSignal
from base.soundcard_audio_processor import SoundcardAudioProcessor
from base.soundcard_calibration_manager import SoundcardCalibrationManager
from base.stimulus_signal_management import StimulusSignalManagement
from consts import error_code, model_consts, ui_style_const
from consts.running_consts import DEFAULT_DIR


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
        self.stimulus_info = {"name": "stimulus_chirps_1", "use_custom_stimulus": True}
        self.stimulus_signal = None
        self.speaker = None
        self.stimulus_signal_time = None
        self.refresh_stimulus_info = False
        self.default_logger = LogManager.set_log_handler("core")
        self.box_checked_enable_list = []
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
        self.stimulus_changed()

    def init_ui(self):
        # set window titlebar stytle
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setWindowTitle("Stimulus Window")
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setFixedSize(540, 700)
        # create layout to strore custom button layout
        custom_stimulus_layout = QGridLayout()
        custom_chk_box = QCheckBox("自定义")
        custom_chk_box.setChecked(True)
        custom_chk_box.stateChanged.connect(lambda: self.change_custom_chk_box(custom_chk_box.isChecked()))
        load_config_btn = QPushButton("导入配置")
        load_config_btn.clicked.connect(self.load_config_btn_clicked)
        save_config_btn = QPushButton("保存配置")
        save_config_btn.clicked.connect(self.save_config_btn_clicked)
        load_wav_btn = QPushButton("导入音频")
        load_wav_btn.clicked.connect(self.load_wav_btn_clicked)
        load_wav_btn.setDisabled(True)
        save_wav_btn = QPushButton("保存音频")
        save_wav_btn.clicked.connect(self.save_wav_btn_clicked)
        sl_btn_h_spacer = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        custom_stimulus_layout.addWidget(custom_chk_box, 0, 0)
        custom_stimulus_layout.addWidget(load_config_btn, 0, 1)
        custom_stimulus_layout.addWidget(save_config_btn, 1, 1)
        custom_stimulus_layout.addItem(sl_btn_h_spacer, 0, 2)
        custom_stimulus_layout.addWidget(load_wav_btn, 0, 3)
        custom_stimulus_layout.addWidget(save_wav_btn, 1, 3)
        custom_stimulus_layout.setContentsMargins(0, 0, 10, 0)

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

        v_spacer_1 = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        v_spacer_2 = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        v_spacer_3 = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)

        layout = QVBoxLayout()
        self.plot_stimulus = pyqtgraph.PlotWidget()
        self.plot_stimulus.setBackground('white')
        self.plot_stimulus.resize(400, 170)
        layout.addWidget(self.plot_stimulus)
        layout.addItem(v_spacer_1)
        layout.addLayout(custom_stimulus_layout)
        layout.addWidget(stimulus_type_group_box)
        layout.addWidget(self.frequency_group_box)
        layout.addWidget(time_group_box)
        layout.addWidget(self.step_group_box)
        layout.addItem(v_spacer_2)
        layout.addLayout(output_layout)
        layout.addItem(v_spacer_3)
        layout.addLayout(function_btn_layout)
        layout.setContentsMargins(25, 10, 25, 20)

        # Set custom_chk_box in different states, can use the function list
        self.box_checked_enable_list = [load_config_btn, save_config_btn, stimulus_type_group_box,
                                        self.frequency_group_box, time_group_box, self.step_group_box]
        self.box_checked_disable_list = [load_wav_btn]

        self.setLayout(layout)

        self.setStyleSheet(ui_style_const.qcombobox_stytle +
                           ui_style_const.qpushbutton_stytle +
                           ui_style_const.qspinbox_stytle +
                           ui_style_const.qdoublespinbox_stytle +
                           ui_style_const.qlabel_stytle + 
                           ui_style_const.qcheckbox_stytle +
                           ui_style_const.qgroupbox_stytle)

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
        self.stimulus_method_combo_box.addItems(["啁啾", "步进", "噪音"])
        stimulus_item = self.STIMULUS_DICT.get("啁啾")
        self.stimulus_type_combo_box.addItems(stimulus_item.get("sub_list"))
        self.stimulus_method_combo_box.currentTextChanged.connect(self.set_stimulus_type_connection)
        self.stimulus_type_combo_box.currentTextChanged.connect(self.stimulus_changed)
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
        self.start_freq_box.setValue(80)
        self.start_freq_box.editingFinished.connect(self.stimulus_changed)
        self.start_freq_box.setMinimumWidth(100)
        stop_freq_label = QLabel("截止频率：")
        self.stop_freq_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.stop_freq_box.setSuffix(" Hz")
        self.stop_freq_box.setRange(10, 24000)
        self.stop_freq_box.setValue(2000)
        self.stop_freq_box.setMinimumWidth(100)
        self.stop_freq_box.editingFinished.connect(self.stimulus_changed)
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
        self.total_time_box.setDecimals(1)      # Allow one decimal place
        self.total_time_box.setRange(0.5, 60)   # Set range 0.5-60 seconds
        self.total_time_box.setValue(4)         # Set default value
        self.total_time_box.setMinimumWidth(100)
        self.total_time_box.editingFinished.connect(self.stimulus_changed)
        repeat_label = QLabel("信号重复：")
        self.repeat_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.repeat_box.setRange(1, 10)
        self.repeat_box.setSuffix(" 次")
        self.repeat_box.setMinimumWidth(100)
        self.repeat_box.valueChanged.connect(self.stimulus_changed)
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
        self.step_box.editingFinished.connect(self.stimulus_changed)
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
        self.voltage_combo_box.addItems(["RMS", "Peak"])
        self.voltage_combo_box.currentTextChanged.connect(self.stimulus_changed)
        self.voltage_spin_box.setSuffix(" V")
        self.voltage_spin_box.setValue(self.load_voltage_from_txt())
        self.voltage_spin_box.setSingleStep(0.1)
        self.voltage_spin_box.setMinimum(0.1)
        self.voltage_spin_box.editingFinished.connect(self.stimulus_changed)

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
        self.sample_rate_combo_box.addItems(["44100", "48000"])
        self.sample_rate_combo_box.currentTextChanged.connect(self.stimulus_changed)
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
        play_btn = QPushButton(" 试  播 ")
        play_btn.setStyleSheet("padding: 3px")
        play_btn.clicked.connect(self.play_btn_clicked)
        ok_btn = QPushButton(" 确  认 ")
        ok_btn.setStyleSheet("padding: 3px")
        ok_btn.clicked.connect(self.ok_btn_clicked)
        cancel_btn = QPushButton(" 取  消 ")
        cancel_btn.setStyleSheet("padding: 3px")
        cancel_btn.clicked.connect(self.cancel_btn_clicked)
        function_btn_h_spacer = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        function_btn_layout.addWidget(play_btn)
        function_btn_layout.addItem(function_btn_h_spacer)
        function_btn_layout.addWidget(ok_btn)
        function_btn_layout.addWidget(cancel_btn)
        function_btn_layout.setSpacing(20)
        return function_btn_layout

    def change_custom_chk_box(self, custom_box_checked):
        """
            Updates the enabled/disabled state and style of related widgets based on the custom checkbox state.

            Parameters:
            custom_box_checked (bool): The checked state of the custom checkbox. If True, the checkbox is checked;
        if False, it is unchecked.
        """
        for widget in self.box_checked_enable_list:
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
        self.stimulus_info["use_custom_stimulus"] = custom_box_checked
        self.stimulus_changed(True)
        if custom_box_checked:
            self.set_stimulus_type_connection()

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

    def stimulus_changed(self, changed_flag=False):
        """
            Handles changes in stimulus parameters and updates relevant information.

            Parameters:
            - changed_flag (bool): A flag indicating whether any parameters have changed. Default is False.

            Returns:
            - None, but updates internal state and may trigger signal generation and graph plotting.
        """
        # Get the currently selected stimulus method and type
        stimulus_method = self.stimulus_method_combo_box.currentText()
        stimulus_method_item = self.STIMULUS_DICT.get(stimulus_method)
        stimulus_type = self.stimulus_type_combo_box.currentText()
        # Build a dictionary containing all stimulus parameters
        change_dict = {
            "stimulus_method": stimulus_method_item.get("name"),
            "stimulus_type": self.STIMULUS_DICT_2.get(stimulus_type),
            "start_freq": self.start_freq_box.value(),
            "stop_freq": self.stop_freq_box.value(),
            "total_time": self.total_time_box.value(),
            "repeat_times": self.repeat_box.value(),
            "num_steps": self.step_box.value(),
            "voltage_type": self.voltage_combo_box.currentText(),
            "voltage": self.voltage_spin_box.value(),
            "amplitude": self.get_predict_amplitude(self.voltage_spin_box.value()),
            "sample_rate": int(self.sample_rate_combo_box.currentText()),
        }
        # Iterate through the parameter dictionary, update stimulus info, and check for changes
        for k, v in change_dict.items():
            changed_flag = self.update_stimulus_info(k, v, changed_flag)
        # If stimulus type is set and there are changes, generate the signal and plot the graph if necessary
        if self.stimulus_info.get("stimulus_type") and changed_flag:
            if self.stimulus_info.get("use_custom_stimulus"):
                self.create_signal_from_stimulus_info()
            self.graph_stimulus()

    @staticmethod
    def get_predict_amplitude(target_voltage):
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
        code, result_amplitude = SoundcardCalibrationManager().calibrate_amplitude(target_voltage,
                                                                                   json_file_name="calibration_coefficients.json")
        if code == error_code.OK:
            predict_amplitude, max_voltage = result_amplitude
            return predict_amplitude
        else:
            return 0.0

    def create_signal_from_stimulus_info(self):
        """
            Generates a signal based on the stimulus information.

            This method selects the appropriate signal generation function based on the `stimulus_method` field 
            in `self.stimulus_info` and calls the function to generate the signal. The generated signal is 
            stored in `self.stimulus_signal`.

            Parameters:
                No explicit parameters, but relies on the following fields in the `self.stimulus_info` dictionary:
                    - stimulus_method: A string specifying the signal generation method, with possible values "chirp",
                 "step", or "noise".
                    - Other fields: Depending on the `stimulus_method`, additional parameters may be required, which
                are passed to the respective signal generation function.

            Returns:
                No explicit return value, but updates the `self.stimulus_signal` attribute with the generated signal.
        """
        create_function_dict = {
            "chirp": StimulusSignal().generate_chirps,
            "step": StimulusSignal().generate_steps,
            "noise": StimulusSignal().generate_noise,
        }
        create_function = create_function_dict.get(self.stimulus_info["stimulus_method"])
        self.stimulus_signal, _ = create_function(**self.stimulus_info)

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
        json_file_path = DEFAULT_DIR + "ui/ui_config/stimulus.json"
         # Generate the name of the stimulus signal based on the stimulus information
        stimulus_name = "_".join(str(value) for value in self.stimulus_info.values())
        # Construct the path for the WAV file and save the stimulus signal as a WAV file
        stimulus_signal_path = model_consts.STORED_STIMULUS_PATH + "/" + stimulus_name + ".wav"
        wavfile.write(stimulus_signal_path, self.stimulus_info["sample_rate"], self.stimulus_signal.astype("float32"))
        # Create a dictionary containing the stimulus information and WAV file path
        data = {
            "stimulus_info": self.stimulus_info,
            "stimulus_signal_path": stimulus_signal_path
        }
        # Write the dictionary data to the JSON file and log the operation
        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=3)
            self.default_logger.info(f"stimulus saved to {json_file_path}.")

    def update_stimulus_ui_value(self):
        """
            Update the user interface values for stimulus parameters.

            This function updates the values of various UI controls based on the stimulus parameters stored in
        `self.stimulus_info`.
            Specifically, it updates the stimulus method, stimulus type, start frequency, stop frequency, total time,
        repeat times, number of steps, voltage type, voltage value, and sample rate.
        """
        for k, v in self.STIMULUS_DICT.items():
            if v["name"] == self.stimulus_info.get("stimulus_method"):
                self.stimulus_method_combo_box.setCurrentText(k)
        for k, v in self.STIMULUS_DICT_2.items():
            if v == self.stimulus_info.get("stimulus_type"):
                self.stimulus_type_combo_box.setCurrentText(k)
        self.start_freq_box.setValue(int(self.stimulus_info["start_freq"]))
        self.stop_freq_box.setValue(int(self.stimulus_info["stop_freq"]))
        self.total_time_box.setValue(float(self.stimulus_info["total_time"]))
        self.repeat_box.setValue(int(self.stimulus_info["repeat_times"]))
        self.step_box.setValue(int(self.stimulus_info["num_steps"]))
        self.voltage_combo_box.setCurrentText(self.stimulus_info["voltage_type"])
        self.voltage_spin_box.setValue(float(self.stimulus_info["voltage"]))
        self.sample_rate_combo_box.setCurrentText(str(self.stimulus_info["sample_rate"]))

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
        sample_rate = self.stimulus_info["sample_rate"]
        signal_duration = np.linspace(0, len(self.stimulus_signal) - 1, len(self.stimulus_signal)) / sample_rate
        self.plot_stimulus.plot(signal_duration, self.stimulus_signal, pen='b')
        self.plot_stimulus.setLabel('left', 'Amplitude')
        self.plot_stimulus.setLabel('bottom', 'Time (s)')

    def load_config_btn_clicked(self):
        """
            Handles the event when the load configuration button is clicked.

            This function is triggered when the user clicks the load configuration button. It opens a dialog
            to load stimulus configuration and updates the loaded configuration into the `stimulus_info` dictionary
            of the current object. Finally, it updates the user interface and emits a signal indicating that the
            stimulus configuration has changed.
        """
        dlg = LoadStimulusConfig()
        loaded_stimulus = dlg.on_exec()
        for stimulus_item in loaded_stimulus:
            self.stimulus_info[stimulus_item] = loaded_stimulus[stimulus_item]
            self.update_stimulus_ui_value()
        self.stimulus_changed(True)

    def save_config_btn_clicked(self):
        """
            Handles the click event of the save configuration button.

            This function calls the `save_stimulus_info_to_db` method of the `StimulusSignalManagement` class
            to save the stimulus information from `self.stimulus_info` to the database. Based on the save result,
            it logs the corresponding message.
        """
        save_code, msg = StimulusSignalManagement().save_stimulus_info_to_db(self.stimulus_info)
        if save_code == error_code.OK:
            self.default_logger.info("Successfully saving stimulus info to database.")

    def load_wav_btn_clicked(self):
        """
            Handles the button click event for loading a WAV file.

            This function opens a file dialog to select a WAV file, loads the audio data upon selection, 
            and stores the audio signal and its time information in the class attributes. 
            It then calls the plotting function to display the audio waveform.
        """
        path, _ = QFileDialog.getOpenFileName(self,
                                              "打开音频",
                                              DEFAULT_DIR + "audio_data/stimulus",
                                              "WAV Files (*.wav)")
        if path:
            self.stimulus_signal, self.stimulus_signal_time = load_audio_simple(path, self.stimulus_info["sample_rate"])
            self.graph_stimulus()

    def save_wav_btn_clicked(self):
        """
            Handles the save audio button click event.

            This function is triggered when the user clicks the save audio button. It opens a file save dialog,
            allowing the user to choose the save path and file name. If a valid file name is selected, the current
            stimulus signal is saved as a WAV audio file.
        """
        file_name, _ = QFileDialog.getSaveFileName(self,
                                                   "保存音频",
                                                   DEFAULT_DIR + "audio_data/stimulus",
                                                   "WAV Files (*.wav)")
        if file_name:
            sr = self.stimulus_info.get("sample_rate", 44100)
            save_audio_simple(file_name + ".wav", self.stimulus_signal, sr)

    def play_btn_clicked(self):
        """
            Handles the play button click event to play the stimulus signal.

            This function retrieves the stimulus signal and related parameters from instance attributes,
            and uses an instance of the SoundcardAudioProcessor class to call the sd_play method
            for playing the signal. If the playback fails, an error log is recorded.
        """
        # Construct the stimulus parameter dictionary, including signal data, amplitude, and sample rate
        stimulus_param = {"data": self.stimulus_signal,
                          "amplitude": self.stimulus_info["amplitude"],
                          "sr": self.stimulus_info["sample_rate"]}
        # Create an instance of SoundcardAudioProcessor and play the stimulus signal
        sap = SoundcardAudioProcessor()
        play_code, msg = sap.sd_play(stimulus_param)
        # If playback fails, log the error
        if play_code != error_code.OK:
            self.default_logger.error(f"Failed to play the stimulus file. {msg}")

    def save_voltage_to_txt(self):
        """
            Save the voltage value to a text file.

            This function retrieves the voltage value from the `self.stimulus_info` dictionary and saves it to a
        specified text file.
            If the target directory does not exist, the function will create it and log the creation information.
            If an exception occurs during the saving process, the function will log the error message.
        """
        # Retrieve the voltage value from the stimulus_info dictionary
        voltage_value = self.stimulus_info["voltage"]
        # If the dir_path does not exist, create it and log the creation
        dir_path = DEFAULT_DIR + 'ui/ui_config'
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
            self.default_logger.info(f"Dir '{dir_path}' created.")
        # Define the file path
        file_path = dir_path + "/" + "voltage_value.txt"
        try:
            # Open the file and write the voltage value, log the action
            with open(file_path, 'w') as f:
                f.write(str(voltage_value))
                self.default_logger.info(f"The voltage value: {voltage_value} saved to voltage_value.txt")
        except Exception as e:
            # If saving fails, log the error~       
            self.default_logger.error("Failed to save voltage value to txt. %s" % (str(e)[:40]))

    def load_voltage_from_txt(self):
        """
            Loads the voltage value from a specified text file.

            This function attempts to read the voltage value from a text file located at the default path and converts
        it to a float.
            If any exception occurs during the reading process, the function logs an error and returns a default value
        of 0.0.

            Returns:
                float: The voltage value read from the file, or 0.0 if the reading fails.
        """
        file_path = DEFAULT_DIR + "ui/ui_config/voltage_value.txt"
        try:
            with open(file_path, 'r') as f:
                voltage_value = float(f.read())
            return voltage_value
        except Exception as e:
            self.default_logger.error("Failed to find voltage value. %s" % (str(e)[:40]))
            return 0.0

    def ok_btn_clicked(self):
        self.refresh_stimulus_info = True
        self.save_stimulus_to_json()
        self.save_voltage_to_txt()
        self.close()

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
        if self.stimulus_info.get(dict_key) != v:
            self.stimulus_info[dict_key] = v
            return True
        changed_flag = changed_flag or False
        return changed_flag


class LoadStimulusConfig(QDialog):

    def __init__(self, ):
        """
            Initialization function to set up the initial state of the class.

            This function performs the following operations:
            1. Calls the initialization method of the parent class.
            2. Initializes an empty dictionary `selected_config` to store the selected configuration.
            3. Loads the stimulus configuration from the database and stores it in `loaded_stimulus`.
            4. Calls the `init_ui` method to initialize the user interface.
        """
        super().__init__()
        self.selected_config = {}
        self.loaded_stimulus = self.load_stimulus_config_from_db()

        self.init_ui()

    def init_ui(self):
        """
            Initializes the user interface, sets the window title, window flags, and creates the stimulus signal
        selection interface.

            This function performs the following operations:
            1. Sets the window title to "Select Stimulus Signal" and disables the close and help buttons.
            2. Creates a QListView to display the list of stimulus signals and uses QStandardItemModel to manage
        the list items.
            3. Iterates through the loaded stimulus signals in self.loaded_stimulus, adds their names to the list,
        and marks the default option.
            4. Sets the selected item in the list view and connects the click event to the on_select_item method.
            5. Creates OK and Cancel buttons and connects their click events to the respective methods.
            6. Adds the list view and button layout to the main layout and sets the styles.
        """
        self.setWindowTitle("选择激励信号")
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)

        # Create a list view and model to display the stimulus signals
        list_view = QListView()
        item_model = QStandardItemModel()
        default_index = None
        # Iterate through the loaded stimulus signals, add them to the model, and mark the default option
        for stimulus in self.loaded_stimulus:
            item_model.appendRow(QStandardItem(stimulus["name"]))
            if stimulus.get('is_default') == 1:
                default_index = item_model.index(item_model.rowCount() - 1, 0)
        list_view.setModel(item_model)
        list_view.setSelectionRectVisible(True)

         # Set the default selected item and trigger the selection event
        if default_index is not None:
            list_view.setCurrentIndex(default_index)
            self.on_select_item(default_index)
        else:
            if item_model.rowCount() > 0:
                list_view.setCurrentIndex(item_model.index(0, 0))
                self.on_select_item(item_model.index(0, 0))
        list_view.clicked.connect(self.on_select_item)

        # Create OK and Cancel buttons and add them to a horizontal layout
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("确认")
        ok_btn.clicked.connect(self.on_click_ok_btn)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.on_click_cancel_btn)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)

        layout = QVBoxLayout()
        layout.addWidget(list_view)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        list_view.setStyleSheet("font-size: 20px;")
        self.setStyleSheet(ui_style_const.qpushbutton_stytle)

    def on_select_item(self, index):
        """
            Handles the event when an item is selected, saving the selected configuration to the `selected_config` attribute.

            Parameters:
            - index: QModelIndex object, representing the index of the item selected by the user.
        """
        self.selected_config = self.loaded_stimulus[index.row()]

    def on_click_ok_btn(self):
        self.close()

    def on_click_cancel_btn(self):
        """
            Handles the cancel button click event.

            This function is triggered when the user clicks the cancel button. It clears the currently selected
        configuration and closes the current window.
        """
        self.selected_config = {}
        self.close()

    def on_exec(self):
        # Executes the configuration selection operation and returns the selected configuration.
        self.exec()
        return self.selected_config

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
                    'name': f"stimulus_{query_data_idx[1]}_{idx + 1}",
                    'stimulus_method': query_data_idx[1],
                    'stimulus_type': query_data_idx[2],
                    'start_freq': query_data_idx[4],
                    'stop_freq': query_data_idx[5],
                    'total_time': query_data_idx[7],
                    'repeat_times': query_data_idx[3],
                    'sample_rate': query_data_idx[6],
                    'num_steps': query_data_idx[8],
                    'is_default': query_data_idx[9]
                }
                # Append the converted stimulus signal information to the list
                stimulus_list.append(stimulus)
        return stimulus_list


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StimulusWindow()
    window.speaker = soundcard.default_speaker()
    # window = LoadStimulusConfig()
    window.show()
    result = window.on_exec()
    print("final result:", result)
