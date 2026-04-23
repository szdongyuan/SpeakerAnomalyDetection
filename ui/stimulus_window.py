import json
import os.path
from copy import deepcopy

import numpy as np
import pyqtgraph
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtWidgets import QDialog, QFileDialog, QSizePolicy, QGridLayout, QHBoxLayout, QVBoxLayout

from base.file_ops import FileOps
from base.load_audio import load_audio_simple
from base.log_manager import LogManager
from base.pre_processing.swept_sine_chirps import StimulusSignal
from base.save_data import save_audio_simple
from base.soundcard_audio_processor import SoundcardAudioProcessor
from base.soundcard_calibration_manager import SoundcardCalibrationManager
from base.stimulus_signal_management import StimulusSignalManagement
from consts import error_code, model_consts, ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.widgets import (
    PushButton,
    ComboBox,
    LineEdit,
    Label,
    CheckBox,
    GroupBox,
    DoubleSpinBox,
    SpinBox,
    MessageBox,
)
from ui.load_stimulus_dialog import LoadStimulusDialog
from ui.ui_src import ui_resources


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

    def __init__(self, stimulus_config_data=None, speaker=None):
        """Initialize stimulus window with default configurations"""
        super().__init__()
        # initialize stimulus signal type， and create variable to store stimulus signal data
        self.speaker = speaker
        self.load_wav_path = ""
        self.load_stimulus_signal_path = None
        self.refresh_stimulus_info = False
        self.is_close_window = False
        self.default_logger = LogManager.set_log_handler("core")
        self.box_checked_enable_dict = {}
        self.box_checked_disable_list = []
        self.final_save_data = None

        # create variable to set stimulus signal data
        self.stimulus_method_combo_box = ComboBox()
        self.stimulus_type_combo_box = ComboBox()
        self.start_freq_box = SpinBox()
        self.stop_freq_box = SpinBox()
        self.total_time_box = DoubleSpinBox()
        self.repeat_box = SpinBox()
        self.step_box = SpinBox()
        self.voltage_combo_box = ComboBox()
        self.voltage_spin_box = DoubleSpinBox()
        self.sample_rate_combo_box = ComboBox()
        # rcreate variable to control whether to use frequency or step
        self.frequency_group_box = self.create_frequency_group_box()
        self.step_group_box = self.create_step_group_box()
        self.stimulus_config_data = deepcopy(stimulus_config_data)
        self.load_stimulus_config_data(self.stimulus_config_data)
        self.start_custom_check_status = self.stimulus_info.get("use_custom_stimulus", False)
        self.init_ui()
        self.update_stimulus_ui_value(self.stimulus_info)
        if self.stimulus_info["use_custom_stimulus"]:
            self.create_signal_from_stimulus_info()
        self.graph_stimulus()

        original_total_time = self.stimulus_info.get("total_time")
        self.original_stimulus_signal_length = self.stimulus_info.get("sample_rate") * original_total_time

    def init_ui(self):
        # set window titlebar style
        self.setObjectName("StimulusWindow")
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))
        self.setWindowTitle("激励信号")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        if ui_style_const._FONT_SCALE < 0.8:
            self.setFixedSize(
                ui_style_const.scale_size_px(630),
                ui_style_const.scale_size_px(780),
            )
        elif ui_style_const._FONT_SCALE < 0.9:
            self.setFixedSize(
                ui_style_const.scale_size_px(580),
                ui_style_const.scale_size_px(750),
            )
        elif ui_style_const._FONT_SCALE < 0.95:
            self.setFixedSize(
                ui_style_const.scale_size_px(530),
                ui_style_const.scale_size_px(730),
            )
        elif ui_style_const._FONT_SCALE < 1.05:
            self.setFixedSize(
                ui_style_const.scale_size_px(515),
                ui_style_const.scale_size_px(650),
            )
        # elif ui_style_const._FONT_SCALE < 1.1:
        else:
            self.setFixedSize(
                ui_style_const.scale_size_px(500),
                ui_style_const.scale_size_px(650),
            )

        self.plot_stimulus = pyqtgraph.PlotWidget()
        self.plot_stimulus.setBackground("white")

        # create layout to strore custom button layout
        custom_stimulus_layout = QGridLayout()
        self.custom_chk_box = CheckBox("自定义")
        self.custom_chk_box.stateChanged.connect(lambda: self.change_custom_chk_box(self.custom_chk_box.isChecked()))
        load_config_btn = PushButton("导入配置")
        load_config_btn.clicked.connect(self.load_config_btn_clicked)
        save_config_btn = PushButton("保存配置")
        save_config_btn.clicked.connect(self.save_config_btn_clicked)
        load_wav_btn = PushButton("导入音频")
        load_wav_btn.clicked.connect(self.load_wav_btn_clicked)
        load_wav_btn.setDisabled(True)
        save_wav_btn = PushButton("保存音频")
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

    def switch_connection_on(self):
        self.stimulus_type_combo_box.currentTextChanged.connect(self.update_stimulus_info_from_stimulus_type_combo_box)
        self.stimulus_method_combo_box.currentTextChanged.connect(self.set_stimulus_type_connection)
        self.start_freq_box.valueChanged.connect(
            lambda: self.update_stimulus_info_from_controller(self.start_freq_box, "start_freq")
        )
        self.stop_freq_box.valueChanged.connect(
            lambda: self.update_stimulus_info_from_controller(self.stop_freq_box, "stop_freq")
        )
        self.total_time_box.valueChanged.connect(
            lambda: self.update_stimulus_info_from_controller(self.total_time_box, "total_time")
        )
        self.repeat_box.valueChanged.connect(
            lambda: self.update_stimulus_info_from_controller(self.repeat_box, "repeat_times")
        )
        self.step_box.valueChanged.connect(
            lambda: self.update_stimulus_info_from_controller(self.step_box, "num_steps")
        )
        self.voltage_combo_box.currentTextChanged.connect(self.update_stimulus_info_from_voltage_combo_box)

        self.voltage_spin_box.valueChanged.connect(
            lambda: self.update_stimulus_info_from_controller(self.voltage_spin_box, "voltage")
        )
        self.sample_rate_combo_box.currentTextChanged.connect(
            lambda: self.update_stimulus_info_from_controller(self.sample_rate_combo_box, "sample_rate")
        )

    def switch_connection_off(self):
        self.stimulus_type_combo_box.disconnect()
        self.stimulus_method_combo_box.disconnect()
        self.start_freq_box.disconnect()
        self.stop_freq_box.disconnect()
        self.total_time_box.disconnect()
        self.repeat_box.disconnect()
        self.step_box.disconnect()
        self.voltage_spin_box.disconnect()
        self.sample_rate_combo_box.disconnect()

    def create_stimulus_type_group_box(self):
        """
        Create a GroupBox for stimulus signal type selection.

        This method constructs a group box containing combo boxes for selecting different stimulus signal types.
        It configures layout components, establishes signal-slot connections for handling selection changes,
        and initializes default values from the stimulus dictionary.

        Returns:
            GroupBox: Configured group box containing stimulus type selection components.
        """
        stimulus_type_group_box = GroupBox("激励信号类型")
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

        Constructs a GroupBox containing start/stop frequency spinboxes for user input.
        Features:
        - Two DoubleSpinBox with Hz suffix
        - Value range: 10-24000 Hz
        - Default values: 80 Hz (start), 2000 Hz (stop)
        - Auto-triggers stimulus_changed signal on edit completion

        Returns:
            GroupBox: Configured group box with frequency range widgets
        """
        frequency_group_box = GroupBox("频率范围 (10 - 24000Hz)")
        start_freq_label = Label("起始频率:")
        self.start_freq_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.start_freq_box.setSuffix(" Hz")
        self.start_freq_box.setRange(10, 24000)
        self.start_freq_box.setMinimumWidth(100)
        stop_freq_label = Label("截止频率:")
        self.stop_freq_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.stop_freq_box.setSuffix(" Hz")
        self.stop_freq_box.setRange(10, 24000)
        self.stop_freq_box.setMinimumWidth(100)
        frequency_layout = QHBoxLayout()
        frequency_layout.addWidget(start_freq_label)
        frequency_layout.addWidget(self.start_freq_box)
        frequency_layout.addWidget(stop_freq_label)
        frequency_layout.addWidget(self.stop_freq_box)
        frequency_group_box.setLayout(frequency_layout)
        frequency_layout.setContentsMargins(10, 10, 20, 10)
        frequency_layout.setSpacing(20)
        return frequency_group_box

    def create_time_group_box(self):
        """
        Create time parameters configuration group box
        Constructs a GroupBox containing signal duration and repetition controls for configuring
        stimulus timing parameters. All value changes will trigger the stimulus_changed signal.

        Returns:
            GroupBox: Container widget with horizontal layout of time configuration controls
        """
        time_group_box = GroupBox()
        total_time_label = Label("信号时长:")
        self.total_time_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.total_time_box.setSuffix(" s")
        self.total_time_box.setDecimals(1)  # Allow one decimal place
        self.total_time_box.setRange(0.5, 60)  # Set range 0.5-60 seconds
        self.total_time_box.setSingleStep(0.5)
        self.total_time_box.setMinimumWidth(100)
        repeat_label = Label("信号重复:")
        self.repeat_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.repeat_box.setRange(1, 10)
        self.repeat_box.setSuffix(" 次")
        self.repeat_box.setMinimumWidth(100)
        time_layout = QHBoxLayout()
        time_layout.addWidget(total_time_label)
        time_layout.addWidget(self.total_time_box)
        time_layout.addWidget(repeat_label)
        time_layout.addWidget(self.repeat_box)
        time_layout.setContentsMargins(10, 10, 20, 10)
        time_layout.setSpacing(20)
        time_group_box.setLayout(time_layout)
        return time_group_box

    def create_step_group_box(self):
        """
        Create and configure the step setting group box
        Return:
            GroupBox: A group box that contains a step number setting control with the following components:
            - Label Displays "Step quantity"
            - SpinBox used to set the step value (range 1-100)
        Signal connection:
            The editingFinished signal of step_box is connected to the stimulus_changed method
        """
        step_group_box = GroupBox()
        step_label = Label("步进数量")
        self.step_box.setMinimumWidth(100)
        self.step_box.setRange(1, 100)
        step_layout = QHBoxLayout()
        step_layout.addWidget(step_label)
        step_layout.addWidget(self.step_box)
        step_layout.setContentsMargins(10, 10, 10, 10)
        step_group_box.setLayout(step_layout)
        return step_group_box

    def create_voltage_group_box(self):
        """
            Creates a GroupBox for setting the output voltage.

            This function creates a GroupBox containing a combo box and a spin box for selecting the type of output
        voltage (RMS or Peak) and setting the voltage value.
            When the values in the combo box or spin box change, the `stimulus_changed` signal is triggered.

            Returns:
                GroupBox: Returns a configured GroupBox containing the controls for setting the output voltage.
        """
        voltage_group_box = GroupBox("输出电压")
        self.voltage_combo_box.addItems(["RMS", "Peak"])
        self.voltage_spin_box.setSuffix(" V")
        max_input_voltage = self.get_max_input_voltage()
        self.voltage_spin_box.setSingleStep(0.1)
        self.voltage_spin_box.setRange(0.1, max_input_voltage)

        voltage_layout = QHBoxLayout()
        voltage_layout.addWidget(self.voltage_combo_box)
        voltage_layout.addWidget(self.voltage_spin_box)
        voltage_layout.setContentsMargins(10, 10, 6, 10)
        voltage_group_box.setLayout(voltage_layout)

        return voltage_group_box

    def create_sample_rate_group_box(self):
        """
        Creates a GroupBox containing a sample rate selection combo box.

        This function generates a GroupBox that includes a combo box for selecting the sample rate.
        The combo box options are 44100 and 48000. When the user changes the sample rate,
        it triggers the `stimulus_changed` signal.

        Returns:
            GroupBox: A GroupBox object containing the sample rate selection combo box.
        """
        sample_rate_group_box = GroupBox("采样率")
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
        default_config_btn = PushButton(" 默认配置 ")
        default_config_btn.clicked.connect(self.default_config_btn_clicked)
        play_btn = PushButton(" 试  播 ")
        play_btn.clicked.connect(self.play_btn_clicked)
        ok_btn = PushButton(" 确  认 ")
        ok_btn.clicked.connect(self.ok_btn_clicked)
        cancel_btn = PushButton(" 取  消 ")
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
        self.stimulus_info["stimulus_type"] = self.STIMULUS_DICT_2.get(stimulus_type)
        if self.stimulus_info.get("use_custom_stimulus", False) and 0 != self.stimulus_type_combo_box.count():
            self.create_signal_from_stimulus_info()
            self.graph_stimulus()

    def update_stimulus_info_from_controller(self, controller, data_type: str):
        if data_type == "sample_rate":
            self.stimulus_info[data_type] = int(controller.currentText())
        elif data_type == "total_time":
            self.stimulus_info[data_type] = float(controller.value())
        elif data_type == "voltage":
            self.stimulus_info[data_type] = float(controller.value())
            self.update_amplitude()
        else:
            self.stimulus_info[data_type] = int(controller.value())
        if self.stimulus_info.get("use_custom_stimulus", False):
            self.create_signal_from_stimulus_info()
            self.graph_stimulus()

    def update_stimulus_info_from_voltage_combo_box(self):
        self.stimulus_info["voltage_type"] = self.voltage_combo_box.currentText()

    def update_amplitude(self):
        self.stimulus_info["amplitude"] = self.get_predict_amplitude(self.voltage_spin_box.value())

    def sync_voltage_info(self):
        """Ensure voltage-related fields mirror the current UI selection before persisting data."""
        self.stimulus_info["voltage_type"] = self.voltage_combo_box.currentText()
        self.stimulus_info["voltage"] = float(self.voltage_spin_box.value())
        self.update_amplitude()

    def switch_group_box_availability(self, enable_status=True):
        stimulus_method = self.STIMULUS_DICT[self.stimulus_method_combo_box.currentText()]["name"]
        for widget in self.box_checked_enable_dict["step"]:
            widget.setDisabled(True)
        if enable_status:
            for widget in self.box_checked_enable_dict[stimulus_method]:
                widget.setEnabled(True)

    def change_custom_chk_box(self, custom_box_checked):
        """
            Updates the enabled/disabled state and style of related widgets based on the custom checkbox state.

            Parameters:
            custom_box_checked (bool): The checked state of the custom checkbox. If True, the checkbox is checked;
        if False, it is unchecked.
        """
        self.switch_group_box_availability(custom_box_checked)
        for widget in self.box_checked_disable_list:
            widget.setDisabled(custom_box_checked)

        self.stimulus_info["use_custom_stimulus"] = custom_box_checked
        if custom_box_checked:
            self.create_signal_from_stimulus_info()
        else:
            self.load_wav_path = self.load_stimulus_signal_path
            self.stimulus_data, _ = load_audio_simple(self.load_wav_path, self.stimulus_info["sample_rate"])
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
        self.stimulus_info["stimulus_method"] = self.STIMULUS_DICT[stimulus_method]["name"]
        self.stimulus_type_combo_box.clear()
        # Fetch the corresponding sublist from the dictionary and add it to the stimulus type combo box
        stimulus_item = self.STIMULUS_DICT.get(stimulus_method, {})
        self.stimulus_type_combo_box.addItems(stimulus_item.get("sub_list", []))
        # Adjust the state and style of related controls based on the selected stimulus method
        if stimulus_method == "啁啾":
            self.step_group_box.setDisabled(True)
            self.frequency_group_box.setEnabled(True)
        elif stimulus_method == "噪音":
            self.frequency_group_box.setDisabled(True)
            self.step_group_box.setDisabled(True)
        else:
            self.frequency_group_box.setEnabled(True)
            self.step_group_box.setEnabled(True)

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
        create_function = create_function_dict.get(self.stimulus_info["stimulus_method"])
        self.stimulus_data, _ = create_function(**self.stimulus_info)

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
        self.sync_voltage_info()

        # Generate the name of the stimulus signal based on the stimulus information
        stimulus_name = "_".join(str(value) for value in self.stimulus_info.values())
        stimulus_signal_path = model_consts.STORED_STIMULUS_PATH + "/" + stimulus_name + ".wav"
        save_audio_simple(
            stimulus_signal_path,
            self.stimulus_data.astype("float32"),
            self.stimulus_info["sample_rate"],
        )
        stimulus_signal_path = FileOps.get_relative_path(stimulus_signal_path, DEFAULT_DIR)
        if self.load_stimulus_signal_path:
            self.load_stimulus_signal_path = FileOps.get_relative_path(self.load_stimulus_signal_path, DEFAULT_DIR)

        data = {
            "stimulus_info": self.stimulus_info,
            "stimulus_signal_path": stimulus_signal_path,
            "load_stimulus_signal_path": self.load_stimulus_signal_path,
        }

        return data

    def update_stimulus_ui_value(self, stimulus_info: dict):
        """
            Update the user interface values for stimulus parameters.

            This function updates the values of various UI controls based on the stimulus parameters stored in
        `stimulus_info`.
            Specifically, it updates the stimulus method, stimulus type, start frequency, stop frequency, total time,
        repeat times, number of steps, voltage type, voltage value, and sample rate.
        """
        self.switch_connection_off()
        for k, v in self.STIMULUS_DICT.items():
            if v["name"] == stimulus_info.get("stimulus_method"):
                self.stimulus_method_combo_box.setCurrentText(k)
                self.set_stimulus_type_connection()
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
        if stimulus_info.get("use_custom_stimulus"):
            self.custom_chk_box.setChecked(stimulus_info.get("use_custom_stimulus", "False"))
        else:
            self.change_custom_chk_box(False)
        self.switch_connection_on()

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
        if self.stimulus_data is not None:
            signal_duration = np.linspace(0, len(self.stimulus_data) - 1, len(self.stimulus_data)) / sample_rate
            self.plot_stimulus.plot(signal_duration, self.stimulus_data, pen=pyqtgraph.mkPen("b", width=2))
            self.plot_stimulus.setLabel("left", "Amplitude", **{"font-size": "20px"})
            self.plot_stimulus.setLabel("bottom", "Time (s)", **{"font-size": "20px"})
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
            self.stimulus_info[stimulus_item] = loaded_stimulus[stimulus_item]
        self.update_stimulus_ui_value(self.stimulus_info)
        self.sync_voltage_info()
        self.switch_group_box_availability(True)
        self.create_signal_from_stimulus_info()
        self.graph_stimulus()

    def save_config_btn_clicked(self):
        """
        Handles the click event of the save configuration button.

        This function calls the `save_stimulus_info_to_db` method of the `StimulusSignalManagement` class
        to save the stimulus information from `stimulus_info` to the database. Based on the save result,
        it logs the corresponding message.
        """
        stimulus_name_dialog = SetConfigName()
        stimulus_name = stimulus_name_dialog.exec()
        if stimulus_name is not None:
            self.stimulus_info["stimulus_name"] = stimulus_name
        else:
            return
        self.sync_voltage_info()
        save_code, msg = StimulusSignalManagement().save_stimulus_info_to_db(self.stimulus_info)
        if save_code == error_code.OK:
            self.default_logger.info("Successfully saving stimulus info to database.")
            MessageBox.information(self, "保存配置", "激励信号保存成功.")
        elif save_code == error_code.INVALID_INSERT:
            self.default_logger.error("This stimulus signals info already exists.")
            MessageBox.warning(self, "保存配置", "激励信号已存在.")
        elif save_code == error_code.INVALID_SAVE:
            self.default_logger.error("Failed to save stimulus info to database.")
            MessageBox.warning(self, "保存配置", "保存激励信号信息到数据库失败.")
        elif save_code == error_code.INVALID_NAME:
            self.default_logger.error("Invalid stimulus name.")
            MessageBox.warning(self, "保存配置", "配置名称已存在.")

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
            self.stimulus_data, _ = load_audio_simple(self.load_wav_path, self.stimulus_info["sample_rate"])
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
                tuple: A dictionary containing the stimulus information.
                    Returns empty dictionary if the loading fails or the configuration is invalid.
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

    def load_stimulus_config_data(self, stimulus_data):
        self.stimulus_info = stimulus_data.get("stimulus_info")
        self.load_wav_path = stimulus_data.get("stimulus_signal_path")
        self.stimulus_data, _ = load_audio_simple(self.load_wav_path, self.stimulus_info["sample_rate"])
        if stimulus_data.get("load_stimulus_signal_path"):
            self.load_stimulus_signal_path = DEFAULT_DIR + stimulus_data.get("load_stimulus_signal_path")

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
        json_file_path = ""
        if default_config_flag:
            json_file_path = DEFAULT_DIR + "ui/ui_config/default_stimulus.json"
        if not os.path.exists(json_file_path):
            return error_code.INVALID_DATA_LOADING, "This json file does not exist."
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            return error_code.OK, data

    def default_config_btn_clicked(self):
        self.stimulus_info = self.get_stimulus_info_from_json(True)
        self.update_stimulus_ui_value(self.stimulus_info)
        self.sync_voltage_info()
        self.switch_group_box_availability(True)
        self.create_signal_from_stimulus_info()
        self.graph_stimulus()

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
            sr = self.stimulus_info.get("sample_rate", 44100)
            save_audio_simple(file_name, self.stimulus_data, sr)

    def play_btn_clicked(self):
        """
        Handles the play button click event to play the stimulus signal.

        This function retrieves the stimulus signal and related parameters from instance attributes,
        and uses an instance of the SoundcardAudioProcessor class to call the sd_play method
        for playing the signal. If the playback fails, an error log is recorded.
        """
        # Extract device index from speaker if available
        device_idx = self.speaker["index"] if self.speaker else None

        # Construct the stimulus parameter dictionary, including signal data, amplitude, and sample rate
        stimulus_param = {
            "data": self.stimulus_data,
            "amplitude": self.stimulus_info["amplitude"],
            "sr": self.stimulus_info["sample_rate"],
            "device": device_idx,
        }
        # Create an instance of SoundcardAudioProcessor and play the stimulus signal
        sap = SoundcardAudioProcessor()
        play_code, msg = sap.sd_play(stimulus_param)
        # If playback fails, log the error
        if play_code != error_code.OK:
            self.default_logger.error(f"Failed to play the stimulus file. {msg}")

    def ok_btn_clicked(self):
        self.refresh_stimulus_info = True
        self.sync_voltage_info()
        if not self.custom_chk_box.isChecked():
            if self.start_custom_check_status:
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
        else:
            data = self.save_stimulus_to_json()
            stimulus_signal_length = self.stimulus_info.get("sample_rate") * self.stimulus_info.get("total_time")
            if not self.start_custom_check_status:
                self.set_ai_popup()
            elif self.original_stimulus_signal_length != stimulus_signal_length:
                self.set_ai_popup()
        self.final_save_data = data
        self.close()

    def load_stimulus_wav(self):
        data = {}
        if self.load_wav_path:
            data = {
                "stimulus_info": self.stimulus_info,
                "stimulus_signal_path": self.load_wav_path,
                "load_stimulus_signal_path": self.load_stimulus_signal_path,
            }
        return data

    def miss_popup(self):
        error_msg = MessageBox(self)
        error_msg.setIcon(MessageBox.Warning)
        error_msg.setText("请先配置激励信号！")
        error_msg.setWindowTitle("未配置激励信号")
        error_msg.setStandardButtons(MessageBox.Ok)
        button = error_msg.exec_()
        return button == MessageBox.Ok

    def set_ai_popup(self):
        error_msg = MessageBox(self)
        error_msg.setIcon(MessageBox.Warning)
        error_msg.setText("激励信号变化, 请重新配置AI分析模型!")
        error_msg.setWindowTitle("配置AI模型")
        error_msg.setStandardButtons(MessageBox.Ok)
        button = error_msg.exec_()
        return button == MessageBox.Ok

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


class SetConfigName(QDialog):

    def __init__(self, parent=None):
        super(SetConfigName, self).__init__(parent)

        self.stimulus_name = None
        self.clicked_ok_close = False

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("设置配置名称")
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))

        congfig_name_layout = self.config_name_layout()
        btn_layout = self.btn_layout()

        layout = QVBoxLayout()
        layout.addLayout(congfig_name_layout)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    @staticmethod
    def config_name_layout():
        name_label = Label("配置名称:")
        name_edit = LineEdit()
        name_edit.setPlaceholderText("请输入配置名称")
        name_layout = QHBoxLayout()
        name_layout.addWidget(name_label)
        name_layout.addWidget(name_edit)

        return name_layout

    def btn_layout(self):
        ok_btn = PushButton("确定")
        cancel_btn = PushButton("取消")
        ok_btn.clicked.connect(self.on_click_ok_btn)
        cancel_btn.clicked.connect(self.on_click_cancel_btn)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        return btn_layout

    def on_click_ok_btn(self):
        self.stimulus_name = self.findChild(LineEdit).text()
        if not self.stimulus_name:
            MessageBox.warning(self, "警告", "请输入配置名称")
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
