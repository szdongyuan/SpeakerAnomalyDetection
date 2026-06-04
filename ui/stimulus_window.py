import json
import math
import os.path
import sys
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
from base.stimulus_signal import generate_frequency_stepped, normalize_stimulus_method, preferred_octave_frequencies
from base.stimulus_signal_management import StimulusSignalManagement
from consts import error_code, model_consts, ui_style_const
from consts.frequency_stepped_consts import (
    FREQUENCY_STEPPED_LABEL,
    FREQUENCY_STEPPED_METHOD,
    FREQUENCY_STEPPED_MODES,
    FREQUENCY_STEPPED_RESOLUTIONS,
)
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


class PreferredFrequencySpinBox(DoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._preferred_frequencies = []
        self.setDecimals(0)
        self.setSingleStep(1.0)

    def setPreferredFrequencies(self, frequencies):
        self._preferred_frequencies = sorted({float(value) for value in frequencies})

    def clearPreferredFrequencies(self):
        self._preferred_frequencies = []

    def nearestPreferredFrequency(self, value, *, tie="lower"):
        if not self._preferred_frequencies:
            return float(value)
        value = float(value)
        distances = [abs(candidate - value) for candidate in self._preferred_frequencies]
        min_distance = min(distances)
        candidates = [
            candidate
            for candidate, distance in zip(self._preferred_frequencies, distances)
            if math.isclose(distance, min_distance, rel_tol=0.0, abs_tol=1e-12)
        ]
        if tie == "upper":
            return max(candidates)
        return min(candidates)

    def stepBy(self, steps):
        if not self._preferred_frequencies or steps == 0:
            super().stepBy(steps)
            return

        value = float(self.value())
        target = value
        for _ in range(abs(int(steps))):
            if steps > 0:
                higher = [candidate for candidate in self._preferred_frequencies if candidate > target + 1e-12]
                target = higher[0] if higher else self._preferred_frequencies[-1]
            else:
                lower = [candidate for candidate in self._preferred_frequencies if candidate < target - 1e-12]
                target = lower[-1] if lower else self._preferred_frequencies[0]
        self.setValue(target)


class StimulusWindow(QDialog):
    STEP_SC_METHOD_DISPLAY_LABEL = "步进（sc）"
    # Define stimulus signal types
    STIMULUS_DICT = {
        "啁啾": {"name": "chirp", "sub_list": ["对数镜像", "线性镜像", "对数", "线性"]},
        "步进": {"name": "step", "sub_list": ["对数", "线性"]},
        STEP_SC_METHOD_DISPLAY_LABEL: {
            "name": "frequency_stepped",
            "sub_list": ["倍频程", "自定义线性", "自定义对数"],
        },
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
        "倍频程": "octave",
        "自定义线性": "custom_linear",
        "自定义对数": "custom_log",
    }
    STEP_SC_MODE_LABELS = {
        "octave": "倍频程",
        "custom_linear": "自定义线性",
        "custom_log": "自定义对数",
    }
    STEP_SC_VALID_MODES = set(FREQUENCY_STEPPED_MODES)
    STEP_SC_RESOLUTION_LABELS = {
        "R3": "R3 (1/1 Oct.)",
        "R10": "R10 (1/3 Oct.)",
        "R20": "R20 (1/6 Oct.)",
        "R40": "R40 (1/12 Oct.)",
        "R80": "R80 (1/24 Oct.)",
    }
    STEP_SC_VALID_RESOLUTIONS = set(FREQUENCY_STEPPED_RESOLUTIONS)
    STEP_SC_FREQUENCY_DRIVERS = {"start_freq", "stop_freq", "num_steps", "frequency_mode", "resolution", "sample_rate"}
    STEP_SC_DEFAULT_MIN_DURATION = 0.1
    STEP_SC_DEFAULT_MIN_CYCLES = 8.0
    STEP_SC_DEFAULT_RESOLUTION = "R10"
    LEGACY_TOTAL_TIME_RANGE = (0.5, 60.0)
    LEGACY_TOTAL_TIME_DECIMALS = 1
    LEGACY_TOTAL_TIME_STEP = 0.5
    STEP_SC_TOTAL_TIME_RANGE = (0.0, 1000000.0)
    STEP_SC_TOTAL_TIME_DECIMALS = 6
    STEP_SC_TOTAL_TIME_STEP = 0.001
    LEGACY_STEP_COUNT_RANGE = (1, 100)
    STEP_SC_STEP_COUNT_RANGE = (1, 2147483647)
    STEP_SC_FILENAME_KEYS = (
        "stimulus_method",
        "frequency_mode",
        "start_freq",
        "stop_freq",
        "num_steps",
        "resolution",
        "min_duration",
        "min_cycles",
        "repeat_times",
        "sample_rate",
        "voltage_type",
        "voltage",
    )
    STEP_SC_RICH_METADATA_KEYS = {
        "stimulus_label",
        "frequency_mode",
        "resolution",
        "effective_start_freq",
        "effective_stop_freq",
        "frequencies",
        "min_duration",
        "min_cycles",
        "schedule_sample_rate",
        "schedule_provenance",
        "transition_hz",
        "safe_max_freq",
        "frequency_clamped",
        "per_repetition_sample_count",
        "alignment_sample_count",
        "playback_sample_count",
        "fadeout_tail_duration_s",
        "fadeout_tail_exponent",
        "fadeout_tail_sample_count",
        "segments",
        "step_durations",
        "schedule_algorithm",
    }
    STEP_SC_LEGACY_EXTERNAL_WAV_KEYS = {
        "load_stimulus_signal_path",
    }
    SUPPORTED_STIMULUS_METHODS = {"chirp", "step", "noise", "frequency_stepped"}

    def __init__(self, stimulus_config_data=None, speaker=None):
        """Initialize stimulus window with default configurations"""
        super().__init__()
        # initialize stimulus signal type， and create variable to store stimulus signal data
        self.speaker = speaker
        self.load_wav_path = ""
        self.load_stimulus_signal_path = None
        self.refresh_stimulus_info = False
        self.is_close_window = False
        self.stimulus_data = None
        self.default_logger = LogManager.set_log_handler("core")
        self.box_checked_enable_dict = {}
        self.box_checked_disable_list = []
        self.final_save_data = None

        # create variable to set stimulus signal data
        self.stimulus_method_combo_box = ComboBox()
        self.stimulus_type_combo_box = ComboBox()
        self.start_freq_box = PreferredFrequencySpinBox()
        self.stop_freq_box = PreferredFrequencySpinBox()
        self.total_time_box = DoubleSpinBox()
        self.repeat_box = SpinBox()
        self.step_box = SpinBox()
        self.voltage_combo_box = ComboBox()
        self.voltage_spin_box = DoubleSpinBox()
        self.sample_rate_combo_box = ComboBox()
        self.min_duration_box = DoubleSpinBox()
        self.min_cycles_box = DoubleSpinBox()
        self.resolution_combo_box = ComboBox()
        self.transition_hz_box = DoubleSpinBox()
        self._step_sc_retained_frequency_state = "none"
        self._step_sc_retained_frequencies = None
        self._step_sc_intended_start_freq = None
        self._step_sc_intended_stop_freq = None
        self._step_sc_last_manual_start_freq = None
        self._step_sc_last_manual_stop_freq = None
        self._step_sc_active_manual_frequency_edit = None
        self._step_sc_active_manual_frequency_previous_direction = None
        self._pre_step_sc_legacy_branch_snapshot = None
        self._step_sc_restore_in_progress = False
        self._warn_on_missing_legacy_wav = False
        self._legacy_external_wav_loaded_by_user = False
        # rcreate variable to control whether to use frequency or step
        self.frequency_group_box = self.create_frequency_group_box()
        self.step_group_box = self.create_step_group_box()
        self.step_sc_group_box = self.create_step_sc_group_box()
        self.stimulus_config_data = deepcopy(stimulus_config_data)
        self.load_stimulus_config_data(self.stimulus_config_data)
        self.start_custom_check_status = self.stimulus_info.get("use_custom_stimulus", False)
        self.init_ui()
        self.update_stimulus_ui_value(self.stimulus_info)
        if self.stimulus_info["use_custom_stimulus"]:
            self.create_signal_from_stimulus_info()
        self.graph_stimulus()

        original_total_time = self.stimulus_info.get("total_time")
        try:
            self.original_stimulus_signal_length = float(self.stimulus_info.get("sample_rate")) * float(
                original_total_time
            )
        except (TypeError, ValueError):
            self.original_stimulus_signal_length = 0 if self.stimulus_data is None else len(self.stimulus_data)

    def init_ui(self):
        # set window titlebar style
        self.setObjectName("StimulusWindow")
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))
        self.setWindowTitle("激励信号")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        # if ui_style_const._FONT_SCALE < 0.8:
        #     self.setFixedSize(
        #         ui_style_const.scale_size_px(630),
        #         ui_style_const.scale_size_px(780),
        #     )
        # elif ui_style_const._FONT_SCALE < 0.9:
        #     self.setFixedSize(
        #         ui_style_const.scale_size_px(580),
        #         ui_style_const.scale_size_px(750),
        #     )
        # elif ui_style_const._FONT_SCALE < 0.95:
        #     self.setFixedSize(
        #         ui_style_const.scale_size_px(530),
        #         ui_style_const.scale_size_px(730),
        #     )
        # elif ui_style_const._FONT_SCALE < 1.05:
        #     self.setFixedSize(
        #         ui_style_const.scale_size_px(515),
        #         ui_style_const.scale_size_px(650),
        #     )
        # # elif ui_style_const._FONT_SCALE < 1.1:
        # else:
        #     self.setFixedSize(
        #         ui_style_const.scale_size_px(500),
        #         ui_style_const.scale_size_px(650),
        #     )

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
        layout.addWidget(self.step_sc_group_box)
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
            "frequency_stepped": [
                load_config_btn,
                save_config_btn,
                stimulus_type_group_box,
                self.frequency_group_box,
                time_group_box,
                self.step_group_box,
                self.step_sc_group_box,
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
        self.min_duration_box.valueChanged.connect(
            lambda: self.update_stimulus_info_from_controller(self.min_duration_box, "min_duration")
        )
        self.min_cycles_box.valueChanged.connect(
            lambda: self.update_stimulus_info_from_controller(self.min_cycles_box, "min_cycles")
        )
        self.resolution_combo_box.currentTextChanged.connect(
            lambda: self.update_stimulus_info_from_controller(self.resolution_combo_box, "resolution")
        )
        self.voltage_combo_box.currentTextChanged.connect(self.update_stimulus_info_from_voltage_combo_box)

        self.voltage_spin_box.valueChanged.connect(
            lambda: self.update_stimulus_info_from_controller(self.voltage_spin_box, "voltage")
        )
        self.sample_rate_combo_box.currentTextChanged.connect(
            lambda: self.update_stimulus_info_from_controller(self.sample_rate_combo_box, "sample_rate")
        )

    def switch_connection_off(self):
        for widget in [
            self.stimulus_type_combo_box,
            self.stimulus_method_combo_box,
            self.start_freq_box,
            self.stop_freq_box,
            self.total_time_box,
            self.repeat_box,
            self.step_box,
            self.min_duration_box,
            self.min_cycles_box,
            self.resolution_combo_box,
            self.voltage_spin_box,
            self.sample_rate_combo_box,
        ]:
            try:
                widget.disconnect()
            except TypeError:
                pass

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
        self.stimulus_method_combo_box.addItems(["啁啾", "步进", self.STEP_SC_METHOD_DISPLAY_LABEL, "噪音"])
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
        self._configure_legacy_total_time_box()
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

    def _configure_legacy_total_time_box(self):
        self.total_time_box.setDecimals(self.LEGACY_TOTAL_TIME_DECIMALS)
        self.total_time_box.setRange(*self.LEGACY_TOTAL_TIME_RANGE)
        self.total_time_box.setSingleStep(self.LEGACY_TOTAL_TIME_STEP)
        self.total_time_box.setReadOnly(False)
        self.total_time_box.setButtonSymbols(DoubleSpinBox.UpDownArrows)

    def _configure_step_sc_total_time_box(self):
        self.total_time_box.setDecimals(self.STEP_SC_TOTAL_TIME_DECIMALS)
        self.total_time_box.setRange(*self.STEP_SC_TOTAL_TIME_RANGE)
        self.total_time_box.setSingleStep(self.STEP_SC_TOTAL_TIME_STEP)
        self.total_time_box.setReadOnly(True)
        self.total_time_box.setButtonSymbols(DoubleSpinBox.NoButtons)

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
        self.step_box.setRange(*self.LEGACY_STEP_COUNT_RANGE)
        step_layout = QHBoxLayout()
        step_layout.addWidget(step_label)
        step_layout.addWidget(self.step_box)
        step_layout.setContentsMargins(10, 10, 10, 10)
        step_group_box.setLayout(step_layout)
        return step_group_box

    def create_step_sc_group_box(self):
        step_sc_group_box = GroupBox(FREQUENCY_STEPPED_LABEL)

        min_duration_label = Label("最短时长:")
        self.min_duration_box.setSuffix(" s")
        self.min_duration_box.setDecimals(4)
        self.min_duration_box.setRange(0.0001, sys.float_info.max)
        self.min_duration_box.setSingleStep(0.0005)
        self.min_duration_box.setValue(self.STEP_SC_DEFAULT_MIN_DURATION)
        self.min_duration_box.setMinimumWidth(90)

        min_cycles_label = Label("最少周期:")
        self.min_cycles_box.setDecimals(1)
        self.min_cycles_box.setRange(0.1, sys.float_info.max)
        self.min_cycles_box.setSingleStep(0.5)
        self.min_cycles_box.setValue(self.STEP_SC_DEFAULT_MIN_CYCLES)
        self.min_cycles_box.setMinimumWidth(90)

        resolution_label = Label("分 辨 率:")
        for code, label in self.STEP_SC_RESOLUTION_LABELS.items():
            self.resolution_combo_box.addItem(label, code)
        self._set_step_sc_resolution_code(self.STEP_SC_DEFAULT_RESOLUTION)

        transition_label = Label("转换频率:")
        self.transition_hz_box.setSuffix(" Hz")
        self.transition_hz_box.setDecimals(1)
        self.transition_hz_box.setRange(0.0, 1000000.0)
        self.transition_hz_box.setReadOnly(True)
        self.transition_hz_box.setButtonSymbols(DoubleSpinBox.NoButtons)
        self.transition_hz_box.setMinimumWidth(90)

        step_sc_layout = QGridLayout()
        step_sc_layout.addWidget(min_duration_label, 0, 0)
        step_sc_layout.addWidget(self.min_duration_box, 0, 1)
        step_sc_layout.addWidget(min_cycles_label, 0, 2)
        step_sc_layout.addWidget(self.min_cycles_box, 0, 3)
        step_sc_layout.addWidget(resolution_label, 1, 0)
        step_sc_layout.addWidget(self.resolution_combo_box, 1, 1)
        step_sc_layout.addWidget(transition_label, 1, 2)
        step_sc_layout.addWidget(self.transition_hz_box, 1, 3)
        step_sc_layout.setContentsMargins(10, 10, 10, 10)
        step_sc_layout.setSpacing(10)
        step_sc_group_box.setLayout(step_sc_layout)
        step_sc_group_box.setVisible(False)
        return step_sc_group_box

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

    def _is_step_sc_active(self):
        return normalize_stimulus_method(self.stimulus_info.get("stimulus_method")) == FREQUENCY_STEPPED_METHOD

    @staticmethod
    def _normalize_stimulus_info_method(stimulus_info):
        if stimulus_info is None:
            return stimulus_info
        stimulus_info["stimulus_method"] = normalize_stimulus_method(stimulus_info.get("stimulus_method", ""))
        return stimulus_info

    @classmethod
    def _is_supported_stimulus_method(cls, stimulus_info):
        return stimulus_info.get("stimulus_method") in cls.SUPPORTED_STIMULUS_METHODS

    def _unsupported_stimulus_method_fallback(self, method):
        MessageBox.warning(
            self,
            "导入配置",
            f"不支持的激励信号类型 {method}，已切换为默认激励。",
        )
        self._step_sc_retained_frequencies = None
        self._step_sc_retained_frequency_state = "none"
        self._clear_loaded_legacy_external_paths()
        return self._fallback_stimulus_info()

    def _ensure_step_sc_defaults(self):
        self.stimulus_info["stimulus_method"] = FREQUENCY_STEPPED_METHOD
        self.stimulus_info["stimulus_label"] = FREQUENCY_STEPPED_LABEL
        self.stimulus_info["use_custom_stimulus"] = True
        mode = self.stimulus_info.get("frequency_mode") or self.stimulus_info.get("stimulus_type") or "octave"
        if mode not in self.STEP_SC_VALID_MODES:
            mode = "octave"
        self.stimulus_info["frequency_mode"] = mode
        self.stimulus_info["stimulus_type"] = mode
        self.stimulus_info.setdefault("min_duration", self.STEP_SC_DEFAULT_MIN_DURATION)
        self.stimulus_info.setdefault("min_cycles", self.STEP_SC_DEFAULT_MIN_CYCLES)
        if mode == "octave":
            if self.stimulus_info.get("resolution") not in self.STEP_SC_VALID_RESOLUTIONS:
                self.stimulus_info["resolution"] = self.STEP_SC_DEFAULT_RESOLUTION

    def _set_step_sc_frequency_mode(self, mode):
        if mode not in self.STEP_SC_VALID_MODES:
            mode = "octave"
        previous_mode = self.stimulus_info.get("frequency_mode")
        self.stimulus_info["frequency_mode"] = mode
        self.stimulus_info["stimulus_type"] = mode
        if mode == "octave":
            if self.stimulus_info.get("resolution") not in self.STEP_SC_VALID_RESOLUTIONS:
                self.stimulus_info["resolution"] = self.STEP_SC_DEFAULT_RESOLUTION
        else:
            self.stimulus_info["resolution"] = None
        if previous_mode is not None and previous_mode != mode:
            self._mark_step_sc_frequency_dirty()
            if mode == "octave":
                self._set_step_sc_intended_frequency_bounds_from_controls()
            else:
                self._clear_step_sc_intended_frequency_bounds()
        self._apply_step_sc_control_state()

    def _mark_step_sc_frequency_dirty(self):
        if self._is_step_sc_active():
            self._step_sc_retained_frequency_state = "dirty"

    def _step_sc_snapshot(self):
        return {
            "stimulus_info": deepcopy(self.stimulus_info),
            "stimulus_data": None if self.stimulus_data is None else self.stimulus_data.copy(),
            "retained_state": self._step_sc_retained_frequency_state,
            "retained_frequencies": (
                None if self._step_sc_retained_frequencies is None else list(self._step_sc_retained_frequencies)
            ),
            "intended_start_freq": self._step_sc_intended_start_freq,
            "intended_stop_freq": self._step_sc_intended_stop_freq,
            "last_manual_start_freq": self._step_sc_last_manual_start_freq,
            "last_manual_stop_freq": self._step_sc_last_manual_stop_freq,
            "active_manual_frequency_edit": self._step_sc_active_manual_frequency_edit,
            "active_manual_frequency_previous_direction": self._step_sc_active_manual_frequency_previous_direction,
        }

    def _restore_step_sc_snapshot(self, snapshot):
        self.stimulus_info = deepcopy(snapshot["stimulus_info"])
        self.stimulus_data = None if snapshot["stimulus_data"] is None else snapshot["stimulus_data"].copy()
        self._step_sc_retained_frequency_state = snapshot["retained_state"]
        self._step_sc_retained_frequencies = (
            None if snapshot["retained_frequencies"] is None else list(snapshot["retained_frequencies"])
        )
        self._step_sc_restore_in_progress = True
        try:
            self.update_stimulus_ui_value(self.stimulus_info)
        finally:
            self._step_sc_restore_in_progress = False
        self._step_sc_intended_start_freq = snapshot.get("intended_start_freq")
        self._step_sc_intended_stop_freq = snapshot.get("intended_stop_freq")
        self._step_sc_last_manual_start_freq = snapshot.get("last_manual_start_freq")
        self._step_sc_last_manual_stop_freq = snapshot.get("last_manual_stop_freq")
        self._step_sc_active_manual_frequency_edit = snapshot.get("active_manual_frequency_edit")
        self._step_sc_active_manual_frequency_previous_direction = snapshot.get(
            "active_manual_frequency_previous_direction"
        )

    def _stimulus_state_snapshot(self):
        snapshot = self._step_sc_snapshot()
        snapshot["load_wav_path"] = self.load_wav_path
        snapshot["load_stimulus_signal_path"] = self.load_stimulus_signal_path
        snapshot["legacy_external_wav_loaded_by_user"] = self._legacy_external_wav_loaded_by_user
        return snapshot

    def _restore_stimulus_state_snapshot(self, snapshot):
        self.load_wav_path = snapshot["load_wav_path"]
        self.load_stimulus_signal_path = snapshot["load_stimulus_signal_path"]
        self._legacy_external_wav_loaded_by_user = snapshot.get("legacy_external_wav_loaded_by_user", False)
        self._restore_step_sc_snapshot(snapshot)

    @staticmethod
    def _legacy_external_wav_snapshot_is_restorable(snapshot):
        if not snapshot:
            return False
        stimulus_info = snapshot.get("stimulus_info") or {}
        if normalize_stimulus_method(stimulus_info.get("stimulus_method")) == FREQUENCY_STEPPED_METHOD:
            return False
        if stimulus_info.get("use_custom_stimulus") is not False:
            return False
        if not snapshot.get("load_stimulus_signal_path"):
            return False
        return snapshot.get("stimulus_data") is not None

    def _restore_pre_step_sc_legacy_external_branch(self):
        snapshot = self._pre_step_sc_legacy_branch_snapshot
        if not self._legacy_external_wav_snapshot_is_restorable(snapshot):
            return False

        target_method = self.stimulus_info.get("stimulus_method")
        snapshot_method = normalize_stimulus_method(snapshot["stimulus_info"].get("stimulus_method"))
        if target_method != snapshot_method:
            return False

        restored_info = self._strip_step_sc_rich_metadata(deepcopy(snapshot["stimulus_info"]))
        restored_info["use_custom_stimulus"] = False
        self.stimulus_info = restored_info
        self.load_wav_path = snapshot.get("load_wav_path") or snapshot.get("load_stimulus_signal_path")
        self.load_stimulus_signal_path = snapshot.get("load_stimulus_signal_path") or self.load_wav_path
        self._legacy_external_wav_loaded_by_user = True
        self.stimulus_data = snapshot["stimulus_data"].copy()

        self.switch_connection_off()
        previous_signal_state = self.custom_chk_box.blockSignals(True)
        try:
            for label, stimulus_item in self.STIMULUS_DICT.items():
                if stimulus_item["name"] == restored_info.get("stimulus_method"):
                    self.stimulus_method_combo_box.setCurrentText(label)
                    break
            for label, stimulus_type in self.STIMULUS_DICT_2.items():
                if stimulus_type == restored_info.get("stimulus_type"):
                    self.stimulus_type_combo_box.setCurrentText(label)
                    break
            self.start_freq_box.setValue(int(restored_info.get("start_freq", 80)))
            self.stop_freq_box.setValue(int(restored_info.get("stop_freq", 2000)))
            self.total_time_box.setValue(float(restored_info.get("total_time", 4)))
            self.repeat_box.setValue(int(restored_info.get("repeat_times", 1)))
            self.step_box.setValue(int(restored_info.get("num_steps", 3)))
            self.voltage_combo_box.setCurrentText(restored_info.get("voltage_type", "RMS"))
            self.voltage_spin_box.setValue(float(restored_info.get("voltage", 2.0)))
            self.sample_rate_combo_box.setCurrentText(str(restored_info.get("sample_rate", 44100)))
            self.custom_chk_box.setChecked(False)
        finally:
            self.custom_chk_box.blockSignals(previous_signal_state)
            self.switch_connection_on()
        self.switch_group_box_availability(False)
        self._configure_legacy_total_time_box()
        self.graph_stimulus()
        return True

    def _clear_loaded_legacy_external_paths(self):
        self.load_wav_path = None
        self.load_stimulus_signal_path = None
        self._legacy_external_wav_loaded_by_user = False

    def _clear_imported_step_sc_legacy_external_authority(self):
        self._clear_loaded_legacy_external_paths()
        self._pre_step_sc_legacy_branch_snapshot = None

    def _adopt_legacy_external_wav_path_from_payload(self, payload):
        external_path = payload.get("load_stimulus_signal_path")
        if not external_path:
            return False
        self.load_stimulus_signal_path = external_path
        self.load_wav_path = external_path
        self._legacy_external_wav_loaded_by_user = True
        return True

    def _has_authoritative_legacy_external_wav(self):
        return bool(self.load_wav_path) and (
            bool(self.load_stimulus_signal_path) or self._legacy_external_wav_loaded_by_user
        )

    @staticmethod
    def _is_missing_legacy_external_path_payload(stimulus_info, payload):
        return (
            normalize_stimulus_method(stimulus_info.get("stimulus_method")) != FREQUENCY_STEPPED_METHOD
            and stimulus_info.get("use_custom_stimulus") is False
            and not payload.get("load_stimulus_signal_path")
        )

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {str(k): StimulusWindow._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [StimulusWindow._json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return StimulusWindow._json_safe(value.tolist())
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        return value

    @staticmethod
    def _float_frequency_list(values):
        if values is None:
            return None
        if isinstance(values, (str, bytes, bytearray)):
            raise ValueError("frequencies must be a list")
        frequencies = [float(value) for value in values]
        if not frequencies:
            raise ValueError("frequencies must not be empty")
        if any(not math.isfinite(value) or value <= 0 for value in frequencies):
            raise ValueError("frequencies must be finite positive values")
        return frequencies

    @staticmethod
    def _is_strictly_monotonic_or_single(values):
        if len(values) <= 1:
            return True
        deltas = [right - left for left, right in zip(values, values[1:])]
        return all(delta > 0 for delta in deltas) or all(delta < 0 for delta in deltas)

    @staticmethod
    def _scalar_bounds_direction_matches_retained(start_freq, stop_freq, retained):
        try:
            start = float(start_freq)
            stop = float(stop_freq)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(start) or not math.isfinite(stop) or start <= 0 or stop <= 0:
            return False
        if len(retained) <= 1:
            return True
        retained_delta = float(retained[-1]) - float(retained[0])
        if math.isclose(retained_delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
            return True
        scalar_delta = stop - start
        if math.isclose(scalar_delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
            return False
        return (scalar_delta > 0) == (retained_delta > 0)

    @staticmethod
    def _apply_retained_octave_scalar_bounds(stimulus_info, retained):
        if StimulusWindow._scalar_bounds_direction_matches_retained(
            stimulus_info.get("start_freq"),
            stimulus_info.get("stop_freq"),
            retained,
        ):
            stimulus_info["start_freq"] = float(stimulus_info.get("start_freq"))
            stimulus_info["stop_freq"] = float(stimulus_info.get("stop_freq"))
        else:
            stimulus_info["start_freq"] = float(retained[0])
            stimulus_info["stop_freq"] = float(retained[-1])
        stimulus_info["effective_start_freq"] = float(retained[0])
        stimulus_info["effective_stop_freq"] = float(retained[-1])

    @staticmethod
    def _require_step_sc_fields(stimulus_info, field_names):
        missing = [
            field_name
            for field_name in field_names
            if field_name not in stimulus_info or stimulus_info.get(field_name) in (None, "")
        ]
        if missing:
            raise ValueError(f"{', '.join(missing)} is required for step(sc) hydration")

    def _prepare_frequency_stepped_info(self, stimulus_info):
        repaired = deepcopy(stimulus_info)
        self._normalize_stimulus_info_method(repaired)
        for key in self.STEP_SC_LEGACY_EXTERNAL_WAV_KEYS:
            repaired.pop(key, None)
        mode = repaired.get("frequency_mode")
        stimulus_type = repaired.get("stimulus_type")
        if mode not in self.STEP_SC_VALID_MODES:
            if stimulus_type not in self.STEP_SC_VALID_MODES:
                raise ValueError("frequency_mode is required")
            mode = stimulus_type
        repaired["stimulus_method"] = FREQUENCY_STEPPED_METHOD
        repaired["stimulus_label"] = FREQUENCY_STEPPED_LABEL
        repaired["frequency_mode"] = mode
        repaired["stimulus_type"] = mode
        repaired["use_custom_stimulus"] = True
        self._require_step_sc_fields(repaired, ("min_duration", "min_cycles"))

        retained = None
        if repaired.get("frequencies") is not None:
            retained = self._float_frequency_list(repaired.get("frequencies"))
            repaired["frequencies"] = list(retained)
            repaired["num_steps"] = len(retained)
            if mode == "octave":
                if repaired.get("resolution") not in self.STEP_SC_VALID_RESOLUTIONS:
                    raise ValueError("resolution is required for editable octave step(sc) hydration")
                if not self._is_strictly_monotonic_or_single(retained):
                    raise ValueError("octave retained frequencies must be strictly monotonic")
                self._apply_retained_octave_scalar_bounds(repaired, retained)
            else:
                repaired.setdefault("start_freq", float(retained[0]))
                repaired.setdefault("stop_freq", float(retained[-1]))
        else:
            if mode == "octave":
                self._require_step_sc_fields(repaired, ("start_freq", "stop_freq", "resolution"))
                if repaired.get("resolution") not in self.STEP_SC_VALID_RESOLUTIONS:
                    raise ValueError("resolution is required for editable octave step(sc) hydration")
            else:
                self._require_step_sc_fields(repaired, ("start_freq", "stop_freq", "num_steps"))

        state = "clean" if retained is not None else "dirty"
        return repaired, retained, state

    def _strip_step_sc_rich_metadata(self, stimulus_info):
        cleaned = deepcopy(stimulus_info)
        self._normalize_stimulus_info_method(cleaned)
        if cleaned.get("stimulus_method") != FREQUENCY_STEPPED_METHOD:
            for key in self.STEP_SC_RICH_METADATA_KEYS:
                cleaned.pop(key, None)
        return cleaned

    def _legacy_stimulus_info_from_payload(self, loaded_stimulus):
        candidate = self._fallback_stimulus_info()
        candidate.update(deepcopy(loaded_stimulus))
        return self._strip_step_sc_rich_metadata(candidate)

    def _sync_current_legacy_stimulus_type(self):
        if self._is_step_sc_active() or self.stimulus_type_combo_box.count() == 0:
            return
        stimulus_type = self.STIMULUS_DICT_2.get(self.stimulus_type_combo_box.currentText())
        if stimulus_type is not None:
            self.stimulus_info["stimulus_type"] = stimulus_type

    def _frequency_stepped_generation_kwargs_from_info(self, stimulus_info, retained, retained_state):
        retained_frequencies = None
        if self._retained_frequencies_compatible_with_info(stimulus_info, retained, retained_state):
            retained_frequencies = list(retained)
        return {
            "sample_rate": int(stimulus_info.get("sample_rate", 44100)),
            "repeat_times": int(stimulus_info.get("repeat_times", 1)),
            "min_duration": float(stimulus_info.get("min_duration", self.STEP_SC_DEFAULT_MIN_DURATION)),
            "min_cycles": float(stimulus_info.get("min_cycles", self.STEP_SC_DEFAULT_MIN_CYCLES)),
            "frequency_mode": stimulus_info.get("frequency_mode"),
            "stimulus_type": stimulus_info.get("stimulus_type"),
            "start_freq": stimulus_info.get("start_freq"),
            "stop_freq": stimulus_info.get("stop_freq"),
            "num_steps": stimulus_info.get("num_steps"),
            "resolution": stimulus_info.get("resolution"),
            "frequencies": retained_frequencies,
            "amplitude": float(stimulus_info.get("amplitude", 1.0)),
            "generate_waveform": True,
        }

    @staticmethod
    def _retained_frequencies_match_payload(payload_frequencies, retained):
        if payload_frequencies is None or retained is None:
            return False
        try:
            payload = [float(value) for value in payload_frequencies]
        except (TypeError, ValueError):
            return False
        if len(payload) != len(retained):
            return False
        return all(
            math.isclose(left, float(right), rel_tol=1e-9, abs_tol=1e-9) for left, right in zip(payload, retained)
        )

    def _retained_frequencies_compatible_with_info(self, stimulus_info, retained, retained_state):
        if retained_state != "clean" or retained is None:
            return False
        if not self._retained_frequencies_match_payload(stimulus_info.get("frequencies"), retained):
            return False
        mode = stimulus_info.get("frequency_mode") or stimulus_info.get("stimulus_type")
        if mode == "octave":
            return stimulus_info.get(
                "resolution"
            ) in self.STEP_SC_VALID_RESOLUTIONS and self._is_strictly_monotonic_or_single(retained)
        return mode in {"custom_linear", "custom_log"}

    def _generate_frequency_stepped_candidate(self, candidate, retained, retained_state):
        result = generate_frequency_stepped(
            **self._frequency_stepped_generation_kwargs_from_info(candidate, retained, retained_state)
        )
        metadata = self._json_safe(result.metadata)
        preserved = {
            key: value
            for key, value in candidate.items()
            if key not in result.metadata or key in {"voltage_type", "voltage", "use_custom_stimulus", "stimulus_name"}
        }
        generated_info = deepcopy(preserved)
        generated_info.update(metadata)
        generated_info["stimulus_method"] = FREQUENCY_STEPPED_METHOD
        generated_info["stimulus_label"] = FREQUENCY_STEPPED_LABEL
        generated_info["use_custom_stimulus"] = True
        generated_info["frequency_mode"] = metadata["frequency_mode"]
        generated_info["stimulus_type"] = metadata["frequency_mode"]
        generated_retained = [float(value) for value in metadata["frequencies"]]
        if metadata["frequency_mode"] == "octave" and self._retained_frequencies_compatible_with_info(
            candidate, retained, retained_state
        ):
            generated_info["start_freq"] = candidate.get("start_freq")
            generated_info["stop_freq"] = candidate.get("stop_freq")
            self._apply_retained_octave_scalar_bounds(generated_info, generated_retained)
        return generated_info, result.data, generated_retained, "clean"

    def _commit_frequency_stepped_candidate(self, generated_info, generated_data, retained, retained_state):
        self.stimulus_info = generated_info
        self.stimulus_data = generated_data
        self._step_sc_retained_frequencies = retained
        self._step_sc_retained_frequency_state = retained_state
        self._update_step_sc_derived_controls()
        self._apply_step_sc_control_state()
        self._display_frequency_stepped_effective_bounds()
        if generated_info.get("frequency_mode") == "octave":
            self._retire_step_sc_manual_frequency_pair_if_complete()
        if "num_steps" in generated_info:
            previous_signal_state = self.step_box.blockSignals(True)
            try:
                self.step_box.setRange(*self.STEP_SC_STEP_COUNT_RANGE)
                self.step_box.setValue(int(generated_info["num_steps"]))
            finally:
                self.step_box.blockSignals(previous_signal_state)

    def _set_step_box_range_for_step_sc_state(self, step_sc):
        previous_signal_state = self.step_box.blockSignals(True)
        try:
            if step_sc:
                self.step_box.setRange(*self.STEP_SC_STEP_COUNT_RANGE)
            else:
                self.step_box.setRange(*self.LEGACY_STEP_COUNT_RANGE)
        finally:
            self.step_box.blockSignals(previous_signal_state)

    def _step_sc_octave_mode_is_active(self):
        return self._is_step_sc_active() and self.stimulus_info.get("frequency_mode") == "octave"

    def _clear_step_sc_intended_frequency_bounds(self):
        self._step_sc_intended_start_freq = None
        self._step_sc_intended_stop_freq = None
        self._step_sc_last_manual_start_freq = None
        self._step_sc_last_manual_stop_freq = None

    def _set_step_sc_intended_frequency_bounds(self, start_freq, stop_freq):
        self._step_sc_intended_start_freq = float(start_freq)
        self._step_sc_intended_stop_freq = float(stop_freq)

    def _clear_step_sc_last_manual_frequency_bounds(self):
        self._step_sc_last_manual_start_freq = None
        self._step_sc_last_manual_stop_freq = None

    def _set_step_sc_intended_frequency_bounds_from_controls(self):
        self._set_step_sc_intended_frequency_bounds(
            self.start_freq_box.value(),
            self.stop_freq_box.value(),
        )
        self._clear_step_sc_last_manual_frequency_bounds()

    def _seed_step_sc_intended_frequency_bounds_from_controls(self):
        if self._step_sc_octave_mode_is_active():
            self._set_step_sc_intended_frequency_bounds_from_controls()
        else:
            self._clear_step_sc_intended_frequency_bounds()

    def _set_step_sc_intended_frequency_bound(self, data_type, value):
        if data_type == "start_freq":
            self._step_sc_intended_start_freq = float(value)
        elif data_type == "stop_freq":
            self._step_sc_intended_stop_freq = float(value)

    def _set_step_sc_last_manual_frequency_bound(self, data_type, value):
        if data_type == "start_freq":
            self._step_sc_last_manual_start_freq = float(value)
        elif data_type == "stop_freq":
            self._step_sc_last_manual_stop_freq = float(value)

    def _retire_step_sc_manual_frequency_pair_if_complete(self):
        if self._step_sc_last_manual_start_freq is None or self._step_sc_last_manual_stop_freq is None:
            return
        self._clear_step_sc_last_manual_frequency_bounds()

    @staticmethod
    def _step_sc_frequency_direction(start_freq, stop_freq):
        start_freq = float(start_freq)
        stop_freq = float(stop_freq)
        if math.isclose(start_freq, stop_freq, rel_tol=0.0, abs_tol=1e-12):
            return 0
        return 1 if start_freq < stop_freq else -1

    def _begin_step_sc_manual_frequency_edit(self, data_type):
        start_freq = self.stimulus_info.get("effective_start_freq", self.stimulus_info.get("start_freq"))
        stop_freq = self.stimulus_info.get("effective_stop_freq", self.stimulus_info.get("stop_freq"))
        if start_freq is None or stop_freq is None:
            start_freq, stop_freq = self._step_sc_intended_frequency_bounds()
        self._step_sc_active_manual_frequency_edit = data_type
        self._step_sc_active_manual_frequency_previous_direction = self._step_sc_frequency_direction(
            start_freq,
            stop_freq,
        )

    def _clear_step_sc_active_manual_frequency_edit(self):
        self._step_sc_active_manual_frequency_edit = None
        self._step_sc_active_manual_frequency_previous_direction = None

    def _step_sc_manual_frequency_reference_is_active(self, data_type):
        active_edit = self._step_sc_active_manual_frequency_edit
        if active_edit is None or active_edit == data_type:
            return True
        previous_direction = self._step_sc_active_manual_frequency_previous_direction
        if previous_direction is None:
            return False
        current_direction = self._step_sc_frequency_direction(*self._step_sc_intended_frequency_bounds())
        return current_direction != previous_direction

    def _step_sc_intended_frequency_bounds(self):
        if self._step_sc_intended_start_freq is None:
            self._step_sc_intended_start_freq = float(self.start_freq_box.value())
        if self._step_sc_intended_stop_freq is None:
            self._step_sc_intended_stop_freq = float(self.stop_freq_box.value())
        return self._step_sc_intended_start_freq, self._step_sc_intended_stop_freq

    def _step_sc_frequency_value_is_preferred_midpoint(self, box, value):
        preferred = getattr(box, "_preferred_frequencies", [])
        if len(preferred) < 2:
            return False
        value = float(value)
        distances = [abs(candidate - value) for candidate in preferred]
        min_distance = min(distances)
        candidates = [
            candidate
            for candidate, distance in zip(preferred, distances)
            if math.isclose(distance, min_distance, rel_tol=0.0, abs_tol=1e-12)
        ]
        return len(candidates) > 1

    def _step_sc_manual_frequency_bound_is_snap_reference(
        self,
        box,
        raw_value,
        intended_value,
        other_raw_value,
    ):
        if raw_value is None:
            return False
        raw_value = float(raw_value)
        intended_value = float(intended_value)
        if math.isclose(raw_value, intended_value, rel_tol=0.0, abs_tol=1e-12):
            return True
        if other_raw_value is not None and math.isclose(
            raw_value,
            float(other_raw_value),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return False
        return self._step_sc_frequency_value_is_preferred_midpoint(box, raw_value)

    def _step_sc_snap_reference_frequency_bounds(self):
        intended_start, intended_stop = self._step_sc_intended_frequency_bounds()
        start_freq = intended_start
        stop_freq = intended_stop
        if self._step_sc_manual_frequency_reference_is_active(
            "start_freq"
        ) and self._step_sc_manual_frequency_bound_is_snap_reference(
            self.start_freq_box,
            self._step_sc_last_manual_start_freq,
            intended_start,
            self._step_sc_last_manual_stop_freq,
        ):
            start_freq = self._step_sc_last_manual_start_freq
        if self._step_sc_manual_frequency_reference_is_active(
            "stop_freq"
        ) and self._step_sc_manual_frequency_bound_is_snap_reference(
            self.stop_freq_box,
            self._step_sc_last_manual_stop_freq,
            intended_stop,
            self._step_sc_last_manual_start_freq,
        ):
            stop_freq = self._step_sc_last_manual_stop_freq
        return start_freq, stop_freq

    def _current_step_sc_sample_rate(self):
        try:
            return int(self.sample_rate_combo_box.currentText())
        except (TypeError, ValueError):
            return int(self.stimulus_info.get("sample_rate", 44100))

    def _resolution_combo_current_code(self):
        resolution = self.resolution_combo_box.currentData()
        if resolution is None:
            resolution = self.resolution_combo_box.currentText()
        if resolution not in self.STEP_SC_VALID_RESOLUTIONS:
            return self.STEP_SC_DEFAULT_RESOLUTION
        return resolution

    def _set_step_sc_resolution_code(self, resolution):
        index = self.resolution_combo_box.findData(resolution)
        if index < 0:
            index = self.resolution_combo_box.findData(self.STEP_SC_DEFAULT_RESOLUTION)
        if index >= 0:
            self.resolution_combo_box.setCurrentIndex(index)

    def _current_step_sc_resolution(self):
        resolution = self.stimulus_info.get("resolution") or self._resolution_combo_current_code()
        if resolution not in self.STEP_SC_VALID_RESOLUTIONS:
            resolution = self.STEP_SC_DEFAULT_RESOLUTION
        return resolution

    def _restore_legacy_frequency_controls(self):
        previous_start_signal_state = self.start_freq_box.blockSignals(True)
        previous_stop_signal_state = self.stop_freq_box.blockSignals(True)
        try:
            for box in (self.start_freq_box, self.stop_freq_box):
                box.clearPreferredFrequencies()
                box.setDecimals(0)
                box.setSingleStep(1.0)
        finally:
            self.start_freq_box.blockSignals(previous_start_signal_state)
            self.stop_freq_box.blockSignals(previous_stop_signal_state)

    def _refresh_step_sc_frequency_preferences(self):
        octave_mode = self._step_sc_octave_mode_is_active()
        if not octave_mode:
            self._restore_legacy_frequency_controls()
            return

        previous_start_signal_state = self.start_freq_box.blockSignals(True)
        previous_stop_signal_state = self.stop_freq_box.blockSignals(True)
        try:
            for box in (self.start_freq_box, self.stop_freq_box):
                box.setDecimals(1)
            try:
                frequencies = preferred_octave_frequencies(
                    self._current_step_sc_resolution(),
                    sample_rate=self._current_step_sc_sample_rate(),
                )
            except ValueError:
                frequencies = []
            minimum = max(float(self.start_freq_box.minimum()), float(self.stop_freq_box.minimum()))
            maximum = min(float(self.start_freq_box.maximum()), float(self.stop_freq_box.maximum()))
            frequencies = [value for value in frequencies if minimum <= value <= maximum]
            self.start_freq_box.setPreferredFrequencies(frequencies)
            self.stop_freq_box.setPreferredFrequencies(frequencies)
        finally:
            self.start_freq_box.blockSignals(previous_start_signal_state)
            self.stop_freq_box.blockSignals(previous_stop_signal_state)

    def _step_sc_octave_frequency_ties(self, start_freq=None, stop_freq=None):
        if start_freq is None or stop_freq is None:
            start_freq, stop_freq = self._step_sc_intended_frequency_bounds()
        start_freq = float(start_freq)
        stop_freq = float(stop_freq)
        if math.isclose(start_freq, stop_freq, rel_tol=0.0, abs_tol=1e-12):
            return "lower", "lower"
        if start_freq > stop_freq:
            return "upper", "lower"
        return "lower", "upper"

    def _snap_step_sc_octave_frequency_box(self, box, *, tie=None):
        if not self._step_sc_octave_mode_is_active():
            return float(box.value())
        if tie is None:
            start_tie, stop_tie = self._step_sc_octave_frequency_ties()
            tie = stop_tie if box is self.stop_freq_box else start_tie
        snapped = box.nearestPreferredFrequency(box.value(), tie=tie)
        previous_signal_state = box.blockSignals(True)
        try:
            box.setValue(snapped)
        finally:
            box.blockSignals(previous_signal_state)
        if box is self.start_freq_box:
            self._set_step_sc_intended_frequency_bound("start_freq", box.value())
        elif box is self.stop_freq_box:
            self._set_step_sc_intended_frequency_bound("stop_freq", box.value())
        return float(box.value())

    def _snap_step_sc_octave_frequency_controls(self):
        if not self._step_sc_octave_mode_is_active():
            return float(self.start_freq_box.value()), float(self.stop_freq_box.value())

        reference_start, reference_stop = self._step_sc_snap_reference_frequency_bounds()
        start_tie, stop_tie = self._step_sc_octave_frequency_ties(reference_start, reference_stop)
        snapped_start = self.start_freq_box.nearestPreferredFrequency(reference_start, tie=start_tie)
        snapped_stop = self.stop_freq_box.nearestPreferredFrequency(reference_stop, tie=stop_tie)
        previous_start_signal_state = self.start_freq_box.blockSignals(True)
        previous_stop_signal_state = self.stop_freq_box.blockSignals(True)
        try:
            self.start_freq_box.setValue(snapped_start)
            self.stop_freq_box.setValue(snapped_stop)
        finally:
            self.start_freq_box.blockSignals(previous_start_signal_state)
            self.stop_freq_box.blockSignals(previous_stop_signal_state)
        self._set_step_sc_intended_frequency_bounds(
            self.start_freq_box.value(),
            self.stop_freq_box.value(),
        )
        return float(self.start_freq_box.value()), float(self.stop_freq_box.value())

    def _display_frequency_stepped_effective_bounds(self):
        if self.stimulus_info.get("frequency_mode") != "octave":
            return
        for box, effective_key, scalar_key in (
            (self.start_freq_box, "effective_start_freq", "start_freq"),
            (self.stop_freq_box, "effective_stop_freq", "stop_freq"),
        ):
            if effective_key not in self.stimulus_info:
                continue
            previous_signal_state = box.blockSignals(True)
            try:
                box.setValue(float(self.stimulus_info[effective_key]))
            finally:
                box.blockSignals(previous_signal_state)
            visible_value = float(box.value())
            self.stimulus_info[scalar_key] = visible_value
            self.stimulus_info[effective_key] = visible_value

    def _apply_step_sc_control_state(self):
        step_sc = self._is_step_sc_active()
        mode = self.stimulus_info.get("frequency_mode")
        self.step_sc_group_box.setVisible(step_sc)
        self._set_step_box_range_for_step_sc_state(step_sc)
        previous_signal_state = self.total_time_box.blockSignals(True)
        if step_sc:
            self._configure_step_sc_total_time_box()
            octave_mode = mode == "octave"
            self.step_box.setReadOnly(octave_mode)
            self.step_box.setEnabled(not octave_mode)
            self.resolution_combo_box.setEnabled(octave_mode)
        else:
            self._configure_legacy_total_time_box()
            self.step_box.setReadOnly(False)
            self.step_box.setEnabled(True)
            self.resolution_combo_box.setEnabled(True)
        self.total_time_box.blockSignals(previous_signal_state)
        self._refresh_step_sc_frequency_preferences()

    def _set_step_sc_combo_from_mode(self, mode):
        label = self.STEP_SC_MODE_LABELS.get(mode, "倍频程")
        self.stimulus_type_combo_box.setCurrentText(label)

    def _update_step_sc_derived_controls(self):
        self.transition_hz_box.blockSignals(True)
        self.total_time_box.blockSignals(True)
        try:
            if self._is_step_sc_active():
                self._configure_step_sc_total_time_box()
            else:
                self._configure_legacy_total_time_box()
            self.transition_hz_box.setValue(float(self.stimulus_info.get("transition_hz", 0.0) or 0.0))
            self.total_time_box.setValue(float(self.stimulus_info.get("total_time", 0.5) or 0.5))
        finally:
            self.transition_hz_box.blockSignals(False)
            self.total_time_box.blockSignals(False)

    def _clamp_legacy_total_time(self, value):
        minimum, maximum = self.LEGACY_TOTAL_TIME_RANGE
        try:
            total_time = float(value)
        except (TypeError, ValueError):
            total_time = minimum
        if not math.isfinite(total_time):
            total_time = minimum
        return min(max(total_time, minimum), maximum)

    def _sync_legacy_total_time_from_control(self):
        total_time = self._clamp_legacy_total_time(self.total_time_box.value())
        previous_signal_state = self.total_time_box.blockSignals(True)
        try:
            self.total_time_box.setValue(total_time)
        finally:
            self.total_time_box.blockSignals(previous_signal_state)
        self.stimulus_info["total_time"] = total_time

    def _sync_legacy_frequency_bounds_from_controls(self):
        self.stimulus_info["start_freq"] = int(self.start_freq_box.value())
        self.stimulus_info["stop_freq"] = int(self.stop_freq_box.value())

    def _clamp_legacy_step_count(self, value):
        minimum, maximum = self.LEGACY_STEP_COUNT_RANGE
        try:
            step_count = int(value)
        except (TypeError, ValueError):
            step_count = minimum
        return min(max(step_count, minimum), maximum)

    def _restore_legacy_step_controls(self):
        step_count = self._clamp_legacy_step_count(self.stimulus_info.get("num_steps", self.step_box.value()))
        previous_signal_state = self.step_box.blockSignals(True)
        try:
            self.step_box.setRange(*self.LEGACY_STEP_COUNT_RANGE)
            self.step_box.setReadOnly(False)
            self.step_box.setEnabled(True)
            self.step_box.setValue(step_count)
        finally:
            self.step_box.blockSignals(previous_signal_state)
        self.stimulus_info["num_steps"] = int(self.step_box.value())

    def update_stimulus_info_from_stimulus_type_combo_box(self):
        stimulus_type = self.stimulus_type_combo_box.currentText()
        snapshot = self._step_sc_snapshot() if self._is_step_sc_active() else None
        if self._is_step_sc_active():
            self._set_step_sc_frequency_mode(self.STIMULUS_DICT_2.get(stimulus_type))
        else:
            self.stimulus_info["stimulus_type"] = self.STIMULUS_DICT_2.get(stimulus_type)
        if self.stimulus_info.get("use_custom_stimulus", False) and 0 != self.stimulus_type_combo_box.count():
            if self._is_step_sc_active() and self._step_sc_retained_frequency_state == "dirty":
                self._sync_step_sc_frequency_drivers_from_controls()
            generated = self.create_signal_from_stimulus_info()
            if generated is False and snapshot is not None:
                self._restore_step_sc_snapshot(snapshot)
                return
            self.graph_stimulus()

    def _sync_step_sc_frequency_drivers_from_controls(self):
        mode = self.STIMULUS_DICT_2.get(self.stimulus_type_combo_box.currentText())
        if mode not in self.STEP_SC_VALID_MODES:
            mode = self.stimulus_info.get("frequency_mode")
        if mode not in self.STEP_SC_VALID_MODES:
            mode = "octave"
        self.stimulus_info["frequency_mode"] = mode
        self.stimulus_info["stimulus_type"] = mode
        self.stimulus_info["num_steps"] = int(self.step_box.value())
        self.stimulus_info["sample_rate"] = int(self.sample_rate_combo_box.currentText())
        if mode == "octave":
            resolution = self._resolution_combo_current_code()
            if resolution not in self.STEP_SC_VALID_RESOLUTIONS:
                resolution = self.STEP_SC_DEFAULT_RESOLUTION
            self.stimulus_info["resolution"] = resolution
            self._refresh_step_sc_frequency_preferences()
            self._snap_step_sc_octave_frequency_controls()
        else:
            self.stimulus_info["resolution"] = None
            self._clear_step_sc_intended_frequency_bounds()
            self._refresh_step_sc_frequency_preferences()
        self.stimulus_info["start_freq"] = float(self.start_freq_box.value())
        self.stimulus_info["stop_freq"] = float(self.stop_freq_box.value())

    def update_stimulus_info_from_controller(self, controller, data_type: str):
        snapshot = self._step_sc_snapshot() if self._is_step_sc_active() else None
        frequency_driver_edit = self._is_step_sc_active() and data_type in self.STEP_SC_FREQUENCY_DRIVERS
        if frequency_driver_edit:
            self._mark_step_sc_frequency_dirty()

        if data_type == "sample_rate":
            self.stimulus_info[data_type] = int(controller.currentText())
        elif data_type == "resolution":
            self.stimulus_info[data_type] = self._resolution_combo_current_code()
        elif data_type == "total_time":
            self.stimulus_info[data_type] = float(controller.value())
        elif data_type in {"min_duration", "min_cycles"}:
            self.stimulus_info[data_type] = float(controller.value())
        elif data_type == "voltage":
            self.stimulus_info[data_type] = float(controller.value())
            self.update_amplitude()
        elif data_type in {"start_freq", "stop_freq"} and self._step_sc_octave_mode_is_active():
            self._begin_step_sc_manual_frequency_edit(data_type)
            self._set_step_sc_last_manual_frequency_bound(data_type, controller.value())
            self._set_step_sc_intended_frequency_bound(data_type, controller.value())
            self.stimulus_info[data_type] = float(controller.value())
        else:
            self.stimulus_info[data_type] = int(controller.value())
        if frequency_driver_edit:
            try:
                if data_type not in {"start_freq", "stop_freq"} and self._step_sc_octave_mode_is_active():
                    self._set_step_sc_intended_frequency_bounds_from_controls()
                self._sync_step_sc_frequency_drivers_from_controls()
            finally:
                if data_type in {"start_freq", "stop_freq"} and self._step_sc_octave_mode_is_active():
                    self._clear_step_sc_active_manual_frequency_edit()
        if self.stimulus_info.get("use_custom_stimulus", False):
            generated = self.create_signal_from_stimulus_info()
            if generated is False and snapshot is not None:
                self._restore_step_sc_snapshot(snapshot)
                return
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
        for widgets in self.box_checked_enable_dict.values():
            for widget in widgets:
                widget.setDisabled(True)
        self.step_sc_group_box.setVisible(stimulus_method == FREQUENCY_STEPPED_METHOD)
        if enable_status:
            for widget in self.box_checked_enable_dict[stimulus_method]:
                widget.setEnabled(True)
        self._apply_step_sc_control_state()

    def change_custom_chk_box(self, custom_box_checked):
        """
            Updates the enabled/disabled state and style of related widgets based on the custom checkbox state.

            Parameters:
            custom_box_checked (bool): The checked state of the custom checkbox. If True, the checkbox is checked;
        if False, it is unchecked.
        """
        if self._is_step_sc_active() and not custom_box_checked:
            self.custom_chk_box.blockSignals(True)
            self.custom_chk_box.setChecked(True)
            self.custom_chk_box.blockSignals(False)
            custom_box_checked = True
        self.switch_group_box_availability(custom_box_checked)
        for widget in self.box_checked_disable_list:
            widget.setDisabled(custom_box_checked)

        self.stimulus_info["use_custom_stimulus"] = custom_box_checked
        if custom_box_checked:
            self.create_signal_from_stimulus_info()
        else:
            if not self._is_step_sc_active():
                if self.load_stimulus_signal_path:
                    self.load_wav_path = self.load_stimulus_signal_path
                    self.stimulus_data, _ = load_audio_simple(self.load_wav_path, self.stimulus_info["sample_rate"])
                elif self._warn_on_missing_legacy_wav:
                    MessageBox.warning(self, "导入配置", "缺少已加载的外部音频路径，已切换为自定义激励。")
                    self.custom_chk_box.blockSignals(True)
                    self.custom_chk_box.setChecked(True)
                    self.custom_chk_box.blockSignals(False)
                    self.switch_group_box_availability(True)
                    for widget in self.box_checked_disable_list:
                        widget.setDisabled(True)
                    self.stimulus_info["use_custom_stimulus"] = True
                    self.create_signal_from_stimulus_info()
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
        previous_method = normalize_stimulus_method(self.stimulus_info.get("stimulus_method"))
        target_method = self.STIMULUS_DICT[stimulus_method]["name"]
        legacy_switch_from_step_sc = previous_method == FREQUENCY_STEPPED_METHOD and target_method != FREQUENCY_STEPPED_METHOD
        total_time_signal_state = self.total_time_box.blockSignals(True) if legacy_switch_from_step_sc else None
        method_entry_snapshot = (
            self._stimulus_state_snapshot()
            if target_method == FREQUENCY_STEPPED_METHOD and previous_method != FREQUENCY_STEPPED_METHOD
            else None
        )
        self.stimulus_info["stimulus_method"] = target_method
        self.stimulus_type_combo_box.blockSignals(True)
        self.stimulus_type_combo_box.clear()
        # Fetch the corresponding sublist from the dictionary and add it to the stimulus type combo box
        stimulus_item = self.STIMULUS_DICT.get(stimulus_method, {})
        self.stimulus_type_combo_box.addItems(stimulus_item.get("sub_list", []))
        self.stimulus_type_combo_box.blockSignals(False)
        if self.stimulus_info["stimulus_method"] != FREQUENCY_STEPPED_METHOD:
            self.stimulus_info = self._strip_step_sc_rich_metadata(self.stimulus_info)
            self._clear_step_sc_intended_frequency_bounds()
            self._restore_legacy_frequency_controls()
            self._sync_current_legacy_stimulus_type()
            if target_method == "step":
                self._restore_legacy_step_controls()
        # Adjust the state and style of related controls based on the selected stimulus method
        if stimulus_method == "啁啾":
            self.step_group_box.setDisabled(True)
            self.frequency_group_box.setEnabled(True)
            self.step_sc_group_box.setVisible(False)
            self._configure_legacy_total_time_box()
        elif stimulus_method == "噪音":
            self.frequency_group_box.setDisabled(True)
            self.step_group_box.setDisabled(True)
            self.step_sc_group_box.setVisible(False)
            self._configure_legacy_total_time_box()
        elif target_method == FREQUENCY_STEPPED_METHOD:
            self._ensure_step_sc_defaults()
            self._set_step_sc_combo_from_mode(self.stimulus_info["frequency_mode"])
            if previous_method != FREQUENCY_STEPPED_METHOD:
                self._seed_step_sc_intended_frequency_bounds_from_controls()
            self.frequency_group_box.setEnabled(True)
            self.step_group_box.setEnabled(True)
            self.step_sc_group_box.setVisible(True)
            self.custom_chk_box.blockSignals(True)
            self.custom_chk_box.setChecked(True)
            self.custom_chk_box.blockSignals(False)
            self.stimulus_info["use_custom_stimulus"] = True
            if previous_method != FREQUENCY_STEPPED_METHOD and self._step_sc_retained_frequency_state not in {
                "clean",
                "dirty",
            }:
                self._step_sc_retained_frequency_state = "none"
            self._apply_step_sc_control_state()
            self.switch_group_box_availability(True)
            if not self._step_sc_restore_in_progress:
                generated = self.create_signal_from_stimulus_info()
                if generated is False and method_entry_snapshot is not None:
                    self._restore_stimulus_state_snapshot(method_entry_snapshot)
                    return
                if method_entry_snapshot is not None:
                    self._pre_step_sc_legacy_branch_snapshot = method_entry_snapshot
                self.graph_stimulus()
        else:
            self.frequency_group_box.setEnabled(True)
            self.step_group_box.setEnabled(True)
            self.step_sc_group_box.setVisible(False)
            self._configure_legacy_total_time_box()
        if legacy_switch_from_step_sc:
            self._sync_legacy_frequency_bounds_from_controls()
            self._sync_legacy_total_time_from_control()
            self.total_time_box.blockSignals(total_time_signal_state)
            restored_external_branch = self._restore_pre_step_sc_legacy_external_branch()
            if not restored_external_branch and not self.load_stimulus_signal_path:
                self._clear_loaded_legacy_external_paths()
        if self.stimulus_info.get("stimulus_method") != FREQUENCY_STEPPED_METHOD and self.stimulus_info.get(
            "use_custom_stimulus", False
        ):
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
        self._normalize_stimulus_info_method(self.stimulus_info)
        if self.stimulus_info["stimulus_method"] == FREQUENCY_STEPPED_METHOD:
            return self._create_frequency_stepped_signal_from_info()

        create_function_dict = {
            "chirp": StimulusSignal().generate_chirps,
            "step": StimulusSignal().generate_steps,
            "noise": StimulusSignal().generate_noise,
        }
        create_function = create_function_dict.get(self.stimulus_info["stimulus_method"])
        if create_function is None:
            self.stimulus_info = self._unsupported_stimulus_method_fallback(
                self.stimulus_info["stimulus_method"]
            )
            self.stimulus_data, _ = StimulusSignal().generate_chirps(**self.stimulus_info)
            return False
        if self.stimulus_info["stimulus_method"] == "step":
            self._restore_legacy_step_controls()
        self.stimulus_data, _ = create_function(**self.stimulus_info)
        return True

    def _frequency_stepped_generation_kwargs(self):
        self._ensure_step_sc_defaults()
        return self._frequency_stepped_generation_kwargs_from_info(
            self.stimulus_info,
            self._step_sc_retained_frequencies,
            self._step_sc_retained_frequency_state,
        )

    def _create_frequency_stepped_signal_from_info(self):
        old_info = deepcopy(self.stimulus_info)
        old_data = None if self.stimulus_data is None else self.stimulus_data.copy()
        old_state = self._step_sc_retained_frequency_state
        old_frequencies = (
            None if self._step_sc_retained_frequencies is None else list(self._step_sc_retained_frequencies)
        )
        try:
            new_info, data, retained, retained_state = self._generate_frequency_stepped_candidate(
                self.stimulus_info,
                self._step_sc_retained_frequencies,
                self._step_sc_retained_frequency_state,
            )
        except Exception as exc:
            self.stimulus_info = old_info
            self.stimulus_data = old_data
            self._step_sc_retained_frequency_state = old_state
            self._step_sc_retained_frequencies = old_frequencies
            self.default_logger.error(f"Failed to generate step(sc) stimulus. {exc}")
            MessageBox.warning(self, FREQUENCY_STEPPED_LABEL, str(exc))
            return False
        self._commit_frequency_stepped_candidate(new_info, data, retained, retained_state)
        return True

    @staticmethod
    def _filename_value(value):
        if value is None:
            return None
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    @staticmethod
    def _sanitize_filename_component(value):
        safe_chars = []
        for char in value:
            if char.isalnum() or char in {"-", "_", "."}:
                safe_chars.append(char)
            else:
                safe_chars.append("-")
        component = "".join(safe_chars).strip(".-")
        while "--" in component:
            component = component.replace("--", "-")
        return component or "stimulus"

    def _build_step_sc_stimulus_filename(self):
        parts = []
        for key in self.STEP_SC_FILENAME_KEYS:
            value = self._filename_value(self.stimulus_info.get(key))
            if value is None:
                continue
            parts.append(f"{key}-{value}")
        filename = self._sanitize_filename_component("_".join(parts))
        return filename[:160].rstrip(".-_") or FREQUENCY_STEPPED_METHOD

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
        self._normalize_stimulus_info_method(self.stimulus_info)
        if self._is_step_sc_active():
            self.stimulus_info = self._json_safe(self.stimulus_info)
            for key in self.STEP_SC_LEGACY_EXTERNAL_WAV_KEYS:
                self.stimulus_info.pop(key, None)
            load_stimulus_signal_path = None
        else:
            self.stimulus_info = self._strip_step_sc_rich_metadata(self.stimulus_info)
            load_stimulus_signal_path = self.load_stimulus_signal_path

        # Generate the name of the stimulus signal based on the stimulus information
        if self._is_step_sc_active():
            stimulus_name = self._build_step_sc_stimulus_filename()
        else:
            stimulus_name = "_".join(str(value) for value in self.stimulus_info.values())
        stimulus_signal_path = model_consts.STORED_STIMULUS_PATH + "/" + stimulus_name + ".wav"
        save_audio_simple(
            stimulus_signal_path,
            self.stimulus_data.astype("float32"),
            self.stimulus_info["sample_rate"],
        )
        stimulus_signal_path = FileOps.get_relative_path(stimulus_signal_path, DEFAULT_DIR)
        if load_stimulus_signal_path:
            load_stimulus_signal_path = FileOps.get_relative_path(load_stimulus_signal_path, DEFAULT_DIR)

        data = {
            "stimulus_info": self._json_safe(self.stimulus_info),
            "stimulus_signal_path": stimulus_signal_path,
            "load_stimulus_signal_path": load_stimulus_signal_path,
        }

        return data

    @staticmethod
    def _set_visible_step_sc_minimum_value(spin_box, value):
        value = float(value)
        spin_box.setValue(value)
        if value > 0 and float(spin_box.value()) == 0.0:
            spin_box.setValue(10 ** -spin_box.decimals())
        return float(spin_box.value())

    def update_stimulus_ui_value(self, stimulus_info: dict):
        """
            Update the user interface values for stimulus parameters.

            This function updates the values of various UI controls based on the stimulus parameters stored in
        `stimulus_info`.
            Specifically, it updates the stimulus method, stimulus type, start frequency, stop frequency, total time,
        repeat times, number of steps, voltage type, voltage value, and sample rate.
        """
        self._normalize_stimulus_info_method(stimulus_info)
        if stimulus_info.get("stimulus_method") == FREQUENCY_STEPPED_METHOD:
            try:
                stimulus_info, retained, retained_state = self._prepare_frequency_stepped_info(stimulus_info)
            except ValueError as exc:
                MessageBox.warning(self, "导入配置", str(exc))
                return False
            self.stimulus_info = stimulus_info
            self._step_sc_retained_frequencies = retained
            self._step_sc_retained_frequency_state = retained_state
        else:
            stimulus_info = self._strip_step_sc_rich_metadata(stimulus_info)
            self.stimulus_info = stimulus_info

        self.switch_connection_off()
        for k, v in self.STIMULUS_DICT.items():
            if v["name"] == stimulus_info.get("stimulus_method"):
                self.stimulus_method_combo_box.setCurrentText(k)
                self.set_stimulus_type_connection()
                break
        for k, v in self.STIMULUS_DICT_2.items():
            if stimulus_info.get("stimulus_method") == FREQUENCY_STEPPED_METHOD:
                target_type = stimulus_info.get("frequency_mode")
            else:
                target_type = stimulus_info.get("stimulus_type")
            if v == target_type:
                self.stimulus_type_combo_box.setCurrentText(k)
                break
        self._sync_current_legacy_stimulus_type()
        octave_mode = (
            stimulus_info.get("stimulus_method") == FREQUENCY_STEPPED_METHOD
            and stimulus_info.get("frequency_mode") == "octave"
        )
        self.start_freq_box.setDecimals(1 if octave_mode else 0)
        self.stop_freq_box.setDecimals(1 if octave_mode else 0)
        if octave_mode:
            start_freq = float(stimulus_info.get("start_freq", "80"))
            stop_freq = float(stimulus_info.get("stop_freq", "2000"))
            visible_start_freq = float(stimulus_info.get("effective_start_freq", start_freq))
            visible_stop_freq = float(stimulus_info.get("effective_stop_freq", stop_freq))
            self._set_step_sc_intended_frequency_bounds(visible_start_freq, visible_stop_freq)
            self._clear_step_sc_last_manual_frequency_bounds()
            self.start_freq_box.setValue(visible_start_freq)
            self.stop_freq_box.setValue(visible_stop_freq)
            stimulus_info["start_freq"] = float(self.start_freq_box.value())
            stimulus_info["stop_freq"] = float(self.stop_freq_box.value())
            if "effective_start_freq" in stimulus_info:
                stimulus_info["effective_start_freq"] = float(self.start_freq_box.value())
            if "effective_stop_freq" in stimulus_info:
                stimulus_info["effective_stop_freq"] = float(self.stop_freq_box.value())
        else:
            self._clear_step_sc_intended_frequency_bounds()
            self._restore_legacy_frequency_controls()
            self.start_freq_box.setValue(int(float(stimulus_info.get("start_freq", "80"))))
            self.stop_freq_box.setValue(int(float(stimulus_info.get("stop_freq", "2000"))))
        total_time = stimulus_info.get("total_time", "4")
        self.total_time_box.setValue(float(total_time if total_time is not None else "4"))
        self.repeat_box.setValue(int(stimulus_info.get("repeat_times", "1")))
        if stimulus_info.get("stimulus_method") == FREQUENCY_STEPPED_METHOD:
            self.step_box.setRange(*self.STEP_SC_STEP_COUNT_RANGE)
        self.step_box.setValue(int(stimulus_info.get("num_steps", "3")))
        if self._is_step_sc_active():
            visible_min_duration = self._set_visible_step_sc_minimum_value(
                self.min_duration_box,
                stimulus_info.get("min_duration", self.STEP_SC_DEFAULT_MIN_DURATION),
            )
            visible_min_cycles = self._set_visible_step_sc_minimum_value(
                self.min_cycles_box,
                stimulus_info.get("min_cycles", self.STEP_SC_DEFAULT_MIN_CYCLES),
            )
            stimulus_info["min_duration"] = visible_min_duration
            stimulus_info["min_cycles"] = visible_min_cycles
            self.stimulus_info["min_duration"] = visible_min_duration
            self.stimulus_info["min_cycles"] = visible_min_cycles
        else:
            self.min_duration_box.setValue(float(stimulus_info.get("min_duration", self.STEP_SC_DEFAULT_MIN_DURATION)))
            self.min_cycles_box.setValue(float(stimulus_info.get("min_cycles", self.STEP_SC_DEFAULT_MIN_CYCLES)))
        if stimulus_info.get("resolution") in self.STEP_SC_VALID_RESOLUTIONS:
            self._set_step_sc_resolution_code(stimulus_info.get("resolution"))
        self.voltage_combo_box.setCurrentText(stimulus_info.get("voltage_type", "RMS"))
        self.voltage_spin_box.setValue(float(stimulus_info.get("voltage", "2.0")))
        self.sample_rate_combo_box.setCurrentText(str(stimulus_info.get("sample_rate", "44100")))
        if stimulus_info.get("use_custom_stimulus"):
            previous_signal_state = self.custom_chk_box.blockSignals(True)
            self.custom_chk_box.setChecked(True)
            self.custom_chk_box.blockSignals(previous_signal_state)
        else:
            previous_signal_state = self.custom_chk_box.blockSignals(True)
            self.custom_chk_box.setChecked(False)
            self.custom_chk_box.blockSignals(previous_signal_state)
            self.change_custom_chk_box(False)
        self._apply_step_sc_control_state()
        self._update_step_sc_derived_controls()
        self.switch_connection_on()
        return True

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
        self._normalize_stimulus_info_method(loaded_stimulus)
        if not self._is_supported_stimulus_method(loaded_stimulus):
            self.stimulus_info = self._unsupported_stimulus_method_fallback(
                loaded_stimulus.get("stimulus_method")
            )
        elif loaded_stimulus.get("stimulus_method") == FREQUENCY_STEPPED_METHOD:
            try:
                candidate, retained, retained_state = self._prepare_frequency_stepped_info(loaded_stimulus)
                candidate, data, retained, retained_state = self._generate_frequency_stepped_candidate(
                    candidate, retained, retained_state
                )
            except Exception as exc:
                MessageBox.warning(self, "导入配置", str(exc))
                return
            self._clear_imported_step_sc_legacy_external_authority()
            self._commit_frequency_stepped_candidate(candidate, data, retained, retained_state)
        else:
            candidate = self._legacy_stimulus_info_from_payload(loaded_stimulus)
            self.stimulus_info = candidate
            if self._adopt_legacy_external_wav_path_from_payload(loaded_stimulus):
                pass
            elif self._is_missing_legacy_external_path_payload(self.stimulus_info, loaded_stimulus):
                self._clear_loaded_legacy_external_paths()
        previous_warn_on_missing_legacy_wav = self._warn_on_missing_legacy_wav
        self._warn_on_missing_legacy_wav = (
            not self._is_step_sc_active()
            and self.stimulus_info.get("use_custom_stimulus") is False
            and not self.load_stimulus_signal_path
        )
        try:
            if self.update_stimulus_ui_value(self.stimulus_info) is False:
                return
        finally:
            self._warn_on_missing_legacy_wav = previous_warn_on_missing_legacy_wav
        self.sync_voltage_info()
        if self._uses_legacy_external_wav_branch():
            self.switch_group_box_availability(False)
        else:
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
        if self._is_step_sc_active():
            return
        load_stimulus_signal_path = self.load_stimulus_signal_path
        load_wav_path = self.load_wav_path
        self.load_stimulus_signal_path = self.load_wav_path
        self.load_wav_path, _ = QFileDialog.getOpenFileName(
            self, "打开音频", DEFAULT_DIR + "audio_data/stimulus", "WAV Files (*.wav)"
        )
        if self.load_wav_path:
            self.stimulus_data, _ = load_audio_simple(self.load_wav_path, self.stimulus_info["sample_rate"])
            self._legacy_external_wav_loaded_by_user = True
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
            self._normalize_stimulus_info_method(info)
            if not self._is_supported_stimulus_method(info):
                return self._unsupported_stimulus_method_fallback(info.get("stimulus_method"))
            if info.get("stimulus_method") == FREQUENCY_STEPPED_METHOD:
                try:
                    info, _, _ = self._prepare_frequency_stepped_info(info)
                except ValueError as exc:
                    MessageBox.warning(self, "导入配置", str(exc))
                    return {}
            else:
                info = self._strip_step_sc_rich_metadata(info)
            self.load_wav_path = result["stimulus_signal_path"]
            if info.get("stimulus_method") == FREQUENCY_STEPPED_METHOD:
                self.load_stimulus_signal_path = None
                self._legacy_external_wav_loaded_by_user = False
            elif result.get("load_stimulus_signal_path"):
                self.load_stimulus_signal_path = DEFAULT_DIR + result["load_stimulus_signal_path"]
                self._legacy_external_wav_loaded_by_user = True
            elif self._is_missing_legacy_external_path_payload(info, result):
                self._clear_loaded_legacy_external_paths()
            return info
        else:
            return {}

    @staticmethod
    def _fallback_stimulus_info():
        return {
            "stimulus_method": "chirp",
            "stimulus_type": "log",
            "start_freq": 80,
            "stop_freq": 2000,
            "total_time": 1.0,
            "repeat_times": 1,
            "num_steps": 3,
            "sample_rate": 44100,
            "voltage_type": "RMS",
            "voltage": 1.0,
            "amplitude": 1.0,
            "use_custom_stimulus": True,
        }

    def load_stimulus_config_data(self, stimulus_data):
        self.stimulus_info = stimulus_data.get("stimulus_info")
        self._normalize_stimulus_info_method(self.stimulus_info)
        invalid_step_sc_fallback = False
        unsupported_method_fallback = not self._is_supported_stimulus_method(self.stimulus_info)
        if unsupported_method_fallback:
            self.stimulus_info = self._unsupported_stimulus_method_fallback(
                self.stimulus_info.get("stimulus_method")
            )
            self.stimulus_data, _ = StimulusSignal().generate_chirps(**self.stimulus_info)
        elif self.stimulus_info.get("stimulus_method") == FREQUENCY_STEPPED_METHOD:
            try:
                candidate, retained, retained_state = self._prepare_frequency_stepped_info(self.stimulus_info)
                self.stimulus_info, self.stimulus_data, retained, retained_state = (
                    self._generate_frequency_stepped_candidate(candidate, retained, retained_state)
                )
            except Exception as exc:
                MessageBox.warning(self, "导入配置", str(exc))
                self.stimulus_info = self._fallback_stimulus_info()
                self.stimulus_data, _ = StimulusSignal().generate_chirps(**self.stimulus_info)
                retained = None
                retained_state = "none"
                invalid_step_sc_fallback = True
            self._step_sc_retained_frequencies = retained
            self._step_sc_retained_frequency_state = retained_state
        else:
            self.stimulus_info = self._strip_step_sc_rich_metadata(self.stimulus_info)
        if unsupported_method_fallback or invalid_step_sc_fallback:
            self.load_wav_path = ""
            self.load_stimulus_signal_path = None
            self._legacy_external_wav_loaded_by_user = False
        elif self.stimulus_info.get("stimulus_method") == FREQUENCY_STEPPED_METHOD:
            self.load_wav_path = stimulus_data.get("stimulus_signal_path")
            self.load_stimulus_signal_path = None
            self._legacy_external_wav_loaded_by_user = False
        elif self._is_missing_legacy_external_path_payload(self.stimulus_info, stimulus_data):
            self._clear_loaded_legacy_external_paths()
        else:
            self.load_wav_path = stimulus_data.get("stimulus_signal_path")
        if self._uses_legacy_external_wav_branch() and not self.load_wav_path:
            MessageBox.warning(self, "导入配置", "缺少已加载的外部音频路径，已切换为自定义激励。")
            self.stimulus_info["use_custom_stimulus"] = True
            self.create_signal_from_stimulus_info()
        if self.stimulus_data is None:
            self.stimulus_data, _ = load_audio_simple(self.load_wav_path, self.stimulus_info["sample_rate"])
        if (
            not unsupported_method_fallback
            and not invalid_step_sc_fallback
            and self.stimulus_info.get("stimulus_method") != FREQUENCY_STEPPED_METHOD
            and stimulus_data.get("load_stimulus_signal_path")
        ):
            self.load_stimulus_signal_path = DEFAULT_DIR + stimulus_data.get("load_stimulus_signal_path")
            self._legacy_external_wav_loaded_by_user = True

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
        snapshot = self._stimulus_state_snapshot()
        candidate = self.get_stimulus_info_from_json(True)
        if not candidate:
            self._restore_stimulus_state_snapshot(snapshot)
            return
        self._normalize_stimulus_info_method(candidate)
        if not self._is_supported_stimulus_method(candidate):
            self.stimulus_info = self._unsupported_stimulus_method_fallback(candidate.get("stimulus_method"))
        elif candidate.get("stimulus_method") == FREQUENCY_STEPPED_METHOD:
            try:
                candidate, retained, retained_state = self._prepare_frequency_stepped_info(candidate)
                candidate, data, retained, retained_state = self._generate_frequency_stepped_candidate(
                    candidate, retained, retained_state
                )
            except Exception as exc:
                MessageBox.warning(self, "导入配置", str(exc))
                self._restore_stimulus_state_snapshot(snapshot)
                return
            self._clear_imported_step_sc_legacy_external_authority()
            self._commit_frequency_stepped_candidate(candidate, data, retained, retained_state)
        else:
            self.stimulus_info = self._strip_step_sc_rich_metadata(candidate)
        previous_warn_on_missing_legacy_wav = self._warn_on_missing_legacy_wav
        self._warn_on_missing_legacy_wav = (
            not self._is_step_sc_active()
            and self.stimulus_info.get("use_custom_stimulus") is False
            and not self.load_stimulus_signal_path
        )
        try:
            self.update_stimulus_ui_value(self.stimulus_info)
        finally:
            self._warn_on_missing_legacy_wav = previous_warn_on_missing_legacy_wav
        self.sync_voltage_info()
        if self._uses_legacy_external_wav_branch():
            self.switch_group_box_availability(False)
        else:
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
        if self._is_step_sc_active():
            self.stimulus_info["use_custom_stimulus"] = True
            self.custom_chk_box.blockSignals(True)
            self.custom_chk_box.setChecked(True)
            self.custom_chk_box.blockSignals(False)
        if not self.custom_chk_box.isChecked():
            if not self._has_authoritative_legacy_external_wav():
                self.miss_popup()
                return
            if self.start_custom_check_status:
                self.set_ai_popup()
            # elif self.load_wav_path != self.load_stimulus_signal_path:
            elif not self._paths_reference_same_file(self.load_wav_path, self.load_stimulus_signal_path):
                self.set_ai_popup()
            self.load_stimulus_signal_path = self.load_wav_path
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

    def _uses_legacy_external_wav_branch(self):
        return not self._is_step_sc_active() and self.stimulus_info.get("use_custom_stimulus") is False

    @staticmethod
    def _paths_reference_same_file(left_path, right_path):
        if not left_path or not right_path:
            return False
        if os.path.exists(left_path) and os.path.exists(right_path):
            return os.path.samefile(left_path, right_path)
        return os.path.normcase(os.path.abspath(left_path)) == os.path.normcase(os.path.abspath(right_path))

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
