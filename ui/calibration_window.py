from re import S
import sys
import threading
from datetime import datetime

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QDialog, QGridLayout, QHBoxLayout
from PyQt5.QtWidgets import QVBoxLayout, QSpacerItem, QSizePolicy, QWidget

from base.log_manager import LogManager
from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis
from base.pre_processing.swept_sine_chirps import StimulusSignal
from base.play_and_record import stream_record_without_play
from base.soundcard_audio_processor import SoundcardAudioProcessor
from base.soundcard_calibration_manager import SoundcardCalibrationManager
from consts import error_code, ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.widgets import (
    PushButton,
    RadioButton,
    SpinBox,
    TabWidget,
    Label,
    LineEdit,
    DoubleSpinBox,
    GroupBox,
    MessageBox,
)
from ui.ui_src import ui_resources


class CalibrationWindow(QDialog):

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """
        Initialize the user interface for the calibration window.
        This function sets up the window icon, title, size, and layout,
        and creates tabs for output and input calibration.
        """
        self.setObjectName("CalibrationWindow")
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))
        self.setWindowTitle("校准窗口")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setMinimumSize(500, 550)
        self.setMaximumSize(600, 580)
        cal_wnd_layout = QVBoxLayout()

        self.input_calibration_flag = False

        self.tabwidget = TabWidget()
        self.output_cal_wnd = OutputCalibration()
        self.input_cal_wnd = InputCalibration()
        self.tabwidget.addTab(self.output_cal_wnd, "输出校准")
        self.tabwidget.addTab(self.input_cal_wnd, "输入校准")

        btn_layout = self.create_btn_box()

        cal_wnd_layout.addWidget(self.tabwidget)
        cal_wnd_layout.addLayout(btn_layout)
        self.setLayout(cal_wnd_layout)

    def create_btn_box(self):
        """
        Create a button box

        This method creates a horizontal layout containing calibration, reset, and cancel buttons.
        Spacers are used to adjust the spacing between the buttons in the layout.
        """
        btn_layout = QHBoxLayout()
        self.cal_btn = PushButton(" 校  准 ")
        self.cal_btn.clicked.connect(self.clicked_calibration_button)
        reset_btn = PushButton(" 重  置 ")
        reset_btn.clicked.connect(self.clicked_reset_button)
        cancel_btn = PushButton(" 退  出 ")
        cancel_btn.clicked.connect(self.clicked_close_button)
        btn_layout.addWidget(self.cal_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(reset_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        return btn_layout

    def clicked_calibration_button(self):
        """
            Handles the event when the calibration button is clicked.

            This function first determines which tab is currently active, and then performs the corresponding
        calibration operation based on the active tab.
            If the first tab is active, it calls the calibration method of the output_cal_wnd object.
            If the second tab is active, it first disables the calibration button to prevent multiple calibrations,
         then calls the clicked_calibration method of the input_cal_wnd object.
        """
        current_tab_index = self.tabwidget.currentIndex()
        if current_tab_index == 0:
            self.output_cal_wnd.calibration()
        elif current_tab_index == 1:
            self.cal_btn.setDisabled(True)
            self.input_cal_wnd.clicked_calibration()
            self.input_calibration_flag = True

    def clicked_reset_button(self):
        """
            Handles the event when the reset button is clicked.

            This function first determines which tab is currently active, and then performs the reset operation
        according to the index of the active tab.
            If the first tab is active, it calls the reset_btn_clicked method of the output_cal_wnd object to perform
        the reset operation.
            If the second tab is active, it calls the reset_btn_clicked method of the input_cal_wnd object to perform
        the reset operation,
            and enables the cal_btn button at the same time to allow the user to perform calculation operations again.
        """
        current_tab_index = self.tabwidget.currentIndex()
        if current_tab_index == 0:
            self.output_cal_wnd.reset_btn_clicked()
        elif current_tab_index == 1:
            self.input_cal_wnd.reset_btn_clicked()
            self.cal_btn.setDisabled(False)
            self.input_calibration_flag = False

    def clicked_close_button(self):
        """
            Handles the event when the close button is clicked.

            This function checks which tab is currently active in the tab widget, and then performs the corresponding
        operation to close the window.
            If the first tab is active, it directly closes the window; if the second tab is active, it first stops the
        timer, and then closes the window.
        """
        current_tab_index = self.tabwidget.currentIndex()
        if current_tab_index == 0:
            self.close()
        elif current_tab_index == 1:
            self.input_cal_wnd.stop_timer = True
            self.close()


class OutputCalibration(QWidget):
    def __init__(self):
        super().__init__()
        self.default_calibration_nums = 1
        self.output_voltage_value = []
        self.default_logger = LogManager.set_log_handler("core")
        self.calibration_param = {"calibration_nums": self.default_calibration_nums}
        self.current_count = 1
        self.countdown = 10
        self.play_flag = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_countdown)
        self.init_ui()
        self.get_calibration_param()

    def init_ui(self):
        """
        Initialize the CalibrationWindow class instance.

        This function initializes various attributes of the class, including the output voltage value list,
        default logger, calibration parameters, current count, countdown timer, play flag, and a QTimer.
        It also connects the QTimer's timeout signal to the update_countdown method and calls methods
        to initialize the user interface and retrieve calibration parameters.
        """
        self.setMinimumSize(305, 373)
        self.setMaximumSize(520, 500)

        calibration_param_box = self.create_calibration_param_box()
        output_box = self.create_output_voltage_box()
        test_box = self.create_test_box()
        self.target_voltage_box.resize(135, 29)

        self.calibration_nums_box.resize(90, 30)
        self.output_voltage_box.resize(135, 29)
        self.target_voltage_box.resize(135, 29)

        layout = QVBoxLayout()
        layout.addWidget(calibration_param_box)
        layout.addStretch()
        layout.addWidget(output_box)
        layout.addStretch()
        layout.addWidget(test_box)
        layout.addStretch()
        layout.setContentsMargins(12, 20, 12, 25)
        self.setLayout(layout)

    def create_calibration_param_box(self):
        """
            Create calibration parameter box

            This method creates a GroupBox containing calibration parameter settings, allowing the user to set the
        number of calibrations.
            It includes a label, a spin box, and layout management.
            Return:
                the created calibration parameter box
        """
        calibration_param_box = GroupBox("校准参数")
        calibration_nums_label = Label("校准次数")
        self.calibration_nums_box = SpinBox()
        self.calibration_nums_box.setSuffix(" 次")
        self.calibration_nums_box.setRange(1, 20)
        self.calibration_nums_box.setValue(1)
        self.calibration_nums_box.editingFinished.connect(self.get_calibration_param)
        param_layout = QHBoxLayout()
        param_layout.addWidget(calibration_nums_label)
        param_layout.addStretch()
        param_layout.addWidget(self.calibration_nums_box)
        param_layout.setContentsMargins(9, 10, 10, 10)
        calibration_param_box.setLayout(param_layout)
        return calibration_param_box

    def create_output_voltage_box(self):
        """
        Create the output voltage settings box.

        This function is responsible for generating a group box containing controls related to output voltage settings.
        It includes a play button, countdown display, output voltage adjustment spin box, and save button.
        Return:
            the created output voltage group box
        """
        output_box = GroupBox("输出电压")
        output_layout = QGridLayout()
        self.play_label = Label(f"第 {self.current_count} 次 ")
        self.play_btn = PushButton(" 播  放 ")
        self.play_btn.clicked.connect(self.play_btn_clicked)
        self.countdown_label = Label(
            f"<span style='color: black;'>倒计时：</span>"
            f"<span style='color: red;'>{self.countdown} </span>"
            f"<span style='color: black;'>s</span>"
        )

        output_layout.addWidget(self.play_label, 0, 0)
        output_layout.setColumnStretch(1, 1)
        output_layout.addWidget(self.countdown_label, 0, 2)
        output_layout.setColumnStretch(3, 1)
        output_layout.addWidget(self.play_btn, 0, 4)

        output_voltage_label = Label("输出电压")
        self.output_voltage_box = DoubleSpinBox()
        self.output_voltage_box.setSuffix(" V")
        self.output_voltage_box.setRange(0, 100)
        self.output_voltage_box.setSingleStep(0.1)
        self.save_btn = PushButton(" 保  存 ")
        self.save_btn.clicked.connect(self.save_btn_clicked)
        self.save_btn.setDisabled(True)
        output_layout.addWidget(output_voltage_label, 1, 0)
        output_layout.addWidget(self.output_voltage_box, 1, 2)
        output_layout.addWidget(self.save_btn, 1, 4)

        output_layout.setVerticalSpacing(10)
        output_layout.setAlignment(Qt.AlignCenter)
        output_box.setLayout(output_layout)
        return output_box

    def create_test_box(self):
        """
            Create a test GroupBox with the necessary widgets and layout.

            This function constructs a GroupBox intended for testing purposes. It includes a label for the target
        voltage, a double spin box to set the target voltage, and a test button to initiate the calibration test. The
        layout is managed using a horizontal box layout, with spacers added to maintain appropriate spacing between the
        widgets.
            Returns:
                GroupBox: A GroupBox containing the test widgets and layout.
        """
        test_box = GroupBox("测    试")
        target_V_label = Label("目标电压")
        self.target_voltage_box = DoubleSpinBox()
        self.target_voltage_box.setMinimumWidth(134)
        self.target_voltage_box.setSuffix(" V")
        test_btn = PushButton(" 测  试 ")
        test_btn.clicked.connect(self.test_calibration)
        test_layout = QHBoxLayout()
        test_layout.addWidget(target_V_label)
        test_layout.addStretch()
        test_layout.addWidget(self.target_voltage_box)
        test_layout.addStretch()
        test_layout.addWidget(test_btn)
        test_box.setLayout(test_layout)
        return test_box

    def save_btn_clicked(self):
        """
        This function is triggered when the save button is clicked.

        It performs the following actions:
        1. Appends the current output voltage value to the output voltage value list.
        2. Resets the value of the output voltage input box to 0.
        3. Updates the current count.
        4. Updates the text of the play label and countdown label to reflect the current count and countdown.
        """
        self.output_voltage_value.append(self.output_voltage_box.value())
        self.output_voltage_box.setValue(0)
        self.create_current_count()
        self.play_label.setText(f"第 {self.current_count} 次 ")
        self.countdown_label.setText(
            f"<span style='color: black;'>倒计时：</span>"
            f"<span style='color: red;'>{self.countdown} </span>"
            f"<span style='color: black;'>s</span>"
        )

    def update_countdown(self):
        """
            Update the countdown status.

            If the countdown is greater than 0, decrement the countdown value and update the countdown display on the
        interface.
            If the countdown ends, stop the timer, update the button states to prepare for saving, and reset the
        countdown to 10 seconds.
        """
        if self.countdown > 0:
            self.countdown -= 1
            self.countdown_label.setText(
                f"<span style='color: black;'>倒计时：</span>"
                f"<span style='color: red;'>{self.countdown} </span>"
                f"<span style='color: black;'>s</span>"
            )
        else:
            self.timer.stop()
            self.play_btn.setText(" 停  止 ")
            self.play_btn.setEnabled(False)
            self.save_btn.setEnabled(True)
            self.play_flag = False
            self.countdown = 10

    def play_btn_clicked(self):
        """
        Handle the play button click event.

        This function controls the audio playback and stop based on the current playback state,
        and updates the countdown display during playback.
        """
        stimulus_dict = self.create_signal()
        sap = SoundcardAudioProcessor()
        if not self.play_flag:
            self.play_flag = True
            self.play_btn.setDisabled(True)
            self.save_btn.setDisabled(True)
            self.countdown_label.setText(
                f"<span style='color: black;'>倒计时：</span>"
                f"<span style='color: red;'>{self.countdown} </span>"
                f"<span style='color: black;'>s</span>"
            )
            self.timer.start(1000)
            threading.Thread(target=sap.sd_play, args=(stimulus_dict,)).start()
        else:
            self.play_flag = False
            self.timer.stop()
            self.play_btn.setText(" 播  放 ")
            if self.current_count >= self.calibration_param["calibration_nums"]:
                self.save_btn.setDisabled(True)

    def create_current_count(self):
        """
        Update the UI state based on the current count.

        This method checks if the current count has reached the calibration number.
        If it has, it disables the play and save buttons and updates the save button text to " 完  成 ".
        If the current count has not reached the calibration number, it increments the current count,
        updates the play button text to " 播  放 ", enables the play button, and disables the save button.
        """
        if self.current_count >= self.calibration_param["calibration_nums"]:
            self.save_btn.setText(" 完  成 ")
            self.save_btn.setDisabled(True)
            self.play_btn.setDisabled(True)
        else:
            self.current_count += 1
            self.play_btn.setText(" 播  放 ")
            self.play_btn.setEnabled(True)
            self.save_btn.setDisabled(True)

    def get_calibration_param(self):
        """
        Retrieve calibration parameters.

        This method fetches the number of calibrations from the calibration numbers box
        and combines it with the output voltage value to generate a dictionary of calibration parameters.
        The calibration parameters include the number of calibrations, output voltage, and a list of amplitudes.

        Returns:
            None. Updates the instance variable self.calibration_param with a dictionary containing the calibration parameters.
        """
        calibration_nums = self.calibration_nums_box.value()
        output_voltage = self.output_voltage_value
        if calibration_nums == 1:
            amplitude_list = np.array([0.95])
        else:
            amplitude_list = np.linspace(0.05, 0.95, calibration_nums)
        self.calibration_param = {
            "calibration_nums": calibration_nums,
            "output_voltage": output_voltage,
            "amplitude_list": amplitude_list,
        }

    def create_signal(self):
        data, sr = StimulusSignal().generate_chirps(
            start_freq=800, stop_freq=800, total_time=10, sample_rate=44100, stimulus_type="linear"
        )
        stimulus_dict = {
            "data": data,
            "sr": sr,
            "amplitude": self.calibration_param["amplitude_list"][self.current_count - 1],
        }
        return stimulus_dict

    def test_calibration(self):
        """
            Create a stimulus signal.

            This function generates a stimulus signal by calling the generate_chirps method of the StimulusSignal class
        with specific parameters.
            It then encapsulates the generated signal data and sampling rate into a dictionary, along with the current
        stimulus amplitude for further processing or storage.

            Parameters:
            - self: The instance of the class itself.

            Returns:
            - stimulus_dict: A dictionary containing the generated stimulus signal data, sampling rate, and current
        stimulus amplitude.
        """
        target_voltage = self.target_voltage_box.value()
        scm = SoundcardCalibrationManager()
        calibrate_code, calibrate_result = scm.calibrate_amplitude(target_voltage)
        if calibrate_code != error_code.OK:
            self.default_logger.error(f"Failed to calculate the amplitude. {calibrate_result}")
        amplitude, max_voltage = calibrate_result
        if target_voltage > max_voltage:
            if self.test_calibration_popup():
                return
        data, sr = StimulusSignal().generate_chirps(
            start_freq=800, stop_freq=800, total_time=10, sample_rate=44100, stimulus_type="linear"
        )
        test_stimulus_dict = {"data": data, "sr": sr, "amplitude": amplitude}
        sap = SoundcardAudioProcessor()
        speaker_code, msg = sap.sd_play(test_stimulus_dict)
        if speaker_code != error_code.OK:
            self.default_logger.error(f"Failed to play the audio. {msg}")

    def test_calibration_popup(self):
        """
        Display a calibration test popup message.

        This function creates a MessageBox instance, sets its icon, text, title, and buttons,
        then displays the message box and waits for user interaction.
        It returns whether the user clicked the OK button.

        Returns:
            bool: True if the user clicked OK, False otherwise.
        """
        test_msg = MessageBox(self)
        test_msg.setIcon(MessageBox.Warning)
        test_msg.setText("目标电压过大，请重新输入!")
        test_msg.setWindowTitle("测试失败")
        test_msg.setStandardButtons(MessageBox.Ok)
        button = test_msg.exec_()
        return button == MessageBox.Ok

    def reset_btn_clicked(self):
        """
        Handles the reset button click event.

        This function resets various settings and displays in the user interface to their default or initial states.
        It clears output voltage values, resets voltage settings, restores calibration counts, stops the timer, etc.
        """
        self.output_voltage_value.clear()
        self.output_voltage_box.setValue(0)
        self.target_voltage_box.setValue(0)
        self.calibration_nums_box.setValue(self.default_calibration_nums)
        self.timer.stop()
        self.current_count = 1
        self.countdown = 10
        self.play_label.setText(f"第 {self.current_count} 次 ")
        self.countdown_label.setText(
            f"<span style='color: black;'>倒计时：</span>"
            f"<span style='color: red;'>{self.countdown} </span>"
            f"<span style='color: black;'>s</span>"
        )
        self.save_btn.setText(" 保  存 ")
        self.play_btn.setText(" 播  放 ")
        self.play_flag = False
        self.play_btn.setEnabled(True)
        self.save_btn.setDisabled(True)

    def calibration(self):
        """
            Performs the soundcard calibration process.

            This method first checks if the number of saved voltage values meets the required calibration times.
            If not, it logs an error and prompts the user through a popup window.
            If the conditions are met, it adds the amplitude and voltage data to the SoundcardCalibrationManager for
        calibration.
            After calibration, it handles the results based on the fit_code: logs and prompts success or failure.
        """
        scm = SoundcardCalibrationManager()
        if len(self.output_voltage_value) != self.calibration_param["calibration_nums"]:
            self.calibration_popup(success_flag=False)
            self.default_logger.error("The saved voltage does not meet the requirement of calibration times.")
        else:
            if self.calibration_param["calibration_nums"] == 1:
                scm.add_data(0, 0, validation=False)
            for amplitude, voltage in zip(self.calibration_param["amplitude_list"], self.output_voltage_value):
                scm.add_data(amplitude, voltage)
            fit_code, msg = scm.fit()
            if fit_code == error_code.OK:
                self.default_logger.info("Calibration success.")
                self.calibration_popup(success_flag=True)
            else:
                self.default_logger.error(f"Failed to calibrate. {msg}")
                self.calibration_popup(success_flag=False)

    def calibration_popup(self, success_flag=True):
        """
        Displays a calibration result popup.

        Depending on whether the calibration was successful, it shows different icons and message texts.
        If the calibration is successful, it displays an information icon and a success message;
        if the calibration fails, it displays a critical icon and a failure message.

        Parameters:
        - success_flag: Boolean indicating whether the calibration was successful. Default value is True.
        """
        cal_msg = MessageBox(self)
        if success_flag:
            cal_msg.setIcon(MessageBox.Information)
            cal_msg.setText("校准成功")
            cal_msg.setWindowTitle("校准成功")
        else:
            cal_msg.setIcon(MessageBox.Critical)
            cal_msg.setText("校准失败，请重试")
            cal_msg.setWindowTitle("校准失败")
        cal_msg.setStandardButtons(MessageBox.Ok)
        cal_msg.exec_()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            event.ignore()
        else:
            super().keyPressEvent(event)


class InputCalibration(QWidget):

    def __init__(self):
        super().__init__()
        self.default_logger = LogManager.set_log_handler("core")  # Configures and retrieves the logger
        self.stop_timer = False  # Initializes the stop timer flag to False
        self.update_ui_timer = QTimer()
        self.update_ui_timer.setInterval(1000)
        self.update_ui_timer.timeout.connect(self.update_recorded_time)

        # Streaming recording state (no waveform display needed)
        self.streaming_processor = None
        self.streaming_poll_timer = QTimer(self)
        self.streaming_poll_timer.timeout.connect(self._poll_streaming_queue)

        self.init_ui()

    def init_ui(self):
        """
        Initializes the user interface.

        This method sets up the window title, window properties, and size constraints.
        It also initializes the UI layout and applies custom stylesheets to various widgets.
        """
        self.setMinimumSize(305, 373)
        self.setMaximumSize(520, 500)
        self.standard_spl_flag = True
        self.recorded_flag = False

        min_height = ui_style_const.scale_size_px(70)
        standard_spl_box = self.create_standard_spl_box()
        standard_spl_box.setMinimumHeight(min_height)
        recorded_box = self.create_recorded_box()
        recorded_box.setMinimumHeight(min_height)
        v2pa_factor_box = self.create_v2pa_factor_box()
        v2pa_factor_box.setMinimumHeight(min_height)

        v_spacer_1 = QSpacerItem(30, 30, QSizePolicy.Minimum, QSizePolicy.Expanding)
        v_spacer_2 = QSpacerItem(30, 30, QSizePolicy.Minimum, QSizePolicy.Expanding)
        v_spacer_3 = QSpacerItem(30, 30, QSizePolicy.Minimum, QSizePolicy.Expanding)

        layout = QVBoxLayout()
        layout.addWidget(standard_spl_box)
        layout.addItem(v_spacer_1)
        layout.addWidget(recorded_box)
        layout.addItem(v_spacer_2)
        layout.addWidget(v2pa_factor_box)
        layout.addItem(v_spacer_3)
        layout.setContentsMargins(12, 20, 12, 25)

        self.setLayout(layout)

    def create_v2pa_factor_box(self):
        """
        Create a GroupBox to display the sound pressure v2pa_factor.

        This method creates a GroupBox containing a label and a read-only line edit
        to show the sound pressure v2pa_factor from the calibration results. The layout
        uses a horizontal box layout to arrange the elements horizontally.

        Returns:
            GroupBox: A GroupBox containing the sound pressure v2pa_factor label and line edit.
        """
        v2pa_factor_box = GroupBox("校准结果")
        v2pa_factor_label = Label("校准系数（V/Pa）：")
        self.v2pa_factor_lineedit = LineEdit()
        self.v2pa_factor_lineedit.setReadOnly(True)

        standard_v2pa_factor_layout = QHBoxLayout()
        h_spacer_v2pa_factor_center = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        standard_v2pa_factor_layout.addWidget(v2pa_factor_label)
        standard_v2pa_factor_layout.addItem(h_spacer_v2pa_factor_center)
        standard_v2pa_factor_layout.addWidget(self.v2pa_factor_lineedit)
        v2pa_factor_box.setLayout(standard_v2pa_factor_layout)

        return v2pa_factor_box

    def create_recorded_box(self):
        """
        Create a GroupBox to display recorded audio information.

        Returns:
            GroupBox: A GroupBox containing the recorded time information.
        """
        recorded_box = GroupBox("录制音频")
        recorded_label = Label("录制时间：")
        self.recorded_label = Label()
        self.recorded_label.setMinimumWidth(70)
        self.recorded_label.resize(70, 30)
        self.recorded_label.setAlignment(Qt.AlignCenter)
        self.recorded_time = 10
        self.recorded_label.setText(
            f"<span style='color: red;'>{self.recorded_time} </span>" f"<span style='color: black;'>s</span>"
        )

        recorded_layout = QHBoxLayout()
        h_spacer_v2pa_factor_center = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        recorded_layout.addWidget(recorded_label)
        recorded_layout.addItem(h_spacer_v2pa_factor_center)
        recorded_layout.addWidget(self.recorded_label)
        recorded_box.setLayout(recorded_layout)

        return recorded_box

    def create_standard_spl_box(self):
        """
        Create a group box containing standard sound pressure options.

        This method generates a GroupBox widget that includes two RadioButton options,
        representing 94 dB and 114 dB standard sound pressure levels. When a different sound
        pressure level is selected, the set_standard_spl method is triggered to handle the logic.

        Returns:
            GroupBox: Group box containing standard sound pressure options.
        """
        standard_spl_box = GroupBox("标准声压")

        self.standard_spl_i = RadioButton("94  dB")
        self.standard_spl_ii = RadioButton("114 dB")
        self.standard_spl_i.clicked.connect(self.set_standard_spl)
        self.standard_spl_ii.clicked.connect(self.set_standard_spl)
        self.standard_spl_i.setChecked(True)

        h_spacer_standard_center = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        standard_spl_layout = QHBoxLayout()
        standard_spl_layout.addWidget(self.standard_spl_i)
        standard_spl_layout.addItem(h_spacer_standard_center)
        standard_spl_layout.addWidget(self.standard_spl_ii)
        standard_spl_layout.setContentsMargins(30, 0, 30, 0)
        standard_spl_box.setLayout(standard_spl_layout)

        return standard_spl_box

    def set_standard_spl(self):
        """
        Sets the value of standard_spl_flag based on the selected SPL standard.

        If self.standard_spl_i is checked, sets self.standard_spl_flag to True.
        If self.standard_spl_ii is checked, sets self.standard_spl_flag to False.
        """
        if self.standard_spl_i.isChecked():
            self.standard_spl_flag = True
        elif self.standard_spl_ii.isChecked():
            self.standard_spl_flag = False

    def clicked_calibration(self):
        """
        Execute the calibration process upon clicking the calibration button using streaming approach.

        This function initializes the recording parameters and starts streaming recording in a non-blocking way,
        allowing the UI timer to update the countdown in real-time. No waveform display is needed.
        """

        # Reset state
        self.recorded_time = 10

        prolong = 1
        recorded_dict = {
            "channels": 1,
            "sample_rate": 44100,
            "num_frames": 10 * 44100,
            "prolong_frames": int(prolong * 44100),
            "device": None,  # Use default input device
        }

        # Start UI countdown timer
        self.update_ui_timer.start()

        # Start streaming recording (non-blocking, no signal connection needed for waveform)
        self.streaming_processor, _ = stream_record_without_play(
            recorded_dict, None, None  # file_path - we don't need to save file  # signal_info
        )

        # Start polling timer to check completion
        self.streaming_poll_timer.start(50)  # Poll every 50ms

    def _poll_streaming_queue(self):
        """
        Poll streaming queue and check for completion.

        Called by QTimer every 50ms from Qt main thread.
        Process audio chunks WITHOUT emitting signals (no waveform display needed).
        """
        if self.streaming_processor is None:
            return

        # Process all available chunks from queue (non-blocking)
        try:
            while True:
                chunk = self.streaming_processor.audio_queue.get_nowait()
                self.streaming_processor.accumulated_chunks.append(chunk)
        except Exception:
            pass

        # Check if recording is complete
        if not self.streaming_processor.is_recording:
            self.streaming_poll_timer.stop()
            self._on_streaming_complete()

    def _on_streaming_complete(self):
        """
        Handle streaming recording completion and calculate calibration result.
        """
        try:
            # Stop UI timer
            self.update_ui_timer.stop()

            # Get complete recorded data from processor
            recorded_data = self.streaming_processor.get_recorded_data()

            # Calculate average SPL from recorded data
            self.average_value = self._calculate_spl_from_data(recorded_data)

            # Calculate v2pa_factor
            v2pa_factor = self.calculate_v2pa_factor(self.average_value)
            self.v2pa_factor_lineedit.setText(str(np.round(v2pa_factor, decimals=3)))

            # Clean up
            self.streaming_processor = None

            # Show result
            if not self.stop_timer:
                if str(v2pa_factor) == "inf":
                    self.calibration_popup(success_flag=False)
                else:
                    self.calibration_popup(success_flag=True)
                    self.default_logger.info("Calibration success.")
                    self.save_v2pa_factor_to_text(v2pa_factor)

        except Exception as e:
            self.default_logger.error(f"Error in streaming calibration completion: {e}")
            self.update_ui_timer.stop()
            self.streaming_processor = None
            self.calibration_popup(success_flag=False)

    def _calculate_spl_from_data(self, recorded_data):
        """
        Calculate average SPL from recorded data (extracted from calculate_average_spl).

        Args:
            recorded_data (np.ndarray): Recorded audio data

        Returns:
            float: Average SPL value
        """
        step = len(recorded_data) // 3
        spl_smooth = AudioThdFrequencyResponseAnalysis().spl_calculation(recorded_data, method="rms", window_size=1201)
        spl_smooth_mid = len(spl_smooth) // 2
        spl_smooth_start = spl_smooth_mid - step
        spl_smooth_end = spl_smooth_mid + step
        spl_sample = spl_smooth[spl_smooth_start:spl_smooth_end]
        return np.mean(spl_sample)

    def calibration_popup(self, success_flag=True):
        """
        Display a calibration result popup.

        Shows different icons and message texts based on whether the calibration was successful.
        If calibration is successful, displays an information icon and success message;
        if calibration fails, displays a critical icon and failure message.

        Parameters:
        - success_flag: Boolean indicating whether the calibration was successful. Default is True.
        """
        cal_msg = MessageBox(self)
        if success_flag:
            cal_msg.setIcon(MessageBox.Information)
            cal_msg.setText("校准成功")
            cal_msg.setWindowTitle("校准成功")
        else:
            cal_msg.setIcon(MessageBox.Critical)
            cal_msg.setText("校准失败，请重试")
            cal_msg.setWindowTitle("校准失败")
        cal_msg.setStandardButtons(MessageBox.Ok)
        cal_msg.exec_()

    def calculate_average_spl(self, recorded_dict):
        """
        Calculate the average sound pressure level (SPL).

        This method records audio data, computes the SPL curve, and then calculates the average value from a selected range.

        Parameters:
        recorded_dict - Dictionary containing information for recording.

        Returns:
        Average SPL value.
        """
        rec_code, recorded_data = SoundcardAudioProcessor().sd_rec(recorded_dict)
        step = len(recorded_data) // 3
        if rec_code == error_code.OK:
            spl_smooth = AudioThdFrequencyResponseAnalysis().spl_calculation(
                recorded_data, method="rms", window_size=1201
            )
            spl_smooth_mid = len(spl_smooth) // 2
            spl_smooth_start = spl_smooth_mid - step
            spl_smooth_end = spl_smooth_mid + step
            spl_sample = spl_smooth[spl_smooth_start:spl_smooth_end]
            self.average_value = np.mean(spl_sample)
            return self.average_value

    def update_recorded_time(self):
        """
        Update the recorded time countdown.

        This function decrements the recorded time and updates the time display on the interface.
        The timer will stop automatically when time reaches 0 or stop_timer flag is set.
        """
        if self.recorded_time > 0 and not self.stop_timer:
            self.recorded_time -= 1
            # Update the time display on the interface, showing the remaining time in red and the unit "s" in black.
            self.recorded_label.setText(
                f"<span style='color: red;'>{self.recorded_time} </span>" f"<span style='color: black;'>s</span>"
            )
        else:
            self.update_ui_timer.stop()
            # Reset time for next calibration
            self.recorded_time = 10

    def calculate_v2pa_factor(self, average_value):
        """
        Calculate the v2pa_factor from the standard sound pressure level.

        This function calculates the v2pa_factor based on whether the standard SPL flag is set to True or False.
        If the flag is True, it uses 94 dB as the standard value; otherwise, it uses 114 dB.

        Args:
            average_value (float): The average sound pressure level value used to calculate the v2pa_factor.

        Returns:
            float: The calculated v2pa_factor value rounded to three decimal places.
        """
        if self.standard_spl_flag:
            deviation_value = round(94 - average_value, 3)
        else:
            deviation_value = round(114 - average_value, 3)
        v2pa_factor = 10 ** (deviation_value / 20)
        return v2pa_factor

    @staticmethod
    def save_v2pa_factor_to_text(v2pa_factor):
        """
        Save the v2pa_factor to a text file.

        This method writes the given v2pa_factor to a specified text file, along with the current date.
        This is particularly useful for tracking and debugging changes in UI configuration.

        Parameters:
        v2pa_factor (float): The v2pa_factor to be saved.
        """
        dir_path = DEFAULT_DIR + "ui/ui_config/"
        file_path = dir_path + "mic_calibration.txt"
        current_time = datetime.now().strftime("%Y-%m-%d")
        with open(file_path, "w") as f:
            f.write(f"v2pa_factor: \n{v2pa_factor}\n")
            f.write(f"Datetime: \n{current_time}\n")

    def reset_btn_clicked(self):
        """
        This method is triggered when the reset button is clicked.

        It resets the recorded time to 10 seconds and updates the recorded label to display the new time in red.
        Additionally, it clears the v2pa_factor line edit and stops any ongoing streaming recording.
        """
        # Stop timers
        self.update_ui_timer.stop()
        self.streaming_poll_timer.stop()

        # Clean up streaming resources
        if self.streaming_processor is not None:
            try:
                self.streaming_processor.stop_streaming()
            except Exception as e:
                self.default_logger.error(f"Error stopping streaming processor: {e}")
            self.streaming_processor = None

        self.stop_timer = False

        # Reset UI
        self.recorded_time = 10
        self.recorded_label.setText(
            f"<span style='color: red;'>{self.recorded_time} </span>" f"<span style='color: black;'>s</span>"
        )
        self.v2pa_factor_lineedit.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CalibrationWindow()
    window.show()
    window.exec()
