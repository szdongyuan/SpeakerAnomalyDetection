import sys
import threading
import time
from datetime import datetime
from concurrent import futures

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QDialog, QDoubleSpinBox, QGroupBox, QGridLayout, QHBoxLayout, QLineEdit, QLabel
from PyQt5.QtWidgets import QMessageBox, QTabWidget, QVBoxLayout, QPushButton, QSpacerItem
from PyQt5.QtWidgets import QSizePolicy, QSpinBox, QWidget, QRadioButton

from base.log_manager import LogManager
from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis
from base.pre_processing.swept_sine_chirps import StimulusSignal
from base.soundcard_audio_processor import SoundcardAudioProcessor
from base.soundcard_calibration_manager import SoundcardCalibrationManager
from consts import ui_style_const, error_code
from consts.running_consts import DEFAULT_DIR


class CalibrationWindow(QDialog):

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setWindowTitle("校准窗口")
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setMinimumSize(500, 550)
        self.setMaximumSize(600, 580)
        cal_wnd_layout = QVBoxLayout()

        self.tabwidget = QTabWidget()
        self.output_cal_wnd = OutputCalibration()
        self.input_cal_wnd = InputCalibration()
        self.tabwidget.addTab(self.output_cal_wnd, "输出校准")
        self.tabwidget.addTab(self.input_cal_wnd, "输入校准")

        btn_layout = self.create_btn_box()

        cal_wnd_layout.addWidget(self.tabwidget)
        cal_wnd_layout.addLayout(btn_layout)
        self.setLayout(cal_wnd_layout)

    def create_btn_box(self):
        btn_layout = QHBoxLayout()
        self.cal_btn = QPushButton(" 校  准 ")
        self.cal_btn.clicked.connect(self.clicked_calibration_button)
        reset_btn = QPushButton(" 重  置 ")
        reset_btn.clicked.connect(self.clicked_reset_button)
        cancel_btn = QPushButton(" 退  出 ")
        cancel_btn.clicked.connect(self.clicked_close_button)
        h_spacer_btn1 = QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        h_spacer_btn2 = QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        btn_layout.addWidget(self.cal_btn)
        btn_layout.addItem(h_spacer_btn1)
        btn_layout.addWidget(reset_btn)
        btn_layout.addItem(h_spacer_btn2)
        btn_layout.addWidget(cancel_btn)
        return btn_layout

    def clicked_calibration_button(self):
        current_tab_index = self.tabwidget.currentIndex()
        if current_tab_index == 0:
            self.output_cal_wnd.calibration()
        elif current_tab_index == 1:
            self.cal_btn.setDisabled(True)
            self.input_cal_wnd.clicked_calibration()

    def clicked_reset_button(self):
        current_tab_index = self.tabwidget.currentIndex()
        if current_tab_index == 0:
            self.output_cal_wnd.reset_btn_clicked()
        elif current_tab_index == 1:
            self.input_cal_wnd.reset_btn_clicked()
            self.cal_btn.setDisabled(False)

    def clicked_close_button(self):
        current_tab_index = self.tabwidget.currentIndex()
        if current_tab_index == 0:
            self.close()
        elif current_tab_index == 1:
            self.input_cal_wnd.stop_timer = True
            self.close()


class OutputCalibration(QWidget):
    def __init__(self):
        super().__init__()
        self.output_voltage_value = []
        self.default_logger = LogManager.set_log_handler("core")
        self.calibration_param = {"calibration_nums": 5}
        self.current_count = 1
        self.countdown = 10
        self.play_flag = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_countdown)
        self.init_ui()
        self.get_calibration_param()

    def init_ui(self):
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/DT_ico.ico"))
        self.setWindowTitle("校准")
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setMinimumSize(305, 373)
        self.setMaximumSize(520, 500)

        calibration_param_box = self.create_calibration_param_box()
        output_box = self.create_output_voltage_box()
        test_box = self.create_test_box()

        v_spacer_1 = QSpacerItem(30, 30, QSizePolicy.Minimum, QSizePolicy.Expanding)
        v_spacer_2 = QSpacerItem(30, 30, QSizePolicy.Minimum, QSizePolicy.Expanding)
        v_spacer_3 = QSpacerItem(30, 30, QSizePolicy.Minimum, QSizePolicy.Expanding)

        layout = QVBoxLayout()
        layout.addWidget(calibration_param_box)
        layout.addItem(v_spacer_1)
        layout.addWidget(output_box)
        layout.addItem(v_spacer_2)
        layout.addWidget(test_box)
        layout.addItem(v_spacer_3)
        layout.setContentsMargins(12, 20, 12, 25)
        self.setLayout(layout)
        self.setStyleSheet(ui_style_const.qcombobox_stytle +
                           ui_style_const.qpushbutton_stytle +
                           ui_style_const.qspinbox_stytle +
                           ui_style_const.qdoublespinbox_stytle +
                           ui_style_const.qgroupbox_stytle +
                           ui_style_const.qlabel_stytle)

    def create_calibration_param_box(self):
        calibration_param_box = QGroupBox("校准参数")
        calibration_nums_label = QLabel("校准次数")
        self.calibration_nums_box = QSpinBox()
        self.calibration_nums_box.setSuffix(" 次")
        self.calibration_nums_box.setRange(3, 20)
        self.calibration_nums_box.setValue(5)
        self.calibration_nums_box.setFixedSize(80, 20)
        self.calibration_nums_box.editingFinished.connect(self.get_calibration_param)
        h_spacer_1 = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        param_layout = QHBoxLayout()
        param_layout.addWidget(calibration_nums_label)
        param_layout.addItem(h_spacer_1)
        param_layout.addWidget(self.calibration_nums_box)
        param_layout.setContentsMargins(9, 10, 10, 10)
        calibration_param_box.setLayout(param_layout)
        return calibration_param_box

    def create_output_voltage_box(self):
        output_box = QGroupBox("输出电压")
        output_layout = QGridLayout()
        self.play_label = QLabel(f"第 {self.current_count} 次 ")
        self.play_label.setFixedSize(68, 14)
        self.play_btn = QPushButton(" 播  放 ")
        self.play_btn.clicked.connect(self.play_btn_clicked)
        self.countdown_label = QLabel(f"<span style='color: black;'>倒计时：</span>"
                                      f"<span style='color: red;'>{self.countdown} </span>"
                                      f"<span style='color: black;'>s</span>")
        self.countdown_label.setStyleSheet("background-color: white;"
                                           "border: 1px solid rgb(122, 122, 122);"
                                           "border-radius: 3px;")
        h_spacer_play_label_left = QSpacerItem(0, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        h_spacer_play_label_right = QSpacerItem(0, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        output_layout.addWidget(self.play_label, 0, 0)
        output_layout.addItem(h_spacer_play_label_left, 0, 1)
        output_layout.addWidget(self.countdown_label, 0, 2)
        output_layout.addItem(h_spacer_play_label_right, 0, 3)
        output_layout.addWidget(self.play_btn, 0, 4)

        output_voltage_label = QLabel("输出电压")
        self.output_voltage_box = QDoubleSpinBox()
        self.output_voltage_box.setFixedSize(80, 16)
        self.output_voltage_box.setSuffix(" V")
        self.output_voltage_box.setFixedSize(105, 23)
        self.output_voltage_box.setRange(0, 100)
        self.output_voltage_box.setSingleStep(0.1)
        self.save_btn = QPushButton(" 保  存 ")
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
        test_box = QGroupBox("测    试")
        target_V_label = QLabel("目标电压")
        self.target_voltage_box = QDoubleSpinBox()
        self.target_voltage_box.setFixedSize(105, 23)
        self.target_voltage_box.setSuffix(" V")
        h_spacer_voltage_box_right = QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        h_spacer_voltage_box_left = QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        test_btn = QPushButton(" 测  试 ")
        test_btn.clicked.connect(self.test_calibration)
        test_layout = QHBoxLayout()
        test_layout.addWidget(target_V_label)
        test_layout.addItem(h_spacer_voltage_box_left)
        test_layout.addWidget(self.target_voltage_box)
        test_layout.addItem(h_spacer_voltage_box_right)
        test_layout.addWidget(test_btn)
        test_box.setLayout(test_layout)
        return test_box

    def save_btn_clicked(self):
        self.output_voltage_value.append(self.output_voltage_box.value())
        self.output_voltage_box.setValue(0)
        self.create_current_count()
        self.play_label.setText(f"第 {self.current_count} 次 ")
        self.countdown_label.setText(f"<span style='color: black;'>倒计时：</span>"
                                     f"<span style='color: red;'>{self.countdown} </span>"
                                     f"<span style='color: black;'>s</span>")

    def update_countdown(self):
        if self.countdown > 0:
            self.countdown -= 1
            self.countdown_label.setText(f"<span style='color: black;'>倒计时：</span>"
                                         f"<span style='color: red;'>{self.countdown} </span>"
                                         f"<span style='color: black;'>s</span>")
        else:
            self.timer.stop()
            self.play_btn.setText(" 停  止 ")
            self.play_btn.setEnabled(False)
            self.save_btn.setEnabled(True)
            self.play_flag = False
            self.countdown = 10

    def play_btn_clicked(self):
        stimulus_dict = self.create_signal()
        sap = SoundcardAudioProcessor()
        if not self.play_flag:
            self.play_flag = True
            self.play_btn.setDisabled(True)
            self.save_btn.setDisabled(True)
            self.countdown_label.setText(f"<span style='color: black;'>倒计时：</span>"
                                         f"<span style='color: red;'>{self.countdown} </span>"
                                         f"<span style='color: black;'>s</span>")
            self.timer.start(1000)
            threading.Thread(target=sap.sd_play, args=(stimulus_dict,)).start()
        else:
            self.play_flag = False
            self.timer.stop()
            self.play_btn.setText(" 播  放 ")
            if self.current_count >= self.calibration_param["calibration_nums"]:
                self.save_btn.setDisabled(True)

    def create_current_count(self):
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
        calibration_nums = self.calibration_nums_box.value()
        output_voltage = self.output_voltage_value
        amplitude_list = np.linspace(0.05, 0.95, calibration_nums)
        self.calibration_param = {
            "calibration_nums": calibration_nums,
            "output_voltage": output_voltage,
            "amplitude_list": amplitude_list
        }

    def create_signal(self):
        data, sr = StimulusSignal().generate_chirps(start_freq=800, stop_freq=800, total_time=10, sample_rate=44100,
                                                    stimulus_type='linear')
        stimulus_dict = {"data": data,
                         "sr": sr,
                         "amplitude": self.calibration_param["amplitude_list"][self.current_count - 1],
                         }
        return stimulus_dict

    def test_calibration(self):
        target_voltage = self.target_voltage_box.value()
        scm = SoundcardCalibrationManager()
        calibrate_code, calibrate_result = scm.calibrate_amplitude(target_voltage)
        if calibrate_code != error_code.OK:
            self.default_logger.error(f"Failed to calculate the amplitude. {calibrate_result}")
        amplitude, max_voltage = calibrate_result
        if target_voltage > max_voltage:
            if self.test_calibration_popup():
                return
        data, sr = StimulusSignal().generate_chirps(start_freq=800, stop_freq=800, total_time=10, sample_rate=44100,
                                                    stimulus_type='linear')
        test_stimulus_dict = {"data": data, "sr": sr, "amplitude": amplitude}
        sap = SoundcardAudioProcessor()
        speaker_code, msg = sap.sd_play(test_stimulus_dict)
        if speaker_code != error_code.OK:
            self.default_logger.error(f"Failed to play the audio. {msg}")

    def test_calibration_popup(self):
        test_msg = QMessageBox(self)
        test_msg.setIcon(QMessageBox.Warning)
        test_msg.setText("目标电压过大，请重新输入!")
        test_msg.setWindowTitle("测试失败")
        test_msg.setStandardButtons(QMessageBox.Ok)
        button = test_msg.exec_()
        return button == QMessageBox.Ok

    def reset_btn_clicked(self):
        self.output_voltage_value.clear()
        self.output_voltage_box.setValue(0)
        self.target_voltage_box.setValue(0)
        self.calibration_nums_box.setValue(5)
        self.timer.stop()
        self.current_count = 1
        self.countdown = 10
        self.play_label.setText(f"第 {self.current_count} 次 ")
        self.countdown_label.setText(f"<span style='color: black;'>倒计时：</span>"
                                     f"<span style='color: red;'>{self.countdown} </span>"
                                     f"<span style='color: black;'>s</span>")
        self.save_btn.setText(" 保  存 ")
        self.play_btn.setText(" 播  放 ")
        self.play_flag = False
        self.play_btn.setEnabled(True)
        self.save_btn.setDisabled(True)

    def calibration(self):
        scm = SoundcardCalibrationManager()
        if len(self.output_voltage_value) != self.calibration_param["calibration_nums"]:
            self.calibration_popup(success_flag=False)
            self.default_logger.error("The saved voltage does not meet the requirement of calibration times.")
        else:
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
        cal_msg = QMessageBox(self)
        if success_flag:
            cal_msg.setIcon(QMessageBox.Information)
            cal_msg.setText("校准成功")
            cal_msg.setWindowTitle("校准成功")
        else:
            cal_msg.setIcon(QMessageBox.Critical)
            cal_msg.setText("校准失败，请重试")
            cal_msg.setWindowTitle("校准失败")
        cal_msg.setStandardButtons(QMessageBox.Ok)
        cal_msg.exec_()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            event.ignore()
        else:
            super().keyPressEvent(event)



class InputCalibration(QWidget):
    def __init__(self):
        super().__init__()
        self.default_logger = LogManager.set_log_handler("core")
        self.stop_timer = False
        self.init_ui()
    def init_ui(self):
        self.setWindowTitle("输入校准")
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setMinimumSize(305, 373)
        self.setMaximumSize(520, 500)
        self.standard_spl_flag = True
        self.recorded_flag = False

        standard_spl_box = self.create_standard_spl_box()
        recorded_box = self.create_recorded_box()
        deviation_spl_box = self.create_deviation_spl_box()

        layout = QVBoxLayout()
        layout.addWidget(standard_spl_box)
        layout.addWidget(recorded_box)
        layout.addWidget(deviation_spl_box)

        self.setLayout(layout)
        self.setStyleSheet(ui_style_const.qcombobox_stytle +
                           ui_style_const.qpushbutton_stytle +
                           ui_style_const.qspinbox_stytle +
                           ui_style_const.qdoublespinbox_stytle +
                           ui_style_const.qgroupbox_stytle +
                           ui_style_const.qlabel_stytle +
                           ui_style_const.qlineedit_stytle
                           # + ui_style_const.qradiobutton_stytle
                           )

    def create_deviation_spl_box(self):
        deviation_spl_box = QGroupBox("校准结果")
        deviation_label = QLabel("声压偏差：")
        self.deviation_lineedit = QLineEdit()
        self.deviation_lineedit.setDisabled(True)

        standard_deviation_layout = QHBoxLayout()
        h_spacer_deviation_center = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        standard_deviation_layout.addWidget(deviation_label)
        standard_deviation_layout.addItem(h_spacer_deviation_center)
        standard_deviation_layout.addWidget(self.deviation_lineedit)
        deviation_spl_box.setLayout(standard_deviation_layout)

        return deviation_spl_box

    def create_recorded_box(self):
        recorded_box = QGroupBox("录制音频")
        recorded_label = QLabel("录制时间：")
        self.recorded_label = QLabel()
        self.recorded_label.setFixedSize(70, 25)
        self.recorded_label.setAlignment(Qt.AlignCenter)
        self.recorded_time = 10
        self.recorded_label.setText(f"<span style='color: red;'>{self.recorded_time} </span>"
                                    f"<span style='color: black;'>s</span>")

        self.recorded_label.setStyleSheet("background-color: white;"
                                          "border: 1px solid rgb(122, 122, 122);"
                                          "border-radius: 3px;")

        recorded_layout = QHBoxLayout()
        h_spacer_deviation_center = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        recorded_layout.addWidget(recorded_label)
        recorded_layout.addItem(h_spacer_deviation_center)
        recorded_layout.addWidget(self.recorded_label)
        recorded_box.setLayout(recorded_layout)

        return recorded_box

    def create_standard_spl_box(self):
        standard_spl_box = QGroupBox("标准声压")

        self.standard_spl_i = QRadioButton("94  dB")
        self.standard_spl_ii = QRadioButton("114 dB")
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
        if self.standard_spl_i.isChecked():
            self.standard_spl_flag = True
        elif self.standard_spl_ii.isChecked():
            self.standard_spl_flag = False

    def clicked_calibration(self):
        prolong = 1
        recorded_dict = {"channels": 1,
                         "sample_rate": 44100,
                         "num_frames": 10 * 44100,
                         "prolong_frames": int(prolong * 44100)
                         }

        with futures.ThreadPoolExecutor(max_workers=1) as executor:
            recorded_thread = executor.submit(self.calculate_average_spl, recorded_dict)
            self.update_recorded_time()

        self.average_value = recorded_thread.result()
        deviation_value = self.calculate_deviation(self.average_value)
        self.deviation_lineedit.setText(str(deviation_value))
        if not self.stop_timer:
            if str(deviation_value) == "inf":
                self.calibration_popup(success_flag=False)
            else:
                self.calibration_popup(success_flag=True)
                self.default_logger.info("Calibration success.")
                self.save_deviation_value_to_text(deviation_value)

    def calibration_popup(self, success_flag=True):
        cal_msg = QMessageBox(self)
        if success_flag:
            cal_msg.setIcon(QMessageBox.Information)
            cal_msg.setText("校准成功")
            cal_msg.setWindowTitle("校准成功")
        else:
            cal_msg.setIcon(QMessageBox.Critical)
            cal_msg.setText("校准失败，请重试")
            cal_msg.setWindowTitle("校准失败")
        cal_msg.setStandardButtons(QMessageBox.Ok)
        cal_msg.exec_()

    def calculate_average_spl(self, recorded_dict):
        rec_code, recorded_data = SoundcardAudioProcessor().sd_rec(recorded_dict)
        step = 100
        if rec_code == error_code.OK:
            spl_smooth = AudioThdFrequencyResponseAnalysis().spl_calculation(recorded_data)
            spl_smooth_mid = len(spl_smooth) // 2
            spl_smooth_start = spl_smooth_mid - step
            spl_smooth_end = spl_smooth_mid + step
            spl_sample = spl_smooth[spl_smooth_start:spl_smooth_end]
            self.average_value = np.sum(spl_sample) / (step * 2)
            return self.average_value

    def update_recorded_time(self):
        while self.recorded_time > 0 and not self.stop_timer:
            time.sleep(1)
            self.recorded_time -= 1
            self.recorded_label.setText(f"<span style='color: red;'>{self.recorded_time} </span>"
                                        f"<span style='color: black;'>s</span>")
            QApplication.processEvents()

    def calculate_deviation(self, average_value):
        if self.standard_spl_flag:
            deviation_value = round(94 - average_value, 3)
        else:
            deviation_value = round(114 - average_value, 3)
        return deviation_value

    @staticmethod
    def save_deviation_value_to_text(deviation_value):
        dir_path = DEFAULT_DIR + 'ui/ui_config/'
        file_path = dir_path + "mic_calibration.txt"
        current_time = datetime.now().strftime("%Y-%m-%d")
        with open(file_path, 'w') as f:
            f.write(f"deviation_value: \n{deviation_value}\n")
            f.write(f"Datetime: \n{current_time}\n")

    def reset_btn_clicked(self):
        self.recorded_time = 10
        self.recorded_label.setText(f"<span style='color: red;'>{self.recorded_time} </span>"
                                    f"<span style='color: black;'>s</span>")
        self.deviation_lineedit.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OutInCalibrationWindow()
    window.show()
    window.exec()

