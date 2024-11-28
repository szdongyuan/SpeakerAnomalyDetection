import numpy as np
import sys
import threading

from PyQt5.QtCore import Qt, QEvent, QTimer
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QGroupBox, QLabel, QSpinBox, QPushButton, QVBoxLayout
from PyQt5.QtWidgets import QApplication, QSpacerItem, QSizePolicy, QDoubleSpinBox, QMessageBox

from base.log_manager import LogManager
from base.pre_processing.swept_sine_chirps import StimulusSignal
from base.soundcard_audio_processor import SoundcardAudioProcessor
from base.soundcard_calibration_manager import SoundcardCalibrationManager
from consts import error_code


class CalibrationWindow(QDialog):

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
        self.setWindowTitle("校准")
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)

        calibration_param_box = self.create_calibration_param_box()
        output_box = self.create_output_voltage_box()
        test_box = self.create_test_box()
        btn_layout = self.create_btn_box()

        v_spacer_1 = QSpacerItem(30, 30, QSizePolicy.Minimum, QSizePolicy.Expanding)
        v_spacer_2 = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)

        layout = QVBoxLayout()
        layout.addWidget(calibration_param_box)
        layout.addItem(v_spacer_1)
        layout.addWidget(output_box)
        layout.addItem(v_spacer_1)
        layout.addWidget(test_box)
        layout.addItem(v_spacer_1)
        layout.addLayout(btn_layout)
        layout.addItem(v_spacer_2)
        self.setLayout(layout)

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
        param_layout.addWidget(self.calibration_nums_box)
        param_layout.addItem(h_spacer_1)
        calibration_param_box.setLayout(param_layout)
        return calibration_param_box

    def create_output_voltage_box(self):
        output_box = QGroupBox("输出电压")
        output_layout = QVBoxLayout()
        play_layout = QHBoxLayout()
        self.play_label = QLabel(f"第 {self.current_count} 次")
        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(self.play_btn_clicked)
        self.countdown_label = QLabel(f"倒计时: {self.countdown} 秒")
        h_spacer_1 = QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        play_layout.addWidget(self.play_label)
        play_layout.addWidget(self.play_btn)
        play_layout.addItem(h_spacer_1)
        play_layout.addWidget(self.countdown_label)

        save_voltage_layout = QHBoxLayout()
        output_voltage_label = QLabel("输出电压")
        self.output_voltage_box = QDoubleSpinBox()
        self.output_voltage_box.setSuffix(" V")
        self.output_voltage_box.setFixedSize(80, 20)
        self.output_voltage_box.setRange(0, 100)
        self.output_voltage_box.setSingleStep(0.1)
        self.output_voltage_box.installEventFilter(self)
        h_spacer_1 = QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.save_btn = QPushButton("保存")
        self.save_btn.clicked.connect(self.save_btn_clicked)
        self.save_btn.setDisabled(True)
        save_voltage_layout.addWidget(output_voltage_label)
        save_voltage_layout.addWidget(self.output_voltage_box)
        save_voltage_layout.addItem(h_spacer_1)
        save_voltage_layout.addWidget(self.save_btn)

        v_spacer_1 = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        output_layout.addLayout(play_layout)
        output_layout.addItem(v_spacer_1)
        output_layout.addLayout(save_voltage_layout)
        output_box.setLayout(output_layout)
        return output_box

    def create_test_box(self):
        test_box = QGroupBox("测试")
        target_V_label = QLabel("目标电压")
        self.target_voltage_box = QDoubleSpinBox()
        self.target_voltage_box.setFixedSize(80, 20)
        self.target_voltage_box.setSuffix(" V")
        h_spacer_1 = QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        test_btn = QPushButton("测试")
        test_btn.clicked.connect(self.test_calibration)
        test_layout = QHBoxLayout()
        test_layout.addWidget(target_V_label)
        test_layout.addWidget(self.target_voltage_box)
        test_layout.addItem(h_spacer_1)
        test_layout.addWidget(test_btn)
        test_box.setLayout(test_layout)
        return test_box

    def create_btn_box(self):
        btn_layout = QHBoxLayout()
        calibration_btn = QPushButton("校准")
        calibration_btn.clicked.connect(self.calibration)
        reset_btn = QPushButton("重置")
        reset_btn.clicked.connect(self.reset_btn_clicked)
        cancel_btn = QPushButton("结束")
        cancel_btn.clicked.connect(self.cancel_btn_clicked)
        btn_layout.addWidget(calibration_btn)
        btn_layout.addWidget(reset_btn)
        btn_layout.addWidget(cancel_btn)
        return btn_layout

    def reset_btn_clicked(self):
        self.output_voltage_value.clear()
        self.output_voltage_box.setValue(0)
        self.target_voltage_box.setValue(0)
        self.timer.stop()
        self.current_count = 1
        self.countdown = 10
        self.play_label.setText(f"第 {self.current_count} 次")
        self.countdown_label.setText(f"倒计时: {self.countdown} 秒")
        self.save_btn.setText("保存")
        self.play_btn.setText("播放")
        self.play_btn.setEnabled(True)

    def save_btn_clicked(self):
        self.output_voltage_value.append(self.output_voltage_box.value())
        self.output_voltage_box.setValue(0)
        self.create_current_count()
        self.play_label.setText(f"第 {self.current_count} 次")
        self.countdown_label.setText(f"倒计时: {self.countdown} 秒")

    def update_countdown(self):
        if self.countdown > 0:
            self.countdown -= 1
            self.countdown_label.setText(f"倒计时: {self.countdown} 秒")
        else:
            self.timer.stop()
            self.play_btn.setText("停止")
            self.play_btn.setEnabled(False)
            self.save_btn.setEnabled(True)
            self.play_flag = False
            self.countdown = 10

    def play_btn_clicked(self):
        stimulus_dict = self.create_signal()
        if not self.play_flag:
            self.play_flag = True
            self.countdown_label.setText(f"倒计时: {self.countdown} 秒")
            self.timer.start(1000)
            threading.Thread(target=SoundcardAudioProcessor().speaker_worker, args=(stimulus_dict,)).start()
            self.save_btn.setDisabled(True)
        else:
            self.play_flag = False
            self.timer.stop()
            self.play_btn.setText("播放")
            if self.current_count >= self.calibration_param["calibration_nums"]:
                self.save_btn.setDisabled(True)
            self.save_btn.setEnabled(True)

    def create_current_count(self):
        if self.current_count >= self.calibration_param["calibration_nums"]:
            self.save_btn.setText("完成")
            self.save_btn.setDisabled(True)
            self.play_btn.setDisabled(True)
        else:
            self.current_count += 1
            self.play_btn.setText("播放")
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
        amplitude = calibrate_result
        data, sr = StimulusSignal().generate_chirps(start_freq=800, stop_freq=800, total_time=10, sample_rate=44100,
                                                    stimulus_type='linear')
        test_stimulus_dict = {"data": data, "sr": sr, "amplitude": amplitude}
        speaker_code, msg = SoundcardAudioProcessor().speaker_worker(test_stimulus_dict)
        if speaker_code != error_code.OK:
            self.default_logger.error(f"Failed to play the audio. {msg}")

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

    def eventFilter(self, source, event):
        if isinstance(source, QDoubleSpinBox):
            if event.type() == QEvent.KeyPress:
                if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
                    return True
        return super().eventFilter(source, event)

    def cancel_btn_clicked(self):
        print(1111)
        self.close()
        print(2222)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CalibrationWindow()
    # sys.exit(app.exec())
    window.exec()
