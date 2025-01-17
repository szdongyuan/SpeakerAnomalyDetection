import sys

import librosa
import numpy as np
import pyqtgraph as pg
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis
from consts.running_consts import DEFAULT_DIR


# class SignalAnalysisWindow(QDialog):
#     def __init__(self, signal_info):
#         super().__init__()
#         self.signal_info = signal_info
#         self.default_logger = LogManager.set_log_handler("core")
#         self.init_ui()

#     def init_ui(self):
#         # set the dialog theme and disable help and close btutton
#         self.setWindowTitle("音频分析窗口")
#         self.setWindowFlag(Qt.WindowCloseButtonHint, False)
#         self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
#         # set widget style
#         self.setStyleSheet(ui_style_const.qpushbutton_stytle +
#                            ui_style_const.qgroupbox_stytle +
#                            ui_style_const.qlabel_stytle +
#                            ui_style_const.qlineedit_stytle +
#                            "QTabWidget {font-size: 11pt;}" +
#                            "QPushButton {background-color: #4472c4; color: white;}" +
#                            "QPushButton:hover {background-color: #4472c4; color: white; border-color: #803333ff;}")
#         signal_analysis_layout = QVBoxLayout()      # create main layout


#         # set three tab, show spl, frequency and thd's result
#         self.tabwidget = QTabWidget()
#         self.spl_wnd = Spl(self.signal_info)
#         self.frequency_wnd = Frequency(self.signal_info)
#         self.distortion_wnd = Distortion(self.signal_info)
#         self.spl_wnd.show()
#         self.frequency_wnd.show()
#         self.distortion_wnd.show()

#         base_btn_layout = QGridLayout()
#         save_btn = QPushButton("保  存")        # use to save current tab analysis data
#         save_btn.setFixedSize(100, 30)
#         # save_btn.clicked.connect(self.save_btn_clicked)

#         base_btn_layout.addWidget(save_btn, 0, 0)

#         signal_analysis_layout.addWidget(self.tabwidget)
#         signal_analysis_layout.addLayout(base_btn_layout)
#         self.setLayout(signal_analysis_layout)

#     def save_failed_popup(self):
#         # the function infor us the right way to save
#         save_failed_msg = QMessageBox(self)
#         save_failed_msg.setIcon(QMessageBox.Critical)
#         save_failed_msg.setText("请先点击绘图按钮")
#         save_failed_msg.setWindowTitle("保存失败")
#         save_failed_msg.setStandardButtons(QMessageBox.Ok)
#         save_failed_msg.exec_()

#     def save_data_to_txt(self, result):
#         # save file to the   selected location
#         file_path, _ = QFileDialog.getSaveFileName(self,
#                                                    "保存数据",
#                                                    "",
#                                                    "Text Files (*.txt)",
#                                                    options=QFileDialog.DontUseNativeDialog)
#         if file_path:
#             try:
#                 with open(file_path, 'w') as f:
#                     f.write(str(result))
#                 self.default_logger.info(f"The file was saved to {file_path}.")
#             except Exception as e:
#                 self.default_logger.error(f"Failed to save the file. {e}")


class Distortion(QWidget):
    def __init__(self, signal_info):
        super().__init__()
        self.signal_info = signal_info
        self.refresh_stimulus_flag = None
        self.selected_label = None
        self.selected_harmonics =  list(range(2, 6))
        self.freq_dict = None
        self.base_freq_list = None
        self.result = {}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("谐波分析")
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        # layout have a child's plotwidget, use to display the thd analysis results
        layout = QVBoxLayout()

        self.thd_plot = pg.PlotWidget()
        self.thd_plot.setBackground('white')

        # layout.addWidget(harmonic_group_box)
        layout.addWidget(self.thd_plot)
        self.setLayout(layout)

    # @staticmethod
    # def get_label_text(value):
    #     # set the thd step
    #     if value == 2:
    #         return str(value)
    #     elif value >= 7:
    #         start = (value - 6) * 5 + 5
    #         end = (value - 6) * 5 + 10
    #         return f"{start}...{end}"
    #     else:
    #         return f"2...{value}"

    def calculate_thd(self):
        freq_value, harmonic, thd = [], [], []
        if self.selected_harmonics:
            kwargs = {"harmonics": self.selected_harmonics}
            stimulus_signal = self.signal_info["stimulus_signal"]
            recorded_signal = self.signal_info["recorded_signal"]
            sample_rate = self.signal_info["sample_rate"]
            atfra = AudioThdFrequencyResponseAnalysis()
            if self.refresh_stimulus_flag or (self.freq_dict is None or self.base_freq_list is None):
                self.freq_dict, self.base_freq_list = atfra.calculate_spectrum(stimulus_signal, sample_rate)
                self.refresh_stimulus_flag = False
            freq_value, harmonic, thd = atfra.calculate_thd(self.freq_dict, self.base_freq_list,
                                                            recorded_signal, sample_rate, **kwargs)
        self.plot_graph(freq_value, thd)
        if isinstance("harmonic", np.ndarray):
            harmonic = harmonic.tolist()
        self.result = {"freq_value": freq_value,
                       "harmonic": harmonic,
                       "thd": thd}
        return self.result

    def plot_graph(self, freq_value, thd):
        # Draw a graph based on the calculated thd
        self.thd_plot.clear()
        if self.check_valid_data(freq_value) and self.check_valid_data(thd):
            self.thd_plot.plot(freq_value, thd, pen='b', name="THD")
        if self.selected_label is not None:
            self.thd_plot.setTitle(f"The Distortion of {self.selected_label.text()} order")
        self.thd_plot.setLabel('left', 'Distortion(%)')
        self.thd_plot.setLabel('bottom', 'Frequency')
        self.thd_plot.setLogMode(x=True, y=False)
        self.thd_plot.showGrid(x=True, y=True)

    @staticmethod
    def check_valid_data(data):
        return isinstance(data, (list, np.ndarray)) and len(data) > 0


class Spl(QWidget):
    def __init__(self, signal_info):
        super().__init__()
        self.signal_info = signal_info
        self.result = {}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("声压分析")
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.spl_plot = pg.PlotWidget(title='Sound Pressure Level')
        self.spl_plot.setBackground('white')
        layout = QVBoxLayout()
        layout.addWidget(self.spl_plot)
        self.setLayout(layout)

    def calculate_spl(self):
        # calculate Sound Pressure Level according to recorded_signal
        recorded_signal = self.signal_info["recorded_signal"]
        sample_rate = self.signal_info["sample_rate"]
        signal_duration = np.linspace(0, len(recorded_signal) / sample_rate, len(recorded_signal))
        reference_pressure = 20e-6
        signal_spl = AudioThdFrequencyResponseAnalysis().spl_calculation(recorded_signal, reference_pressure)
        self.plot_spl(signal_duration, recorded_signal, signal_spl)
        self.result = {"signal_duration": signal_duration.tolist(),
                       "recorded_signal": recorded_signal.tolist(),
                       "signal_spl": signal_spl.tolist(),
                       }
        return self.result

    def plot_spl(self, signal_duration, recorded_signal, signal_spl):
        self.spl_plot.clear()
        self.spl_plot.plot(signal_duration, signal_spl, pen='r')
        self.spl_plot.setLabel('left', 'SPL (dB)')
        self.spl_plot.setLabel('bottom', 'Time (s)')


class Frequency(QWidget):

    def __init__(self, signal_info):
        super().__init__()
        self.signal_info = signal_info
        self.smooth_flag = False
        self.temp_frequency_list = None
        self.result = {}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("频响分析")
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.fr_plot = pg.PlotWidget(title='Frequency Response')
        # self.fr_plot.setFixedSize(400, 320)
        self.fr_plot.setBackground('white')
        layout = QVBoxLayout()
        layout.addWidget(self.fr_plot)
        self.setLayout(layout)

    def calculate_fr(self):
        stimulus_signal = self.signal_info["stimulus_signal"]
        recorded_signal = self.signal_info["recorded_signal"]
        sr = self.signal_info["sample_rate"]
        fr, frequency_list = AudioThdFrequencyResponseAnalysis().calculate_fr(stimulus_signal, recorded_signal, sr)
        self.plot_fr(frequency_list, fr)
        self.result = {"fr": fr.tolist(),
                       "frequency_list": frequency_list.tolist()
                       }
        return self.result

    def plot_fr(self, frequency_list, fr):
        # drawing the Frequency Response
        self.fr_plot.clear()
        self.fr_plot.plot(frequency_list, fr, pen='b')
        self.fr_plot.setLabel('left', 'Amplitude (dB)')
        self.fr_plot.setLabel('bottom', 'Frequency (Hz)')
        self.fr_plot.setLogMode(x=True, y=False)


if __name__ == "__main__":
    stimulus, sr = librosa.load("../audio_data/analysis_samples/stimulus.wav", sr=44100)
    recorded, _ = librosa.load("../audio_data/analysis_samples/recording.wav", sr=44100)
    signal_info = {"stimulus_signal": stimulus,
                   "recorded_signal": recorded,
                   "sample_rate": sr}
    app = QApplication(sys.argv)
    window = Spl(signal_info)
    window.show()
    window.exec()
