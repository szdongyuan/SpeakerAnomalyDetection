import librosa
import numpy as np
import pyqtgraph as pg
import sys
from functools import partial
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QApplication, QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QMessageBox
from PyQt5.QtWidgets import QGridLayout, QScrollArea, QCheckBox, QGroupBox, QFileDialog
from PyQt5.QtWidgets import QSizePolicy, QTabWidget, QSpacerItem, QLineEdit, QComboBox

from base.log_manager import LogManager
from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis


class SignalAnalysisWindow(QDialog):
    def __init__(self, signal_info):
        super().__init__()
        self.signal_info = signal_info
        self.default_logger = LogManager.set_log_handler("core")
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("音频分析窗口")
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        signal_analysis_layout = QVBoxLayout()

        self.tabwidget = QTabWidget()
        spl_wnd = Spl(self.signal_info)
        frequency_wnd = Frequency(self.signal_info)
        distortion_wnd = Distortion(self.signal_info)
        self.tabwidget.addTab(spl_wnd, "声压级")
        self.tabwidget.addTab(frequency_wnd, "频响")
        self.tabwidget.addTab(distortion_wnd, "失真")

        base_btn_layout = QGridLayout()
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self.save_btn_clicked)

        base_btn_layout.addWidget(save_btn, 0, 0)

        signal_analysis_layout.addWidget(self.tabwidget)
        signal_analysis_layout.addLayout(base_btn_layout)
        self.setLayout(signal_analysis_layout)

    def save_btn_clicked(self):
        current_widget = self.tabwidget.currentWidget()
        result = current_widget.result
        if result:
            self.save_data_to_txt(result)
        else:
            self.save_failed_popup()

    def save_failed_popup(self):
        save_failed_msg = QMessageBox(self)
        save_failed_msg.setIcon(QMessageBox.Critical)
        save_failed_msg.setText("请先点击绘图按钮")
        save_failed_msg.setWindowTitle("保存失败")
        save_failed_msg.setStandardButtons(QMessageBox.Ok)
        save_failed_msg.exec_()

    def save_data_to_txt(self, result):
        file_path, _ = QFileDialog.getSaveFileName(self, "保存数据", "", "Text Files (*.txt)")
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(str(result))
                self.default_logger.info(f"The file was saved to {file_path}.")
            except Exception as e:
                self.default_logger.error(f"Failed to save the file. {e}")


class Distortion(QWidget):
    def __init__(self, signal_info):
        super().__init__()
        self.signal_info = signal_info
        self.selected_label = None
        self.selected_harmonics = []
        self.freq_dict = None
        self.base_freq_list = None
        self.result = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        harmonic_group_box = QGroupBox("Harmonics")
        harmonic_group_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        harmonic_slider_layout = self.create_harmonic_slider_layout()
        harmonic_group_box.setLayout(harmonic_slider_layout)

        self.thd_plot = pg.PlotWidget()
        self.thd_plot.setFixedSize(400, 300)
        self.thd_plot.setBackground('white')

        layout.addWidget(harmonic_group_box)
        layout.addWidget(self.thd_plot)
        self.setLayout(layout)

    def create_harmonic_slider_layout(self):
        harmonic_slider_layout = QHBoxLayout()
        scroll_area = QScrollArea()
        scroll_area.setFixedSize(130, 120)
        box_container = QWidget()
        box_layout = QVBoxLayout()

        for i in range(2, 9):
            label_text = self.get_label_text(i)
            label = QLabel(label_text)
            label.setAlignment(Qt.AlignLeft)
            label.setAutoFillBackground(True)
            label.mousePressEvent = partial(self.on_label_click, value=i, label=label)
            box_layout.addWidget(label)
        box_container.setLayout(box_layout)
        scroll_area.setWidget(box_container)

        btn_layout = QVBoxLayout()
        plt_btn = QPushButton('绘图')
        plt_btn.setFixedSize(100, 30)
        plt_btn.clicked.connect(self.calculate_thd)
        btn_layout.addWidget(plt_btn)
        harmonic_slider_layout.addWidget(scroll_area)
        harmonic_slider_layout.addStretch()
        harmonic_slider_layout.addLayout(btn_layout)
        return harmonic_slider_layout

    @staticmethod
    def get_label_text(value):
        if value == 2:
            return str(value)
        elif value >= 7:
            start = (value - 6) * 5 + 5
            end = (value - 6) * 5 + 10
            return f"{start}...{end}"
        else:
            return f"2...{value}"

    def calculate_thd(self):
        freq_value, harmonic, thd = [], [], []
        if self.selected_harmonics:
            kwargs = {"harmonics": self.selected_harmonics}
            stimulus_signal = self.signal_info["stimulus_signal"]
            recorded_signal = self.signal_info["recorded_signal"]
            sample_rate = self.signal_info["sample_rate"]
            atfra = AudioThdFrequencyResponseAnalysis()
            if self.freq_dict is None and self.base_freq_list is None:
                self.freq_dict, self.base_freq_list = atfra.calculate_spectrum(stimulus_signal, sample_rate)
            freq_value, harmonic, thd = atfra.calculate_thd(self.freq_dict, self.base_freq_list, recorded_signal,
                                                            sample_rate, **kwargs)
        self.plot_graph(freq_value, thd)
        if isinstance("harmonic", np.ndarray):
            harmonic = harmonic.tolist()
        self.result = {"freq_value": freq_value,
                       "harmonic": harmonic,
                       "thd": thd}
        return self.result

    def plot_graph(self, freq_value, thd):
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

    def on_label_click(self, event, value, label):
        palette = label.palette()
        current_color = palette.color(label.backgroundRole()).name()
        background_color = self.palette().color(self.backgroundRole()).name()
        if self.selected_label:
            self.selected_label.setStyleSheet(f"background-color: {background_color}; border: none;")
        if current_color != '#add8e6':
            self.get_selected_harmonics(value)
            label.setStyleSheet("background-color: lightblue;")
            self.selected_label = label
        else:
            self.selected_harmonics = []
            label.setStyleSheet(f"background-color: {background_color}; border: none;")
            self.selected_label = None

    def get_selected_harmonics(self, value):
        if value >= 7:
            step = 5
            start = (value - step) * step - 1
            self.selected_harmonics = list(range(start, (start + step + 1)))
        else:
            self.selected_harmonics = list(range(1, value))


class Spl(QWidget):
    def __init__(self, signal_info):
        super().__init__()
        self.signal_info = signal_info
        self.result = {}
        self.init_ui()

    def init_ui(self):
        spl_box = self.spl_box()
        self.waveform_plot = pg.PlotWidget(title='Waveform')
        self.waveform_plot.setFixedSize(400, 200)
        self.waveform_plot.setBackground('white')
        self.spl_plot = pg.PlotWidget(title='Sound Pressure Level')
        self.spl_plot.setFixedSize(400, 200)
        self.spl_plot.setBackground('white')
        layout = QVBoxLayout()
        layout.addWidget(spl_box)
        layout.addWidget(self.waveform_plot)
        layout.addWidget(self.spl_plot)
        self.setLayout(layout)

    def spl_box(self):
        spl_box = QGroupBox("声压级")
        spl_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        ref_pre_box_label = QLabel("参考声压:")
        ref_pre_box = QLineEdit()
        ref_pre_box.setText("20µPa")
        ref_pre_box.setAlignment(Qt.AlignCenter)
        ref_pre_box.setReadOnly(True)
        ref_pre_box.setFixedSize(100, 20)
        h_spacer_1 = QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        plt_btn = QPushButton('绘图')
        plt_btn.setFixedSize(100, 25)
        plt_btn.clicked.connect(self.calculate_spl)
        spl_box_layout = QHBoxLayout()
        spl_box_layout.addWidget(ref_pre_box_label)
        spl_box_layout.addWidget(ref_pre_box)
        spl_box_layout.addItem(h_spacer_1)
        spl_box_layout.addWidget(plt_btn)
        spl_box.setLayout(spl_box_layout)
        return spl_box

    def calculate_spl(self):
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
        self.waveform_plot.clear()
        self.spl_plot.clear()
        self.waveform_plot.plot(signal_duration, recorded_signal, pen='b')
        self.waveform_plot.setLabel('left', 'Amplitude')
        self.waveform_plot.setLabel('bottom', 'Time (s)')
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
        frequency_response_box = self.frequency_response_box()
        self.fr_plot = pg.PlotWidget(title='Frequency Response')
        self.fr_plot.setFixedSize(400, 320)
        self.fr_plot.setBackground('white')
        layout = QVBoxLayout()
        layout.addWidget(frequency_response_box)
        layout.addWidget(self.fr_plot)
        self.setLayout(layout)

    def frequency_response_box(self):
        fr_box = QGroupBox("频响")
        fr_box.setFixedSize(400, 80)
        plt_btn = QPushButton('绘图')
        plt_btn.setFixedSize(100, 25)
        plt_btn.clicked.connect(self.calculate_fr)
        fr_box_layout = QHBoxLayout()
        fr_box_layout.addWidget(plt_btn)
        fr_box.setLayout(fr_box_layout)
        return fr_box

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
    window = SignalAnalysisWindow(signal_info)
    window.show()
    window.exec()
