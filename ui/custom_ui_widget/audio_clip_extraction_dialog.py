import sys

import librosa
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor, QIcon
from PyQt5.QtWidgets import (
    QDialog, QPushButton, QHBoxLayout, QLineEdit, QToolTip, QVBoxLayout, QCheckBox,
    QDoubleSpinBox, QApplication, QFileDialog, QMessageBox
)
import pyqtgraph as pg

from base.file_ops import FileOps
from base.save_data import save_audio_simple
from consts.running_consts import DEFAULT_DIR
from ui.graph_widget import DraggablePlotWidget


class AudioClipExtractionDialog(QDialog):

    def __init__(self,
                 source_file=None,
                 source_audio=None,
                 clip_len=None,
                 sample_rate=None,
                 save_clip=False,
                 save_clip_path=None,
                 dialog_title="提取音频片段",
    ):
        super(AudioClipExtractionDialog, self).__init__()
        self.source_file = source_file
        self.source_audio = source_audio
        self.clip_len = clip_len
        self.sample_rate = sample_rate
        self.save_clip = save_clip
        self.save_clip_path = save_clip_path
        self.dialog_title = dialog_title

        self.audio_data = None
        self.selected_region_time = (None, None)
        self.return_value = (None, None, None)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.dialog_title)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        self.open_file_layout = self.create_open_file_layout()
        self.plot_widget = self.create_plot_widget()
        self.option_layout = self.create_option_layout()

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(self.open_file_layout)
        main_layout.addWidget(self.plot_widget)
        main_layout.addLayout(self.option_layout)
        main_layout.addLayout(self.create_btn_layout())

        self.setLayout(main_layout)

    def create_open_file_layout(self):
        layout = QHBoxLayout()
        open_file_btn = QPushButton("加载音频")
        open_file_btn.clicked.connect(self.open_audio_file)

        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setPlaceholderText("请上传源音频文件...")
        layout.addWidget(open_file_btn)
        layout.addWidget(self.file_path_edit)
        return layout

    def create_plot_widget(self):
        self.plot_curve = pg.PlotDataItem(pen='k')
        self.region = pg.LinearRegionItem(values=[0, 0], brush=(50, 150, 250, 50),
                                          pen={'color': (0, 0, 255), 'width': 2})
        self.region.setZValue(10)
        self.region.sigRegionChanged.connect(self.on_region_changed)

        plot_widget = DraggablePlotWidget(region_item=self.region)
        plot_widget.setBackground("white")
        plot_widget.setLabel("left", "Amplitude(V)", **{"font-size": "20px"})
        plot_widget.setLabel("bottom", "Time(s)", **{"font-size": "20px"})
        plot_widget.showGrid(x=True, y=True, alpha=0.5)
        plot_widget.addItem(self.plot_curve)
        plot_widget.addItem(self.region)
        self.region.hide()

        plot_widget.sigSelectionCancelled.connect(self.on_selection_cancelled)
        plot_widget.viewport().setMouseTracking(True)
        plot_widget.viewport().installEventFilter(self)
        plot_widget.setActive(False)
        return plot_widget

    def create_option_layout(self):
        self.fixed_len_checkbox = QCheckBox("固定片段时长")
        self.clip_len_spinbox = QDoubleSpinBox()
        self.clip_len_spinbox.setRange(0.01, 10)
        self.clip_len_spinbox.setDecimals(3)
        self.clip_len_spinbox.setSingleStep(0.1)
        self.clip_len_spinbox.setValue(0.1)

        if self.clip_len is not None:
            self.fixed_len_checkbox.setChecked(True)
        self.fixed_len_checkbox.stateChanged.connect(self.change_fixed_len_box_state)

        layout = QHBoxLayout()
        layout.addWidget(self.fixed_len_checkbox)
        layout.addWidget(self.clip_len_spinbox)
        return layout

    def create_btn_layout(self):
        layout = QHBoxLayout()
        ok_btn = QPushButton("确认")
        ok_btn.clicked.connect(self.on_click_ok_btn)
        layout.addStretch()
        layout.addWidget(ok_btn)
        return layout

    def open_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择音频文件", "", "音频文件 (*.wav)")
        if file_path:
            self.source_file = file_path
            self.file_path_edit.setText(file_path)
            self.load_and_display_waveform(file_path)
            self.region.hide()
            self.selected_region_time = (None, None)

    def load_and_display_waveform(self, file_path):
        try:
            audio_data, sample_rate = librosa.load(file_path, sr=self.sample_rate, mono=True)
            self.audio_data, self.sample_rate = audio_data, sample_rate
            time_array = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
            self.plot_curve.setData(time_array, audio_data)
            self.region.setBounds([0, time_array[-1]])
            self.plot_widget.setActive(True)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载音频文件失败:\n{e}")
            self.audio_data, self.sample_rate = None, None
            self.plot_curve.clear()
            self.plot_widget.setActive(False)

    def on_region_changed(self):
        start_time, end_time = self.region.getRegion()
        duration = abs(end_time - start_time)
        if self.region.isVisible() and duration > 1e-9:
            self.selected_region_time = tuple(sorted((start_time, end_time)))
            tooltip_text = f"时长: {duration:.3f}s"
            QToolTip.showText(QCursor.pos(), tooltip_text, self.plot_widget.viewport())
        else:
            self.selected_region_time = (None, None)
            QToolTip.hideText()

    def on_selection_cancelled(self):
        self.selected_region_time = (None, None)
        QToolTip.hideText()

    def change_fixed_len_box_state(self):
        self.region.hide()
        self.selected_region_time = (None, None)

    def on_click_ok_btn(self):
        if self.selected_region_time != (None, None):
            start_time, end_time = self.selected_region_time
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            clip_data = self.audio_data[start_sample:end_sample]

            relative_path = self.save_clip_path
            if self.save_clip:
                if not self.save_clip_path:
                    self.save_clip_path, _ = QFileDialog.getSaveFileName(self, "保存片段", "", "WAV 文件 (*.wav)")

                relative_path = FileOps.get_relative_path(self.save_clip_path, DEFAULT_DIR)
                save_audio_simple(self.save_clip_path, clip_data, self.sample_rate)

            self.accept()
            self.return_value = (clip_data, relative_path, len(clip_data))
            self.close()
        else:
            QMessageBox.warning(self, "提示", "请选择音频片段")

    def on_exec(self):
        self.exec()
        return self.return_value


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioClipExtractionDialog(save_clip=True)
    window.show()
    app.exec_()
