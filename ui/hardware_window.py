import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItem, QStandardItemModel, QIcon
from PyQt5.QtWidgets import QApplication, QAbstractItemView, QDialog, QGroupBox, QHBoxLayout, QLabel, QListView, QFrame
from PyQt5.QtWidgets import QPushButton, QSpacerItem, QSizePolicy, QVBoxLayout, QComboBox

from base.sound_device_manager import SoundDeviceManager
from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR

class HardwareWindow(QDialog):

    def __init__(self, current_speaker=None, current_mic=None):
        super().__init__()
        self.speaker = current_speaker
        self.mic = current_mic
        # save initial devices to allow cancel action(future use)
        self.initial_speaker = current_speaker
        self.initial_mic = current_mic

        self.init_ui()
        self.refresh_device_display()

    def init_ui(self):
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setWindowTitle("硬件设置")
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        speaker_box = self.create_speaker_box()
        mic_box = self.create_mic_box()
        btn_layout = QHBoxLayout()
        h_spacer = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        refresh_btn = QPushButton(" 刷  新 ")
        refresh_btn.clicked.connect(self.refresh_device_display)
        ok_btn = QPushButton(" 确  定 ")
        ok_btn.clicked.connect(self.ok_btn_clicked)
        btn_layout.addWidget(refresh_btn)
        btn_layout.addItem(h_spacer)
        btn_layout.setContentsMargins(0, 0, 11, 0)
        btn_layout.addWidget(ok_btn)
        layout = QVBoxLayout()
        layout.addWidget(speaker_box)
        layout.addWidget(mic_box)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.setStyleSheet(ui_style_const.qpushbutton_style +
                           ui_style_const.qgroupbox_style +
                           ui_style_const.qgroupbox_style +
                           ui_style_const.qlabel_style)

    def create_speaker_box(self):
        speaker_label_layout = QVBoxLayout()
        self.speaker_label = QLabel(" " * 30)
        self.speaker_channel_label = QLabel(" " * 30)

        speaker_label_layout.addWidget(self.speaker_label)
        speaker_label_layout.addWidget(self.speaker_channel_label)

        speaker_btn_layout = QHBoxLayout()
        select_speaker_btn = QPushButton("选择扬声器")
        select_speaker_btn.clicked.connect(self.select_speaker_btn_clicked)
        h_spacer = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        # calibrate_speaker_btn = QPushButton("输出校准")
        # calibrate_speaker_btn.clicked.connect(self.calibrate_speaker_btn_clicked)
        speaker_btn_layout.addWidget(select_speaker_btn)
        speaker_btn_layout.addItem(h_spacer)
        # speaker_btn_layout.addWidget(calibrate_speaker_btn)
        speaker_box = QGroupBox("扬声器")
        layout = QVBoxLayout()
        layout.addLayout(speaker_label_layout)
        layout.addLayout(speaker_btn_layout)
        speaker_box.setLayout(layout)
        return speaker_box

    def create_mic_box(self):
        mic_label_layout = QVBoxLayout()
        self.mic_label = QLabel(" " * 30)
        self.mic_channel_label = QLabel(" " * 30)

        mic_label_layout.addWidget(self.mic_label)
        mic_label_layout.addWidget(self.mic_channel_label)

        mic_btn_layout = QHBoxLayout()
        select_mic_btn = QPushButton("选择麦克风")
        select_mic_btn.clicked.connect(self.select_mic_btn_clicked)
        h_spacer = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        mic_btn_layout.addWidget(select_mic_btn)
        mic_btn_layout.addItem(h_spacer)
        mic_box = QGroupBox("麦克风")
        layout = QVBoxLayout()
        layout.addLayout(mic_label_layout)
        layout.addLayout(mic_btn_layout)
        mic_box.setLayout(layout)
        return mic_box

    def select_speaker_btn_clicked(self):
        dlg = DeviceListWindow("speaker")
        selected_speaker = dlg.on_exec()
        if selected_speaker:
            self.speaker = selected_speaker
            self.speaker_label.setText("设  备：   %s" % self.speaker["name"])
            self.speaker_channel_label.setText(
                "驱  动：   %s" % SoundDeviceManager().get_api_info(self.speaker["hostapi"])["name"]
            )

    def select_mic_btn_clicked(self):
        dlg = DeviceListWindow("mic")
        selected_mic = dlg.on_exec()
        if selected_mic:
            self.mic = selected_mic
            self.mic_label.setText("设  备：   %s" % self.mic["name"])
            self.mic_channel_label.setText(
                "驱  动：   %s" % SoundDeviceManager().get_api_info(self.mic["hostapi"])["name"]
            )

    def refresh_device_display(self):
        sdm = SoundDeviceManager()

        last_speaker = self.speaker
        last_mic = self.mic
        sdm.refresh_available_device()
        _, default_speaker = sdm.get_default_device("speaker", refresh=False)
        _, default_mic = sdm.get_default_device("mic", refresh=False)
        all_devices = sdm.get_device_info()
        speaker_list = []
        mic_list = []
        ## Because there are unusable devices in all_device (no API), get the devices with API
        for api in all_devices:
            speaker_list.extend(all_devices[api]["output"])
            mic_list.extend(all_devices[api]["input"])

        if not speaker_list:
            self.speaker = None
        else:
            if last_speaker and any(d['name'] == last_speaker['name'] for d in speaker_list):
                self.speaker = next(
                    (d for d in speaker_list
                    if d['name'] == last_speaker['name'] and d['hostapi'] == last_speaker['hostapi']),
                    default_speaker
                )
            else:
                self.speaker = default_speaker

        if not mic_list:
            self.mic = None
        else:
            if last_mic and any(d['name'] == last_mic['name'] for d in mic_list):
                self.mic = next(
                    (d for d in mic_list
                    if d['name'] == last_mic['name'] and d['hostapi'] == last_mic['hostapi']),
                    default_mic
                )
            else:
                self.mic = default_mic

        if not self.speaker:
            self.speaker_label.setText("设  备：   无可用输出设备")
            self.speaker_channel_label.setText("驱  动：   无可用输出设备")
        else:
            self.speaker_label.setText("设  备：   %s" % self.speaker["name"])
            self.speaker_channel_label.setText(
                "驱  动：   %s" % SoundDeviceManager().get_api_info(self.speaker["hostapi"])["name"]
            )

        if not self.mic:
            self.mic_label.setText("设  备：   无可用输入设备")
            self.mic_channel_label.setText("驱  动：   无可用输入设备")
        else:
            self.mic_label.setText("设  备：   %s" % self.mic["name"])
            self.mic_channel_label.setText(
                "驱  动：   %s" % SoundDeviceManager().get_api_info(self.mic["hostapi"])["name"]
            )

    def ok_btn_clicked(self):
        mic_idx = self.mic["index"] if self.mic else -1
        speaker_idx = self.speaker["index"] if self.speaker else -1
        SoundDeviceManager().change_default_device(mic_idx, speaker_idx)
        self.accept()

    def on_exec(self):
        result = self.exec()
        if result == QDialog.Accepted:
            return self.speaker, self.mic
        else:
            return self.initial_speaker, self.initial_mic


class DeviceListWindow(QDialog):

    def __init__(self, device_type):
        super().__init__()
        if device_type == "speaker":
            self.device_type = "output"
            self.device_title = " —— 扬声器"
        elif device_type == "mic":
            self.device_type = "input"
            self.device_title = " —— 麦克风"

        self.selected_device = None
        self.api_info = SoundDeviceManager().get_device_info()

        self.api_combo_box = QComboBox()
        self.list_view = QListView()
        self.init_ui()

    def init_ui(self):
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setWindowTitle("选择设备%s" % self.device_title)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        api_layout = QHBoxLayout()
        api_label = QLabel("选择驱动")
        self.api_combo_box.addItems([api for api in self.api_info])
        self.api_combo_box.currentTextChanged.connect(self.update_api_device)
        api_layout.addWidget(api_label)
        api_layout.addWidget(self.api_combo_box)
        self.list_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        item_model = QStandardItemModel()
        current_api = self.api_combo_box.currentText()
        self.device_list = self.api_info[current_api][self.device_type]
        for device in self.device_list:
            item_model.appendRow(QStandardItem(device["name"]))
        self.list_view.setModel(item_model)
        self.list_view.setSelectionRectVisible(True)
        self.list_view.clicked.connect(self.on_select_item)
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton(" 确  认 ")
        ok_btn.clicked.connect(self.on_click_ok_btn)
        cancel_btn = QPushButton(" 取  消 ")
        cancel_btn.clicked.connect(self.on_click_cancel_btn)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(ok_btn)
        btn_layout.setSpacing(105)
        btn_layout.setContentsMargins(30, 0, 30, 0)

        item_spacer = QSpacerItem(7, 15, QSizePolicy.Minimum, QSizePolicy.Minimum)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFixedHeight(20)
        line.setStyleSheet(ui_style_const.hardware_qframe_style)
        layout = QVBoxLayout()
        layout.addLayout(api_layout)
        layout.addWidget(line)
        layout.addWidget(self.list_view)
        layout.addItem(item_spacer)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.setStyleSheet(ui_style_const.qpushbutton_style +
                           ui_style_const.qlabel_style +
                           ui_style_const.qcombobox_style +
                           ui_style_const.qlistview_style)

    def update_api_device(self):
        item_model = QStandardItemModel()
        current_api = self.api_combo_box.currentText()
        self.device_list = self.api_info[current_api][self.device_type]
        for device in self.device_list:
            item_model.appendRow(QStandardItem(device["name"]))
        self.list_view.setModel(item_model)

    def on_select_item(self, index):
        self.selected_device = self.device_list[index.row()]

    def on_click_ok_btn(self):
        self.close()

    def on_click_cancel_btn(self):
        self.selected_device = None
        self.close()

    def on_exec(self):
        self.exec()
        return self.selected_device


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HardwareWindow()
    # window = DeviceListWindow("speaker")
    # window = CalibrationWizard()
    window.show()
    result = window.on_exec()
    print(result)
