import sys
import soundcard as sc
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter, QPixmap, QStandardItem, QStandardItemModel, QIcon
from PyQt5.QtWidgets import QApplication, QAbstractItemView, QDialog, QGroupBox, QHBoxLayout, QLabel, QListView
from PyQt5.QtWidgets import QPushButton, QSpacerItem, QSizePolicy, QVBoxLayout, QWizard, QWizardPage

from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.calibaration_window import CalibrationWindow


class HardwareWindow(QDialog):

    def __init__(self):
        super().__init__()
        self.speaker = get_default_device("speaker")
        self.mic = get_default_device("mic")

        self.init_ui()

    def init_ui(self):
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/DT_ico.ico"))
        self.setWindowTitle("硬件设置")
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)

        speaker_box = self.create_speaker_box()
        mic_box = self.create_mic_box()

        btn_layout = QHBoxLayout()
        h_spacer = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        ok_btn = QPushButton(" 确  定 ")
        ok_btn.clicked.connect(self.ok_btn_clicked)
        btn_layout.addItem(h_spacer)
        btn_layout.setContentsMargins(0, 0, 11, 0)
        btn_layout.addWidget(ok_btn)

        layout = QVBoxLayout()
        layout.addWidget(speaker_box)
        layout.addWidget(mic_box)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        self.setStyleSheet(ui_style_const.qpushbutton_stytle +
                           ui_style_const.qgroupbox_stytle)

    def create_speaker_box(self):
        speaker_label_layout = QVBoxLayout()
        self.speaker_label = QLabel("设  备： %s" % self.speaker.name)
        self.speaker_channel_label = QLabel("通道数： %s" % self.speaker.channels)
        speaker_label_layout.addWidget(self.speaker_label)
        speaker_label_layout.addWidget(self.speaker_channel_label)

        speaker_btn_layout = QHBoxLayout()
        select_speaker_btn = QPushButton("选择扬声器")
        select_speaker_btn.clicked.connect(self.select_speaker_btn_clicked)
        h_spacer = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        calibrate_speaker_btn = QPushButton("输出校准")
        calibrate_speaker_btn.clicked.connect(self.calibrate_speaker_btn_clicked)
        speaker_btn_layout.addWidget(select_speaker_btn)
        speaker_btn_layout.addItem(h_spacer)
        speaker_btn_layout.addWidget(calibrate_speaker_btn)

        speaker_box = QGroupBox("扬声器")
        layout = QVBoxLayout()
        layout.addLayout(speaker_label_layout)
        layout.addLayout(speaker_btn_layout)
        speaker_box.setLayout(layout)
        return speaker_box

    def create_mic_box(self):
        mic_label_layout = QVBoxLayout()
        self.mic_label = QLabel("设  备： %s" % self.mic.name)
        self.mic_channel_label = QLabel("通道数： %s" % self.mic.channels)
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
            self.speaker_label.setText("设  备：   %s" % self.speaker.name)
            self.speaker_channel_label.setText("通道数： %s" % self.speaker.channels)

    def select_mic_btn_clicked(self):
        dlg = DeviceListWindow("mic")
        selected_mic = dlg.on_exec()
        if selected_mic:
            self.mic = selected_mic
            self.mic_label.setText("设  备：   %s" % self.mic.name)
            self.mic_channel_label.setText("通道数： %s" % self.mic.channels)

    def calibrate_speaker_btn_clicked(self):
        dlg = CalibrationWizard()
        dlg.on_exec()
        dlg2 = CalibrationWindow()
        dlg2.speaker = self.speaker
        dlg2.exec()

    def ok_btn_clicked(self):
        self.close()

    def on_exec(self):
        self.exec()
        return self.speaker, self.mic

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QColor(174, 171, 162, 123))
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())
        super().paintEvent(event)


class DeviceListWindow(QDialog):

    def __init__(self, device_type):
        super().__init__()
        if device_type == "speaker":
            self.device_list = sc.all_speakers()
            self.device_title = " —— 扬声器"
        elif device_type == "mic":
            self.device_list = sc.all_microphones()
            self.device_title = " —— 麦克风"
        else:
            self.device_list = []
            self.device_title = ""
        self.selected_device = None

        self.init_ui()

    def init_ui(self):
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/DT_ico.ico"))
        self.setWindowTitle("选择设备%s" % self.device_title)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)

        list_view = QListView()
        list_view.setSelectionMode(QAbstractItemView.SingleSelection)
        list_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        item_model = QStandardItemModel()
        for device in self.device_list:
            item_model.appendRow(QStandardItem(device.name))

        list_view.setModel(item_model)
        list_view.setSelectionRectVisible(True)
        list_view.clicked.connect(self.on_select_item)

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton(" 确  认 ")
        ok_btn.clicked.connect(self.on_click_ok_btn)
        cancel_btn = QPushButton(" 取  消 ")
        cancel_btn.clicked.connect(self.on_click_cancel_btn)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        btn_layout.setSpacing(65)
        btn_layout.setContentsMargins(40, 0, 40, 0)

        layout = QVBoxLayout()
        layout.addWidget(list_view)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

        self.setStyleSheet(ui_style_const.qpushbutton_stytle)

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

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QColor(174, 171, 162, 123))
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())
        super().paintEvent(event)


class CalibrationWizard(QWizard):

    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/DT_ico.ico"))
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWizardStyle(QWizard.ModernStyle)
        self.setWindowTitle("输出校准向导")

        # Todo: add pic to wizard
        page_1 = self.create_wizard_page(title="步骤一：连接设备",
                                         label_txt="将功放输出端正确连接至电压表或示波仪。",
                                         wizard_pic=DEFAULT_DIR + "ui/ui_pic/calibration_pic/fig1_device.png")
        page_2 = self.create_wizard_page(title="步骤二：播放激励信号",
                                         label_txt="点击“播放”按钮，观察电压读数。",
                                         wizard_pic=DEFAULT_DIR + "ui/ui_pic/calibration_pic/fig2_play.png")
        page_3 = self.create_wizard_page(title="步骤三：记录电压",
                                         label_txt="待电压稳定后，记录读数。重复步骤二、三若干次（建议5次以上）。",
                                         wizard_pic=DEFAULT_DIR + "ui/ui_pic/calibration_pic/fig3_save.png")
        page_4 = self.create_wizard_page(title="步骤四：校准输出",
                                         label_txt="点击“校准”按钮。",
                                         wizard_pic=DEFAULT_DIR + "ui/ui_pic/calibration_pic/fig4_calibrate.png")
        page_5 = self.create_wizard_page(title="步骤五：测试",
                                         label_txt="点击“测试”按钮，若电压读数与预期差距较大可点击“重置”按钮重新校准。\n校准完成后点击“退出”即可。",
                                         wizard_pic=DEFAULT_DIR + "ui/ui_pic/calibration_pic/fig5_test.png")
        page_list = [page_1, page_2, page_3, page_4, page_5]
        for i, page in enumerate(page_list):
            self.setPage(i, page)
        pix = QPixmap(640, 64)
        pix.fill(QColor(52, 104, 192))
        self.setPixmap(QWizard.BannerPixmap, pix)

        self.setButtonText(QWizard.NextButton, '下一步')
        self.setButtonText(QWizard.BackButton, '上一步')
        self.setButtonText(QWizard.FinishButton, '校  准')
        # self.setButtonText(QWizard.CancelButton, '跳过')
        self.setOption(QWizard.NoCancelButton)
        self.setStyleSheet(ui_style_const.qpushbutton_stytle)

    @staticmethod
    def create_wizard_page(title, subtitle=" ", label_txt=" ", wizard_pic=None):
        layout = QVBoxLayout()
        layout.addWidget(QLabel(label_txt))
        if wizard_pic:
            pic_layout = QHBoxLayout()
            pic_label = QLabel()
            pic_label.setPixmap(QPixmap(wizard_pic))
            pic_layout.addWidget(pic_label)
            pic_layout.setAlignment(Qt.AlignCenter)
            layout.addLayout(pic_layout)
        page = QWizardPage()
        page.setTitle("<font color='white' size='6'>%s</font>" % title)
        page.setSubTitle(subtitle)
        page.setLayout(layout)
        return page

    def on_exec(self):
        self.exec()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QColor(174, 171, 162, 123))
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())
        super().paintEvent(event)

def get_default_device(device):
    if device == "mic":
        return sc.default_microphone()
    elif device == "speaker":
        return sc.default_speaker()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HardwareWindow()
    # window = DeviceListWindow("speaker")
    # window = CalibrationWizard()
    window.show()
    result = window.on_exec()
