import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor
from PyQt5.QtWidgets import QAction, QApplication, QLabel, QMainWindow, QStatusBar, QWidget, QVBoxLayout, QHBoxLayout, \
    QSpacerItem, QSizePolicy, QPushButton, QMenuBar

from base.log_manager import LogManager
from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.ai_window import AiWindow
from ui.hardware_window import HardwareWindow, get_default_device
from ui.login_window import AddAccountWindow, ChangePwdWindow, LoginWindow
from ui.calibration_window import CalibrationWindow
from ui.sequence_widget import SequenceWindow
from ui.stimulus_window import StimulusWindow


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.user_name = None
        self.access_lvl = None
        self.refresh_stimulus_flag = None
        self.mic = get_default_device("mic")
        self.speaker = get_default_device("speaker")

        self.function_action_stimulus = QAction("激励信号", self)
        self.function_action_test_sequence = QAction("测试程序", self)
        self.function_action_ai_training = QAction("训练AI模型", self)
        self.function_action_exit = QAction("退出", self)
        self.hardware_action_selection = QAction("硬件选择", self)
        self.hardware_action_calibration = QAction("校准", self)
        self.user_action_switch_account = QAction("切换用户", self)
        self.user_action_add_account = QAction("添加用户", self)
        self.user_action_change_pwd = QAction("修改密码", self)
        self.widget_list_operator = [self.user_action_change_pwd]
        self.widget_list_engineer = self.widget_list_operator + [
            self.function_action_stimulus,
            self.function_action_test_sequence,
            self.function_action_ai_training,
            self.hardware_action_selection,
            self.hardware_action_calibration,
        ]
        self.widget_list_admin = self.widget_list_engineer + [self.user_action_add_account]

        self.init_ui()

    def init_ui(self):
        self.set_title()
        self.init_menu()
        self.init_sequence_widget()
        self.sequence_window.close()
        self.on_access_lvl_changed()
        self.show_statusbar_layout()
        self.showMaximized()
        self.on_login_window_init()

    def set_title(self):
        self.setWindowFlags(Qt.FramelessWindowHint)
        title_layout = QHBoxLayout()
        title_btn_layout = self.set_title_btn()
        icon_label = QLabel()
        icon_label.setStyleSheet("background-color: transparent")
        title_icon = QPixmap(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico")
        icon_label.setPixmap(title_icon)
        icon_label.setFixedSize(25, 25)
        icon_label.setScaledContents(True)
        title_label = QLabel("谛听异音检测 -0.12 beta")
        h_spacer = QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        title_layout.addWidget(icon_label)
        title_layout.addWidget(title_label)
        title_layout.addItem(h_spacer)
        title_layout.addLayout(title_btn_layout)
        self.setMinimumSize(1030, 760)
        title_layout.setContentsMargins(3, 3, 5, 0)
        self.setStyleSheet(ui_style_const.qlabel_stytle +
                           ui_style_const.qpushbutton_stytle)

        return (title_layout)

    def set_title_btn(self):
        self.min_btn = QPushButton()
        self.min_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/main_window_pic/minsize.svg"))
        self.min_btn.setStyleSheet("border: None; background-color: transparent")
        self.min_btn.clicked.connect(self.showMinimized)
        self.max_flag = True
        self.max_btn = QPushButton()
        self.max_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/main_window_pic/normalsize.svg"))
        self.max_btn.clicked.connect(self.show_window_size)
        self.max_btn.setStyleSheet("border: None; background-color: transparent")
        self.close_btn = QPushButton()
        self.close_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/main_window_pic/close.svg"))
        self.close_btn.setStyleSheet("border: None; background-color: transparent")
        self.close_btn.clicked.connect(self.close)

        title_btn_layout = QHBoxLayout()
        title_btn_layout.addWidget(self.min_btn)
        title_btn_layout.addWidget(self.max_btn)
        title_btn_layout.addWidget(self.close_btn)

        return title_btn_layout

    def show_window_size(self):
        if self.max_flag:
            self.max_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/main_window_pic/maxsize.svg"))
            self.showNormal()
            self.max_flag = False
        else:
            self.max_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/main_window_pic/normalsize.svg"))
            self.showMaximized()
            self.max_flag = True

    def init_sequence_widget(self):
        main_window = QWidget()
        layout = QVBoxLayout()
        self.sequence_window = SequenceWindow()
        menu_bar = self.init_menu()
        title_layout = self.set_title()
        layout.addLayout(title_layout)
        layout.addWidget(menu_bar)
        layout.addWidget(self.sequence_window)
        layout.setAlignment(Qt.AlignTop)
        layout.setContentsMargins(0, 0, 0, 0)
        main_window.setLayout(layout)
        self.setCentralWidget(main_window)
        self.sequence_window.mic = self.mic
        self.sequence_window.speaker = self.speaker

    def init_menu(self):
        menu_bar = QMenuBar()
        menu_bar.setStyleSheet(ui_style_const.main_window_menubar_stytle)
        function_menu = menu_bar.addMenu("功能")
        hardware_menu = menu_bar.addMenu("硬件")
        user_menu = menu_bar.addMenu("用户")
        help_menu = menu_bar.addMenu("帮助")

        function_menu.addAction(self.function_action_stimulus)
        self.function_action_stimulus.triggered.disconnect()
        self.function_action_stimulus.triggered.connect(self.on_stimulus_window_init)
        function_menu.addAction(self.function_action_test_sequence)
        self.function_action_test_sequence.triggered.disconnect()
        function_menu.addSeparator()
        function_menu.addAction(self.function_action_ai_training)
        self.function_action_ai_training.triggered.disconnect()
        self.function_action_ai_training.triggered.connect(self.on_ai_window_init)
        function_menu.addSeparator()

        function_menu.addAction(self.function_action_exit)
        self.function_action_exit.triggered.disconnect()
        self.function_action_exit.triggered.connect(self.on_window_close)
        hardware_menu.addAction(self.hardware_action_selection)
        self.hardware_action_selection.triggered.disconnect()
        self.hardware_action_selection.triggered.connect(self.on_hardware_window_init)
        hardware_menu.addAction(self.hardware_action_calibration)
        self.hardware_action_calibration.triggered.disconnect()
        self.hardware_action_calibration.triggered.connect(self.on_calibration_window_init)

        user_menu.addAction(self.user_action_switch_account)
        self.user_action_switch_account.triggered.disconnect()
        self.user_action_switch_account.triggered.connect(self.on_login_window_init)
        user_menu.addAction(self.user_action_add_account)
        self.user_action_add_account.triggered.disconnect()
        self.user_action_add_account.triggered.connect(self.on_add_account_window_init)
        user_menu.addAction(self.user_action_change_pwd)
        self.user_action_change_pwd.triggered.disconnect()
        self.user_action_change_pwd.triggered.connect(self.on_change_pwd_window_init)

        return menu_bar

    def on_stimulus_window_init(self):
        dlg = StimulusWindow()
        dlg.speaker = self.speaker
        self.refresh_stimulus_flag = dlg.on_exec()
        self.sequence_window.refresh_stimulus_flag = self.refresh_stimulus_flag

    def show_statusbar_layout(self):
        self.user_label = QLabel()
        self.user_label.setAlignment(Qt.AlignLeft)
        self.user_label.setStyleSheet(ui_style_const.qlabel_stytle)
        self.user_label.setText("当前用户：{name}  用户等级：{level}".format(name=self.user_name, level=self.access_lvl))
        self.device_label = QLabel()
        self.device_label.setStyleSheet(ui_style_const.qlabel_stytle)
        device_txt = "麦克风：{mic}  扬声器：{speaker}".format(mic=self.mic["name"], speaker=self.speaker["name"])
        self.device_label.setText(device_txt)

        statusbar = QStatusBar()
        statusbar.addWidget(self.user_label)
        statusbar.addPermanentWidget(self.device_label)
        self.setStatusBar(statusbar)

    def update_statusbar(self):
        device_txt = "麦克风：{mic}  扬声器：{speaker}".format(mic=self.mic["name"], speaker=self.speaker["name"])
        self.device_label.setText(device_txt)
        self.user_label.setText("当前用户：{name}  用户等级：{level}".format(name=self.user_name, level=self.access_lvl))

    @staticmethod
    def on_ai_window_init():
        dlg = AiWindow()
        dlg.exec()

    def on_access_lvl_changed(self):
        widget_dict = {"Operator": self.widget_list_operator,
                       "Engineer": self.widget_list_engineer,
                       "Admin": self.widget_list_admin}
        for widget in self.widget_list_admin:
            widget.setDisabled(True)
        for widget in widget_dict.get(self.access_lvl, []):
            widget.setEnabled(True)

    def on_login_window_init(self):
        dlg = LoginWindow()
        access_lvl, user_name = dlg.on_exec()
        if access_lvl is not None:
            self.access_lvl, self.user_name = access_lvl, user_name
            self.sequence_window.show()
            self.update_statusbar()
        self.on_access_lvl_changed()

    @staticmethod
    def on_add_account_window_init():
        dlg = AddAccountWindow(LogManager.set_log_handler("core"))
        dlg.exec()

    def on_change_pwd_window_init(self):
        dlg = ChangePwdWindow(self.user_name, LogManager.set_log_handler("core"))
        dlg.exec()

    def on_hardware_window_init(self):
        dlg = HardwareWindow()
        self.speaker, self.mic = dlg.on_exec()
        self.update_statusbar()
        self.sequence_window.mic = self.mic
        self.sequence_window.speaker = self.speaker

    def on_calibration_window_init(self):
        dlg = CalibrationWindow()
        dlg.speaker = self.speaker
        dlg.exec()

    def on_window_close(self):
        self.close()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            if not self.max_flag:
                self.move(event.globalPos() - self.drag_position)
            event.accept()

    def paintEvent(self, event):
        # Set the window Background-color
        painter = QPainter(self)
        width = self.width()
        height = self.height()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(208, 206, 202))
        painter.drawRect(0, 0, width, 31)
        painter.setBrush(QColor(208, 206, 202, 124))
        painter.drawRect(0, 31, width, 41)
        painter.drawRect(0, height - 24, width, 24)
        painter.end()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
