import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QLabel, QSizePolicy, QSpacerItem, QHBoxLayout, \
    QStatusBar, QWidget

from base.log_manager import LogManager
from consts import ui_style_const
from ui.calibaration_window import CalibrationWindow
from ui.hardware_window import HardwareWindow, get_default_device
from ui.login_window import LoginWindow, AddAccountWindow, ChangePwdWindow
from ui.ai_window import AiWindow
from ui.sequence_widget import SequenceWindow
from ui.stimulus_window import StimulusWindow


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.user_name = None
        self.access_lvl = None
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
        self.showMaximized()
        self.init_menu()
        self.init_sequence_widget()
        self.on_access_lvl_changed()
        self.show()

        self.on_login_window_init()

    def init_sequence_widget(self):
        self.sequence_window = SequenceWindow()
        self.setCentralWidget(self.sequence_window)
        self.sequence_window.mic = self.mic
        self.sequence_window.speaker = self.speaker

    def init_menu(self):
        menu_bar = self.menuBar()
        self.setStyleSheet(ui_style_const.main_window_menubar_stytle)
        function_menu = menu_bar.addMenu("功能")
        hardware_menu = menu_bar.addMenu("硬件")
        user_menu = menu_bar.addMenu("用户")
        help_menu = menu_bar.addMenu("帮助")

        function_menu.addAction(self.function_action_stimulus)
        self.function_action_stimulus.triggered.connect(self.on_stimulus_window_init)
        function_menu.addAction(self.function_action_test_sequence)
        function_menu.addSeparator()
        function_menu.addAction(self.function_action_ai_training)
        self.function_action_ai_training.triggered.connect(self.on_ai_window_init)
        function_menu.addSeparator()

        function_menu.addAction(self.function_action_exit)
        self.function_action_exit.triggered.connect(self.on_window_close)
        hardware_menu.addAction(self.hardware_action_selection)
        self.hardware_action_selection.triggered.connect(self.on_hardware_window_init)
        hardware_menu.addAction(self.hardware_action_calibration)
        self.hardware_action_calibration.triggered.connect(self.on_calibration_window_init)

        user_menu.addAction(self.user_action_switch_account)
        self.user_action_switch_account.triggered.connect(self.on_login_window_init)
        user_menu.addAction(self.user_action_add_account)
        self.user_action_add_account.triggered.connect(self.on_add_account_window_init)
        user_menu.addAction(self.user_action_change_pwd)
        self.user_action_change_pwd.triggered.connect(self.on_change_pwd_window_init)

    def on_stimulus_window_init(self):
        dlg = StimulusWindow()
        dlg.speaker = self.speaker
        dlg.on_exec()

    def show_statusbar_layout(self):
        statusbar_widget = QWidget()
        statusbar_widget.setMinimumWidth(self.width())
        self.user_label = QLabel()
        self.user_label.setText("当前用户：{name}  用户等级：{level}".format(name=self.user_name, level=self.access_lvl))
        self.device_label = QLabel()
        self.device_label.setText("麦克风：{mic}  扬声器：{speaker}".format(mic=self.mic.name, speaker=self.speaker.name))
        h_spacer = QSpacerItem(30, 30, QSizePolicy.Expanding, QSizePolicy.Minimum)

        statusbar_layout = QHBoxLayout()
        statusbar_layout.addWidget(self.user_label)
        statusbar_layout.addItem(h_spacer)
        statusbar_layout.addWidget(self.device_label)
        statusbar_widget.setLayout(statusbar_layout)

        statusbar = QStatusBar()
        statusbar.addWidget(statusbar_widget)
        self.setStatusBar(statusbar)
        self.setStyleSheet(ui_style_const.qlabel_stytle)

    def update_statusbar(self):
        self.device_label.setText("麦克风：{mic}  扬声器：{speaker}".format(mic=self.mic.name, speaker=self.speaker.name))
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
        self.login_falgs = True
        dlg = LoginWindow()
        access_lvl, user_name = dlg.on_exec()
        if access_lvl is not None:
            self.access_lvl, self.user_name = access_lvl, user_name
        self.on_access_lvl_changed()
        if self.login_falgs:
            self.login_falgs = False
            self.show_statusbar_layout()
        else:
            self.update_statusbar()

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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
