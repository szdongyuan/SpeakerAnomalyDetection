import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction, QApplication, QLabel, QMainWindow, QStatusBar

from base.log_manager import LogManager
from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.ai_window import AiWindow
from ui.calibaration_window import CalibrationWindow
from ui.hardware_window import HardwareWindow, get_default_device
from ui.login_window import AddAccountWindow, ChangePwdWindow, LoginWindow
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
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/DT_ico.ico"))
        self.setWindowTitle("谛听异音检测 -0.12 beta")
        self.setMinimumSize(1030, 760)

    def init_sequence_widget(self):
        self.sequence_window = SequenceWindow()
        self.setCentralWidget(self.sequence_window)

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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
