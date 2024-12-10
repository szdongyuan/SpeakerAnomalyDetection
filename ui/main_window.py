import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QAction

from base.log_manager import LogManager
from ui.calibaration_window import CalibrationWindow
from ui.hardware_window import HardwareWindow, get_default_device
from ui.login_window import LoginWindow, AddAccountWindow, ChangePwdWindow
from ui.ai_window import AiWindow
from ui.stimulus_window import StimulusWindow


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.user_name = None
        self.access_lvl = None
        self.stimulus = None
        self.stimulus_info = None
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
        self.widget_list_operator = [self.user_action_switch_account]
        self.widget_list_engineer = self.widget_list_operator + [
            self.function_action_stimulus,
            self.function_action_test_sequence,
            self.function_action_ai_training,
            self.hardware_action_selection,
            self.hardware_action_calibration,
            self.user_action_change_pwd,
        ]
        self.widget_list_admin = self.widget_list_engineer + [self.user_action_add_account]

        self.init_ui()

    def init_ui(self):
        self.showMaximized()
        self.init_menu()

        self.on_access_lvl_changed()
        self.show()
        self.showMaximized()

        self.on_login_window_init()

    def init_menu(self):
        menu_bar = self.menuBar()
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
        stimulus_info, stimulus = dlg.on_exec()
        if stimulus is not None:
            self.stimulus_info = stimulus_info
            self.stimulus = stimulus

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
        self.access_lvl, self.user_name = dlg.on_exec()
        self.on_access_lvl_changed()
        print(self.access_lvl)

    @staticmethod
    def on_add_account_window_init():
        dlg = AddAccountWindow(LogManager.set_log_handler("core"))
        dlg.exec()

    def on_change_pwd_window_init(self):
        try:
            dlg = ChangePwdWindow(self.user_name, LogManager.set_log_handler("core"))
            dlg.exec()
        except Exception as e:
            print(e)

    def on_hardware_window_init(self):
        dlg = HardwareWindow()
        self.speaker, self.mic = dlg.on_exec()
        print(self.speaker, self.mic)

    @staticmethod
    def on_calibration_window_init():
        dlg = CalibrationWindow()
        dlg.exec()

    def on_window_close(self):
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
