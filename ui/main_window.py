import os
import sys

from PyQt5.QtCore import Qt, QEventLoop, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QGroupBox, QLabel, QApplication, QComboBox, QVBoxLayout, QMessageBox, \
    QGridLayout, QLineEdit, QFileDialog, QMainWindow, QAction
from PyQt5.QtWidgets import QSpacerItem, QSizePolicy, QTextEdit, QWidget, QPushButton

from consts import error_code
from ui.login_window import LoginWindow


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.access_lvl = None

        self.function_action_stimulus = QAction("激励信号", self)
        self.function_action_test_sequence = QAction("测试程序", self)
        self.function_action_ai_training = QAction("训练AI模型", self)
        self.function_action_exit = QAction("退出", self)
        self.hardware_action_selection = QAction("硬件选择", self)
        self.hardware_action_calibration = QAction("校准", self)
        self.user_action_switch_account = QAction("切换用户", self)
        self.user_action_add_account = QAction("添加用户", self)
        self.user_action_change_pwd = QAction("修改密码", self)
        self.menu_list_operator = [self.user_action_switch_account]
        self.menu_list_engineer = self.menu_list_operator + [
            self.function_action_stimulus,
            self.function_action_test_sequence,
            self.function_action_ai_training,
            self.hardware_action_selection,
            self.hardware_action_calibration,
            self.user_action_change_pwd
        ]
        self.menu_list_admin = self.menu_list_engineer + [self.user_action_add_account]

        self.init_ui()

    def init_ui(self):
        self.init_menu()

        self.show()

        dlg = LoginWindow()
        self.access_lvl = dlg.on_exec()
        print(self.access_lvl)

    def init_menu(self):
        menu_bar = self.menuBar()
        function_menu = menu_bar.addMenu("功能")
        hardware_menu = menu_bar.addMenu("硬件")
        user_menu = menu_bar.addMenu("用户")
        help_menu = menu_bar.addMenu("帮助")

        function_menu.addAction(self.function_action_stimulus)
        function_menu.addAction(self.function_action_test_sequence)
        function_menu.addSeparator()
        function_menu.addAction(self.function_action_ai_training)
        function_menu.addSeparator()
        function_menu.addAction(self.function_action_exit)

        hardware_menu.addAction(self.hardware_action_selection)
        hardware_menu.addAction(self.hardware_action_calibration)

        user_menu.addAction(self.user_action_switch_account)
        user_menu.addAction(self.user_action_add_account)
        user_menu.addAction(self.user_action_change_pwd)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
