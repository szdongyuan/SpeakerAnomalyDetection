import hashlib
import sys

from getmac import get_mac_address
from PyQt5.QtWidgets import QApplication, QDialog, QLineEdit, QLabel, QMessageBox
from PyQt5.QtWidgets import QPushButton, QComboBox
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout


class LoginWindow(QDialog):
    def __init__(self, access_lvl=None):
        super().__init__()

        self.pwd_checked = False
        self.access_lvl = access_lvl

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Login Window")
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()

        access_layout = QHBoxLayout()
        label_access = QLabel("权限")
        self.access_selection = QComboBox()
        self.access_selection.addItem("管理员")
        self.access_selection.addItem("工程师")
        self.access_selection.addItem("操作员")
        self.access_selection.currentTextChanged.connect(self.access_select_change)
        self.setup_botton = QPushButton("设置")
        self.setup_botton.clicked.connect(self.setup_click)
        access_layout.addWidget(label_access)
        access_layout.addWidget(self.access_selection)
        access_layout.addWidget(self.setup_botton)

        user_layout = QHBoxLayout()
        label_user = QLabel("账号")
        self.username_input = QLineEdit()
        user_layout.addWidget(label_user)
        user_layout.addWidget(self.username_input)

        pwd_layout = QHBoxLayout()
        label_pwd = QLabel("密码")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        pwd_layout.addWidget(label_pwd)
        pwd_layout.addWidget(self.password_input)

        button_layout = QHBoxLayout()
        login_button = QPushButton("登录")
        login_button.clicked.connect(self.login_click)
        button_layout.addWidget(login_button)

        layout.addLayout(access_layout)
        layout.addLayout(user_layout)
        layout.addLayout(pwd_layout)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def access_select_change(self):
        if self.access_selection.currentText() != "管理员":
            self.setup_botton.setDisabled(True)
        else:
            self.setup_botton.setEnabled(True)

    def login_click(self):
        if self.check_credentials():
            self.pwd_checked = True
            self.close()
        else:
            QMessageBox.warning(self, "Error", "Username or Password is incorrect")

    def setup_click(self):
        if self.check_credentials():
            print("start setup procedure")
        else:
            QMessageBox.warning(self, "Error", "Username or Password is incorrect")

    def check_credentials(self):
        username = self.username_input.text()
        password = self.password_input.text()
        access_lvl_dict = {"管理员": "admin", "工程师": "engineer", "操作员": "operator"}
        self.access_lvl = access_lvl_dict[self.access_selection.currentText()]

        mac_pwd = get_mac_address() + password
        sh = hashlib.sha1()
        sh.update(mac_pwd.encode("utf-8"))
        enc_pwd = sh.hexdigest()
        user_info = self.get_user_info_from_db(username)

        if user_info.get("access_lvl") == self.access_lvl and user_info.get("password") == enc_pwd:
            return True
        else:
            return False

    # Todo
    @staticmethod
    def get_user_info_from_db(user_name):
        return {"user_name": "admin",
                "access_lvl": "operator",
                "password": "b7760302acfd1cd80cb3e22d3eeae9b3ae9cf238"}

    def on_exec(self):
        self.exec()
        return self.access_lvl if self.pwd_checked else None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LoginWindow()
    window.show()
    sys.exit(app.exec_())
