import hashlib
import sys

from getmac import get_mac_address
from PyQt5.QtWidgets import QApplication, QDialog, QLineEdit, QLabel, QMessageBox
from PyQt5.QtWidgets import QPushButton, QComboBox
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout

from base.db_manager import DataSave
from consts import model_consts, error_code, running_consts


class LoginWindow(QDialog):

    def __init__(self, access_lvl=None):
        super().__init__()

        self.pwd_checked = False
        self.access_lvl = access_lvl

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("登录")
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()

        access_layout = QHBoxLayout()
        label_access = QLabel("权限")
        self.access_selection = QComboBox()
        self.access_selection.addItem("管理员")
        self.access_selection.addItem("工程师")
        self.access_selection.addItem("操作员")
        self.access_selection.currentTextChanged.connect(self.access_add_account)
        access_layout.addWidget(label_access)
        access_layout.addWidget(self.access_selection)

        user_layout = QHBoxLayout()
        label_user = QLabel("账号")
        self.username_input = QLineEdit()
        self.add_account_botton = QPushButton("添加账号")
        self.add_account_botton.clicked.connect(self.add_account_click)
        user_layout.addWidget(label_user)
        user_layout.addWidget(self.username_input)
        user_layout.addWidget(self.add_account_botton)

        pwd_layout = QHBoxLayout()
        label_pwd = QLabel("密码")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.change_pwd_botton = QPushButton("修改密码")
        self.change_pwd_botton.clicked.connect(self.change_pwd_click)
        pwd_layout.addWidget(label_pwd)
        pwd_layout.addWidget(self.password_input)
        pwd_layout.addWidget(self.change_pwd_botton)

        button_layout = QHBoxLayout()
        login_button = QPushButton("登录")
        login_button.clicked.connect(self.login_click)
        button_layout.addWidget(login_button)

        layout.addLayout(access_layout)
        layout.addLayout(user_layout)
        layout.addLayout(pwd_layout)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def access_add_account(self):
        if self.access_selection.currentText() != "管理员":
            self.add_account_botton.setDisabled(True)
        else:
            self.add_account_botton.setEnabled(True)

    def add_account_click(self):
        if self.check_credentials():
            dlg = AddAccountWindow()
            dlg.exec()
        else:
            QMessageBox.warning(self, "Error", "Username or Password is incorrect")

    def change_pwd_click(self):
        if self.check_credentials():
            dlg = ChangePwdWindow(self.username_input.text())
            dlg.exec()
        else:
            QMessageBox.warning(self, "Error", "Username or Password is incorrect")

    def login_click(self):
        if self.check_credentials():
            self.pwd_checked = True
            self.close()
        else:
            QMessageBox.warning(self, "Error", "Username or Password is incorrect")

    def check_credentials(self):
        username = self.username_input.text()
        password = self.password_input.text()
        self.access_lvl = running_consts.ACCESS_LVL_DICT[self.access_selection.currentText()]

        enc_pwd = encrypt_password(username, password)
        user_info = self.get_user_info_from_db(username)

        if user_info.get("access_level") == self.access_lvl and user_info.get("password") == enc_pwd:
            return True
        else:
            return False

    @staticmethod
    def get_user_info_from_db(user_name):
        query_clause_data = {"user_name": user_name}
        with DataSave(model_consts.DATABASE_PATH) as database:
            query_code, query_data = database.query("users_table",
                                                    ["user_name", "access_level", "password"],
                                                    query_clause_data)
        if query_code == error_code.OK and query_data:
            user_data = query_data[0]
            return {
                "user_name": user_data[0],
                "access_level": user_data[1],
                "password": user_data[2]
            }
        else:
            return {}

    def on_exec(self):
        self.exec()
        return self.access_lvl if self.pwd_checked else None


class AddAccountWindow(QDialog):

    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("添加账号")

        layout = QVBoxLayout()

        access_layout = QHBoxLayout()
        label_access = QLabel("权限")
        self.access_selection = QComboBox()
        self.access_selection.addItem("工程师")
        self.access_selection.addItem("操作员")
        access_layout.addWidget(label_access)
        access_layout.addWidget(self.access_selection)

        user_layout = QHBoxLayout()
        label_user = QLabel("新建账号")
        self.username_input = QLineEdit()
        user_layout.addWidget(label_user)
        user_layout.addWidget(self.username_input)

        pwd_layout = QHBoxLayout()
        label_pwd = QLabel("输入密码")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        pwd_layout.addWidget(label_pwd)
        pwd_layout.addWidget(self.password_input)

        info_layout = QHBoxLayout()
        self.info = QLabel()
        info_layout.addWidget(self.info)

        button_layout = QHBoxLayout()
        add_user_button = QPushButton("添加账号")
        add_user_button.clicked.connect(self.add_user_click)
        exit_button = QPushButton("退出")
        exit_button.clicked.connect(self.exit_click)
        button_layout.addWidget(add_user_button)
        button_layout.addWidget(exit_button)

        layout.addLayout(access_layout)
        layout.addLayout(user_layout)
        layout.addLayout(pwd_layout)
        layout.addLayout(info_layout)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def add_user_click(self):
        username = self.username_input.text()
        password = self.password_input.text()
        access_lvl = running_consts.ACCESS_LVL_DICT[self.access_selection.currentText()]
        enc_pwd = encrypt_password(username, password)
        if self.add_user_info_to_db(username, enc_pwd, access_lvl):
            self.info.setText("添加账号成功")
            self.username_input.clear()
            self.password_input.clear()
        else:
            self.info.setText("添加账号失败")

    @staticmethod
    def add_user_info_to_db(username, password, access_lvl):
        # Todo
        print(username, password, access_lvl)
        return False

    def exit_click(self):
        self.close()


class ChangePwdWindow(QDialog):

    def __init__(self, user_name):
        super().__init__()

        self.user_name = user_name
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("修改密码")

        layout = QVBoxLayout()

        info_layout = QHBoxLayout()
        self.info = QLabel("账号： " + self.user_name)
        info_layout.addWidget(self.info)

        pwd_layout = QHBoxLayout()
        label_pwd = QLabel("新建密码")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        pwd_layout.addWidget(label_pwd)
        pwd_layout.addWidget(self.password_input)

        confirm_pwd_layout = QHBoxLayout()
        label_confirm_pwd = QLabel("确认密码")
        self.confirm_password_input = QLineEdit()
        self.confirm_password_input.setEchoMode(QLineEdit.Password)
        confirm_pwd_layout.addWidget(label_confirm_pwd)
        confirm_pwd_layout.addWidget(self.confirm_password_input)

        button_layout = QHBoxLayout()
        change_pwd_button = QPushButton("修改密码")
        change_pwd_button.clicked.connect(self.change_pwd_click)
        button_layout.addWidget(change_pwd_button)

        layout.addLayout(info_layout)
        layout.addLayout(pwd_layout)
        layout.addLayout(confirm_pwd_layout)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def change_pwd_click(self):
        if self.password_input.text() != self.confirm_password_input.text():
            QMessageBox.warning(self, "Error", "两次输入的密码不一致")
        else:
            enc_pwd = encrypt_password(self.user_name, self.password_input.text())
            if self.change_pwd_in_db(self.user_name, enc_pwd):
                QMessageBox.information(self, "Success", "修改密码成功")
                self.close()
            else:
                QMessageBox.warning(self, "Error", "修改密码失败")

    @staticmethod
    def change_pwd_in_db(user_name, enc_pwd):
        # Todo
        print(user_name, enc_pwd)
        return True


def encrypt_password(user_name, password):
    mac_pwd = get_mac_address() + user_name + password
    sh = hashlib.sha1()
    sh.update(mac_pwd.encode("utf-8"))
    enc_pwd = sh.hexdigest()
    return enc_pwd


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LoginWindow()
    window.show()
    sys.exit(app.exec_())
