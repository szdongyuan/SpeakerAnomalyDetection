import hashlib
import sys

from PyQt5.QtGui import QFont, QIcon
from getmac import get_mac_address
from PyQt5.QtWidgets import QApplication, QDialog, QLineEdit, QLabel, QMessageBox, QSpacerItem, QSizePolicy
from PyQt5.QtWidgets import QPushButton, QComboBox
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout

from base.db_manager import DataSave
from base.log_manager import LogManager
from consts import model_consts, error_code, ui_style_const


ACCESS_LVL_DICT = {"管理员": "Admin", "工程师": "Engineer", "操作员": "Operator"}


class LoginWindow(QDialog):

  

    def __init__(self, access_lvl=None):
        super().__init__()

        self.pwd_checked = False
        self.access_lvl = access_lvl
        self.logger = LogManager.set_log_handler("core")
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("登录")
        self.setGeometry(100, 100, 300, 200)
        self.setMaximumSize(600, 400)
        self.setMinimumSize(300, 200)
        self.setWindowIcon(QIcon("./ui_pic/DT_ico.ico"))

        layout = QVBoxLayout()

        self.access_layout = QHBoxLayout()
        self.label_access = QLabel("权 限：")
        self.access_selection = QComboBox()
        self.access_selection.addItem("管理员")
        self.access_selection.addItem("工程师")
        self.access_selection.addItem("操作员")
        self.access_selection.currentTextChanged.connect(self.access_add_account)
        self.access_layout.addWidget(self.label_access)
        self.access_layout.addWidget(self.access_selection)
        self.access_selection.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        user_layout = QHBoxLayout()
        self.label_user = QLabel("账 号：")
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("请输入你的账号")
        self.add_account_botton = QPushButton("添加账号")
        self.add_account_botton.clicked.connect(self.add_account_click)
        user_layout.addWidget(self.label_user)
        user_layout.addWidget(self.username_input)
        user_layout.addWidget(self.add_account_botton)

        pwd_layout = QHBoxLayout()
        self.label_pwd = QLabel("密 码：")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入账号密码")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.change_pwd_botton = QPushButton("修改密码")
        self.change_pwd_botton.clicked.connect(self.change_pwd_click)
        pwd_layout.addWidget(self.label_pwd)
        pwd_layout.addWidget(self.password_input)
        pwd_layout.addWidget(self.change_pwd_botton)

        button_layout = QHBoxLayout()
        self.login_button = QPushButton(" 登  录 ")
        self. login_button.clicked.connect(self.login_click)
        h_spacer_login_i = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        button_layout.addItem(h_spacer_login_i)
        button_layout.addWidget(self.login_button)
        h_spacer_login_ii = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        button_layout.addItem(h_spacer_login_ii)

        layout.addLayout(self.access_layout)
        layout.addLayout(user_layout)
        layout.addLayout(pwd_layout)
        layout.addLayout(button_layout)
        layout.setContentsMargins(30, 20, 30, 5)

        self.setLayout(layout)

        self.setStyleSheet(ui_style_const.qcombobox_stytle +
                           ui_style_const.qpushbutton_stytle +
                           ui_style_const.qlineedit_stytle)

    def access_add_account(self):
        if self.access_selection.currentText() != "管理员":
            self.add_account_botton.setDisabled(True)
        else:
            self.add_account_botton.setEnabled(True)

    def add_account_click(self):
        if self.check_credentials():
            dlg = AddAccountWindow(self.logger)
            dlg.exec()
        else:
            QMessageBox.warning(self, "Error", "Username or Password is incorrect")

    def change_pwd_click(self):
        if self.check_credentials():
            dlg = ChangePwdWindow(self.username_input.text(), self.logger)
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
        self.access_lvl = ACCESS_LVL_DICT[self.access_selection.currentText()]

        enc_pwd = encrypt_password(username, password)
        user_info = self.get_user_info_from_db(username)

        if user_info.get("access_level") == self.access_lvl and user_info.get("password") == enc_pwd:
            return True
        else:
            self.logger.error("The password or access_level is incorrect")
            return False

    def resizeEvent(self, event):
        new_Dialog_font = QFont("SimSun", int(9 * (event.size().height() / 200)))
        self.label_access.setFont(new_Dialog_font)
        self.access_selection.setFont(new_Dialog_font)
        self.label_user.setFont(new_Dialog_font)
        self.username_input.setFont(new_Dialog_font)
        self.add_account_botton.setFont(new_Dialog_font)
        self.label_pwd.setFont(new_Dialog_font)
        self.password_input.setFont(new_Dialog_font)
        self.change_pwd_botton.setFont(new_Dialog_font)
        self.login_button.setFont(new_Dialog_font)
        self.access_layout.setContentsMargins(0, 0, int(self.add_account_botton.sizeHint().width()) + 6, 0)

    @staticmethod
    def get_user_info_from_db(user_name):
        with DataSave(model_consts.DATABASE_PATH) as database:
            query_code, query_data = database.query("users_table",
                                                    ["user_name", "access_level", "password"],
                                                    {"user_name": user_name})
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

    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("添加账号")
        self.setFixedSize(300, 200)
        layout = QVBoxLayout()

        access_layout = QHBoxLayout()
        label_access = QLabel("权    限：")
        self.access_selection = QComboBox()
        self.access_selection.addItem("工程师")
        self.access_selection.addItem("操作员")
        access_layout.addWidget(label_access)
        access_layout.addWidget(self.access_selection)
        self.access_selection.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        user_layout = QHBoxLayout()
        label_user = QLabel("新建账号：")
        self.username_input = QLineEdit()
        user_layout.addWidget(label_user)
        user_layout.addWidget(self.username_input)
        self.username_input.setPlaceholderText("请输入用户名")

        pwd_layout = QHBoxLayout()
        label_pwd = QLabel("输入密码：")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        pwd_layout.addWidget(label_pwd)
        pwd_layout.addWidget(self.password_input)
        self.password_input.setPlaceholderText("请输入密码")

        info_layout = QHBoxLayout()
        self.info = QLabel("")
        self.info.setMaximumHeight(15)
        self.info.setMinimumHeight(10)
        info_layout.addWidget(self.info)

        button_layout = QHBoxLayout()
        add_user_button = QPushButton("添加账号")
        add_user_button.clicked.connect(self.add_user_click)
        exit_button = QPushButton(" 退  出 ")
        exit_button.clicked.connect(self.exit_click)
        button_layout.addWidget(add_user_button)
        button_layout.addWidget(exit_button)

        layout.addLayout(access_layout)
        layout.addLayout(user_layout)
        layout.addLayout(pwd_layout)
        layout.addLayout(info_layout)
        layout.addLayout(button_layout)

        layout.setContentsMargins(25, 10, 25, 10)

        self.setLayout(layout)

        self.setStyleSheet(ui_style_const.qcombobox_stytle +
                           ui_style_const.qpushbutton_stytle +
                           ui_style_const.qlineedit_stytle)

    def add_user_click(self):
        username = self.username_input.text()
        password = self.password_input.text()
        access_lvl = ACCESS_LVL_DICT[self.access_selection.currentText()]
        if not password:
            self.info.setText("添加账号失败")
        else:
            enc_pwd = encrypt_password(username, password)
            if self.add_user_info_to_db(username, enc_pwd, access_lvl):
                self.info.setText("添加账号成功")
                self.username_input.clear()
                self.password_input.clear()
            else:
                self.info.setText("添加账号失败")

    def add_user_info_to_db(self, username, password, access_lvl):
        if not username or not password or not access_lvl:
            self.logger.error("Username, password, and access level cannot be empty.")
            return False
        try:
            with DataSave(model_consts.DATABASE_PATH) as database:
                result = database.query_matching_data([(username,)],
                                                      "users_table", ["user_name"],
                                                      ["user_id"])
                if not result:
                    insert_code, msg = database.insert_data_into_db("users_table",
                                                                      model_consts.USERS_COLUMNS,
                                                                      [(username, password, access_lvl)])

                    if insert_code == error_code.OK:
                        self.logger.info(f"Successful to create user {username}.")
                        return True
                    else:
                        self.logger.error(f"Failed to create user. {msg}")
                        return False
                self.logger.warning(f"This user {username} already exists.")
                return False
        except Exception as e:
            self.logger.error("Failed to create user. %s" % (str(e)[:40]))
            return False

    def exit_click(self):
        self.close()


class ChangePwdWindow(QDialog):

    def __init__(self, user_name, logger):
        super().__init__()
        self.logger = logger
        self.user_name = user_name
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("修改密码")
        self.setFixedSize(300, 200)

        layout = QVBoxLayout()

        self.info = QLabel("账    号： " + self.user_name)
        self.info.setMaximumHeight(25)
        self.info.setMinimumHeight(10)

        pwd_layout = QHBoxLayout()
        label_pwd = QLabel("新建密码：")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        pwd_layout.addWidget(label_pwd)
        pwd_layout.addWidget(self.password_input)
        self.password_input.setPlaceholderText("请输入新的密码")

        confirm_pwd_layout = QHBoxLayout()
        label_confirm_pwd = QLabel("确认密码：")
        self.confirm_password_input = QLineEdit()
        self.confirm_password_input.setEchoMode(QLineEdit.Password)
        confirm_pwd_layout.addWidget(label_confirm_pwd)
        confirm_pwd_layout.addWidget(self.confirm_password_input)
        self.confirm_password_input.setPlaceholderText("请再次输入密码")

        button_layout = QHBoxLayout()
        change_pwd_button = QPushButton("修改密码")
        change_pwd_button.clicked.connect(self.change_pwd_click)
        h_spacer_change_pwd_i = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        h_spacer_change_pwd_ii = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        button_layout.addItem(h_spacer_change_pwd_i)
        button_layout.addWidget(change_pwd_button)
        button_layout.addItem(h_spacer_change_pwd_ii)
        button_layout.setContentsMargins(0, 10, 0, 0)

        layout.addWidget(self.info)
        layout.addLayout(pwd_layout)
        layout.addLayout(confirm_pwd_layout)
        layout.addLayout(button_layout)

        layout.setContentsMargins(25, 10, 25, 10)

        self.setLayout(layout)

        self.setStyleSheet(ui_style_const.qpushbutton_stytle + ui_style_const.qlineedit_stytle)

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

    def change_pwd_in_db(self, user_name, enc_pwd):
        try:
            with DataSave(model_consts.DATABASE_PATH) as database:
                result = database.query_matching_data([(user_name,)], "users_table",
                                                      ["user_name"], ["password"])
                if result:
                    new_password_data = {"password": enc_pwd}
                    update_code, msg = database.update_table_data("users_table", new_password_data,
                                                                  {"user_name": user_name}, update_time=True)
                    if update_code == error_code.OK:
                        self.logger.info("Password reset succeeded.")
                        return True
                    else:
                        self.logger.error(msg)
                        return False
                else:
                    self.logger.warning(f"The user {user_name} does not exist.")
                    return False
        except Exception as e:
            self.logger.error("Failed to reset password. %s" % (str(e)[:40]))
            return False


def encrypt_password(user_name, password):
    mac_pwd = get_mac_address() + user_name + password
    sh = hashlib.sha1()
    sh.update(mac_pwd.encode("utf-8"))
    enc_pwd = sh.hexdigest()
    return enc_pwd


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LoginWindow()
    # window = ChangePwdWindow('admin', LogManager.set_log_handler("core"))
    # window = AddAccountWindow(LogManager.set_log_handler("core"))
    window.show()
    sys.exit(app.exec_())
