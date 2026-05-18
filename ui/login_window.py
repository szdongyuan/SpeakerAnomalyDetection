import hashlib
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QHBoxLayout, QSizePolicy, QVBoxLayout

from base.db_manager import DataSave, ensure_system_database_ready
from base.system_intervction.hardware_intervction import get_mac_address
from base.log_manager import LogManager
from consts import error_code, model_consts
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.widgets import PushButton, ComboBox, LineEdit, Label, MessageBox
from ui.ui_src import ui_resources


ACCESS_LVL_DICT = {"管理员": "Admin", "工程师": "Engineer", "操作员": "Operator"}


class LoginWindow(QDialog):

    def __init__(self, access_lvl=None):
        super().__init__()

        self.pwd_checked = False
        self.access_lvl = access_lvl
        self.logger = LogManager.set_log_handler("core")
        self.init_ui()

    def init_ui(self):
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))
        self.setWindowTitle("登录")
        self.setMaximumSize(600, 600)
        self.setMinimumSize(300, 400)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setAutoFillBackground(True)

        label_background_layout = QHBoxLayout()
        self.label_background_image = Label(self)
        self.label_background_image.setFixedHeight(200)
        self.label_background_image.setScaledContents(True)
        label_background_layout.addWidget(self.label_background_image)
        label_background_layout.setAlignment(Qt.AlignCenter | Qt.AlignTop)

        self.login_layout = QVBoxLayout()

        self.access_layout = QHBoxLayout()
        self.label_access = Label("权 限：")
        self.access_selection = ComboBox()
        self.access_selection.addItem("管理员")
        self.access_selection.addItem("工程师")
        self.access_selection.addItem("操作员")
        self.access_selection.currentTextChanged.connect(self.access_add_account)
        self.access_layout.addWidget(self.label_access)
        self.access_layout.addWidget(self.access_selection)
        self.access_selection.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        user_layout = QHBoxLayout()
        self.label_user = Label("账 号：")
        self.username_input = LineEdit()
        self.username_input.setPlaceholderText("请输入你的账号")
        self.add_account_botton = PushButton("添加账号")
        self.add_account_botton.clicked.connect(self.add_account_click)
        user_layout.addWidget(self.label_user)
        user_layout.addWidget(self.username_input)
        user_layout.addWidget(self.add_account_botton)

        pwd_layout = QHBoxLayout()
        self.label_pwd = Label("密 码：")
        self.password_input = LineEdit()
        self.password_input.setPlaceholderText("请输入账号密码")
        self.password_input.setEchoMode(LineEdit.Password)
        self.change_pwd_botton = PushButton("修改密码")
        self.change_pwd_botton.clicked.connect(self.change_pwd_click)
        pwd_layout.addWidget(self.label_pwd)
        pwd_layout.addWidget(self.password_input)
        pwd_layout.addWidget(self.change_pwd_botton)

        button_layout = QHBoxLayout()
        self.login_button = PushButton(" 登  录 ")
        self.login_button.clicked.connect(self.login_click)
        button_layout.addStretch()
        button_layout.addWidget(self.login_button)
        button_layout.addStretch()

        self.login_layout.addLayout(self.access_layout)
        self.login_layout.addLayout(user_layout)
        self.login_layout.addLayout(pwd_layout)
        self.login_layout.addLayout(button_layout)
        self.login_layout.setAlignment(Qt.AlignCenter)
        self.login_layout.setSpacing(15)
        self.login_layout.setContentsMargins(30, 0, 30, 5)

        layout = QVBoxLayout()
        layout.addLayout(label_background_layout)
        layout.addLayout(self.login_layout)
        layout.setContentsMargins(0, 0, 0, 0)

        self.setLayout(layout)
        self.login_button.setDefault(True)

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
            MessageBox.warning(self, "Error", "Username or Password is incorrect")

    def change_pwd_click(self):
        if self.check_credentials():
            dlg = ChangePwdWindow(self.username_input.text(), self.logger)
            dlg.exec()
        else:
            MessageBox.warning(self, "Error", "Username or Password is incorrect")

    def login_click(self):
        if self.check_credentials():
            self.pwd_checked = True
            self.close()
        else:
            MessageBox.warning(self, "Error", "Username or Password is incorrect")

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
        if event.size().width() / (event.size().height() / 2) > 1.5:
            space_size = int(15 / 400 * event.size().height())
        else:
            space_size = int(15 / 300 * event.size().width())

        self.access_layout.setContentsMargins(0, 0, int(self.add_account_botton.sizeHint().width()) + space_size, 0)

        self.login_layout.setSpacing(space_size)

        original_pixmap = QPixmap(":/ui/icon/ui_login_icon.png")
        scaled_pixmap = original_pixmap.scaledToHeight(200, Qt.SmoothTransformation)
        pixmap = QPixmap(event.size().width(), 200)
        pixmap.fill(QColor(174, 171, 162))

        painter = QPainter(pixmap)
        painter.drawPixmap((event.size().width() - scaled_pixmap.width()) // 2, 0, scaled_pixmap)
        painter.end()

        self.label_background_image.setPixmap(pixmap)

    def paintEvent(self, event):
        painter = QPainter(self)
        width = self.width()
        height = self.height()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(174, 171, 162))
        painter.drawRect(0, 0, width, 200)
        painter.setBrush(QColor(174, 171, 162, 123))
        painter.drawRect(0, 200, width, height - 200)
        painter.end()

    @staticmethod
    def get_user_info_from_db(user_name):
        ensure_system_database_ready()
        with DataSave(model_consts.SYSTEM_DATABASE_PATH) as database:
            query_code, query_data = database.query(
                "users_table", ["user_name", "access_level", "password"], {"user_name": user_name}
            )
        if query_code == error_code.OK and query_data:
            user_data = query_data[0]
            return {"user_name": user_data[0], "access_level": user_data[1], "password": user_data[2]}
        else:
            return {}

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            pass
        else:
            super().keyPressEvent(event)

    def on_exec(self):
        self.exec()
        return (self.access_lvl, self.username_input.text()) if self.pwd_checked else (None, None)

    def showEvent(self, event):
        super().showEvent(event)
        self.username_input.setFocus()


class AddAccountWindow(QDialog):

    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        self.init_ui()

    def init_ui(self):
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))
        self.setWindowTitle("添加账号")
        self.setFixedSize(350, 240)
        layout = QVBoxLayout()

        access_layout = QHBoxLayout()
        label_access = Label("权    限：")
        self.access_selection = ComboBox()
        self.access_selection.addItem("工程师")
        self.access_selection.addItem("操作员")
        access_layout.addWidget(label_access)
        access_layout.addWidget(self.access_selection)
        self.access_selection.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        user_layout = QHBoxLayout()
        label_user = Label("新建账号：")
        self.username_input = LineEdit()
        user_layout.addWidget(label_user)
        user_layout.addWidget(self.username_input)
        self.username_input.setPlaceholderText("请输入用户账号")

        pwd_layout = QHBoxLayout()
        label_pwd = Label("输入密码：")
        self.password_input = LineEdit()
        self.password_input.setEchoMode(LineEdit.Password)
        pwd_layout.addWidget(label_pwd)
        pwd_layout.addWidget(self.password_input)
        self.password_input.setPlaceholderText("请输入用户密码")

        info_layout = QHBoxLayout()
        self.info = Label("")
        self.info.set_font_size(15)
        self.info.setObjectName("infoLabel")
        self.info.setAlignment(Qt.AlignCenter)
        self.info.setMaximumHeight(15)
        self.info.setMinimumHeight(10)
        info_layout.addWidget(self.info)

        button_layout = QHBoxLayout()
        add_user_button = PushButton("添加账号")
        add_user_button.clicked.connect(self.add_user_click)
        exit_button = PushButton(" 退  出 ")
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
            ensure_system_database_ready()
            with DataSave(model_consts.SYSTEM_DATABASE_PATH) as database:
                result = database.query_matching_data([(username,)], "users_table", ["user_name"], ["user_id"])
                if not result:
                    insert_code, msg = database.insert_data_into_db(
                        "users_table", model_consts.USERS_COLUMNS, [(username, password, access_lvl)]
                    )

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
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))
        self.setWindowTitle("修改密码")
        self.setFixedSize(350, 200)

        layout = QVBoxLayout()

        self.info = Label("账    号： " + self.user_name)
        self.info.setMaximumHeight(25)
        self.info.setMinimumHeight(10)

        pwd_layout = QHBoxLayout()
        label_pwd = Label("新建密码：")
        self.password_input = LineEdit()
        self.password_input.setEchoMode(LineEdit.Password)
        pwd_layout.addWidget(label_pwd)
        pwd_layout.addWidget(self.password_input)
        self.password_input.setPlaceholderText("请输入新的密码")

        confirm_pwd_layout = QHBoxLayout()
        label_confirm_pwd = Label("确认密码：")
        self.confirm_password_input = LineEdit()
        self.confirm_password_input.setEchoMode(LineEdit.Password)
        confirm_pwd_layout.addWidget(label_confirm_pwd)
        confirm_pwd_layout.addWidget(self.confirm_password_input)
        self.confirm_password_input.setPlaceholderText("请再次输入密码")

        button_layout = QHBoxLayout()
        change_pwd_button = PushButton("修改密码")
        change_pwd_button.clicked.connect(self.change_pwd_click)
        button_layout.addStretch()
        button_layout.addWidget(change_pwd_button)
        button_layout.addStretch()
        button_layout.setContentsMargins(0, 10, 0, 0)

        layout.addWidget(self.info)
        layout.addLayout(pwd_layout)
        layout.addLayout(confirm_pwd_layout)
        layout.addLayout(button_layout)

        layout.setContentsMargins(25, 10, 25, 10)

        self.setLayout(layout)

    def change_pwd_click(self):
        if not self.password_input.text():
            MessageBox.warning(self, "错误", "密码不能为空")
        elif self.password_input.text() != self.confirm_password_input.text():
            MessageBox.warning(self, "错误", "两次输入的密码不一致")
        else:
            enc_pwd = encrypt_password(self.user_name, self.password_input.text())
            if self.change_pwd_in_db(self.user_name, enc_pwd):
                MessageBox.information(self, "成功", "修改密码成功")
                self.close()
            else:
                MessageBox.warning(self, "失败", "修改密码失败")

    def change_pwd_in_db(self, user_name, enc_pwd):
        try:
            ensure_system_database_ready()
            with DataSave(model_consts.SYSTEM_DATABASE_PATH) as database:
                result = database.query_matching_data([(user_name,)], "users_table", ["user_name"], ["password"])
                if result:
                    new_password_data = {"password": enc_pwd}
                    update_code, msg = database.update_table_data(
                        "users_table", new_password_data, {"user_name": user_name}, update_time=True
                    )
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
    # window = LoginWindow()
    window = ChangePwdWindow("admin", LogManager.set_log_handler("core"))
    # window = AddAccountWindow(LogManager.set_log_handler("core"))
    window.show()
    sys.exit(app.exec_())
