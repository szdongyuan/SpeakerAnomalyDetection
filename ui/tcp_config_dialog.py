import re

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QDialog, QMessageBox

from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.widgets import PushButton, LineEdit, CheckBox, GroupBox


class TcpConfigDialog(QDialog):

    def __init__(self, is_tcp_flag, ip, port, parent=None):
        super(TcpConfigDialog, self).__init__(parent)

        self.ip = ip
        self.port = port
        self.is_tcp_flag = is_tcp_flag
        self.clicked_ok_flag = False
        self.ip_format = False
        self.port_format = False
        self.groupbox_list = list()

        self.ip_lineedit = LineEdit()
        self.port_lineedit = LineEdit()
        self.tcp_checkbox = CheckBox("TCP")
        self.ok_btn = PushButton(" 确  定 ")
        self.cancel_btn = PushButton(" 取  消 ")

        self.set_member_connect()
        self.set_lineedit_text()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("TCP配置")
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        self.setMinimumSize(300, 200)
        self.resize(300, 250)

        self.set_main_layout()
        self.swap_able_status()

    def set_member_connect(self):
        self.cancel_btn.clicked.connect(self.close)
        self.ok_btn.clicked.connect(self.on_ok_btn_clicked)
        self.ip_lineedit.editingFinished.connect(self.on_ip_lineedit_editingfinshed)
        self.port_lineedit.editingFinished.connect(self.on_port_lineedit_editingfinshed)
        self.tcp_checkbox.clicked.connect(self.on_tcp_checkbox_clicked)

    def set_main_layout(self):
        ip_gruopbox = self.create_ip_gruopbox()
        port_groupbox = self.create_port_gruopbox()
        btn_layout = self.create_btn_layout()

        self.groupbox_list.append(ip_gruopbox)
        self.groupbox_list.append(port_groupbox)

        layout = QVBoxLayout()
        layout.addWidget(self.tcp_checkbox)
        layout.addStretch()
        layout.addWidget(ip_gruopbox)
        layout.addWidget(port_groupbox)
        layout.addStretch()
        layout.addLayout(btn_layout)
        layout.addSpacing(5)

        self.setLayout(layout)

    def set_lineedit_text(self):
        self.ip_lineedit.setText(self.ip)
        self.port_lineedit.setText(str(self.port))
        self.tcp_checkbox.setChecked(self.is_tcp_flag)

    def create_ip_gruopbox(self):
        ip_layout = QHBoxLayout()
        ip_layout.addWidget(self.ip_lineedit)

        ip_groupbox = GroupBox("网络地址")
        ip_groupbox.setLayout(ip_layout)
        return ip_groupbox

    def create_port_gruopbox(self):
        port_layout = QHBoxLayout()
        port_layout.addWidget(self.port_lineedit)

        port_groupbox = GroupBox("监听端口 ")
        port_groupbox.setLayout(port_layout)

        return port_groupbox

    def create_btn_layout(self):
        a = PushButton()
        a.setVisible(False)
        a.setDefault(True)

        btn_layoput = QHBoxLayout()
        btn_layoput.addStretch()
        btn_layoput.addWidget(a)
        btn_layoput.addWidget(self.cancel_btn)
        btn_layoput.addWidget(self.ok_btn)

        return btn_layoput

    def swap_able_status(self):
        if self.is_tcp_flag:
            for i in self.groupbox_list:
                i.setEnabled(True)
        else:
            for i in self.groupbox_list:
                i.setEnabled(False)

    def on_tcp_checkbox_clicked(self):
        self.is_tcp_flag = self.sender().isChecked()
        self.swap_able_status()

    def on_ip_lineedit_editingfinshed(self):
        if self.ip_format:
            return
        self.ip_format = True
        self._editing_ip = True
        ip = self.ip_lineedit.text()
        pattern = r"^((25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)$"
        if not re.match(pattern, ip):
            QMessageBox.warning(self, "警告", "无效IP格式。")
            self.ip_lineedit.setFocus()
            self.ip_lineedit.setText(self.ip)
            self._editing_ip = False
            self.ip_format = False
            return
        self.ip = ip
        self._editing_ip = False
        self.ip_format = False

    def on_port_lineedit_editingfinshed(self):
        if self.port_format:
            return
        self.port_format = True
        self._editing_port = True
        port_text = self.port_lineedit.text()
        if not port_text.isdigit():
            QMessageBox.warning(self, "无效端口", "端口号必须是数字")
            self.port_lineedit.setFocus()
            self.port_lineedit.setText(str(self.port))
            self.port_format = False
            return
        port = int(port_text)
        if not (0 < port < 65536):
            QMessageBox.warning(self, "无效端口", "请输入 1 到 65535 之间的端口号")
            self.port_lineedit.setFocus()
            self.port_lineedit.setText(str(self.port))
            self.port_format = False
            return
        self.port = port_text
        self._editing_port = False
        self.port_format = False

    def on_ok_btn_clicked(self):
        self.clicked_ok_flag = True
        self.close()

    def exec(self):
        super().exec()
        if self.clicked_ok_flag:
            result = (self.is_tcp_flag, self.ip, self.port)
        else:
            result = None
        return result


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication

    app = QApplication([])
    dialog = TcpConfigDialog(False, "127.0.0.1", "50000")
    # dialog = TcpConfigDialog(True, "127.0.0.1", "50000")

    a = dialog.exec()
    print(a)
