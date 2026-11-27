"""TCP 监听及工况启动指令。"""
import json
import logging
from PyQt5.QtNetwork import QHostAddress, QTcpServer, QTcpSocket
from PyQt5.QtWidgets import (QDialog, QFormLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QSpinBox, QVBoxLayout)
from ui.config_dialog_theme import build_config_dialog_stylesheet


class TcpServiceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('TCP 服务设置')
        self.setStyleSheet(build_config_dialog_stylesheet())
        self.server = QTcpServer(self)
        self._clients = {}
        self._start_button = None
        self.server.newConnection.connect(self._accept)
        self.host = QLineEdit('127.0.0.1')
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(9000)
        self.status = QLabel('未启动')
        self._station_sn_by_client = {}
        self.active_station_sn_codes = None
        self.start_button = QPushButton('启动服务')
        self.stop_button = QPushButton('停止服务')
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_service)
        self.stop_button.clicked.connect(self.stop_service)
        form = QFormLayout()
        form.addRow('监听 IP：', self.host)
        form.addRow('端口：', self.port)
        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.status)
        layout.addLayout(buttons)

    def start_service(self):
        address = QHostAddress()
        if not address.setAddress(self.host.text().strip()):
            self.status.setText('IP 地址无效')
            return
        if not self.server.listen(address, self.port.value()):
            self.status.setText('启动失败：' + self.server.errorString())
            return
        self.status.setText('监听中：%s:%s' % (self.host.text().strip(), self.port.value()))
        self.host.setEnabled(False)
        self.port.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_service(self):
        self.server.close()
        for client in self.server.findChildren(QTcpSocket):
            client.abort()
            client.deleteLater()
        self.host.setEnabled(True)
        self.port.setEnabled(True)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status.setText('未启动')

    def set_start_button(self, button):
        self._start_button = button

    def publish_result(self, result):
        message = (json.dumps(result, ensure_ascii=False) + '\n').encode('utf-8')
        for client in list(self._clients):
            if client.state() == QTcpSocket.ConnectedState:
                client.write(message)

    def _accept(self):
        while self.server.hasPendingConnections():
            client = self.server.nextPendingConnection()
            self._clients[client] = bytearray()
            client.readyRead.connect(lambda c=client: self._read(c))
            client.disconnected.connect(lambda c=client: self._disconnect(c))

    def _disconnect(self, client):
        self._clients.pop(client, None)
        self._station_sn_by_client.pop(client, None)
        client.deleteLater()

    def _read(self, client):
        data = self._clients.get(client)
        if data is None:
            return
        data.extend(bytes(client.readAll()))
        while b'\n' in data:
            line, _, remainder = data.partition(b'\n')
            data[:] = remainder
            if len(line) > 4096:
                client.abort()
                return
            if line.strip().startswith(b'{'):
                try:
                    message = json.loads(line.decode('utf-8'))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    message = None
                if isinstance(message, dict) and message.get('type') == 'station_sn_codes':
                    codes = message.get('sn_codes')
                    if isinstance(codes, list) and len(codes) == 4 and all(isinstance(code, str) for code in codes):
                        self._station_sn_by_client[client] = tuple(codes)
                        logging.getLogger('core').info('收到 PLC 工位 SN 报文: %s',
                                                       json.dumps(message, ensure_ascii=False))
                        client.write((json.dumps({'type': 'station_sn_codes_received'}, ensure_ascii=False) + '\n').encode('utf-8'))
                        continue
                    logging.getLogger('core').warning('收到无效的 PLC 工位 SN 报文: %r', line)
            if line.strip() != b'true':
                response = {'success': False, 'error': '未知指令'}
            elif self._start_button is None or not self._start_button.isEnabled():
                self._station_sn_by_client.pop(client, None)
                response = {'success': False, 'error': '检测项目未就绪或正在运行'}
            else:
                self.active_station_sn_codes = self._station_sn_by_client.pop(client, None)
                try:
                    self._start_button.click()
                finally:
                    self.active_station_sn_codes = None
                response = {'success': True, 'status': '启动指令已提交'}
            client.write((json.dumps(response, ensure_ascii=False) + '\n').encode('utf-8'))
        if len(data) > 4096:
            client.abort()
