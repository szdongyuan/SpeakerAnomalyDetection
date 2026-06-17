import json
import socket
import threading
import time

from base.log_manager import LogManager
from consts.action_code import RequestTypeEnum
from consts.running_consts import tcp_service_recv_bytes


class TcpServer:
    def __init__(self, host=None, port=None, callback=None):
        self.host = host
        self.port = port
        self.callback = callback
        self.server_socket = None
        self.server_thread = None
        self.stop_flag = False  # False 服务停止标记, True 开启
        self.request_id = None
        self.client_socket = None
        self.client_address = None
        self.client_lock = threading.RLock()
        self.default_logger = LogManager.set_log_handler("core")

    def start(self):
        if not self.server_thread or not self.server_thread.is_alive():
            self.stop_flag = True
            self.server_thread = threading.Thread(target=self._run)
            self.server_thread.start()

    def stop(self):
        self.stop_flag = False
        self._clear_current_client()
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None
            self.default_logger.info("Tcp server stopped.")
        else:
            self.default_logger.info("no tcp server running")

    def _set_current_client(self, client_socket, client_address):
        with self.client_lock:
            self.client_socket = client_socket
            self.client_address = client_address

    def _clear_current_client(self, expected_socket=None):
        with self.client_lock:
            current_socket = self.client_socket
            if expected_socket is not None and current_socket is not expected_socket:
                return
            self.client_socket = None
            self.client_address = None

        if current_socket:
            try:
                current_socket.close()
            except OSError as e:
                self.default_logger.error(f"client_socket close error: {e}")

    def send_to_current_client(self, message):
        with self.client_lock:
            client_socket = self.client_socket
            if client_socket is None:
                self.default_logger.warning("tcp_send_skip: no active client socket")
                return False
            payload = f"{message}".encode()
            try:
                client_socket.sendall(payload)
                return True
            except OSError as e:
                self.default_logger.error(f"tcp_send_error: {e}")

        self._clear_current_client(expected_socket=client_socket)
        return False

    def _run(self):
        while self.stop_flag:
            client_socket = None
            try:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.bind((self.host, self.port))
                self.server_socket.listen(5)
                client_socket, client_address = self.server_socket.accept()
                self._set_current_client(client_socket, client_address)
                self.default_logger.info(f"client_address from {client_address} build")

                try:
                    while True:
                        if not self.stop_flag:
                            break
                        try:
                            recved = client_socket.recv(tcp_service_recv_bytes)
                            if not recved:
                                self.default_logger.info(f"client No data sent or has been closed")
                                break
                            info = recved.decode()

                            res = self.callback(info)
                            if not self.send_to_current_client(res):
                                break

                        except OSError as e:
                            self.default_logger.error(f"server_socket accept error: {e}")
                            break
                finally:
                    self.default_logger.info(f"client close")
                    self._clear_current_client(expected_socket=client_socket)
            except Exception as e:
                self.default_logger.error(f"tcp service except: {e}")
                if self.server_socket:
                    try:
                        self.server_socket.close()
                    except OSError as close_error:
                        self.default_logger.error(f"server_socket close error: {close_error}")
                    finally:
                        self.server_socket = None
                time.sleep(1)


def check_tcp_msg_format(info):
    try:
        data = json.loads(info)
    except json.JSONDecodeError as e:
        return False, "error, json format error"
    req_type = int(data.get("RequestType"))
    is_sync = data.get("IsSync")
    timestamp = data.get("Timestamp")
    if req_type not in [rte.value for rte in RequestTypeEnum]:
        return False, "error, RequestType error"
    if not isinstance(is_sync, bool):
        return False, "error, IsSync type error"
    if not timestamp or not isinstance(timestamp, str):
        return False, "error, Timestamp type error "
    return True, data
