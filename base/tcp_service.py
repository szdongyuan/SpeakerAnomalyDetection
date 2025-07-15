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
        self.client_address = None
        self.default_logger = LogManager.set_log_handler("core")

    def start(self):
        if not self.server_thread or not self.server_thread.is_alive():
            self.stop_flag = True
            self.server_thread = threading.Thread(target=self._run)
            self.server_thread.start()

    def stop(self):
        self.stop_flag = False
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None
            self.default_logger.info("Tcp server stopped.")
        else:
            self.default_logger.info("no tcp server running")

    def _run(self):
        while self.stop_flag:
            try:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.bind((self.host, self.port))
                self.server_socket.listen(5)
                print("Tcp_service is waiting for a connection...")
                client_socket, self.client_address = self.server_socket.accept()
                self.default_logger.info(f"client_address from {self.client_address} build")

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
                        client_socket.send(f"{res}".encode())

                    except OSError as e:
                        self.default_logger.error(f"server_socket accept error: {e}")
                        break
                print("client close")
                self.default_logger.info(f"client close")
                client_socket.close()
            except Exception as e:
                self.default_logger.error(f"tcp service except: {e}")
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
