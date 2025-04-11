import socket
import threading

from base.log_manager import LogManager
from consts.running_consts import tcp_service_recv_bytes


class TcpServer:
    def __init__(self, host='127.0.0.1', port=50000, callback=None):
        self.host = host
        self.port = port
        self.callback = callback
        self.server_socket = None
        self.server_thread = None
        self.stop_flag = False   # False 服务停止标记, True 开启
        self.request_id = None
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
                client_socket, client_address = self.server_socket.accept()
                self.default_logger.info(f"client_address from {client_address} build")

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
                        client_socket.send(f'{res}'.encode())

                    except OSError as e:
                        print(f"server_socket accept error: {e}")
                        break
                print("client close")
                self.default_logger.info(f"client close")
                client_socket.close()
            except Exception as e:
                self.default_logger.info(f"tcp service except: {e}")
                print(f"tcp service except: {e}")