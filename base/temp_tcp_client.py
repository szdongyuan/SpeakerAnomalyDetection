import socket


class TempTcpClient(object):
    def __init__(self, ip, port, information):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((ip, port))
        self.sock.send(information.encode())
        self.sock.close()
