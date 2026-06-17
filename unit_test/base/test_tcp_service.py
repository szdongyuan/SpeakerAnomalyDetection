import pytest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from base.tcp_service import TcpServer


class FakeSocket:
    def __init__(self, fail_send=False):
        self.fail_send = fail_send
        self.sent = []
        self.closed = False

    def sendall(self, payload):
        if self.fail_send:
            raise OSError("send failed")
        self.sent.append(payload)

    def close(self):
        self.closed = True


def test_send_to_current_client_sends_encoded_message():
    server = TcpServer(host="127.0.0.1", port=50000, callback=lambda info: "ok")
    fake_socket = FakeSocket()
    server.client_socket = fake_socket
    server.client_address = ("127.0.0.1", 64902)

    assert server.send_to_current_client("ok") is True

    assert fake_socket.sent == [b"ok"]
    assert server.client_socket is fake_socket
    assert server.client_address == ("127.0.0.1", 64902)


def test_send_to_current_client_returns_false_without_connected_client():
    server = TcpServer(host="127.0.0.1", port=50000, callback=lambda info: "ok")

    assert server.send_to_current_client("ok") is False


def test_send_to_current_client_closes_and_clears_state_on_send_error():
    server = TcpServer(host="127.0.0.1", port=50000, callback=lambda info: "ok")
    fake_socket = FakeSocket(fail_send=True)
    server.client_socket = fake_socket
    server.client_address = ("127.0.0.1", 64902)

    assert server.send_to_current_client("ok") is False

    assert fake_socket.closed is True
    assert server.client_socket is None
    assert server.client_address is None


def test_run_routes_callback_response_through_send_to_current_client(monkeypatch):
    class FakeClientSocket:
        def __init__(self):
            self.closed = False

        def recv(self, _size):
            return b"request"

        def send(self, _payload):
            raise AssertionError("direct client_socket.send should not be used")

        def close(self):
            self.closed = True

    class FakeListeningSocket:
        def __init__(self, client_socket):
            self.client_socket = client_socket
            self.closed = False

        def bind(self, _address):
            return None

        def listen(self, _backlog):
            return None

        def accept(self):
            return self.client_socket, ("127.0.0.1", 64902)

        def close(self):
            self.closed = True

    client_socket = FakeClientSocket()
    listening_socket = FakeListeningSocket(client_socket)

    def callback(_info):
        server.stop_flag = False
        return "ok"

    server = TcpServer(host="127.0.0.1", port=50000, callback=callback)
    sent_messages = []
    server.send_to_current_client = lambda message: sent_messages.append(message) or True
    monkeypatch.setattr("base.tcp_service.socket.socket", lambda *_args, **_kwargs: listening_socket)
    server.stop_flag = True

    server._run()

    assert sent_messages == ["ok"]


def test_run_closes_and_clears_current_client_when_callback_raises(monkeypatch):
    class FakeClientSocket:
        def __init__(self):
            self.closed = False

        def recv(self, _size):
            return b"request"

        def close(self):
            self.closed = True

    class FakeListeningSocket:
        def __init__(self, client_socket):
            self.client_socket = client_socket
            self.closed = False

        def bind(self, _address):
            return None

        def listen(self, _backlog):
            return None

        def accept(self):
            return self.client_socket, ("127.0.0.1", 64902)

        def close(self):
            self.closed = True

    client_socket = FakeClientSocket()
    listening_socket = FakeListeningSocket(client_socket)

    def callback(_info):
        server.stop_flag = False
        raise RuntimeError("callback failed")

    server = TcpServer(host="127.0.0.1", port=50000, callback=callback)
    monkeypatch.setattr("base.tcp_service.socket.socket", lambda *_args, **_kwargs: listening_socket)
    monkeypatch.setattr("base.tcp_service.time.sleep", lambda _seconds: None)
    server.stop_flag = True

    server._run()

    assert client_socket.closed is True
    assert server.client_socket is None
    assert server.client_address is None


def test_run_closes_listening_socket_before_retry_when_callback_raises(monkeypatch):
    class FakeClientSocket:
        def __init__(self):
            self.closed = False

        def recv(self, _size):
            return b"request"

        def close(self):
            self.closed = True

    class FakeListeningSocket:
        def __init__(self, client_socket=None):
            self.client_socket = client_socket
            self.closed = False

        def bind(self, _address):
            return None

        def listen(self, _backlog):
            return None

        def accept(self):
            return self.client_socket, ("127.0.0.1", 64902)

        def close(self):
            self.closed = True

    class RetryListeningSocket(FakeListeningSocket):
        def bind(self, _address):
            retry_saw_first_closed.append(first_listening.closed)
            server.stop_flag = False
            raise RuntimeError("stop after retry cleanup check")

    client_socket = FakeClientSocket()
    first_listening = FakeListeningSocket(client_socket)
    second_listening = RetryListeningSocket()
    sockets = iter([first_listening, second_listening])
    retry_saw_first_closed = []

    def callback(_info):
        raise RuntimeError("callback failed")

    server = TcpServer(host="127.0.0.1", port=50000, callback=callback)
    monkeypatch.setattr("base.tcp_service.socket.socket", lambda *_args, **_kwargs: next(sockets))
    monkeypatch.setattr("base.tcp_service.time.sleep", lambda _seconds: None)
    server.stop_flag = True

    server._run()

    assert retry_saw_first_closed == [True]
    assert first_listening.closed is True
    assert server.server_socket is None
