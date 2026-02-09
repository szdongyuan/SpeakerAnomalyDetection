from PyQt5.QtCore import Qt, QMetaObject, Q_ARG, pyqtSlot

from base.load_config import LoadUiConfig
from base.log_manager import LogManager
from base.tcp_service import TcpServer, check_tcp_msg_format
from consts.action_code import RequestTypeEnum
from ui.tcp_config_dialog import TcpConfigDialog


def _sequence_window_cls():
    from ui.sequence.sequence_widget import SequenceWindow

    return SequenceWindow


class SequenceWidgetTcpOpsMixin:

    @pyqtSlot(str)
    def _tcp_run_test(self, label: str = "not_labeled"):
        """
        TCP 回调线程通过 QueuedConnection 投递到 Qt 主线程的入口。
        直接调用 start_this_play 可能发生跨线程 UI 操作风险，因此统一走这里。
        """
        self.start_this_play(label)

    def swap_tcp_status(self):
        if self.tcp_flag:
            self.barcode_scanner_box.setEnabled(False)
            self.tcp_ip, self.tcp_port = LoadUiConfig.get_tcp_config()
            if hasattr(self, "tcp_server") and self.__class__.tcp_server:
                self.__class__.tcp_server.stop()
                self.__class__.tcp_server = None
            self.__class__.tcp_server = TcpServer(host=self.tcp_ip, port=self.tcp_port, callback=self.deal_package)
            self.__class__.tcp_server.start()
        else:
            self.barcode_scanner_box.setEnabled(True)
            if hasattr(self, "tcp_server") and self.__class__.tcp_server:
                self.__class__.tcp_server.stop()
                self.__class__.tcp_server = None

    @staticmethod
    def deal_package(info):
        """
        info: {
                  "RequestType": "0-9999",
                  "RequestContent": {
                    "User": "Alice",
                    "Action": "ScanBarcode",
                    "label": "NG"
                  },
                  "IsSync": false,
                  "Timestamp": "2025-04-09T16:30:00"
              }
        """
        ok, data = check_tcp_msg_format(info)
        if not ok:
            return data
        request_type = int(data.get("RequestType"))
        request_content = data.get("RequestContent", {})
        is_sync = data.get("IsSync")
        timestamp = data.get("Timestamp")
        request_id = f"{request_type}@{timestamp}"
        if request_id == _sequence_window_cls().tcp_server.request_id:
            return "pass"
        else:
            _sequence_window_cls().tcp_server.request_id = request_id
        # allocating task
        if request_type == RequestTypeEnum.RUN_TEST.value:
            # 兼容多种客户端字段命名：
            # - 老客户端: Label
            # - 示例文档: label
            label = request_content.get("Label") or request_content.get("label") or "not_labeled"
            # Dispatch to current SequenceWindow instance in Qt main thread
            try:
                ref = getattr(_sequence_window_cls(), "_active_instance_ref", None)
                inst = ref() if callable(ref) else None
            except Exception:
                inst = None
            if inst is None:
                try:
                    LogManager.set_log_handler("core").warning(
                        "TCP RUN_TEST received, but no active SequenceWindow instance is available."
                    )
                except Exception:
                    pass
            else:
                try:
                    # 必须用 QueuedConnection 投递到 inst 所在线程（Qt 主线程），避免跨线程 UI 调用
                    QMetaObject.invokeMethod(inst, "_tcp_run_test", Qt.QueuedConnection, Q_ARG(str, str(label)))
                except Exception as e:
                    try:
                        LogManager.set_log_handler("core").error(f"TCP dispatch RUN_TEST failed: {e}")
                    except Exception:
                        pass
        return "ok"

    def on_tcp_btn_clicked(self):
        tcp_config_dialog = TcpConfigDialog(self.tcp_flag, self.tcp_ip, self.tcp_port)
        result = tcp_config_dialog.exec()
        if result:
            self.tcp_flag, self.tcp_ip, self.tcp_port = result
            LoadUiConfig.write_tcp_config(self.tcp_ip, self.tcp_port, self.default_logger)
            self.swap_tcp_status()
