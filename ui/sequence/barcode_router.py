from typing import Protocol, Any

from PyQt5.QtCore import QObject


class TriggerRouterPort(Protocol):
    def handle_barcode_return_pressed(self) -> bool: ...

    def handle_barcode_text_changed(self, text: str) -> None: ...

    def handle_barcode_debounce_timeout(self) -> bool: ...

    def handle_keypress(self, obj: Any, event: Any) -> bool | None: ...


class BarcodeRouter(QObject):
    """Thin Qt signal adapter with no SequenceWindow back-reference."""

    def __init__(self, port: TriggerRouterPort, parent: QObject | None = None):
        if parent is None and isinstance(port, QObject):
            parent = port
        super().__init__(parent)
        self._port = port

    # -----------------------------
    # 复用/搬迁：S/N 输入框的 Enter / textChanged 防抖提交
    # -----------------------------
    def on_barcode_return_pressed(self):
        return self._port.handle_barcode_return_pressed()

    def on_barcode_text_changed(self, _text: str):
        return self._port.handle_barcode_text_changed(_text)

    def on_barcode_debounce_timeout(self):
        return self._port.handle_barcode_debounce_timeout()

    # -----------------------------
    # 复用/搬迁：eventFilter 主逻辑（仅处理 KeyPress）
    # 返回值：
    # - True: 吞掉事件
    # - False: 不处理，让 Qt 继续派发
    # - None: 不是扫码逻辑的处理范围
    # -----------------------------
    def handle_keypress(self, obj, event):
        return self._port.handle_keypress(obj, event)


