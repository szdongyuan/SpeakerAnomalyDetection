import time

try:
    import serial
except Exception:  # pragma: no cover - import failure is surfaced via status signal
    serial = None

from PyQt5.QtCore import QThread, pyqtSignal

from base.log_manager import LogManager


class SerialDiscreteInputWorker(QThread):
    sig_state_changed = pyqtSignal(object)
    sig_status = pyqtSignal(object)

    def __init__(self, config):
        super().__init__()
        self.config = config or {}
        self.logger = LogManager.set_log_handler("core")
        self._is_running = False
        self.last_state = None
        self.serial_port = None
        self.last_raw_hex = None
        self._last_no_response_log_time = 0.0

    @staticmethod
    def _debug_print(message):
        print(f"[serial-trigger][worker] {message}")

    def _emit_status(self, **kwargs):
        payload = {
            "connected": False,
            "running": self._is_running,
            "message": "",
            "raw_hex": "",
            "value": "",
            "mode": "",
            "error": "",
        }
        payload.update(kwargs)
        self.sig_status.emit(payload)

    def stop(self):
        self._is_running = False
        try:
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
        except Exception:
            pass
        self.wait(1500)

    @staticmethod
    def _extract_state_value(received_bytes, raw_hex, decoder_mode, state_byte_index):
        if decoder_mode == "full_frame":
            return raw_hex
        if decoder_mode == "state_byte" and len(received_bytes) > state_byte_index:
            return f"{received_bytes[state_byte_index]:02X}"
        return None

    def run(self):
        self._is_running = True
        serial_settings = self.config.get("serial_settings", {})
        polling_settings = self.config.get("polling_settings", {})
        decoder = self.config.get("decoder", {})

        port = serial_settings.get("port", "COM3")
        baudrate = serial_settings.get("baudrate", 9600)
        bytesize = serial_settings.get("bytesize", 8)
        parity = serial_settings.get("parity", "N")
        stopbits = serial_settings.get("stopbits", 1)
        timeout = serial_settings.get("timeout", 0.1)

        interval_ms = polling_settings.get("interval_ms", 50)
        query_command_hex = polling_settings.get("query_command_hex", "")
        decoder_mode = decoder.get("mode", "full_frame")
        state_byte_index = int(decoder.get("state_byte_index", 3) or 3)
        self._debug_print(
            f"启动监听: port={port}, baudrate={baudrate}, mode={decoder_mode}, "
            f"state_byte_index={state_byte_index}, query={query_command_hex}"
        )

        if serial is None:
            msg = "pyserial 未安装，无法启用串口离散输入触发"
            self.logger.error(msg)
            self._debug_print(msg)
            self._emit_status(running=False, connected=False, message=msg, error=msg, mode=decoder_mode)
            return

        try:
            query_bytes = bytes.fromhex(str(query_command_hex or "").strip())
        except ValueError as e:
            msg = f"query_command_hex 配置非法: {e}"
            self.logger.error(msg)
            self._debug_print(msg)
            self._emit_status(running=False, connected=False, message=msg, error=msg, mode=decoder_mode)
            return

        try:
            self.serial_port = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=bytesize,
                parity=parity,
                stopbits=stopbits,
                timeout=timeout,
            )
            try:
                self.serial_port.reset_input_buffer()
            except Exception:
                pass
            msg = f"串口已连接: {port}"
            self.logger.info(msg)
            self._debug_print(msg)
            self._emit_status(running=True, connected=True, message=msg, mode=decoder_mode)
        except Exception as e:
            msg = f"串口打开失败: {port}, {e}"
            self.logger.error(msg)
            self._debug_print(msg)
            self._emit_status(running=False, connected=False, message=msg, error=str(e), mode=decoder_mode)
            return

        while self._is_running:
            try:
                self.serial_port.write(query_bytes)
                time.sleep(interval_ms / 1000.0)

                waiting = getattr(self.serial_port, "in_waiting", 0)
                if waiting <= 0:
                    now = time.monotonic()
                    if (now - self._last_no_response_log_time) >= 1.0:
                        self._last_no_response_log_time = now
                        self._debug_print("轮询已发送，但当前未收到设备响应")
                    continue

                received_bytes = self.serial_port.read(waiting)
                if not received_bytes:
                    continue

                raw_hex = " ".join(f"{b:02X}" for b in received_bytes)
                state_value = self._extract_state_value(received_bytes, raw_hex, decoder_mode, state_byte_index)
                if raw_hex != self.last_raw_hex:
                    self._debug_print(f"收到原始帧: raw_hex={raw_hex}, extracted={state_value}")
                    self.last_raw_hex = raw_hex
                self._emit_status(
                    running=True,
                    connected=True,
                    message="收到串口响应",
                    raw_hex=raw_hex,
                    value=state_value or "",
                    mode=decoder_mode,
                )

                if state_value is not None and state_value != self.last_state:
                    self._debug_print(f"状态变化: last={self.last_state}, current={state_value}")
                    self.sig_state_changed.emit(
                        {
                            "mode": decoder_mode,
                            "value": state_value,
                            "raw_hex": raw_hex,
                        }
                    )
                    self.last_state = state_value

            except Exception as e:
                # 忽略因主动关闭串口或线程停止导致的无效句柄错误
                err_str = str(e)
                if not self._is_running and ("句柄无效" in err_str or "ClearCommError" in err_str or "Bad file descriptor" in err_str):
                    self._debug_print("检测到主动停止导致的句柄失效，已忽略该异常并退出轮询")
                    break
                msg = f"串口轮询异常: {e}"
                self.logger.error(msg)
                self._debug_print(msg)
                self._emit_status(
                    running=self._is_running,
                    connected=False,
                    message=msg,
                    error=str(e),
                    mode=decoder_mode,
                )
                time.sleep(1)

        try:
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
        except Exception:
            pass
        self._debug_print(f"监听结束: port={port}")
        self._emit_status(running=False, connected=False, message=f"串口已断开: {port}", mode=decoder_mode)
