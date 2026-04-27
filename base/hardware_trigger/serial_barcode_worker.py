"""Passive serial barcode reader.

Opens a COM port, waits for the scanner to push data, splits the byte
stream on a configured terminator (default ``\\r\\n``) and emits each
decoded barcode via :pyattr:`sig_barcode_received`. Kept intentionally
similar to :class:`SerialDiscreteInputWorker` so the two worker types
can share debugging habits, but this path is passive-read only --
there's no polling query to write back to the device.

Any serial I/O failure is treated as recoverable: the worker logs it,
emits a status payload so the UI can surface it, and either retries
the read loop or exits cleanly on ``stop()``. It never raises out of
``run()``, so a misconfigured port cannot take the hardware manager
down with it.
"""

import time
import traceback

try:
    import serial
except Exception:  # pragma: no cover - import failure is surfaced via status signal
    serial = None

from PyQt5.QtCore import QThread, pyqtSignal

from base.log_manager import LogManager


_DEFAULT_TERMINATOR = "\r\n"
_DEFAULT_ENCODING = "utf-8"
_MAX_BUFFER_BYTES = 4096  # hard cap so a broken scanner cannot OOM us
_READ_BYTE_FALLBACK = 1   # fall back to blocking 1-byte read when in_waiting==0


class SerialBarcodeWorker(QThread):
    """QThread that reads barcodes from a serial scanner.

    Signals:
        sig_barcode_received(str): one decoded barcode (terminator stripped,
            whitespace trimmed). Empty barcodes are never emitted.
        sig_status(object): dict payload describing current listener state,
            mirrors the contract used by SerialDiscreteInputWorker so the
            UI can reuse status rendering if needed later.
    """

    sig_barcode_received = pyqtSignal(str)
    sig_status = pyqtSignal(object)

    def __init__(self, config):
        super().__init__()
        self.config = config if isinstance(config, dict) else {}
        self.logger = LogManager.set_log_handler("core")
        self._is_running = False
        self.serial_port = None

    def _debug_print(self, message):
        """Route chatter to both stdout (PyCharm console) and the
        core logger (goes into the rotating log file).

        Using INFO level so the operator does not have to flip log
        levels to troubleshoot a silent serial scanner. Volume is
        naturally low because we only call this on actual events
        (open / close / raw chunk / decoded barcode / idle flush),
        not on every idle read tick.
        """
        line = f"[serial-barcode][worker] {message}"
        print(line)
        try:
            self.logger.info(line)
        except Exception:
            pass

    def _emit_status(self, **kwargs):
        payload = {
            "connected": False,
            "running": self._is_running,
            "message": "",
            "error": "",
        }
        payload.update(kwargs)
        try:
            self.sig_status.emit(payload)
        except Exception:
            # Never let status emission crash the worker.
            pass

    def stop(self):
        """Request shutdown. Safe to call from any thread, idempotent.

        Logs the caller's stack so we can tell WHO asked the worker to
        stop when chasing "why did my scanner listener die mid-run"
        bugs. The traceback is only captured if the worker is actually
        transitioning from running -> stopping, so repeated idempotent
        ``stop()`` calls stay quiet.
        """
        if self._is_running:
            try:
                stack = "".join(traceback.format_stack(limit=10))
                self._debug_print(
                    "stop() 被调用, 即将关闭 worker. 调用栈(最多10层):\n" + stack
                )
            except Exception:
                pass
        self._is_running = False
        try:
            if self.serial_port is not None and getattr(self.serial_port, "is_open", False):
                self.serial_port.close()
        except Exception:
            pass
        # Give the read loop up to 1.5s to wind down. Matches SerialDiscreteInputWorker.
        self.wait(1500)

    @staticmethod
    def _resolve_terminator_bytes(raw, encoding):
        """Convert the configured terminator into raw bytes.

        Accepts either a plain string (``"\\r\\n"``) or a hex-style
        string (``"0D 0A"``). Falls back to CRLF when the value is
        missing/unusable, so the loop is never left with ``b""`` which
        would make splitting ambiguous.
        """
        if isinstance(raw, bytes):
            return raw if raw else _DEFAULT_TERMINATOR.encode(encoding, errors="ignore")
        if not isinstance(raw, str) or not raw:
            return _DEFAULT_TERMINATOR.encode(encoding, errors="ignore")

        # Heuristic: only letters 0-9/A-F and spaces -> treat as hex string.
        stripped = raw.strip()
        if stripped and all(c in "0123456789abcdefABCDEF " for c in stripped):
            try:
                decoded = bytes.fromhex(stripped.replace(" ", ""))
                if decoded:
                    return decoded
            except ValueError:
                pass

        try:
            encoded = raw.encode(encoding, errors="ignore")
        except (LookupError, TypeError):
            encoded = raw.encode(_DEFAULT_ENCODING, errors="ignore")
        return encoded or _DEFAULT_TERMINATOR.encode(encoding, errors="ignore")

    def _decode_and_emit(self, line_bytes, encoding):
        """Decode one raw line (terminator already stripped) and emit.

        Empty / whitespace-only lines are skipped so a stray extra
        terminator does not generate a bogus commit downstream.
        """
        if not line_bytes:
            return
        try:
            text = line_bytes.decode(encoding, errors="ignore")
        except (LookupError, TypeError):
            text = line_bytes.decode(_DEFAULT_ENCODING, errors="ignore")

        text = text.strip()
        if not text:
            return

        self._debug_print(f"decoded barcode: '{text}' (len={len(text)})")
        try:
            self.sig_barcode_received.emit(text)
            self._debug_print(f"sig_barcode_received.emit -> manager: '{text}'")
        except Exception as e:
            # Signal slot raising should never kill the worker loop.
            self.logger.warning(f"[serial-barcode] emit 异常: {e}")

    def run(self):
        self._is_running = True

        port = str(self.config.get("port", "") or "").strip()
        baudrate = int(self.config.get("baudrate", 9600) or 9600)
        bytesize = int(self.config.get("bytesize", 8) or 8)
        parity = str(self.config.get("parity", "N") or "N")
        stopbits = self.config.get("stopbits", 1) or 1
        timeout = float(self.config.get("timeout", 0.1) or 0.1)
        encoding = str(self.config.get("encoding", _DEFAULT_ENCODING) or _DEFAULT_ENCODING)
        terminator_bytes = self._resolve_terminator_bytes(
            self.config.get("terminator", _DEFAULT_TERMINATOR), encoding
        )

        if not port:
            msg = "串口扫码枪: 未配置 port, 监听未启动"
            self.logger.info(msg)
            self._debug_print(msg)
            self._emit_status(running=False, connected=False, message=msg)
            return

        if serial is None:
            msg = "pyserial 未安装, 无法启用串口扫码枪"
            self.logger.error(msg)
            self._debug_print(msg)
            self._emit_status(running=False, connected=False, message=msg, error=msg)
            return

        self._debug_print(
            f"启动监听: port={port}, baudrate={baudrate}, "
            f"terminator={terminator_bytes!r} (hex={terminator_bytes.hex()}), "
            f"encoding={encoding}"
        )

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
            msg = f"串口扫码枪已打开: {port}"
            self.logger.info(msg)
            self._emit_status(running=True, connected=True, message=msg)
        except Exception as e:
            msg = f"串口扫码枪打开失败: {port}, {e}"
            self.logger.error(msg)
            self._debug_print(msg)
            self._emit_status(running=False, connected=False, message=msg, error=str(e))
            return

        buffer = b""
        # Idle-flush fallback: if the scanner does NOT send the configured
        # terminator (many serial scanners default to CR-only, LF-only or
        # no terminator at all), the while-loop below would sit on the
        # accumulated bytes forever. Instead, when the port has been idle
        # for a short window after receiving something, treat the current
        # buffer as one complete barcode and emit it. This is the same
        # heuristic hid/wedge uses (first-char -> last-char gap).
        idle_flush_seconds = 0.15
        last_rx_monotonic = 0.0

        # Heartbeat: prove to the log that run() is alive even when the
        # scanner has not sent anything. Without this, a silent scanner
        # and a hung read-loop look identical from the operator's side.
        heartbeat_interval_seconds = 10.0
        last_heartbeat_at = time.monotonic()
        ticks_since_heartbeat = 0

        while self._is_running:
            try:
                # Use whatever bytes have already arrived; fall back to a
                # small blocking read so the loop still respects timeout
                # when the scanner is idle, instead of spinning hot.
                waiting = getattr(self.serial_port, "in_waiting", 0) or 0
                chunk = self.serial_port.read(waiting if waiting > 0 else _READ_BYTE_FALLBACK)

                ticks_since_heartbeat += 1
                now_mono = time.monotonic()
                if now_mono - last_heartbeat_at >= heartbeat_interval_seconds:
                    port_is_open = False
                    try:
                        port_is_open = bool(getattr(self.serial_port, "is_open", False))
                    except Exception:
                        pass
                    self._debug_print(
                        f"heartbeat: port_open={port_is_open}, "
                        f"ticks_since_last_heartbeat={ticks_since_heartbeat}, "
                        f"buffered_bytes={len(buffer)}"
                    )
                    last_heartbeat_at = now_mono
                    ticks_since_heartbeat = 0

                if not chunk:
                    # No new bytes this tick: if we have a pending buffer
                    # and it has been idle long enough, flush it out as
                    # one barcode even without a matching terminator.
                    if buffer and last_rx_monotonic and (
                        now_mono - last_rx_monotonic >= idle_flush_seconds
                    ):
                        self._debug_print(
                            f"idle flush (未匹配到 terminator): hex={buffer.hex()} "
                            f"raw={buffer!r}"
                        )
                        self._decode_and_emit(buffer, encoding)
                        buffer = b""
                        last_rx_monotonic = 0.0
                    continue

                last_rx_monotonic = now_mono
                # Diagnostic trace: log every raw chunk so the operator
                # can see exactly what the scanner pushed. Useful when
                # the terminator / baudrate / encoding is misconfigured.
                self._debug_print(
                    f"raw chunk: len={len(chunk)} hex={chunk.hex()} bytes={chunk!r}"
                )

                buffer += chunk
                if len(buffer) > _MAX_BUFFER_BYTES:
                    # Protect against a scanner that keeps sending without
                    # ever flushing a terminator. Keep the tail so the next
                    # terminator still has a chance to arrive.
                    self.logger.warning(
                        f"[serial-barcode] 缓冲区超过 {_MAX_BUFFER_BYTES} 字节, 裁剪保留尾部"
                    )
                    buffer = buffer[-_MAX_BUFFER_BYTES // 2 :]

                while terminator_bytes and terminator_bytes in buffer:
                    line, _, buffer = buffer.partition(terminator_bytes)
                    self._debug_print(
                        f"terminator hit: line_hex={line.hex()} "
                        f"remaining_buffer_bytes={len(buffer)}"
                    )
                    self._decode_and_emit(line, encoding)
                    # Anything left in buffer belongs to the next barcode,
                    # reset the idle timer so we do not flush too early.
                    if buffer:
                        last_rx_monotonic = time.monotonic()
                    else:
                        last_rx_monotonic = 0.0
            except Exception as e:
                err_str = str(e)
                # Any error raised after we already asked for shutdown is
                # just the port being closed from stop() - swallow it.
                if not self._is_running and (
                    "句柄无效" in err_str
                    or "ClearCommError" in err_str
                    or "Bad file descriptor" in err_str
                    or "PortNotOpenError" in repr(e)
                ):
                    self._debug_print("检测到主动停止导致的句柄失效, 已忽略并退出")
                    break
                msg = f"串口扫码枪读取异常: {e}"
                self.logger.error(msg)
                self._debug_print(msg)
                self._emit_status(
                    running=self._is_running,
                    connected=False,
                    message=msg,
                    error=str(e),
                )
                time.sleep(1)

        port_is_open_at_exit = False
        try:
            port_is_open_at_exit = bool(
                self.serial_port is not None and getattr(self.serial_port, "is_open", False)
            )
        except Exception:
            pass
        self._debug_print(
            f"run() while 循环退出, _is_running={self._is_running}, "
            f"port_open={port_is_open_at_exit}, port={port}"
        )
        try:
            if port_is_open_at_exit:
                self.serial_port.close()
        except Exception:
            pass
        self._debug_print(f"监听结束: port={port}")
        self._emit_status(running=False, connected=False, message=f"串口扫码枪已断开: {port}")
