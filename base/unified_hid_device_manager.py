import keyboard
import pywinusb.hid as hid
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
import json
import os
import time

from base.load_config import LoadUiConfig
from base.log_manager import LogManager
from consts import error_code
from consts.running_consts import DEFAULT_DIR


class UnifiedHardwareManager(QObject):
    """
    统一硬件管理器
    - 扫码枪：HID（pywinusb）中断回调 + Qt 信号槽
    - 光电开关：全局键盘热键（keyboard 库）监听
    """

    DEFAULT_CONFIG = DEFAULT_DIR + "configs/scanner_barcode_config/scanner_hid_config.json"
    AUTO_POLL_INTERVAL_MS = 1000
    # “免配置自动识别”下的设备记忆（可选）：用于下次启动或复插时优先绑定同一把枪
    # 与 scanner_hid_config.json 配置分离：配置只负责“静态下发”，记忆只负责“运行时状态”。
    SCANNER_MEMORY_PATH = DEFAULT_DIR + "ui/ui_config/scanner_last_device.json"

    sig_barcode = pyqtSignal(str)
    sig_trigger = pyqtSignal()
    sig_directional_trigger = pyqtSignal(str)
    sig_serial_trigger_status = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.logger = LogManager.set_log_handler("core")
        # key -> {device_id: device_handle}
        self.hid_handles = {}
        self.hotkey_registered = False
        self.hotkey_string = None

        # 配置缓存
        self.config_path = self.DEFAULT_CONFIG
        self.scanner_conf = None
        self.sensor_conf = None
        self.sensor_hotkey = None

        # 扫码枪自动识别/热插拔
        self._scanner_enabled = False
        self._scanner_locked_device_id = None
        self._scanner_preferred_device_id = None
        self._scanner_preferred_vidpid = None
        self._auto_mode = False
        self._auto_no_device_logged = False

        self._hid_poll_timer = QTimer(self)
        self._hid_poll_timer.setInterval(self.AUTO_POLL_INTERVAL_MS)
        self._hid_poll_timer.timeout.connect(self._poll_hid_devices)

        # 尝试加载上次成功连接的扫码枪记忆（只在自动模式下作为优先候选）
        self._load_scanner_memory()

        # 条码去重机制：同一条码在短时间窗口内只发射一次（防止多HID接口重复触发）
        self._last_barcode = None
        self._last_barcode_time = 0.0
        self._barcode_dedup_window_sec = 0.5  # 500ms 去重窗口

        self.serial_worker = None
        self.serial_config = {}
        self.serial_trigger_armed = True
        self.serial_last_trigger_direction = ""
        self.serial_listener_status = {
            "enabled": False,
            "running": False,
            "connected": False,
            "message": "未启用",
            "raw_hex": "",
            "value": "",
            "action": "",
            "direction": "",
            "mode": "",
            "error": "",
            "device_model": "",
        }

    @staticmethod
    def _debug_print(message):
        print(f"[serial-trigger][manager] {message}")

    def _load_scanner_memory(self):
        try:
            if not os.path.exists(self.SCANNER_MEMORY_PATH):
                return
            with open(self.SCANNER_MEMORY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            if not isinstance(data, dict):
                return
            dev_id = data.get("device_id")
            vid = data.get("vid")
            pid = data.get("pid")
            if dev_id:
                self._scanner_preferred_device_id = str(dev_id)
            if vid is not None and pid is not None:
                self._scanner_preferred_vidpid = (int(vid), int(pid))
        except Exception as e:
            self.logger.warning(f"扫码枪记忆加载失败: {e}")

    def _save_scanner_memory(self, dev):
        """保存本次确认成功的扫码枪信息，用于下次启动优先匹配（写入独立记忆文件）"""
        try:
            os.makedirs(os.path.dirname(self.SCANNER_MEMORY_PATH), exist_ok=True)
            payload = {
                "device_id": self._device_id(dev),
                "vid": int(getattr(dev, "vendor_id", 0)),
                "pid": int(getattr(dev, "product_id", 0)),
                "vendor_name": getattr(dev, "vendor_name", "") or "",
                "product_name": getattr(dev, "product_name", "") or "",
            }
            with open(self.SCANNER_MEMORY_PATH, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            self._scanner_preferred_device_id = payload["device_id"]
            self._scanner_preferred_vidpid = (payload["vid"], payload["pid"])
        except Exception as e:
            self.logger.warning(f"扫码枪记忆保存失败: {e}")

    def ensure_config_loaded(self, config_path=None):
        """已加载则直接返回 True"""
        if self.scanner_conf or self.sensor_conf:
            return True

        path = config_path or self.config_path
        err_code, config_data = LoadUiConfig.load_data_from_json(path)
        if err_code != error_code.OK:
            self.logger.warning(f"HID 配置加载失败 (code={err_code}): {config_data}")
            return False

        self.parse_config(config_data)
        return True

    def parse_config(self, config_data):
        """解析配置 JSON"""
        if not config_data or not isinstance(config_data, dict):
            self.logger.warning("HID 配置格式错误, 期望字典")
            return

        scanner = config_data.get("scanner") or {}
        vid, pid = scanner.get("vid"), scanner.get("pid")
        if vid and pid:
            self.scanner_conf = (int(vid, 16), int(pid, 16))
            self.logger.info(f"扫码枪配置: VID={vid}, PID={pid}")

        sensor = config_data.get("sensor") or {}
        vid, pid = sensor.get("vid"), sensor.get("pid")
        if vid and pid:
            self.sensor_conf = (int(vid, 16), int(pid, 16))

        self.sensor_hotkey = sensor.get("hotkey")
        if self.sensor_hotkey:
            self.logger.info(f"光电开关热键: {self.sensor_hotkey}")
        elif sensor:
            self.logger.warning("光电开关缺少 hotkey, 光电触发将不可用")

    def _emit_serial_status(self, **kwargs):
        status = dict(self.serial_listener_status or {})
        status.update(kwargs)
        self.serial_listener_status = status
        self.sig_serial_trigger_status.emit(status)

    @staticmethod
    def _serial_config_path():
        return DEFAULT_DIR + "configs/scanner_barcode_config/serial_discrete_input.json"

    def load_serial_discrete_input_config(self):
        err_code, data = LoadUiConfig.load_serial_discrete_input_config()
        if err_code != error_code.OK or not isinstance(data, dict):
            return False, data
        self.serial_config = data
        return True, data

    def get_serial_discrete_input_status(self):
        return dict(self.serial_listener_status or {})

    def start_scanner_and_sensor_listeners(self):
        """
        仅启动扫码枪和光电热键监听。
        串口离散输入触发有自己的独立入口，不从这里拉起。
        """
        self.ensure_config_loaded()

        self._scanner_enabled = True
        self._auto_mode = not bool(self.scanner_conf)

        if self.scanner_conf:
            self._poll_hid_devices()
        else:
            self.logger.info("扫码枪未配置 VID/PID，进入自动识别模式：开启后插入扫码枪即可使用。")
            self._poll_hid_devices()

        if self.sensor_hotkey:
            self._register_hotkey(self.sensor_hotkey)
        elif self.sensor_conf:
            self.logger.warning("光电开关已配置 VID/PID，但未配置 hotkey，光电监听未启用")

        if self._scanner_enabled and not self._hid_poll_timer.isActive():
            self._hid_poll_timer.start()
        return True

    def stop_scanner_and_sensor_listeners(self):
        self._scanner_enabled = False
        self._scanner_locked_device_id = None
        self._auto_mode = False
        self._auto_no_device_logged = False
        self._last_barcode = None
        self._last_barcode_time = 0.0

        if self._hid_poll_timer.isActive():
            self._hid_poll_timer.stop()

        for key in list(self.hid_handles.keys()):
            self.close_hid_device(key)

        if self.hotkey_registered:
            try:
                keyboard.unhook_all_hotkeys()
                self.hotkey_registered = False
                self.logger.info(f"光电热键已移除: {self.hotkey_string}")
            except Exception as e:
                self.logger.warning(f"热键移除异常: {e}")

    def start_serial_discrete_input_listener(self, config_data=None):
        if self.serial_worker is not None and self.serial_worker.isRunning():
            self._debug_print("start 请求被忽略: 监听已在运行")
            self._emit_serial_status(message="串口离散输入监听已在运行")
            return {"ok": True, "message": "already running"}

        if config_data is not None:
            self.serial_config = LoadUiConfig.normalize_serial_discrete_input_config(config_data)
        else:
            ok, loaded = self.load_serial_discrete_input_config()
            if not ok:
                msg = f"串口离散输入配置加载失败: {loaded}"
                self._emit_serial_status(enabled=False, running=False, connected=False, message=msg, error=str(loaded))
                return {"ok": False, "message": msg}

        if not self.serial_config.get("enabled", False):
            self.serial_trigger_armed = True
            self.serial_last_trigger_direction = ""
            self._debug_print("start 请求结束: 配置存在但 enabled=False")
            self._emit_serial_status(
                enabled=False,
                running=False,
                connected=False,
                message="串口离散输入触发未启用",
                device_model=str(self.serial_config.get("device_model", "") or ""),
            )
            return {"ok": True, "message": "disabled"}

        from base.hardware_trigger.serial_discrete_input_worker import SerialDiscreteInputWorker

        self.serial_trigger_armed = True
        self.serial_last_trigger_direction = ""
        serial_settings = self.serial_config.get("serial_settings", {}) or {}
        decoder = self.serial_config.get("decoder", {}) or {}
        self._debug_print(
            "启动监听: "
            f"port={serial_settings.get('port', 'COM3')}, "
            f"baudrate={serial_settings.get('baudrate', 9600)}, "
            f"mode={decoder.get('mode', 'full_frame')}"
        )
        self.serial_worker = SerialDiscreteInputWorker(self.serial_config)
        self.serial_worker.sig_state_changed.connect(self._on_serial_state_changed)
        self.serial_worker.sig_status.connect(self._on_serial_worker_status)
        self.serial_worker.start()
        self._emit_serial_status(
            enabled=True,
            running=True,
            connected=False,
            message="正在启动串口离散输入监听",
            device_model=str(self.serial_config.get("device_model", "") or ""),
        )
        return {"ok": True, "message": "starting"}

    def stop_serial_discrete_input_listener(self):
        if self.serial_worker is not None:
            self._debug_print("停止监听")
            self.serial_worker.stop()
            self.serial_worker = None
        self.serial_trigger_armed = True
        self.serial_last_trigger_direction = ""
        self._emit_serial_status(
            enabled=bool(self.serial_config.get("enabled", False)),
            running=False,
            connected=False,
            message="串口离散输入监听已停止",
            action="",
            direction="",
            raw_hex="",
            value="",
        )

    def test_serial_discrete_input_connection(self, config_data=None):
        cfg = LoadUiConfig.normalize_serial_discrete_input_config(config_data or self.serial_config or {})
        serial_settings = cfg.get("serial_settings", {}) or {}
        polling_settings = cfg.get("polling_settings", {}) or {}
        port = serial_settings.get("port", "COM3")
        self._debug_print(
            "测试连接: "
            f"port={port}, "
            f"baudrate={serial_settings.get('baudrate', 9600)}"
        )
        try:
            import serial
        except Exception as e:
            return {"ok": False, "message": "pyserial 未安装", "raw_hex": "", "error": str(e)}

        try:
            ser = serial.Serial(
                port=port,
                baudrate=serial_settings.get("baudrate", 9600),
                bytesize=serial_settings.get("bytesize", 8),
                parity=serial_settings.get("parity", "N"),
                stopbits=serial_settings.get("stopbits", 1),
                timeout=serial_settings.get("timeout", 0.1),
            )
            try:
                ser.reset_input_buffer()
            except Exception:
                pass
            ser.write(bytes.fromhex(str(polling_settings.get("query_command_hex", "")).strip()))
            time.sleep(float(polling_settings.get("interval_ms", 50)) / 1000.0)
            received = ser.read(getattr(ser, "in_waiting", 0))
            ser.close()
            raw_hex = " ".join(f"{b:02X}" for b in received) if received else ""
            if raw_hex:
                self._debug_print(f"测试连接收到响应: raw_hex={raw_hex}")
                return {"ok": True, "message": "测试连接成功", "raw_hex": raw_hex}
            self._debug_print("测试连接成功，但未收到设备响应")
            return {"ok": False, "message": "测试连接成功，但未收到设备响应", "raw_hex": ""}
        except Exception as e:
            err_text = str(e)
            if "PermissionError" in repr(e) or "拒绝访问" in err_text:
                msg = f"测试连接失败: 串口 {port} 当前被占用，请关闭其他占用程序后重试"
                self._debug_print(msg)
                return {"ok": False, "message": msg, "raw_hex": "", "error": err_text}
            self._debug_print(f"测试连接失败: {e}")
            return {"ok": False, "message": f"测试连接失败: {e}", "raw_hex": "", "error": str(e)}

    def start(self):
        """
        为兼容旧逻辑保留的入口：
        仅启动扫码枪和光电热键监听。
        串口离散输入触发必须走独立入口 `start_serial_discrete_input_listener()`。
        """
        return self.start_scanner_and_sensor_listeners()

    def stop(self):
        """停止所有监听（扫码枪/光电/串口）"""
        self.stop_scanner_and_sensor_listeners()
        self.stop_serial_discrete_input_listener()

    def _register_hotkey(self, hotkey_string):
        """注册全局键盘热键"""
        if self.hotkey_registered:
            return
        self.hotkey_string = hotkey_string
        try:
            keyboard.add_hotkey(
                self.hotkey_string,
                self._on_hotkey_triggered,
                suppress=True
            )
            self.hotkey_registered = True
            self.logger.info(f"光电热键已注册: {self.hotkey_string}")
        except Exception as e:
            self.logger.error(f"热键注册失败: {e}（热键: {self.hotkey_string}）")

    def _on_hotkey_triggered(self):
        """热键触发回调"""
        self.logger.info(f"光电开关触发 (热键: {self.hotkey_string})")
        self.sig_trigger.emit()

    def _on_serial_worker_status(self, payload):
        if not isinstance(payload, dict):
            return
        self._emit_serial_status(
            enabled=bool(self.serial_config.get("enabled", False)),
            device_model=str(self.serial_config.get("device_model", "") or ""),
            **payload,
        )

    def _on_serial_state_changed(self, payload):
        """串口离散输入状态变化回调"""
        if not isinstance(payload, dict):
            return

        mode = str(payload.get("mode", self.serial_config.get("decoder", {}).get("mode", "full_frame")))
        state_code = str(payload.get("value", "") or "")
        raw_hex = str(payload.get("raw_hex", "") or "")
        state_map = (self.serial_config.get("state_maps", {}) or {}).get(mode, {}) or {}
        state_config = state_map.get(state_code)
        self._debug_print(
            f"收到状态变化: mode={mode}, state={state_code}, raw_hex={raw_hex}, armed={self.serial_trigger_armed}"
        )

        if not state_config:
            self.logger.warning(f"Unknown serial state code [{mode}]: {state_code}")
            self._debug_print(f"未匹配到状态映射: mode={mode}, state={state_code}")
            self._emit_serial_status(
                enabled=bool(self.serial_config.get("enabled", False)),
                mode=mode,
                raw_hex=raw_hex,
                value=state_code,
                action="unknown",
                direction="",
                message=f"未识别的状态码: {state_code}",
            )
            return

        action = str(state_config.get("action", "") or "")
        direction = str(state_config.get("direction", "") or "")
        desc = str(state_config.get("description", "") or action or "Unknown")
        self._debug_print(
            f"状态映射成功: action={action}, direction={direction}, desc={desc}, armed_before={self.serial_trigger_armed}"
        )

        if action == "start_record":
            allow_trigger = self.serial_trigger_armed or (
                bool(direction) and direction != self.serial_last_trigger_direction
            )
            if allow_trigger:
                self.serial_trigger_armed = False
                self.serial_last_trigger_direction = direction
                self.logger.info(f"串口离散输入触发响应: 方向={direction} (状态码={state_code})")
                self._debug_print(f"发出方向触发信号: direction={direction}")
                self.sig_directional_trigger.emit(direction)
                msg = f"串口触发: {direction}"
            else:
                msg = f"串口触发已锁定，等待 rearm: {direction}"
                self.logger.debug(msg)
                self._debug_print(msg)
        elif action in ("idle", "ignore"):
            self.serial_trigger_armed = True
            if action == "idle":
                self.serial_last_trigger_direction = ""
            msg = f"串口状态更新: {desc}，已 rearm"
            self.logger.debug(msg)
            self._debug_print(msg)
        else:
            msg = f"串口状态更新: {desc}"
            self.logger.debug(msg)
            self._debug_print(msg)

        self._emit_serial_status(
            enabled=bool(self.serial_config.get("enabled", False)),
            mode=mode,
            raw_hex=raw_hex,
            value=state_code,
            action=action,
            direction=direction,
            message=msg,
            armed=self.serial_trigger_armed,
        )

    def _attach_hid_device(self, key, conf, handler):
        """查找并打开 HID 设备，注册回调"""
        vid, pid = conf

        # 先关闭旧连接
        self.close_hid_device(key)

        # 查找匹配设备
        try:
            all_devices = hid.find_all_hid_devices()
        except Exception as e:
            self.logger.error(f"[{key}] HID 枚举失败: {e}")
            return

        matching = [d for d in all_devices if d.vendor_id == vid and d.product_id == pid]
        if not matching:
            self.logger.warning(f"[{key}] 未找到设备 (VID={hex(vid)}, PID={hex(pid)})")
            return

        # 连接所有匹配设备
        opened = {}
        for dev in matching:
            try:
                dev_id = self._device_id(dev)
                dev.open()
                dev.set_raw_data_handler(self._wrap_handler(handler, dev_id))
                opened[dev_id] = dev
            except Exception as e:
                self.logger.warning(f"[{key}] 接口连接失败: {e}")

        if opened:
            self.hid_handles[key] = opened
            self.logger.info(f"[{key}] HID 监听已启动 ({len(opened)} 个接口)")
        else:
            self.logger.error(f"[{key}] 所有接口连接失败")

    def close_hid_device(self, key):
        """关闭指定 key 的 HID 设备"""
        handles = self.hid_handles.pop(key, None)
        if not handles:
            return
        # 兼容历史 list/单对象结构，并支持 dict(device_id -> handle)
        if isinstance(handles, dict):
            iterable = list(handles.values())
        elif isinstance(handles, list):
            iterable = handles
        else:
            iterable = [handles]

        for h in iterable:
            try:
                h.close()
            except:
                pass
        self.logger.info(f"[{key}] HID 设备已断开")

    def on_scanner_data(self, report, device_id=None):
        """扫码枪数据回调"""
        try:
            payload = ''.join(chr(b) for b in report if 32 <= b <= 126)
            target_key = '"data":"'
            start_index = payload.find(target_key)
            if start_index != -1:
                start_index += len(target_key)
                end_index = payload.find('"', start_index)
                if end_index != -1:
                    barcode = payload[start_index:end_index]

                    # 条码去重：在短时间窗口内收到相同条码则忽略（防止多HID接口重复触发）
                    now = time.monotonic()
                    if (
                        self._last_barcode == barcode
                        and (now - self._last_barcode_time) < self._barcode_dedup_window_sec
                    ):
                        self.logger.debug(f"[scanner] 忽略重复条码 (去重窗口内): {barcode}")
                        return
                    # 更新去重记录
                    self._last_barcode = barcode
                    self._last_barcode_time = now

                    # 自动模式：第一次成功解析到条码后锁定设备，避免打开太多 HID
                    if self._auto_mode and device_id and not self._scanner_locked_device_id:
                        self._scanner_locked_device_id = device_id
                        # 给出"已确认/已锁定"的提示，便于现场确认识别成功
                        try:
                            dev = (self.hid_handles.get("scanner") or {}).get(device_id)
                            if dev is not None:
                                self.logger.info(
                                    f"[scanner] 已锁定扫码枪设备: {self._device_name(dev)} "
                                    f"(VID={hex(getattr(dev, 'vendor_id', 0))}, PID={hex(getattr(dev, 'product_id', 0))})"
                                )
                            else:
                                self.logger.info(f"[scanner] 已锁定扫码枪设备: {device_id}")
                        except Exception:
                            self.logger.info(f"[scanner] 已锁定扫码枪设备: {device_id}")
                        # 保存设备记忆：下次启动/复插优先只连这把枪
                        try:
                            dev = (self.hid_handles.get("scanner") or {}).get(device_id)
                            if dev is not None:
                                self._save_scanner_memory(dev)
                        except Exception:
                            pass
                        QTimer.singleShot(0, self._poll_hid_devices)
                    self.sig_barcode.emit(barcode)
        except Exception as e:
            self.logger.warning(f"扫码枪数据解析异常: {e}")

    def _wrap_handler(self, handler, device_id):
        """pywinusb 的 raw handler 只传 report，本方法把 device_id 透传进去"""
        def _inner(report):
            try:
                handler(report, device_id=device_id)
            except TypeError:
                # 兼容旧 handler 签名 handler(report)
                handler(report)
        return _inner

    @staticmethod
    def _device_id(dev):
        # pywinusb 通常有 device_path，可作为稳定唯一键
        return getattr(dev, "device_path", None) or f"{hex(dev.vendor_id)}:{hex(dev.product_id)}:{getattr(dev, 'product_name', '')}"

    @staticmethod
    def _device_name(dev):
        vendor = getattr(dev, "vendor_name", "") or ""
        product = getattr(dev, "product_name", "") or ""
        return (vendor + " " + product).strip()

    @staticmethod
    def _is_keyboard_or_mouse(dev):
        # 避免误打开系统键盘/鼠标（很多扫码枪是键盘模式；若是键盘模式，通常不需要 HID 解析，直接输入框扫码即可）
        try:
            caps = getattr(dev, "hid_caps", None)
            usage_page = getattr(caps, "usage_page", None)
            usage = getattr(caps, "usage", None)
            # Generic Desktop: 0x01; Mouse:0x02; Keyboard:0x06
            return usage_page == 0x01 and usage in (0x02, 0x06)
        except Exception:
            return False

    @staticmethod
    def _looks_like_scanner_by_name(dev):
        name = (getattr(dev, "product_name", "") or "") + " " + (getattr(dev, "vendor_name", "") or "")
        name = name.lower()
        keywords = ("barcode", "scanner", "scan", "qr", "2d", "code")
        return any(k in name for k in keywords)

    def _sync_open_devices(self, key, current_devices, handler):
        """
        current_devices: {device_id: dev}
        """
        existing = self.hid_handles.get(key) or {}
        if not isinstance(existing, dict):
            # 兼容历史结构
            try:
                existing = {self._device_id(d): d for d in (existing if isinstance(existing, list) else [existing])}
            except Exception:
                existing = {}

        # 关闭已拔出设备（仅在状态变化时提示一次，避免轮询刷屏）
        for dev_id in list(existing.keys()):
            if dev_id not in current_devices:
                try:
                    dev = existing.get(dev_id)
                    existing[dev_id].close()
                except Exception:
                    pass
                existing.pop(dev_id, None)
                if dev is not None:
                    try:
                        self.logger.info(
                            f"[{key}] 设备已断开: {self._device_name(dev)} "
                            f"(VID={hex(getattr(dev, 'vendor_id', 0))}, PID={hex(getattr(dev, 'product_id', 0))})"
                        )
                    except Exception:
                        self.logger.info(f"[{key}] 设备已断开: {dev_id}")

        # 打开新插入设备（仅在状态变化时提示一次，避免轮询刷屏）
        for dev_id, dev in current_devices.items():
            if dev_id in existing:
                continue
            try:
                dev.open()
                dev.set_raw_data_handler(self._wrap_handler(handler, dev_id))
                existing[dev_id] = dev
                try:
                    self.logger.info(
                        f"[{key}] 设备已连接: {self._device_name(dev)} "
                        f"(VID={hex(getattr(dev, 'vendor_id', 0))}, PID={hex(getattr(dev, 'product_id', 0))})"
                    )
                except Exception:
                    self.logger.info(f"[{key}] 设备已连接: {dev_id}")
            except Exception as e:
                self.logger.warning(f"[{key}] 接口连接失败: {e}")

        self.hid_handles[key] = existing

    def _poll_hid_devices(self):
        """定时轮询 HID 设备，支持扫码枪热插拔/自动识别"""
        if not self._scanner_enabled:
            return

        try:
            all_devices = hid.find_all_hid_devices()
        except Exception as e:
            self.logger.error(f"[scanner] HID 枚举失败: {e}")
            return

        # 配置模式：严格 VID/PID
        if self.scanner_conf:
            vid, pid = self.scanner_conf
            current = {}
            for d in all_devices:
                if d.vendor_id == vid and d.product_id == pid:
                    current[self._device_id(d)] = d
            self._sync_open_devices("scanner", current, self.on_scanner_data)
            return

        # 自动模式
        non_km = [d for d in all_devices if not self._is_keyboard_or_mouse(d)]

        # 先尝试“记住的设备”（若存在），避免每次都打开一堆候选设备
        if not self._scanner_locked_device_id and self._scanner_preferred_device_id:
            preferred = [d for d in non_km if self._device_id(d) == self._scanner_preferred_device_id]
            if preferred:
                current = {self._device_id(d): d for d in preferred}
                self._sync_open_devices("scanner", current, self.on_scanner_data)
                return

        # 次优：若换电脑/换 USB 口导致 device_path 变化，但 VID/PID 不变，可用 VID/PID 先收敛范围
        if not self._scanner_locked_device_id and self._scanner_preferred_vidpid:
            pvid, ppid = self._scanner_preferred_vidpid
            preferred_by_vidpid = [d for d in non_km if d.vendor_id == pvid and d.product_id == ppid]
            if preferred_by_vidpid:
                current = {self._device_id(d): d for d in preferred_by_vidpid}
                self._sync_open_devices("scanner", current, self.on_scanner_data)
                return

        # 若已锁定设备：只保留锁定的那个（插拔后自动解锁）
        if self._scanner_locked_device_id:
            current = {self._device_id(d): d for d in non_km if self._device_id(d) == self._scanner_locked_device_id}
            if not current:
                # 锁定设备已拔出，回到未锁定状态
                self._scanner_locked_device_id = None
            self._sync_open_devices("scanner", current, self.on_scanner_data)
            return

        # 优先按名称关键词筛一遍
        name_matched = [d for d in non_km if self._looks_like_scanner_by_name(d)]
        candidates = name_matched if name_matched else non_km

        current = {self._device_id(d): d for d in candidates}
        self._sync_open_devices("scanner", current, self.on_scanner_data)
        if current:
            self._auto_no_device_logged = False

        if not current and not self._auto_no_device_logged:
            self._auto_no_device_logged = True
            self.logger.info("自动识别模式下暂未发现可疑扫码枪 HID 设备：请插入扫码枪后再扫码。")