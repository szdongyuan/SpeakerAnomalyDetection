import keyboard
import pywinusb.hid as hid
from PyQt5.QtCore import QObject, pyqtSignal

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

    sig_barcode = pyqtSignal(str)
    sig_trigger = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.logger = LogManager.set_log_handler("core")
        self.hid_handles = {}
        self.hotkey_registered = False
        self.hotkey_string = None

        # 配置缓存
        self.config_path = self.DEFAULT_CONFIG
        self.scanner_conf = None
        self.sensor_conf = None
        self.sensor_hotkey = None

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

    def start(self):
        """启动硬件监听（使用已加载的配置）"""
        if not self.scanner_conf and not self.sensor_conf:
            return False

        if self.scanner_conf:
            self._attach_hid_device(
                key="scanner",
                conf=self.scanner_conf,
                handler=self.on_scanner_data
            )

        if self.sensor_hotkey:
            self._register_hotkey(self.sensor_hotkey)
        elif self.sensor_conf:
            self.logger.warning("光电开关已配置 VID/PID，但未配置 hotkey，光电监听未启用")

        return True

    def stop(self):
        """停止所有监听"""
        # 关闭所有 HID 设备
        for key in list(self.hid_handles.keys()):
            self.close_hid_device(key)

        # 移除热键
        if self.hotkey_registered:
            try:
                keyboard.unhook_all_hotkeys()
                self.hotkey_registered = False
                self.logger.info(f"光电热键已移除: {self.hotkey_string}")
            except Exception as e:
                self.logger.warning(f"热键移除异常: {e}")

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
        opened = []
        for dev in matching:
            try:
                dev.open()
                dev.set_raw_data_handler(handler)
                opened.append(dev)
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
        for h in (handles if isinstance(handles, list) else [handles]):
            try:
                h.close()
            except:
                pass
        self.logger.info(f"[{key}] HID 设备已断开")

    def on_scanner_data(self, report):
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
                    self.sig_barcode.emit(barcode)
        except Exception as e:
            self.logger.warning(f"扫码枪数据解析异常: {e}")