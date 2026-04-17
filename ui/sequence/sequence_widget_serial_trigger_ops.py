from PyQt5.QtWidgets import QMessageBox

from base.load_config import LoadUiConfig
from consts import error_code
from ui.serial_discrete_input_config_dialog import SerialDiscreteInputConfigDialog


class SequenceWidgetSerialTriggerOpsMixin:
    def _test_serial_trigger_connection(self, config):
        normalized_config = LoadUiConfig.normalize_serial_discrete_input_config(dict(config or {}))
        restart_config = None
        test_port = str((normalized_config.get("serial_settings", {}) or {}).get("port", "") or "")
        running_config = getattr(self.hw_manager, "serial_config", {}) or {}
        running_port = str((running_config.get("serial_settings", {}) or {}).get("port", "") or "")
        worker = getattr(self.hw_manager, "serial_worker", None)
        worker_running = bool(worker is not None and worker.isRunning())

        if worker_running and test_port and test_port == running_port:
            restart_config = LoadUiConfig.normalize_serial_discrete_input_config(dict(running_config))
            self.hw_manager.stop_serial_discrete_input_listener()

        ret = self.hw_manager.test_serial_discrete_input_connection(normalized_config)

        if restart_config and restart_config.get("enabled", False):
            restart_ret = self.hw_manager.start_serial_discrete_input_listener(restart_config)
            if not restart_ret.get("ok", False):
                QMessageBox.warning(
                    self,
                    "串口离散输入触发",
                    restart_ret.get("message", "测试后恢复监听失败"),
                )

        raw_hex = str(ret.get("raw_hex", "") or "")
        return {
            "connected": bool(ret.get("ok", False)),
            "has_response": bool(raw_hex),
            "message": str(ret.get("message", "") or "测试连接失败"),
            "raw_hex": raw_hex,
        }

    def init_serial_trigger_runtime(self):
        if getattr(self, "_serial_trigger_runtime_initialized", False):
            return
        self._serial_trigger_runtime_initialized = True
        err_code, data = LoadUiConfig.load_serial_discrete_input_config()
        if err_code == error_code.OK and isinstance(data, dict):
            self._serial_trigger_config = data
        else:
            self._serial_trigger_config = {}
        self.on_serial_trigger_status_changed(self.hw_manager.get_serial_discrete_input_status())
        if self._serial_trigger_config.get("enabled", False):
            self.hw_manager.start_serial_discrete_input_listener(self._serial_trigger_config)

    def on_serial_trigger_btn_clicked(self):
        current_config = getattr(self, "_serial_trigger_config", None)
        if not isinstance(current_config, dict) or not current_config:
            err_code, data = LoadUiConfig.load_serial_discrete_input_config()
            current_config = data if err_code == error_code.OK and isinstance(data, dict) else {}

        dialog = SerialDiscreteInputConfigDialog(
            current_config,
            runtime_status=self.hw_manager.get_serial_discrete_input_status(),
            test_connection_callback=self._test_serial_trigger_connection,
            parent=self,
        )
        result = dialog.exec()
        if not result:
            return

        _action, config = result
        self._serial_trigger_config = LoadUiConfig.normalize_serial_discrete_input_config(dict(config or {}))

        if not LoadUiConfig.save_serial_discrete_input_config(self._serial_trigger_config):
            QMessageBox.warning(self, "保存失败", "无法保存串口离散输入触发配置。")
            return

        if self._serial_trigger_config.get("enabled", False):
            ret = self.hw_manager.start_serial_discrete_input_listener(self._serial_trigger_config)
        else:
            self.hw_manager.stop_serial_discrete_input_listener()
            ret = {"ok": True, "message": "已关闭串口离散输入触发"}

        if not ret.get("ok", False):
            QMessageBox.warning(self, "串口离散输入触发", ret.get("message", "启动失败"))
        self.on_serial_trigger_status_changed(self.hw_manager.get_serial_discrete_input_status())

    def on_serial_trigger_status_changed(self, status):
        status = status or {}
        self._serial_trigger_runtime_status = dict(status)
        connected = bool(status.get("connected", False))
        has_response = bool(status.get("has_response", False))
        message = str(status.get("message", "") or "")
        if connected and has_response:
            status_text = "已连接"
        elif connected:
            status_text = "已打开"
        else:
            status_text = "未连接"

        try:
            self.serial_trigger_status_label.setText(status_text)
            self.serial_trigger_status_label.setToolTip(message)
        except Exception:
            pass
