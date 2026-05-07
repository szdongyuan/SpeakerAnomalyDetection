from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR

try:
    from serial.tools import list_ports
except Exception:  # pragma: no cover - optional until pyserial is installed
    list_ports = None


class SerialDiscreteInputConfigDialog(QDialog):
    COMMON_BAUDRATES = ["1200", "2400", "4800", "9600", "19200", "38400", "57600", "115200"]
    NO_PORTS_TEXT = "未检测到可用串口"

    # Explicit contract: dotted paths in the saved config that the user can
    # actually change through this dialog. UnifiedHardwareManager uses this
    # list to decide whether the running serial worker needs to be restarted
    # after the operator clicks 确定:
    #   * any listed path differs between the new and currently-running
    #     config -> stop + start the worker with the new config
    #   * none of the listed paths differ -> keep the worker running and
    #     only refresh self.serial_config in place
    #
    # If you add a new control to this dialog that influences how the worker
    # opens or polls the serial port, ALSO append the matching dotted path
    # here, otherwise the change will be saved to disk but won't take effect
    # until the app is restarted. The companion unit test
    # ``unit_test/ui/test_serial_discrete_input_config_dialog.py`` locks
    # this list against ``_build_config()``'s output so drift fails fast.
    EDITABLE_PATHS = (
        "enabled",
        "device_model",
        "serial_settings.port",
        "serial_settings.baudrate",
    )

    def __init__(self, config: dict, runtime_status: dict = None, test_connection_callback=None, parent=None):
        super().__init__(parent)
        self.config = dict(config or {})
        self.runtime_status = dict(runtime_status or {})
        self._test_connection_callback = test_connection_callback
        self._dialog_action = None

        self.enabled_checkbox = QCheckBox("启用串口离散输入触发")
        self.port_combobox = QComboBox()
        self.baudrate_combobox = QComboBox()
        self.device_model_lineedit = QLineEdit()
        self.runtime_status_label = QLabel("当前状态：未连接")
        self.test_btn = QPushButton(" 测试连接 ")
        self.ok_btn = QPushButton(" 确  定 ")
        self.cancel_btn = QPushButton(" 取  消 ")

        self._init_ui()
        self._set_member_connect()
        self._set_values_from_config()

    def _init_ui(self):
        self.setWindowTitle("串口离散输入触发配置")
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowCloseButtonHint, True)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setMinimumSize(420, 360)
        self.resize(520, 420)

        self.port_combobox.setEditable(True)
        self.baudrate_combobox.setEditable(True)
        self.baudrate_combobox.addItems(self.COMMON_BAUDRATES)

        layout = QVBoxLayout()
        layout.addWidget(self.enabled_checkbox)
        layout.addWidget(self._create_groupbox("串口号", self.port_combobox))
        layout.addWidget(self._create_groupbox("波特率", self.baudrate_combobox))
        layout.addWidget(self._create_groupbox("设备型号", self.device_model_lineedit))
        layout.addStretch()
        layout.addLayout(self._create_btn_layout())
        self.setLayout(layout)

        self.setStyleSheet(
            ui_style_const.qpushbutton_style
            + ui_style_const.qlineedit_style
            + ui_style_const.qcheckbox_style
            + ui_style_const.qgroupbox_style
            + ui_style_const.qcombobox_style
            + ui_style_const.qlabel_style
            + ui_style_const.qdialog_style
        )

    def _set_member_connect(self):
        self.cancel_btn.clicked.connect(self.close)
        self.ok_btn.clicked.connect(self._on_ok_btn_clicked)
        self.test_btn.clicked.connect(self._on_test_btn_clicked)

    @staticmethod
    def _create_groupbox(title, widget):
        layout = QHBoxLayout()
        layout.addWidget(widget)
        groupbox = QGroupBox(title)
        groupbox.setLayout(layout)
        return groupbox

    def _create_btn_layout(self):
        layout = QHBoxLayout()
        layout.addWidget(self.test_btn)
        layout.addStretch()
        layout.addWidget(self.cancel_btn)
        layout.addWidget(self.ok_btn)
        return layout

    @staticmethod
    def _format_port_label(device, description):
        device = str(device or "").strip()
        description = str(description or "").strip()
        if not device:
            return description
        if not description or description == device:
            return device
        return f"{device} - {description}"

    def _get_selected_port_value(self):
        current_data = self.port_combobox.currentData()
        if current_data:
            return str(current_data).strip()

        current_text = str(self.port_combobox.currentText() or "").strip()
        if current_text == self.NO_PORTS_TEXT:
            return ""
        if " - " in current_text:
            return current_text.split(" - ", 1)[0].strip()
        return current_text

    def _load_port_options(self):
        self.port_combobox.clear()
        ports = []
        if list_ports is not None:
            try:
                ports = list(list_ports.comports())
            except Exception:
                ports = []
        self._available_ports = [str(p.device).strip() for p in ports if getattr(p, "device", None)]
        if ports:
            for port_info in ports:
                device = str(getattr(port_info, "device", "") or "").strip()
                if not device:
                    continue
                description = str(getattr(port_info, "description", "") or "").strip()
                self.port_combobox.addItem(self._format_port_label(device, description), device)
        else:
            self.port_combobox.addItem(self.NO_PORTS_TEXT, "")

    def _set_values_from_config(self):
        enabled = bool(self.config.get("enabled", False))
        serial_settings = self.config.get("serial_settings", {}) or {}

        # 先刷新当前可用串口，再决定默认选中项
        self._load_port_options()

        saved_port = str(serial_settings.get("port", "COM3") or "COM3")
        if getattr(self, "_available_ports", []):
            selected_port = saved_port if saved_port in self._available_ports else self._available_ports[0]
            selected_index = self.port_combobox.findData(selected_port)
            if selected_index >= 0:
                self.port_combobox.setCurrentIndex(selected_index)
            else:
                self.port_combobox.setEditText(selected_port)
        else:
            self.port_combobox.setCurrentIndex(0)

        self.enabled_checkbox.setChecked(enabled)
        self.baudrate_combobox.setCurrentText(str(serial_settings.get("baudrate", 9600) or 9600))
        self.device_model_lineedit.setText(str(self.config.get("device_model", "JY-DAM0404D") or "JY-DAM0404D"))
        self.update_runtime_status(self.runtime_status)

    def update_runtime_status(self, status):
        status = status or {}
        connected = bool(status.get("connected", False))
        has_response = bool(status.get("has_response", False))
        message = str(status.get("message", "") or "")
        raw_hex = str(status.get("raw_hex", "") or "")
        if connected and has_response:
            status_text = "已连接"
        elif connected:
            status_text = "待响应"
        else:
            status_text = "未连接"

        lines = [status_text]
        if message and message != status_text:
            lines.append(message)
        if raw_hex:
            lines.append(f"最近接收码：{raw_hex}")

        self.runtime_status = dict(status)
        self.runtime_status_label.setText("\n".join(lines))
        self.runtime_status_label.setToolTip("\n".join(lines))

    def _build_config(self):
        config = dict(self.config or {})
        serial_settings = dict(config.get("serial_settings", {}) or {})
        port = self._get_selected_port_value()
        baudrate_text = str(self.baudrate_combobox.currentText() or "").strip()
        device_model = str(self.device_model_lineedit.text() or "").strip()

        if not port:
            raise ValueError("未检测到可用串口，请检查设备连接")
        if not baudrate_text.isdigit():
            raise ValueError("波特率必须是数字")
        if not device_model:
            raise ValueError("设备型号不能为空")

        config["enabled"] = bool(self.enabled_checkbox.isChecked())
        config["device_model"] = device_model
        serial_settings["port"] = port
        serial_settings["baudrate"] = int(baudrate_text)
        config["serial_settings"] = serial_settings
        return config

    def _on_ok_btn_clicked(self):
        try:
            self.config = self._build_config()
        except ValueError as e:
            QMessageBox.warning(self, "配置无效", str(e))
            return
        self._dialog_action = "save"
        self.accept()

    def _on_test_btn_clicked(self):
        try:
            self.config = self._build_config()
        except ValueError as e:
            QMessageBox.warning(self, "配置无效", str(e))
            return
        callback = self._test_connection_callback
        if not callable(callback):
            self.update_runtime_status(
                {
                    "connected": False,
                    "has_response": False,
                    "message": "当前窗口未配置测试连接回调",
                    "raw_hex": "",
                }
            )
            return
        try:
            result = callback(dict(self.config or {})) or {}
        except Exception as e:
            result = {
                "connected": False,
                "has_response": False,
                "message": f"测试连接失败: {e}",
                "raw_hex": "",
            }
        self._show_test_result_popup(result)

    def _show_test_result_popup(self, result: dict):
        result = result or {}
        connected = bool(result.get("connected", False))
        has_response = bool(result.get("has_response", False))
        message = str(result.get("message", "") or "测试连接失败")
        raw_hex = str(result.get("raw_hex", "") or "")

        text = message
        if raw_hex:
            text = f"{text}\n\n最近接收码: {raw_hex}"

        # Only treat as successful when serial is reachable and response code is received.
        if connected and has_response:
            QMessageBox.information(self, "测试连接", text)
            return
        QMessageBox.warning(self, "测试连接", text)

    def exec(self):
        super().exec()
        if self._dialog_action:
            return self._dialog_action, self.config
        return None
