"""One settings form; capture choices are requests, not advertised hardware capabilities."""

from dataclasses import replace
from collections import Counter
from fractions import Fraction
from hashlib import sha256

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QGridLayout,
    QLabel, QMessageBox, QPushButton, QSizePolicy, QVBoxLayout,
)

from consts import ui_style_const
from ui.dialog_enter_policy import install_dialog_enter_policy


class VideoSettingsDialog(QDialog):
    probe_requested = pyqtSignal()
    configuration_requested = pyqtSignal(object)

    def __init__(self, config, parent=None, *, read_only=False):
        super().__init__(parent)
        self.config = config
        self.read_only = read_only
        self._saving = False
        self._access = None
        self.setWindowTitle("摄像头设置")
        self.setMinimumWidth(440)
        self.setStyleSheet(
            f"QDialog {{background:#F7FAFD; font-family:{ui_style_const.MAIN_UI_SMALL_FONT_FAMILY}; font-size:13px;}}"
            f"QLabel,QCheckBox,QComboBox,QPushButton {{font-family:{ui_style_const.MAIN_UI_SMALL_FONT_FAMILY};"
            "font-size:13px; color:#30485E;}"
            "QLabel,QCheckBox {background:transparent;}"
            "QCheckBox {spacing:7px;}"
            "QComboBox {background:white; border:1px solid #B8CADC; border-radius:3px;"
            "min-height:28px; padding:0 7px; selection-background-color:#3B6FB6; selection-color:white;}"
            "QComboBox:disabled {background:#EEF2F6; color:#526273;}"
            "QCheckBox:disabled {color:#526273;}"
            "QPushButton {background:#EDF4FC; border:1px solid #B8CADC; border-radius:3px;"
            "min-height:28px; padding:0 12px;}"
            "QPushButton:hover {background:#DDEBFA;}"
            "QPushButton:disabled {background:#EEF2F6; color:#68788A;}"
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(14)
        form = QGridLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(10)
        form.setColumnStretch(1, 1)
        root.addLayout(form)
        self.enabled = QCheckBox("启用摄像头预览")
        self.enabled.setToolTip("保存后生效，下次启动沿用。")
        self.enabled.setChecked(config.enabled)
        form.addWidget(self.enabled, 0, 0, 1, 3)
        self.devices = QComboBox()
        self.devices.setPlaceholderText("请选择摄像头")
        self.devices.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.devices.setMinimumContentsLength(12)
        if config.device_id:
            self.devices.addItem(config.device_name or config.device_id, config.device_id)
            self.devices.setCurrentIndex(0)
        self.refresh_button = QPushButton("刷新")
        self.refresh_button.setFixedWidth(64)
        self.refresh_button.clicked.connect(self.probe_requested.emit)
        form.addWidget(QLabel("摄像头"), 1, 0)
        form.addWidget(self.devices, 1, 1)
        form.addWidget(self.refresh_button, 1, 2)
        self.resolution = QComboBox()
        resolutions = list(dict.fromkeys([
            (config.width, config.height),
            (640, 480), (1280, 720), (1920, 1080),
            (2560, 1440),
        ]))
        # A saved value must not reintroduce the removed preset.
        resolutions = [size for size in resolutions if size != (2048, 1536)]
        for width, height in resolutions:
            self.resolution.addItem(f"{width} × {height}", (width, height))
        if (config.width, config.height) not in resolutions:
            self.resolution.insertItem(0, "原分辨率已移除，请重新选择", None)
            self.resolution.model().item(0).setEnabled(False)
            self.resolution.setCurrentIndex(0)
        self.resolution.setToolTip("决定摄像头采集分辨率；所选分辨率必须由摄像头支持。")
        form.addWidget(QLabel("摄像头分辨率"), 2, 0)
        form.addWidget(self.resolution, 2, 1)
        self.rate = QComboBox()
        rates = [
            Fraction(config.fps_num, config.fps_den),
            10, 15, 20, 25, 30, 60,
        ]
        for rate in dict.fromkeys(rates):
            self.rate.addItem(f"{float(rate):g} 帧/秒", Fraction(rate))
        self.rate.setToolTip("影响画面流畅度；所选帧率必须由摄像头支持。")
        form.addWidget(QLabel("摄像头帧率"), 3, 0)
        form.addWidget(self.rate, 3, 1)
        self.message = QLabel()
        self.message.setTextFormat(Qt.PlainText)
        self.message.setStyleSheet("color:#657789; font-size:12px;")
        self.message.ensurePolished()
        self.message.setFixedHeight(self.message.fontMetrics().height())
        self.message.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.message.setToolTip(self.message.text())
        root.addWidget(self.message)
        self.read_only_label = QLabel("当前为只读：操作员权限下不能修改设置。")
        self.read_only_label.setWordWrap(True)
        self.read_only_label.setStyleSheet("color:#80551C; font-size:12px;")
        self.read_only_label.setVisible(read_only)
        root.addWidget(self.read_only_label)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Close)
        self.buttons.setStyleSheet(f"QDialogButtonBox {{ button-layout: {QDialogButtonBox.GnomeLayout}; }}")
        self.buttons.button(QDialogButtonBox.Save).setText("保存设置")
        self.buttons.button(QDialogButtonBox.Close).setText("关闭")
        self.buttons.accepted.connect(self.submit)
        self.buttons.rejected.connect(self.close)
        root.addWidget(self.buttons)
        self.set_access(read_only=read_only)

    def show_apply_status(self, message):
        self.message.setToolTip(message)
        self.message.setText(self.message.fontMetrics().elidedText(message, Qt.ElideRight, self.message.width()))

    def set_access(self, *, read_only, saving=False):
        access = (read_only, saving)
        if access == self._access:
            return
        self._access = access
        self.read_only, self._saving = access
        editable = not read_only and not saving
        self.enabled.setEnabled(editable)
        for widget in (self.devices, self.resolution, self.rate, self.refresh_button):
            widget.setEnabled(editable)
        self.buttons.button(QDialogButtonBox.Save).setEnabled(editable)
        self.read_only_label.setText(
            "正在保存设置…" if saving else "当前为只读：操作员权限下不能修改设置。"
        )
        self.read_only_label.setVisible(read_only or saving)
        install_dialog_enter_policy(
            self, self.buttons.button(QDialogButtonBox.Save if editable else QDialogButtonBox.Close)
        )

    def set_devices(self, devices, error=""):
        selected = self.devices.currentData() or self.config.device_id
        self.devices.clear()
        counts = Counter(device.name for device in devices)
        for device in devices:
            label = device.name
            if counts[label] > 1:
                label += f" [{sha256(device.device_id.encode()).hexdigest()[:8]}]"
            self.devices.addItem(label, device.device_id)
            index = self.devices.count() - 1
            self.devices.setItemData(index, device.device_id, Qt.ToolTipRole)
            self.devices.setItemData(index, device.name, Qt.UserRole + 1)
        index = self.devices.findData(selected)
        if index < 0 and selected:
            prefix = "列表读取失败，上次选择" if error else "未连接"
            self.devices.addItem(f"{prefix}：{self.config.device_name or selected}", selected)
            index = self.devices.count() - 1
        if not self.devices.count():
            self.devices.addItem("摄像头列表读取失败" if error else "未发现摄像头")
            self.devices.model().item(0).setEnabled(False)
        self.devices.setCurrentIndex(max(0, index))
        hint = "" if devices else "请连接 USB 摄像头后点“刷新”。"
        self.devices.setToolTip("摄像头列表读取失败，请点“刷新”重试。" if error else hint)
        self.refresh_button.setEnabled(not self.read_only and not self._saving)

    def submit(self):
        if self.read_only or self._saving:
            return
        try:
            resolution = self.resolution.currentData()
            if resolution is None:
                raise ValueError("请选择摄像头分辨率后再保存。")
            width, height = resolution
            rate = self.rate.currentData()
            config = replace(
                self.config, enabled=self.enabled.isChecked(),
                device_id=self.devices.currentData() or "",
                device_name=(
                    self.devices.itemData(self.devices.currentIndex(), Qt.UserRole + 1) or self.config.device_name
                ),
                width=width, height=height, fps_num=rate.numerator, fps_den=rate.denominator,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "配置无效", str(exc))
            return
        self.configuration_requested.emit(config)
