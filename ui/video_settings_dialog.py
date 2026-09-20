"""One settings form; capture choices are requests, not advertised hardware capabilities."""

from dataclasses import replace
from collections import Counter
from fractions import Fraction
from hashlib import sha256

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QGridLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QSizePolicy, QSpinBox, QVBoxLayout,
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
        self.setWindowTitle("摄像头与录像设置")
        self.setMinimumWidth(560)
        self.setStyleSheet(
            f"QDialog {{background:#F7FAFD; font-family:{ui_style_const.MAIN_UI_SMALL_FONT_FAMILY}; font-size:13px;}}"
            f"QLabel,QCheckBox,QComboBox,QSpinBox,QLineEdit,QPushButton {{font-family:{ui_style_const.MAIN_UI_SMALL_FONT_FAMILY};"
            "font-size:13px; color:#30485E;}"
            "QLabel,QCheckBox {background:transparent;}"
            "QCheckBox {spacing:7px;}"
            "QCheckBox::indicator {width:14px; height:14px; border:1px solid #8FA7BD;"
            "border-radius:2px; background:white;}"
            "QCheckBox::indicator:checked {background:#3B6FB6; border-color:#3B6FB6;}"
            "QCheckBox::indicator:disabled {border-color:#B8CADC; background:#EEF2F6;}"
            "QCheckBox::indicator:checked:disabled {background:#8DA6C4;}"
            "QComboBox,QSpinBox,QLineEdit {background:white; border:1px solid #B8CADC; border-radius:3px;"
            "min-height:28px; padding:0 7px; selection-background-color:#3B6FB6; selection-color:white;}"
            "QComboBox:disabled,QSpinBox:disabled,QLineEdit:disabled {background:#EEF2F6; color:#526273;}"
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
        self.enabled.setToolTip("只开启预览；录像由主界面的开始/停止按钮控制。")
        self.enabled.setChecked(config.enabled)
        form.addWidget(self.enabled, 0, 0, 1, 3)
        self.devices = QComboBox()
        self.devices.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.devices.setMinimumContentsLength(12)
        if config.device_id:
            self.devices.addItem(config.device_name or config.device_id, config.device_id)
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
            (2048, 1536), (2560, 1440),
        ]))
        for width, height in resolutions:
            self.resolution.addItem(f"{width} × {height}", (width, height))
        self.resolution.setToolTip("影响画面清晰度；所选分辨率必须由摄像头支持。")
        form.addWidget(QLabel("分辨率"), 2, 0)
        form.addWidget(self.resolution, 2, 1)
        self.rate = QComboBox()
        rates = [
            Fraction(config.fps_num, config.fps_den),
            10, 15, 20, 25, 30, 60,
        ]
        for rate in dict.fromkeys(rates):
            self.rate.addItem(f"{float(rate):g} 帧/秒", Fraction(rate))
        self.rate.setToolTip("影响画面流畅度；所选帧率必须由摄像头支持。")
        form.addWidget(QLabel("帧率"), 3, 0)
        form.addWidget(self.rate, 3, 1)
        self.bitrate = QSpinBox()
        self.bitrate.setRange(100, 100_000)
        self.bitrate.setSuffix(" kbps")
        self.bitrate.setValue(config.target_bitrate_bps // 1000)
        self.bitrate.setToolTip("影响录像画质与容量；数值越大，通常占用的磁盘空间越多。")
        form.addWidget(QLabel("录像码率"), 4, 0)
        form.addWidget(self.bitrate, 4, 1)
        self.folder = QLineEdit(config.recording_root)
        self.folder.setPlaceholderText("选择根目录，录像存入其下的video文件夹")
        self.folder.setToolTip("选择video的上一级目录；已有video时直接复用，没有则创建。\n连续录像每2小时换一个文件；空间不足时停止保存并提示。")
        self.browse_button = QPushButton("浏览…")
        self.browse_button.setFixedWidth(64)
        self.browse_button.clicked.connect(self.choose_directory)
        form.addWidget(QLabel("保存位置"), 5, 0)
        form.addWidget(self.folder, 5, 1)
        form.addWidget(self.browse_button, 5, 2)
        self.message = QLabel("点击“刷新”查找USB摄像头。")
        self.message.setTextFormat(Qt.PlainText)
        self.message.setStyleSheet("color:#657789; font-size:12px;")
        self.message.ensurePolished()
        self.message.setFixedHeight(self.message.fontMetrics().height())
        self.message.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.message.setToolTip(self.message.text())
        root.addWidget(self.message)
        self.read_only_label = QLabel("当前为只读：录像期间或操作员权限下不能修改设置。")
        self.read_only_label.setWordWrap(True)
        self.read_only_label.setStyleSheet("color:#80551C; font-size:12px;")
        self.read_only_label.setVisible(read_only)
        root.addWidget(self.read_only_label)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Close)
        self.buttons.button(QDialogButtonBox.Save).setText("保存设置")
        self.buttons.button(QDialogButtonBox.Close).setText("关闭")
        self.buttons.accepted.connect(self.submit)
        self.buttons.rejected.connect(self.close)
        root.addWidget(self.buttons)
        if read_only:
            for widget in (
                self.enabled, self.devices, self.resolution, self.rate,
                self.bitrate, self.folder, self.browse_button,
            ):
                widget.setEnabled(False)
            self.buttons.button(QDialogButtonBox.Save).setEnabled(False)
        install_dialog_enter_policy(
            self, self.buttons.button(QDialogButtonBox.Close if read_only else QDialogButtonBox.Save)
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
            self.devices.addItem(f"未连接：{self.config.device_name or selected}", selected)
            index = self.devices.count() - 1
        if index >= 0:
            self.devices.setCurrentIndex(index)
        if not self.devices.count():
            self.devices.setPlaceholderText("未发现摄像头")
        message = error or ("" if devices else "未发现USB摄像头，请连接后刷新。")
        # Keep the same row even when blank; long errors remain available in the tooltip.
        self.message.setToolTip(message)
        self.message.setText(self.message.fontMetrics().elidedText(message, Qt.ElideRight, self.message.width()))
        self.refresh_button.setEnabled(True)

    def choose_directory(self):
        path = QFileDialog.getExistingDirectory(self, "选择录像根目录（录像存入其下的video文件夹）", self.folder.text())
        if path:
            self.folder.setText(path)

    def submit(self):
        if self.read_only:
            return
        try:
            width, height = self.resolution.currentData()
            rate = self.rate.currentData()
            config = replace(
                self.config, enabled=self.enabled.isChecked(),
                device_id=self.devices.currentData() or "",
                device_name=(
                    self.devices.itemData(self.devices.currentIndex(), Qt.UserRole + 1) or self.config.device_name
                ),
                width=width, height=height, fps_num=rate.numerator, fps_den=rate.denominator,
                target_bitrate_bps=self.bitrate.value() * 1000, recording_root=self.folder.text().strip(),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "配置无效", str(exc))
            return
        self.configuration_requested.emit(config)
