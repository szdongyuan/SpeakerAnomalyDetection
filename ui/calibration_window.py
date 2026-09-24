import os
import sys
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from numbers import Integral

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QComboBox,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QLabel,
)
from PyQt5.QtWidgets import QMessageBox, QVBoxLayout, QPushButton, QSpacerItem
from PyQt5.QtWidgets import QSizePolicy, QWidget, QRadioButton

from base.log_manager import LogManager
from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis
from base.recording_process_protocol import RecordingRequest
from base.recording_service import RecordingCallbacks, RecordingService
from base.ve3668n_calibration import verify_ve_calibration_result
from base.ve3668n_input import calibration_fingerprint, validate_device_snapshot
from base.ve3668n_stores import VEStoreIOError
from ui.recording_service_bridge import RecordingProcessorFacade, RecordingServiceBridge
from base.soundcard_audio_processor import SoundcardAudioProcessor
from base.soundcard_calibration_manager import (
    MicCalibrationFormatError,
    MicCalibrationIOError,
    clear_mic_channel_calibrations,
    load_mic_channel_v2pa_factors,
    save_mic_channel_calibration,
)
from consts import ui_style_const, error_code
from consts.running_consts import DEFAULT_DIR
from ui.vkinging_presentation import device_display_name, ve_failure_text
from ui.dialog_enter_policy import install_dialog_enter_policy


class CalibrationWindow(QDialog):

    def __init__(self, input_device=None, input_channels=None, *, recording_bridge=None,
                 ve_profile_store=None, ve_calibration_store=None, ve_queue_config_provider=None):
        super().__init__()
        self.recording_bridge = recording_bridge
        self.ve_profile_store = ve_profile_store
        self.ve_calibration_store = ve_calibration_store
        self.ve_queue_config_provider = ve_queue_config_provider
        self.input_device = input_device
        self.input_channels = list(input_channels or [])
        self.init_ui()
        install_dialog_enter_policy(self, None)

    def init_ui(self):
        """
        Initialize the user interface for the calibration window.
        This function sets up the window icon, title, size, and layout,
        and creates the embedded input calibration panel.
        """
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setWindowTitle("校准窗口")
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setMinimumSize(500, 550)
        self.setMaximumSize(600, 580)
        cal_wnd_layout = QVBoxLayout()

        self.input_calibration_flag = False

        self.input_cal_wnd = InputCalibration(
            input_device=self.input_device,
            input_channels=self.input_channels,
            recording_bridge=self.recording_bridge,
            ve_profile_store=self.ve_profile_store,
            ve_calibration_store=self.ve_calibration_store,
            ve_queue_config_provider=self.ve_queue_config_provider,
        )
        input_panel = QGroupBox(self)
        panel_layout = QVBoxLayout(input_panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(self.input_cal_wnd)
        self.input_cal_wnd.calibration_finished.connect(
            self._on_input_calibration_finished
        )
        self.input_cal_wnd.calibration_state_changed.connect(
            self._on_input_calibration_state_changed
        )
        self.input_cal_wnd.calibration_availability_changed.connect(self._sync_calibration_button_state)

        btn_layout = self.create_btn_box()
        self._sync_calibration_button_state()

        cal_wnd_layout.addWidget(input_panel)
        cal_wnd_layout.addLayout(btn_layout)
        self.setLayout(cal_wnd_layout)
        self.setStyleSheet(ui_style_const.qpushbutton_style + ui_style_const.qgroupbox_style)

    def create_btn_box(self):
        """
        Create a button box

        This method creates a horizontal layout containing calibration, reset, and cancel buttons.
        Spacers are used to adjust the spacing between the buttons in the layout.
        """
        btn_layout = QHBoxLayout()
        self.cal_btn = QPushButton(" 校  准 ")
        self.cal_btn.clicked.connect(self.clicked_calibration_button)
        self.reset_btn = QPushButton(" 重  置 ")
        self.reset_btn.clicked.connect(self.clicked_reset_button)
        cancel_btn = QPushButton(" 退  出 ")
        cancel_btn.clicked.connect(self.clicked_close_button)
        h_spacer_btn1 = QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        h_spacer_btn2 = QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        btn_layout.addWidget(self.cal_btn)
        btn_layout.addItem(h_spacer_btn1)
        btn_layout.addWidget(self.reset_btn)
        btn_layout.addItem(h_spacer_btn2)
        btn_layout.addWidget(cancel_btn)
        return btn_layout

    def clicked_calibration_button(self):
        """Start input calibration when its controls are available."""
        if not self._input_calibration_controls_available():
            self._sync_calibration_button_state()
            return
        self.cal_btn.setDisabled(True)
        self.input_cal_wnd.clicked_calibration()
        self._sync_calibration_button_state()

    def _on_input_calibration_finished(self, _success):
        self._sync_calibration_button_state()

    def _on_input_calibration_state_changed(self, changed):
        if changed:
            self.input_calibration_flag = True
        self._sync_calibration_button_state()

    def _input_calibration_controls_available(self):
        bridge = self.input_cal_wnd.recording_bridge
        if (self.input_cal_wnd._ve_input and bridge is not None
                and not self.input_cal_wnd._can_start_recording_workflow()):
            return False
        return (
            self.input_cal_wnd.calibration_available
            and self.input_cal_wnd.current_channel is not None
            and self.input_cal_wnd.streaming_processor is None
        )

    def _sync_calibration_button_state(self, _index=None):
        enabled = self._input_calibration_controls_available()
        self.cal_btn.setEnabled(enabled)
        self.reset_btn.setEnabled(enabled)
        if self.input_cal_wnd._ve_input:
            self.input_cal_wnd.channel_combo_box.setEnabled(enabled)
            self.input_cal_wnd.standard_spl_i.setEnabled(enabled)
            self.input_cal_wnd.standard_spl_ii.setEnabled(enabled)

    def clicked_reset_button(self):
        """Reset input calibration when its controls are available."""
        if not self._input_calibration_controls_available():
            self._sync_calibration_button_state()
            return
        self.input_cal_wnd.reset_btn_clicked()
        self._sync_calibration_button_state()

    def clicked_close_button(self):
        """Close the window and release input recording resources."""
        self.reject()

    def reject(self):
        self.input_cal_wnd.close_recording()
        super().reject()

    def closeEvent(self, event):
        self.input_cal_wnd.close_recording()
        super().closeEvent(event)

    def done(self, result):
        self.input_cal_wnd.close_recording()
        super().done(result)


@dataclass(frozen=True)
class _VECalibrationContext:
    request: RecordingRequest
    standard_spl: float
    calibrated_at: str


class InputCalibration(QWidget):
    calibration_finished = pyqtSignal(bool)
    calibration_state_changed = pyqtSignal(bool)
    calibration_availability_changed = pyqtSignal()

    def __init__(self, input_device=None, input_channels=None, *, recording_bridge=None,
                 ve_profile_store=None, ve_calibration_store=None, ve_queue_config_provider=None):
        super().__init__()
        self.recording_bridge = recording_bridge
        self.ve_profile_store = ve_profile_store
        self.ve_calibration_store = ve_calibration_store
        self.ve_queue_config_provider = ve_queue_config_provider
        self._ve_input = (input_device or {}).get("backend") == "vkinging"
        self._ve_capture_context = None
        self._ve_accepted_audio = None
        self._ve_calibration_records = {}
        self._ve_display_device = None
        self._owns_recording_bridge = False
        self._recording_closed = False
        self._release_notice_session = None
        if recording_bridge is not None:
            recording_bridge.shutting_down.connect(self.close_recording)
        self.input_device = input_device
        self.input_channels = self._normalize_input_channels(input_channels)
        self.saved_v2pa_factors = {}
        self.current_channel = None
        self.calibration_available = False
        self.calibration_unavailable_message = None
        self.default_logger = LogManager.set_log_handler("core")  # Configures and retrieves the logger
        self.stop_timer = False  # Initializes the stop timer flag to False
        self.update_ui_timer = QTimer()
        self.update_ui_timer.setInterval(1000)
        self.update_ui_timer.timeout.connect(self.update_recorded_time)

        # Streaming recording state (no waveform display needed)
        self.streaming_processor = None
        self.active_capture_channel = None

        self.init_ui()
        self._initialize_calibration_state()

    def _can_start_recording_workflow(self):
        bridge = self.recording_bridge
        return bridge is None or not bridge.service.busy

    @staticmethod
    def _normalize_input_channels(input_channels):
        normalized = []
        seen = set()
        for channel in input_channels or []:
            if isinstance(channel, bool) or not isinstance(channel, Integral):
                continue
            physical_channel = int(channel)
            if physical_channel < 0 or physical_channel in seen:
                continue
            seen.add(physical_channel)
            normalized.append(physical_channel)
        return normalized

    def _initialize_calibration_state(self):
        if self._ve_input:
            self.refresh_ve_calibration_state()
            return
        if self.input_device is None:
            self._set_calibration_unavailable("未选择输入设备")
            return
        if not self.input_channels:
            self._set_calibration_unavailable("未选择有效输入通道")
            return

        try:
            self.saved_v2pa_factors = load_mic_channel_v2pa_factors(
                self.input_device
            )
        except (MicCalibrationFormatError, MicCalibrationIOError) as exc:
            self.default_logger.error(
                f"Failed to load input calibration registry: {exc}"
            )
            self._set_calibration_unavailable("状态: 输入校准文件错误")
            QMessageBox.critical(
                self,
                "输入校准错误",
                "输入校准文件错误，无法进行输入校准",
            )
            return

        self.calibration_available = True
        self.calibration_unavailable_message = None
        self.channel_combo_box.setEnabled(True)
        initial_channel = next(
            (
                channel
                for channel in self.input_channels
                if channel not in self.saved_v2pa_factors
            ),
            self.input_channels[0],
        )
        self.channel_combo_box.setCurrentIndex(
            self.channel_combo_box.findData(initial_channel)
        )
        self._on_channel_changed(self.channel_combo_box.currentIndex())

    def _current_ve_device(self):
        if self.ve_profile_store is None or self.ve_calibration_store is None:
            raise ValueError("VE 输入校准需要共享设备配置和校准存储")
        if not isinstance(self.input_device, Mapping):
            raise ValueError("VE 输入设备身份无效")
        if self.ve_queue_config_provider is not None:
            from base.ve3668n_recording_config import resolve_ve_recording_device

            context = self._ve_capture_context
            if context is not None:
                # Active capture owns its complete config. Only live identity and
                # channel availability remain subject to the stale-result checks.
                return validate_device_snapshot({
                    **self.input_device, "input_config": context.request.device["input_config"],
                })
            detail = self.ve_queue_config_provider()
            fallback = (self.ve_profile_store.load(self.input_device, self.ve_calibration_store)
                        if "sample_rate" not in detail else None)
            return resolve_ve_recording_device(self.input_device, detail, fallback_profile=fallback)
        identity = calibration_fingerprint(self.input_device, 0, self.input_device.get("input_config"))
        try:
            self.ve_calibration_store.observe(self.input_device)
            profile = self.ve_profile_store.load(self.input_device, self.ve_calibration_store)
        except ValueError:
            # Observation can durably invalidate records before an unsupported
            # profile is rejected. Read with the last supported identity to
            # show that sticky status, including a failed attempt to start.
            previous = self._ve_display_device
            if previous is not None and identity["machine_id"] == previous["machine_id"]:
                self._set_ve_records(self.ve_calibration_store.observe(previous))
            raise
        return validate_device_snapshot({**self.input_device, "input_config": profile})

    def _set_ve_records(self, records):
        self._ve_calibration_records = records
        self.saved_v2pa_factors = {
            channel: record["v2pa_factor"] for channel, record in records.items()
            if record["status"] == "valid"
        }

    def refresh_ve_calibration_state(self):
        """Refresh shared VE state without resetting selection or provenance.

        The active queue supplies recording parameters; standalone calibration
        uses the hardware profile. A rate-only refresh neither writes
        calibration nor produces a recalibration popup.
        """
        if (not self._ve_input or self._recording_closed or self.streaming_processor is not None
                or not self._can_start_recording_workflow()):
            return False
        try:
            device = self._current_ve_device()
            if (not device["available"] or not self.input_channels
                    or not set(self.input_channels).issubset(device["physical_channels"])):
                raise ValueError("VE 输入设备或物理通道不可用")
            records = self.ve_calibration_store.observe(device)
        except (ValueError, VEStoreIOError) as exc:
            machine_id = self.input_device.get("machine_id") if isinstance(self.input_device, Mapping) else None
            self.default_logger.error(
                f"VE input calibration unavailable machine_id={machine_id}: {exc}")
            self._set_calibration_unavailable(ve_failure_text("unavailable"))
            return False
        self._ve_display_device = device
        self._set_ve_records(records)
        self.calibration_available = True
        self.calibration_unavailable_message = None
        self.channel_combo_box.setEnabled(True)
        self.standard_spl_i.setEnabled(True)
        self.standard_spl_ii.setEnabled(True)
        selected = self.current_channel
        if selected not in self.input_channels:
            selected = next((channel for channel in self.input_channels
                             if channel not in self.saved_v2pa_factors), self.input_channels[0])
        self._select_channel(selected)
        return True

    def _set_calibration_unavailable(self, message):
        self.calibration_available = False
        self.calibration_unavailable_message = message
        self.current_channel = None
        self.channel_combo_box.blockSignals(True)
        self.channel_combo_box.setCurrentIndex(-1)
        self.channel_combo_box.blockSignals(False)
        self.channel_combo_box.setEnabled(False)
        self.channel_status_label.setText(message)
        self.v2pa_factor_lineedit.clear()
        if self._ve_input:
            self.standard_spl_i.setEnabled(False)
            self.standard_spl_ii.setEnabled(False)

    def _on_channel_changed(self, index):
        if not self.calibration_available or index < 0:
            self.current_channel = None
            return
        if self.active_capture_channel is not None:
            pinned_index = self.channel_combo_box.findData(
                self.active_capture_channel
            )
            if index != pinned_index:
                self.channel_combo_box.blockSignals(True)
                self.channel_combo_box.setCurrentIndex(pinned_index)
                self.channel_combo_box.blockSignals(False)
            self.current_channel = self.active_capture_channel
            return
        self.current_channel = self.channel_combo_box.itemData(index)
        self._refresh_channel_display()

    def _begin_capture(self, physical_channel):
        self.active_capture_channel = physical_channel
        self.channel_combo_box.setEnabled(False)
        self.channel_status_label.setText("状态: 录制中")
        if self._ve_input:
            self.standard_spl_i.setEnabled(False)
            self.standard_spl_ii.setEnabled(False)

    def _clear_active_capture(self, refresh_display=True):
        captured_channel = self.active_capture_channel
        self.active_capture_channel = None
        if self._ve_input:
            self._ve_capture_context = None
            self._ve_accepted_audio = None
            enabled = self.calibration_available and not self._recording_closed
            enabled = enabled and self._can_start_recording_workflow()
            self.standard_spl_i.setEnabled(enabled)
            self.standard_spl_ii.setEnabled(enabled)
        if not self.calibration_available:
            return
        if captured_channel is not None:
            captured_index = self.channel_combo_box.findData(captured_channel)
            if captured_index >= 0:
                self.channel_combo_box.blockSignals(True)
                self.channel_combo_box.setCurrentIndex(captured_index)
                self.channel_combo_box.blockSignals(False)
                self.current_channel = captured_channel
        self.channel_combo_box.setEnabled(enabled if self._ve_input else True)
        if refresh_display:
            self._refresh_channel_display()

    def _refresh_channel_display(self):
        if self._ve_input:
            record = self._ve_calibration_records.get(self.current_channel)
            status = record["status"] if record is not None else "none"
            text = {"none": "未校准，仅电压数据", "valid": "实测校准有效",
                    "invalidated": "配置已变更，需重新校准"}[status]
            self.channel_status_label.setText("状态: " + text)
            factor = self.saved_v2pa_factors.get(self.current_channel)
            self.v2pa_factor_lineedit.setText(
                str(np.round(float(factor), decimals=6)) if factor is not None else "")
            return
        factor = self.saved_v2pa_factors.get(self.current_channel)
        if factor is None:
            self.channel_status_label.setText("状态: 未校准")
            self.v2pa_factor_lineedit.clear()
            return
        self.channel_status_label.setText("状态: 已校准")
        self.v2pa_factor_lineedit.setText(
            str(np.round(float(factor), decimals=6))
        )

    def _select_channel(self, physical_channel):
        index = self.channel_combo_box.findData(physical_channel)
        self.channel_combo_box.blockSignals(True)
        self.channel_combo_box.setCurrentIndex(index)
        self.channel_combo_box.blockSignals(False)
        self.current_channel = physical_channel if index >= 0 else None
        self._refresh_channel_display()

    def _next_uncalibrated_channel(self, completed_channel):
        current_index = self.input_channels.index(completed_channel)
        search_order = (
            self.input_channels[current_index + 1 :]
            + self.input_channels[:current_index]
        )
        return next(
            (
                channel
                for channel in search_order
                if channel not in self.saved_v2pa_factors
            ),
            None,
        )

    def _success_popup_message(self, factor, next_channel):
        message = (
            "校准成功\n"
            f"本次校准结果：{float(factor):.6f} Pa/V"
        )
        if next_channel is not None:
            index = self.channel_combo_box.findData(next_channel)
            label = self.channel_combo_box.itemText(index)
            message += f"\n下次校准通道：{label}"
        return message

    def init_ui(self):
        """
        Initializes the user interface.

        This method sets up the window title, window properties, and size constraints.
        It also initializes the UI layout and applies custom stylesheets to various widgets.
        """
        self.setMinimumSize(305, 373)
        self.setMaximumSize(520, 500)
        self.standard_spl_flag = True
        self.recorded_flag = False

        input_device_box = self.create_input_device_box()
        standard_spl_box = self.create_standard_spl_box()
        recorded_box = self.create_recorded_box()
        v2pa_factor_box = self.create_v2pa_factor_box()

        v_spacer_1 = QSpacerItem(30, 30, QSizePolicy.Minimum, QSizePolicy.Expanding)
        v_spacer_2 = QSpacerItem(30, 30, QSizePolicy.Minimum, QSizePolicy.Expanding)
        v_spacer_3 = QSpacerItem(30, 30, QSizePolicy.Minimum, QSizePolicy.Expanding)

        layout = QVBoxLayout()
        layout.addWidget(input_device_box)
        layout.addWidget(standard_spl_box)
        layout.addItem(v_spacer_1)
        layout.addWidget(recorded_box)
        layout.addItem(v_spacer_2)
        layout.addWidget(v2pa_factor_box)
        if self._ve_input:
            reminder = QLabel("更换麦克风后请清除该通道旧校准并重新校准")
            reminder.setWordWrap(True)
            layout.addWidget(reminder)
        layout.addItem(v_spacer_3)
        layout.setContentsMargins(12, 20, 12, 25)

        self.setLayout(layout)
        self.setStyleSheet(
            ui_style_const.qcombobox_style
            + ui_style_const.qpushbutton_style
            + ui_style_const.qspinbox_style
            + ui_style_const.qdoublespinbox_style
            + ui_style_const.qgroupbox_style
            + ui_style_const.qlabel_style
            + ui_style_const.qlineedit_style
            + ui_style_const.qradiobutton_style
        )

    def create_input_device_box(self):
        input_device_box = QGroupBox("校准输入")
        device_name = "未选择输入设备"
        if self.input_device is not None:
            getter = getattr(self.input_device, "get", None)
            if callable(getter):
                device_name = (device_display_name(self.input_device) if self._ve_input
                               else str(getter("name") or device_name))

        self.input_device_label = QLabel(device_name)
        self.input_device_label.setWordWrap(True)
        self.channel_combo_box = QComboBox()
        for channel in self.input_channels:
            self.channel_combo_box.addItem(f"In{channel + 1}", channel)
        self.channel_combo_box.currentIndexChanged.connect(
            self._on_channel_changed
        )
        self.channel_status_label = QLabel("状态: 未校准")

        input_device_layout = QGridLayout()
        input_device_layout.addWidget(self.input_device_label, 0, 0, 1, 2)
        input_device_layout.addWidget(QLabel("输入通道"), 1, 0)
        input_device_layout.addWidget(self.channel_combo_box, 1, 1)
        input_device_layout.addWidget(self.channel_status_label, 2, 0, 1, 2)
        input_device_box.setLayout(input_device_layout)
        return input_device_box

    def create_v2pa_factor_box(self):
        """
        Create a QGroupBox to display the sound pressure v2pa_factor.

        This method creates a QGroupBox containing a label and a read-only line edit
        to show the sound pressure v2pa_factor from the calibration results. The layout
        uses a horizontal box layout to arrange the elements horizontally.

        Returns:
            QGroupBox: A QGroupBox containing the sound pressure v2pa_factor label and line edit.
        """
        v2pa_factor_box = QGroupBox("校准结果")
        v2pa_factor_label = QLabel("校准系数（Pa/V）：")
        self.v2pa_factor_lineedit = QLineEdit()
        self.v2pa_factor_lineedit.setStyleSheet("background-color: white;")
        self.v2pa_factor_lineedit.setReadOnly(True)

        standard_v2pa_factor_layout = QHBoxLayout()
        h_spacer_v2pa_factor_center = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        standard_v2pa_factor_layout.addWidget(v2pa_factor_label)
        standard_v2pa_factor_layout.addItem(h_spacer_v2pa_factor_center)
        standard_v2pa_factor_layout.addWidget(self.v2pa_factor_lineedit)
        v2pa_factor_box.setLayout(standard_v2pa_factor_layout)

        return v2pa_factor_box

    def create_recorded_box(self):
        """
        Create a QGroupBox to display recorded audio information.

        Returns:
            QGroupBox: A QGroupBox containing the recorded time information.
        """
        recorded_box = QGroupBox("录制音频")
        recorded_label = QLabel("录制时间：")
        self.recorded_label = QLabel()
        self.recorded_label.setFixedSize(70, 30)
        self.recorded_label.setAlignment(Qt.AlignCenter)
        self.recorded_time = 10
        self.recorded_label.setText(
            f"<span style='color: red;'>{self.recorded_time} </span>" f"<span style='color: black;'>s</span>"
        )

        self.recorded_label.setStyleSheet(
            "background-color: white;" "border: 1px solid rgb(122, 122, 122);" "border-radius: 3px;"
        )

        recorded_layout = QHBoxLayout()
        h_spacer_v2pa_factor_center = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        recorded_layout.addWidget(recorded_label)
        recorded_layout.addItem(h_spacer_v2pa_factor_center)
        recorded_layout.addWidget(self.recorded_label)
        recorded_box.setLayout(recorded_layout)

        return recorded_box

    def create_standard_spl_box(self):
        """
        Create a group box containing standard sound pressure options.

        This method generates a QGroupBox widget that includes two QRadioButton options,
        representing 94 dB and 114 dB standard sound pressure levels. When a different sound
        pressure level is selected, the set_standard_spl method is triggered to handle the logic.

        Returns:
            QGroupBox: Group box containing standard sound pressure options.
        """
        standard_spl_box = QGroupBox("标准声压")

        self.standard_spl_i = QRadioButton("94  dB")
        self.standard_spl_ii = QRadioButton("114 dB")
        self.standard_spl_i.clicked.connect(self.set_standard_spl)
        self.standard_spl_ii.clicked.connect(self.set_standard_spl)
        self.standard_spl_i.setChecked(True)

        h_spacer_standard_center = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        standard_spl_layout = QHBoxLayout()
        standard_spl_layout.addWidget(self.standard_spl_i)
        standard_spl_layout.addItem(h_spacer_standard_center)
        standard_spl_layout.addWidget(self.standard_spl_ii)
        standard_spl_layout.setContentsMargins(30, 0, 30, 0)
        standard_spl_box.setLayout(standard_spl_layout)

        return standard_spl_box

    def set_standard_spl(self):
        """
        Sets the value of standard_spl_flag based on the selected SPL standard.

        If self.standard_spl_i is checked, sets self.standard_spl_flag to True.
        If self.standard_spl_ii is checked, sets self.standard_spl_flag to False.
        """
        if self.standard_spl_i.isChecked():
            self.standard_spl_flag = True
        elif self.standard_spl_ii.isChecked():
            self.standard_spl_flag = False

    def _get_recording_bridge(self):
        if self.recording_bridge is None:
            self.recording_bridge = RecordingServiceBridge(RecordingService(), self)
            self._owns_recording_bridge = True
            # Qt destruction can bypass closeEvent. Capture the owned bridge,
            # never dereference the already-destroyed widget or its controls.
            bridge = self.recording_bridge
            def shutdown_owned_bridge(_=None, bridge=bridge):
                # Invalidate Python delivery before service cleanup can enqueue
                # cancellation/release events against the deleted Qt child.
                bridge._delivery_closed = True
                if not bridge._shutdown_requested:
                    bridge.service.shutdown()
            self.destroyed.connect(shutdown_owned_bridge)
            self.recording_bridge.shutting_down.connect(self.close_recording)
            QApplication.instance().aboutToQuit.connect(self.recording_bridge.shutdown)
        return self.recording_bridge


    def clicked_calibration(self):
        """Start one asynchronous, single-channel ten-second calibration."""
        if self._recording_closed:
            return False
        if not self.calibration_available or self.current_channel is None:
            self.calibration_popup(success_flag=False,
                message=self.calibration_unavailable_message or "请先选择输入设备和有效输入通道。")
            return False
        bridge = self._get_recording_bridge()
        if (self.streaming_processor is not None
                or not self._can_start_recording_workflow()):
            self.calibration_popup(success_flag=False, message="录音设备忙，请等待当前录音结束后再校准。")
            return False
        self.stop_timer = False
        self.recorded_time = 10
        if not self._ve_input:
            self.v2pa_factor_lineedit.clear()
        self._begin_capture(self.current_channel)
        self._capture_standard_spl_db = 94.0 if self.standard_spl_flag else 114.0
        is_ve = self._ve_input
        machine_id = self.input_device.get("machine_id") if isinstance(self.input_device, Mapping) else None
        try:
            device = self._current_ve_device() if self._ve_input else self.input_device
            sample_rate = device["input_config"]["sample_rate"] if self._ve_input else 44100
            request = RecordingRequest(
                request_id=uuid.uuid4().hex, purpose="calibration", sample_rate=sample_rate,
                target_samples=sample_rate * 10, channels=(self.active_capture_channel,),
                device=device,
                # The service replaces this unused placeholder in its background
                # allocator; the widget neither creates nor removes temp files.
                path=os.path.abspath("calibration-unused.wav"),
                streaming=False, trim_samples=0, monitor={},
                calibration_metadata=None, validation_thresholds={},
            )
            if self._ve_input:
                self._ve_capture_context = _VECalibrationContext(
                    request, self._capture_standard_spl_db, datetime.now(timezone.utc).isoformat())
                self._ve_accepted_audio = None
            session = bridge.start(request, RecordingCallbacks(
                result_ready=self._on_calibration_result_ready,
                accepted=self._on_calibration_accepted,
                failed=self._on_calibration_failed,
                cancelled=self._on_calibration_cancelled,
                released=self._on_calibration_released,
                release_failed=self._on_calibration_release_failed,
            ))
            self.streaming_processor = RecordingProcessorFacade(session)
            self._release_notice_session = session
        except (RuntimeError, ValueError, TypeError, VEStoreIOError) as exc:
            self.streaming_processor = None
            self._clear_active_capture()
            self.default_logger.error(
                f"Failed to start input calibration recording machine_id={machine_id}: {exc}")
            self.calibration_popup(success_flag=False,
                message=ve_failure_text("calibration_start") if is_ve else f"输入校准录音启动失败：{str(exc)[:80]}")
            return False
        self.update_ui_timer.start()
        return True

    def _is_current_session(self, session):
        return (not self._recording_closed and not self.stop_timer
                and self.streaming_processor is not None
                and self.streaming_processor.session is session)

    def _on_calibration_result_ready(self, session, audio):
        if not self._is_current_session(session):
            session.reject_result("calibration window no longer owns this result")
            return
        if self._ve_input:
            try:
                self._verified_ve_audio(session, audio)
            except ValueError as exc:
                session.reject_result(str(exc))
                return
            session.accept_result()
            return
        request, descriptor = session.request, audio.descriptor
        if (descriptor.request_id != request.request_id
                or descriptor.purpose != "calibration"
                or descriptor.channels != (self.active_capture_channel,)
                or descriptor.sample_rate != request.sample_rate
                or descriptor.raw_frames != request.target_samples
                or descriptor.final_frames != request.target_samples
                or descriptor.path != request.path
                or audio.multi.shape != (request.target_samples, 1)
                or audio.mono.shape != (request.target_samples,)
                or audio.multi.dtype != np.float32 or audio.mono.dtype != np.float32
                or not np.isfinite(audio.multi).all()
                or not np.array_equal(audio.mono, audio.multi[:, 0])):
            session.reject_result("输入校准录音长度或通道数据无效")
            return
        session.accept_result()

    def _on_calibration_accepted(self, session, audio):
        if not self._is_current_session(session):
            return
        self.streaming_processor.set_recorded_audio(audio)
        if self._ve_input:
            self._ve_accepted_audio = audio
        self._on_streaming_complete()

    def _on_calibration_failed(self, session, failure):
        if self._is_current_session(session):
            device = session.request.device
            self.default_logger.error(
                f"Input calibration recording failed machine_id={device.get('machine_id')}: {failure}")
            self._finish_failed_calibration(
                ve_failure_text("calibration") if device.get("backend") == "vkinging"
                else f"输入校准录音失败：{failure.message}")

    def _on_calibration_cancelled(self, session, _cancelled):
        if self._is_current_session(session):
            self.cancel_calibration()

    def _on_calibration_released(self, session):
        owned = self._release_notice_session is session
        if owned:
            self._release_notice_session = None
        if self._ve_input and self._is_current_session(session):
            self._on_streaming_complete()
        if self._ve_input and owned and not self._recording_closed:
            if self.streaming_processor is None:
                self._clear_active_capture()
            self.calibration_availability_changed.emit()

    def _on_calibration_release_failed(self, session, error):
        current = self._release_notice_session is session
        if current:
            self._release_notice_session = None
        self.default_logger.error(
            f"Input calibration temporary cleanup failed machine_id={session.request.device.get('machine_id')} "
            f"for {session.request.path}: {error}")
        if session.request.device.get("backend") == "vkinging":
            if current and self._is_current_session(session):
                self._finish_failed_calibration(ve_failure_text("calibration_release"))
            if current and not self._recording_closed:
                if self.streaming_processor is None:
                    self._clear_active_capture()
                self.calibration_availability_changed.emit()
            return
        if current and not self._recording_closed and not self.stop_timer:
            QMessageBox.warning(self, "录音资源未释放",
                "输入校准临时文件未能清理，文件仍被保留。此问题不会改变本次校准结果。\n"
                "请检查路径与访问权限：\n"
                + session.request.path + "\n" + str(error))

    def _on_streaming_complete(self):
        """
        Handle streaming recording completion and calculate calibration result.
        """
        if self._ve_input:
            self._finish_ve_calibration()
            return
        processor = self.streaming_processor
        if (self.stop_timer or self._recording_closed or processor is None
                or processor.session.state != "completed"):
            return
        if not self.calibration_available:
            self._finish_failed_calibration(
                self.calibration_unavailable_message
                or "输入校准当前不可用。"
            )
            return
        captured_channel = self.active_capture_channel
        if captured_channel is None:
            self._finish_failed_calibration("输入校准录音通道状态无效")
            return

        try:
            recorded_data = processor.get_recorded_data()
            if recorded_data.size != processor.target_samples:
                raise ValueError(
                    f"录音长度不完整：{recorded_data.size}/{processor.target_samples}"
                )

        except (RuntimeError, ValueError) as exc:
            self.default_logger.error(
                f"Failed to finish input calibration audio capture: {exc}"
            )
            self._finish_failed_calibration(str(exc))
            return

        try:
            self.average_value = self._calculate_spl_from_data(recorded_data)
            v2pa_factor = self.calculate_v2pa_factor(self.average_value, self._capture_standard_spl_db)
            if (
                not np.isfinite(self.average_value)
                or not np.isfinite(v2pa_factor)
                or v2pa_factor <= 0.0
            ):
                raise ValueError("输入校准计算结果无效")
            if (
                isinstance(processor.sample_rate, bool)
                or not isinstance(processor.sample_rate, Integral)
                or processor.sample_rate <= 0
            ):
                raise ValueError("输入校准采样率无效")

            standard_spl_db = self._capture_standard_spl_db
            save_mic_channel_calibration(
                v2pa_factor=v2pa_factor,
                input_device=processor.session.request.device.to_dict(),
                input_channel=captured_channel,
                standard_spl_db=standard_spl_db,
                sample_rate_hz=processor.sample_rate,
                duration_seconds=processor.target_samples / processor.sample_rate,
            )
        except (
            ValueError,
            ArithmeticError,
            MicCalibrationFormatError,
            MicCalibrationIOError,
        ) as exc:
            self.default_logger.error(
                f"Failed to calculate or save input calibration: {exc}"
            )
            self._finish_failed_calibration("输入校准保存失败，请重试。")
            return

        self._stop_calibration_timers()
        self.streaming_processor = None
        self.active_capture_channel = None
        self.saved_v2pa_factors[captured_channel] = float(v2pa_factor)
        self.channel_combo_box.setEnabled(True)
        next_channel = self._next_uncalibrated_channel(captured_channel)
        self._select_channel(next_channel if next_channel is not None else captured_channel)
        self.calibration_state_changed.emit(True)
        if not self.stop_timer:
            self.calibration_popup(
                success_flag=True,
                message=self._success_popup_message(
                    v2pa_factor, next_channel
                ),
            )
            self.default_logger.info("Input calibration succeeded and was saved.")
            self.calibration_finished.emit(True)

    def _verified_ve_audio(self, session, audio):
        context, request = self._ve_capture_context, session.request
        # The service allocates a private path asynchronously AFTER start().
        # Only that path may differ from the frozen admission context.
        if (session.cancel_requested or context is None
                or replace(context.request, path=request.path) != request):
            raise ValueError("输入校准请求已变更")
        column = verify_ve_calibration_result(request, audio.descriptor, audio.multi)
        if (audio.multi.shape != (request.target_samples, 1)
                or audio.mono.shape != (request.target_samples,) or audio.mono.dtype != np.float32
                or not np.array_equal(audio.mono, column)):
            raise ValueError("输入校准单通道原始电压数据不匹配")
        return audio.mono

    def _check_ve_current_context(self, request):
        channel = request.channels[0]
        expected = calibration_fingerprint(request.device, channel, request.device["input_config"])
        if not isinstance(self.input_device, Mapping):
            raise ValueError("输入设备身份已变更")
        current = calibration_fingerprint(self.input_device, channel, self.input_device.get("input_config"))
        if (self.current_channel != channel or self.active_capture_channel != channel
                or channel not in self.input_channels
                or any(current[key] != expected[key] for key in ("backend", "model", "machine_id"))):
            raise ValueError("输入设备身份或校准物理通道已变更")
        # Observe changed conditions through the shared store before rejecting:
        # invalidation is sticky, even if an external editor restores the values.
        device = self._current_ve_device()
        if (not device["available"] or channel not in device["physical_channels"]
                or (self.ve_queue_config_provider is None and current != expected)
                or calibration_fingerprint(device, channel, device["input_config"]) != expected):
            raise ValueError("输入校准配置已变更，需重新校准")

    def _finish_ve_calibration(self):
        processor = self.streaming_processor
        if processor is None or not self._is_current_session(processor.session):
            return
        session = processor.session
        audio, context = self._ve_accepted_audio, self._ve_capture_context
        if (audio is None or context is None or session.state != "completed"
                or session.cancel_requested or not session.released.is_set()
                or session.release_error is not None):
            return
        try:
            request = session.request
            volts = self._verified_ve_audio(session, audio)
            self._check_ve_current_context(request)
            raw_level = self._calculate_spl_from_data(volts)
            factor = self.calculate_v2pa_factor(raw_level, context.standard_spl)
            if not np.isfinite(raw_level) or not np.isfinite(factor) or factor <= 0:
                raise ValueError("输入校准计算结果无效")
            channel = request.channels[0]
            record = self.ve_calibration_store.save(
                request.device, channel, v2pa_factor=float(factor),
                standard_spl=context.standard_spl, calibration_sample_rate=request.sample_rate,
                calibration_duration_seconds=10.0, calibrated_at=context.calibrated_at)
        except (ValueError, ArithmeticError, VEStoreIOError) as exc:
            self.default_logger.error(
                f"Failed to calculate or save VE calibration machine_id={session.request.device.get('machine_id')}: {exc}")
            self._finish_failed_calibration(ve_failure_text("calibration"))
            return
        self._stop_calibration_timers()
        self.streaming_processor = None
        self._ve_calibration_records[channel] = record
        self.saved_v2pa_factors[channel] = float(factor)
        self._clear_active_capture(refresh_display=False)
        next_channel = self._next_uncalibrated_channel(channel)
        self._select_channel(next_channel if next_channel is not None else channel)
        self.calibration_state_changed.emit(True)
        self.calibration_popup(
            success_flag=True,
            message=self._success_popup_message(factor, next_channel),
        )
        self.calibration_finished.emit(True)

    def _stop_calibration_timers(self):
        self.update_ui_timer.stop()

    def _finish_failed_calibration(self, message):
        self._stop_calibration_timers()
        processor = self.streaming_processor
        self.streaming_processor = None
        self._clear_active_capture()
        if processor is not None:
            processor.stop_streaming()
        if not self.stop_timer:
            self.calibration_popup(success_flag=False, message=message)
            self.calibration_finished.emit(False)

    def _calculate_spl_from_data(self, recorded_data):
        """
        Calculate average SPL from recorded data (extracted from calculate_average_spl).

        Args:
            recorded_data (np.ndarray): Recorded audio data

        Returns:
            float: Average SPL value
        """
        step = len(recorded_data) // 3
        spl_smooth = AudioThdFrequencyResponseAnalysis().spl_calculation(recorded_data, method="rms", window_size=1201)
        spl_smooth_mid = len(spl_smooth) // 2
        spl_smooth_start = spl_smooth_mid - step
        spl_smooth_end = spl_smooth_mid + step
        spl_sample = spl_smooth[spl_smooth_start:spl_smooth_end]
        return np.mean(spl_sample)

    def calibration_popup(self, success_flag=True, message=None):
        """
        Display a calibration result popup.

        Shows different icons and message texts based on whether the calibration was successful.
        If calibration is successful, displays an information icon and success message;
        if calibration fails, displays a critical icon and failure message.

        Parameters:
        - success_flag: Boolean indicating whether the calibration was successful. Default is True.
        """
        cal_msg = QMessageBox(self)
        if success_flag:
            cal_msg.setIcon(QMessageBox.Information)
            cal_msg.setText(message or "校准成功")
            cal_msg.setWindowTitle("校准成功")
        else:
            cal_msg.setIcon(QMessageBox.Critical)
            cal_msg.setText(message or "校准失败，请重试")
            cal_msg.setWindowTitle("校准失败")
        cal_msg.setStandardButtons(QMessageBox.Ok)
        cal_msg.exec_()

    def calculate_average_spl(self, recorded_dict):
        """
        Calculate the average sound pressure level (SPL).

        This method records audio data, computes the SPL curve, and then calculates the average value from a selected range.

        Parameters:
        recorded_dict - Dictionary containing information for recording.

        Returns:
        Average SPL value.
        """
        rec_code, recorded_data = SoundcardAudioProcessor().sd_rec(recorded_dict)
        step = len(recorded_data) // 3
        if rec_code == error_code.OK:
            spl_smooth = AudioThdFrequencyResponseAnalysis().spl_calculation(
                recorded_data, method="rms", window_size=1201
            )
            spl_smooth_mid = len(spl_smooth) // 2
            spl_smooth_start = spl_smooth_mid - step
            spl_smooth_end = spl_smooth_mid + step
            spl_sample = spl_smooth[spl_smooth_start:spl_smooth_end]
            self.average_value = np.mean(spl_sample)
            return self.average_value

    def update_recorded_time(self):
        """
        Update the recorded time countdown.

        This function decrements the recorded time and updates the time display on the interface.
        The timer will stop automatically when time reaches 0 or stop_timer flag is set.
        """
        if self.recorded_time > 0 and not self.stop_timer:
            self.recorded_time -= 1
            # Update the time display on the interface, showing the remaining time in red and the unit "s" in black.
            self.recorded_label.setText(
                f"<span style='color: red;'>{self.recorded_time} </span>" f"<span style='color: black;'>s</span>"
            )
        else:
            self.update_ui_timer.stop()
            # Reset time for next calibration
            self.recorded_time = 10

    def calculate_v2pa_factor(self, average_value, standard_spl_db=None):
        """
        Calculate the v2pa_factor from the standard sound pressure level.

        This function calculates the v2pa_factor based on whether the standard SPL flag is set to True or False.
        If the flag is True, it uses 94 dB as the standard value; otherwise, it uses 114 dB.

        Args:
            average_value (float): The average sound pressure level value used to calculate the v2pa_factor.

        Returns:
            float: The calculated v2pa_factor value rounded to three decimal places.
        """
        if standard_spl_db is None:
            standard_spl_db = 94.0 if self.standard_spl_flag else 114.0
        deviation_value = round(standard_spl_db - average_value, 3)
        v2pa_factor = 10 ** (deviation_value / 20)
        return v2pa_factor

    def reset_btn_clicked(self):
        """
        This method is triggered when the reset button is clicked.

        It resets the recorded time to 10 seconds and updates the recorded label to display the new time in red.
        Additionally, it clears the v2pa_factor line edit and stops any ongoing streaming recording.
        """
        if self._ve_input:
            self._reset_ve_calibration()
            return
        if not self.calibration_available:
            return

        self._release_notice_session = None
        self._stop_calibration_timers()
        processor = self.streaming_processor
        self.streaming_processor = None
        if processor is not None:
            processor.stop_streaming()
        self._clear_active_capture(refresh_display=False)
        self.stop_timer = False

        self.recorded_time = 10
        self.recorded_label.setText(
            f"<span style='color: red;'>{self.recorded_time} </span>" f"<span style='color: black;'>s</span>"
        )
        try:
            changed = clear_mic_channel_calibrations(
                self.input_device,
                self.input_channels,
            )
        except (
            ValueError,
            MicCalibrationFormatError,
            MicCalibrationIOError,
        ) as exc:
            self.default_logger.error(f"Failed to reset input calibration: {exc}")
            self.channel_combo_box.setEnabled(True)
            self._refresh_channel_display()
            self.calibration_popup(
                success_flag=False,
                message="输入校准重置失败，请重试。",
            )
            return

        if changed:
            self.calibration_state_changed.emit(True)
        try:
            self.saved_v2pa_factors = load_mic_channel_v2pa_factors(self.input_device)
        except (MicCalibrationFormatError, MicCalibrationIOError) as exc:
            self.default_logger.error(
                f"Failed to reload input calibration after reset: {exc}"
            )
            self._set_calibration_unavailable("状态: 输入校准文件错误")
            self.calibration_popup(
                success_flag=False,
                message="输入校准文件错误，无法进行输入校准",
            )
            return

        self.channel_combo_box.setEnabled(True)
        self._select_channel(self.input_channels[0])

    def _reset_ve_calibration(self):
        if (self._recording_closed or not self.calibration_available or self.current_channel is None
                or self.streaming_processor is not None
                or not self._can_start_recording_workflow()):
            return
        channel = self.current_channel
        try:
            changed = self.ve_calibration_store.reset(self.input_device, channel)
        except (ValueError, VEStoreIOError) as exc:
            machine_id = self.input_device.get("machine_id") if isinstance(self.input_device, Mapping) else None
            self.default_logger.error(
                f"Failed to reset VE input calibration machine_id={machine_id}: {exc}")
            self.calibration_popup(success_flag=False, message=ve_failure_text("calibration_reset"))
            return
        self._ve_calibration_records.pop(channel, None)
        self.saved_v2pa_factors.pop(channel, None)
        self._refresh_channel_display()
        if changed:
            self.calibration_state_changed.emit(True)

    def cancel_calibration(self):
        if not self._ve_input or self._recording_closed:
            self._release_notice_session = None
        self.stop_timer = True
        self._stop_calibration_timers()
        processor = self.streaming_processor
        self.streaming_processor = None
        self._clear_active_capture()
        if processor is not None:
            processor.stop_streaming()

    def close_recording(self):
        if self._recording_closed:
            return
        self._recording_closed = True
        self.cancel_calibration()
        if self._owns_recording_bridge:
            self.recording_bridge.shutdown()

    def closeEvent(self, event):
        self.close_recording()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CalibrationWindow()
    window.show()
    window.exec()
