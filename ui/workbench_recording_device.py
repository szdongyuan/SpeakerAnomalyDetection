"""GUI-thread preparation and immutable capture capabilities for the workbench."""

from dataclasses import dataclass
from uuid import uuid4

from PyQt5.QtCore import QObject, pyqtSignal

from base.recording_process_protocol import FrozenConfig, SoundcardPrepareRequest, VePrewarmRequest


@dataclass(frozen=True)
class WorkbenchRecordingDeviceSnapshot:
    probe_id: str
    generation: int
    device: FrozenConfig
    channels: tuple[int, ...]
    sample_rate: int
    ve_signature: tuple | None = None

    @property
    def physical_to_column(self):
        return {channel: column for column, channel in enumerate(self.channels)}


class WorkbenchRecordingDevice(QObject):
    changed = pyqtSignal()

    def __init__(self, parent, *, bridge, lifetime, load_ve_profile, prewarm_ve, rediscover_ve, logger):
        super().__init__(parent)
        self._bridge = bridge
        self._lifetime = lifetime
        self._load_ve_profile = load_ve_profile
        self._prewarm_ve = prewarm_ve
        self._rediscover_ve = rediscover_ve
        self._logger = logger
        self._worker = bridge.worker_status
        self._closed = False
        self._login_ready = False
        self._snapshot = None
        self._soundcard_identity = None
        self._soundcard_pending = None
        self._soundcard_needs_probe = False
        self._ve_selected = False
        self._ve_needs_discovery = False
        self._ve_request = None
        self.detail = "正在准备录制设备…"
        bridge.worker_status_changed.connect(self._on_worker_status)
        bridge.shutting_down.connect(self.close)

    @property
    def snapshot(self):
        status = self._bridge.worker_status
        if (not self._closed and self._snapshot is not None and status.state == "ready"
                and status.generation == self._snapshot.generation):
            return self._snapshot
        return None

    @property
    def login_ready(self):
        return self._login_ready and not self._closed

    def set_login_ready(self, ready):
        if not self._closed and self._login_ready != ready:
            self._login_ready = ready
            self.changed.emit()

    @property
    def ve_signature(self):
        return self._ve_request.signature if self._ve_request is not None else None

    def _publish(self, snapshot, detail):
        if (self._snapshot, self.detail) != (snapshot, detail):
            self._snapshot, self.detail = snapshot, detail
            self.changed.emit()

    def _clear_selection(self):
        self._soundcard_identity = None
        self._soundcard_pending = None
        self._soundcard_needs_probe = False
        self._ve_selected = False
        self._ve_needs_discovery = False
        self._ve_request = None

    def invalidate_selection(self):
        if not self._closed:
            self._clear_selection()
            self._publish(None, "正在准备录制设备…")

    def select_soundcard(self, identity):
        if self._closed:
            return
        self._clear_selection()
        if identity is not None and not all(isinstance(value, str) and value.strip() for value in identity):
            self._publish(None, "已保存的声卡身份不完整，请在硬件设置中重新选择输入设备。")
            return
        self._soundcard_identity = identity if identity is not None else (None, None)
        self._soundcard_needs_probe = True
        self._publish(None, "正在准备声卡输入…")
        self.refresh()

    def select_ve(self):
        if not self._closed:
            self._clear_selection()
            self._ve_selected = True
            self._publish(None, "正在检查 VE 输入设备…")

    def prepare_ve(self, device, source):
        if self._closed or not self._ve_selected or self._ve_needs_discovery:
            return "invalid"
        self._ve_request = None
        if not device.get("available"):
            self._publish(None, "VE 输入设备不可用，请检查连接或在硬件设置中重新选择。")
            return "invalid"
        try:
            config = self._load_ve_profile(device)
            self._ve_request = VePrewarmRequest.create(
                uuid4().hex, {**device, "input_config": config}, device["physical_channels"],
                config["sample_rate"], attempt=1)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            self._logger.warning("Workbench VE configuration failed for %s: %s", device.get("machine_id"), exc)
            self._publish(None, "VE 输入设备配置不可用，请检查硬件设置。")
            return "invalid"
        result = self._prewarm_ve(device, self._ve_request.channels, source, request=self._ve_request)
        self.refresh()
        return result

    def refresh(self):
        if self._closed:
            return
        status = self._bridge.worker_status
        if status.state in ("unavailable", "closing"):
            return
        if self._soundcard_needs_probe and not self._bridge.hardware_busy:
            self._start_soundcard_probe()
        request = self._ve_request
        if request is None or status != self._worker:
            return
        if self._snapshot is not None and self._snapshot.probe_id == request.warmup_id:
            return
        signature = request.signature
        admission = self._lifetime.admission_for(signature)
        if admission == "failed_signature":
            self._publish(None, "VE 设备初始化失败，请检查硬件设置。")
        elif admission == "pending" or status.state != "ready":
            self._publish(None, "VE 设备正在初始化…")
        else:
            self._publish(WorkbenchRecordingDeviceSnapshot(
                request.warmup_id, status.generation, request.device, request.channels,
                request.sample_rate, signature), "")

    def _start_soundcard_probe(self):
        request = SoundcardPrepareRequest(uuid4().hex, *self._soundcard_identity)
        self._soundcard_pending = request
        self._soundcard_needs_probe = False
        result = self._bridge.prepare_soundcard(request, lambda outcome: self._on_soundcard_result(request, outcome))
        if self._soundcard_pending is request and result != "accepted":
            self._soundcard_pending = None
            self._soundcard_needs_probe = result == "busy"
            if result == "closing":
                self.close()

    def _on_soundcard_result(self, request, result):
        status = self._bridge.worker_status
        if (self._closed or self._soundcard_pending is not request or result.probe_id != request.probe_id
                or status.state != "ready" or status.generation != result.generation):
            return
        self._soundcard_pending = None
        if result.success:
            identity = result.device["name"], result.device["hostapi_name"]
            if request.name is None or identity == self._soundcard_identity:
                self._publish(WorkbenchRecordingDeviceSnapshot(
                    result.probe_id, result.generation, result.device, result.channels, result.sample_rate), "")
                return
        self._logger.warning("Workbench soundcard preparation failed for %r: %s",
                             self._soundcard_identity, result.detail or "unexpected input identity")
        detail = ("默认声卡不可用，请在硬件设置中选择输入设备。" if request.name is None
                  else f"已保存的声卡 {request.name} 不可用，请在硬件设置中重新选择输入设备。")
        self._publish(None, detail)

    def _on_worker_status(self, status):
        if self._closed or status.generation < self._worker.generation:
            return
        if status.state == "closing":
            self.close()
            return
        previous, self._worker = self._worker, status
        if (status.state == "unavailable"
                or (previous.generation and status.generation != previous.generation)):
            self._soundcard_pending = None
            self._soundcard_needs_probe = self._soundcard_identity is not None
            self._ve_request = None
            self._ve_needs_discovery = self._ve_selected
            self._publish(None, "录制进程正在恢复，等待设备重新准备…")
        if status.state == "ready" and self._ve_needs_discovery:
            self._ve_needs_discovery = False
            self._rediscover_ve()
        self.refresh()

    def close(self):
        if not self._closed:
            self._closed = True
            self._clear_selection()
            self._publish(None, "应用正在退出。")
