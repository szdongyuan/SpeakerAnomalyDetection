import os
import threading
from contextlib import contextmanager

os.environ["SD_ENABLE_ASIO"] = "1"

import sounddevice as sd

from consts import error_code


class SoundDeviceManager(object):

    _convenience_lock = threading.Lock()

    @classmethod
    @contextmanager
    def convenience_operation(cls):
        """Serialize project convenience calls without waiting on busy audio.

        Hold through the ownership check, start and any retry. Blocking calls
        retain ownership until completion; contenders get a busy result.
        """
        acquired = cls._convenience_lock.acquire(blocking=False)
        try:
            yield acquired
        finally:
            if acquired:
                cls._convenience_lock.release()

    def get_default_device(self, device, refresh=True):
        if refresh:
            self.refresh_available_device()
        try:
            if device == "mic":
                snapshot = dict(sd.query_devices(sd.default._default_device[0]))
                snapshot["hostapi_name"] = sd.query_hostapis(snapshot["hostapi"])["name"]
                return error_code.OK, snapshot
            elif device == "speaker":
                # The backend host API reports the system default independently
                # of sd.default.device overrides. Unlike _default_device, this
                # path never resolves ordinary input (which VE does not use).
                index = sd.query_hostapis(sd.default.hostapi).get("default_output_device")
                if index is None or index < 0:
                    return error_code.MISSING_HARDWARE_DEVICE, None
                return error_code.OK, sd.query_devices(index)
        except Exception as e:
            return error_code.MISSING_HARDWARE_DEVICE, None

    @staticmethod
    def change_default_device(mic_id, speaker_id):
        """Legacy signature: apply input only; speaker_id is inert."""
        SoundDeviceManager.change_default_input_device(mic_id)

    @staticmethod
    def change_default_input_device(mic_id):
        """Apply input without reading or replacing the output slot."""
        sd.default.device[0] = mic_id

    @staticmethod
    def change_default_output_device(speaker_id):
        """Inert legacy API: output follows the backend default at playback."""

    @staticmethod
    def output_would_interrupt(owned_stream=None):
        """Convenience playback stops the last convenience stream before open.

        Called within convenience_operation. Refuse to replace an active stream
        unless this caller owns it. Completed nonblocking streams can remain
        open; allow the next convenience call to close and replace them.
        """
        try:
            stream = sd.get_stream()
        except RuntimeError:
            # sounddevice raises this when no convenience stream exists yet.
            return False
        return stream is not owned_stream and not stream.closed and stream.active

    @staticmethod
    def get_api_info(api_index=None):
        return sd.query_hostapis(api_index)

    @staticmethod
    def get_device_info():
        api_info = sd.query_hostapis()
        device_list = sd.query_devices()
        host_dict = {}
        for api in api_info:
            api_input = []
            api_output = []
            host_dict[api.get("name")] = {"input": [], "output": []}
            for device_id in api.get("devices"):
                device = device_list[device_id]
                if device.get("max_input_channels") > 0:
                    api_input.append({**device, "hostapi_name": api["name"]})
                if device.get("max_output_channels") > 0:
                    api_output.append(device)
            host_dict[api.get("name")] = {"input": api_input, "output": api_output}
        return host_dict

    @staticmethod
    def refresh_available_device():
        sd._terminate()
        sd._initialize()

    def get_default_device_all_channels(self, device_type: str, refresh: bool = False):
        """
        获取默认麦克风/扬声器的所有通道序号（0-based）。

        - mic：基于 max_input_channels，返回 [0..max-1]
        - speaker：基于 max_output_channels，返回 [0..max-1]
        - 无默认设备或 max<=0：返回 []
        """
        if device_type not in ("mic", "speaker"):
            return []

        code, device = self.get_default_device(device_type, refresh=refresh)
        if code != error_code.OK or not device:
            return []

        if device_type == "mic":
            max_channels = int(device.get("max_input_channels") or 0)
        else:
            max_channels = int(device.get("max_output_channels") or 0)

        if max_channels <= 0:
            return []

        return list(range(max_channels))
