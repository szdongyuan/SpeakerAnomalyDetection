import os

os.environ["SD_ENABLE_ASIO"] = "1"

import sounddevice as sd

from consts import error_code


class SoundDeviceManager(object):

    def get_default_device(self, device, refresh=True):
        if refresh:
            self.refresh_available_device()
        try:
            if device == "mic":
                return error_code.OK, sd.query_devices(sd.default._default_device[0])
            elif device == "speaker":
                return error_code.OK, sd.query_devices(sd.default._default_device[1])
        except Exception as e:
            return error_code.MISSING_HARDWARE_DEVICE, None

    @staticmethod
    def change_default_device(mic_id, speaker_id):
        sd.default.device = (mic_id, speaker_id)

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
                    api_input.append(device)
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
