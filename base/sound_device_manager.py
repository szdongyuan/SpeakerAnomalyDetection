import os

os.environ["SD_ENABLE_ASIO"] = "1"

import sounddevice as sd


class SoundDeviceManager(object):

    @staticmethod
    def get_default_device(device):
        if device == "mic":
            return sd.query_devices(sd.default.device[0])
        elif device == "speaker":
            return sd.query_devices(sd.default.device[1])

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
