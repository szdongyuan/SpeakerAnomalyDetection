import json
import os

os.environ["SD_ENABLE_ASIO"] = "1"

import sounddevice as sd

from consts import error_code
from consts.running_consts import AUDIO_DEVICE_CONFIG_PATH


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
    def get_device_hostapi_name(device):
        if not device:
            return None

        return SoundDeviceManager.get_api_info(device.get("hostapi")).get("name")

    @staticmethod
    def serialize_device(device):
        if not device:
            return None
        hostapi_name = SoundDeviceManager.get_device_hostapi_name(device)
        return {
            "name": device.get("name"),
            "hostapi_name": hostapi_name,
            "default_samplerate": device.get("default_samplerate"),
        }

    @staticmethod
    def _normalize_mic_channels(mic_channels=None, mic=None, fallback=None):
        if fallback is None:
            fallback = [0]

        max_channels = 0
        try:
            if mic:
                max_channels = int(mic.get("max_input_channels") or 0)
        except (TypeError, ValueError):
            max_channels = 0

        if mic_channels is None:
            candidates = []
        elif isinstance(mic_channels, (str, bytes)):
            candidates = [mic_channels]
        else:
            try:
                candidates = list(mic_channels)
            except TypeError:
                candidates = [mic_channels]

        normalized = set()
        for channel in candidates:
            try:
                channel = int(channel)
            except (TypeError, ValueError):
                continue
            if channel < 0:
                continue
            if max_channels > 0 and channel >= max_channels:
                continue
            normalized.add(channel)

        if normalized:
            return sorted(normalized)

        if fallback is mic_channels:
            return [0]
        return SoundDeviceManager._normalize_mic_channels(fallback, mic=mic, fallback=[0])

    @staticmethod
    def normalize_channel_indices(channels):
        if not channels:
            return []
        normalized = []
        for channel in channels:
            try:
                normalized.append(int(channel))
            except (TypeError, ValueError):
                continue
        return sorted(set(normalized))

    @staticmethod
    def channels_for_device(device, device_type="mic"):
        if not device:
            return []
        if device_type == "speaker":
            max_channels = int(device.get("max_output_channels") or 0)
        else:
            max_channels = int(device.get("max_input_channels") or 0)
        if max_channels <= 0:
            return []
        return list(range(max_channels))

    @staticmethod
    def restore_mic_channels(mic, saved_channels):
        available_channels = SoundDeviceManager.channels_for_device(mic, "mic")
        if not available_channels:
            return []

        if saved_channels is None:
            return available_channels
        try:
            saved_channel_items = list(saved_channels)
        except TypeError:
            saved_channel_items = [saved_channels]
        if not saved_channel_items:
            return available_channels

        saved_channel_set = set(SoundDeviceManager.normalize_channel_indices(saved_channel_items))
        valid_channels = [channel for channel in available_channels if channel in saved_channel_set]
        return valid_channels or [available_channels[0]]

    @staticmethod
    def restore_saved_mic_channels_strict(mic, saved_channels):
        available_channels = SoundDeviceManager.channels_for_device(mic, "mic")
        if not available_channels or saved_channels is None:
            return []

        try:
            saved_channel_items = list(saved_channels)
        except TypeError:
            saved_channel_items = [saved_channels]
        if not saved_channel_items:
            return []

        normalized_channels = set()
        for saved_channel in saved_channel_items:
            if isinstance(saved_channel, bool):
                return []
            if isinstance(saved_channel, float) and not saved_channel.is_integer():
                return []
            try:
                normalized_channel = int(saved_channel)
            except (TypeError, ValueError):
                return []
            if normalized_channel < 0:
                return []
            normalized_channels.add(normalized_channel)

        if not normalized_channels:
            return []

        available_set = set(available_channels)
        if any(channel not in available_set for channel in normalized_channels):
            return []

        return [channel for channel in available_channels if channel in normalized_channels]

    @staticmethod
    def save_selected_devices(mic, speaker, mic_channels=None):
        payload = {
            "mic": SoundDeviceManager.serialize_device(mic),
            "speaker": SoundDeviceManager.serialize_device(speaker),
            "mic_channels": SoundDeviceManager._normalize_mic_channels(mic_channels, mic=mic),
        }
        os.makedirs(os.path.dirname(AUDIO_DEVICE_CONFIG_PATH), exist_ok=True)
        with open(AUDIO_DEVICE_CONFIG_PATH, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    @staticmethod
    def load_selected_devices():
        if not os.path.exists(AUDIO_DEVICE_CONFIG_PATH):
            return None
        try:
            with open(AUDIO_DEVICE_CONFIG_PATH, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def get_saved_mic_channels(self, fallback=None, mic=None):
        payload = self.load_selected_devices()
        if not payload:
            return self._normalize_mic_channels(fallback, mic=mic)
        return self._normalize_mic_channels(payload.get("mic_channels"), mic=mic, fallback=fallback)

    @staticmethod
    def _flatten_device_info(device_info, device_type):
        devices = []
        for api_name in device_info:
            devices.extend(device_info[api_name][device_type])
        return devices

    @staticmethod
    def _match_saved_device(saved_device, current_devices):
        if not isinstance(saved_device, dict):
            return None

        saved_name = saved_device.get("name")
        saved_hostapi_name = saved_device.get("hostapi_name")
        saved_samplerate = saved_device.get("default_samplerate")
        if saved_name is None or saved_hostapi_name is None or saved_samplerate is None:
            return None

        for device in current_devices:
            current_hostapi_name = SoundDeviceManager.get_device_hostapi_name(device)
            if (
                device.get("name") == saved_name
                and current_hostapi_name == saved_hostapi_name
                and float(device.get("default_samplerate")) == float(saved_samplerate)
            ):
                return device
        return None

    @staticmethod
    def _saved_devices_are_retryable(saved_devices):
        if not isinstance(saved_devices, dict):
            return False
        return (
            SoundDeviceManager._saved_device_descriptor_is_retryable(saved_devices.get("mic"))
            and SoundDeviceManager._saved_device_descriptor_is_retryable(saved_devices.get("speaker"))
        )

    @staticmethod
    def _saved_device_descriptor_is_retryable(saved_device):
        if not isinstance(saved_device, dict):
            return False

        name = saved_device.get("name")
        hostapi_name = saved_device.get("hostapi_name")
        default_samplerate = saved_device.get("default_samplerate")
        if not name or not hostapi_name or default_samplerate is None:
            return False

        try:
            float(default_samplerate)
        except (TypeError, ValueError):
            return False

        return True

    @staticmethod
    def _build_unavailable_startup_result(reason, can_retry_saved_devices=False):
        return {
            "mic": None,
            "speaker": None,
            "mic_channels": [],
            "fallback_targets": ["麦克风", "扬声器"],
            "startup_notice_message": reason,
            "startup_device_error_reason": reason,
            "device_available": False,
            "can_retry_saved_devices": bool(can_retry_saved_devices),
        }

    def _resolve_saved_startup_devices(self, saved_devices):
        try:
            all_devices = self.get_device_info()
        except Exception as e:
            return self._build_unavailable_startup_result(
                f"音频设备枚举失败，请检查设备连接或重新选择设备。{e}",
                can_retry_saved_devices=self._saved_devices_are_retryable(saved_devices),
            )

        mic_devices = self._flatten_device_info(all_devices, "input")
        speaker_devices = self._flatten_device_info(all_devices, "output")

        saved_mic = saved_devices.get("mic")
        saved_speaker = saved_devices.get("speaker")
        saved_mic_channels = saved_devices.get("mic_channels")

        mic = self._match_saved_device(saved_mic, mic_devices) if isinstance(saved_mic, dict) else None
        speaker = self._match_saved_device(saved_speaker, speaker_devices) if isinstance(saved_speaker, dict) else None

        if mic is None or speaker is None:
            return self._build_unavailable_startup_result(
                "已保存的音频设备不存在或配置无效，请检查设备连接或重新选择设备。",
                can_retry_saved_devices=True,
            )

        mic_channels = self.restore_saved_mic_channels_strict(mic, saved_mic_channels)
        if not mic_channels:
            return self._build_unavailable_startup_result(
                "已保存的麦克风通道不存在或配置无效，请检查设备连接或重新选择设备。",
                can_retry_saved_devices=True,
            )

        self.change_default_device(mic["index"], speaker["index"])
        self.save_selected_devices(mic, speaker, mic_channels)

        return {
            "mic": mic,
            "speaker": speaker,
            "mic_channels": mic_channels,
            "fallback_targets": [],
            "startup_notice_message": None,
            "startup_device_error_reason": None,
            "device_available": True,
            "can_retry_saved_devices": True,
        }

    def get_startup_devices(self):
        try:
            self.refresh_available_device()
        except Exception as e:
            saved_devices = self.load_selected_devices()
            return self._build_unavailable_startup_result(
                f"音频设备枚举失败，请检查设备连接或重新选择设备。{e}",
                can_retry_saved_devices=self._saved_devices_are_retryable(saved_devices),
            )

        if not os.path.exists(AUDIO_DEVICE_CONFIG_PATH):
            return self._build_unavailable_startup_result(
                "未找到已保存的音频设备配置，请检查设备连接或重新选择设备。",
                can_retry_saved_devices=False,
            )

        saved_devices = self.load_selected_devices()
        if not self._saved_devices_are_retryable(saved_devices):
            return self._build_unavailable_startup_result(
                "已保存的音频设备配置无效，请检查设备连接或重新选择设备。",
                can_retry_saved_devices=False,
            )

        return self._resolve_saved_startup_devices(saved_devices)

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

        return self.channels_for_device(device, device_type)
