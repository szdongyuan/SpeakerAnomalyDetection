import json
import os
from dataclasses import dataclass

os.environ["SD_ENABLE_ASIO"] = "1"

import sounddevice as sd

from base.audio_sample_rate import resolve_input_sample_rate, resolve_output_sample_rate
from base.hardware_management import (
    HardwareManagementRepository,
    MissingHardwareTablesError,
    augment_runtime_device,
    build_selected_device_payload,
)
from consts import error_code, model_consts
from consts.running_consts import AUDIO_DEVICE_CONFIG_PATH


@dataclass(frozen=True)
class SelectedDeviceClearResult:
    status: str
    matched: bool = False
    cleared: bool = False
    clear_failed: bool = False
    error: str = ""

    def __bool__(self):
        return self.cleared


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
        payload = {
            "name": device.get("name"),
            "hostapi_name": hostapi_name,
            "default_samplerate": device.get("default_samplerate"),
        }
        if device.get("samplerate") is not None:
            payload["samplerate"] = device.get("samplerate")
        return payload

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
        normalized_channels = SoundDeviceManager._normalize_mic_channels(mic_channels, mic=mic)
        if SoundDeviceManager._devices_have_registered_hardware_ids(mic, speaker):
            payload = build_selected_device_payload(mic, speaker, normalized_channels)
        else:
            payload = {
                "mic": SoundDeviceManager.serialize_device(mic),
                "speaker": SoundDeviceManager.serialize_device(speaker),
                "mic_channels": normalized_channels,
            }
        os.makedirs(os.path.dirname(AUDIO_DEVICE_CONFIG_PATH), exist_ok=True)
        with open(AUDIO_DEVICE_CONFIG_PATH, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    def _save_selected_devices_with_config_rollback(self, mic, speaker, mic_channels=None):
        had_config = os.path.exists(AUDIO_DEVICE_CONFIG_PATH)
        previous_config = None
        if had_config:
            with open(AUDIO_DEVICE_CONFIG_PATH, "rb") as file:
                previous_config = file.read()

        try:
            self.save_selected_devices(mic, speaker, mic_channels)
        except Exception as save_error:
            try:
                if had_config:
                    os.makedirs(os.path.dirname(AUDIO_DEVICE_CONFIG_PATH), exist_ok=True)
                    with open(AUDIO_DEVICE_CONFIG_PATH, "wb") as file:
                        file.write(previous_config)
                elif os.path.exists(AUDIO_DEVICE_CONFIG_PATH):
                    os.remove(AUDIO_DEVICE_CONFIG_PATH)
            except OSError as rollback_error:
                rollback_action = "restoration" if had_config else "cleanup"
                raise RuntimeError(
                    "selected-device config save failed: "
                    f"{save_error}; rollback {rollback_action} failed: {rollback_error}"
                ) from rollback_error
            raise

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

    @staticmethod
    def clear_selected_devices_for_deleted_hardware(hardware_id):
        if hardware_id is None:
            return SelectedDeviceClearResult(status="no_match")

        payload = SoundDeviceManager.load_selected_devices()
        if not isinstance(payload, dict):
            return SelectedDeviceClearResult(status="no_match")

        deleted_hardware_id = str(hardware_id)
        for device_key in ("mic", "speaker"):
            device = payload.get(device_key)
            if isinstance(device, dict) and str(device.get("hardware_id")) == deleted_hardware_id:
                try:
                    os.remove(AUDIO_DEVICE_CONFIG_PATH)
                except FileNotFoundError:
                    return SelectedDeviceClearResult(status="cleared", matched=True, cleared=True)
                except OSError as exc:
                    return SelectedDeviceClearResult(
                        status="clear_failed",
                        matched=True,
                        clear_failed=True,
                        error=str(exc),
                    )
                return SelectedDeviceClearResult(status="cleared", matched=True, cleared=True)
        return SelectedDeviceClearResult(status="no_match")

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
    def _flatten_all_runtime_devices(device_info):
        devices = []
        seen = set()
        for api_name in device_info:
            for device_type in ("input", "output"):
                for device in device_info[api_name][device_type]:
                    device_index = device.get("index")
                    key = ("index", device_index) if device_index is not None else ("object", id(device))
                    if key in seen:
                        continue
                    seen.add(key)
                    devices.append(device)
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
    def _devices_have_registered_hardware_ids(mic, speaker):
        return (
            isinstance(mic, dict)
            and isinstance(speaker, dict)
            and bool(mic.get("hardware_id"))
            and bool(speaker.get("hardware_id"))
        )

    @staticmethod
    def _saved_devices_have_registered_hardware_ids(saved_devices):
        if not isinstance(saved_devices, dict):
            return False
        return SoundDeviceManager._devices_have_registered_hardware_ids(
            saved_devices.get("mic"),
            saved_devices.get("speaker"),
        )

    @staticmethod
    def _saved_devices_are_retryable(saved_devices):
        if not isinstance(saved_devices, dict):
            return False
        return (
            SoundDeviceManager._saved_device_descriptor_is_retryable(saved_devices.get("mic"))
            and SoundDeviceManager._saved_device_descriptor_is_retryable(saved_devices.get("speaker"))
        )

    @staticmethod
    def _saved_devices_can_retry(saved_devices):
        return (
            SoundDeviceManager._saved_devices_are_retryable(saved_devices)
            or SoundDeviceManager._saved_devices_have_registered_hardware_ids(saved_devices)
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
    def _build_unavailable_startup_result(reason, can_retry_saved_devices=False, startup_recovery_action=None):
        return {
            "mic": None,
            "speaker": None,
            "mic_channels": [],
            "fallback_targets": ["麦克风", "扬声器"],
            "startup_notice_message": reason,
            "startup_device_error_reason": reason,
            "device_available": False,
            "can_retry_saved_devices": bool(can_retry_saved_devices),
            "startup_recovery_action": startup_recovery_action,
        }

    @staticmethod
    def _augment_legacy_startup_device_with_saved_samplerate(runtime_device, saved_device, resolver):
        resolved = resolver(saved_device)
        if not resolved.ok:
            return None, resolved.message
        device = dict(runtime_device)
        device["samplerate"] = resolved.sample_rate
        return device, None

    @staticmethod
    def _match_registered_runtime_device(asset, current_devices):
        matches = [
            device
            for device in current_devices
            if device.get("name") == asset.get("device_name")
            and SoundDeviceManager.get_device_hostapi_name(device) == asset.get("hostapi_name")
        ]
        if len(matches) == 1:
            return "ok", matches[0]
        if matches:
            return "ambiguous", None
        return "missing", None

    @staticmethod
    def _runtime_mic_supports_channels(runtime_mic, mic_channels):
        try:
            max_input_channels = int(runtime_mic.get("max_input_channels") or 0)
        except (TypeError, ValueError):
            max_input_channels = 0
        return max_input_channels > 0 and all(channel < max_input_channels for channel in mic_channels)

    @staticmethod
    def _runtime_speaker_has_output_capacity(runtime_speaker):
        try:
            max_output_channels = int(runtime_speaker.get("max_output_channels") or 0)
        except (TypeError, ValueError):
            max_output_channels = 0
        return max_output_channels > 0

    @staticmethod
    def _restore_registered_mic_channels(asset, channel_rows, saved_channels):
        if not saved_channels:
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
                channel_index = int(saved_channel)
            except (TypeError, ValueError):
                return []
            if channel_index < 0:
                return []
            normalized_channels.add(channel_index)

        if not normalized_channels:
            return []

        available_channels = set()
        for row in channel_rows:
            if not hasattr(row, "get"):
                return []
            if row.get("direction") != "input":
                continue
            raw_channel_index = row.get("channel_index")
            if raw_channel_index is None:
                return []
            if isinstance(raw_channel_index, bool):
                return []
            if isinstance(raw_channel_index, float) and not raw_channel_index.is_integer():
                return []
            try:
                available_channel = int(raw_channel_index)
            except (TypeError, ValueError):
                return []
            if available_channel < 0:
                return []
            available_channels.add(available_channel)
        if any(channel not in available_channels for channel in normalized_channels):
            return []

        try:
            max_input_channels = int(asset.get("max_input_channels") or 0)
        except (TypeError, ValueError):
            return []
        if max_input_channels <= 0:
            return []
        if any(channel >= max_input_channels for channel in normalized_channels):
            return []
        return sorted(normalized_channels)

    @staticmethod
    def _registered_channel_index_is_valid(value):
        if value is None or isinstance(value, bool):
            return False
        if isinstance(value, int):
            channel_index = value
        elif isinstance(value, str):
            stripped = value.strip()
            if not stripped.isdecimal():
                return False
            channel_index = int(stripped)
        else:
            return False
        return channel_index >= 0

    @staticmethod
    def _registered_asset_has_selectable_input_channel(repository, asset):
        channel_rows = repository.list_channels(asset.get("hardware_id"), direction="input")
        if not channel_rows:
            return False
        for row in channel_rows:
            if not hasattr(row, "get"):
                return False
            if row.get("direction") != "input":
                return False
            if not SoundDeviceManager._registered_channel_index_is_valid(row.get("channel_index")):
                return False
        return True

    @staticmethod
    def _selection_group_can_complete_selection(repository, grouped_assets):
        if not grouped_assets:
            return False
        for api_group in grouped_assets.values():
            if not api_group.get("output"):
                continue
            if any(
                SoundDeviceManager._registered_asset_has_selectable_input_channel(repository, asset)
                for asset in api_group.get("input", [])
            ):
                return True
        return False

    @staticmethod
    def _registered_assets_can_complete_selection(repository):
        list_for_selection = getattr(repository, "list_assets_for_selection", None)
        if callable(list_for_selection):
            grouped_assets = list_for_selection() or {}
            return SoundDeviceManager._selection_group_can_complete_selection(repository, grouped_assets)

        registered_assets = repository.list_assets()
        has_input = any(asset.get("max_input_channels", 0) > 0 for asset in registered_assets)
        has_output = any(asset.get("max_output_channels", 0) > 0 for asset in registered_assets)
        return has_input and has_output

    def _try_resolve_registered_startup_devices(self, saved_devices, runtime_devices):
        if not self._saved_devices_have_registered_hardware_ids(saved_devices):
            return None

        repository = HardwareManagementRepository()
        saved_mic = saved_devices.get("mic")
        saved_speaker = saved_devices.get("speaker")
        try:
            tables_exist = repository.tables_exist()
        except MissingHardwareTablesError:
            return self._build_unavailable_startup_result(
                "已注册硬件当前不可用，请检查设备连接后重试。",
                can_retry_saved_devices=True,
            )
        except model_consts.SQLITE_REPOSITORY_EXCEPTIONS:
            return self._build_unavailable_startup_result(
                "已注册硬件当前不可用，请检查设备连接后重试。",
                can_retry_saved_devices=True,
            )

        if not tables_exist:
            return self._build_unavailable_startup_result(
                "已注册硬件当前不可用，请检查设备连接后重试。",
                can_retry_saved_devices=True,
            )

        try:
            mic_asset = repository.get_asset(saved_mic.get("hardware_id"))
            speaker_asset = repository.get_asset(saved_speaker.get("hardware_id"))
        except MissingHardwareTablesError:
            return self._build_unavailable_startup_result(
                "已注册硬件当前不可用，请检查设备连接后重试。",
                can_retry_saved_devices=True,
            )
        except model_consts.SQLITE_REPOSITORY_EXCEPTIONS:
            return self._build_unavailable_startup_result(
                "已注册硬件当前不可用，请检查设备连接后重试。",
                can_retry_saved_devices=True,
            )

        if mic_asset is None or speaker_asset is None:
            try:
                can_complete_selection = self._registered_assets_can_complete_selection(repository)
            except MissingHardwareTablesError:
                return self._build_unavailable_startup_result(
                    "已注册硬件当前不可用，请检查设备连接后重试。",
                    can_retry_saved_devices=True,
                )
            except model_consts.SQLITE_REPOSITORY_EXCEPTIONS:
                return self._build_unavailable_startup_result(
                    "已注册硬件当前不可用，请检查设备连接后重试。",
                    can_retry_saved_devices=True,
                )
            if not can_complete_selection:
                return self._build_unavailable_startup_result(
                    "已保存的注册硬件已删除或不存在，请先在硬件管理中注册硬件。",
                    can_retry_saved_devices=False,
                    startup_recovery_action="register_hardware",
                )
            return self._build_unavailable_startup_result(
                "已保存的注册硬件已删除或不存在，请在硬件管理中重新选择设备。",
                can_retry_saved_devices=False,
            )

        try:
            channel_rows = repository.list_channels(mic_asset["hardware_id"], direction="input")
        except MissingHardwareTablesError:
            return self._build_unavailable_startup_result(
                "已注册硬件当前不可用，请检查设备连接后重试。",
                can_retry_saved_devices=True,
            )
        except model_consts.SQLITE_REPOSITORY_EXCEPTIONS:
            return self._build_unavailable_startup_result(
                "已注册硬件当前不可用，请检查设备连接后重试。",
                can_retry_saved_devices=True,
            )

        mic_channels = self._restore_registered_mic_channels(
            mic_asset,
            channel_rows,
            saved_devices.get("mic_channels"),
        )

        if not mic_channels:
            return self._build_unavailable_startup_result(
                "已保存的麦克风通道不存在或配置无效，请检查设备连接或重新选择设备。",
                can_retry_saved_devices=True,
            )

        mic_match_status, runtime_mic = self._match_registered_runtime_device(mic_asset, runtime_devices)
        speaker_match_status, runtime_speaker = self._match_registered_runtime_device(speaker_asset, runtime_devices)
        if mic_match_status == "missing" or speaker_match_status == "missing":
            return self._build_unavailable_startup_result(
                "已注册硬件当前不可用，请检查设备连接后重试。",
                can_retry_saved_devices=True,
            )
        if mic_match_status == "ambiguous" or speaker_match_status == "ambiguous":
            return self._build_unavailable_startup_result(
                "已注册硬件匹配到多个当前设备，无法安全应用。",
                can_retry_saved_devices=True,
            )

        if not self._runtime_mic_supports_channels(runtime_mic, mic_channels):
            return self._build_unavailable_startup_result(
                "已保存的麦克风通道不存在或配置无效，请检查设备连接或重新选择设备。",
                can_retry_saved_devices=True,
            )
        if not self._runtime_speaker_has_output_capacity(runtime_speaker):
            return self._build_unavailable_startup_result(
                "已注册硬件当前不可用，请检查设备连接后重试。",
                can_retry_saved_devices=True,
            )

        mic = augment_runtime_device(runtime_mic, mic_asset)
        speaker = augment_runtime_device(runtime_speaker, speaker_asset)
        try:
            self.change_default_device(mic["index"], speaker["index"])
        except Exception as e:
            return self._build_unavailable_startup_result(
                f"音频设备应用失败，请检查设备连接或重新选择设备。{e}",
                can_retry_saved_devices=True,
            )

        try:
            self._save_selected_devices_with_config_rollback(mic, speaker, mic_channels)
        except Exception as e:
            return self._build_unavailable_startup_result(
                f"音频设备配置保存失败，请检查设备连接或重新选择设备。{e}",
                can_retry_saved_devices=True,
            )

        return {
            "mic": mic,
            "speaker": speaker,
            "mic_channels": mic_channels,
            "fallback_targets": [],
            "startup_notice_message": None,
            "startup_device_error_reason": None,
            "device_available": True,
            "can_retry_saved_devices": True,
            "startup_recovery_action": None,
        }

    def _resolve_saved_startup_devices(self, saved_devices):
        try:
            all_devices = self.get_device_info()
        except Exception as e:
            return self._build_unavailable_startup_result(
                f"音频设备枚举失败，请检查设备连接或重新选择设备。{e}",
                can_retry_saved_devices=self._saved_devices_can_retry(saved_devices),
            )

        mic_devices = self._flatten_device_info(all_devices, "input")
        speaker_devices = self._flatten_device_info(all_devices, "output")
        runtime_devices = self._flatten_all_runtime_devices(all_devices)

        registered_result = self._try_resolve_registered_startup_devices(saved_devices, runtime_devices)
        if registered_result is not None:
            return registered_result

        if not self._saved_devices_are_retryable(saved_devices):
            return self._build_unavailable_startup_result(
                "已保存的音频设备配置无效，请检查设备连接或重新选择设备。",
                can_retry_saved_devices=False,
            )

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

        mic, mic_samplerate_error = self._augment_legacy_startup_device_with_saved_samplerate(
            mic,
            saved_mic,
            resolve_input_sample_rate,
        )
        speaker, speaker_samplerate_error = self._augment_legacy_startup_device_with_saved_samplerate(
            speaker,
            saved_speaker,
            resolve_output_sample_rate,
        )
        if mic_samplerate_error or speaker_samplerate_error:
            return self._build_unavailable_startup_result(
                mic_samplerate_error or speaker_samplerate_error,
                can_retry_saved_devices=True,
            )

        try:
            self.change_default_device(mic["index"], speaker["index"])
        except Exception as e:
            return self._build_unavailable_startup_result(
                f"音频设备应用失败，请检查设备连接或重新选择设备。{e}",
                can_retry_saved_devices=True,
            )

        if not self._saved_devices_have_registered_hardware_ids(saved_devices):
            try:
                self._save_selected_devices_with_config_rollback(mic, speaker, mic_channels)
            except Exception as e:
                return self._build_unavailable_startup_result(
                    f"音频设备配置保存失败，请检查设备连接或重新选择设备。{e}",
                    can_retry_saved_devices=True,
                )

        return {
            "mic": mic,
            "speaker": speaker,
            "mic_channels": mic_channels,
            "fallback_targets": [],
            "startup_notice_message": None,
            "startup_device_error_reason": None,
            "device_available": True,
            "can_retry_saved_devices": True,
            "startup_recovery_action": None,
        }

    def get_startup_devices(self):
        try:
            self.refresh_available_device()
        except Exception as e:
            saved_devices = self.load_selected_devices()
            return self._build_unavailable_startup_result(
                f"音频设备枚举失败，请检查设备连接或重新选择设备。{e}",
                can_retry_saved_devices=self._saved_devices_can_retry(saved_devices),
            )

        if not os.path.exists(AUDIO_DEVICE_CONFIG_PATH):
            return self._build_unavailable_startup_result(
                "未找到已保存的音频设备配置，请检查设备连接或重新选择设备。",
                can_retry_saved_devices=False,
            )

        saved_devices = self.load_selected_devices()
        if not isinstance(saved_devices, dict):
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
