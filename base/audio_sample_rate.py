from dataclasses import dataclass
from typing import Optional

from consts.audio_consts import VALID_SAMPLE_RATES


@dataclass(frozen=True)
class SampleRateResolution:
    ok: bool
    sample_rate: Optional[int] = None
    message: str = ""


def _fail(message):
    return SampleRateResolution(False, None, message)


def _coerce_exact_integer_sample_rate(value):
    if isinstance(value, bool):
        return None
    try:
        sample_rate = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        if not isinstance(value, str) and float(value) != float(sample_rate):
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return sample_rate


def resolve_device_sample_rate(device, role_label: str) -> SampleRateResolution:
    if not device:
        return _fail(f"未选择{role_label}，请在硬件管理中选择设备。")
    if not isinstance(device, dict):
        return _fail(f"{role_label}信息无效，请在硬件管理中重新选择设备。")
    if "samplerate" not in device or device.get("samplerate") in (None, ""):
        return _fail(f"{role_label}缺少采样率，请在硬件管理中设置采样率。")
    sample_rate = _coerce_exact_integer_sample_rate(device.get("samplerate"))
    if sample_rate is None:
        return _fail(f"{role_label}采样率无效，请在硬件管理中设置为44100或48000。")
    if sample_rate not in VALID_SAMPLE_RATES:
        return _fail(f"{role_label}采样率无效，请在硬件管理中设置为44100或48000。")
    return SampleRateResolution(True, sample_rate, "")


def resolve_input_sample_rate(mic) -> SampleRateResolution:
    return resolve_device_sample_rate(mic, "输入设备")


def resolve_output_sample_rate(speaker) -> SampleRateResolution:
    return resolve_device_sample_rate(speaker, "输出设备")


def resolve_duplex_sample_rate(mic, speaker) -> SampleRateResolution:
    input_result = resolve_input_sample_rate(mic)
    if not input_result.ok:
        return input_result
    output_result = resolve_output_sample_rate(speaker)
    if not output_result.ok:
        return output_result
    if input_result.sample_rate != output_result.sample_rate:
        return _fail("输入设备与输出设备的采样率不一致，请在硬件管理中设置为一致后重新选择。")
    return SampleRateResolution(True, input_result.sample_rate, "")
