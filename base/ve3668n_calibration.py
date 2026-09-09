"""VE raw-voltage calibration validation, separate from the legacy SPL maths."""
from dataclasses import replace

import numpy as np

from base.recording_process_protocol import FrozenConfig, RecordingRequest, RecordingResult


def _single_channel(volts):
    data = np.asarray(volts)
    if data.ndim == 2 and data.shape[1] == 1:
        data = data[:, 0]
    if data.ndim != 1:
        raise ValueError("校准数据必须是单通道原始电压")
    return data


def require_ac_signal(volts):
    """Unconditional numerical AC floor, not a new SPL/quality algorithm.

    Demeaning is local to this check. The legacy RMS/dB calculation still
    receives the unchanged raw voltage, including any DC component.
    """
    data = np.asarray(_single_channel(volts), dtype=np.float64)
    if len(data) < 1201 or not np.isfinite(data).all():
        raise ValueError("校准数据过短或包含非有限值")
    step, middle = len(data) // 3, len(data) // 2
    central = data[middle - step:middle + step]
    with np.errstate(over="ignore", invalid="ignore"):
        ac_rms = np.sqrt(np.mean((central - central.mean()) ** 2))
    if not np.isfinite(ac_rms) or ac_rms <= 1e-10:
        raise ValueError("无有效交流信号，无法校准")


def verify_ve_calibration_result(request, descriptor, volts):
    """Validate a frozen ten-second request and its reader-owned raw V.

    Capture and ResultReader independently verify the WAV's FLOAT format before
    publishing. RecordingResult has no subtype field; do not reopen its path,
    which the service may already have deleted. ``handles_released`` attests to
    capture handles only: the UI must ALSO await accepted + session.released.
    Returns the original mono array (or an Nx1 view), never scaled by a K.
    """
    if not isinstance(request, RecordingRequest) or not isinstance(request.device, FrozenConfig):
        raise ValueError("校准请求必须是冻结的 RecordingRequest")
    # Re-run the common protocol boundary, including stable VE identity,
    # availability, physical channel, IEPE/V, rate, duration, trim and monitor.
    replace(request)
    if (request.device.get("backend") != "vkinging" or request.purpose != "calibration"
            or request.streaming or request.calibration_metadata is not None):
        raise ValueError("VE 校准必须录制十秒单通道原始电压")
    if not isinstance(descriptor, RecordingResult):
        raise ValueError("校准录音结果描述无效")
    if (descriptor.request_id != request.request_id or descriptor.purpose != request.purpose
            or descriptor.path != request.path or descriptor.channels != request.channels
            or type(descriptor.channels) is not tuple
            or any(type(channel) is not int for channel in descriptor.channels)
            or descriptor.handles_released is not True):
        raise ValueError("校准录音身份、通道或资源释放状态不匹配")
    for field, expected in (("sample_rate", request.sample_rate),
                            ("raw_frames", request.target_samples),
                            ("final_frames", request.target_samples)):
        value = getattr(descriptor, field)
        if type(value) is not int or value != expected:
            raise ValueError(f"校准录音 {field} 与冻结请求不匹配")
    data = _single_channel(volts)
    if data.dtype != np.float32 or not np.isfinite(data).all():
        raise ValueError("校准数据必须是有限 float32 原始电压")
    step, middle = len(data) // 3, len(data) // 2
    if (len(data) != request.target_samples or len(data) < 1201 or step == 0
            or not 0 <= middle - step < middle + step <= len(data)):
        raise ValueError("校准录音长度或 RMS 统计区间无效")
    require_ac_signal(data)
    return data
