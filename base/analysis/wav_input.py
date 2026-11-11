"""WAV data and calibration boundaries; no device or active-config lookup."""

import math

import numpy as np
import soundfile as sf

from base.wav_calibration_metadata import (
    WavCalibrationMetadataReadStatus,
    inspect_wav_calibration_metadata,
    resolve_wav_channel_v2pa_factor,
)


def load_task_wav(task):
    samples, sample_rate = sf.read(task.wav_path, dtype="float32", always_2d=True)
    if samples.shape[0] == 0:
        raise ValueError("WAV 没有音频帧")
    if samples.shape[1] != len(task.channels):
        raise ValueError("WAV 通道数与任务 channels 数量不一致")
    metadata = inspect_wav_calibration_metadata(task.wav_path)
    if not metadata.handles_released:
        # The reader explicitly transfers uncertain handles to its caller.
        for _, handle in metadata.retained_handles:
            handle.close()
        raise OSError("WAV 标定元数据读取时文件句柄关闭异常")
    return samples, sample_rate, metadata


def resolve_calibration(channel, metadata):
    """Resolve exactly one multiplier from decoded WAV values to Pa."""
    if metadata.status is WavCalibrationMetadataReadStatus.INVALID:
        raise ValueError("WAV 标定元数据损坏，不能确定样本标定来源")
    file_factor = None
    if metadata.status is WavCalibrationMetadataReadStatus.VALID:
        resolution = resolve_wav_channel_v2pa_factor(
            metadata.metadata, channel.channel_index,
        )
        if resolution.used_file_metadata:
            file_factor = resolution.factor
    supplied = channel.v2pa_factor
    if supplied is not None and (not math.isfinite(supplied) or supplied <= 0):
        raise ValueError("任务标定系数必须为有限正数")
    if supplied is not None and file_factor is not None:
        if not math.isclose(supplied, file_factor, rel_tol=1e-9, abs_tol=0.0):
            raise ValueError("任务标定系数与 WAV 对应列的标定系数不一致")
        return supplied, "task_and_wav"
    if supplied is not None:
        return supplied, "task"
    if file_factor is not None:
        return file_factor, "wav"
    raise ValueError("任务和 WAV 均未提供该通道的有效声压标定系数")


def sample_window(config, frame_count, sample_rate):
    if not config.get("analysis_time_range_enabled", False):
        return 0, frame_count
    start = float(config.get("analysis_start_time_sec", 0.0))
    end = float(config.get("analysis_end_time_sec", 0.0))
    duration = frame_count / sample_rate
    if end == 0.0:
        end = duration
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
        raise ValueError("分析时间范围无效")
    if end > duration and not math.isclose(end, duration, abs_tol=1e-12, rel_tol=0):
        raise ValueError("分析结束时间超过 WAV 实际时长")
    first = int(np.floor(start * sample_rate))
    stop = min(frame_count, int(np.ceil(end * sample_rate)))
    if stop <= first:
        raise ValueError("分析时间范围内没有音频帧")
    return first, stop
