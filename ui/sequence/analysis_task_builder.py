"""Build immutable analysis tasks from one exact selected WAV."""

from __future__ import annotations

import copy
import os
from uuid import uuid4

import soundfile as sf

from base.analysis_process_protocol import (
    AnalysisChannelMapping,
    AnalysisInstanceRequest,
    AnalysisTaskRequest,
    SUPPORTED_ANALYSIS_TYPES,
    build_runtime_key,
)
from base.wav_calibration_metadata import (
    WavCalibrationMetadataReadStatus,
    inspect_wav_calibration_metadata,
    resolve_wav_channel_v2pa_factor,
)
from base.wav_channel_mapping import resolve_wav_plot_channels
from ui.ui_analysis_config.config_normalization import normalize_analysis_channels


class AnalysisTaskBuildError(ValueError):
    """The selected recording/config cannot form one deterministic task."""


def build_analysis_task_request(
    *,
    condition_key,
    wav_path,
    source,
    sequence_config,
    analysis_config,
    storage_snapshot=None,
    saved_active_input_channels=None,
    fallback_v2pa_factors=None,
    task_id=None,
):
    absolute_path = os.path.abspath(str(wav_path or "").strip())
    if not absolute_path or not os.path.isfile(absolute_path):
        raise AnalysisTaskBuildError("所选档位的 WAV 文件不存在")
    if not isinstance(analysis_config, dict):
        raise AnalysisTaskBuildError("当前分析配置无效")

    try:
        with sf.SoundFile(absolute_path, mode="r") as source_file:
            channel_count = int(source_file.channels)
            frame_count = int(source_file.frames)
            sample_rate = int(source_file.samplerate)
    except Exception as error:
        raise AnalysisTaskBuildError(f"WAV 文件无法读取：{error}") from error
    if channel_count <= 0 or frame_count <= 0 or sample_rate <= 0:
        raise AnalysisTaskBuildError("WAV 文件描述信息无效")

    metadata_result = inspect_wav_calibration_metadata(absolute_path)
    if metadata_result.status is WavCalibrationMetadataReadStatus.INVALID:
        raise AnalysisTaskBuildError("WAV 通道或校准元数据无效")
    try:
        raw_channels = resolve_wav_plot_channels(
            metadata_result,
            column_count=channel_count,
            saved_active_input_channels=saved_active_input_channels,
        )
    except ValueError as error:
        raise AnalysisTaskBuildError(f"WAV 通道映射无效：{error}") from error

    channel_mapping = tuple(
        AnalysisChannelMapping(raw_channel, column)
        for column, raw_channel in enumerate(raw_channels)
    )
    column_by_raw_channel = {
        mapping.raw_channel: mapping.source_wav_column
        for mapping in channel_mapping
    }
    fallback_factors = {
        int(key): float(value)
        for key, value in dict(fallback_v2pa_factors or {}).items()
        if not isinstance(key, bool)
        and not isinstance(value, bool)
        and float(value) > 0.0
    }

    display_sequence = analysis_config.get("display_sequence", [])
    if not isinstance(display_sequence, (list, tuple)):
        raise AnalysisTaskBuildError("分析项顺序配置无效")
    instances = []
    missing_channels = []
    for config_key in display_sequence:
        key = str(config_key or "").strip()
        parameters = analysis_config.get(key)
        if not key or not isinstance(parameters, dict):
            continue
        analysis_type = str(parameters.get("type") or "").strip()
        if analysis_type not in SUPPORTED_ANALYSIS_TYPES:
            continue
        for raw_channel in normalize_analysis_channels(parameters):
            source_column = column_by_raw_channel.get(raw_channel)
            if source_column is None:
                missing_channels.append(f"{key}: CH{raw_channel + 1}")
                continue
            calibration = resolve_wav_channel_v2pa_factor(
                metadata_result.metadata,
                source_column,
            )
            factor = (
                calibration.factor
                if calibration.used_file_metadata
                else fallback_factors.get(raw_channel, 1.0)
            )
            calibration_available = bool(
                calibration.used_file_metadata
                or raw_channel in fallback_factors
            )
            runtime_parameters = copy.deepcopy(parameters)
            runtime_parameters["analysis_channel"] = source_column
            instances.append(
                AnalysisInstanceRequest(
                    config_key=key,
                    runtime_key=build_runtime_key(key, raw_channel),
                    analysis_type=analysis_type,
                    raw_channel=raw_channel,
                    source_wav_column=source_column,
                    v2pa_factor=factor,
                    parameters=runtime_parameters,
                    calibration_available=calibration_available,
                )
            )
    if missing_channels:
        raise AnalysisTaskBuildError(
            "以下分析通道不在该 WAV 中：" + "、".join(missing_channels)
        )
    if not instances:
        raise AnalysisTaskBuildError(
            "当前配置没有可执行的 SPL、Spec、FBA、AI 或 FFT 分析项"
        )

    sequence_snapshot = {
        "sequence_config": copy.deepcopy(sequence_config or []),
        "sample_rate": sample_rate,
        "frame_count": frame_count,
    }
    return AnalysisTaskRequest(
        task_id=str(task_id or uuid4()),
        condition_key=str(condition_key or "").strip(),
        wav_path=absolute_path,
        source=source,
        channel_mapping=channel_mapping,
        sequence_config_snapshot=sequence_snapshot,
        analysis_config_snapshot=copy.deepcopy(analysis_config),
        storage_snapshot=copy.deepcopy(storage_snapshot or {}),
        instances=tuple(instances),
    )
