"""Build a station recording request from committed configuration and prepared hardware."""

import logging
import math
import os
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import PureWindowsPath

from base.file_ops import FileOps
from base.play_and_record import resolve_startup_trim_samples
from base.product_model_validation import validate_product_model
from base.recording_calibration_snapshot import build_recording_wav_calibration_metadata
from base.recording_process_protocol import RecordingRequest
from base.recording_settings import resolve_audio_validation_thresholds, resolve_startup_trim_ms
from base.soundcard_calibration_manager import MicCalibrationError
from base.workbench_recording_protocol import (
    StationRecordingGroup, StationRecordingPlan, StationWavChannel, StationWavName, validate_station_wav_length,
)


class WorkbenchRecordingRequestError(ValueError):
    """A workbench condition cannot be recorded with the current settings."""


def _plain_detail(parameters):
    # Legacy setting resolvers require dicts. The repository exposes frozen mappings.
    detail = dict(parameters)
    if isinstance(detail.get("audio_validation"), Mapping):
        detail["audio_validation"] = dict(detail["audio_validation"])
    return detail


def _recording_root(root):
    if not isinstance(root, str) or not root.strip() or "\0" in root:
        raise ValueError("产品结果根目录必须为非空的有效路径。")
    return os.path.abspath(root)


def _safe_name(value, fallback):
    # Spaces within human names are valid; only invalid Windows characters and
    # terminal dots/spaces need changing for these path components.
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(value or "")).strip(" .") or fallback
    return "_" + name if PureWindowsPath(name).is_reserved() else name


def _applicable_stations(condition, stations):
    return tuple(station for station in stations if station.station_id in condition.applicable_station_ids)


def _workbench_validation_thresholds(detail):
    thresholds = resolve_audio_validation_thresholds(detail)
    if not thresholds["enabled"]:
        return {"enabled": False}
    effective = {"enabled": True}
    for name, value in thresholds.items():
        if name == "enabled":
            continue
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"audio_validation.{name} 的数字超出有限范围。")
        try:
            limit = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"audio_validation.{name} 必须能转换为数值。") from exc
        # The gate compares against a lower bound. Retain special string
        # values in serializable form, with NaN acting as no lower bound.
        if math.isnan(limit):
            limit = -math.inf
        effective[name] = limit if math.isfinite(limit) else str(limit)
    return effective


def _recording_options_key(detail):
    try:
        trim = resolve_startup_trim_ms(detail)
    except OverflowError:
        # Keep invalid input distinct so the request boundary can diagnose it.
        trim = detail["startup_trim_ms"]
    try:
        thresholds = _workbench_validation_thresholds(detail)
    except ValueError as exc:
        # The operator can still click to see which field needs correction.
        thresholds = str(exc)
    return detail["total_time"], trim, thresholds


def recording_config_key(recording, *, source_identity):
    """Compare effective recording work, independent of analysis and UI settings."""
    context = recording.context
    return context.project_id, context.project_name, context.result_root_directory, os.path.normcase(source_identity), tuple(
        (condition.condition_id, condition.condition_name,
         _recording_options_key(_plain_detail(recording.parameters_for(condition.condition_id))),
         tuple((station.station_id, station.station_name, station.channels)
               for station in _applicable_stations(condition, context.stations)))
        for condition in context.conditions
    )


def _channel_mappings(stations, physical_to_column):
    device_ref = None
    for station in stations:
        if not station.channels:
            raise ValueError(f"工位「{station.station_name}」({station.station_id}) 没有录制通道。")
        physical_channels = set()
        mapping = []
        for wav_index, channel in enumerate(station.channels):
            hardware = channel.hardware_channel
            physical_index = hardware.physical_channel_index
            source = (f"工位「{station.station_name}」({station.station_id}) "
                      f"通道「{channel.channel_name}」({channel.channel_id}, 物理索引 {physical_index})")
            if physical_index not in physical_to_column:
                raise ValueError(f"{source} 不在当前录制设备的输入通道中。")
            if physical_index in physical_channels:
                raise ValueError(f"{source} 在该工位内重复使用。")
            if device_ref is None:
                device_ref = hardware.device_ref
            if hardware.device_ref != device_ref:
                raise ValueError(f"{source} 的设备分组 {hardware.device_ref!r} 与 {device_ref!r} 不一致。")
            physical_channels.add(physical_index)
            mapping.append(StationWavChannel(
                channel.channel_id, channel.channel_name, channel.signal_type,
                hardware.device_ref, physical_index, wav_index,
            ))
        yield station, tuple(mapping)


def _calibration_snapshot(device, channels, ve_calibration_store, condition_label):
    try:
        return build_recording_wav_calibration_metadata(
            channels, device, ve_calibration_store=ve_calibration_store,
        )
    except MicCalibrationError as exc:
        # Soundcard WAV metadata is optional. Keep the calibration error diagnosable.
        logging.getLogger("core").error("%s 声卡校准元数据读取失败：%s", condition_label, exc)
        return None


def _required_wav_component(value, label):
    if not isinstance(value, str):
        raise ValueError(f"{label}必须为文本。")
    component = FileOps.sanitize_file_name_component(value)
    if not component:
        raise ValueError(f"{label}为空或无法用于 WAV 文件名，请修正。")
    return component


def _station_wav_names(stations, timestamp, condition_name, *, product_model, sample_number, serial_number):
    if not isinstance(product_model, str):
        raise ValueError("型号必须为文本。")
    model = _required_wav_component(validate_product_model(product_model), "型号")
    sample = _required_wav_component(sample_number, "样本编号")
    if not isinstance(serial_number, str):
        raise ValueError("S/N 必须为文本。")
    serial = FileOps.sanitize_file_name_component(serial_number)
    if serial_number.strip() and not serial:
        raise ValueError("S/N 无法用于 WAV 文件名，请修正或清空 S/N。")
    condition = _required_wav_component(condition_name, "工况名称")
    used_names = set()
    for station in stations:
        label = f"工位「{station.station_name}」({station.station_id})"
        station_name = _required_wav_component(station.station_name, label + "的名称")
        candidate = StationWavName("_".join((timestamp, station_name, condition, model, sample)), serial)
        number = 1
        filename = candidate.filename_for(number)
        while filename.casefold() in used_names:
            number += 1
            filename = candidate.filename_for(number)
        try:
            validate_station_wav_length(filename)
        except ValueError as exc:
            raise ValueError(f"{label} {exc}") from exc
        used_names.add(filename.casefold())
        yield filename, candidate


def build_recording_request(recording, condition_id, device_snapshot, recording_id, *,
                            recorded_at, product_model, sample_number, serial_number="",
                            ve_calibration_store=None, preview_time_mode=None):
    """Build from recording-only parameters and the prepared device snapshot."""
    context = recording.context
    condition = context.get_condition(condition_id)
    if condition is None:
        raise WorkbenchRecordingRequestError(
            f"产品 {context.project_id} 未知录制工况 {condition_id!r}。"
        )
    return _build_recording_request(
        condition, context.stations, _plain_detail(recording.parameters_for(condition_id)),
        device_snapshot, recording_id, project_id=context.project_id,
        project_name=context.project_name, result_root_directory=context.result_root_directory,
        recorded_at=recorded_at, product_model=product_model, sample_number=sample_number,
        serial_number=serial_number, ve_calibration_store=ve_calibration_store,
        preview_time_mode=preview_time_mode,
    )


def _build_recording_request(condition, stations, detail, device_snapshot, recording_id, *,
                             project_id, project_name, result_root_directory, recorded_at,
                             product_model, sample_number, serial_number,
                             ve_calibration_store, preview_time_mode=None):
    """Freeze the full capture and each station's ordered PCM24 WAV mapping."""
    label = f"产品 {project_id} 工况「{condition.condition_name}」({condition.condition_id})"
    recorded_at = datetime.now() if recorded_at is None else recorded_at
    try:
        if not isinstance(recording_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", recording_id):
            raise ValueError("录制 ID 必须为安全的文件名标识。")
        applicable = _applicable_stations(condition, stations)
        if not applicable:
            raise ValueError("没有关联录制工位，请检查产品测试配置。")
        sample_rate = device_snapshot.sample_rate
        kept_frames = int(detail["total_time"] * sample_rate)
        if kept_frames < 1:
            raise ValueError(f"录制时长在设备采样率 {sample_rate} Hz 下不足一帧。")
        trim_samples = resolve_startup_trim_samples(detail, sample_rate)
        wav_directory = os.path.join(
            _recording_root(result_root_directory),
            _safe_name(project_name, "产品"),
            "audio",
        )
        timestamp = recorded_at.strftime("%Y-%m-%d_%H-%M-%S-") + f"{recorded_at.microsecond // 1000:03d}"
        names = tuple(_station_wav_names(
            applicable, timestamp, condition.condition_name, product_model=product_model,
            sample_number=sample_number, serial_number=serial_number,
        ))
        thresholds = _workbench_validation_thresholds(detail)
        mappings = tuple(_channel_mappings(applicable, device_snapshot.physical_to_column))
        device = device_snapshot.device.to_dict()
        metadata = _calibration_snapshot(device, device_snapshot.channels, ve_calibration_store, label)
        by_physical = {} if metadata is None else {
            entry["physical_input_channel"]: entry for entry in metadata["recorded_channels"]
        }
        plans = []
        for (station, channels), (filename, candidate) in zip(mappings, names, strict=True):
            station_metadata = None if metadata is None else {
                **metadata,
                "recorded_channels": [
                    {**by_physical[channel.physical_channel_index], "wav_channel_index": channel.wav_channel_index}
                    for channel in channels
                ],
            }
            plans.append(StationRecordingPlan(station.station_id, filename,
                                             channels, station_metadata, wav_name=candidate))
        group = StationRecordingGroup(
            project_id, condition.condition_id, wav_directory, tuple(plans),
            prepared_generation=device_snapshot.generation,
            publish_into_parent=True,
        )
        preview_options = {} if preview_time_mode is None else {"preview_time_mode": preview_time_mode}
        return RecordingRequest(
            recording_id, "main", sample_rate, kept_frames + trim_samples, device_snapshot.channels,
            device, os.path.join(wav_directory, f".{timestamp}-{recording_id}.staging"),
            True, trim_samples, {}, None,
            thresholds, station_group=group, **preview_options,
        )
    except (ValueError, OverflowError, OSError) as exc:
        raise WorkbenchRecordingRequestError(f"{label} 录制配置错误：{exc}") from exc
