"""One-shot analysis worker.  One process handles exactly one WAV task."""

from __future__ import annotations

import base64
from collections import OrderedDict
import gc
import math
import os
from pathlib import Path
import tempfile
import time
import traceback

import numpy as np
import soundfile as sf

from base.analysis_artifact_paths import (
    build_channel_image_path,
    storage_context_from_metadata,
)
from base.analysis_algorithm_adapters import calculate_analysis_instance
from base.analysis_csv_exporter import export_item_csvs
from base.analysis_display_payload import min_max_envelope, sample_curve_at_interval
from base.analysis_plot_renderer import render_analysis_png
from base.analysis_process_protocol import (
    AnalysisArtifactResult,
    AnalysisInstanceResult,
    AnalysisProgress,
    AnalysisWorkerFailure,
)
from base.analysis_result_summary import summarize_analysis_task


def analysis_worker_main(request, event_queue, log_queue=None):
    """Process entrypoint; returns only serializable value objects."""
    started_at = time.monotonic()
    _emit_log(
        log_queue,
        "INFO",
        "analysis_task_started",
        request,
        pid=os.getpid(),
    )
    try:
        wav_started_at = time.monotonic()
        audio, sample_rate = _load_wav_once(request)
        _emit_log(
            log_queue,
            "INFO",
            "analysis_wav_loaded",
            request,
            frame_count=int(audio.shape[0]),
            channel_count=int(audio.shape[1]),
            sample_rate=sample_rate,
            duration_seconds=round(time.monotonic() - wav_started_at, 3),
        )
        result = _execute_task(request, audio, sample_rate, event_queue, log_queue)
        event_queue.put(("result", result))
        _emit_log(
            log_queue,
            "INFO",
            "analysis_task_finished",
            request,
            pid=os.getpid(),
            execution_status=result.execution_status,
            final_judgement=result.final_judgement or "",
            duration_seconds=round(time.monotonic() - started_at, 3),
        )
    except BaseException as error:
        failure = AnalysisWorkerFailure(
            task_id=request.task_id,
            stage="任务执行",
            error_type=type(error).__name__,
            message=str(error) or type(error).__name__,
            traceback_text=traceback.format_exc(),
        )
        event_queue.put(("failure", failure))
        _emit_log(
            log_queue,
            "ERROR",
            "analysis_task_failed",
            request,
            pid=os.getpid(),
            error_type=failure.error_type,
            error_message=failure.message,
            traceback_text=failure.traceback_text,
            duration_seconds=round(time.monotonic() - started_at, 3),
        )
    finally:
        try:
            del audio
        except UnboundLocalError:
            pass
        gc.collect()


def _load_wav_once(request):
    audio, sample_rate = sf.read(
        request.wav_path,
        dtype="float32",
        always_2d=True,
    )
    audio = np.asarray(audio, dtype=np.float32, order="C")
    if audio.ndim != 2 or audio.shape[0] <= 0:
        raise ValueError("WAV 没有可分析的音频帧")
    expected_columns = len(request.channel_mapping)
    if audio.shape[1] != expected_columns:
        raise ValueError(
            f"WAV 通道数已变化：任务={expected_columns}，文件={audio.shape[1]}"
        )
    return audio, int(sample_rate)


def _execute_task(request, audio, sample_rate, event_queue, log_queue):
    storage_metadata = request.storage_snapshot.to_dict()
    channel_labels = dict(storage_metadata.get("channel_labels", {}) or {})
    storage_context = None
    storage_error = ""
    if request.source == "自动分析":
        try:
            storage_context = storage_context_from_metadata(
                storage_metadata
            )
        except (TypeError, ValueError) as error:
            storage_error = str(error)

    groups = OrderedDict()
    for instance_request in request.instances:
        groups.setdefault(instance_request.config_key, []).append(instance_request)

    all_results = []
    completed_count = 0
    total_count = len(request.instances)
    analysis_config = request.analysis_config_snapshot.to_dict()
    sequence_snapshot = request.sequence_config_snapshot.to_dict()

    for config_key, instance_requests in groups.items():
        item_started_at = time.monotonic()
        channel_outputs = []
        pending_results = []
        item_config = analysis_config.get(config_key, {})
        for instance_request in instance_requests:
            event_queue.put(
                (
                    "progress",
                    AnalysisProgress(
                        request.task_id,
                        "分析中",
                        completed_count,
                        total_count,
                        f"{config_key} CH{instance_request.raw_channel + 1}",
                        instance_request.runtime_key,
                    ),
                )
            )
            started_instance = time.monotonic()
            try:
                _emit_log(
                    log_queue,
                    "INFO",
                    "analysis_instance_started",
                    request,
                    config_key=instance_request.config_key,
                    analysis_type=instance_request.analysis_type,
                    runtime_key=instance_request.runtime_key,
                    raw_channel=instance_request.raw_channel,
                )
                output, image_payload = _execute_instance(
                    request,
                    instance_request,
                    audio[:, instance_request.source_wav_column],
                    sample_rate,
                    sequence_snapshot,
                )
                artifacts = []
                if request.source == "自动分析":
                    if storage_context is None:
                        image_artifact = AnalysisArtifactResult(
                            "图片",
                            "保存失败",
                            error_message=storage_error or "缺少结果目录上下文",
                        )
                    else:
                        image_artifact = _save_instance_image(
                            storage_context,
                            Path(request.wav_path).stem,
                            config_key,
                            instance_request.raw_channel,
                            image_payload,
                            channel_labels,
                        )
                    artifacts.append(image_artifact)
                    _emit_log(
                        log_queue,
                        "INFO" if image_artifact.status == "已保存" else "WARNING",
                        (
                            "analysis_image_saved"
                            if image_artifact.status == "已保存"
                            else "analysis_image_save_failed"
                        ),
                        request,
                        runtime_key=instance_request.runtime_key,
                        artifact_path=image_artifact.path,
                        error_message=image_artifact.error_message,
                    )
                output["artifacts"] = artifacts
                channel_outputs.append(output)
                pending_results.append((instance_request, output))
                _emit_log(
                    log_queue,
                    "INFO",
                    "analysis_instance_finished",
                    request,
                    config_key=instance_request.config_key,
                    analysis_type=instance_request.analysis_type,
                    runtime_key=instance_request.runtime_key,
                    raw_channel=instance_request.raw_channel,
                    duration_seconds=round(time.monotonic() - started_instance, 3),
                )
            except BaseException as error:
                contributes = _contributes_to_final(
                    instance_request.analysis_type,
                    item_config,
                )
                failed_output = {
                    "raw_channel": instance_request.raw_channel,
                    "source_wav_column": instance_request.source_wav_column,
                    "execution_status": "分析失败",
                    "contributes_to_final": contributes,
                    "judgement": None,
                    "metrics": {},
                    "display_payload": {},
                    "artifacts": (
                        AnalysisArtifactResult("图片", "不保存"),
                    ),
                    "error_stage": "算法计算",
                    "error_type": type(error).__name__,
                    "error_message": str(error) or type(error).__name__,
                }
                pending_results.append((instance_request, failed_output))
                _emit_log(
                    log_queue,
                    "ERROR",
                    "analysis_instance_failed",
                    request,
                    config_key=instance_request.config_key,
                    analysis_type=instance_request.analysis_type,
                    runtime_key=instance_request.runtime_key,
                    raw_channel=instance_request.raw_channel,
                    error_type=type(error).__name__,
                    error_message=str(error),
                    traceback_text=traceback.format_exc(),
                    duration_seconds=round(time.monotonic() - started_instance, 3),
                )
            completed_count += 1
            event_queue.put(
                (
                    "progress",
                    AnalysisProgress(
                        request.task_id,
                        "分析中",
                        completed_count,
                        total_count,
                        f"已完成 {completed_count}/{total_count}",
                        instance_request.runtime_key,
                    ),
                )
            )

        csv_artifacts = ()
        if request.source == "自动分析" and channel_outputs:
            if storage_context is None:
                csv_artifacts = (
                    AnalysisArtifactResult(
                        "CSV",
                        "保存失败",
                        error_message=storage_error or "缺少结果目录上下文",
                    ),
                )
            else:
                csv_records = export_item_csvs(
                    storage_context,
                    Path(request.wav_path).stem,
                    config_key,
                    instance_requests[0].analysis_type,
                    item_config,
                    channel_outputs,
                    channel_labels,
                )
                csv_artifacts = tuple(
                    AnalysisArtifactResult(
                        f"CSV:{record.config_item_name}",
                        "已保存" if record.ok else "保存失败",
                        path=record.file_path,
                        error_message=record.error_message,
                    )
                    for record in csv_records
                )
                for record in csv_records:
                    _emit_log(
                        log_queue,
                        "INFO" if record.ok else "WARNING",
                        "analysis_csv_saved" if record.ok else "analysis_csv_save_failed",
                        request,
                        config_key=config_key,
                        config_item_name=record.config_item_name,
                        artifact_path=record.file_path,
                        error_message=record.error_message,
                    )

        csv_attached = False
        for instance_request, output in pending_results:
            artifacts = tuple(output.get("artifacts") or ())
            if csv_artifacts and not csv_attached and output["execution_status"] == "分析完成":
                artifacts = artifacts + csv_artifacts
                csv_attached = True
            all_results.append(
                AnalysisInstanceResult(
                    task_id=request.task_id,
                    config_key=instance_request.config_key,
                    runtime_key=instance_request.runtime_key,
                    analysis_type=instance_request.analysis_type,
                    raw_channel=instance_request.raw_channel,
                    source_wav_column=instance_request.source_wav_column,
                    execution_status=output["execution_status"],
                    contributes_to_final=output["contributes_to_final"],
                    judgement=output["judgement"],
                    metrics=_plain_mapping(output.get("metrics") or {}),
                    display_payload=_plain_mapping(
                        output.get("display_payload") or {}
                    ),
                    artifacts=artifacts,
                    error_stage=output.get("error_stage", ""),
                    error_type=output.get("error_type", ""),
                    error_message=output.get("error_message", ""),
                )
            )

        successful_channels = sum(
            1
            for _instance_request, output in pending_results
            if output["execution_status"] == "分析完成"
        )
        _emit_log(
            log_queue,
            "INFO" if successful_channels == len(pending_results) else "WARNING",
            "analysis_item_finished",
            request,
            config_key=config_key,
            analysis_type=instance_requests[0].analysis_type,
            channel_count=len(pending_results),
            successful_channels=successful_channels,
            failed_channels=len(pending_results) - successful_channels,
            duration_seconds=round(time.monotonic() - item_started_at, 3),
        )

    return summarize_analysis_task(request, all_results)


def _execute_instance(
    request,
    instance_request,
    signal,
    sample_rate,
    sequence_snapshot,
):
    config = instance_request.parameters.to_dict()
    calculation = calculate_analysis_instance(
        instance_request.analysis_type,
        signal,
        sample_rate,
        config,
        instance_request.v2pa_factor,
        source=request.source,
        sequence_snapshot=sequence_snapshot,
    )
    try:
        image_payload = render_analysis_png(calculation.get("plot"))
        contributes = _contributes_to_final(
            instance_request.analysis_type,
            config,
        )
        judgement = calculation.get("judgement")
        if contributes and judgement is None:
            raise RuntimeError("该分析项未产生必需的 OK/NG 判定")
        normalized = _extract_output(
            instance_request,
            calculation,
            judgement,
            request.source,
            image_payload,
        )
        normalized.update(
            {
                "raw_channel": instance_request.raw_channel,
                "source_wav_column": instance_request.source_wav_column,
                "execution_status": "分析完成",
                "contributes_to_final": contributes,
                "judgement": judgement if contributes else None,
            }
        )
        return normalized, image_payload
    finally:
        calculation.clear()
        gc.collect()


def _extract_output(instance_request, calculation, judgement, source, image_payload):
    metrics = dict(calculation.get("metrics") or {})
    curve = dict(calculation.get("curve") or {})
    analysis_type = instance_request.analysis_type
    if analysis_type == "SPL":
        x = _array_value(curve.get("x"))
        y = _array_value(curve.get("y"))
        upper = _array_value(curve.get("upper"))
        lower = _array_value(curve.get("lower"))
        display_x, display_y = min_max_envelope(x, y)
        display_upper = _interpolate_optional(x, upper, display_x)
        display_lower = _interpolate_optional(x, lower, display_x)
        csv_x, csv_y = sample_curve_at_interval(x, y)
        csv_upper = _interpolate_optional(x, upper, csv_x)
        csv_lower = _interpolate_optional(x, lower, csv_x)
        return {
            "metrics": metrics,
            "display_payload": {
                "kind": "curve",
                "x": display_x,
                "y": display_y,
                "lower": display_lower,
                "upper": display_upper,
                "x_label": "Time (s)",
                "y_label": "SPL (dB)",
                "overall_spl": metrics.get("overall_spl"),
            }
            if source == "手动查看"
            else {},
            "csv_curve": {
                "x": csv_x,
                "y": csv_y,
                "lower": csv_lower,
                "upper": csv_upper,
            },
        }
    if analysis_type == "FBA":
        x_values = _array_value(curve.get("x"))
        y_values = _array_value(curve.get("y"))
        lower_values = _array_value(curve.get("lower"))
        upper_values = _array_value(curve.get("upper"))
        plot = dict(calculation.get("plot") or {})
        return {
            "metrics": metrics,
            "display_payload": {
                "kind": "bar",
                "x": x_values.tolist(),
                "y": y_values.tolist(),
                "labels": [str(label) for label in plot.get("labels", ())],
                "lower": lower_values.tolist(),
                "upper": upper_values.tolist(),
                "out_mask": np.asarray(
                    plot.get("out_mask")
                    if plot.get("out_mask") is not None
                    else [],
                    dtype=bool,
                ).reshape(-1).tolist(),
                "x_label": str(plot.get("x_label") or "Frequency Band"),
                "y_label": str(plot.get("y_label") or "Band SPL (dB)"),
            }
            if source == "手动查看"
            else {},
            "csv_curve": {
                "x": x_values.tolist(),
                "y": y_values.tolist(),
                "lower": lower_values.tolist(),
                "upper": upper_values.tolist(),
            },
        }
    if analysis_type == "FFT":
        x_values = _array_value(curve.get("x"))
        y_values = _array_value(curve.get("y"))
        lower_values = _array_value(curve.get("lower"))
        upper_values = _array_value(curve.get("upper"))
        display_x, display_y = min_max_envelope(x_values, y_values)
        display_lower = _interpolate_optional(x_values, lower_values, display_x)
        display_upper = _interpolate_optional(x_values, upper_values, display_x)
        plot = dict(calculation.get("plot") or {})
        return {
            "metrics": metrics,
            "display_payload": {
                "kind": "curve",
                "x": display_x,
                "y": display_y,
                "lower": display_lower,
                "upper": display_upper,
                "x_label": "Frequency (Hz)",
                "y_label": "FFT (dB)",
                "log_x": bool(plot.get("log_x")),
            }
            if source == "手动查看"
            else {},
            "csv_curve": {
                "x": x_values.tolist(),
                "y": y_values.tolist(),
                "lower": lower_values.tolist(),
                "upper": upper_values.tolist(),
            },
        }
    if analysis_type == "AI":
        return {
            "metrics": metrics,
            "display_payload": {"kind": "values", **metrics, "result": judgement}
            if source == "手动查看"
            else {},
            "csv_curve": {},
        }
    if analysis_type == "Spec":
        return {
            "metrics": {},
            "display_payload": {
                "kind": "image",
                "png_base64": base64.b64encode(image_payload).decode("ascii"),
            }
            if source == "手动查看"
            else {},
            "csv_curve": {},
        }
    raise ValueError(f"不支持的分析类型：{analysis_type}")


def _contributes_to_final(analysis_type, config):
    if analysis_type == "AI":
        return True
    if analysis_type in {"SPL", "FBA", "FFT"}:
        return bool((config or {}).get("limit_checked", False))
    return False


def _save_instance_image(
    storage_context,
    wav_stem,
    config_key,
    raw_channel,
    png_data,
    channel_labels,
):
    path = build_channel_image_path(
        storage_context,
        wav_stem,
        config_key,
        raw_channel,
        channel_labels,
    )
    try:
        _write_bytes_atomic(path, png_data)
    except (OSError, ValueError) as error:
        return AnalysisArtifactResult(
            "图片",
            "保存失败",
            error_message=str(error),
        )
    return AnalysisArtifactResult("图片", "已保存", path=str(path))


def _write_bytes_atomic(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _interpolate_optional(source_x, source_y, target_x):
    if len(source_x) == 0 or len(source_y) != len(source_x) or not target_x:
        return []
    return np.interp(
        np.asarray(target_x, dtype=np.float64),
        np.asarray(source_x, dtype=np.float64),
        np.asarray(source_y, dtype=np.float64),
    ).tolist()


def _plain_mapping(mapping):
    return {str(key): _plain_value(value) for key, value in dict(mapping).items()}


def _plain_value(value):
    if isinstance(value, np.ndarray):
        return [_plain_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if isinstance(value, dict):
        return _plain_mapping(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _array_value(value):
    if value is None:
        return np.array([], dtype=np.float64)
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _emit_log(log_queue, level, event, request, **fields):
    if log_queue is None:
        return
    record = {
        "level": level,
        "event": event,
        "task_id": request.task_id,
        "condition_key": request.condition_key,
        "wav_path": request.wav_path,
        "source": request.source,
        **fields,
    }
    try:
        log_queue.put(record)
    except (OSError, ValueError):
        pass
