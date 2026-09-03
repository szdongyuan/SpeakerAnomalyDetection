"""Pure aggregation rules for multichannel analysis results."""

from __future__ import annotations

from base.analysis_process_protocol import (
    AnalysisInstanceResult,
    AnalysisTaskResult,
)


def summarize_analysis_task(request, instance_results):
    """Separate execution completeness from judgment participation."""
    results = tuple(instance_results)
    participating = [item for item in results if item.contributes_to_final]
    successful_results = tuple(
        item for item in results if item.execution_status == "分析完成"
    )
    failed_results = tuple(
        item for item in results if item.execution_status != "分析完成"
    )
    if successful_results and failed_results:
        execution_status = "结果不完整"
        judgement_status = "不可用"
        final_judgement = None
    elif failed_results or not results:
        execution_status = "分析失败"
        judgement_status = "不可用"
        final_judgement = None
    elif not participating:
        execution_status = "分析完成"
        judgement_status = "未产生判定"
        final_judgement = None
    else:
        execution_status = "分析完成"
        judgement_status = "已判定"
        final_judgement = (
            "OK"
            if all(item.judgement == "OK" for item in participating)
            else "NG"
        )
    return AnalysisTaskResult(
        task_id=request.task_id,
        condition_key=request.condition_key,
        wav_path=request.wav_path,
        source=request.source,
        execution_status=execution_status,
        judgement_status=judgement_status,
        final_judgement=final_judgement,
        instance_results=results,
    )


def failed_analysis_task_result(request, failure, instance_results=()):
    """Build a task-level failure without manufacturing an OK/NG value."""
    results = tuple(instance_results)
    execution_status = (
        "结果不完整"
        if any(item.execution_status == "分析完成" for item in results)
        else "分析失败"
    )
    return AnalysisTaskResult(
        task_id=request.task_id,
        condition_key=request.condition_key,
        wav_path=request.wav_path,
        source=request.source,
        execution_status=execution_status,
        judgement_status="不可用",
        final_judgement=None,
        instance_results=results,
        error_stage=failure.stage,
        error_type=failure.error_type,
        error_message=failure.message,
    )
