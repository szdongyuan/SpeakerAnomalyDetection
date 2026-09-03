import multiprocessing
import pickle

import pytest

from base.analysis_process_protocol import (
    AnalysisArtifactResult,
    AnalysisChannelMapping,
    AnalysisInstanceRequest,
    AnalysisInstanceResult,
    AnalysisTaskRequest,
    AnalysisWorkerFailure,
    build_runtime_key,
    parse_runtime_key,
)
from base.analysis_result_summary import (
    failed_analysis_task_result,
    summarize_analysis_task,
)


def _instance_request(raw_channel=0, column=0, analysis_type="SPL"):
    config_key = f"{analysis_type}-item"
    return AnalysisInstanceRequest(
        config_key=config_key,
        runtime_key=build_runtime_key(config_key, raw_channel),
        analysis_type=analysis_type,
        raw_channel=raw_channel,
        source_wav_column=column,
        v2pa_factor=1.0,
        parameters={"limit_checked": True},
    )


def _task(tmp_path, *, source="自动分析", instances=None):
    wav_path = tmp_path / "selected.wav"
    wav_path.write_bytes(b"RIFF")
    return AnalysisTaskRequest(
        task_id="task-001",
        condition_key="0.3",
        wav_path=str(wav_path.resolve()),
        source=source,
        channel_mapping=(AnalysisChannelMapping(0, 0),),
        sequence_config_snapshot={"acq": {"mode": "RECORD_ONLY"}},
        analysis_config_snapshot={"display_sequence": ["SPL-item"]},
        storage_snapshot={},
        instances=instances or (_instance_request(),),
    )


def _result(
    tmp_path,
    *,
    analysis_type="SPL",
    execution_status="分析完成",
    contributes=True,
    judgement="OK",
    artifact_status="已保存",
):
    artifact = AnalysisArtifactResult(
        "图片",
        artifact_status,
        str((tmp_path / "CH1.png").resolve()) if artifact_status == "已保存" else "",
        "磁盘已满" if artifact_status == "保存失败" else "",
    )
    config_key = f"{analysis_type}-item"
    return AnalysisInstanceResult(
        task_id="task-001",
        config_key=config_key,
        runtime_key=build_runtime_key(config_key, 0),
        analysis_type=analysis_type,
        raw_channel=0,
        source_wav_column=0,
        execution_status=execution_status,
        contributes_to_final=contributes,
        judgement=judgement,
        metrics={"value": 1.0},
        display_payload={"kind": "curve", "x": [0.0], "y": [1.0]},
        artifacts=(artifact,),
        error_message="计算失败" if execution_status == "分析失败" else "",
    )


def _echo(value, queue):
    queue.put(value)


def test_runtime_key_round_trips_physical_channel_identity():
    keys = [build_runtime_key("声压级", channel) for channel in range(5)]
    assert keys == [f"声压级--通道{number}" for number in range(1, 6)]
    assert [parse_runtime_key(key) for key in keys] == [
        ("声压级", channel) for channel in range(5)
    ]


def test_request_is_frozen_picklable_and_crosses_spawn(tmp_path):
    request = _task(tmp_path)
    assert pickle.loads(pickle.dumps(request)) == request

    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_echo, args=(request, queue))
    process.start()
    restored = queue.get(timeout=15)
    process.join(timeout=15)
    try:
        assert process.exitcode == 0
        assert restored == request
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        queue.close()
        queue.join_thread()


def test_display_only_failure_makes_result_incomplete(tmp_path):
    request = _task(tmp_path)
    judged = _result(tmp_path)
    display_failure = _result(
        tmp_path,
        analysis_type="Spec",
        execution_status="分析失败",
        contributes=False,
        judgement=None,
        artifact_status="保存失败",
    )

    result = summarize_analysis_task(request, (judged, display_failure))

    assert result.execution_status == "结果不完整"
    assert result.judgement_status == "不可用"
    assert result.final_judgement is None


def test_only_failed_instances_make_task_fail(tmp_path):
    request = _task(tmp_path)
    failed = _result(
        tmp_path,
        execution_status="分析失败",
        contributes=True,
        judgement=None,
        artifact_status="保存失败",
    )
    result = summarize_analysis_task(request, (failed,))

    assert result.execution_status == "分析失败"
    assert result.judgement_status == "不可用"
    assert result.final_judgement is None


def test_no_participating_items_finishes_without_judgment(tmp_path):
    request = _task(tmp_path)
    spec = _result(
        tmp_path,
        analysis_type="Spec",
        contributes=False,
        judgement=None,
    )
    result = summarize_analysis_task(request, (spec,))

    assert result.execution_status == "分析完成"
    assert result.judgement_status == "未产生判定"
    assert result.final_judgement is None


def test_artifact_failure_does_not_change_successful_judgment(tmp_path):
    request = _task(tmp_path)
    judged = _result(tmp_path, artifact_status="保存失败")
    result = summarize_analysis_task(request, (judged,))

    assert result.execution_status == "分析完成"
    assert result.final_judgement == "OK"


def test_task_level_worker_failure_is_reported_as_failed(tmp_path):
    request = _task(tmp_path)
    failure = AnalysisWorkerFailure(
        request.task_id,
        "子进程退出",
        "WorkerExit",
        "子进程异常退出",
    )

    result = failed_analysis_task_result(request, failure)

    assert result.execution_status == "分析失败"
    assert result.judgement_status == "不可用"
    assert result.final_judgement is None


def test_task_level_failure_after_a_success_is_result_incomplete(tmp_path):
    request = _task(tmp_path)
    failure = AnalysisWorkerFailure(
        request.task_id,
        "子进程退出",
        "WorkerExit",
        "子进程异常退出",
    )

    result = failed_analysis_task_result(request, failure, (_result(tmp_path),))

    assert result.execution_status == "结果不完整"
    assert result.judgement_status == "不可用"
    assert result.final_judgement is None


@pytest.mark.parametrize("analysis_type", ["LOUD", "LP", "RSC"])
def test_first_release_rejects_types_outside_ui_scope(analysis_type):
    with pytest.raises(ValueError, match="analysis_type"):
        _instance_request(analysis_type=analysis_type)
