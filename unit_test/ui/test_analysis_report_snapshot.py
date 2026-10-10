from base.analysis_process_protocol import (
    AnalysisArtifactResult,
    AnalysisInstanceResult,
    AnalysisTaskResult,
)

from ui.sequence.analysis_report_snapshot import (
    build_analysis_report_items_from_task_result,
)


def test_process_result_report_snapshot_reads_saved_channel_image(tmp_path):
    image_path = tmp_path / "CH1.png"
    image_path.write_bytes(b"png-data")
    instance = AnalysisInstanceResult(
        task_id="task-1",
        config_key="声压级",
        runtime_key="声压级--通道1",
        analysis_type="SPL",
        raw_channel=0,
        source_wav_column=0,
        execution_status="分析完成",
        contributes_to_final=True,
        judgement="OK",
        metrics={
            "overall_spl": 72.5,
            "overall_lower_limit": 60.0,
            "overall_upper_limit": 80.0,
            "unit": "dB",
        },
        display_payload={},
        artifacts=(
            AnalysisArtifactResult("图片", "已保存", path=str(image_path)),
        ),
    )
    result = AnalysisTaskResult(
        task_id="task-1",
        condition_key="0.3",
        wav_path=str((tmp_path / "record.wav").resolve()),
        source="自动分析",
        execution_status="分析完成",
        judgement_status="已判定",
        final_judgement="OK",
        instance_results=(instance,),
    )

    items = build_analysis_report_items_from_task_result(
        result,
        {
            "声压级": {
                "type": "SPL",
                "limit_checked": True,
                "limit_metric": "overall_spl",
            }
        },
    )

    assert items[0]["status"] == "OK"
    assert items[0]["measurement"] == "72.5"
    assert items[0]["lower_limit"] == "60"
    assert items[0]["upper_limit"] == "80"
    assert items[0]["images"][0]["png_data"] == b"png-data"


def test_process_result_report_snapshot_records_artifact_save_failure(tmp_path):
    instance = AnalysisInstanceResult(
        task_id="task-2",
        config_key="声压级",
        runtime_key="声压级--通道1",
        analysis_type="SPL",
        raw_channel=0,
        source_wav_column=0,
        execution_status="分析完成",
        contributes_to_final=True,
        judgement="OK",
        metrics={"overall_spl": 72.5, "unit": "dB"},
        display_payload={},
        artifacts=(
            AnalysisArtifactResult(
                "CSV:实时声压级",
                "保存失败",
                error_message="磁盘空间不足",
            ),
        ),
    )
    result = AnalysisTaskResult(
        task_id="task-2",
        condition_key="0.3",
        wav_path=str((tmp_path / "record.wav").resolve()),
        source="自动分析",
        execution_status="分析完成",
        judgement_status="已判定",
        final_judgement="OK",
        instance_results=(instance,),
    )

    items = build_analysis_report_items_from_task_result(
        result,
        {"声压级": {"type": "SPL", "limit_checked": False}},
    )

    assert items[0]["status"] == "OK"
    assert "CSV:实时声压级保存失败：磁盘空间不足" in items[0]["error"]
