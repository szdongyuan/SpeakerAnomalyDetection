from datetime import datetime
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from base.analysis_artifact_paths import AnalysisStorageContext
from base.analysis_service import AnalysisProcessService
from base.analysis_worker import _extract_output
from base.wav_calibration_metadata import append_wav_calibration_metadata
from ui.sequence.analysis_task_builder import build_analysis_task_request


def test_manual_fba_output_preserves_discrete_band_plot_semantics():
    output = _extract_output(
        SimpleNamespace(analysis_type="FBA"),
        {
            "metrics": {"weighting": "A"},
            "curve": {
                "x": np.asarray([100.0, 1000.0, 10_000.0]),
                "y": np.asarray([51.0, 62.0, 48.0]),
                "lower": np.asarray([40.0, 40.0, 40.0]),
                "upper": np.asarray([70.0, 70.0, 70.0]),
            },
            "plot": {
                "kind": "bar",
                "labels": ["100", "1k", "10k"],
                "out_mask": np.asarray([False, False, True]),
                "x_label": "Frequency Band",
                "y_label": "Band SPL [dB(A)]",
            },
        },
        "NG",
        "手动查看",
        b"",
    )

    assert output["display_payload"] == {
        "kind": "bar",
        "x": [100.0, 1000.0, 10_000.0],
        "y": [51.0, 62.0, 48.0],
        "labels": ["100", "1k", "10k"],
        "lower": [40.0, 40.0, 40.0],
        "upper": [70.0, 70.0, 70.0],
        "out_mask": [False, False, True],
        "x_label": "Frequency Band",
        "y_label": "Band SPL [dB(A)]",
    }


def test_spawn_worker_runs_real_spl_and_writes_atomic_artifacts(tmp_path):
    wav_path = tmp_path / "source.wav"
    time_axis = np.arange(12_000, dtype=np.float32) / 48_000.0
    audio = (0.01 * np.sin(2.0 * np.pi * 1000.0 * time_axis)).reshape(-1, 1)
    sf.write(wav_path, audio, 48_000, subtype="FLOAT")
    assert append_wav_calibration_metadata(
        wav_path,
        {
            "recorded_channels": [
                {
                    "wav_channel_index": 0,
                    "physical_input_channel": 0,
                    "v2pa_factor": 1.0,
                    "standard_spl": 94.0,
                    "calibrated": True,
                }
            ]
        },
    )
    storage = AnalysisStorageContext(
        str(tmp_path.resolve()),
        "项目",
        "型号",
        "样本",
        1,
        "端口",
        "0.3",
        datetime(2026, 9, 1, 8, 0, 0),
    )
    request = build_analysis_task_request(
        task_id="real-spl-task",
        condition_key="0.3",
        wav_path=str(wav_path),
        source="自动分析",
        sequence_config=[{"seq1": {"acq": {"mode": "RECORD_ONLY"}}}],
        analysis_config={
            "auto_analysis": True,
            "display_sequence": ["声压级"],
            "声压级": {
                "type": "SPL",
                "analysis_channels": [0],
                "limit_checked": False,
                "show_overall_spl": True,
                "weighting": "Z",
            },
        },
        storage_snapshot={
            **storage.to_metadata(),
            "channel_labels": {"CH1": "前"},
        },
    )

    service = AnalysisProcessService()
    service.start(request)
    deadline = time.monotonic() + 30
    terminal = None
    logs = []
    while time.monotonic() < deadline and service.active:
        events, new_logs = service.poll()
        logs.extend(new_logs)
        for kind, payload in events:
            if kind in {"result", "failure"}:
                terminal = (kind, payload)
        time.sleep(0.02)

    if service.active:
        service._process.terminate()
        service._process.join(timeout=5)
        service.poll()
    assert terminal is not None, [record.get("event") for record in logs]
    assert terminal[0] == "result"
    result = terminal[1]
    assert result.execution_status == "分析完成"
    assert result.judgement_status == "未产生判定"
    artifacts = [
        artifact
        for instance in result.instance_results
        for artifact in instance.artifacts
    ]
    assert sorted(artifact.status for artifact in artifacts) == [
        "已保存",
        "已保存",
        "已保存",
    ]
    assert all(artifact.path and __import__("os").path.isfile(artifact.path) for artifact in artifacts)
    assert "声压级_CH1(前).png" in {
        Path(artifact.path).name for artifact in artifacts
    }
    wav_logs = [record for record in logs if record.get("event") == "analysis_wav_loaded"]
    assert len(wav_logs) == 1
    assert wav_logs[0]["duration_seconds"] >= 0.0
    instance_logs = [
        record for record in logs if record.get("event") == "analysis_instance_finished"
    ]
    assert instance_logs[0]["config_key"] == "声压级"
    assert instance_logs[0]["analysis_type"] == "SPL"
    item_logs = [record for record in logs if record.get("event") == "analysis_item_finished"]
    assert len(item_logs) == 1
    assert item_logs[0]["config_key"] == "声压级"
    assert item_logs[0]["analysis_type"] == "SPL"
    assert item_logs[0]["channel_count"] == 1
    assert item_logs[0]["duration_seconds"] >= instance_logs[0]["duration_seconds"]
