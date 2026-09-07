from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from base.wav_calibration_metadata import append_wav_calibration_metadata
from ui.sequence.analysis_task_builder import (
    AnalysisTaskBuildError,
    build_analysis_task_request,
)


def _write_wav(path: Path, physical_channels=(0, 2)):
    audio = np.zeros((480, len(physical_channels)), dtype=np.float32)
    sf.write(path, audio, 48_000, subtype="FLOAT")
    metadata = {
        "recorded_channels": [
            {
                "wav_channel_index": column,
                "physical_input_channel": raw_channel,
                "v2pa_factor": float(column + 1),
                "standard_spl": 94.0,
                "calibrated": True,
            }
            for column, raw_channel in enumerate(physical_channels)
        ]
    }
    assert append_wav_calibration_metadata(path, metadata)


def _analysis_config(channels=(0, 2)):
    return {
        "auto_analysis": True,
        "display_sequence": ["声压级", "频谱", "忽略项"],
        "声压级": {
            "type": "SPL",
            "analysis_channels": list(channels),
            "limit_checked": True,
        },
        "频谱": {
            "type": "Spec",
            "analysis_channels": [channels[0]],
        },
        "忽略项": {"type": "LOUD", "analysis_channels": [channels[0]]},
    }


def test_builder_uses_exact_wav_mapping_and_file_calibration(tmp_path):
    wav_path = tmp_path / "not-derived-from-name.wav"
    _write_wav(wav_path)

    request = build_analysis_task_request(
        task_id="task-1",
        condition_key="0.3",
        wav_path=str(wav_path),
        source="自动分析",
        sequence_config=[{"seq1": {"acq": {"mode": "RECORD_ONLY"}}}],
        analysis_config=_analysis_config(),
        storage_snapshot={"project_name": "项目A"},
    )

    assert request.wav_path == str(wav_path.resolve())
    assert [(item.raw_channel, item.source_wav_column) for item in request.channel_mapping] == [
        (0, 0),
        (2, 1),
    ]
    assert [item.runtime_key for item in request.instances] == [
        "声压级--通道1",
        "声压级--通道3",
        "频谱--通道1",
    ]
    assert [item.v2pa_factor for item in request.instances] == [1.0, 2.0, 1.0]
    assert all(item.calibration_available for item in request.instances)


def test_builder_marks_default_factor_when_wav_has_no_calibration(tmp_path):
    wav_path = tmp_path / "uncalibrated.wav"
    sf.write(
        wav_path,
        np.zeros((480, 1), dtype=np.float32),
        48_000,
        subtype="FLOAT",
    )

    request = build_analysis_task_request(
        task_id="task-uncalibrated",
        condition_key="0.3",
        wav_path=str(wav_path),
        source="手动查看",
        sequence_config=[],
        analysis_config=_analysis_config((0,)),
    )

    assert all(item.v2pa_factor == 1.0 for item in request.instances)
    assert all(not item.calibration_available for item in request.instances)


def test_builder_rejects_configured_channel_missing_from_wav(tmp_path):
    wav_path = tmp_path / "recording.wav"
    _write_wav(wav_path, (0, 2))

    with pytest.raises(AnalysisTaskBuildError, match="CH8"):
        build_analysis_task_request(
            condition_key="0.3",
            wav_path=str(wav_path),
            source="手动查看",
            sequence_config=[],
            analysis_config=_analysis_config((7,)),
        )


def test_builder_rejects_missing_file_and_empty_supported_scope(tmp_path):
    with pytest.raises(AnalysisTaskBuildError, match="不存在"):
        build_analysis_task_request(
            condition_key="0.3",
            wav_path=str(tmp_path / "missing.wav"),
            source="手动查看",
            sequence_config=[],
            analysis_config=_analysis_config(),
        )

    wav_path = tmp_path / "recording.wav"
    _write_wav(wav_path, (0,))
    with pytest.raises(AnalysisTaskBuildError, match="没有可执行"):
        build_analysis_task_request(
            condition_key="0.3",
            wav_path=str(wav_path),
            source="手动查看",
            sequence_config=[],
            analysis_config={
                "display_sequence": ["响度"],
                "响度": {"type": "LOUD"},
            },
        )
