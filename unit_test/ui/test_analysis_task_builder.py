from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from base.wav_calibration_metadata import append_wav_calibration_metadata, inspect_wav_calibration_metadata
from unit_test.base.ve3668n_fakes import wav_metadata
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


@pytest.mark.parametrize("sources", [("none", "none"), ("measured", "none"),
                                     ("none", "measured"), ("measured", "measured")])
@pytest.mark.parametrize("source", ["自动分析", "手动查看"])
def test_vk_file_factors_override_other_soundcard_cache_without_rewriting_wav(tmp_path, sources, source):
    wav_path = tmp_path / "existing-v1-voltage.wav"
    audio = np.tile(np.float32([8.25, 2.5]), (480, 1))
    sf.write(wav_path, audio, 44_100, subtype="FLOAT")
    metadata = wav_metadata(sources)
    assert append_wav_calibration_metadata(wav_path, metadata)
    original = wav_path.read_bytes()

    request = build_analysis_task_request(
        condition_key="0.3", wav_path=str(wav_path), source=source,
        sequence_config=[], analysis_config=_analysis_config((7, 1)),
        saved_active_input_channels=[1, 7], fallback_v2pa_factors={7: 99.0, 1: 42.0},
    )

    assert [(item.raw_channel, item.source_wav_column) for item in request.channel_mapping] == [(7, 0), (1, 1)]
    for item in request.instances:
        measured = sources[item.source_wav_column] == "measured"
        assert item.v2pa_factor == (10.0 if measured else 1.0)
        assert item.calibration_available is measured
    assert wav_path.read_bytes() == original
    assert inspect_wav_calibration_metadata(wav_path).metadata == metadata
    np.testing.assert_array_equal(sf.read(wav_path, dtype="float32")[0], audio)


def test_builder_keeps_ordinary_soundcard_cache_fallback(tmp_path):
    wav_path = tmp_path / "ordinary.wav"
    sf.write(wav_path, np.zeros(480, dtype=np.float32), 48_000, subtype="FLOAT")
    request = build_analysis_task_request(
        condition_key="0.3", wav_path=str(wav_path), source="手动查看",
        sequence_config=[], analysis_config=_analysis_config((0,)),
        fallback_v2pa_factors={0: 42.0},
    )
    assert all(item.v2pa_factor == 42.0 and item.calibration_available for item in request.instances)


def test_builder_rejects_damaged_vk_metadata_even_with_cached_factors(tmp_path):
    from unit_test.base.test_wav_calibration_metadata import _append_raw_chunk, _metadata_list_chunk
    import json

    wav_path = tmp_path / "damaged-vk.wav"
    sf.write(wav_path, np.zeros((480, 2), dtype=np.float32), 44_100, subtype="FLOAT")
    metadata = wav_metadata(("none", "none"))
    metadata["recorded_channels"][1]["v2pa_factor"] = 1.0
    _append_raw_chunk(wav_path, _metadata_list_chunk(json.dumps(metadata).encode()))
    with pytest.raises(AnalysisTaskBuildError, match="元数据无效"):
        build_analysis_task_request(
            condition_key="0.3", wav_path=str(wav_path), source="手动查看",
            sequence_config=[], analysis_config=_analysis_config((7, 1)),
            fallback_v2pa_factors={7: 99.0, 1: 42.0},
        )


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
