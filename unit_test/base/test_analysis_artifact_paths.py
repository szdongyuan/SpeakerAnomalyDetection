from datetime import datetime
from pathlib import Path

import pytest

from base.analysis_artifact_paths import (
    AnalysisStorageContext,
    build_channel_image_path,
    build_csv_path,
    build_pdf_path,
    build_recording_stem,
    build_wav_path,
    format_channel_name,
    next_available_path,
    sanitize_path_component,
    storage_context_from_metadata,
)


def _context(tmp_path):
    return AnalysisStorageContext(
        result_root_directory=str(tmp_path.resolve()),
        project_name="项目A",
        product_model="型号01",
        sample_number="样本001",
        port_name="端口1",
        test_round=7,
        condition_name="0.3",
        recording_started_at=datetime(2026, 9, 1, 8, 9, 10, 123000),
        serial_number="SN123456",
    )


def test_storage_paths_share_one_stem_without_label_directories(tmp_path):
    context = _context(tmp_path)
    stem = build_recording_stem(context)

    assert stem == "项目A_型号01_样本001_端口1_R0007_0.3_20260901-080910_SN-SN123456"

    wav_path = build_wav_path(context)
    image_path = build_channel_image_path(context, stem, "声压级", 2)
    csv_path = build_csv_path(context, stem, "声压级", "实时声压级")

    assert wav_path == (
        tmp_path / "项目A" / "型号01" / "样本001" / "wav" / f"{stem}.wav"
    )
    assert image_path == (
        tmp_path
        / "项目A"
        / "型号01"
        / "样本001"
        / "images"
        / stem
        / "声压级_CH3.png"
    )
    assert csv_path == (
        tmp_path
        / "项目A"
        / "型号01"
        / "样本001"
        / "csv"
        / stem
        / "声压级_实时声压级.csv"
    )
    assert not ({"OK", "NG", "not_labeled"} & set(wav_path.parts))


def test_one_wav_keeps_all_analysis_item_images_in_one_directory(tmp_path):
    context = _context(tmp_path)
    stem = build_recording_stem(context)

    spl_path = build_channel_image_path(
        context,
        stem,
        "声压级",
        0,
        {"CH1": "前"},
    )
    spec_path = build_channel_image_path(context, stem, "频谱分析", 0)

    assert spl_path.parent == spec_path.parent
    assert spl_path.parent.name == stem
    assert spl_path.name == "声压级_CH1(前).png"
    assert spec_path.name == "频谱分析_CH1.png"


def test_channel_name_normalizes_label_keys_and_omits_empty_labels():
    assert format_channel_name(0, {"ch1": " 前 "}) == "CH1(前)"
    assert format_channel_name(1, {"CH2": ""}) == "CH2"


def test_one_wav_keeps_all_analysis_item_csvs_in_one_directory(tmp_path):
    context = _context(tmp_path)
    stem = build_recording_stem(context)

    spl_overall_path = build_csv_path(context, stem, "声压级1", "总体声压级")
    spl_realtime_path = build_csv_path(context, stem, "声压级1", "实时声压级")
    ai_path = build_csv_path(context, stem, "AI1", "模型输出")

    assert spl_overall_path.parent == spl_realtime_path.parent == ai_path.parent
    assert spl_overall_path.parent.name == stem
    assert spl_overall_path.name == "声压级1_总体声压级.csv"
    assert spl_realtime_path.name == "声压级1_实时声压级.csv"
    assert ai_path.name == "AI1_模型输出.csv"


def test_pdf_name_and_directory_follow_report_boundary(tmp_path):
    context = _context(tmp_path)
    path = build_pdf_path(
        context,
        generated_at=datetime(2026, 9, 1, 9, 10, 11, 456000),
    )

    assert path == (
        tmp_path
        / "项目A"
        / "型号01"
        / "pdf"
        / "项目A_型号01_样本001_R0007_20260901-091011-456.pdf"
    )


def test_storage_context_round_trips_plain_metadata(tmp_path):
    context = _context(tmp_path)
    assert storage_context_from_metadata(context.to_metadata()) == context


def test_recording_stem_omits_serial_component_when_sn_is_empty(tmp_path):
    context = AnalysisStorageContext(
        result_root_directory=str(tmp_path.resolve()),
        project_name="项目A",
        product_model="型号01",
        sample_number="样本001",
        port_name="端口1",
        test_round=7,
        condition_name="0.3",
        recording_started_at=datetime(2026, 9, 1, 8, 9, 10, 999000),
    )

    assert build_recording_stem(context) == (
        "项目A_型号01_样本001_端口1_R0007_0.3_20260901-080910"
    )


def test_second_precision_name_uses_numbered_suffix_on_collision(tmp_path):
    wav_path = build_wav_path(_context(tmp_path))
    wav_path.parent.mkdir(parents=True)
    wav_path.write_bytes(b"existing")

    assert next_available_path(wav_path).name == (
        f"{wav_path.stem}_01.wav"
    )


@pytest.mark.parametrize("value", ["CON", "a/b", "x" * 200])
def test_path_components_are_windows_safe_and_bounded(value):
    component = sanitize_path_component(value, max_length=32)
    assert component
    assert len(component) <= 32
    assert not any(character in component for character in '<>:"/\\|?*')
