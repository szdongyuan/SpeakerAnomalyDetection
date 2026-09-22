import csv
from datetime import datetime
from pathlib import Path
import sqlite3

import pytest

from base.analysis_report_source import (
    AnalysisItemIdentity,
    ReportCandidate,
    catalog_analysis_items,
    default_report_path,
    filter_candidates,
    scan_project,
)


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _create_database(path, wav_path, label="OK"):
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE audio_data_table ("
            "file_path TEXT UNIQUE, product_model TEXT, "
            "record_date TEXT, labels TEXT)"
        )
        connection.execute(
            "INSERT INTO audio_data_table VALUES (?, ?, ?, ?)",
            (str(wav_path).replace("\\", "/"), "M1", "2026-09-02", label),
        )


def _build_project(tmp_path, *, wav_name=None):
    project = tmp_path / "ProjectA"
    stem = wav_name or "ProjectA_M1_S001_PORT-A_R0002_0.3_20260902-080000-123"
    wav_path = project / "M1" / "S001" / "wav" / f"{stem}.wav"
    wav_path.parent.mkdir(parents=True)
    wav_path.write_bytes(b"RIFF-test")

    csv_directory = project / "M1" / "S001" / "csv" / stem
    image_directory = project / "M1" / "S001" / "images" / stem
    _write_csv(
        csv_directory / "custom_spl_总体声压级.csv",
        ("通道", "总体声压级dB", "总体下限dB", "总体上限dB", "result"),
        (("CH1(前)", "72.5", "60", "80", "OK"),),
    )
    _write_csv(
        csv_directory / "custom_spl_实时声压级.csv",
        ("X轴时间秒", "CH1(前)_SPL_Y轴dB", "下限_Y轴dB", "上限_Y轴dB"),
        (("0", "72.5", "60", "80"),),
    )
    image_directory.mkdir(parents=True)
    (image_directory / "频谱项_CH1(前).png").write_bytes(b"png")
    (image_directory / "频谱项_CH2(后).png").write_bytes(b"png")
    return project, wav_path


def test_scan_supports_audio_directory_and_legacy_wavs_without_raw_csv_pollution(tmp_path):
    project, legacy_wav = _build_project(tmp_path)
    sample = legacy_wav.parent.parent
    new_wav = sample / "audio" / "wav" / legacy_wav.name
    new_wav.parent.mkdir(parents=True)
    new_wav.write_bytes(legacy_wav.read_bytes())
    _write_csv(
        sample / "audio" / "raw_csv" / f"{new_wav.stem}.csv",
        ("sample_index", "time_s", "CH99"),
        (("0", "0.000000000", "0.1"),),
    )
    database = tmp_path / "audio.db"
    _create_database(database, new_wav)

    index = scan_project(str(project), database_path=str(database))

    assert {item.wav_path for item in index.candidates} == {
        str(new_wav.resolve()), str(legacy_wav.resolve())
    }
    new_candidate = next(item for item in index.candidates if item.wav_path == str(new_wav.resolve()))
    assert new_candidate.result_text == "OK"
    assert new_candidate.channel_labels == (("CH1", "前"), ("CH2", "后"))
    assert AnalysisItemIdentity("custom_spl", "SPL") in {
        item.identity for item in new_candidate.analysis_items
    }


def test_default_report_path_uses_application_audio_data_directory(
    tmp_path,
    monkeypatch,
):
    project = tmp_path / "stored_data" / "ProjectA"
    project.mkdir(parents=True)
    application_root = tmp_path / "application"
    monkeypatch.setattr(
        "base.analysis_report_source.DEFAULT_DIR",
        str(application_root),
    )

    suggested = Path(
        default_report_path(
            str(project),
            now=datetime(2026, 9, 4, 14, 30, 0),
        )
    )

    assert suggested.parent == application_root / "audio_data"
    assert suggested.name == "ProjectA_声学测试报告_20260904-143000.pdf"


def test_scan_project_parses_wav_and_joins_label_by_exact_path(tmp_path):
    project, wav_path = _build_project(tmp_path)
    database = tmp_path / "audio.db"
    _create_database(database, wav_path)
    index = scan_project(str(project), database_path=str(database))

    assert len(index.candidates) == 1
    candidate = index.candidates[0]
    assert candidate.model == "M1"
    assert candidate.sample == "S001"
    assert candidate.port == "PORT-A"
    assert candidate.condition == "0.3"
    assert candidate.round_text == "R0002"
    assert candidate.recorded_at_text == "2026-09-02 08:00:00"
    assert candidate.result_text == "OK"
    assert candidate.channel_labels == (("CH1", "前"), ("CH2", "后"))
    assert candidate.channel_mapping_source == "分析结果文件"
    identities = {item.identity for item in candidate.analysis_items}
    assert AnalysisItemIdentity("custom_spl", "SPL") in identities
    assert AnalysisItemIdentity("频谱项", "Spec") in identities
    spec_item = next(
        item
        for item in candidate.analysis_items
        if item.identity == AnalysisItemIdentity("频谱项", "Spec")
    )
    assert tuple(channel for channel, _path in spec_item.image_files) == (1, 2)
    options = {option.identity: option for option in catalog_analysis_items([candidate])}
    assert options[AnalysisItemIdentity("custom_spl", "SPL")].has_scalar_values
    assert not options[AnalysisItemIdentity("频谱项", "Spec")].has_scalar_values


def test_scan_project_parses_second_precision_name_with_sn(tmp_path):
    project, _wav_path = _build_project(
        tmp_path,
        wav_name=(
            "ProjectA_M1_S001_PORT-A_R0002_0.3_"
            "20260902-080000_SN-SN123456"
        ),
    )

    candidate = scan_project(str(project)).candidates[0]

    assert candidate.port == "PORT-A"
    assert candidate.condition == "0.3"
    assert candidate.recorded_at_text == "2026-09-02 08:00:00"
    assert not any("文件名" in issue for issue in candidate.issues)


def test_filter_and_catalog_keep_same_type_items_separate(tmp_path):
    project, wav_path = _build_project(tmp_path)
    stem = wav_path.stem
    csv_directory = project / "M1" / "S001" / "csv" / stem
    _write_csv(
        csv_directory / "second_spl_总体声压级.csv",
        ("通道", "总体声压级dB", "总体下限dB", "总体上限dB", "result"),
        (("CH1", "70", "60", "80", "OK"),),
    )
    database = tmp_path / "audio.db"
    _create_database(database, wav_path, label="NG")

    candidates = scan_project(str(project), database_path=str(database)).candidates
    matched = filter_candidates(
        candidates,
        {
            "models": ["M1"],
            "ports": ["PORT-A"],
            "conditions": ["0.3"],
            "rounds": ["R0002"],
            "sample_numbers": ["S001"],
            "result": "NG",
        },
    )
    options = catalog_analysis_items(matched)

    assert matched == list(candidates)
    assert {option.identity for option in options} >= {
        AnalysisItemIdentity("custom_spl", "SPL"),
        AnalysisItemIdentity("second_spl", "SPL"),
    }
    assert all(option.coverage_text == "1/1" for option in options)
    assert filter_candidates(candidates, {"models": []}) == []


def test_scan_project_ignores_legacy_mapping_csv_when_results_have_labels(
    tmp_path,
):
    project, wav_path = _build_project(tmp_path)
    mapping_path = (
        project / "M1" / "S001" / "csv" / wav_path.stem / "channel_mapping.csv"
    )
    _write_csv(
        mapping_path,
        ("physical_channel", "label"),
        (("CH1", "旧前"), ("CH2", "旧后")),
    )
    database = tmp_path / "audio.db"
    _create_database(database, wav_path)

    candidate = scan_project(str(project), database_path=str(database)).candidates[0]

    assert candidate.channel_labels == (("CH1", "前"), ("CH2", "后"))
    assert candidate.channel_mapping_source == "分析结果文件"


def test_scan_project_keeps_unparseable_wav_visible_and_uses_saved_result_labels(
    tmp_path,
):
    project, wav_path = _build_project(tmp_path, wav_name="legacy-name")
    database = tmp_path / "audio.db"
    _create_database(database, wav_path, label="")
    candidate = scan_project(str(project), database_path=str(database)).candidates[0]

    assert candidate.wav_path == str(wav_path.resolve())
    assert candidate.port == ""
    assert candidate.result_text == "无判定结果"
    assert candidate.channel_labels == (("CH1", "前"), ("CH2", "后"))
    assert candidate.channel_mapping_source == "分析结果文件"
    assert any("文件名" in issue for issue in candidate.issues)
    assert not any("通道映射" in issue for issue in candidate.issues)
    assert filter_candidates(
        [candidate],
        {
            "models": ["M1"],
            "ports": ["—"],
            "conditions": ["—"],
            "rounds": ["—"],
        },
    ) == [candidate]


@pytest.mark.parametrize("label,status,expected", [
    ("OK", "matched", "OK"),
    (" ng ", "matched", "NG"),
    ("", "matched", "无判定结果"),
    ("not_labeled", "matched", "无判定结果"),
    ("", "not_found", "无判定结果"),
    ("", "unavailable", "无判定结果"),
])
def test_candidate_result_groups_missing_results_without_mutating_source(label, status, expected):
    candidate = ReportCandidate("demo.wav", "项目", "型号", "样本",
                                label=label, database_status=status)
    assert candidate.result_text == expected
    assert candidate.label == label
    assert candidate.database_status == status
    assert candidate.issues == ()
    assert candidate.data_status == "完整"


def test_scan_distinguishes_missing_record_from_database_failure(tmp_path):
    project, _wav_path = _build_project(tmp_path)
    database = tmp_path / "audio.db"
    failed = scan_project(str(project), database_path=str(database))
    assert failed.warnings == ("数据库文件不存在",)
    assert failed.candidates[0].database_status == "unavailable"
    assert failed.candidates[0].result_text == "无判定结果"

    _create_database(database, tmp_path / "other.wav")
    missing = scan_project(str(project), database_path=str(database))
    assert not missing.warnings
    assert missing.candidates[0].database_status == "not_found"
    assert missing.candidates[0].result_text == "无判定结果"
    assert "数据库中未找到该 WAV 的判定记录" in missing.candidates[0].issues


def test_scan_project_database_access_does_not_modify_rows(tmp_path):
    project, wav_path = _build_project(tmp_path)
    database = tmp_path / "audio.db"
    _create_database(database, wav_path)

    scan_project(str(project), database_path=str(database))

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM audio_data_table").fetchone()[0] == 1


def test_scan_project_honors_cancellation(tmp_path):
    project, _wav_path = _build_project(tmp_path)

    try:
        scan_project(str(project), cancel_requested=lambda: True)
    except InterruptedError as error:
        assert "取消" in str(error)
    else:
        raise AssertionError("scan should stop when cancellation is requested")
