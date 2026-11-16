import builtins
import os
from pathlib import Path

import pytest

from base.audio_record_filter import (
    AudioFilterMetadata,
    filter_audio_records,
    parse_archive_audio_filter_metadata,
)
from unit_test.base.test_audio_record_filter import audio_row


def current_recording_path(root="C:/results", project="项目A", station="工位1",
                           condition="正转", suffix=""):
    return (f"{root}/{project}/audio/"
            f"2026-11-16_09-30-00-123_{station}_{condition}_M-02_007{suffix}.wav")


def select(rows, filters):
    metadata = {row[0]: parse_archive_audio_filter_metadata(row[1]) for row in rows}
    return filter_audio_records(rows, metadata, filters)


def test_station_and_condition_are_fields_after_timestamp():
    info = parse_archive_audio_filter_metadata(current_recording_path(suffix="_2_SN_008"))
    assert (info.project_name, info.station_name, info.condition_name) == ("项目A", "工位1", "正转")
    assert info.port_name is None
    assert info.sample_number is None
    assert info.test_round is None


@pytest.mark.parametrize("station,condition", [("装配工位", "反转"), ("001", "02")])
@pytest.mark.parametrize("suffix", ["", "_2", "_SN_产品_008", "_2_SN_产品_008"])
def test_names_and_serial_number_suffix_keep_field_positions(station, condition, suffix):
    info = parse_archive_audio_filter_metadata(current_recording_path(
        station=station, condition=condition, suffix=suffix,
    ))
    assert (info.station_name, info.condition_name) == (station, condition)


def test_relative_path_is_resolved_against_explicit_application_root():
    absolute = parse_archive_audio_filter_metadata(current_recording_path(root="C:/app/results"))
    relative = parse_archive_audio_filter_metadata(
        current_recording_path(root="results"), application_root="C:/app",
    )
    assert relative == absolute


def test_project_identity_normalizes_case_slashes_and_dot_segments():
    first = parse_archive_audio_filter_metadata(current_recording_path(root="C:/Results"))
    same = parse_archive_audio_filter_metadata(
        current_recording_path(root="c:/results/unused/..").replace("/", "\\").replace(".wav", ".WAV")
        .replace("\\audio\\", "\\AuDiO\\"),
    )
    assert first == same
    assert first.project_key == "c:\\results\\项目a"


def test_unc_path_preserves_project_directory_identity():
    info = parse_archive_audio_filter_metadata(current_recording_path(root="//server/share"))
    assert info.project_name == "项目A"
    assert info.project_key == "\\\\server\\share\\项目a"
    assert (info.station_name, info.condition_name) == ("工位1", "正转")


def test_same_named_projects_in_different_directories_remain_distinct():
    rows = [audio_row("one", current_recording_path()),
            audio_row("two", current_recording_path(root="D:/results"))]
    first, second = [parse_archive_audio_filter_metadata(row[1]) for row in rows]
    assert first.project_name == second.project_name
    assert first.project_key != second.project_key
    assert select(rows, {"select_project": first.project_key}) == [rows[0]]


@pytest.mark.parametrize("path", [
    None,
    "",
    "  ",
    current_recording_path(station=""),
    current_recording_path(condition=""),
    current_recording_path().replace("2026-11-16_", "_"),
    current_recording_path().replace("09-30-00-123_", "_"),
    "C:/results/项目A/audio/2026-11-16_09-30-00-123_工位1_正转.wav",
    "C:/results/项目A/audio/2026-11-16_09-30-00-123_工位1_正转_.wav",
    current_recording_path().replace("/audio/", "/wav/"),
    current_recording_path().replace(".wav", ".mp3"),
    "C:/audio/2026-11-16_09-30-00-123_工位1_正转_M-02_007.wav",
])
def test_missing_required_path_fields_raise_with_original_path(path):
    with pytest.raises(ValueError) as error:
        parse_archive_audio_filter_metadata(path)
    assert repr(path) in str(error.value)


def test_valid_path_parsing_does_not_open_files_or_enumerate_directories(monkeypatch):
    path = current_recording_path()

    def no_io(*args, **kwargs):
        pytest.fail("Current audio metadata parsing must not access the filesystem")

    monkeypatch.setattr(builtins, "open", no_io)
    monkeypatch.setattr(Path, "open", no_io)
    monkeypatch.setattr(Path, "iterdir", no_io)
    monkeypatch.setattr(os, "listdir", no_io)
    monkeypatch.setattr(os, "scandir", no_io)
    info = parse_archive_audio_filter_metadata(path)
    assert (info.project_name, info.station_name, info.condition_name) == ("项目A", "工位1", "正转")


def test_station_field_is_appended_to_preserve_existing_positional_contract():
    info = AudioFilterMetadata("project-key", "project", "007", 2, "port", "condition")
    assert info.condition_name == "condition"
    assert info.station_name is None


@pytest.fixture
def current_rows():
    return [
        audio_row("first", current_recording_path(), barcode="ABC/123"),
        audio_row("station10", current_recording_path(station="工位10"), barcode="ABC/123"),
        audio_row("reverse", current_recording_path(condition="反转"), barcode="ABC/123"),
        audio_row("other_project", current_recording_path(project="项目B"), barcode="ABC/123"),
        audio_row("other_model", current_recording_path(), model="型号010", barcode="ABC/123"),
        audio_row("other_barcode", current_recording_path(), barcode="ABC/123X"),
        audio_row("other_label", current_recording_path(), label="NG", barcode="ABC/123"),
        audio_row("other_rate", current_recording_path(), rate=48000, barcode="ABC/123"),
        audio_row("other_day", current_recording_path(), day="2026-09-15", barcode="ABC/123"),
        audio_row("last", current_recording_path(suffix="_2_SN_008"), barcode="ABC/123"),
    ]


@pytest.mark.parametrize("filters,excluded", [
    ({"select_station": "工位1"}, "station10"),
    ({"select_condition": "正转"}, "reverse"),
    ({"select_project": "c:\\results\\项目a"}, "other_project"),
    ({"select_product_model": "型号01"}, "other_model"),
    ({"select_barcode": "ABC/123"}, "other_barcode"),
    ({"select_labels": ["OK"]}, "other_label"),
    ({"select_sample_rate": [44100]}, "other_rate"),
    ({"select_record_date": "2026-09-14"}, "other_day"),
])
def test_retained_filters_each_match_only_their_own_exact_value(current_rows, filters, excluded):
    expected = [row for row in current_rows if row[0] != excluded]
    matched = select(current_rows, filters)
    assert matched == expected
    assert all(actual is original for actual, original in zip(matched, expected))


def test_all_filters_combine_with_and_and_preserve_original_row_identity_and_order(current_rows):
    matched = select(current_rows, {
        "select_project": "c:\\results\\项目a",
        "select_station": "工位1",
        "select_condition": "正转",
        "select_product_model": "型号01",
        "select_barcode": "ABC/123",
        "select_labels": ["OK"],
        "select_sample_rate": [44100],
        "select_record_date": "2026-09-14",
    })
    assert matched == [current_rows[0], current_rows[-1]]
    assert matched[0] is current_rows[0]
    assert matched[1] is current_rows[-1]


def test_database_record_date_does_not_use_filename_timestamp(current_rows):
    assert select(current_rows, {"select_record_date": "2026-11-16"}) == []
    assert select(current_rows, {"select_record_date": "2026-09-15"}) == [current_rows[8]]


def test_empty_records_and_zero_match_are_empty_lists():
    assert select([], {"select_station": "工位1"}) == []
    rows = [audio_row("one", current_recording_path()),
            audio_row("two", current_recording_path(station="工位10", condition="反转"))]
    assert select(rows, {"select_station": "工位1", "select_condition": "反转"}) == []
    matched = select(rows, {})
    assert matched == rows
    assert all(actual is original for actual, original in zip(matched, rows))
