from pathlib import Path

import pytest

from base.audio_record_filter import (
    UNKNOWN,
    filter_audio_records,
    parse_audio_filter_metadata,
)


def recording_path(root="C:/results", project="项目A", sample="003", port="端口1",
                   round_number=2, gear="高速档", layout="audio/wav", suffix=""):
    stem = f"{project}_型号01_{sample}_{port}_R{round_number:04d}_{gear}_20260914-093000{suffix}"
    return f"{root}/{project}/型号01/{sample}/{layout}/{stem}.wav"


def audio_row(identifier, path=None, *, model="型号01", day="2026-09-14", rate=44100, label="OK"):
    return (identifier, path or recording_path(), model, rate, day, label, None, None)


def select(rows, filters):
    metadata = {row[0]: parse_audio_filter_metadata(row[1]) for row in rows}
    return filter_audio_records(rows, metadata, filters)


@pytest.mark.parametrize("layout", ["audio/wav", "wav"])
@pytest.mark.parametrize("suffix", ["", "-123", "_SN-编号_01", "_02", "-123_SN-编号_01_02"])
def test_parse_known_recording_layouts_without_opening_files(layout, suffix, monkeypatch):
    def no_io(*args, **kwargs):
        pytest.fail("Filtering must not read WAVs or enumerate directories")
    monkeypatch.setattr(Path, "open", no_io)
    monkeypatch.setattr(Path, "iterdir", no_io)
    info = parse_audio_filter_metadata(recording_path(layout=layout, suffix=suffix))
    assert (info.project_name, info.sample_number, info.port_name, info.test_round, info.condition_name) == (
        "项目A", "003", "端口1", 2, "高速档",
    )


def test_names_with_underscores_and_numeric_gear_keep_their_meaning():
    info = parse_audio_filter_metadata(recording_path(
        project="项目_A", sample="样本_003", port="USB_C_1", gear="0.30",
    ))
    assert (info.project_name, info.sample_number, info.port_name, info.condition_name) == (
        "项目_A", "样本_003", "USB_C_1", "0.30",
    )


def test_legacy_sample_named_audio_is_not_confused_with_new_layout():
    info = parse_audio_filter_metadata(recording_path(sample="audio", layout="wav"))
    assert info.sample_number == "audio"
    assert info.project_name == "项目A"


def test_project_identity_uses_full_directory_and_normalizes_windows_paths():
    first = parse_audio_filter_metadata(recording_path())
    same = parse_audio_filter_metadata(recording_path().replace("/", "\\"))
    other = parse_audio_filter_metadata(recording_path(root="D:/results"))
    network = parse_audio_filter_metadata(recording_path(root="//server/share"))
    assert first.project_key == same.project_key
    assert first.project_name == other.project_name
    assert first.project_key != other.project_key
    assert network.project_name == "项目A"


def test_relative_path_is_resolved_against_application_root():
    absolute = parse_audio_filter_metadata(recording_path(root="C:/app/results"))
    relative = parse_audio_filter_metadata(recording_path(root="results"), application_root="C:/app")
    assert relative == absolute


@pytest.mark.parametrize("path", [
    "old/OK/recording_6000rpm.wav",
    "", "unstructured.wav", "C:/wrong/audio/wav/a.wav",
    recording_path().replace("项目A_型号01_", "项目B_型号01_"),
    recording_path().replace("20260914", "20261399"),
    recording_path().replace("20260914-093000", "abcdef123456"),
])
def test_unknown_names_are_not_guessed(path):
    info = parse_audio_filter_metadata(path)
    assert info.project_key is None
    assert info.port_name is None
    assert info.condition_name is None


def test_combined_fields_are_exact_and_keep_original_records():
    rows = [audio_row("chosen"), audio_row("port10", recording_path(port="端口10")),
            audio_row("other_round", recording_path(round_number=3)),
            audio_row("other_sample", recording_path(sample="3")),
            audio_row("other_gear", recording_path(gear="低速档"))]
    matched = select(rows, {"select_sample_number": "003", "select_port": "端口1",
                            "select_test_round": 2, "select_condition": "高速档"})
    assert matched == [rows[0]]
    assert matched[0] is rows[0]


def test_project_selection_does_not_merge_same_named_projects():
    rows = [audio_row("one"), audio_row("two", recording_path(root="D:/results"))]
    project = parse_audio_filter_metadata(rows[0][1]).project_key
    assert select(rows, {"select_project": project}) == [rows[0]]


def test_unknown_is_distinct_from_all_and_real_name():
    rows = [audio_row("known"), audio_row("unknown", "old.wav"),
            audio_row("literal", recording_path(gear="未识别"))]
    assert select(rows, {}) == rows
    assert select(rows, {"select_condition": UNKNOWN}) == [rows[1]]
    assert select(rows, {"select_condition": "未识别"}) == [rows[2]]


def test_database_values_only_match_their_own_column():
    rows = [audio_row("OK", model="NG", label="not_labeled"),
            audio_row("wanted", model="型号01", label="OK")]
    assert select(rows, {"select_labels": ["OK"]}) == [rows[1]]
    assert select(rows, {"select_product_model": "OK"}) == []
    assert select(rows, {"select_labels": ["OK", "not_labeled"]}) == rows
    assert select(rows, {"select_sample_rate": [48000]}) == []


def test_single_date_matches_database_column_and_ignores_wav_timestamp():
    rows = [audio_row("before", day="2026-09-12"), audio_row("chosen", day="2026-09-13"),
            audio_row("after", day="2026-09-14"), audio_row("2026-09-13", day="2026-09-15"),
            audio_row("invalid", day="bad date")]
    assert select(rows, {"select_record_date": "2026-09-13"}) == [rows[1]]
    assert select(rows, {"select_record_date": "2026-09-16"}) == []
    assert select(rows, {}) == rows
