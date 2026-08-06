import os
from pathlib import Path

import pytest

from base.file_ops import FileOps
from base import play_and_record
from consts import model_consts


def _set_recording_paths(monkeypatch, root):
    monkeypatch.setattr(model_consts, "STORED_RECORDED_PATH", str(root))
    monkeypatch.setattr(model_consts, "STORED_RECORDED_UNLABELED_PATH", str(root / "not_labeled"))
    monkeypatch.setattr(model_consts, "STORED_RECORDED_OK_PATH", str(root / "OK"))
    monkeypatch.setattr(model_consts, "STORED_RECORDED_NG_PATH", str(root / "NG"))


def test_product_condition_file_name_uses_confirmed_field_order():
    file_name = FileOps.build_recorded_file_name(
        "S004-1",
        "2026-08-04-14-32-15",
        "107C610BB999",
        "SN240727018",
        "_6000_rpm",
        True,
    )

    assert file_name == "2026-08-04-14-32-15_BCSN240727018_S004-1_6000_107C610BB999.wav"


def test_product_condition_file_name_omits_empty_barcode():
    file_name = FileOps.build_recorded_file_name(
        "S004-1",
        "2026-08-04-14-32-15",
        "107C610BB999",
        "",
        "_7000_rpm",
        True,
    )

    assert file_name == "2026-08-04-14-32-15_S004-1_7000_107C610BB999.wav"


def test_product_condition_file_name_sanitizes_path_components():
    file_name = FileOps.build_recorded_file_name(
        "Motor/A:01",
        "2026-08-04-14-32-15",
        "107C610BB999",
        "SN/001",
        "_6000 rpm",
        True,
    )

    assert file_name == "2026-08-04-14-32-15_BCSN_001_Motor_A_01_6000_107C610BB999.wav"


@pytest.mark.parametrize(
    "condition_name",
    ["6000", "6000rpm", "6000 rpm", "6000 RPM", "_6000_rpm"],
)
def test_product_condition_numeric_speed_is_canonicalized(condition_name):
    file_name = FileOps.build_recorded_file_name(
        "S004-1",
        "2026-08-04-14-32-15",
        "107C610BB999",
        "",
        condition_name,
        True,
    )

    assert file_name == "2026-08-04-14-32-15_S004-1_6000_107C610BB999.wav"


def test_legacy_file_name_keeps_existing_field_order():
    file_name = FileOps.build_recorded_file_name(
        "S004-1",
        "2026-08-04-14-32-15",
        "107C610BB999",
        "SN240727018",
        "_forward",
        False,
    )

    assert file_name == "S004-1_2026-08-04-14-32-15_107C610BB999_BCSN240727018_forward.wav"


def test_product_condition_recording_dir_uses_product_model_folder(tmp_path, monkeypatch):
    root = tmp_path / "stored_data"
    _set_recording_paths(monkeypatch, root)

    recording_dir = FileOps.get_recording_store_dir(
        "S004-1",
        "not_labeled",
        True,
    )

    assert Path(recording_dir) == root / "S004-1" / "not_labeled"


def test_product_condition_recording_dir_uses_unspecified_model_for_empty_model(tmp_path, monkeypatch):
    root = tmp_path / "stored_data"
    _set_recording_paths(monkeypatch, root)

    recording_dir = FileOps.get_recording_store_dir(
        "",
        "not_labeled",
        True,
    )

    assert Path(recording_dir) == root / "unspecified_model" / "not_labeled"
    assert FileOps.resolve_recording_product_model("", True) == "unspecified_model"


def test_product_condition_recording_dir_sanitizes_model_folder(tmp_path, monkeypatch):
    root = tmp_path / "stored_data"
    _set_recording_paths(monkeypatch, root)

    recording_dir = FileOps.get_recording_store_dir("Motor/A:01", "not_labeled", True)

    assert Path(recording_dir) == root / "Motor_A_01" / "not_labeled"


def test_legacy_recording_dir_keeps_global_label_folder(tmp_path, monkeypatch):
    root = tmp_path / "stored_data"
    _set_recording_paths(monkeypatch, root)

    recording_dir = FileOps.get_recording_store_dir("S004-1", "not_labeled", False)

    assert Path(recording_dir) == root / "not_labeled"


def test_product_condition_recording_dir_uses_queue_selected_root(tmp_path):
    recording_dir = FileOps.get_recording_store_dir(
        "S004-1",
        "not_labeled",
        True,
        str(tmp_path),
    )

    assert Path(recording_dir) == tmp_path / "S004-1" / "not_labeled"


def test_get_recorded_info_uses_product_condition_storage_contract(tmp_path, monkeypatch):
    root = tmp_path / "stored_data"
    _set_recording_paths(monkeypatch, root)
    monkeypatch.setattr(play_and_record, "get_mac_address", lambda: "10:7C:61:0B:B9:99")

    recorded_path, recorded_info = play_and_record.get_recorded_info(
        "S004-1",
        "recording-token",
        "SN240727018",
        "not_labeled",
        name_suffix="_6000_rpm",
        use_product_model_dir=True,
    )

    path = Path(recorded_path)
    assert path.parent == root / "S004-1" / "not_labeled"
    assert path.name.endswith("_BCSN240727018_S004-1_6000_107C610BB999.wav")
    assert recorded_info["file_path"] == recorded_path
    assert recorded_info["product_model"] == "S004-1"
    assert recorded_info["record_name_suffix"] == "_6000_rpm"


def test_get_recorded_info_uses_queue_selected_root(tmp_path, monkeypatch):
    monkeypatch.setattr(play_and_record, "get_mac_address", lambda: "10:7C:61:0B:B9:99")

    recorded_path, recorded_info = play_and_record.get_recorded_info(
        "S004-1",
        "recording-token",
        "SN240727018",
        "not_labeled",
        name_suffix="_7000_rpm",
        use_product_model_dir=True,
        recording_root=str(tmp_path),
    )

    assert Path(recorded_path).parent == tmp_path / "S004-1" / "not_labeled"
    assert recorded_info[model_consts.RECORDING_ROOT_CONFIG_KEY] == str(tmp_path)


def test_relabel_product_recording_keeps_product_model_folder(tmp_path, monkeypatch):
    root = tmp_path / "stored_data"
    _set_recording_paths(monkeypatch, root)
    source_dir = root / "S004-1" / "not_labeled"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "sample_6000_rpm.wav"
    source_path.write_bytes(b"RIFF")

    target_path = FileOps.move_wav_to_dir(str(source_path), "OK")

    assert Path(target_path) == root / "S004-1" / "OK" / source_path.name
    assert Path(target_path).is_file()
    assert not source_path.exists()


def test_relabel_empty_recording_path_does_not_create_or_move_anything(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "stored_data"
    _set_recording_paths(monkeypatch, root)

    target_path = FileOps.move_wav_to_dir("", "OK")

    assert target_path == ""
    assert not root.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows paths are case-insensitive")
def test_relabel_root_comparison_ignores_windows_path_case(tmp_path):
    root = tmp_path / "FactoryAudio"
    source_dir = root / "S004-1" / "not_labeled"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "sample.wav"
    source_path.write_bytes(b"RIFF")

    target_path = FileOps.move_wav_to_dir(
        str(source_path),
        "OK",
        str(root).swapcase(),
    )

    expected_path = root / "S004-1" / "OK" / source_path.name
    assert Path(target_path) == expected_path
    assert expected_path.is_file()


def test_relabel_legacy_recording_keeps_global_label_folder(tmp_path, monkeypatch):
    root = tmp_path / "stored_data"
    _set_recording_paths(monkeypatch, root)
    source_dir = root / "not_labeled"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "legacy.wav"
    source_path.write_bytes(b"RIFF")

    target_path = FileOps.move_wav_to_dir(str(source_path), "NG")

    assert Path(target_path) == root / "NG" / source_path.name
    assert Path(target_path).is_file()


def test_relabel_product_recording_uses_queue_selected_root(tmp_path):
    source_dir = tmp_path / "S004-1" / "not_labeled"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "sample_8000_rpm.wav"
    source_path.write_bytes(b"RIFF")

    target_path = FileOps.move_wav_to_dir(
        str(source_path),
        "OK",
        str(tmp_path),
    )

    assert Path(target_path) == tmp_path / "S004-1" / "OK" / source_path.name
    assert Path(target_path).is_file()
    assert not source_path.exists()
