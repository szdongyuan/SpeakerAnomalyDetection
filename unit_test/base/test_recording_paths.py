from pathlib import Path

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

    assert file_name == "2026-08-04-14-32-15_BCSN240727018_S004-1_6000_rpm_107C610BB999.wav"


def test_product_condition_file_name_omits_empty_barcode():
    file_name = FileOps.build_recorded_file_name(
        "S004-1",
        "2026-08-04-14-32-15",
        "107C610BB999",
        "",
        "_7000_rpm",
        True,
    )

    assert file_name == "2026-08-04-14-32-15_S004-1_7000_rpm_107C610BB999.wav"


def test_product_condition_file_name_sanitizes_path_components():
    file_name = FileOps.build_recorded_file_name(
        "Motor/A:01",
        "2026-08-04-14-32-15",
        "107C610BB999",
        "SN/001",
        "_6000 rpm",
        True,
    )

    assert file_name == "2026-08-04-14-32-15_BCSN_001_Motor_A_01_6000_rpm_107C610BB999.wav"


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
    assert path.name.endswith("_BCSN240727018_S004-1_6000_rpm_107C610BB999.wav")
    assert recorded_info["file_path"] == recorded_path
    assert recorded_info["product_model"] == "S004-1"
    assert recorded_info["record_name_suffix"] == "_6000_rpm"


def test_relabel_product_recording_keeps_existing_global_label_folder_logic(tmp_path, monkeypatch):
    root = tmp_path / "stored_data"
    _set_recording_paths(monkeypatch, root)
    source_dir = root / "S004-1" / "not_labeled"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "sample_6000_rpm.wav"
    source_path.write_bytes(b"RIFF")

    target_path = FileOps.move_wav_to_dir(str(source_path), "OK")

    assert Path(target_path) == root / "OK" / source_path.name
    assert Path(target_path).is_file()
    assert not source_path.exists()


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
