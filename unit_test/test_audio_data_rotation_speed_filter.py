import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QComboBox, QGroupBox, QMessageBox

from ui.custom_ui_widget.audio_data_manage_dialog import (
    AudioDataManageDialog,
    FilterAudioDialog,
    extract_audio_file_name,
    extract_rotation_speed,
)


class _AudioDataFilterState:
    pass


def _audio_row(audio_id, file_path, product_model="S004-1", sample_rate=44100, label="OK"):
    return (
        audio_id,
        file_path,
        product_model,
        sample_rate,
        "2026-07-31",
        label,
        None,
        None,
    )


def _find_rotation_speed_combobox(dialog):
    speed_group = next(
        group
        for group in dialog.findChildren(QGroupBox)
        if group.title() == "转速"
    )
    return speed_group.findChild(QComboBox)


def test_extract_audio_file_name_supports_windows_and_posix_paths():
    file_name = "2026-08-05-09-39-18_S004-1_6000_107c610bb999.wav"

    assert extract_audio_file_name(f"audio_data/stored_data/S004-1/not_labeled/{file_name}") == file_name
    assert extract_audio_file_name(rf"audio_data\stored_data\S004-1\not_labeled\{file_name}") == file_name


def test_extract_rotation_speed_supports_legacy_and_product_condition_names():
    assert extract_rotation_speed("S004-1_2026-07-31_6000rpm.wav") == 6000
    assert extract_rotation_speed(r"C:\audio\S004-1_7000_RPM.wav") == 7000
    assert extract_rotation_speed(
        "2026-08-04-14-32-15_BCSN240727018_S004-1_6000_rpm_107C610BB999.wav"
    ) == 6000
    assert extract_rotation_speed(
        r"audio_data\stored_data\S004-1\not_labeled\2026-08-05-09-39-18_S004-1_6000_107c610bb999.wav"
    ) == 6000
    assert extract_rotation_speed("S004-1_2026-07-31_107c610bb999.wav") is None
    assert extract_rotation_speed("S004-1_2026-07-31_6000.wav") is None


def test_rotation_speed_filter_matches_only_corresponding_wav_files():
    manager = _AudioDataFilterState()
    manager.all_audio_data = [
        _audio_row("1", "audio_data/stored_data/OK/S004-1_6000rpm.wav"),
        _audio_row("2", "audio_data/stored_data/NG/S004-1_7000rpm.wav", label="NG"),
        _audio_row("3", "audio_data/stored_data/not_labeled/S004-1_2026-07-31.wav", label="not_labeled"),
    ]

    AudioDataManageDialog.filter_audio_data_at_filter_config(
        manager,
        {"select_rotation_speed": 7000},
    )

    assert [item[0] for item in manager.filter_audio_data] == ["2"]


def test_rotation_speed_filter_matches_new_product_condition_file_names():
    manager = _AudioDataFilterState()
    manager.all_audio_data = [
        _audio_row(
            "1",
            r"audio_data\stored_data\S004-1\not_labeled\2026-08-05-09-39-18_S004-1_6000_107c610bb999.wav",
            label="not_labeled",
        ),
        _audio_row(
            "2",
            r"audio_data\stored_data\S004-1\not_labeled\2026-08-05-09-41-49_S004-1_7000_107c610bb999.wav",
            label="not_labeled",
        ),
    ]

    AudioDataManageDialog.filter_audio_data_at_filter_config(
        manager,
        {"select_rotation_speed": 6000},
    )

    assert [item[0] for item in manager.filter_audio_data] == ["1"]


def test_rotation_speed_filter_combines_with_existing_filters():
    manager = _AudioDataFilterState()
    manager.all_audio_data = [
        _audio_row("1", "audio_data/stored_data/OK/S004-1_6000rpm.wav"),
        _audio_row("2", "audio_data/stored_data/NG/S004-1_6000rpm.wav", label="NG"),
        _audio_row("3", "audio_data/stored_data/OK/S004-1_7000rpm.wav"),
    ]

    AudioDataManageDialog.filter_audio_data_at_filter_config(
        manager,
        {
            "select_rotation_speed": 6000,
            "select_labels": ["OK"],
        },
    )

    assert [item[0] for item in manager.filter_audio_data] == ["1"]


def test_filter_dialog_lists_available_rotation_speeds():
    app = QApplication.instance() or QApplication([])
    dialog = FilterAudioDialog(
        {"S004-1"},
        {"2026-07-31"},
        filter_config={},
        rotation_speed_set={7000, 6000},
    )

    speed_combobox = _find_rotation_speed_combobox(dialog)

    assert speed_combobox.isEditable()
    assert [speed_combobox.itemText(index) for index in range(speed_combobox.count())] == [
        "ALL",
        "6000",
        "7000",
    ]
    app.processEvents()


def test_filter_dialog_accepts_edited_rotation_speed():
    app = QApplication.instance() or QApplication([])
    dialog = FilterAudioDialog(
        {"S004-1"},
        {"2026-07-31"},
        filter_config={},
        rotation_speed_set={6000, 7000},
    )

    speed_combobox = _find_rotation_speed_combobox(dialog)
    speed_combobox.setEditText("7500")
    dialog.on_click_ok_btn()

    assert dialog.filter_config["select_rotation_speed"] == 7500
    app.processEvents()


def test_rotation_speed_combobox_matches_date_combobox_width():
    app = QApplication.instance() or QApplication([])
    dialog = FilterAudioDialog(
        {"S004-1"},
        {"2026-07-31"},
        filter_config={},
        rotation_speed_set=set(),
    )
    dialog.show()

    speed_combobox = _find_rotation_speed_combobox(dialog)
    date_group = next(
        group
        for group in dialog.findChildren(QGroupBox)
        if group.title() == "日期"
    )
    date_combobox = date_group.findChild(QComboBox)
    speed_editor = speed_combobox.lineEdit()
    speed_combobox.setEditText("6000")
    speed_editor.setCursorPosition(len(speed_editor.text()))
    app.processEvents()

    assert speed_editor.text() == "6000"
    assert speed_combobox.width() == date_combobox.width()


def test_filter_dialog_rejects_invalid_edited_rotation_speed(monkeypatch):
    app = QApplication.instance() or QApplication([])
    dialog = FilterAudioDialog(
        {"S004-1"},
        {"2026-07-31"},
        filter_config={},
        rotation_speed_set={6000, 7000},
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, _text: None,
    )

    speed_combobox = _find_rotation_speed_combobox(dialog)
    speed_combobox.setEditText("高速")
    dialog.on_click_ok_btn()

    assert dialog.is_clicked_ok == 0
    assert "select_rotation_speed" not in dialog.filter_config
    app.processEvents()
