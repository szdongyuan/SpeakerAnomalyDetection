import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QComboBox, QFileDialog, QInputDialog, QLabel, QMessageBox

from base.load_config import LoadUiConfig
from base.sequence_queue_references import SequenceQueueReferenceScanner
from base.product_test_project_config import ProductTestProjectConfigManager
from consts import ui_style_const
from consts.product_test_project_consts import EXPORT_RAW_AUDIO_CSV_KEY
from ui.product_test_project_config_dialog import (
    ProductTestProjectConfigDialog,
)
from ui.sequence.direction_waveform_panel import DirectionWaveformPanel
from ui.sequence.recent_session_panel import RecentSessionPanel
from ui.sequence.sequence_widget_config_ops import SequenceWidgetConfigOpsMixin


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def make_manager(tmp_path):
    program_dir = tmp_path / "product_test_programs"
    queue_dir = tmp_path / "analysis_sequence_config"
    program_dir.mkdir()
    queue_dir.mkdir()
    return ProductTestProjectConfigManager(
        str(program_dir),
        str(program_dir / "program_registry.json"),
        str(queue_dir / "sequence_config_registry.json"),
    )


def test_contextual_queue_editor_collects_visible_draft_without_saving(app, tmp_path):
    manager = make_manager(tmp_path)
    queue = register_queue(manager)
    data = project_data(tmp_path)
    success, file_name = manager.save_project(None, data)
    assert success
    calls = []
    dialog = ProductTestProjectConfigDialog(
        manager, contextual_queue_editor_callback=lambda path, provider: calls.append((path, provider))
    )
    dialog._show_project(data, file_name)
    before = Path(manager.program_dir, file_name).read_bytes()
    dialog.project_name_input.setText("Renamed draft")
    dialog.condition_table.item(0, 1).setText("Visible draft condition")
    _, button = dialog._queue_controls_for_row(0)
    assert button.isEnabled()
    dialog._edit_queue_for_button(button)
    path, provider = calls[0]
    first = provider()[0]
    assert first.product_path == os.path.join(manager.program_dir, file_name)
    assert first.data["project_name"] == "Renamed draft"
    assert first.data["test_groups"][0]["test_conditions"][0]["condition_name"] == "Visible draft condition"
    assert provider()[0] is first
    assert Path(manager.program_dir, file_name).read_bytes() == before
    scanner = SequenceQueueReferenceScanner(manager.program_dir, manager.registry_path, manager.queue_registry_path)
    result = scanner.find_references(path, drafts=provider())
    assert len(result.references) == 2
    assert all(ref.sources == {"saved", "draft"} for ref in result.references)
    dialog._show_project(data, None)
    unsaved = provider()[0]
    assert unsaved.product_path is None
    assert provider()[0] is unsaved
    other = ProductTestProjectConfigDialog(manager)
    other._show_project(data, None)
    assert other._queue_reference_drafts()[0] is not unsaved
    other._dirty = False
    other.close()
    dialog._dirty = False
    dialog.close()


def make_queue_config(duration=600.0):
    return [
        {
            "sequence_1": {
                "acq": {
                    "mode": "RECORD_ONLY",
                    "detail": {"total_time": duration, "sample_rate": 44100},
                },
                "analysis_list": {
                    "display_sequence": [
                        "声压级 (SPL) 1",
                        "频谱分析 (FFT) 1",
                        "1/3倍频程 (FBA) 1",
                    ],
                    "声压级 (SPL) 1": {
                        "type": "SPL",
                        "limit_checked": True,
                    },
                    "频谱分析 (FFT) 1": {"type": "FFT"},
                    "1/3倍频程 (FBA) 1": {"type": "FBA"},
                },
            }
        }
    ]


def register_queue(manager, queue_name="低噪声基础测试", duration=600.0):
    queue_dir = os.path.dirname(manager.queue_registry_path)
    queue_path = os.path.join(queue_dir, f"{queue_name}.json")
    assert LoadUiConfig.save_data_to_json(make_queue_config(duration), queue_path)
    assert LoadUiConfig.save_data_to_json(
        {queue_name: queue_path}, manager.queue_registry_path
    )
    return queue_path


def project_data(tmp_path, conditions=None):
    conditions = conditions or [
        {
            "condition_name": "档位1",
            "trigger_state": "",
            "test_queue": "低噪声基础测试",
        }
    ]
    target_trigger = ""
    if any(condition.get("trigger_state") for condition in conditions):
        target_trigger = "01 04 02 01 01 29 30"
    return {
        "project_name": "PB-A01充电宝",
        "result_root_directory": str(tmp_path / "results"),
        "test_groups": [
            {
                "group_name": "USB-C输出口",
                "test_conditions": conditions,
            },
            {
                "group_name": "USB-A输出口",
                "test_conditions": [
                    {
                        "condition_name": "档位1",
                        "trigger_state": target_trigger,
                        "test_queue": "低噪声基础测试",
                    }
                ],
            },
        ],
    }


def prepare_project(manager, tmp_path, conditions=None):
    register_queue(manager)
    success, file_name = manager.save_project(
        None, project_data(tmp_path, conditions)
    )
    assert success, file_name
    return file_name


def close_dialog(dialog):
    dialog._set_dirty(False)
    dialog.close()


@pytest.mark.parametrize("action", ["discard", "decline", "overwrite", "save-as", "rename", "save-failure"])
def test_import_external_duplicate_stays_draft_until_save(app, tmp_path, monkeypatch, action):
    manager = make_manager(tmp_path)
    file_name = prepare_project(manager, tmp_path)
    saved_path = Path(manager.program_dir, file_name)
    before = saved_path.read_bytes()
    registry_before = Path(manager.registry_path).read_bytes()
    imported = project_data(tmp_path)
    imported[EXPORT_RAW_AUDIO_CSV_KEY] = True
    # Same basename outside the managed directory is still an external draft.
    source = tmp_path / file_name
    assert LoadUiConfig.save_data_to_json(imported, str(source))
    source_before = source.read_bytes()
    dialog = ProductTestProjectConfigDialog(manager)
    events = []
    questions = []
    warnings = []
    dialog.projects_changed.connect(lambda: events.append("changed"))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(source), ""))
    monkeypatch.setattr(QMessageBox, "information", lambda *a: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: warnings.append(a[2]))

    def answer(*args):
        questions.append(args[1])
        return QMessageBox.No if action == "decline" else QMessageBox.Yes

    monkeypatch.setattr(QMessageBox, "question", answer)
    try:
        dialog._import_project()
        assert dialog.current_file is None
        assert dialog._dirty
        assert dialog.wav_and_csv_radio.isChecked()
        assert saved_path.read_bytes() == before
        assert Path(manager.registry_path).read_bytes() == registry_before
        assert events == []
        assert questions == []

        if action == "discard":
            monkeypatch.setattr(QMessageBox, "exec_", lambda self: QMessageBox.Discard)
            dialog.reject()
        elif action == "save-as":
            monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("导入副本", True))
            dialog._save_project_as()
            assert dialog.current_file == "导入副本.json"
        elif action == "rename":
            dialog.project_name_input.setText("导入新名称")
            assert dialog._save_project(close_dialog=False)
            assert dialog.current_file == "导入新名称.json"
        else:
            if action == "save-failure":
                monkeypatch.setattr(manager, "save_registry", lambda registry: False)
            assert dialog._save_project(close_dialog=False) is (action == "overwrite")
            assert questions == ["覆盖已有配置"]

        if action == "overwrite":
            assert saved_path.read_bytes() != before
            assert manager.load_project(file_name)[1][EXPORT_RAW_AUDIO_CSV_KEY] is True
            assert dialog.current_file == file_name
        else:
            assert saved_path.read_bytes() == before
        if action in {"discard", "decline", "save-failure"}:
            assert Path(manager.registry_path).read_bytes() == registry_before
            assert events == []
        else:
            assert events == ["changed"]
            assert not dialog._dirty
            assert manager.load_project(dialog.current_file)[1][EXPORT_RAW_AUDIO_CSV_KEY] is True
        assert source.read_bytes() == source_before
        assert bool(warnings) is (action == "save-failure")
    finally:
        close_dialog(dialog)


@pytest.mark.parametrize("missing_queue", [False, True])
def test_import_new_draft_validates_queue_only_on_save(app, tmp_path, monkeypatch, missing_queue):
    manager = make_manager(tmp_path)
    original_file = prepare_project(manager, tmp_path)
    original_path = Path(manager.program_dir, original_file)
    before = original_path.read_bytes()
    imported = project_data(tmp_path)
    imported["project_name"] = "外部项目"
    if missing_queue:
        imported["test_groups"][0]["test_conditions"][0]["test_queue"] = "未安装队列"
    source = tmp_path / "external.json"
    assert LoadUiConfig.save_data_to_json(imported, str(source))
    target = Path(manager.program_dir, "外部项目.json")
    dialog = ProductTestProjectConfigDialog(manager)
    warnings = []
    questions = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(source), ""))
    monkeypatch.setattr(QMessageBox, "information", lambda *a: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: warnings.append(a[2]))
    monkeypatch.setattr(QMessageBox, "question", lambda *a: questions.append(a[1]))
    try:
        dialog._import_project()
        assert dialog.project_name_input.text() == "外部项目"
        assert dialog.current_file is None
        assert dialog._dirty
        assert not target.exists()
        assert not warnings
        assert dialog._save_project(close_dialog=False) is (not missing_queue)
        assert target.exists() is (not missing_queue)
        assert bool(warnings) is missing_queue
        assert questions == []
        assert original_path.read_bytes() == before
    finally:
        close_dialog(dialog)


@pytest.mark.parametrize("confirm_overwrite", [False, True])
def test_import_managed_file_also_confirms_overwrite(app, tmp_path, monkeypatch, confirm_overwrite):
    manager = make_manager(tmp_path)
    file_name = prepare_project(manager, tmp_path)
    saved_path = Path(manager.program_dir, file_name)
    before = saved_path.read_bytes()
    registry_before = Path(manager.registry_path).read_bytes()
    dialog = ProductTestProjectConfigDialog(manager)
    questions = []
    events = []
    dialog.projects_changed.connect(lambda: events.append("changed"))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(saved_path), ""))
    monkeypatch.setattr(QMessageBox, "information", lambda *a: None)
    def answer(*args):
        questions.append(args[1])
        return QMessageBox.Yes if confirm_overwrite else QMessageBox.No

    monkeypatch.setattr(QMessageBox, "question", answer)
    try:
        dialog._import_project()
        assert dialog.current_file is None
        assert dialog._dirty
        assert saved_path.read_bytes() == before
        assert events == []
        dialog.wav_and_csv_radio.setChecked(True)
        assert dialog._save_project(close_dialog=False) is confirm_overwrite
        assert questions == ["覆盖已有配置"]
        if confirm_overwrite:
            assert events == ["changed"]
            assert dialog.current_file == file_name
            assert not dialog._dirty
            assert manager.load_project(file_name)[1][EXPORT_RAW_AUDIO_CSV_KEY] is True
        else:
            assert events == []
            assert dialog.current_file is None
            assert dialog._dirty
            assert saved_path.read_bytes() == before
            assert Path(manager.registry_path).read_bytes() == registry_before
    finally:
        close_dialog(dialog)


@pytest.mark.parametrize("confirm_overwrite", [False, True])
@pytest.mark.parametrize("import_name", ["demo", "DEMO"])
def test_case_variant_import_updates_one_project(app, tmp_path, monkeypatch, confirm_overwrite, import_name):
    manager = make_manager(tmp_path)
    register_queue(manager)
    data = project_data(tmp_path)
    data["project_name"] = "Demo"
    assert manager.save_project(None, data) == (True, "Demo.json")
    registry_before = manager.load_registry()
    saved_path = Path(manager.program_dir, "Demo.json")
    before = saved_path.read_bytes()
    data["project_name"] = import_name
    data[EXPORT_RAW_AUDIO_CSV_KEY] = True
    source = tmp_path / "external.json"
    assert LoadUiConfig.save_data_to_json(data, str(source))
    source_before = source.read_bytes()
    dialog = ProductTestProjectConfigDialog(manager)
    questions = []
    warnings = []

    def answer(*args):
        questions.append(args[2])
        return QMessageBox.Yes if confirm_overwrite else QMessageBox.No

    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(source), ""))
    monkeypatch.setattr(QMessageBox, "question", answer)
    monkeypatch.setattr(QMessageBox, "information", lambda *a: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: warnings.append(a[2]))
    try:
        dialog._import_project()
        assert saved_path.read_bytes() == before
        assert dialog._save_project(close_dialog=False) is confirm_overwrite
        assert len(questions) == 1
        assert "Demo" in questions[0]
        assert not warnings
        assert manager.load_registry() == registry_before
        assert source.read_bytes() == source_before
        if confirm_overwrite:
            assert dialog.current_file == "Demo.json"
            assert dialog.project_name_input.text() == "Demo"
            assert dialog.collect_project()["project_name"] == "Demo"
            saved = manager.load_project("Demo.json")[1]
            assert saved["project_name"] == "Demo"
            assert saved[EXPORT_RAW_AUDIO_CSV_KEY] is True
        else:
            assert saved_path.read_bytes() == before
            assert dialog.project_name_input.text() == import_name
    finally:
        close_dialog(dialog)


def test_import_invalid_file_keeps_displayed_project(app, tmp_path, monkeypatch):
    manager = make_manager(tmp_path)
    file_name = prepare_project(manager, tmp_path)
    source = tmp_path / "broken.json"
    source.write_text('{"project_name": "broken", "test_groups": null}', encoding="utf-8")
    dialog = ProductTestProjectConfigDialog(manager)
    before = dialog.collect_project()
    warnings = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(source), ""))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: warnings.append(a[2]))
    try:
        dialog._import_project()
        assert warnings
        assert dialog.current_file == file_name
        assert dialog.collect_project() == before
    finally:
        close_dialog(dialog)


def test_time_segments_and_voltage_survive_save_reopen_copy_and_runtime(app, tmp_path):
    manager = make_manager(tmp_path)
    register_queue(manager)
    condition = {'condition_name': '最高功率充电', 'test_queue': '低噪声基础测试',
                 'input_voltage': '230Vac/50Hz', 'segmented_analysis': {
                     'mode': 'time', 'interval_seconds': 60, 'display_time_unit': 'min', 'analysis_seconds': 10}}
    data = project_data(tmp_path, [condition])
    dialog = ProductTestProjectConfigDialog(manager)
    dialog._show_project(data, None)
    assert dialog.condition_table.cellWidget(0, 7).text() == '230Vac/50Hz'
    assert dialog.condition_table.cellWidget(0, 5).status_label.text() == '按时间 · 10 段'
    assert dialog._copy_conditions_to_groups([1], confirm_replace=False)
    copied = dialog.project_data['test_groups'][1]['test_conditions'][0]
    assert copied['segmented_analysis'] == condition['segmented_analysis']
    assert copied['input_voltage'] == condition['input_voltage']
    ok, filename = manager.save_project(None, dialog.project_data)
    assert ok
    loaded = manager.load_project(filename)
    if isinstance(loaded, tuple):
        loaded = loaded[1]
    assert loaded['test_groups'][0]['test_conditions'][0]['segmented_analysis'] == condition['segmented_analysis']
    close_dialog(dialog)


def test_dialog_uses_project_port_condition_layout(app, tmp_path):
    manager = make_manager(tmp_path)
    prepare_project(manager, tmp_path)

    dialog = ProductTestProjectConfigDialog(manager)
    app.processEvents()

    assert dialog.windowTitle() == "产品测试配置"
    assert dialog.project_name_input.text() == "PB-A01充电宝"
    assert dialog.port_count_spinbox.value() == 2
    assert dialog.port_tabs.count() == 2
    assert dialog.port_tabs.tabText(0) == "USB-C输出口"
    assert not hasattr(dialog, "group_name_input")
    assert dialog.condition_section_title.parent() is dialog.condition_header
    assert dialog.add_condition_btn.parent() is dialog.condition_header
    assert dialog.condition_table.columnCount() == 8
    assert dialog.condition_table.horizontalHeaderItem(2).text() == "状态码"
    assert dialog.condition_table.horizontalHeaderItem(3).text() == "测试队列配置"
    assert dialog.condition_table.horizontalHeaderItem(4).text() == "录音时长"
    assert dialog.condition_table.horizontalHeaderItem(5).text() == "分段分析"
    assert dialog.condition_table.horizontalHeaderItem(6).text() == "判定与分析"
    assert dialog.add_condition_btn.text() == "+ 添加工况"
    assert dialog.delete_condition_btn.text() == "删除工况"
    assert dialog.delete_project_btn.text() == "删除配置"
    assert dialog.delete_project_btn.isEnabled()
    assert dialog.wav_only_radio.text() == "仅 WAV"
    assert dialog.wav_and_csv_radio.text() == "WAV + CSV"
    assert dialog.wav_only_radio.parentWidget() is dialog.wav_and_csv_radio.parentWidget()
    assert dialog.wav_only_radio.isChecked()
    assert not dialog.wav_and_csv_radio.isChecked()
    assert not hasattr(dialog, "status_label")
    assert dialog.delete_project_btn.objectName() != "productProjectDangerButton"
    assert "#D4E1F2" in dialog.styleSheet()
    assert "#1F2937" in dialog.styleSheet()
    assert ui_style_const.UI_FONT_FAMILY in dialog.styleSheet()
    assert ui_style_const.MAIN_UI_SMALL_FONT_FAMILY not in dialog.styleSheet()
    assert "font-weight: 500" in dialog.styleSheet()
    assert "border-top: 1px solid #AFC0D6" not in dialog.styleSheet()
    assert "border-bottom: 1px solid #AFC0D6" not in dialog.styleSheet()
    assert "productProjectDangerButton" not in dialog.styleSheet()
    root_margins = dialog.layout().contentsMargins()
    assert root_margins.left() == 0
    assert root_margins.right() == 0
    footer_layout = dialog.layout().itemAt(dialog.layout().count() - 1).layout()
    assert footer_layout.contentsMargins().top() == 40
    assert footer_layout.contentsMargins().left() == 10
    assert footer_layout.contentsMargins().right() == 10
    assert dialog.condition_table.cellWidget(0, 2).placeholderText() == "选填"
    assert not hasattr(dialog, "close_trigger_input")
    assert not hasattr(dialog, "pdf_report_checkbox")
    assert all(
        label.text() != "按项目名称建立目录"
        for label in dialog.findChildren(QLabel)
    )
    close_dialog(dialog)


@pytest.mark.parametrize("unit", ["A", "Ω"])
def test_output_load_settings_round_trip_and_copy(app, tmp_path, monkeypatch, unit):
    from ui.output_load_config_dialog import OutputLoadConfigDialog

    manager = make_manager(tmp_path)
    file_name = prepare_project(manager, tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)

    def edit_loads(editor):
        editor.mode_buttons["output_load"].setChecked(True)
        editor.loads_input.setPlainText("0.3，0 0.15\n0.3")
        editor.duration_input.setValue(10)
        editor.unit_input.setCurrentText(unit)
        assert editor.settings()["load_unit"] == unit
        return editor.Accepted

    monkeypatch.setattr(OutputLoadConfigDialog, "exec", edit_loads)
    dialog.condition_table.cellWidget(0, 5).settings_button.click()
    assert dialog.condition_table.cellWidget(0, 5).status_label.text() == "按负载 · 4 段"
    dialog.port_tabs.setCurrentIndex(1)
    assert dialog.condition_table.cellWidget(0, 5).status_label.text() == "未启用"
    dialog.port_tabs.setCurrentIndex(0)
    assert dialog._copy_conditions_to_groups([1], confirm_replace=False)
    collected = dialog.collect_project()
    expected = {"mode": "output_load", "load_values": [0.3, 0, 0.15, 0.3], "load_unit": unit, "analysis_seconds": 10}
    for group in collected["test_groups"]:
        assert group["test_conditions"][0]["segmented_analysis"] == expected
    success, saved_file = manager.save_project(file_name, collected)
    assert success, saved_file
    _, loaded = manager.load_project(saved_file)
    assert loaded["test_groups"][0]["test_conditions"][0]["segmented_analysis"] == expected
    close_dialog(dialog)


@pytest.mark.parametrize("text", ["0.12345678", "1234567.89", "0.000123456789"])
def test_load_values_keep_precision_when_confirmed_and_reopened(app, text):
    from PyQt5.QtWidgets import QDialogButtonBox
    from ui.output_load_config_dialog import OutputLoadConfigDialog

    dialog = OutputLoadConfigDialog("档位1", None, 60)
    dialog.mode_buttons["output_load"].setChecked(True)
    dialog.loads_input.setPlainText(text)
    dialog.buttons.button(QDialogButtonBox.Ok).click()
    assert dialog.result() == dialog.Accepted
    saved = dialog.settings()
    assert saved["load_values"] == [float(text)]
    dialog.close()

    reopened = OutputLoadConfigDialog("档位1", saved, 60)
    assert float(reopened.loads_input.toPlainText()) == float(text)
    reopened.buttons.button(QDialogButtonBox.Ok).click()
    assert reopened.result() == reopened.Accepted
    assert reopened.settings() == saved
    reopened.close()


@pytest.mark.parametrize("text", ["", "-1", "nan", "inf", "abc"])
def test_output_load_editor_rejects_invalid_values(app, text):
    from PyQt5.QtWidgets import QDialogButtonBox
    from ui.output_load_config_dialog import OutputLoadConfigDialog

    dialog = OutputLoadConfigDialog("档位1", None, 60)
    dialog.mode_buttons["output_load"].setChecked(True)
    dialog.loads_input.setPlainText(text)
    assert not dialog.buttons.button(QDialogButtonBox.Ok).isEnabled()
    dialog.close()


def test_load_unit_can_be_typed_validated_and_reopened(app):
    from PyQt5.QtWidgets import QDialogButtonBox
    from ui.output_load_config_dialog import OutputLoadConfigDialog

    dialog = OutputLoadConfigDialog("档位1", {
        "mode": "output_load", "load_values": [0, 0.3],
        "load_unit": "Ω", "analysis_seconds": 10,
    }, 600)
    dialog.show()
    app.processEvents()
    assert dialog.unit_input.isEditable()
    assert dialog.unit_input.currentText() == "Ω"
    edit = dialog.unit_input.lineEdit()
    edit.selectAll()
    QTest.keyClicks(edit, "  mW  ")
    assert dialog.settings()["load_unit"] == "mW"
    assert dialog.buttons.button(QDialogButtonBox.Ok).isEnabled()
    saved = dialog.settings()
    edit.clear()
    assert not dialog.buttons.button(QDialogButtonBox.Ok).isEnabled()
    assert dialog.error_label.text() == "请选择或输入负载单位"
    dialog.unit_input.setCurrentIndex(0)
    assert dialog.settings()["load_unit"] == "A"
    assert dialog.buttons.button(QDialogButtonBox.Ok).isEnabled()
    dialog.close()
    reopened = OutputLoadConfigDialog("档位1", saved, 600)
    assert reopened.unit_input.currentText() == "mW"
    assert reopened.settings() == saved
    reopened.close()


@pytest.mark.parametrize("seconds,unit,expected_unit", [
    (1, "min", "s"),
    (1, "h", "s"),
    (0.125, "s", "s"),
    (0.125, "min", "s"),
    (90, "min", "min"),
    (1800, "h", "h"),
    (90000, "s", "s"),
])
def test_time_interval_survives_reopen_without_changing_segment_windows(
    app, seconds, unit, expected_unit,
):
    from PyQt5.QtWidgets import QDialogButtonBox
    from base.analysis_segments import build_segment_plan
    from ui.output_load_config_dialog import OutputLoadConfigDialog

    saved = {"mode": "time", "interval_seconds": seconds,
             "display_time_unit": unit, "analysis_seconds": 0.01}
    total_duration = seconds * 60
    original_plan = build_segment_plan(saved, total_duration, 48000)
    original_windows = [(s.start_sample, s.end_sample, s.window_start_sample,
                         s.window_end_sample) for s in original_plan]
    for _ in range(2):
        dialog = OutputLoadConfigDialog("档位1", saved, total_duration)
        assert dialog.time_unit_input.currentData() == expected_unit
        assert dialog.buttons.button(QDialogButtonBox.Ok).isEnabled()
        dialog.buttons.button(QDialogButtonBox.Ok).click()
        assert dialog.result() == dialog.Accepted
        saved = dialog.settings()
        assert saved["interval_seconds"] == seconds
        plan = build_segment_plan(saved, total_duration, 48000)
        assert [(s.start_sample, s.end_sample, s.window_start_sample,
                 s.window_end_sample) for s in plan] == original_windows
        dialog.close()


@pytest.mark.parametrize("mode", ["time", "output_load"])
@pytest.mark.parametrize("duration", [0.125, 0.001, 90000])
def test_analysis_duration_is_preserved_when_reopening(app, mode, duration):
    from PyQt5.QtWidgets import QDialogButtonBox
    from base.analysis_segments import build_segment_plan
    from ui.output_load_config_dialog import OutputLoadConfigDialog

    saved = {"mode": mode, "analysis_seconds": duration}
    if mode == "time":
        saved.update(interval_seconds=duration * 2, display_time_unit="s")
    else:
        saved.update(load_values=[0, 0.3], load_unit="A")
    original_plan = build_segment_plan(saved, duration * 4, 48000)
    for _ in range(2):
        dialog = OutputLoadConfigDialog("档位1", saved, duration * 4)
        assert dialog.duration_input.value() == duration
        assert dialog.buttons.button(QDialogButtonBox.Ok).isEnabled()
        dialog.buttons.button(QDialogButtonBox.Ok).click()
        assert dialog.result() == dialog.Accepted
        saved = dialog.settings()
        assert saved["analysis_seconds"] == duration
        assert build_segment_plan(saved, duration * 4, 48000) == original_plan
        dialog.close()


@pytest.mark.parametrize("mode", ["time", "output_load"])
@pytest.mark.parametrize("duration,allowed", [(0.1, True), (0.11, False)])
def test_decimal_segment_duration_dialog_validation(app, mode, duration, allowed):
    from PyQt5.QtWidgets import QDialogButtonBox
    from ui.output_load_config_dialog import OutputLoadConfigDialog

    settings = {"mode": mode, "analysis_seconds": duration}
    if mode == "time":
        settings["interval_seconds"] = 0.1
    else:
        settings["load_values"] = list(range(101))
    dialog = OutputLoadConfigDialog("档位1", settings, 10.1)
    try:
        assert dialog.buttons.button(QDialogButtonBox.Ok).isEnabled() is allowed
        assert bool(dialog.error_label.text()) is (not allowed)
        assert dialog.settings()["analysis_seconds"] == duration
    finally:
        dialog.close()


def test_recording_duration_summaries_preserve_numeric_precision(app, tmp_path):
    from PyQt5.QtWidgets import QLabel
    from ui.output_load_config_dialog import OutputLoadConfigDialog

    manager = make_manager(tmp_path)
    prepare_project(manager, tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)
    combo, _ = dialog._queue_controls_for_row(0)
    dialog.queue_catalog[combo.currentText()]["duration"] = 1.23456789
    dialog._update_row_summary(0)
    assert dialog.condition_table.item(0, 4).text() == "1.23456789秒"
    close_dialog(dialog)
    editor = OutputLoadConfigDialog("档位1", None, 1.23456789)
    assert "录音时长：1.23456789 秒" in [label.text() for label in editor.findChildren(QLabel)]
    editor.close()


def test_hour_time_unit_converts_to_seconds_and_reopens(app):
    from PyQt5.QtWidgets import QDialogButtonBox
    from ui.output_load_config_dialog import OutputLoadConfigDialog

    settings = {"mode": "time", "interval_seconds": 3600,
                "display_time_unit": "h", "analysis_seconds": 10}
    dialog = OutputLoadConfigDialog("档位1", settings, 7200)
    assert not dialog.time_unit_input.isEditable()
    assert dialog.time_unit_input.currentText() == "小时"
    assert dialog.interval_input.value() == 1
    assert dialog.settings() == settings
    dialog.interval_input.setValue(0.5)
    saved = dialog.settings()
    assert saved["interval_seconds"] == 1800
    assert dialog.buttons.button(QDialogButtonBox.Ok).isEnabled()
    dialog.close()
    reopened = OutputLoadConfigDialog("档位1", saved, 7200)
    assert reopened.time_unit_input.currentData() == "h"
    assert reopened.interval_input.value() == 0.5
    assert reopened.settings() == saved
    reopened.close()


def test_output_load_editor_duration_and_cancel(app, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QDialogButtonBox
    from ui.output_load_config_dialog import OutputLoadConfigDialog

    editor = OutputLoadConfigDialog("档位1", None, 60)
    editor.mode_buttons["output_load"].setChecked(True)
    editor.loads_input.setPlainText("0, 1, 2")
    editor.duration_input.setValue(20)
    assert editor.buttons.button(QDialogButtonBox.Ok).isEnabled()
    editor.duration_input.setValue(21)
    assert not editor.buttons.button(QDialogButtonBox.Ok).isEnabled()
    editor.mode_buttons["none"].setChecked(True)
    assert editor.buttons.button(QDialogButtonBox.Ok).isEnabled()
    editor.close()

    manager = make_manager(tmp_path)
    prepare_project(manager, tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)
    before = dialog.collect_project()
    monkeypatch.setattr(OutputLoadConfigDialog, "exec", lambda self: self.Rejected)
    dialog.condition_table.cellWidget(0, 5).settings_button.click()
    assert dialog.collect_project() == before
    close_dialog(dialog)


def test_raw_audio_save_choice_round_trips_through_project_json(app, tmp_path):
    manager = make_manager(tmp_path)
    file_name = prepare_project(manager, tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)

    dialog.wav_and_csv_radio.setChecked(True)
    project = dialog.collect_project()
    assert project[EXPORT_RAW_AUDIO_CSV_KEY] is True
    success, saved_file_name = manager.save_project(file_name, project)
    assert success is True
    assert saved_file_name == file_name

    load_code, saved = manager.load_project(file_name)
    assert load_code == 0
    assert saved[EXPORT_RAW_AUDIO_CSV_KEY] is True
    close_dialog(dialog)


def test_port_name_can_be_edited_inline_from_selector(app, tmp_path):
    manager = make_manager(tmp_path)
    prepare_project(manager, tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)
    dialog.show()
    app.processEvents()
    initial_row_count = dialog.condition_table.rowCount()

    QTest.mouseDClick(dialog.port_tabs._buttons[0], Qt.LeftButton)
    app.processEvents()
    assert dialog.port_tabs._name_editor.isVisible()

    dialog.port_tabs._name_editor.setText("USB-C PD")
    QTest.keyClick(dialog.port_tabs._name_editor, Qt.Key_Return)
    app.processEvents()

    assert not dialog.port_tabs._name_editor.isVisible()
    assert dialog.port_tabs.tabText(0) == "USB-C PD"
    assert dialog.condition_section_title.text() == "工况配置   ·   USB-C PD"
    assert dialog.condition_table.rowCount() == initial_row_count
    assert not dialog.add_condition_btn.autoDefault()
    assert dialog.collect_project()["test_groups"][0]["group_name"] == "USB-C PD"
    close_dialog(dialog)


def test_queue_duration_summary_and_operation_are_derived(app, tmp_path):
    manager = make_manager(tmp_path)
    prepare_project(manager, tmp_path)
    dialog = ProductTestProjectConfigDialog(manager, lambda _path: None)
    app.processEvents()

    assert dialog.condition_table.item(0, 4).text() == "600秒"
    summary = dialog.condition_table.item(0, 6).text()
    assert not summary.startswith("自动判定")
    assert dialog.condition_table.item(0, 6).toolTip().splitlines() == [
        "声压级 (SPL) 1", "频谱分析 (FFT) 1", "1/3倍频程 (FBA) 1"
    ]
    assert "声压级 (SPL) 1" in summary
    assert "频谱分析 (FFT) 1" in summary
    _queue_combobox, operation_button = dialog._queue_controls_for_row(0)
    assert operation_button.text() == "编辑"
    assert dialog.collect_project()["test_groups"][0]["test_conditions"][0] == {
        "condition_name": "档位1",
        "trigger_state": "",
        "test_queue": "低噪声基础测试",
        "input_voltage": "",
        "segmented_analysis": {"mode": "none"},
    }
    close_dialog(dialog)


def test_more_than_twenty_conditions_and_port_switch_preserve_edits(
    app, tmp_path
):
    manager = make_manager(tmp_path)
    conditions = [
        {
            "condition_name": f"档位{index}",
            "trigger_state": "",
            "test_queue": "低噪声基础测试",
        }
        for index in range(1, 22)
    ]
    prepare_project(manager, tmp_path, conditions)
    dialog = ProductTestProjectConfigDialog(manager)

    assert dialog.condition_table.rowCount() == 21
    dialog.condition_table.item(20, 1).setText("自定义档位")
    dialog.port_tabs.setCurrentIndex(1)
    dialog.port_tabs.setCurrentIndex(0)
    app.processEvents()

    assert dialog.condition_table.rowCount() == 21
    assert dialog.condition_table.item(20, 1).text() == "自定义档位"
    close_dialog(dialog)


def test_delete_condition_is_immediate_and_allows_empty_table(
    app, tmp_path, monkeypatch
):
    manager = make_manager(tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)
    dialog._add_condition()
    app.processEvents()
    assert dialog.condition_table.rowCount() == 2
    assert dialog.delete_condition_btn.isEnabled()

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: pytest.fail("删除工况不应弹出确认窗口"),
    )
    dialog.condition_table.selectRow(1)
    dialog.delete_condition_btn.click()
    app.processEvents()

    assert dialog.condition_table.rowCount() == 1
    assert dialog.delete_condition_btn.isEnabled()

    dialog.delete_condition_btn.click()
    app.processEvents()

    assert dialog.condition_table.rowCount() == 0
    assert dialog.delete_condition_btn.isEnabled()
    close_dialog(dialog)


def test_port_count_adds_named_port_on_same_tab_bar(app, tmp_path):
    manager = make_manager(tmp_path)
    prepare_project(manager, tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)

    dialog.port_count_spinbox.setValue(3)
    app.processEvents()

    assert dialog.port_tabs.count() == 3
    assert dialog.port_tabs.tabText(2) == "新端口3"
    collected = dialog.collect_project()
    assert len(collected["test_groups"]) == 3
    assert collected["test_groups"][2]["test_conditions"][0][
        "condition_name"
    ] == "档位1"
    close_dialog(dialog)


def test_port_button_width_stays_fixed_when_port_count_changes(app, tmp_path):
    manager = make_manager(tmp_path)
    prepare_project(manager, tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)
    dialog.show()
    app.processEvents()

    initial_width = dialog.port_tabs.tabRect(0).width()
    assert initial_width == dialog.port_tabs.BUTTON_WIDTH

    dialog.port_count_spinbox.setValue(2)
    app.processEvents()

    assert dialog.port_tabs.tabRect(0).width() == initial_width
    assert dialog.port_tabs.tabRect(1).width() == initial_width
    close_dialog(dialog)


def test_queue_controls_keep_height_when_port_count_changes(app, tmp_path):
    manager = make_manager(tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)
    dialog.show()
    app.processEvents()

    trigger_input = dialog.condition_table.cellWidget(0, 2)
    queue_combobox, operation_button = dialog._queue_controls_for_row(0)
    initial_heights = (queue_combobox.height(), operation_button.height())
    initial_fonts = [
        (widget.font().family(), widget.font().pixelSize())
        for widget in (trigger_input, queue_combobox, operation_button)
    ]
    assert initial_heights == (dialog.CONDITION_CONTROL_HEIGHT,) * 2
    assert initial_fonts == [
        (
            dialog.CONDITION_CONTROL_FONT_FAMILY,
            dialog.CONDITION_CONTROL_FONT_SIZE,
        )
    ] * 3

    dialog.port_count_spinbox.setValue(2)
    app.processEvents()
    trigger_input = dialog.condition_table.cellWidget(0, 2)
    queue_combobox, operation_button = dialog._queue_controls_for_row(0)

    assert (queue_combobox.height(), operation_button.height()) == initial_heights
    assert [
        (widget.font().family(), widget.font().pixelSize())
        for widget in (trigger_input, queue_combobox, operation_button)
    ] == initial_fonts
    close_dialog(dialog)


def test_port_tab_scrollbar_supports_overflow_navigation(app, tmp_path):
    manager = make_manager(tmp_path)
    prepare_project(manager, tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)
    dialog.show()
    app.processEvents()

    scroll_bar = dialog.port_tabs_scroll_area.horizontalScrollBar()
    assert scroll_bar.maximum() == 0
    assert not scroll_bar.isVisible()
    assert dialog.port_tabs.width() < dialog.port_tabs_scroll_area.viewport().width()

    dialog.port_count_spinbox.setValue(12)
    app.processEvents()

    class WheelEvent:
        def __init__(self):
            self.accepted = False

        @staticmethod
        def angleDelta():
            return QPoint(0, -120)

        def accept(self):
            self.accepted = True

    assert scroll_bar.maximum() > 0
    assert scroll_bar.isVisible()
    assert dialog.port_tabs.currentIndex() == 0

    event = WheelEvent()
    dialog.port_tabs.wheelEvent(event)
    app.processEvents()

    assert event.accepted
    assert dialog.port_tabs.currentIndex() == 0
    assert scroll_bar.value() > 0

    scroll_bar.setValue(scroll_bar.maximum())
    assert scroll_bar.value() == scroll_bar.maximum()
    close_dialog(dialog)


def test_queue_cell_contains_selector_and_full_operation_button(app, tmp_path):
    manager = make_manager(tmp_path)
    prepare_project(manager, tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)

    required_width = max(
        dialog.condition_table.fontMetrics().horizontalAdvance(text)
        for text in ("新建", "编辑")
    ) + 32

    queue_combobox, operation_button = dialog._queue_controls_for_row(0)

    assert queue_combobox is not None
    assert operation_button.minimumWidth() >= required_width - 8
    assert dialog.condition_table.cellWidget(0, 3) is not queue_combobox
    close_dialog(dialog)


def test_copy_conditions_replaces_targets_without_trigger_states(app, tmp_path):
    manager = make_manager(tmp_path)
    conditions = [
        {
            "condition_name": "档位1",
            "trigger_state": "01 04 02 00 01 78 F0",
            "test_queue": "低噪声基础测试",
        },
        {
            "condition_name": "档位2",
            "trigger_state": "01 04 02 00 02 38 F1",
            "test_queue": "低噪声基础测试",
        },
    ]
    prepare_project(manager, tmp_path, conditions)
    dialog = ProductTestProjectConfigDialog(manager)

    assert dialog._copy_conditions_to_groups([1], confirm_replace=False)
    target_conditions = dialog.project_data["test_groups"][1]["test_conditions"]

    assert [item["condition_name"] for item in target_conditions] == [
        "档位1",
        "档位2",
    ]
    assert [item["trigger_state"] for item in target_conditions] == ["", ""]
    assert all(
        item["test_queue"] == "低噪声基础测试"
        for item in target_conditions
    )
    close_dialog(dialog)


def test_result_directory_keeps_root_without_directory_status_note(app, tmp_path):
    manager = make_manager(tmp_path)
    prepare_project(manager, tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)

    result_root = str(tmp_path / "results")
    assert os.path.normpath(dialog.result_root_input.text()) == os.path.normpath(
        result_root
    )
    assert not hasattr(dialog, "project_directory_preview")
    assert dialog.select_result_root_btn.text() == "选择"

    dialog.project_name_input.setText("PB-A02充电宝")
    dialog._on_project_field_changed()
    assert dialog.result_root_input.text() == result_root
    assert dialog.collect_project()["result_root_directory"] == result_root
    close_dialog(dialog)


def test_save_button_closes_dialog_after_success(app, tmp_path, monkeypatch):
    manager = make_manager(tmp_path)
    prepare_project(manager, tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)

    dialog.save_btn.click()
    app.processEvents()

    assert dialog.result() == dialog.Accepted


def test_unsaved_changes_dialog_uses_chinese_button_text(
    app, tmp_path, monkeypatch
):
    manager = make_manager(tmp_path)
    prepare_project(manager, tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)
    captured = {}

    def capture_message_box(message_box):
        captured["save"] = message_box.button(QMessageBox.Save).text()
        captured["discard"] = message_box.button(QMessageBox.Discard).text()
        captured["cancel"] = message_box.button(QMessageBox.Cancel).text()
        return QMessageBox.Discard

    monkeypatch.setattr(QMessageBox, "exec_", capture_message_box)
    dialog._set_dirty(True)

    assert dialog._confirm_leave_changes()
    assert captured == {
        "save": "保存",
        "discard": "不保存",
        "cancel": "取消",
    }
    close_dialog(dialog)


def test_delete_project_removes_registry_entry_and_configuration(
    app, tmp_path, monkeypatch
):
    manager = make_manager(tmp_path)
    file_name = prepare_project(manager, tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes
    )

    dialog._delete_project()

    assert not os.path.exists(os.path.join(manager.program_dir, file_name))
    assert manager.load_registry()["configs"] == []
    assert dialog.current_file is None
    close_dialog(dialog)


def test_new_queue_is_selected_when_editor_creates_one_queue(app, tmp_path):
    manager = make_manager(tmp_path)
    register_queue(manager)
    new_project = project_data(
        tmp_path,
        [
            {
                "condition_name": "档位1",
                "trigger_state": "",
                "test_queue": "",
            }
        ],
    )
    new_project["test_groups"] = new_project["test_groups"][:1]

    def create_queue(_queue_path):
        queue_dir = os.path.dirname(manager.queue_registry_path)
        new_path = os.path.join(queue_dir, "新测试队列.json")
        assert LoadUiConfig.save_data_to_json(make_queue_config(10), new_path)
        assert LoadUiConfig.save_data_to_json(
            {
                "低噪声基础测试": os.path.join(
                    queue_dir, "低噪声基础测试.json"
                ),
                "新测试队列": new_path,
            },
            manager.queue_registry_path,
        )

    dialog = ProductTestProjectConfigDialog(manager, create_queue)
    dialog._show_project(new_project, None)
    queue_combobox, button = dialog._queue_controls_for_row(0)

    dialog._edit_queue_for_button(button)
    app.processEvents()

    assert queue_combobox.currentData() == "新测试队列"
    assert dialog.condition_table.item(0, 4).text() == "10秒"
    assert button.text() == "编辑"
    close_dialog(dialog)


def test_unavailable_queue_opens_editor_in_new_mode(app, tmp_path):
    manager = make_manager(tmp_path)
    prepare_project(manager, tmp_path)
    queue_path = manager.load_queue_catalog()["低噪声基础测试"]["path"]
    os.remove(queue_path)
    opened_paths = []
    dialog = ProductTestProjectConfigDialog(manager, opened_paths.append)
    _queue_combobox, button = dialog._queue_controls_for_row(0)

    assert button.text() == "新建"
    dialog._edit_queue_for_button(button)

    assert opened_paths == [None]
    close_dialog(dialog)


def test_main_selector_uses_project_name_from_registry(app):
    combobox = QComboBox()
    registry = {
        "active_file": "PB-A01充电宝.json",
        "configs": [
            {
                "file": "PB-A01充电宝.json",
                "project_name": "PB-A01充电宝",
            }
        ],
    }
    host = SimpleNamespace(
        using_file_combobox=combobox,
        _get_product_program_registry=lambda: registry,
    )

    SequenceWidgetConfigOpsMixin.add_file_to_using_file_combobox(host)

    assert combobox.count() == 1
    assert combobox.currentText() == "PB-A01充电宝"
    assert combobox.currentData() == "PB-A01充电宝.json"


def test_project_condition_display_uses_group_and_composite_key():
    conditions = [
        {
            "key": "group_1:condition_1",
            "group_name": "USB-C输出口",
            "condition_name": "档位1",
            "display_name": "USB-C输出口 / 档位1",
            "trigger_state": "01 04 02 00 01 78 F0",
            "test_queue": "低噪声基础测试",
        }
    ]

    waveform_conditions = DirectionWaveformPanel._normalize_conditions(conditions)
    recent_conditions = RecentSessionPanel._normalize_conditions(conditions)

    assert waveform_conditions == [
        {"key": "group_1:condition_1", "name": "USB-C输出口 / 档位1", "test_queue": "低噪声基础测试"}
    ]
    assert recent_conditions == [
        {"key": "group_1:condition_1", "name": "USB-C输出口 / 档位1"}
    ]
