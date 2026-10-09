import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt5 import sip
from PyQt5.QtCore import QCoreApplication, QEvent, QPoint, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QComboBox, QFileDialog, QInputDialog, QLabel, QMenu, QMessageBox

from base.load_config import LoadUiConfig
from base.sequence_queue_references import SequenceQueueReferenceScanner
from base.product_test_project_config import ProductTestProjectConfigManager
from consts.ve3668n_consts import VE_BACKEND
from consts import ui_style_const
from consts.product_test_project_consts import EXPORT_RAW_AUDIO_CSV_KEY
from ui.product_test_project_config_dialog import (
    ProductTestProjectConfigDialog,
)
from ui.sequence.direction_waveform_panel import DirectionWaveformPanel
from ui.sequence.recent_session_panel import RecentSessionPanel
from ui.sequence.sequence_widget_config_ops import SequenceWidgetConfigOpsMixin


@pytest.fixture(scope="module")
def app(qt_app):
    yield qt_app
    # Leave Qt-owned widgets and pyqtgraph menus to their actual owners.
    windows = [widget for widget in qt_app.topLevelWidgets()
               if widget.parent() is None and sip.ispyowned(widget)
               and not isinstance(widget, QMenu)]
    for window in windows:
        window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    qt_app.processEvents()


def make_manager(tmp_path, *, input_device_provider=lambda: {"backend": VE_BACKEND}):
    program_dir = tmp_path / "product_test_programs"
    queue_dir = tmp_path / "analysis_sequence_config"
    program_dir.mkdir()
    queue_dir.mkdir()
    return ProductTestProjectConfigManager(
        str(program_dir),
        str(program_dir / "program_registry.json"),
        str(queue_dir / "sequence_config_registry.json"),
        input_device_provider=input_device_provider,
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


def make_queue_config(duration=600.0, sample_rate=48000, ve_range_index=0):
    return [
        {
            "sequence_1": {
                "acq": {
                    "mode": "RECORD_ONLY",
                    "detail": {"total_time": duration, "sample_rate": sample_rate, "ve_range_index": ve_range_index},
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


@pytest.mark.parametrize("save_as", [False, True])
@pytest.mark.parametrize("variant", ["missing", "invalid", "different", "no_device", "rate_mismatch", "unavailable_vk", "late_vk"])
def test_device_context_save_paths(app, tmp_path, monkeypatch, save_as, variant):
    import ui.product_test_project_config_dialog as dialog_module

    device = None if variant == "no_device" else {"backend": "soundcard"}
    if variant == "unavailable_vk":
        device = {"backend": VE_BACKEND, "available": False}
    calls = []
    def provider():
        calls.append(device)
        if variant == "late_vk" and len(calls) == 2:
            return {"backend": VE_BACKEND}
        return device
    setup_manager = make_manager(tmp_path, input_device_provider=None)
    first_path = register_queue(setup_manager)
    second_path = str(Path(first_path).with_name("队列B.json"))
    first, second = make_queue_config(), make_queue_config()
    first[0]["sequence_1"]["acq"]["detail"].pop("ve_range_index")
    detail = second[0]["sequence_1"]["acq"]["detail"]
    if variant == "invalid":
        detail["ve_range_index"] = "bad"
    elif variant == "different":
        first[0]["sequence_1"]["acq"]["detail"]["ve_range_index"] = 0
        detail["ve_range_index"] = 1
    else:
        detail.pop("ve_range_index")
    if variant == "rate_mismatch":
        detail["sample_rate"] = 44100
    assert LoadUiConfig.save_data_to_json(first, first_path)
    assert LoadUiConfig.save_data_to_json(second, second_path)
    assert LoadUiConfig.save_data_to_json({"低噪声基础测试": first_path, "队列B": second_path}, setup_manager.queue_registry_path)
    # Keep the real default-manager construction, redirecting only its file paths.
    def create_manager(**kwargs):
        return ProductTestProjectConfigManager(setup_manager.program_dir, setup_manager.registry_path,
                                               setup_manager.queue_registry_path, **kwargs)
    monkeypatch.setattr(dialog_module, "ProductTestProjectConfigManager", create_manager)
    dialog = ProductTestProjectConfigDialog(input_device_provider=provider)
    data = project_data(tmp_path)
    data["test_groups"][1]["test_conditions"][0]["test_queue"] = "队列B"
    dialog._show_project(data, None)
    dialog.condition_table.item(0, 1).setText("草稿工况")
    contents = dialog.collect_project()
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    messages, conflicts, changes = [], [], []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: messages.append(args[2]))
    monkeypatch.setattr(QMessageBox, "information", lambda *args: None)
    capture_conflict_dialog(monkeypatch, conflicts)
    def accept_name(name_dialog):
        name_dialog.setTextValue("副本")
        if variant == "late_vk" and save_as:
            device["backend"] = VE_BACKEND
        return QInputDialog.Accepted
    monkeypatch.setattr(QInputDialog, "exec_", accept_name)
    dialog.programs_changed.connect(lambda: changes.append(True))
    assert calls == []
    try:
        if save_as:
            dialog._save_project_as()
        else:
            dialog._save_project(close_dialog=False)
        if variant in {"rate_mismatch", "unavailable_vk", "late_vk"}:
            assert not changes
            assert dialog._dirty and dialog.collect_project() == contents
            assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == before
            if variant == "rate_mismatch":
                assert conflicts[0][1] == "采样率不一致："
                assert len(conflicts[0][2]) == 2
                assert all("量程" not in row for row in conflicts[0][2])
            else:
                assert "ve_range_index" in messages[0]
        else:
            assert changes == [True]
            assert not messages and not conflicts
            assert len(calls) == (1 if save_as else 2)
            assert Path(setup_manager.program_dir, "副本.json" if save_as else data["project_name"] + ".json").exists()
            assert {p: p.read_bytes() for p in before} == before
        if variant == "late_vk":
            assert len(calls) == (1 if save_as else 2)
    finally:
        close_dialog(dialog)


def test_injected_manager_keeps_its_device_context(app, tmp_path):
    manager = make_manager(tmp_path)
    def unexpected_provider():
        pytest.fail("Injected manager must keep its own provider")
    dialog = ProductTestProjectConfigDialog(manager, input_device_provider=unexpected_provider)
    try:
        assert dialog.manager is manager
        assert manager.input_device_provider()["backend"] == VE_BACKEND
    finally:
        close_dialog(dialog)


def capture_conflict_dialog(monkeypatch, observed):
    from ui.product_queue_conflict_dialog import ProductQueueConflictDialog

    def capture(window):
        rows = tuple(window.conflict_list.item(i).text() for i in range(window.conflict_list.count()))
        observed.append((window.windowTitle(), window.prompt_label.text(), rows))
        return QMessageBox.Ok

    monkeypatch.setattr(ProductQueueConflictDialog, "exec_", capture)


@pytest.mark.parametrize("save_as", [False, True])
@pytest.mark.parametrize("invalid", [False, True])
def test_acquisition_save_rejection_keeps_dialog_draft(app, tmp_path, monkeypatch, save_as, invalid):
    manager = make_manager(tmp_path)
    first_path = register_queue(manager)
    second_path = str(Path(first_path).with_name("队列B.json"))
    assert LoadUiConfig.save_data_to_json(make_queue_config(), second_path)
    assert LoadUiConfig.save_data_to_json({"低噪声基础测试": first_path, "队列B": second_path}, manager.queue_registry_path)
    data = project_data(tmp_path)
    data["test_groups"][1]["test_conditions"][0]["test_queue"] = "队列B"
    success, file_name = manager.save_project(None, data)
    assert success
    dialog = ProductTestProjectConfigDialog(manager)
    dialog._show_project(data, file_name)
    dialog.project_name_input.setText("未保存草稿")
    dialog.condition_table.item(0, 1).setText("未保存工况")
    assert dialog._dirty
    # Disk changes after the dialog captured its catalog must be seen on save.
    changed = make_queue_config(sample_rate="bad" if invalid else 44100, ve_range_index=1)
    assert LoadUiConfig.save_data_to_json(changed, second_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    contents = dialog.collect_project()
    messages, changes, accepted, conflicts = [], [], [], []
    if not invalid:
        capture_conflict_dialog(monkeypatch, conflicts)
    dialog.projects_changed.connect(lambda: changes.append("projects"))
    dialog.programs_changed.connect(lambda: changes.append("programs"))
    dialog.accepted.connect(lambda: accepted.append(True))
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: messages.append((args[1], args[2])))
    monkeypatch.setattr(QMessageBox, "information", lambda *args: pytest.fail("unexpected success"))
    def accept_name(name_dialog):
        name_dialog.setTextValue("另存草稿")
        return QInputDialog.Accepted
    monkeypatch.setattr(QInputDialog, "exec_", accept_name)
    try:
        if save_as:
            dialog._save_project_as()
        else:
            assert dialog._save_project() is False
        if invalid:
            assert len(messages) == 1 and not conflicts
            title, message = messages[0]
            assert "\n" not in message
        else:
            assert not messages and len(conflicts) == 1
            title, summary, rows = conflicts[0]
            assert summary == "采样率、量程不一致："
            assert rows == (
                "USB-C输出口/未保存工况的测试队列“低噪声基础测试”（采样率 48000 Hz，量程 ±10 V）",
                "USB-A输出口/档位1的测试队列“队列B”（采样率 44100 Hz，量程 ±5 V）",
            )
            message = "\n".join((summary, *rows))
        assert title == ("另存为失败" if save_as else "无法保存")
        for text in ["队列B", "USB-A输出口/档位1"]:
            assert text in message
        for text in (["sample_rate", "bad"] if invalid else ["低噪声基础测试", "48000 Hz", "44100 Hz", "±10 V", "±5 V", "采样率", "量程"]):
            assert text in message
        assert dialog.current_file == file_name and dialog._dirty
        assert dialog.collect_project() == contents
        assert not changes and not accepted
        assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == before
        assert not (tmp_path / "results" / "未保存草稿").exists()
        assert not (tmp_path / "results" / "另存草稿").exists()
    finally:
        close_dialog(dialog)


def test_acquisition_save_sees_corrected_disk_queue(app, tmp_path, monkeypatch):
    manager = make_manager(tmp_path)
    path = register_queue(manager)
    assert LoadUiConfig.save_data_to_json(make_queue_config(sample_rate="bad"), path)
    dialog = ProductTestProjectConfigDialog(manager)
    dialog._show_project(project_data(tmp_path), None)
    assert LoadUiConfig.save_data_to_json(make_queue_config(), path)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: pytest.fail(args[2]))
    monkeypatch.setattr(QMessageBox, "information", lambda *args: None)
    try:
        assert dialog._save_project(close_dialog=False)
        assert not dialog._dirty
    finally:
        close_dialog(dialog)


@pytest.mark.parametrize("invalid", [False, True])
@pytest.mark.parametrize("overwrite", [False, True])
def test_acquisition_imported_draft_rejection_keeps_import_state(app, tmp_path, monkeypatch, overwrite, invalid):
    manager = make_manager(tmp_path)
    file_name = prepare_project(manager, tmp_path)
    data = project_data(tmp_path)
    if not overwrite:
        data["project_name"] = "导入草稿"
    data["result_root_directory"] = str(tmp_path / "new-results")
    source = tmp_path / "import.json"
    queue_path = manager.load_queue_catalog()["低噪声基础测试"]["path"]
    if invalid:
        malformed = make_queue_config()
        del malformed[0]["sequence_1"]["acq"]["detail"]["ve_range_index"]
        assert LoadUiConfig.save_data_to_json(malformed, queue_path)
    else:
        second_path = str(Path(queue_path).with_name("队列B.json"))
        assert LoadUiConfig.save_data_to_json(make_queue_config(sample_rate=44100), second_path)
        assert LoadUiConfig.save_data_to_json({"低噪声基础测试": queue_path, "队列B": second_path}, manager.queue_registry_path)
        data["test_groups"][1]["test_conditions"][0]["test_queue"] = "队列B"
    assert LoadUiConfig.save_data_to_json(data, str(source))
    dialog = ProductTestProjectConfigDialog(manager)
    messages, changes, conflicts = [], [], []
    if not invalid:
        capture_conflict_dialog(monkeypatch, conflicts)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(source), ""))
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: messages.append(args[2]))
    monkeypatch.setattr(QMessageBox, "information", lambda *args: pytest.fail("unexpected success"))
    monkeypatch.setattr(QMessageBox, "question", lambda *args: pytest.fail("invalid import must fail before overwrite confirmation"))
    dialog.projects_changed.connect(lambda: changes.append(True))
    try:
        dialog._import_project()
        assert dialog._imported_draft and dialog._dirty and not messages
        contents = dialog.collect_project()
        before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
        assert dialog._save_project() is False
        if invalid:
            assert len(messages) == 1 and "ve_range_index" in messages[0]
        else:
            assert not messages and len(conflicts) == 1
            assert conflicts[0] == ("无法保存", "采样率不一致：", (
                "USB-C输出口/档位1的测试队列“低噪声基础测试”（采样率 48000 Hz，量程 ±10 V）",
                "USB-A输出口/档位1的测试队列“队列B”（采样率 44100 Hz，量程 ±10 V）",
            ))
        assert dialog.current_file is None and dialog._imported_draft and dialog._dirty
        assert dialog.collect_project() == contents
        assert not changes
        assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == before
        assert Path(manager.program_dir, file_name).exists()
        assert not (tmp_path / "new-results").exists()
    finally:
        close_dialog(dialog)


@pytest.mark.parametrize("save_path", ["validation", "stale_catalog", "save_as"])
def test_many_save_errors_show_one_problem_without_writing(app, tmp_path, monkeypatch, save_path):
    manager = make_manager(tmp_path)
    register_queue(manager, duration=100)
    conditions = [
        {
            "condition_name": f"档位{index + 1}",
            "trigger_state": "",
            "test_queue": "低噪声基础测试",
            "segmented_analysis": {
                "mode": "time", "interval_seconds": 60,
                "analysis_seconds": 60, "display_time_unit": "s",
            },
        }
        for index in range(200)
    ]
    dialog = ProductTestProjectConfigDialog(manager)
    dialog._show_project(project_data(tmp_path, conditions), None)
    if save_path == "stale_catalog":
        # Save preflight must reload even when the editor has an older catalog.
        dialog.queue_catalog["低噪声基础测试"]["duration"] = 120
    observed = []

    monkeypatch.setattr(
        QMessageBox, "warning",
        lambda parent, title, message: observed.append((title, message)),
    )
    def accept_name(name_dialog):
        name_dialog.setTextValue("另存配置")
        return QInputDialog.Accepted

    monkeypatch.setattr(QInputDialog, "exec_", accept_name)
    try:
        if save_path == "save_as":
            dialog._save_project_as()
        else:
            assert dialog._save_project(close_dialog=False) is False
        assert observed == [(
            {"validation": "无法保存", "stale_catalog": "无法保存", "save_as": "另存为失败"}[save_path],
            "录音时长必须是分段间隔的整数倍，请调整分段间隔",
        )]
        assert not list(Path(manager.program_dir).glob("*.json"))
    finally:
        close_dialog(dialog)


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
            def accept_name(name_dialog):
                name_dialog.setTextValue("导入副本")
                return QInputDialog.Accepted

            monkeypatch.setattr(QInputDialog, "exec_", accept_name)
            dialog._save_project_as()
            assert dialog.current_file is None
            assert dialog._imported_draft and dialog._dirty
            assert dialog.project_name_input.text() == imported["project_name"]
            assert manager.load_project("导入副本.json")[1][EXPORT_RAW_AUDIO_CSV_KEY] is True
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
            if action != "save-as":
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


@pytest.mark.parametrize("window_width", [1020, 1387, 1700])
def test_segment_summary_fits_with_settings_button(app, tmp_path, window_width):
    manager = make_manager(tmp_path)
    register_queue(manager)
    conditions = [
        {
            "condition_name": f"档位{count}",
            "test_queue": "低噪声基础测试",
            "segmented_analysis": {
                "mode": "time",
                "interval_seconds": 600 / count,
                "display_time_unit": "s",
                "analysis_seconds": 0.01,
            },
        }
        for count in (10, 100, 10000)
    ]
    dialog = ProductTestProjectConfigDialog(manager)
    try:
        dialog._show_project(project_data(tmp_path, conditions), None)
        dialog.resize(window_width, 747)
        if window_width == 1020:
            # Force overflow independently of the platform's font metrics.
            dialog.condition_table.setMaximumWidth(900)
        dialog.show()
        app.processEvents()
        table = dialog.condition_table
        for row, count in enumerate((10, 100, 10000)):
            cell = table.cellWidget(row, 5)
            label = cell.status_label
            button = cell.settings_button
            assert label.text() == f"按时间 · {count} 段"
            assert label.width() >= label.sizeHint().width()
            assert label.geometry().right() < button.geometry().left()
            assert cell.rect().contains(button.geometry())
            assert button.height() == dialog.CONDITION_CONTROL_HEIGHT
        if window_width == 1020:
            assert table.horizontalScrollBar().maximum() > 0
            table.horizontalScrollBar().setValue(table.horizontalScrollBar().maximum())
            app.processEvents()
            cell = table.cellWidget(2, 5)
            assert table.viewport().rect().contains(cell.geometry())
    finally:
        close_dialog(dialog)


def test_segment_column_grows_without_shrinking_after_edit_or_port_switch(app, tmp_path):
    manager = make_manager(tmp_path)
    register_queue(manager)
    dialog = ProductTestProjectConfigDialog(manager)
    try:
        dialog._show_project(project_data(tmp_path), None)
        dialog.show()
        app.processEvents()
        table = dialog.condition_table
        original_width = table.columnWidth(5)
        assert original_width >= 205
        cell = table.cellWidget(0, 5)
        cell.load_settings = {
            "mode": "time", "interval_seconds": 0.06,
            "display_time_unit": "s", "analysis_seconds": 0.01,
        }
        dialog._update_output_load_status(cell)
        app.processEvents()
        expanded_width = table.columnWidth(5)
        assert expanded_width > original_width
        assert cell.status_label.width() >= cell.status_label.sizeHint().width()

        cell.load_settings = {"mode": "none"}
        dialog._update_output_load_status(cell)
        dialog.port_tabs.setCurrentIndex(1)
        dialog.port_tabs.setCurrentIndex(0)
        app.processEvents()
        assert table.columnWidth(5) == expanded_width
        assert table.cellWidget(0, 5).status_label.text() == "未启用"
    finally:
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
    assert dialog.wav_and_csv_radio.text() == "WAV + CSV（ZIP 压缩）"
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


@pytest.mark.parametrize("changed,followup", [(False, None), (True, "save"), (True, "discard")])
def test_save_as_keeps_editor_context_and_original_save_target(
    app, tmp_path, monkeypatch, changed, followup,
):
    manager = make_manager(tmp_path)
    register_queue(manager)
    data = project_data(tmp_path)
    data["project_name"] = "xxx"
    for group in data["test_groups"]:
        group["test_conditions"][0]["input_voltage"] = "10"
    assert manager.save_project(None, data) == (True, "xxx.json")
    original_path = Path(manager.program_dir, "xxx.json")
    original_bytes = original_path.read_bytes()
    dialog = ProductTestProjectConfigDialog(manager)
    messages = []
    changes = []
    dialog.projects_changed.connect(lambda: changes.append("changed"))
    monkeypatch.setattr(QMessageBox, "information", lambda *args: messages.append(args[2]))

    def accept_name(name_dialog):
        name_dialog.setTextValue("555")
        return QInputDialog.Accepted

    monkeypatch.setattr(QInputDialog, "exec_", accept_name)
    try:
        dialog.show()
        dialog.port_tabs.setCurrentIndex(1)
        dialog.condition_table.setCurrentCell(0, 1)
        voltage = dialog.condition_table.cellWidget(0, 7)
        if changed:
            voltage.setFocus()
            voltage.selectAll()
            QTest.keyClicks(voltage, "12")
        assert dialog._dirty is changed
        contents = dialog.collect_project()
        dialog.save_as_btn.click()

        assert dialog.current_file == "xxx.json"
        assert dialog.project_name_input.text() == "xxx"
        assert dialog._dirty is changed and not dialog._imported_draft
        assert dialog.collect_project() == contents
        assert dialog.port_tabs.currentIndex() == 1
        assert dialog.condition_table.currentRow() == 0
        assert dialog.condition_table.cellWidget(0, 7) is voltage
        assert original_path.read_bytes() == original_bytes
        assert manager.load_registry()["active_file"] == "xxx.json"
        copy_path = Path(manager.program_dir, "555.json")
        copy_bytes = copy_path.read_bytes()
        assert manager.load_project("555.json")[1]["test_groups"][1]["test_conditions"][0]["input_voltage"] == ("12" if changed else "10")
        assert changes == ["changed"]
        assert messages == ["已另存为“555”，当前仍在编辑“xxx”。"]

        if followup == "save":
            dialog.save_btn.click()
            assert manager.load_project("xxx.json")[1]["test_groups"][1]["test_conditions"][0]["input_voltage"] == "12"
            assert not dialog.isVisible()
        elif followup == "discard":
            monkeypatch.setattr(QMessageBox, "exec_", lambda _: QMessageBox.Discard)
            dialog.cancel_btn.click()
            assert original_path.read_bytes() == original_bytes
            assert not dialog.isVisible()
        assert copy_path.read_bytes() == copy_bytes
    finally:
        close_dialog(dialog)


@pytest.mark.parametrize("outcome", ["cancel", "empty-name", "same-name", "file-failure", "registry-failure"])
def test_save_as_cancel_or_failure_preserves_draft_and_files(app, tmp_path, monkeypatch, outcome):
    manager = make_manager(tmp_path)
    file_name = prepare_project(manager, tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)
    messages = []
    changes = []
    dialog.projects_changed.connect(lambda: changes.append("changed"))
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: messages.append(args[2]))
    monkeypatch.setattr(QMessageBox, "information", lambda *args: pytest.fail("Save unexpectedly succeeded"))

    def respond(name_dialog):
        name = {"empty-name": "", "same-name": Path(file_name).stem}.get(outcome, "555")
        name_dialog.setTextValue(name)
        return QInputDialog.Rejected if outcome == "cancel" else QInputDialog.Accepted

    monkeypatch.setattr(QInputDialog, "exec_", respond)
    if outcome == "file-failure":
        monkeypatch.setattr(LoadUiConfig, "save_data_to_json", lambda *args: False)
    elif outcome == "registry-failure":
        monkeypatch.setattr(manager, "save_registry", lambda _: False)
    try:
        dialog.port_tabs.setCurrentIndex(1)
        dialog.condition_table.item(0, 1).setText("未保存的工况")
        assert dialog._dirty
        contents = dialog.collect_project()
        before = {p: p.read_bytes() for p in Path(manager.program_dir).glob("*.json")}
        dialog.save_as_btn.click()
        assert dialog.current_file == file_name and dialog._dirty
        assert not dialog._imported_draft
        assert dialog.port_tabs.currentIndex() == 1
        assert dialog.collect_project() == contents
        assert {p: p.read_bytes() for p in Path(manager.program_dir).glob("*.json")} == before
        assert not changes
        assert bool(messages) is (outcome != "cancel")
    finally:
        close_dialog(dialog)


def test_save_as_from_new_draft_does_not_adopt_copy(app, tmp_path, monkeypatch):
    manager = make_manager(tmp_path)
    register_queue(manager)
    dialog = ProductTestProjectConfigDialog(manager)
    data = project_data(tmp_path)
    data["project_name"] = "未保存草稿"
    messages = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args: messages.append(args[2]))

    def accept_name(name_dialog):
        name_dialog.setTextValue("555")
        return QInputDialog.Accepted

    monkeypatch.setattr(QInputDialog, "exec_", accept_name)
    try:
        dialog._show_project(data, None)
        dialog._set_dirty(True)
        dialog.save_as_btn.click()
        assert dialog.current_file is None and dialog._dirty
        assert not dialog._imported_draft
        assert dialog.project_name_input.text() == "未保存草稿"
        assert manager.load_registry()["active_file"] is None
        assert manager.load_project("555.json")[1]["project_name"] == "555"
        assert not Path(manager.program_dir, "未保存草稿.json").exists()
        assert messages == ["已另存为“555”，当前草稿保持不变。"]
    finally:
        close_dialog(dialog)


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
    from ui.config_delete_dialog import ConfigDeleteDialog

    def choose_and_delete(window):
        assert not window.checked_targets()
        window.config_list.item(0).setCheckState(Qt.Checked)
        window.delete_button.click()
        return window.result()

    monkeypatch.setattr(ConfigDeleteDialog, "exec_", choose_and_delete)

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


def test_main_selector_uses_project_name_from_registry(app, tmp_path):
    manager = make_manager(tmp_path)
    Path(manager.program_dir, "PB-A01充电宝.json").write_text("{}", encoding="utf-8")
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
        _get_product_program_manager=lambda: manager,
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


def test_acquisition_late_manager_failure_routes_full_conflict(app, tmp_path, monkeypatch):
    manager = make_manager(tmp_path)
    first_path = register_queue(manager)
    second_path = str(Path(first_path).with_name("队列B.json"))
    assert LoadUiConfig.save_data_to_json(make_queue_config(), second_path)
    assert LoadUiConfig.save_data_to_json({"低噪声基础测试": first_path, "队列B": second_path}, manager.queue_registry_path)
    data = project_data(tmp_path)
    data["test_groups"][1]["test_conditions"][0]["test_queue"] = "队列B"
    success, file_name = manager.save_project(None, data)
    assert success
    dialog = ProductTestProjectConfigDialog(manager)
    dialog.project_name_input.setText("晚到冲突草稿")
    dialog._set_dirty(True)
    assert dialog._dirty
    original_validate = manager.validate_project
    def validate_then_change(*args, **kwargs):
        result = original_validate(*args, **kwargs)
        assert result["can_save"]
        assert LoadUiConfig.save_data_to_json(make_queue_config(sample_rate=44100), second_path)
        return result
    monkeypatch.setattr(manager, "validate_project", validate_then_change)
    conflicts, accepted = [], []
    capture_conflict_dialog(monkeypatch, conflicts)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: pytest.fail("conflict lost its type"))
    monkeypatch.setattr(QMessageBox, "information", lambda *args: pytest.fail("unexpected success"))
    dialog.accepted.connect(lambda: accepted.append(True))
    before = {p: p.read_bytes() for p in Path(manager.program_dir).glob("*.json")}
    contents = dialog.collect_project()
    try:
        assert dialog._save_project() is False
        assert conflicts == [("保存失败", "采样率不一致：", (
            "USB-C输出口/档位1的测试队列“低噪声基础测试”（采样率 48000 Hz，量程 ±10 V）",
            "USB-A输出口/档位1的测试队列“队列B”（采样率 44100 Hz，量程 ±10 V）",
        ))]
        assert dialog.current_file == file_name and dialog._dirty
        assert dialog.collect_project() == contents and not accepted
        assert {p: p.read_bytes() for p in Path(manager.program_dir).glob("*.json")} == before
    finally:
        close_dialog(dialog)


@pytest.mark.parametrize("damage", ["missing", "invalid_json", "groups_null", "group_scalar", "conditions_null", "condition_scalar", "segments_invalid"])
def test_unreadable_project_keeps_management_access(app, tmp_path, monkeypatch, damage):
    manager = make_manager(tmp_path)
    name = prepare_project(manager, tmp_path)
    path = Path(manager.program_dir, name)
    if damage == "missing":
        path.rename(path.with_suffix(".moved"))
    elif damage == "invalid_json":
        path.write_text("{", encoding="utf-8")
    else:
        data = manager.load_project(name)[1]
        if damage == "groups_null":
            data["test_groups"] = None
        elif damage == "group_scalar":
            data["test_groups"] = [1]
        elif damage == "conditions_null":
            data["test_groups"][0]["test_conditions"] = None
        elif damage == "condition_scalar":
            data["test_groups"][0]["test_conditions"] = [1]
        else:
            data["test_groups"][0]["test_conditions"][0]["segmented_analysis"] = {"mode": []}
        assert LoadUiConfig.save_data_to_json(data, str(path))
    before = {p.name: p.read_bytes() for p in Path(manager.program_dir).iterdir()}
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    dialog = ProductTestProjectConfigDialog(manager)
    try:
        assert not dialog.initial_load_succeeded
        assert dialog.current_file is None
        assert dialog.project_name_input.text() == ""
        blank_draft = damage == "missing"
        assert dialog.condition_table.rowCount() == (1 if blank_draft else 0)
        assert dialog.project_name_input.isEnabled() is blank_draft
        assert dialog.save_btn.isEnabled() is blank_draft
        assert dialog.save_as_btn.isEnabled() is blank_draft
        assert dialog.load_status_label.isHidden() is blank_draft
        assert dialog.new_project_btn.isEnabled()
        assert dialog.import_project_btn.isEnabled()
        assert dialog.delete_project_btn.isEnabled()
        if not blank_draft:
            assert dialog._save_project(close_dialog=False) is False
        assert warnings == []
        dialog.reject()
        assert before == {p.name: p.read_bytes() for p in Path(manager.program_dir).iterdir()}
    finally:
        close_dialog(dialog)


@pytest.mark.parametrize("action", ["new", "import", "cancel_import", "delete"])
@pytest.mark.parametrize("damage", ["missing", "invalid_json"])
def test_failed_load_reuses_existing_actions(app, tmp_path, monkeypatch, action, damage):
    from ui.config_delete_dialog import ConfigDeleteDialog
    manager = make_manager(tmp_path)
    name = prepare_project(manager, tmp_path)
    path = Path(manager.program_dir, name)
    moved = path.with_suffix(".moved")
    path.rename(moved)
    if damage == "invalid_json":
        path.write_text("{", encoding="utf-8")
    before = Path(manager.registry_path).read_bytes()
    monkeypatch.setattr(QMessageBox, "warning", lambda *_: None)
    monkeypatch.setattr(QMessageBox, "information", lambda *_: None)
    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.Yes)
    dialog = ProductTestProjectConfigDialog(manager)
    try:
        assert dialog.save_btn.isEnabled() is (damage == "missing")
        if action == "delete":
            def remove_missing(window):
                item = window.config_list.item(0)
                assert item.data(Qt.UserRole).remove_file is (damage != "missing")
                item.setCheckState(Qt.Checked)
                window.delete_button.click()
                return window.result()
            monkeypatch.setattr(ConfigDeleteDialog, "exec_", remove_missing)
            dialog.delete_project_btn.click()
            assert manager.load_registry() == {"active_file": None, "configs": []}
            assert moved.exists()
            assert dialog.save_btn.isEnabled() is (damage == "missing")
        elif action == "new":
            dialog.new_project_btn.click()
            assert dialog.current_file is None
            assert dialog.project_name_input.isEnabled()
            assert dialog.save_btn.isEnabled()
            assert dialog.save_as_btn.isEnabled()
            assert dialog.condition_table.rowCount() == 1
            assert Path(manager.registry_path).read_bytes() == before
        else:
            monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_: (str(moved) if action == "import" else "", ""))
            dialog.import_project_btn.click()
            assert Path(manager.registry_path).read_bytes() == before
            if action == "import":
                assert dialog.current_file is None and dialog._imported_draft
                assert dialog.save_btn.isEnabled()
                assert dialog._save_project(close_dialog=False)
                assert path.exists()
                assert manager.load_registry()["active_file"] == name
            else:
                assert dialog.save_btn.isEnabled() is (damage == "missing")
                assert dialog.condition_table.rowCount() == (1 if damage == "missing" else 0)
    finally:
        close_dialog(dialog)


def test_missing_queue_does_not_prevent_editing_product(app, tmp_path):
    manager = make_manager(tmp_path)
    name = prepare_project(manager, tmp_path)
    assert LoadUiConfig.save_data_to_json({}, manager.queue_registry_path)
    dialog = ProductTestProjectConfigDialog(manager)
    try:
        assert dialog.initial_load_succeeded
        assert dialog.current_file == name
        assert dialog.project_name_input.isEnabled()
        assert dialog.save_btn.isEnabled()
        assert dialog.condition_table.rowCount() > 0
    finally:
        close_dialog(dialog)


def test_explicit_structurally_invalid_load_preserves_current_draft(app, tmp_path, monkeypatch):
    manager = make_manager(tmp_path)
    name = prepare_project(manager, tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)
    try:
        dialog.project_name_input.setText("Unsaved draft")
        dialog._set_dirty(True)
        bad = manager.default_project()
        bad["test_groups"] = None
        assert LoadUiConfig.save_data_to_json(bad, str(Path(manager.program_dir, "bad.json")))
        warnings = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
        assert dialog._load_project("bad.json") is False
        assert dialog.current_file == name
        assert dialog.project_name_input.text() == "Unsaved draft"
        assert dialog._dirty
        assert len(warnings) == 1
    finally:
        close_dialog(dialog)


def test_missing_file_blank_draft_cannot_silently_replace_old_registration(app, tmp_path, monkeypatch):
    manager = make_manager(tmp_path)
    name = prepare_project(manager, tmp_path)
    path = Path(manager.program_dir, name)
    path.rename(path.with_suffix(".moved"))
    before = Path(manager.registry_path).read_bytes()
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    monkeypatch.setattr(QMessageBox, "information", lambda *_: None)
    dialog = ProductTestProjectConfigDialog(manager)
    try:
        assert dialog.current_file is None and not dialog._imported_draft
        dialog.project_name_input.setText(Path(name).stem)
        dialog.result_root_input.setText(str(tmp_path / "results"))
        combo, _ = dialog._queue_controls_for_row(0)
        combo.setCurrentIndex(1)
        assert dialog._save_project(close_dialog=False) is False
        assert "已存在" in warnings[-1]
        assert not path.exists()
        assert Path(manager.registry_path).read_bytes() == before
        dialog.project_name_input.setText("Recovered")
        assert dialog._save_project(close_dialog=False)
        assert Path(manager.program_dir, "Recovered.json").exists()
        assert manager.load_registry()["active_file"] == name
        assert not path.exists()
    finally:
        close_dialog(dialog)
