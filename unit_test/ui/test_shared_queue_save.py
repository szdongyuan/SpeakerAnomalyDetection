import copy
import json

import pytest
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox, QPushButton

from base.load_config import LoadUiConfig
from base.sequence_queue_references import QueueReferenceDraft, SequenceQueueReferenceScanner
from ui.acquisition_config_window import RecordConfigWindow
from ui.operation_sequence import AnalysisModelSelect
from unit_test.base.test_sequence_queue_references import project, write_json
from unit_test.base.ve3668n_fakes import device_info


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def recording_only_notices(monkeypatch):
    notices = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args: notices.append(args[2]))
    return notices


@pytest.fixture
def editor(app, tmp_path, monkeypatch):
    products = tmp_path / "products"
    registry = products / "registry.json"
    queues = tmp_path / "queues.json"
    target = tmp_path / "Q.json"
    payload = [{"seq1": {"acq": {"name": "录制音频", "mode": "RECORD_ONLY",
                                "detail": {"sample_rate": 44100, "total_time": 4.0,
                                           "use_streaming_recording": False,
                                           "recording_preview_time_mode": "relative_latest",
                                           "recording_root": ""}},
                         "analysis_list": {"display_sequence": [],
                                           "auto_analysis": True}}}]
    write_json(target, payload)
    write_json(queues, {"Q": str(target), "using_config_path": str(target)})
    write_json(registry, {"configs": [{"file": "one.json", "project_name": "Product"}]})
    write_json(products / "one.json", project("Q", "Q"))
    monkeypatch.setattr("base.load_config.SEQUENCE_CONFIG_REGISTRY_PATH", str(queues))
    window = AnalysisModelSelect(str(target))
    window.reference_scanner = SequenceQueueReferenceScanner(products, registry, queues)
    window.confirm_shared_save = lambda path, result: False
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    yield window, target, queues, products, warnings
    window._allow_close = True
    window.close()


def change(window):
    window.select_list.config[0].detail["sample_rate"] = 48000
    window.select_list._notify_config_changed()


@pytest.mark.parametrize("action", ["save", "save_as"])
@pytest.mark.parametrize("has_analysis", [False, True])
def test_explicit_save_explains_recording_only_product_queue_limit(
    editor, monkeypatch, recording_only_notices, action, has_analysis
):
    window, target, *_ = editor
    config = window.select_list.config[0]
    config.detail["sample_rate"] = 48000
    if has_analysis:
        config.analysis_list["spl"] = {"type": "SPL", "limit_checked": True}
        config.display_sequence = ["spl"]
    window.confirm_shared_save = lambda *args: True
    if action == "save_as":
        target = target.with_name("copy.json")
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
        window.save_btn_clicked()
    else:
        window.ok_btn_clicked()

    saved = json.loads(target.read_text(encoding="utf-8"))[0]["seq1"]
    assert saved["acq"]["detail"]["sample_rate"] == 48000
    assert bool(saved["analysis_list"]["display_sequence"]) is has_analysis
    if has_analysis:
        assert not recording_only_notices
    else:
        assert len(recording_only_notices) == 1
        assert "未配置分析项" in recording_only_notices[0]
        assert "添加分析项后，才会显示在产品测试配置中" in recording_only_notices[0]


def test_explicit_save_still_explains_already_autosaved_recording_only_queue(
    editor, recording_only_notices
):
    window, target, _, products, _ = editor
    write_json(products / "one.json", project("Q"))
    change(window)
    assert not window.dirty
    assert not recording_only_notices
    before = target.read_bytes()

    window.ok_btn_clicked()

    assert target.read_bytes() == before
    assert len(recording_only_notices) == 1


@pytest.mark.parametrize("next_action,notice_count", [
    ("save", 1), ("edit_then_save", 2), ("save_as", 2),
])
def test_recording_only_notice_is_not_repeated_for_same_saved_queue(
    editor, monkeypatch, recording_only_notices, next_action, notice_count
):
    window, target, *_ = editor
    copy_path = target.with_name("copy.json")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(copy_path), ""))
    window.save_btn_clicked()
    assert len(recording_only_notices) == 1
    saved = copy_path.read_bytes()

    if next_action == "save_as":
        other_path = target.with_name("other.json")
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(other_path), ""))
        window.save_btn_clicked()
        assert other_path.read_bytes() == saved
    else:
        if next_action == "edit_then_save":
            change(window)
        window.ok_btn_clicked()
        assert not window.isVisible()

    assert len(recording_only_notices) == notice_count
    if next_action == "save":
        assert copy_path.read_bytes() == saved


@pytest.mark.parametrize("outcome", ["cancelled", "failed", "deferred"])
def test_unsuccessful_save_does_not_report_recording_only_queue_saved(
    editor, monkeypatch, recording_only_notices, outcome
):
    window, target, *_ = editor
    window.select_list.config[0].detail["sample_rate"] = 48000
    window.confirm_shared_save = lambda *args: outcome != "cancelled"
    if outcome == "failed":
        monkeypatch.setattr(LoadUiConfig, "save_sequence_config_to_json", lambda *args: False)

    result = window._save_queue(str(target), explicit=outcome != "deferred")

    assert result == outcome
    assert not recording_only_notices


@pytest.fixture
def save_selection(monkeypatch):
    calls = []

    def select(path):
        def execute(dialog):
            calls.append(dialog.windowTitle())
            return QDialog.Accepted if path is not None else QDialog.Rejected
        monkeypatch.setattr(QFileDialog, "exec_", execute)
        monkeypatch.setattr(QFileDialog, "selectedFiles", lambda dialog: [str(path)] if path else [])
        return calls

    return select


def test_first_save_writes_selected_file(editor, monkeypatch, tmp_path, save_selection):
    window, original, registry, _, _ = editor
    placeholder = tmp_path / "ui" / "ui_config" / "none_path.json"
    selected = tmp_path / "queues" / "Chosen.json"
    before = original.read_bytes()
    window.using_config_path = str(placeholder)
    change(window)
    monkeypatch.setattr("ui.operation_sequence.DEFAULT_DIR", tmp_path.as_posix() + "/")
    calls = save_selection(selected)
    window.show()
    window.ok_btn_clicked()
    assert selected.exists()
    assert calls == ["保存测试队列"]
    assert not placeholder.with_name("sequence_config.json").exists()
    assert original.read_bytes() == before
    assert json.loads(selected.read_text(encoding="utf-8"))[0]["seq1"]["acq"]["detail"]["sample_rate"] == 48000
    assert json.loads(registry.read_text(encoding="utf-8"))["using_config_path"] == str(selected)
    assert window.using_config_path == str(selected)
    assert not window.isVisible()


@pytest.mark.parametrize("operation", ["ok_btn_clicked", "save_btn_clicked"])
def test_save_dialog_default_directory_is_canonical(editor, monkeypatch, tmp_path, save_selection, operation):
    window, *_ = editor
    (tmp_path / "consts").mkdir()
    monkeypatch.setattr("ui.operation_sequence.DEFAULT_DIR", tmp_path.as_posix() + "/consts/../")
    expected_dir = tmp_path / "ui" / "ui_config" / "analysis_sequence_config"
    window.using_config_path = None
    change(window)
    draft = window.format_config_data(window.select_list.config)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    directories = []
    real_set_directory = QFileDialog.setDirectory

    def set_directory(dialog, directory):
        # Capture the exact native-dialog input: directory() normalizes it in Qt.
        directories.append(directory)
        real_set_directory(dialog, directory)

    def save_as_dialog(parent, title, directory, **kwargs):
        directories.append(directory)
        return "", ""

    monkeypatch.setattr(QFileDialog, "setDirectory", set_directory)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", save_as_dialog)
    save_selection(None)
    window.show()
    getattr(window, operation)()

    assert directories == [str(expected_dir)]
    assert window.isVisible() and window.dirty
    assert window.using_config_path is None
    assert window.format_config_data(window.select_list.config) == draft
    assert {path: path.read_bytes() for path in tmp_path.rglob("*.json")} == before


def test_first_save_directory_failure_retains_draft(editor, monkeypatch, tmp_path, save_selection):
    window, original, registry, _, warnings = editor
    before = original.read_bytes(), registry.read_bytes()
    window.using_config_path = None
    change(window)
    calls = save_selection(tmp_path / "Chosen.json")
    # A file in place of the default directory produces a real filesystem error.
    write_json(tmp_path / "ui", {})
    monkeypatch.setattr("ui.operation_sequence.DEFAULT_DIR", tmp_path.as_posix() + "/")
    window.show()
    window.ok_btn_clicked()
    assert warnings and not calls
    assert window.using_config_path is None
    assert window.dirty and window.isVisible()
    assert (original.read_bytes(), registry.read_bytes()) == before


@pytest.mark.parametrize("path", [None, "", "none_path.json", "ui/ui_config/none_path.json", r"ui\ui_config\none_path.json"])
def test_first_save_path_states_and_second_save(editor, monkeypatch, tmp_path, save_selection, path):
    window, original, registry, _, _ = editor
    before = original.read_bytes()
    window.using_config_path = path
    window.select_list.config[0].detail["sample_rate"] = 48000
    window.dirty = True
    selected = tmp_path / "Chosen.json"
    monkeypatch.setattr("ui.operation_sequence.DEFAULT_DIR", tmp_path.as_posix() + "/")
    calls = save_selection(selected)
    window.ok_btn_clicked()
    window.select_list.config[0].detail["sample_rate"] = 96000
    window.ok_btn_clicked()
    assert calls == ["保存测试队列"]
    assert window.using_config_path == str(selected)
    assert original.read_bytes() == before
    assert json.loads(selected.read_text(encoding="utf-8"))[0]["seq1"]["acq"]["detail"]["sample_rate"] == 96000
    assert json.loads(registry.read_text(encoding="utf-8"))["using_config_path"] == str(selected)
    assert "Chosen" in window.current_config_label.text()


@pytest.mark.parametrize("dirty", [False, True])
def test_first_save_cancel_preserves_state(editor, monkeypatch, tmp_path, save_selection, dirty):
    window, original, registry, _, _ = editor
    window.using_config_path = None
    window.dirty = dirty
    draft = window.format_config_data(window.select_list.config)
    before = original.read_bytes(), registry.read_bytes()
    monkeypatch.setattr("ui.operation_sequence.DEFAULT_DIR", tmp_path.as_posix() + "/")
    calls = save_selection(None)
    window.show()
    window.ok_btn_clicked()
    assert calls == ["保存测试队列"]
    assert window.isVisible() and window.dirty is dirty
    assert window.using_config_path is None
    assert window.format_config_data(window.select_list.config) == draft
    assert (original.read_bytes(), registry.read_bytes()) == before
    assert not (tmp_path / "ui/ui_config/sequence_config.json").exists()


@pytest.mark.parametrize("invalid", ["empty", "rate"])
@pytest.mark.parametrize("prompt", [False, True])
def test_first_save_validates_before_selection(editor, monkeypatch, save_selection, invalid, prompt):
    window, original, registry, _, warnings = editor
    window.using_config_path = None
    window.dirty = True
    before = original.read_bytes(), registry.read_bytes()
    if invalid == "empty":
        window.select_list.clear_option_list()
    else:
        window.select_list.config[0].detail["sample_rate"] = "invalid"
    calls = save_selection(None)
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.Save)
    window.show()
    if prompt:
        window.close()
    else:
        window.ok_btn_clicked()
    assert warnings and not calls
    assert window.isVisible() and window.dirty
    assert window.using_config_path is None
    assert (original.read_bytes(), registry.read_bytes()) == before


@pytest.mark.parametrize("preselected", [False, True])
def test_first_save_formal_target_never_prompts(editor, save_selection, preselected):
    window, original, registry, _, _ = editor
    target = original.with_name("New.json") if preselected else original
    window.using_config_path = str(target)
    window._new_target_path_selected = preselected
    window.confirm_shared_save = lambda *a: True
    before = json.loads(registry.read_text(encoding="utf-8"))
    calls = save_selection(None)
    window.select_list.config[0].detail["sample_rate"] = 48000
    window.ok_btn_clicked()
    assert not calls and target.exists()
    assert window.using_config_path == str(target)
    assert json.loads(registry.read_text(encoding="utf-8"))["using_config_path"] == before["using_config_path"]
    assert json.loads(target.read_text(encoding="utf-8"))[0]["seq1"]["acq"]["detail"]["sample_rate"] == 48000
    persisted = target.read_bytes()
    window.select_list.config[0].detail["sample_rate"] = 96000
    window.select_list._notify_config_changed()
    # Shared files and preselected new targets retain their deferred autosave policy.
    assert target.read_bytes() == persisted
    assert window.dirty


@pytest.mark.parametrize("outcome", ["saved", "selection_cancel", "shared_cancel", "write_failure"])
@pytest.mark.parametrize("operation", ["ok_btn_clicked", "close", "load_btn_clicked", "new_btn_clicked"])
def test_first_save_transition_guards(editor, monkeypatch, tmp_path, operation, outcome):
    window, original, registry, _, warnings = editor
    window.using_config_path = None
    change(window)
    before = original.read_bytes(), registry.read_bytes()
    selected = original if outcome == "shared_cancel" else tmp_path / "Chosen.json"
    calls, imports, confirmations = [], [], []
    monkeypatch.setattr("ui.operation_sequence.DEFAULT_DIR", tmp_path.as_posix() + "/")
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.Save)
    def execute(dialog):
        calls.append(dialog.windowTitle())
        if dialog.windowTitle() == "新建配置文件" or outcome == "selection_cancel":
            return QDialog.Rejected
        return QDialog.Accepted
    monkeypatch.setattr(QFileDialog, "exec_", execute)
    monkeypatch.setattr(QFileDialog, "selectedFiles", lambda *a: [str(selected)])
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: imports.append(True) or ("", ""))
    window.confirm_shared_save = lambda path, result: confirmations.append(path) or False
    if outcome == "write_failure":
        monkeypatch.setattr(LoadUiConfig, "save_sequence_config_to_json", lambda *a: False)
    window.show()
    getattr(window, operation)()
    assert calls[0] == "保存测试队列"
    assert bool(imports) is (operation == "load_btn_clicked" and outcome == "saved")
    assert ("新建配置文件" in calls) is (operation == "new_btn_clicked" and outcome == "saved")
    assert confirmations == ([str(original)] if outcome == "shared_cancel" else [])
    assert window.select_list.config[0].detail["sample_rate"] == 48000
    if outcome == "saved":
        assert selected.exists() and not window.dirty
        assert window.using_config_path == str(selected)
        assert window.isVisible() is (operation in {"load_btn_clicked", "new_btn_clicked"})
    else:
        assert window.isVisible() and window.dirty
        assert window.using_config_path is None
        assert (original.read_bytes(), registry.read_bytes()) == before
        assert warnings if outcome == "write_failure" else not warnings


def test_first_save_registration_retry_does_not_rewrite(editor, monkeypatch, tmp_path, save_selection):
    window, original, registry, _, warnings = editor
    window.using_config_path = None
    change(window)
    selected = tmp_path / "Chosen.json"
    monkeypatch.setattr("ui.operation_sequence.DEFAULT_DIR", tmp_path.as_posix() + "/")
    calls = save_selection(selected)
    before = registry.read_bytes()
    real_register = LoadUiConfig.save_data_to_json
    real_save = LoadUiConfig.save_sequence_config_to_json
    writes = []
    def save(payload, path):
        writes.append(path)
        return real_save(payload, path)
    monkeypatch.setattr(LoadUiConfig, "save_sequence_config_to_json", save)
    monkeypatch.setattr(LoadUiConfig, "save_data_to_json", lambda data, path, *a: False if str(path) == str(registry) else real_register(data, path, *a))
    window.show()
    window.ok_btn_clicked()
    assert warnings and window.dirty and window.isVisible()
    assert window.using_config_path is None
    assert selected.exists() and str(selected) in window._pending_registration
    assert registry.read_bytes() == before
    monkeypatch.setattr(LoadUiConfig, "save_data_to_json", real_register)
    window.ok_btn_clicked()
    assert len(calls) == 2 and writes == [str(selected)]
    assert not window.dirty and not window.isVisible() and not window._pending_registration
    assert window.using_config_path == str(selected)
    registered = json.loads(registry.read_text(encoding="utf-8"))
    assert registered["using_config_path"] == registered["默认配置"] == str(selected)


def test_first_save_missing_template_can_add_recording(editor, monkeypatch, tmp_path, save_selection):
    _, original, registry, _, warnings = editor
    monkeypatch.setattr("ui.operation_sequence.DEFAULT_DIR", tmp_path.as_posix() + "/")
    window = AnalysisModelSelect(None, reference_scanner=editor[0].reference_scanner)
    selected = tmp_path / "Chosen.json"
    calls = save_selection(selected)
    before = original.read_bytes(), registry.read_bytes()
    try:
        window.show()
        assert window.select_list.config == []
        window.ok_btn_clicked()
        assert warnings == ["没有配置测试内容"] and not calls
        assert window.isVisible()
        assert (original.read_bytes(), registry.read_bytes()) == before
        window.select_list.set_sound_item("录制音频")
        window.ok_btn_clicked()
        assert calls == ["保存测试队列"]
        assert selected.exists() and window.using_config_path == str(selected)
        assert not window.isVisible()
    finally:
        window._allow_close = True
        window.close()


@pytest.mark.parametrize("existing", [False, True])
def test_first_save_qt_suffix_and_overwrite_cancel(editor, monkeypatch, tmp_path, existing):
    window, original, registry, _, _ = editor
    window.using_config_path = None
    change(window)
    monkeypatch.setattr("ui.operation_sequence.DEFAULT_DIR", tmp_path.as_posix() + "/")
    selected = tmp_path / "Chosen.json"
    if existing:
        write_json(selected, [{"untouched": True}])
    before = original.read_bytes(), registry.read_bytes()
    selected_before = selected.read_bytes() if existing else None
    real_exec = QFileDialog.exec_
    observations = []

    def execute(dialog):
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setDirectory(str(tmp_path))
        dialog.selectFile("Chosen")
        observations.append((dialog.acceptMode(), dialog.fileMode(), dialog.defaultSuffix(),
                             dialog.testOption(QFileDialog.DontConfirmOverwrite)))
        timer = QTimer(dialog)
        timer.setSingleShot(True)
        timer.timeout.connect(dialog.reject)
        timer.start(3000)
        watcher = QTimer(dialog)
        def decline_overwrite():
            modal = QApplication.activeModalWidget()
            if isinstance(modal, QMessageBox):
                observations.append("overwrite")
                modal.done(QMessageBox.No)
                QTimer.singleShot(0, dialog.reject)
        watcher.timeout.connect(decline_overwrite)
        watcher.start(10)
        QTimer.singleShot(0, dialog.accept)
        result = real_exec(dialog)
        timer.stop()
        watcher.stop()
        return result

    monkeypatch.setattr(QFileDialog, "exec_", execute)
    window.show()
    window.ok_btn_clicked()
    assert observations[0] == (QFileDialog.AcceptSave, QFileDialog.AnyFile, "json", False)
    assert not (tmp_path / "Chosen").exists()
    assert original.read_bytes() == before[0]
    if existing:
        assert observations[1:] == ["overwrite"]
        assert selected.read_bytes() == selected_before
        assert registry.read_bytes() == before[1]
        assert window.isVisible() and window.dirty and window.using_config_path is None
    else:
        assert selected.exists() and not window.isVisible()
        assert window.using_config_path.replace("\\", "/") == selected.as_posix()


@pytest.fixture
def recording_dialog(editor, monkeypatch):
    window, target, *_ = editor
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload[0]["seq1"]["acq"].update(name="录制音频", mode="RECORD_ONLY")
    write_json(target, payload)
    window.select_list.load_model_config(str(target))
    window.select_list.mic = {"name": "Soundcard"}
    window.select_list.speaker = {"name": "Output"}

    def edit(expected, value):
        def execute(dialog):
            assert isinstance(dialog, RecordConfigWindow)
            assert dialog.startup_trim_input.value() == expected
            dialog.startup_trim_input.setValue(value)
            dialog.on_click_ok_btn()
            assert dialog.final_data is not None
            return dialog.final_data

        monkeypatch.setattr(RecordConfigWindow, "exec", execute)
        window.select_list.show_dialog(window.select_list.config[0].name)

    return edit


@pytest.mark.parametrize("vk", [False, True])
def test_startup_delay_dialog_shared_save_and_reload(editor, recording_dialog, vk):
    window, target, registry, products, _ = editor
    if vk:
        window.select_list.mic = device_info()
    before = target.read_bytes(), registry.read_bytes()
    product_before = (products / "one.json").read_bytes()
    recording_dialog(100, 123)
    assert window.select_list.config[0].detail["startup_trim_ms"] == 123
    assert window.dirty
    assert (target.read_bytes(), registry.read_bytes()) == before
    window.ok_btn_clicked()
    assert (target.read_bytes(), registry.read_bytes()) == before

    confirmations = []
    window.confirm_shared_save = lambda path, result: confirmations.append(path) or True
    window.ok_btn_clicked()
    for expected, value in ((123, 0), (0, 0)):
        saved = json.loads(target.read_text(encoding="utf-8"))[0]["seq1"]["acq"]["detail"]
        assert saved["startup_trim_ms"] == expected
        assert type(saved["startup_trim_ms"]) is int
        window.select_list.load_model_config(str(target))
        recording_dialog(expected, value)
        window.ok_btn_clicked()
    assert confirmations == [str(target), str(target)]
    assert registry.read_bytes() == before[1]
    assert (products / "one.json").read_bytes() == product_before


def test_shared_notifier_defers_and_ok_cancel_preserves_draft(editor):
    window, target, registry, _, _ = editor
    before = target.read_bytes(), registry.read_bytes()
    change(window)
    assert (target.read_bytes(), registry.read_bytes()) == before
    assert window.dirty
    window.show()
    window.ok_btn_clicked()
    assert window.isVisible()
    assert window.select_list.config[0].detail["sample_rate"] == 48000
    assert (target.read_bytes(), registry.read_bytes()) == before


def test_confirm_writes_once_and_unchanged_does_not_warn(editor, monkeypatch):
    window, target, _, _, _ = editor
    change(window)
    confirmations, writes = [], []
    real_save = LoadUiConfig.save_sequence_config_to_json
    def save(payload, path):
        writes.append(path)
        return real_save(payload, path)
    monkeypatch.setattr(LoadUiConfig, "save_sequence_config_to_json", save)
    window.confirm_shared_save = lambda path, result: confirmations.append(result) or True
    window.ok_btn_clicked()
    assert writes == [str(target)]
    assert len(confirmations) == 1
    window.ok_btn_clicked()
    assert len(confirmations) == 1
    assert writes == [str(target)]


@pytest.mark.parametrize("count", [0, 1])
def test_single_or_unreferenced_autosave_is_preserved(editor, count):
    window, target, _, products, _ = editor
    write_json(products / "one.json", project(*(["Q"] * count)))
    change(window)
    assert json.loads(target.read_text(encoding="utf-8"))[0]["seq1"]["acq"]["detail"]["sample_rate"] == 48000
    assert not window.dirty


def test_payload_formatting_does_not_mutate_draft(editor):
    window, *_ = editor
    item = window.select_list.config[0]
    item.analysis_list["old"] = {"type": "retired", "golden_sample_checked": True}
    before = copy.deepcopy(item.config_info)
    window.format_config_data(window.select_list.config)
    assert item.config_info == before


@pytest.mark.parametrize("rate", [96000, 48000.5, True, None, "broken"])
def test_recording_load_preserves_owned_detail_for_repair_and_save(editor, rate):
    window, target, _, products, _ = editor
    payload = json.loads(target.read_text(encoding="utf-8"))
    detail = {"sample_rate": rate, "ve_range_index": 5, "total_time": 4.0,
              "audio_validation": {"enabled": False}, "startup_trim_ms": 123}
    payload[0]["seq1"]["acq"].update(name="录制音频", mode="RECORD_ONLY", detail=detail)
    write_json(target, payload)
    window.select_list.load_model_config(str(target))
    loaded = window.select_list.config[0].detail
    for key, value in detail.items():
        assert loaded[key] == value
        assert type(loaded[key]) is type(value)
    if rate == 96000:
        write_json(products / "one.json", project("Q"))
        window.select_list._notify_config_changed()
        saved = json.loads(target.read_text(encoding="utf-8"))[0]["seq1"]["acq"]["detail"]
        assert all(saved[key] == value for key, value in detail.items())


@pytest.mark.parametrize("overwrite", [False, True])
def test_save_as_checks_actual_target_only(editor, recording_dialog, monkeypatch, overwrite):
    window, target, registry, products, _ = editor
    new_target = target.with_name("R.json")
    if overwrite:
        write_json(new_target, [])
        write_json(registry, {"Q": str(target), "R": str(new_target)})
        write_json(products / "one.json", project("Q", "Q", "R", "R"))
    before = target.read_bytes()
    recording_dialog(100, 123)
    change(window)
    confirmations = []
    window.confirm_shared_save = lambda path, result: confirmations.append((path, result)) or True
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(new_target), ""))
    window.save_btn_clicked()
    assert target.read_bytes() == before
    assert json.loads(new_target.read_text(encoding="utf-8"))[0]["seq1"]["acq"]["detail"]["sample_rate"] == 48000
    saved = json.loads(new_target.read_text(encoding="utf-8"))[0]["seq1"]["acq"]["detail"]
    assert saved["startup_trim_ms"] == 123
    assert type(saved["startup_trim_ms"]) is int
    if overwrite:
        assert confirmations[0][0] == str(new_target)
        assert [r.condition_name for r in confirmations[0][1].references] == ["C", "D"]
    else:
        assert confirmations == []


def test_overwrite_cancel_changes_neither_target_nor_registry(editor, monkeypatch):
    window, target, registry, _, _ = editor
    before = target.read_bytes(), registry.read_bytes()
    change(window)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    window.save_btn_clicked()
    assert (target.read_bytes(), registry.read_bytes()) == before
    assert window.dirty


@pytest.mark.parametrize("choice,confirm,closes", [
    (QMessageBox.Save, True, True), (QMessageBox.Save, False, False),
    (QMessageBox.Discard, False, True), (QMessageBox.Cancel, False, False),
])
def test_close_unsaved_draft(editor, monkeypatch, choice, confirm, closes):
    window, target, _, _, _ = editor
    before = target.read_bytes()
    change(window)
    window.show()
    prompts, warnings = [], []
    monkeypatch.setattr(QMessageBox, "question", lambda *a: prompts.append(a) or choice)
    window.confirm_shared_save = lambda *a: warnings.append(a) or confirm
    window.close()
    assert window.isVisible() is not closes
    assert len(prompts) == 1
    assert len(warnings) == (1 if choice == QMessageBox.Save else 0)
    if choice == QMessageBox.Save and confirm:
        assert target.read_bytes() != before
    else:
        assert target.read_bytes() == before


@pytest.mark.parametrize("operation", ["load_btn_clicked", "new_btn_clicked"])
@pytest.mark.parametrize("choice,confirm,proceed", [
    (QMessageBox.Save, True, True), (QMessageBox.Save, False, False),
    (QMessageBox.Discard, False, True), (QMessageBox.Cancel, False, False),
])
def test_switch_unsaved_draft(editor, monkeypatch, operation, choice, confirm, proceed):
    window, target, _, _, _ = editor
    before = target.read_bytes()
    change(window)
    next_target = target.with_name("next.json")
    write_json(next_target, [])
    selected = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a: choice)
    window.confirm_shared_save = lambda *a: confirm
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: selected.append(True) or (str(next_target), ""))
    monkeypatch.setattr(QFileDialog, "exec_", lambda *a: selected.append(True) or QDialog.Accepted)
    monkeypatch.setattr(QFileDialog, "selectedFiles", lambda *a: [str(next_target)])
    getattr(window, operation)()
    assert bool(selected) is proceed
    assert window.using_config_path == (str(next_target).replace("\\", "/") if proceed else str(target))
    assert (target.read_bytes() != before) is (choice == QMessageBox.Save and confirm)


def test_incomplete_scan_defers_then_requires_explicit_decision(editor):
    window, target, registry, products, _ = editor
    product_registry = products / "registry.json"
    write_json(product_registry, {"configs": [{"file": "one.json"}, {"file": "bad.json", "project_name": "Broken"}]})
    (products / "bad.json").write_text("{", encoding="utf-8")
    before = target.read_bytes(), registry.read_bytes()
    change(window)
    assert (target.read_bytes(), registry.read_bytes()) == before
    confirmations = []
    window.confirm_shared_save = lambda path, result: confirmations.append(result) or False
    window.ok_btn_clicked()
    assert len(confirmations[0].references) == 2
    assert confirmations[0].issues[0].product_name == "Broken"
    assert (target.read_bytes(), registry.read_bytes()) == before
    window.confirm_shared_save = lambda *a: True
    window.ok_btn_clicked()
    assert target.read_bytes() != before[0]


def test_each_save_rescans_disk_and_parent_union(editor):
    window, target, _, products, _ = editor
    draft = QueueReferenceDraft(project("Q", ""), products / "one.json")
    window.parent_draft_provider = lambda: (draft,)
    change(window)
    confirmations = []
    window.confirm_shared_save = lambda path, result: confirmations.append(result) or True
    window.ok_btn_clicked()
    assert [r.sources for r in confirmations[0].references] == [{"saved", "draft"}, {"saved"}]
    # Parent cancellation leaves its original bytes; later saves still see A/B.
    window.parent_draft_provider = None
    window.select_list.config[0].detail["sample_rate"] = 96000
    window.ok_btn_clicked()
    assert len(confirmations[1].references) == 2
    write_json(products / "one.json", project("Q"))
    window.select_list.config[0].detail["sample_rate"] = 32000
    window.select_list._notify_config_changed()
    assert not window.dirty
    assert len(confirmations) == 2


def test_failed_queue_write_preserves_registry_draft_and_window(editor, monkeypatch):
    window, target, registry, _, errors = editor
    before = target.read_bytes(), registry.read_bytes()
    change(window)
    window.show()
    window.confirm_shared_save = lambda *a: True
    monkeypatch.setattr(LoadUiConfig, "save_sequence_config_to_json", lambda *a: False)
    window.ok_btn_clicked()
    assert window.isVisible() and window.dirty and errors
    assert (target.read_bytes(), registry.read_bytes()) == before


def test_confirmation_reentrant_notifier_cannot_write_early(editor, monkeypatch):
    window, target, _, _, _ = editor
    before = target.read_bytes()
    change(window)
    def confirm(*args):
        window.select_list._notify_config_changed()
        assert target.read_bytes() == before
        return True
    window.confirm_shared_save = confirm
    window.ok_btn_clicked()
    assert not window.dirty


def test_dialog_long_list_and_incomplete_details_are_readable(editor):
    from ui.shared_queue_save_dialog import SharedQueueSaveDialog
    window, target, _, products, _ = editor
    data = project(*(["Q"] * 100))
    for index, condition in enumerate(data["test_groups"][0]["test_conditions"]):
        condition["condition_name"] = f"Condition {index + 1}"
    write_json(products / "one.json", data)
    result = window.reference_scanner.find_references(str(target))
    dialog = SharedQueueSaveDialog(str(target), result)
    dialog.show()
    QApplication.processEvents()
    assert "Condition 100" in dialog.details.toPlainText()
    assert str(products) not in dialog.details.toPlainText()
    assert dialog.details.verticalScrollBar().maximum() > 0
    assert dialog.save_button.text() == "确认"
    assert dialog.cancel_button.text() == "取消"
    dialog.reject()

    (products / "one.json").write_text("{", encoding="utf-8")
    dialog = SharedQueueSaveDialog(str(target), window.reference_scanner.find_references(str(target)))
    dialog.show()
    QApplication.processEvents()
    assert dialog.details.toPlainText() == ""
    assert not dialog.details.isVisible()
    assert "未列出的工况" in dialog.text()
    assert "one.json" in dialog.toolTip()
    dialog.reject()


def test_dialog_paths_deduplicate_saved_draft_but_keep_renames(editor):
    from ui.shared_queue_save_dialog import SharedQueueSaveDialog
    window, target, _, products, _ = editor
    saved = project("Q", "Q", name="<b>Product</b>")
    saved["test_groups"][0]["test_conditions"][1]["condition_name"] = "A"
    write_json(products / "one.json", saved)
    draft = copy.deepcopy(saved)
    draft["test_groups"][0]["test_conditions"][1]["condition_name"] = "Renamed"
    result = window.reference_scanner.find_references(str(target), drafts=(
        QueueReferenceDraft(draft, products / "one.json"),))
    dialog = SharedQueueSaveDialog(str(target), result)
    assert dialog.details.toPlainText().splitlines() == [
        "<b>Product</b>/Port/A", "<b>Product</b>/Port/Renamed"]
    assert dialog.informativeText() == "此队列的修改将同时影响以上所有工况。建议另存为后重新选择。"
    assert len(result.references) == 2
    dialog.close()


@pytest.mark.parametrize("action", ["confirm", "cancel", "enter", "escape", "close"])
@pytest.mark.parametrize("incomplete", [False, True])
def test_real_shared_message_decision_controls_queue_write(editor, action, incomplete):
    from ui.shared_queue_save_dialog import SharedQueueSaveDialog
    window, target, registry, products, _ = editor
    if incomplete:
        (products / "one.json").write_text("{", encoding="utf-8")
    before = target.read_bytes(), registry.read_bytes()
    change(window)
    window.confirm_shared_save = window._confirm_shared_save
    window.show()
    observed = []

    def decide():
        dialog = QApplication.activeModalWidget()
        observed.append(isinstance(dialog, SharedQueueSaveDialog))
        if action == "confirm":
            QTest.mouseClick(dialog.save_button, Qt.LeftButton)
        elif action == "cancel":
            QTest.mouseClick(dialog.cancel_button, Qt.LeftButton)
        elif action == "close":
            dialog.close()
        else:
            QTest.keyClick(dialog, Qt.Key_Return if action == "enter" else Qt.Key_Escape)

    QTimer.singleShot(0, decide)
    window.ok_btn_clicked()
    assert observed == [True]
    assert (target.read_bytes() != before[0]) is (action in ("confirm", "enter"))
    assert registry.read_bytes() == before[1]
    if action not in ("confirm", "enter"):
        assert window.dirty and window.isVisible()
        assert window.select_list.config[0].detail["sample_rate"] == 48000


def test_auto_analysis_checkbox_marks_shared_draft_dirty(editor):
    window, target, _, _, _ = editor
    before = target.read_bytes()
    window.auto_analysis_box.setChecked(False)
    assert window.dirty
    assert target.read_bytes() == before


@pytest.mark.parametrize("rate", [48000.5, True, None, "broken"])
def test_malformed_recording_rate_cannot_be_saved(editor, rate):
    window, target, registry, _, errors = editor
    before = target.read_bytes(), registry.read_bytes()
    window.select_list.config[0].mode = "RECORD_ONLY"
    window.select_list.config[0].detail.update(sample_rate=rate, ve_range_index=5)
    window.confirm_shared_save = lambda *a: True
    window.ok_btn_clicked()
    assert (target.read_bytes(), registry.read_bytes()) == before
    assert any("sample_rate" in error for error in errors)


def test_registry_failure_retains_draft_and_retry_finishes_registration(editor, monkeypatch):
    window, target, registry, _, errors = editor
    new_target = target.with_name("new.json")
    change(window)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(new_target), ""))
    real_save = LoadUiConfig.save_data_to_json
    def fail_registry(data, path, *args):
        return False if str(path) == str(registry) else real_save(data, path, *args)
    monkeypatch.setattr(LoadUiConfig, "save_data_to_json", fail_registry)
    window.save_btn_clicked()
    assert window.dirty and errors and new_target.exists()
    monkeypatch.setattr(LoadUiConfig, "save_data_to_json", real_save)
    window.save_btn_clicked()
    assert json.loads(registry.read_text(encoding="utf-8"))["new"] == str(new_target)
    assert not window.dirty


def test_default_queue_registration_happens_only_after_success(editor, monkeypatch, tmp_path, save_selection):
    window, target, registry, products, _ = editor
    default = tmp_path / "ui" / "ui_config" / "sequence_config.json"
    write_json(default, [])
    write_json(registry, {"using_config_path": None, "default": str(default)})
    write_json(products / "one.json", project("default", "default"))
    monkeypatch.setattr("ui.operation_sequence.DEFAULT_DIR", str(tmp_path).replace("\\", "/") + "/")
    window.using_config_path = str(tmp_path / "ui" / "ui_config" / "none_path.json")
    save_selection(default.as_posix())
    before = default.read_bytes(), registry.read_bytes()
    window.ok_btn_clicked()
    assert (default.read_bytes(), registry.read_bytes()) == before
    window.confirm_shared_save = lambda *a: True
    window.ok_btn_clicked()
    assert json.loads(registry.read_text(encoding="utf-8"))["using_config_path"] == str(default).replace("\\", "/")


def test_clear_control_retains_unsaved_draft_boundary(editor, monkeypatch):
    window, target, _, _, _ = editor
    before = target.read_bytes()
    button = next(button for button in window.findChildren(QPushButton) if button.toolTip() == "清空")
    button.click()
    assert window.dirty
    assert window.select_list.config == []
    window.show()
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.Cancel)
    window.close()
    assert window.isVisible()
    assert target.read_bytes() == before


def test_save_as_new_target_continues_editing_that_file(editor, monkeypatch):
    window, target, _, _, _ = editor
    before = target.read_bytes()
    change(window)
    new_target = target.with_name("new.json")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(new_target), ""))
    window.save_btn_clicked()
    assert window.using_config_path == str(new_target)
    window.select_list.config[0].detail["sample_rate"] = 96000
    window.select_list._notify_config_changed()
    assert target.read_bytes() == before
    assert json.loads(new_target.read_text(encoding="utf-8"))[0]["seq1"]["acq"]["detail"]["sample_rate"] == 96000


def test_main_window_passes_parent_context_to_real_queue_editor(editor, monkeypatch):
    from types import SimpleNamespace
    import main_window
    window, target, _, products, _ = editor
    draft = QueueReferenceDraft(project("Q", ""), products / "one.json")
    provider = lambda: (draft,)
    observed = []
    def construct(path, **kwargs):
        assert path == str(target)
        assert kwargs["parent_draft_provider"] is provider
        window.parent_draft_provider = kwargs["parent_draft_provider"]
        return window
    def execute():
        change(window)
        window.confirm_shared_save = lambda path, result: observed.append(result) or False
        window.ok_btn_clicked()
    monkeypatch.setattr(main_window, "AnalysisModelSelect", construct)
    monkeypatch.setattr(window, "exec", execute)
    refreshes = []
    host = SimpleNamespace(mic=None, speaker=None, mic_channels=[0], speaker_channels=[],
                           sequence_window=SimpleNamespace(
                               on_sequence_config_updated=lambda: refreshes.append(True),
                               update_player_btn_is_paused=lambda: None))
    main_window.MainWindow._open_analysis_model_select(host, str(target), provider)
    assert [r.sources for r in observed[0].references] == [{"saved", "draft"}, {"saved"}]
    assert refreshes == [True]


def test_import_restores_auto_analysis_without_creating_a_change(editor, monkeypatch):
    window, target, registry, products, _ = editor
    next_target = target.with_name("next.json")
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload[0]["seq1"]["analysis_list"]["auto_analysis"] = False
    write_json(next_target, payload)
    write_json(registry, {"Q": str(target), "next": str(next_target)})
    write_json(products / "one.json", project("next", "next"))
    before = next_target.read_bytes()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(next_target), ""))
    window.load_btn_clicked()
    assert not window.auto_analysis_box.isChecked()
    confirmations = []
    window.confirm_shared_save = lambda *a: confirmations.append(a) or False
    window.ok_btn_clicked()
    assert not confirmations
    assert next_target.read_bytes() == before


@pytest.mark.parametrize("retry_registration", [False, True])
def test_save_as_same_basename_preserves_existing_aliases(editor, monkeypatch, retry_registration):
    window, target, registry_path, products, errors = editor
    next_target = target.parent / "new" / "Q.json"
    existing_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    existing_registry["Q (2)"] = str(target.parent / "other" / "Q.json")
    write_json(registry_path, existing_registry)
    before = target.read_bytes()
    change(window)
    confirmations = []
    window.confirm_shared_save = lambda *args: confirmations.append(args) or True
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(next_target), ""))
    real_save = LoadUiConfig.save_data_to_json
    if retry_registration:
        def fail_registry(data, path, *args):
            return False if str(path) == str(registry_path) else real_save(data, path, *args)
        monkeypatch.setattr(LoadUiConfig, "save_data_to_json", fail_registry)
        window.save_btn_clicked()
        assert window.dirty and errors and next_target.exists()
        assert json.loads(registry_path.read_text(encoding="utf-8")) == existing_registry
        assert len(window.reference_scanner.find_references(str(target)).references) == 2
        monkeypatch.setattr(LoadUiConfig, "save_data_to_json", real_save)
    window.save_btn_clicked()
    registered = json.loads(registry_path.read_text(encoding="utf-8"))
    assert all(registered[name] == path for name, path in existing_registry.items())
    assert registered["Q (3)"] == str(next_target)
    assert len(window.reference_scanner.find_references(str(target)).references) == 2
    assert not window.reference_scanner.find_references(str(next_target)).references
    assert not confirmations
    assert target.read_bytes() == before
    window.save_btn_clicked()
    window.select_list.config[0].detail["sample_rate"] = 96000
    window.select_list._notify_config_changed()
    assert json.loads(registry_path.read_text(encoding="utf-8")) == registered
    assert target.read_bytes() == before
    write_json(products / "one.json", project("Q", "Q (3)"))
    assert len(window.reference_scanner.find_references(str(target)).references) == 1
    assert len(window.reference_scanner.find_references(str(next_target)).references) == 1


@pytest.mark.parametrize("selection_key_present", [False, True])
def test_save_as_reserved_basename_never_sets_selection_key(editor, monkeypatch, selection_key_present):
    window, target, registry_path, _, _ = editor
    next_target = target.parent / "new" / "using_config_path.json"
    existing_registry = {"Q": str(target)}
    if selection_key_present:
        existing_registry["using_config_path"] = str(target)
    write_json(registry_path, existing_registry)
    before = target.read_bytes()
    change(window)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(next_target), ""))
    window.save_btn_clicked()
    registered = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registered.get("using_config_path") == existing_registry.get("using_config_path")
    assert registered["using_config_path (2)"] == str(next_target)
    assert registered["Q"] == str(target)
    assert len(window.reference_scanner.find_references(str(target)).references) == 2
    assert not window.reference_scanner.find_references(str(next_target)).references
    assert target.read_bytes() == before


@pytest.mark.parametrize("basename", ["Q", "using_config_path"])
@pytest.mark.parametrize("retry_registration", [False, True])
def test_import_uses_safe_alias_registration_without_writing_queues(editor, monkeypatch, basename, retry_registration):
    window, target, registry_path, _, errors = editor
    imported = target.parent / "imported" / f"{basename}.json"
    write_json(imported, json.loads(target.read_text(encoding="utf-8")))
    before = target.read_bytes(), imported.read_bytes(), registry_path.read_bytes()
    original_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(imported), ""))
    writes = []
    monkeypatch.setattr(LoadUiConfig, "save_sequence_config_to_json", lambda *a: writes.append(a) or False)
    real_save = LoadUiConfig.save_data_to_json
    if retry_registration:
        def fail_registry(data, path, *args):
            return False if str(path) == str(registry_path) else real_save(data, path, *args)
        monkeypatch.setattr(LoadUiConfig, "save_data_to_json", fail_registry)
    window.load_btn_clicked()
    if retry_registration:
        assert errors and window.dirty
        assert registry_path.read_bytes() == before[2]
        monkeypatch.setattr(LoadUiConfig, "save_data_to_json", real_save)
        window.ok_btn_clicked()
    registered = json.loads(registry_path.read_text(encoding="utf-8"))
    assert all(registered[name] == path for name, path in original_registry.items())
    assert registered[f"{basename} (2)"] == str(imported).replace("\\", "/")
    assert len(window.reference_scanner.find_references(str(target)).references) == 2
    assert not window.reference_scanner.find_references(str(imported)).references
    assert (target.read_bytes(), imported.read_bytes()) == before[:2]
    assert not writes
    assert not window.dirty


@pytest.mark.parametrize("basename,alias", [
    ("R", "R"), ("Q", "Q (2)"), ("using_config_path", "using_config_path (2)"),
])
@pytest.mark.parametrize("retry_registration", [False, True])
def test_save_as_identical_unregistered_file_completes_registration(
    editor, monkeypatch, basename, alias, retry_registration,
):
    window, target, registry_path, _, errors = editor
    next_target = target.parent / "identical" / f"{basename}.json"
    write_json(next_target, window.format_config_data(window.select_list.config))
    before = target.read_bytes(), next_target.read_bytes(), registry_path.read_bytes()
    original_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    confirmations, queue_writes = [], []
    window.confirm_shared_save = lambda *a: confirmations.append(a) or False
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(next_target), ""))
    monkeypatch.setattr(LoadUiConfig, "save_sequence_config_to_json", lambda *a: queue_writes.append(a) or False)
    real_save = LoadUiConfig.save_data_to_json
    if retry_registration:
        def fail_registry(data, path, *args):
            return False if str(path) == str(registry_path) else real_save(data, path, *args)
        monkeypatch.setattr(LoadUiConfig, "save_data_to_json", fail_registry)
        window.save_btn_clicked()
        assert errors and window.dirty
        assert window.using_config_path == str(target)
        assert registry_path.read_bytes() == before[2]
        monkeypatch.setattr(LoadUiConfig, "save_data_to_json", real_save)
    window.save_btn_clicked()
    registered = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registered[alias] == str(next_target)
    assert all(registered[name] == path for name, path in original_registry.items())
    assert len(window.reference_scanner.find_references(str(target)).references) == 2
    assert not window.reference_scanner.find_references(str(next_target)).references
    assert window.using_config_path == str(next_target)
    assert not window.dirty
    registered_bytes = registry_path.read_bytes()
    window.save_btn_clicked()
    assert registry_path.read_bytes() == registered_bytes
    assert (target.read_bytes(), next_target.read_bytes()) == before[:2]
    assert not confirmations and not queue_writes


@pytest.mark.parametrize("selection", ["absent", "null", "existing"])
@pytest.mark.parametrize("identical", [False, True])
@pytest.mark.parametrize("retry_registration", [False, True])
def test_save_as_from_placeholder_preserves_global_selection(
    editor, monkeypatch, selection, identical, retry_registration,
):
    window, target, registry_path, _, errors = editor
    placeholder = target.parent / "ui" / "ui_config" / "none_path.json"
    next_target = target.with_name("R.json")
    window.using_config_path = str(placeholder)
    original_registry = {"Q": str(target)}
    if selection != "absent":
        original_registry["using_config_path"] = None if selection == "null" else str(target)
    write_json(registry_path, original_registry)
    if identical:
        write_json(next_target, window.format_config_data(window.select_list.config))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(next_target), ""))
    real_save = LoadUiConfig.save_data_to_json
    if retry_registration:
        def fail_registry(data, path, *args):
            return False if str(path) == str(registry_path) else real_save(data, path, *args)
        monkeypatch.setattr(LoadUiConfig, "save_data_to_json", fail_registry)
        window.save_btn_clicked()
        assert errors and window.dirty
        assert window.using_config_path == str(placeholder)
        assert json.loads(registry_path.read_text(encoding="utf-8")) == original_registry
        monkeypatch.setattr(LoadUiConfig, "save_data_to_json", real_save)
    window.save_btn_clicked()
    registered = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registered == {**original_registry, "R": str(next_target)}
    assert window.using_config_path == str(next_target)


@pytest.mark.parametrize("operation", ["save_btn_clicked", "load_btn_clicked"])
def test_save_as_or_import_builtin_file_does_not_activate_default(editor, monkeypatch, operation):
    window, target, registry_path, _, _ = editor
    default = target.parent / "ui" / "ui_config" / "sequence_config.json"
    placeholder = default.with_name("none_path.json")
    monkeypatch.setattr("ui.operation_sequence.DEFAULT_DIR", str(target.parent).replace("\\", "/") + "/")
    window.using_config_path = str(placeholder)
    write_json(default, window.format_config_data(window.select_list.config))
    original_registry = {"Q": str(target), "using_config_path": str(target)}
    write_json(registry_path, original_registry)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(default), ""))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(default), ""))
    getattr(window, operation)()
    registered = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registered["using_config_path"] == str(target)
    assert "默认配置" not in registered
    assert registered["sequence_config"].replace("\\", "/") == str(default).replace("\\", "/")


@pytest.mark.parametrize("identical", [False, True])
def test_ok_builtin_default_activation_survives_registration_retry(editor, monkeypatch, identical, save_selection):
    window, target, registry_path, _, errors = editor
    default = target.parent / "ui" / "ui_config" / "sequence_config.json"
    monkeypatch.setattr("ui.operation_sequence.DEFAULT_DIR", str(target.parent).replace("\\", "/") + "/")
    window.using_config_path = str(default.with_name("none_path.json"))
    save_selection(default.as_posix())
    if identical:
        write_json(default, window.format_config_data(window.select_list.config))
    original_registry = {"Q": str(target), "using_config_path": None}
    write_json(registry_path, original_registry)
    real_save = LoadUiConfig.save_data_to_json
    def fail_registry(data, path, *args):
        return False if str(path) == str(registry_path) else real_save(data, path, *args)
    monkeypatch.setattr(LoadUiConfig, "save_data_to_json", fail_registry)
    window.ok_btn_clicked()
    assert errors and window.dirty
    assert json.loads(registry_path.read_text(encoding="utf-8")) == original_registry
    monkeypatch.setattr(LoadUiConfig, "save_data_to_json", real_save)
    window.ok_btn_clicked()
    registered = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registered["using_config_path"] == str(default).replace("\\", "/")
    assert registered["默认配置"] == str(default).replace("\\", "/")
    assert not window.dirty
