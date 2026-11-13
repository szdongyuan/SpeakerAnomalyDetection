"""Current editor notifications and product refresh through the real MainWindow."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt5.QtWidgets import QMessageBox

import main_window
from base import load_config
from base.motor_workbench_config import MotorWorkbenchConfigRepository
from ui.product_test_project_config_dialog import ProductTestProjectConfigDialog
from unit_test.product_config_fixtures import make_manager, make_project
from unit_test.ui.test_workbench_recording_startup import startup


@pytest.fixture
def current_main(startup, tmp_path, monkeypatch):
    manager = make_manager(tmp_path)
    project = make_project(tmp_path, project_name='A', condition_count=2)
    assert manager.save_project(None, project) == (True, 'A.json')
    repository = MotorWorkbenchConfigRepository(manager.registry_path, manager.queue_registry_path,
                                                plot_layout_path=tmp_path / 'missing-layout.json')
    monkeypatch.setattr(main_window, 'MotorWorkbenchConfigRepository', lambda: repository)
    monkeypatch.setattr(main_window, 'ProductTestProjectConfigManager', lambda: manager)
    monkeypatch.setattr(load_config, 'SEQUENCE_CONFIG_REGISTRY_PATH', manager.queue_registry_path)
    messages = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: messages.append(args[1:]))
    monkeypatch.setattr(QMessageBox, 'information', lambda *args: messages.append(args[1:]))
    def login(_):
        startup.bridge.change_worker(1)
        startup.bridge.answer_probe()
        return 'Engineer', 'test'
    window = startup.launch(login)
    queue_path = Path(manager.load_queue_catalog()['测试队列A']['path'])
    return SimpleNamespace(window=window, manager=manager, project=project,
                           queue_path=queue_path, messages=messages, startup=startup)


def save_b(current):
    project = deepcopy(current.project)
    project['project_id'] = '00000000-0000-4000-8000-000000000002'
    project['project_name'] = 'B'
    assert current.manager.save_project(None, project) == (True, 'B.json')
    current.window.workbench_controller.refresh_product_catalog()
    return project


def test_main_connects_committed_notification_before_opening_editor(current_main, monkeypatch):
    current = current_main
    window = current.window
    events = []
    original = window.on_workbench_configuration_committed
    monkeypatch.setattr(window, 'on_workbench_configuration_committed',
                        lambda: (events.append('committed'), original()))
    class Dialog:
        initial_load_succeeded = True
        def __init__(self, manager, queue_editor, parent, **options):
            assert options['input_device_provider']() is window.mic
            assert options['contextual_queue_editor_callback'] == queue_editor
            self.projects_changed = SimpleNamespace(connect=lambda callback: setattr(self, 'committed', callback))
        def exec(self):
            events.append('opened')
            assert window._configuration_editor_open
            self.committed()
    monkeypatch.setattr(main_window, 'ProductTestProjectConfigDialog', Dialog)
    window.on_product_test_program_config()
    assert events == ['opened', 'committed']
    assert not window._configuration_editor_open


def test_exceptional_editor_exit_restores_current_admission(current_main, monkeypatch):
    window = current_main.window
    class Dialog:
        initial_load_succeeded = True
        def __init__(self, *args, **options):
            self.projects_changed = SimpleNamespace(connect=lambda _: None)
        def exec(self):
            assert window._configuration_editor_open
            raise RuntimeError('dialog failed')
    monkeypatch.setattr(main_window, 'ProductTestProjectConfigDialog', Dialog)
    with pytest.raises(RuntimeError, match='dialog failed'):
        window.on_product_test_program_config()
    assert not window._configuration_editor_open
    assert not window._configuration_deletion_busy()


def test_switch_saved_project_edit_save_and_reopen(current_main):
    current = current_main
    window, manager = current.window, current.manager
    original_a = Path(manager.program_dir, 'A.json').read_bytes()
    project_b = save_b(current)
    window.workbench_controller.select_product('B.json')
    assert manager.load_registry()['active_file'] == 'B.json'
    assert window.sequence_window.using_file_combobox.currentData() == 'B.json'
    editor = ProductTestProjectConfigDialog(manager)
    editor.projects_changed.connect(window.on_workbench_configuration_committed)
    try:
        assert editor.collect_project() == project_b
        editor.project_name_input.setText('B已修改')
        assert editor._save_project()
        assert not Path(manager.program_dir, 'B.json').exists()
        assert editor.current_file == 'B已修改.json'
        assert window.sequence_window.using_file_combobox.currentData() == 'B已修改.json'
    finally:
        editor._dirty = False
        editor.close()
    reopened = ProductTestProjectConfigDialog(manager)
    try:
        assert reopened.current_file == 'B已修改.json'
        assert reopened.collect_project()['project_name'] == 'B已修改'
    finally:
        reopened.close()
    assert Path(manager.program_dir, 'A.json').read_bytes() == original_a
    window.workbench_controller.select_product('A.json')
    assert manager.load_registry()['active_file'] == 'A.json'


@pytest.mark.parametrize('failure', ['corrupt', 'missing', 'write_false', 'write_error', 'locked'])
def test_failed_switch_preserves_previous_current_selection(current_main, monkeypatch, failure):
    current = current_main
    window, manager = current.window, current.manager
    save_b(current)
    registry = Path(manager.registry_path)
    before = registry.read_bytes()
    path_b = Path(manager.program_dir, 'B.json')
    if failure == 'corrupt':
        path_b.write_text('{', encoding='utf-8')
    elif failure == 'missing':
        path_b.rename(path_b.with_suffix('.moved'))
    elif failure == 'write_false':
        monkeypatch.setattr(manager, 'save_registry', lambda _: False)
    elif failure == 'write_error':
        def fail(_):
            raise OSError('write failed')
        monkeypatch.setattr(manager, 'save_registry', fail)
    else:
        current.startup.analysis_service.pending_task_count = 1
    window.workbench_controller.select_product('B.json')
    assert registry.read_bytes() == before
    assert window.sequence_window.using_file_combobox.currentData() == 'A.json'
    if failure == "locked":
        window.workbench_controller.refresh_recording_state()
        assert not window.sequence_window.using_file_combobox.isEnabled()
    else:
        assert current.messages
