import ast
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest
from PyQt5.QtWidgets import QComboBox, QWidget

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("filename", ["main_window.py", "main_window_Launcher.py"])
def test_bootstrap_reaches_gui_after_freeze_support_without_tools_imports(filename):
    # Stop at the first GUI import: even frozen children must hit bootstrap first.
    script = '''
import builtins, multiprocessing, runpy
called = []
multiprocessing.freeze_support = lambda: called.append(True)
original = builtins.__import__
class ReachedGui(Exception): pass
def guarded(name, *args, **kwargs):
    assert name != "tools" and not name.startswith("tools."), name
    if name.startswith(("PyQt5", "ui.")):
        assert called, "GUI imported before freeze_support"
        raise ReachedGui()
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
try:
    runpy.run_path(sys.argv[1], run_name="__main__")
except ReachedGui:
    pass
else:
    raise AssertionError("bootstrap did not reach GUI imports")
assert called
'''
    result = subprocess.run([sys.executable, "-c", "import sys\n" + script, str(ROOT / filename)],
                            cwd=ROOT, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_launcher_creates_once_and_injects_bridge():
    tree = ast.parse((ROOT / "main_window_Launcher.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert sum(isinstance(node.func, ast.Name) and node.func.id == "RawAudioCsvService" for node in calls) == 1
    main_call = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "MainWindow")
    assert any(kw.arg == "raw_audio_csv_bridge" for kw in main_call.keywords)


@pytest.mark.parametrize("filename", ["main_window.py", "main_window_Launcher.py"])
def test_spawn_reimport_does_not_start_application_or_csv_service(filename):
    script = '''
import multiprocessing, runpy, sys, threading
from PyQt5.QtWidgets import QApplication
runpy.run_path(sys.argv[1], run_name="__mp_main__")
assert QApplication.instance() is None
assert not multiprocessing.active_children()
assert not any(thread.name.startswith("raw-csv-") for thread in threading.enumerate())
'''
    result = subprocess.run([sys.executable, "-c", script, str(ROOT / filename)],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("injected", [False, True])
def test_main_owns_fallback_and_shares_with_sequence(ui_qapp, monkeypatch, injected):
    import main_window as module
    from base.raw_audio_csv_service import RawAudioCsvService
    from ui.raw_audio_csv_service_bridge import RawAudioCsvServiceBridge
    borrowed = RawAudioCsvServiceBridge(RawAudioCsvService(), ui_qapp) if injected else None
    monkeypatch.setattr(module, "restore_or_default", lambda **kwargs: (None, None, [], []))
    monkeypatch.setattr(module.MainWindow, "_init_ve_hardware_runtime", lambda self: None)
    monkeypatch.setattr(module.MainWindow, "init_ui", lambda self: None)
    recording = Mock()
    window = module.MainWindow(recording_bridge=recording, raw_audio_csv_bridge=borrowed)
    try:
        assert window._owns_raw_audio_csv_service is (not injected)
        if injected:
            assert window.raw_audio_csv_bridge is borrowed
        sequence = QWidget()
        sequence.left_panel = Mock()
        sequence.plot_grid = Mock()
        sequence.show_workbench_round = Mock()
        sequence.set_workbench_active_product = Mock()
        sequence.show_workbench_products = Mock()
        sequence.set_workbench_product_selection_enabled = Mock()
        sequence.using_file_combobox = QComboBox(sequence)
        sequence.update_v2pa_factor = Mock()
        factory = Mock(return_value=sequence)
        monkeypatch.setattr(module, "SequenceWindow", factory)
        monkeypatch.setattr(module, "MotorWorkbenchController", Mock())
        monkeypatch.setattr(window, "init_menu", lambda: QWidget())
        monkeypatch.setattr(window, "set_title", lambda: QWidget())
        monkeypatch.setattr(window, "_create_menu_row", lambda menu: QWidget())
        window.init_sequence_widget()
        assert factory.call_args.kwargs == {"can_commit": window._workbench_scanner_admission_available}
        assert window.raw_audio_csv_service._process is None
    finally:
        window.raw_audio_csv_service.begin_shutdown()
        assert window.raw_audio_csv_service.closed.wait(3)
        window.raw_audio_csv_bridge.close_delivery()
        window.video_controller.shutdown()
        window.deleteLater()
