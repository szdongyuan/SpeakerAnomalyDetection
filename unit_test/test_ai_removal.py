import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from base.analysis_config_validation import validate_analysis_config
from base.load_config import LoadUiConfig
from consts import error_code


@pytest.mark.parametrize("analysis_type", ["ai", " AI "])
def test_noncanonical_ai_type_is_rejected_without_rewriting_file(tmp_path, analysis_type):
    path = tmp_path / "queue.json"
    payload = [{"seq1": {"analysis_list": {
        "display_sequence": ["legacy", "spl"],
        "legacy": {"type": analysis_type},
        "spl": {"type": "SPL", "limit_checked": True},
    }}}]
    original = json.dumps(payload).encode()
    path.write_bytes(original)

    code, message = LoadUiConfig.load_sequence_config_from_json(path)

    assert code == error_code.INVALID_DATA_LOADING
    assert "AI 类型必须写为 AI" in message
    assert path.read_bytes() == original


def test_rule_queue_with_legacy_default_field_is_still_loadable(tmp_path):
    path = tmp_path / "queue.json"
    payload = [{"seq1": {"analysis_list": {
        "display_sequence": ["spl"], "default_ai": None,
        "spl": {"type": "SPL", "limit_checked": True},
    }}}]
    path.write_text(json.dumps(payload), encoding="utf-8")
    code, actual = LoadUiConfig.load_sequence_config_from_json(path)
    assert code == error_code.OK
    assert actual == payload


def test_ai_draft_loads_without_rewriting_file_but_execution_is_blocked(tmp_path):
    path = tmp_path / "queue.json"
    config = {"display_sequence": ["AI 1"], "AI 1": {
        "type": "AI", "analyse_model_name": "", "analysis_channel": 0,
    }}
    payload = [{"seq1": {"analysis_list": config}}]
    original = json.dumps(payload).encode()
    path.write_bytes(original)
    code, loaded = LoadUiConfig.load_sequence_config_from_json(path)
    assert code == error_code.OK
    assert loaded == payload
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="尚未选择模型"):
        validate_analysis_config(config, for_execution=True)


def test_ai_model_name_must_be_text():
    with pytest.raises(ValueError, match="模型名称必须为字符串"):
        validate_analysis_config({"AI 1": {"type": "AI", "analyse_model_name": 123}})


def test_startup_and_rule_analysis_work_without_tensorflow_or_keras(tmp_path):
    script = r'''
import importlib
import importlib.abc
import logging
import sys
from types import SimpleNamespace
from unittest.mock import patch

class BlockModelLibraries(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'tensorflow', 'keras', 'onnxruntime'}:
            raise AssertionError('Forbidden dependency: ' + fullname)

sys.meta_path.insert(0, BlockModelLibraries())
from PyQt5.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from consts.running_consts import MODULES_LOAD
for _, module in MODULES_LOAD:
    if module:
        importlib.import_module(module)
import main_window as m
with patch.object(m.LogManager, 'set_log_handler', logging.getLogger), \
     patch.object(m, 'restore_or_default', return_value=(None, None, [], [])), \
     patch.object(m.MainWindow, 'init_ui', lambda self: None), \
     patch.object(m.MainWindow, '_init_ve_hardware_runtime', lambda self: None):
    window = m.MainWindow(recording_bridge=SimpleNamespace(shutdown=lambda: None))
    for role in ('Operator', 'Engineer', 'Admin'):
        window.access_lvl = role
        window.on_access_lvl_changed()
        menu = window.init_menu()
        actions = menu.actions()[0].menu().actions()
        assert '训练AI模型' not in [a.text() for a in actions]
        assert [a.text() for a in actions].count('AI模型管理') == 1
        assert window.function_action_ai_model_manager.isVisible()
        assert window.function_action_ai_model_manager.isEnabled() == (role != 'Operator')
    from ui.ai.model_manager_dialog import AiModelManagerDialog
    with patch.object(AiModelManagerDialog, 'exec', return_value=0) as opened, \
         patch.object(m, 'DATABASE_PATH', 'missing/model_frontend.db'):
        window.function_action_ai_model_manager.trigger()
        opened.assert_called_once_with()
from base.analysis_algorithm_adapters import _HANDLERS
from base.analysis_algorithm_adapters import calculate_analysis_instance
from base.analysis.executor import _SUPPORTED_TYPES
assert 'AI' not in _HANDLERS and 'AI' not in _SUPPORTED_TYPES
import numpy as np
signal = .01 * np.sin(2 * np.pi * 1000 * np.arange(12000) / 48000)
for kind, config in (
    ('SPL', {'weighting': 'Z', 'show_overall_spl': True}),
    ('FFT', {'n_fft': 1024}),
    ('FBA', {'f_min': 20, 'f_max': 20000, 'weighting': 'Z'}),
    ('Spec', {'n_fft': 1024, 'hop_length': 256}),
):
    result = calculate_analysis_instance(
        kind, signal, 48000, config, 1.0,
        source='自动分析', sequence_snapshot={},
    )
    assert result['plot']
assert not any(n.split('.')[0] in {'tensorflow', 'keras', 'onnxruntime'} for n in sys.modules)
print('startup_without_model_libraries_ok')
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env={
            **os.environ,
            "QT_QPA_PLATFORM": "offscreen",
            "NUMBA_CACHE_DIR": str(tmp_path / "numba_cache"),
        },
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "startup_without_model_libraries_ok" in result.stdout
