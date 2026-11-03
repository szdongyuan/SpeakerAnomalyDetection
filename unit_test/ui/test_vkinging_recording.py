"""Real sequence methods, service, SDK double, WAV and temporary SQLite integration."""
import ast
import copy
import logging
import math
import os
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace
import uuid

import numpy as np
import pytest
import soundfile as sf
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QWidget

from base.acquisition_recording_defaults import normalize_record_only_detail, normalize_play_record_detail
from base.audio_sample_rate import resolve_input_sample_rate, resolve_duplex_sample_rate
from base.data_struct.data_deal_struct import DataDealStruct
from base.db_manager import DataSave
from base.hardware_management import HardwareManagementRepository
from base.load_config import LoadUiConfig
from base.recording_management import RecordingManager
from base.recording_calibration_snapshot import build_vkinging_recording_wav_calibration_metadata
from base.vkinging_input import VkingingCaptureRequest, is_vkinging_device, resolve_vkinging_input_config
from base.vkinging_service import VkingingService
from base.wav_calibration_metadata import inspect_wav_calibration_metadata, resolve_wav_channel_v2pa_factor
from consts import error_code
from consts.audio_consts import normalize_float_bit_depth
from ui.vkinging_bridge import VkingingBridge
from unit_test.base.vkinging_fakes import FakeSDK

ROOT = Path(__file__).resolve().parents[2]


def sequence_class(namespace):
    tree = ast.parse((ROOT / 'ui/sequence/sequence_widget.py').read_text(encoding='utf-8'))
    originals = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SequenceWindow')
    names = {'judge_play_and_record', 'start_this_play', 'checked_work_status_message', 'reset_work_pram',
             '_begin_recording_output_attempt', '_recording_output_path', '_clear_recording_output_attempt',
             '_finish_recording_success', '_ensure_pending_recorded_count', '_clear_pending_recorded_count',
             '_commit_pending_recorded_count', '_run_post_recording_followup', '_delete_successful_replay_backup',
             '_delete_path_best_effort', 'closeEvent'}
    methods = [n for n in originals.body if isinstance(n, ast.FunctionDef)
               and (n.name in names or 'vkinging' in n.name)]
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef)
               and (n.name.startswith('_vkinging') or n.name.startswith('_clear_')
                    or n.name == '_resolve_runtime_sample_rate_for_mode')]
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom)
               and n.module == 'base.vkinging_recording_preparation']
    cls = ast.ClassDef(name='SequenceWindow', bases=[ast.Name(id='QWidget', ctx=ast.Load())],
                       keywords=[], body=methods, decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=imports + helpers + [cls], type_ignores=[])),
                 str(ROOT / 'ui/sequence/sequence_widget.py'), 'exec'), namespace)
    return namespace['SequenceWindow']


class Button:
    def __init__(self): self.enabled = True
    def isEnabled(self): return self.enabled
    def setEnabled(self, value): self.enabled = value
    def setDisabled(self, value): self.enabled = not value


class Edit:
    def __init__(self, value): self.value = value
    def text(self): return self.value
    def setText(self, value): self.value = value


@pytest.fixture
def recording(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    path = tmp_path / 'system.db'
    db = DataSave(str(path))
    assert db.create_table()[0] == error_code.OK
    db.close()
    repo = HardwareManagementRepository(str(path))
    channels = tuple(range(32))
    device = dict(hostapi_name='vkinging', hardware_id='machine-A', name='old-name',
                  samplerate=51200, bit_depth=64, physical_channels=channels, max_input_channels=32)
    repo.register_asset(device, 'vkinging', 'Lab input', 51200)
    repo.update_mic_channel_calibrations('machine-A', {31: 2.5}, {31: 94},
        channel_conditions={31: dict(input_mode='IEPE', unit='V', range_min=-10., range_max=10.)})
    warnings, plots, analyses, saves, snapshots = [], [], [], [], []
    ns = dict(globals(), MessageBox=SimpleNamespace(warning=lambda *args: warnings.append(args[-1])),
              save_recorded_data_to_json=lambda *args: saves.append(args),
              build_vkinging_recording_wav_calibration_metadata=lambda *args, **kwargs:
                  (snapshots.append(1) or build_vkinging_recording_wav_calibration_metadata(*args, db_path=str(path), **kwargs)))
    win = sequence_class(ns)()
    output = tmp_path / 'recording.wav'
    ns['get_recorded_info'] = lambda *args: (str(output), dict(file_path=str(output), product_model='test',
        record_date='2026-11-02', barcode=None, labels=args[-1]))
    class Manager(RecordingManager):
        def __init__(self): self.db_path = str(path)
        def save_signal_info_to_db(self, info, stimulus, **kwargs):
            assert sdk.closed
            assert any(call[0] == 'clear_task' for call in sdk.calls)
            saves.append('db')
            return super().save_signal_info_to_db(info, stimulus, **kwargs)
    ns['RecordingManager'] = Manager
    win.sequence_config = [{'seq1': {'acq': {'mode': 'RECORD_ONLY', 'detail':
        dict(total_time=.1, recording_start_delay_ms=1, use_streaming_recording=False,
             sample_rate=44100, vkinging_range_index=2)}}}]
    win.mic, win.speaker, win.mic_channels = device, None, list(reversed(channels))
    win.data_struct = SimpleNamespace(stimulus_data=None, stimulus_info=None, clear_data=lambda: None)
    win.analysis_config = dict(auto_analysis=True)
    win.lineedit_type, win.lineedit_count, win.lineedit_s_or_n = Edit('test'), Edit('0'), Edit('')
    win.barcode_scanner_box = SimpleNamespace(isChecked=lambda: False)
    win.data_btn, win.replayer_btn = Button(), Button()
    win.current_recorded_count, win.last_play_count = 0, None
    win.tcp_flag, win.clicked_player_flag, win.player_status_flag = False, True, False
    win.analysis_window, win._analysis_result_summary_window = [], None
    win.default_logger = logging.getLogger('vkinging-integration')
    win._validate_sn_regex_before_start = lambda **kwargs: True
    win._set_sn_input_recording_read_only = lambda value: None
    win._clear_plot_area = win._cleanup_streaming_resources = lambda: None
    win.update_player_btn_is_playing = win.update_player_btn_is_paused = lambda: None
    win._should_use_streaming_recording = lambda: pytest.fail('soundcard routing used')
    win._build_current_wav_calibration_metadata = lambda: pytest.fail('soundcard snapshot used')
    win.plot_waveform_to_workspace = lambda data, rate: plots.append((data, rate))
    win.run = lambda: analyses.append(win.data_struct.wav_calibration_inspection)
    # 5171 raw frames, with short reads and exact tail; values exceed +/-1 V.
    raw = np.arange(5171 * 32, dtype=np.float64).reshape(-1, 32) / 1000 + 2.5
    reads = [(raw[i:i+400].ravel(), len(raw[i:i+400])) for i in range(0, len(raw), 400)]
    sdk = FakeSDK(devices=(('addr', 'new-name'),),
        attributes={('new-name', 'MachineId'): 'machine-A', ('new-name', 'Model'): ''},
        channels={'new-name': tuple(f'new-name/AIN{ch+1}' for ch in channels)}, reads=reads)
    service = VkingingService(sdk_factory=lambda: sdk)
    bridge = VkingingBridge(service)
    win.vkinging_service, win.vkinging_bridge = service, bridge
    ctx = SimpleNamespace(win=win, ns=ns, sdk=sdk, service=service, bridge=bridge, output=output,
                          db=path, raw=raw, warnings=warnings, plots=plots, analyses=analyses,
                          saves=saves, snapshots=snapshots, app=app)
    yield ctx
    if hasattr(win, '_dispose_vkinging_recording'): win._dispose_vkinging_recording()
    bridge.close()


def wait_until(ctx, predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, (ctx.service.state, ctx.warnings)
        ctx.bridge.poll()
        ctx.app.processEvents()
        threading.Event().wait(.001)


@pytest.mark.parametrize('bits', [32, 64])
@pytest.mark.parametrize('stream', [False, True])
def test_start_entry_32_channels_float_file_db_waveform_and_analysis(recording, stream, bits):
    c = recording
    c.win.mic['bit_depth'] = bits
    c.win.sequence_config[0]['seq1']['acq']['detail']['use_streaming_recording'] = stream
    c.win.start_this_play('not_labeled')
    wait_until(c, lambda: bool(c.analyses) or bool(c.warnings))
    assert not c.warnings
    assert len(c.snapshots) == len(c.plots) == len(c.analyses) == 1
    assert c.win.current_recorded_count == c.win.last_play_count == 1
    assert len([v for v in c.saves if v == 'db']) == 1
    actual, rate = sf.read(c.output, always_2d=True)
    expected = c.raw[51:5171].astype('float32' if bits == 32 else 'float64')
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (5120, 32) and rate == 51200
    inspection = inspect_wav_calibration_metadata(c.output)
    assert inspection.metadata['acquisition']['range_max'] == 2.5
    assert c.win.data_struct.wav_physical_channel_indices == tuple(reversed(range(32)))
    from unit_test.ui.test_vkinging_analysis import signal_namespace
    assert signal_namespace()['resolve_imported_wav_v2pa_factor'](c.win.data_struct, 0) == 2.5
    with sqlite3.connect(c.db) as db:
        assert db.execute('SELECT COUNT(*) FROM audio_data_table').fetchone()[0] == 1
    assert {call[2] for call in c.sdk.calls} != {threading.get_ident()}
    assert len({call[2] for call in c.sdk.calls}) == 1


def test_db_failure_retains_valid_file_and_no_success(recording):
    c = recording
    c.ns['RecordingManager'] = lambda: SimpleNamespace(save_signal_info_to_db=lambda *args: (99, 'read only'))
    c.win.start_this_play()
    wait_until(c, lambda: bool(c.warnings))
    assert c.output.exists()
    assert inspect_wav_calibration_metadata(c.output).source == 'vkinging'
    assert '文件已保存、入库失败' in c.warnings[-1] and str(c.output) in c.warnings[-1]
    assert c.win.current_recorded_count == 0 and not c.analyses and not c.plots


def test_existing_normal_output_is_never_overwritten(recording):
    c = recording
    c.output.write_bytes(b'user-existing')
    c.win.start_this_play()
    wait_until(c, lambda: bool(c.warnings))
    assert c.output.read_bytes() == b'user-existing'
    assert not c.analyses and c.win.current_recorded_count == 0


@pytest.mark.parametrize('value', [True, 1.0, -1, 7])
def test_explicit_invalid_queue_range_fails_before_sdk(recording, value):
    c = recording
    c.win.sequence_config[0]['seq1']['acq']['detail']['vkinging_range_index'] = value
    c.win.start_this_play()
    assert c.warnings and not c.sdk.calls


def test_sparse_selection_is_frozen_and_unscaled(recording):
    c = recording
    c.win.mic_channels = [31, 0, 8]
    c.sdk.reads.clear()
    values = c.raw[:, :3].copy()
    c.sdk.reads.extend((values[i:i+400].ravel(), len(values[i:i+400])) for i in range(0, len(values), 400))
    c.win.start_this_play()
    c.win.mic_channels[:] = [0]
    c.win.sequence_config[0]['seq1']['acq']['detail']['vkinging_range_index'] = 0
    wait_until(c, lambda: bool(c.analyses) or bool(c.warnings))
    assert not c.warnings
    np.testing.assert_array_equal(c.win.data_struct.store_wave_data_multi, values[51:])
    assert c.win.data_struct.wav_physical_channel_indices == (31, 0, 8)
    assert c.analyses[0].metadata['acquisition']['range_max'] == 2.5


def test_saved_path_and_labels_are_frozen_before_worker(recording):
    c = recording
    c.win.start_this_play('accepted-label')
    c.win.recorded_path = str(c.output.with_name('later.wav'))
    c.win.recorded_signal_info['labels'] = 'later-label'
    wait_until(c, lambda: bool(c.analyses) or bool(c.warnings))
    assert not c.warnings
    with sqlite3.connect(c.db) as db:
        path, label = db.execute('SELECT file_path, labels FROM audio_data_table').fetchone()
    assert Path(path) == c.output and label == 'accepted-label'
    assert c.win.recorded_path == str(c.output)


@pytest.mark.parametrize('stage', ['write_chunk', 'finalize', 'commit'])
def test_disk_and_metadata_failure_retain_owned_artifacts_without_db(recording, monkeypatch, stage):
    from base.vkinging_wav import VkingingVoltageWriter
    c = recording
    def fail(*args, **kwargs): raise OSError('injected ' + stage)
    monkeypatch.setattr(VkingingVoltageWriter, stage, fail)
    c.win.start_this_play()
    wait_until(c, lambda: bool(c.warnings))
    assert not c.output.exists() and not c.analyses and not c.plots
    assert list(c.output.parent.glob('*.partial.wav'))
    assert c.win.current_recorded_count == 0 and 'db' not in c.saves


@pytest.mark.parametrize('duration', [0, .09, 601, float('nan'), float('inf'), None])
def test_invalid_duration_rejected_before_sdk(recording, duration):
    c = recording
    c.win.sequence_config[0]['seq1']['acq']['detail']['total_time'] = duration
    c.win.start_this_play()
    assert c.warnings and not c.sdk.calls


def test_record_only_ignores_previous_stimulus_length(recording):
    c = recording
    c.win.data_struct.stimulus_data = np.zeros(23)
    c.win.data_struct.stimulus_info = {'total_time': 9}
    c.win.start_this_play()
    wait_until(c, lambda: bool(c.analyses) or bool(c.warnings))
    assert not c.warnings
    assert c.win.data_struct.store_wave_data_multi.shape == (5120, 32)
    assert c.win.data_struct.stimulus_data is None


@pytest.mark.parametrize('delay', [-1, 1500, float('nan'), float('inf'), -float('inf'),
                                 'invalid', None, True, False])
def test_explicit_invalid_start_delay_rejected_before_sdk_or_output(recording, delay):
    c = recording
    c.win.sequence_config[0]['seq1']['acq']['detail']['recording_start_delay_ms'] = delay
    c.win.start_this_play()
    assert c.warnings and not c.sdk.calls and not c.snapshots
    assert '启动延迟' in c.warnings[-1]
    assert not c.output.exists() and not list(c.output.parent.glob('*.partial.wav'))


@pytest.mark.parametrize('delay,expected_frames', [(0, 0), (1000, 51200), (.125, 6), ('0.125', 6), ('missing', 5120)])
def test_valid_start_delay_and_missing_default_are_frozen(recording, delay, expected_frames):
    c = recording
    detail = c.win.sequence_config[0]['seq1']['acq']['detail']
    if delay == 'missing': detail.pop('recording_start_delay_ms')
    else: detail['recording_start_delay_ms'] = delay
    requests = []
    c.bridge.start_capture = lambda owner, request, callback: (requests.append(request) or True)
    c.win.start_this_play()
    assert not c.warnings and len(requests) == 1
    assert requests[0].discard_initial_samples == expected_frames
    assert requests[0].target_samples == 5120


def test_actual_sqlite_insert_failure_keeps_file_without_success(recording):
    c = recording
    with sqlite3.connect(c.db) as db:
        db.execute("CREATE TRIGGER reject_recording BEFORE INSERT ON audio_data_table "
                   "BEGIN SELECT RAISE(FAIL, 'real recording insert failed'); END")
    c.win.start_this_play()
    wait_until(c, lambda: bool(c.analyses) or bool(c.warnings))
    assert c.warnings and '文件已保存、入库失败' in c.warnings[-1]
    assert 'real recording insert failed' in c.warnings[-1] and str(c.output) in c.warnings[-1]
    assert c.output.exists() and not c.analyses and c.win.current_recorded_count == 0
    with sqlite3.connect(c.db) as db:
        assert db.execute('SELECT COUNT(*) FROM audio_data_table').fetchone()[0] == 0


def test_actual_record_then_replay_reuses_existing_database_row(recording):
    c = recording
    c.win.start_this_play()
    wait_until(c, lambda: bool(c.analyses) or bool(c.warnings))
    assert not c.warnings and len(c.analyses) == 1
    with sqlite3.connect(c.db) as db:
        original_id = db.execute('SELECT audio_data_id FROM audio_data_table').fetchone()[0]
    c.sdk.reads.extend((c.raw[i:i+400].ravel(), len(c.raw[i:i+400])) for i in range(0, len(c.raw), 400))
    c.win.judge_play_and_record(is_replay=True)
    wait_until(c, lambda: len(c.analyses) == 2 or bool(c.warnings))
    assert not c.warnings and len(c.analyses) == 2
    assert c.win.current_recorded_count == c.win.last_play_count == 1
    with sqlite3.connect(c.db) as db:
        assert db.execute('SELECT audio_data_id FROM audio_data_table').fetchall() == [(original_id,)]


def test_actual_sqlite_replay_insert_failure_restores_prior_and_retains_new_file(recording):
    c = recording
    c.output.write_bytes(b'prior recording')
    c.win.last_play_count = c.win.current_recorded_count = 1
    with sqlite3.connect(c.db) as db:
        db.execute("CREATE TRIGGER reject_recording BEFORE INSERT ON audio_data_table "
                   "BEGIN SELECT RAISE(FAIL, 'real replay insert failed'); END")
    c.win.judge_play_and_record(is_replay=True)
    wait_until(c, lambda: bool(c.analyses) or bool(c.warnings))
    assert c.warnings and '文件已保存、入库失败' in c.warnings[-1]
    assert c.output.read_bytes() == b'prior recording'
    retained = list(c.output.parent.glob('*.replay.wav'))
    assert len(retained) == 1 and str(retained[0]) in c.warnings[-1]
    assert inspect_wav_calibration_metadata(retained[0]).source == 'vkinging'
    assert not c.analyses and c.win.current_recorded_count == 1


@pytest.mark.parametrize('update_failure', [False, True])
def test_replay_current_hardware_rate_updates_database_or_restores_prior(recording, update_failure):
    c = recording
    c.win.start_this_play()
    wait_until(c, lambda: bool(c.analyses) or bool(c.warnings))
    assert not c.warnings and len(c.analyses) == 1
    prior_audio = c.output.read_bytes()
    with sqlite3.connect(c.db) as db:
        prior_row = db.execute('SELECT * FROM audio_data_table').fetchall()
        if update_failure:
            db.execute("CREATE TRIGGER reject_update BEFORE UPDATE ON audio_data_table "
                       "BEGIN SELECT RAISE(FAIL, 'real replay update failed'); END")
    c.win.mic['samplerate'] = 64000
    HardwareManagementRepository(str(c.db)).update_asset_fields('machine-A', {'samplerate': 64000})
    values = np.arange(6464 * 32, dtype=np.float64).reshape(-1, 32) / 1000 + 2.5
    c.sdk.reads.extend((values[i:i+400].ravel(), len(values[i:i+400])) for i in range(0, len(values), 400))
    c.win.judge_play_and_record(is_replay=True)
    wait_until(c, lambda: len(c.analyses) == 2 or bool(c.warnings))
    assert c.win.current_recorded_count == c.win.last_play_count == 1
    with sqlite3.connect(c.db) as db:
        current_row = db.execute('SELECT * FROM audio_data_table').fetchall()
    if update_failure:
        assert 'real replay update failed' in c.warnings[-1] and '文件已保存、入库失败' in c.warnings[-1]
        assert current_row == prior_row and c.output.read_bytes() == prior_audio
        assert len(c.analyses) == 1
        retained = list(c.output.parent.glob('*.replay.wav'))
        assert len(retained) == 1 and str(retained[0]) in c.warnings[-1]
        assert sf.info(retained[0]).samplerate == 64000
    else:
        assert not c.warnings and len(c.analyses) == 2
        assert current_row[0][0] == prior_row[0][0] and current_row[0][3] == 64000
        actual, rate = sf.read(c.output, always_2d=True)
        assert rate == 64000
        np.testing.assert_array_equal(actual, values[64:])
