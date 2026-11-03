"""Exercise production analysis methods without importing the full application graph."""
import ast
import copy
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from base.data_struct.data_deal_struct import DataDealStruct
from base.vkinging_wav import VkingingVoltageWriter
from base.wav_calibration_metadata import (
    WavCalibrationInspection, inspect_wav_calibration_metadata,
    resolve_wav_channel_v2pa_factor,
)

ROOT = Path(__file__).resolve().parents[2]


def load_functions(relative, names, namespace=None, class_name=None):
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = tree.body if class_name is None else next(
        n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    selected = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = namespace if namespace is not None else {}
    for node in selected:
        node.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(path), 'exec'), ns)
    return ns


def signal_namespace():
    ns = dict(np=np, DataDealStruct=DataDealStruct,
              resolve_wav_channel_v2pa_factor=resolve_wav_channel_v2pa_factor,
              MessageBox=SimpleNamespace(warning=lambda *args: None),
              resolve_analysis_v2pa_factor_for_channel=lambda *args, **kwargs: pytest.fail('live DB queried'))
    load_functions('ui/signal_analysis_window.py',
                   ['_missing_file_calibration_warning_text', 'resolve_imported_wav_v2pa_factor',
                    'resolve_analysis_channel_signal', '_record_calibration_prerequisite_failure'], ns)
    return ns


def metadata():
    return {'backend': 'vkinging', 'schema_version': 1,
            'acquisition': {'hardware_id': 'frozen-device', 'model': None, 'unit': 'V',
                            'input_mode': 'IEPE', 'range_min': -5., 'range_max': 5.,
                            'sample_rate': 51200, 'bit_depth': 64},
            'recorded_channels': [
                {'wav_channel_index': i, 'physical_channel_index': ch, 'calibrated': True,
                 'v2pa_factor': factor, 'standard_spl': None,
                 'calibration_conditions': {'input_mode': 'IEPE', 'unit': 'V',
                                            'range_min': -10., 'range_max': 10.}}
                for i, (ch, factor) in enumerate(zip((15, 0, 8), (2., 3., 7.)))]}


def data_for(inspection):
    return SimpleNamespace(wav_calibration_inspection=inspection,
                           wav_calibration_metadata=inspection.metadata,
                           wav_calibration_metadata_authoritative=True,
                           wav_calibration_warning_shown=False,
                           analysis_prerequisite_failures={}, analysis_result_dict={},
                           store_wave_data_multi=np.array([[2., 4., 8.], [-3., -5., -9.]]),
                           sample_rate=51200)


@pytest.mark.parametrize('status', ['vkinging_invalid', 'unsupported'])
def test_declared_source_never_becomes_factor_one(status):
    data = data_for(WavCalibrationInspection('vkinging', status, reason='damaged JSON', unit='V'))
    with pytest.raises(ValueError, match='vkinging'):
        signal_namespace()['resolve_imported_wav_v2pa_factor'](data, 0)


def test_file_snapshot_uses_wav_column_and_applies_factor_once(tmp_path):
    path = tmp_path / 'voltage.wav'
    samples = np.array([[2., 4., 8.], [-3., -5., -9.]])
    writer = VkingingVoltageWriter(path, sample_rate=51200, bit_depth=64,
                                  physical_channels=(15, 0, 8), target_frames=2, metadata=metadata())
    writer.write_chunk(samples)
    writer.finalize()
    writer.commit()
    inspection = inspect_wav_calibration_metadata(path)
    data = data_for(inspection)
    ns = signal_namespace()
    for col, factor in enumerate((2., 3., 7.)):
        raw = ns['resolve_analysis_channel_signal'](data, {'analysis_channel': col}, 'SPL')
        resolved = ns['resolve_imported_wav_v2pa_factor'](data, col)
        np.testing.assert_array_equal(raw * resolved, samples[:, col] * factor)
    np.testing.assert_array_equal(data.store_wave_data_multi, samples)


def test_invalid_file_authority_precedes_pre_resolved_factor():
    ns = signal_namespace()
    load_functions('ui/signal_analysis_window.py', ['_resolve_v2pa_factor_for_analysis'], ns, 'AnalysisGraphWidget')
    widget = SimpleNamespace(data_struct=data_for(WavCalibrationInspection('vkinging', 'vkinging_invalid')),
                             _use_pre_resolved_v2pa_factor=True, v2pa_factor=999.,
                             analysis_config={'analysis_channel': 0}, title_name='required SPL')
    assert ns['_resolve_v2pa_factor_for_analysis'](widget) is False
    assert widget.v2pa_factor is None
    assert 'required SPL' in widget.data_struct.analysis_prerequisite_failures


def sequence_namespace():
    ns = signal_namespace()
    ns.update(copy=copy, GOLDEN_SAMPLE_RESULT_PATH_KEY='golden_sample_result_path')
    load_functions('ui/sequence/sequence_widget.py', ['_clear_wav_calibration_runtime_state'], ns)
    load_functions('ui/sequence/sequence_widget.py', ['instance_analysis_class', '_summarize_ok_ng'], ns, 'SequenceWindow')
    return ns


class FakeAnalysis:
    def __init__(self, title):
        self.title_name = title
        self.v2pa_factor = None

    def _resolve_v2pa_factor_for_analysis(self):
        return True


@pytest.mark.parametrize('kind', ['SPL', 'SPLF', 'FFT', 'PRB', 'LP', 'PD', 'ED', 'FBA', 'LOUD'])
@pytest.mark.parametrize('col,physical,factor', [(0, 15, 2.), (1, 0, 3.), (2, 8, 7.)])
def test_just_recorded_uses_frozen_column_including_ed_nested(kind, col, physical, factor):
    ns = sequence_namespace()
    ns['get_class_mapping'] = lambda: {kind: FakeAnalysis}
    window = SimpleNamespace(mode='RECORD_ONLY', data_struct=data_for(
        WavCalibrationInspection('vkinging', 'vkinging_valid', metadata(), unit='V')),
        analysis_window=[], analysis_config={}, _active_input_channels=[15, 0, 8],
        analysis_types_requiring_v2pa={kind})
    params = ({'head': {'config': {'analysis_channel': physical}}, 'tail': {'config': {}}}
              if kind == 'ED' else {'analysis_channel': physical})
    original = copy.deepcopy(params)
    ns['instance_analysis_class'](window, 'required', kind, params)
    assert len(window.analysis_window) == 1
    widget = window.analysis_window[0]
    assert widget.v2pa_factor == factor
    actual_config = widget.analysis_config['head']['config'] if kind == 'ED' else widget.analysis_config
    assert actual_config['analysis_channel'] == col
    assert params == original


def test_uncalibrated_required_item_cannot_disappear_into_overall_ok():
    ns = sequence_namespace()
    ns['get_class_mapping'] = lambda: {'SPL': FakeAnalysis, 'AI': FakeAnalysis}
    meta = metadata()
    meta['recorded_channels'][1].update(calibrated=False, v2pa_factor=None,
                                      standard_spl=None, calibration_conditions=None)
    data = data_for(WavCalibrationInspection('vkinging', 'vkinging_uncalibrated', meta, unit='V'))
    window = SimpleNamespace(mode='IMPORT_AUDIO', data_struct=data, analysis_window=[], analysis_config={},
                             analysis_types_requiring_v2pa={'SPL'})
    ns['instance_analysis_class'](window, 'required SPL', 'SPL', {'analysis_channel': 1})
    ns['instance_analysis_class'](window, 'raw AI', 'AI', {'analysis_channel': 1})
    data.analysis_result_dict['raw AI'] = (True, 0.)
    assert len(window.analysis_window) == 1
    assert data.analysis_prerequisite_failures
    assert ns['_summarize_ok_ng'](window) == (False, 'NG')
    data.analysis_prerequisite_failures.clear()
    assert ns['_summarize_ok_ng'](window) == (True, 'OK')


def test_clear_and_new_recording_do_not_leak_source_state():
    DataDealStruct._instance = None
    data = DataDealStruct()
    data.wav_calibration_inspection = WavCalibrationInspection('vkinging', 'vkinging_invalid')
    data.wav_physical_channel_indices = (15, 0, 8)
    data.analysis_prerequisite_failures = {'SPL': 'missing'}
    data.clear_data()
    assert data.wav_calibration_inspection is None
    assert data.wav_physical_channel_indices == ()
    assert data.analysis_prerequisite_failures == {}
    data.wav_calibration_inspection = WavCalibrationInspection('vkinging', 'unsupported')
    sequence_namespace()['_clear_wav_calibration_runtime_state'](data)
    assert data.wav_calibration_inspection is None

def import_namespace(path):
    from base.load_audio import load_audio_preserve_rate
    from base.wav_calibration_metadata import read_wav_calibration_metadata
    ns = sequence_namespace()
    warnings = []
    ns.update(SimpleNamespace=SimpleNamespace, DEFAULT_DIR='',
              QFileDialog=SimpleNamespace(getOpenFileName=lambda *args: (str(path), '')),
              load_audio_preserve_rate=load_audio_preserve_rate,
              read_wav_calibration_metadata=read_wav_calibration_metadata,
              inspect_wav_calibration_metadata=inspect_wav_calibration_metadata,
              MessageBox=SimpleNamespace(warning=lambda parent, title, text: warnings.append(text)))
    load_functions('ui/sequence/sequence_widget.py',
                   ['_clear_import_analysis_runtime_state', '_clear_data_struct_stimulus_runtime_state',
                    '_missing_file_calibration_warning_text'], ns)
    load_functions('ui/sequence/sequence_widget.py', ['import_audio_and_analyze'], ns, 'SequenceWindow')
    window = SimpleNamespace(sequence_config=[{'seq1': {'acq': {'mode': 'IMPORT_AUDIO', 'detail': {}}}}],
                             data_struct=SimpleNamespace(), analysis_config={'auto_analysis': False},
                             _clear_plot_area=lambda: None, plot_waveform_to_workspace=lambda *args: None,
                             data_btn=SimpleNamespace(setEnabled=lambda enabled: None),
                             default_logger=SimpleNamespace(warning=lambda text: None))
    return ns, window, warnings


def make_voltage_file(tmp_path):
    path = tmp_path / 'saved.wav'
    samples = np.array([[2., 4., 8.], [-3., -5., -9.]])
    writer = VkingingVoltageWriter(path, sample_rate=51200, bit_depth=64,
                                  physical_channels=(15, 0, 8), target_frames=2, metadata=metadata())
    writer.write_chunk(samples)
    writer.finalize()
    writer.commit()
    return path, samples


def rewrite_chunks(path, mutation):
    data = path.read_bytes()
    offset, parts = 12, []
    while offset < len(data):
        name, size = struct.unpack_from('<4sI', data, offset)
        payload = mutation(name, data[offset+8:offset+8+size])
        if payload is not None:
            parts.append(name + struct.pack('<I', len(payload)) + payload + b'\0' * (len(payload) % 2))
        offset += 8 + size + size % 2
    payload = b'WAVE' + b''.join(parts)
    path.write_bytes(b'RIFF' + struct.pack('<I', len(payload)) + payload)


@pytest.mark.parametrize('damage', ['none', 'remove_json', 'bad_json', 'schema', 'remove_marker'])
def test_actual_import_preserves_source_and_raw_voltage(tmp_path, damage):
    path, samples = make_voltage_file(tmp_path)
    def mutation(name, payload):
        if name == b'LIST':
            if damage == 'remove_json':
                return None
            if damage == 'bad_json':
                return payload.replace(b'mic_calibration={', b'mic_calibration=!')
            if damage == 'schema':
                return payload.replace(b'"schema_version":1', b'"schema_version":9')
        if name == b'vkin' and damage == 'remove_marker':
            return None
        return payload
    rewrite_chunks(path, mutation)
    ns, window, warnings = import_namespace(path)
    ns['import_audio_and_analyze'](window)
    inspection = getattr(window.data_struct, 'wav_calibration_inspection', None)
    assert inspection is not None
    assert inspection.source == 'vkinging'
    np.testing.assert_array_equal(window.data_struct.store_wave_data_multi, samples)
    if damage == 'none':
        assert window.data_struct.wav_physical_channel_indices == (15, 0, 8)
        assert ns['resolve_imported_wav_v2pa_factor'](window.data_struct, 1) == 3.
    else:
        with pytest.raises(ValueError):
            ns['resolve_imported_wav_v2pa_factor'](window.data_struct, 1)
        assert warnings


def test_malformed_legacy_riff_cannot_borrow_factor_one_import(tmp_path):
    import soundfile as sf
    path = tmp_path / 'broken.wav'
    sf.write(path, [2., -3.], 48000, subtype='FLOAT')
    data = bytearray(path.read_bytes())
    struct.pack_into('<I', data, 4, len(data) + 100)
    path.write_bytes(data)
    ns, window, warnings = import_namespace(path)
    ns['import_audio_and_analyze'](window)
    assert getattr(window.data_struct, 'store_wave_data_multi', None) is None
    assert warnings


def test_attach_verified_recording_snapshot_owns_nested_mapping():
    original = metadata()
    data = SimpleNamespace()
    DataDealStruct.set_wav_calibration_inspection(data,
        WavCalibrationInspection('vkinging', 'vkinging_valid', original, unit='V'))
    original['recorded_channels'][0]['v2pa_factor'] = 999.
    assert data.wav_physical_channel_indices == (15, 0, 8)
    assert signal_namespace()['resolve_imported_wav_v2pa_factor'](data, 0) == 2.


def test_blocked_required_item_prevents_partial_success_publication():
    ns = sequence_namespace()
    load_functions('ui/sequence/sequence_widget.py',
                   ['_maybe_show_analysis_result_summary', '_maybe_export_excel_results', '_can_output_ok_ng'],
                   ns, 'SequenceWindow')
    data = SimpleNamespace(analysis_result_dict={'raw AI': (True, 0.)},
                           analysis_prerequisite_failures={'required SPL': 'missing factor'})
    window = SimpleNamespace(data_struct=data,
                             analysis_config={'display_sequence': ['raw AI'], 'raw AI': {'type': 'AI'}})
    assert ns['_can_output_ok_ng'](window)[0] is False
    # Both must stop before constructing a partial-success summary or touching export state.
    ns['_maybe_show_analysis_result_summary'](window, 0, 0)
    ns['_maybe_export_excel_results'](window)

@pytest.mark.parametrize('authoritative', [True, False])
def test_ed_source_authority_cannot_be_bypassed_by_pre_resolved_factor(authoritative):
    ns = signal_namespace()
    load_functions('ui/signal_analysis_window.py',
                   ['_prepare_pipeline_context', 'calculate_pipeline_pd_pm'], ns, 'PipelinePdPm')
    ns['get_class_mapping'] = lambda: {'PD': FakeAnalysis, 'PM': FakeAnalysis}
    data = data_for(WavCalibrationInspection('vkinging', 'vkinging_valid', metadata(), unit='V'))
    data.wav_calibration_metadata_authoritative = authoritative
    seen = []
    widget = SimpleNamespace(data_struct=data,
                             analysis_config={'head': {'config': {'analysis_channel': 2}},
                                              'tail': {'config': {}}},
                             _use_pre_resolved_v2pa_factor=True, v2pa_factor=999., title_name='ED',
                             windowTitle=lambda: 'ED', _render_plots=lambda *args: None,
                             _update_table=lambda *args: None,
                             _summarize_and_notify=lambda *args: {'factor': widget.v2pa_factor})
    widget._prepare_pipeline_context = lambda: ns['_prepare_pipeline_context'](widget)
    widget._execute_pd = lambda *args: (seen.append(widget.v2pa_factor) or {}, [])
    result = ns['calculate_pipeline_pd_pm'](widget)
    assert result == {'factor': 7.}
    assert seen == [7.]
    data.wav_calibration_inspection = WavCalibrationInspection('vkinging', 'unsupported')
    widget._use_pre_resolved_v2pa_factor = True
    widget.v2pa_factor = 999.
    assert ns['calculate_pipeline_pd_pm'](widget) is None
    assert seen == [7.]
    assert data.analysis_prerequisite_failures


def test_real_qt_fft_uses_raw_column_and_exactly_one_frozen_factor(monkeypatch):
    from PyQt5.QtWidgets import QApplication
    from base.core_algorithm.response import FftAnalyzer
    from ui import signal_analysis_window as signal_ui
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(signal_ui, 'resolve_analysis_v2pa_factor_for_channel',
                        lambda *args, **kwargs: pytest.fail('current DB must not interpret old voltage'))
    monkeypatch.setattr(signal_ui.MessageBox, 'warning', lambda *args: None)
    data = data_for(WavCalibrationInspection('vkinging', 'vkinging_valid', metadata(), unit='V'))
    values = np.sin(2 * np.pi * 800 * np.arange(4096) / 51200)
    data.store_wave_data_multi = values[:, None] * np.array([2., 4., 8.])
    original = data.store_wave_data_multi.copy()
    for column, factor in enumerate((2., 3., 7.)):
        widget = signal_ui.FftAnalysis('FFT')
        widget.data_struct = data
        widget.analysis_config = {'analysis_channel': column, 'n_fft': 1024,
                                  'focus_range_enabled': False, 'x_axis_scale': 'linear'}
        widget.v2pa_factor = 999.
        widget._use_pre_resolved_v2pa_factor = True
        result = widget.calculate_fft()
        raw = FftAnalyzer().analyze(np.asarray(original[:, column], dtype=np.float32),
                                   51200, n_fft=1024, v2pa_factor=1.)
        peak = np.argmax(raw.spectrum_db)
        assert result['fft_db'][peak] == pytest.approx(raw.spectrum_db[peak] + 20*np.log10(factor), abs=1e-8)
        assert widget.v2pa_factor == factor
        widget.close()
    np.testing.assert_array_equal(data.store_wave_data_multi, original)
    assert app is not None


def test_corrupt_nested_metadata_still_allows_raw_import(tmp_path):
    path, samples = make_voltage_file(tmp_path)
    def mutation(name, payload):
        if name == b'LIST':
            payload = bytearray(payload)
            struct.pack_into('<I', payload, 8, len(payload)+100)
            return bytes(payload)
        return payload
    rewrite_chunks(path, mutation)
    ns, window, warnings = import_namespace(path)
    ns['import_audio_and_analyze'](window)
    np.testing.assert_array_equal(window.data_struct.store_wave_data_multi, samples)
    assert window.data_struct.wav_calibration_inspection.source == 'vkinging'
    with pytest.raises(ValueError):
        ns['resolve_imported_wav_v2pa_factor'](window.data_struct, 0)
    assert warnings

@pytest.mark.parametrize('mode,selected', [('RECORD_ONLY', [15, 0, 8]), ('IMPORT_AUDIO', [0, 1, 2])])
def test_run_batch_resolves_only_frozen_file_factors(mode, selected):
    from base.soundcard_calibration_manager import AnalysisV2paBatch
    ns = sequence_namespace()
    ns.update(AnalysisV2paBatch=lambda resolver=None: AnalysisV2paBatch(
                  resolver=resolver or ns['resolve_analysis_v2pa_factor_for_channel']),
              ui_style_const=SimpleNamespace(scale_size_px=lambda value: value))
    load_functions('ui/sequence/sequence_widget.py', ['run'], ns, 'SequenceWindow')
    data = data_for(WavCalibrationInspection('vkinging', 'vkinging_valid', metadata(), unit='V'))
    seen = []
    window = SimpleNamespace(mode=mode, data_struct=data, analysis_window=[],
                             sequence_config=[{'seq1': {'acq': {'mode': mode}}}],
                             analysis_config={'display_sequence': ['a', 'b', 'c'],
                                              **{key: {'type': 'SPL', 'analysis_channel': channel}
                                                 for key, channel in zip('abc', selected)}},
                             _analysis_result_summary_window=None, count_board=SimpleNamespace(mode='normal'),
                             screen=lambda: SimpleNamespace(size=lambda: SimpleNamespace(width=lambda: 1200,
                                                                                         height=lambda: 900)),
                             _handle_post_analysis_exports=lambda: None,
                             _maybe_show_analysis_result_summary=lambda *args: None,
                             _send_tcp_analysis_result_callback=lambda: None)
    def prepare(key, kind, params):
        seen.append(window._analysis_v2pa_batch.resolve(params['analysis_channel']).factor)
    window.instance_analysis_class = prepare
    ns['run'](window)
    assert seen == [2., 3., 7.]


@pytest.mark.parametrize('kind', ['HD', 'RB', 'FR', 'AI', 'Spec', 'PM'])
def test_raw_analysis_still_instantiates_with_invalid_voltage_calibration(kind):
    ns = sequence_namespace()
    ns['get_class_mapping'] = lambda: {kind: FakeAnalysis}
    data = data_for(WavCalibrationInspection('vkinging', 'vkinging_invalid'))
    window = SimpleNamespace(mode='IMPORT_AUDIO', data_struct=data,
                             analysis_window=[], analysis_config={}, analysis_types_requiring_v2pa={'HD', 'RB'})
    ns['instance_analysis_class'](window, 'raw', kind, {'analysis_channel': 2})
    assert len(window.analysis_window) == 1
    assert data.analysis_prerequisite_failures == {}


@pytest.mark.parametrize('invalid_metadata', [False, True])
def test_ordinary_wav_keeps_legacy_warning_and_factor_one(tmp_path, invalid_metadata):
    import soundfile as sf
    path = tmp_path / 'old.wav'
    sf.write(path, [2., -3.], 48000, subtype='FLOAT')
    if invalid_metadata:
        comment = b'mic_calibration=broken\0'
        comment_chunk = b'ICMT' + struct.pack('<I', len(comment)) + comment + b'\0' * (len(comment) % 2)
        payload = b'INFO' + comment_chunk
        data = bytearray(path.read_bytes())
        data.extend(b'LIST' + struct.pack('<I', len(payload)) + payload)
        struct.pack_into('<I', data, 4, len(data)-8)
        path.write_bytes(data)
    ns, window, warnings = import_namespace(path)
    ns['import_audio_and_analyze'](window)
    assert window.data_struct.wav_calibration_inspection.source == 'legacy'
    assert ns['resolve_imported_wav_v2pa_factor'](window.data_struct, 0) == 1.
    assert warnings == ['该音频文件未包含有效校准数据，分析结果仅供参考。']
    np.testing.assert_array_equal(window.data_struct.store_wave_data, [2., -3.])


def tcp_sender_window(data):
    """Bind the production sender/payload methods to an in-memory transport."""
    import json
    import os
    from datetime import datetime

    sent = []
    logger = SimpleNamespace(info=lambda text: None, warning=lambda text: None,
                             error=lambda text: pytest.fail(text))
    transport = SimpleNamespace(send_to_current_client=lambda message: sent.append(json.loads(message)) or True)
    ns = sequence_namespace()
    ns.update(json=json, os=os,
              datetime=SimpleNamespace(now=lambda: datetime(2026, 11, 2, 12, 30, 45, 123000)),
              SequenceWindow=SimpleNamespace(tcp_server=transport))
    names = ['_get_tcp_callback_file_name', '_get_tcp_callback_label',
             '_build_tcp_analysis_result_payload', '_send_tcp_analysis_result_callback']
    load_functions('ui/sequence/sequence_widget.py', names, ns, 'SequenceWindow')
    window = SimpleNamespace(data_struct=data, tcp_flag=True,
                             recorded_path='saved/frozen-voltage.wav', default_logger=logger)
    for name in ['_summarize_ok_ng', *names]:
        setattr(window, name, ns[name].__get__(window, type(window)))
    return window, sent


def test_tcp_sender_suppresses_blocked_calibration_with_passing_raw_item():
    data = data_for(WavCalibrationInspection('vkinging', 'vkinging_uncalibrated', metadata(), unit='V'))
    data.analysis_prerequisite_failures = {'required SPL': 'uncalibrated'}
    data.analysis_result_dict = {'raw AI': (True, 0.)}
    window, sent = tcp_sender_window(data)

    window._send_tcp_analysis_result_callback()

    assert sent == []
    # Once all requested measurements are available, publication resumes normally.
    data.analysis_prerequisite_failures.clear()
    data.analysis_result_dict['required SPL'] = (True, 0.)
    window._send_tcp_analysis_result_callback()
    assert sent == [{'TimeStamp': '2026-11-02 12:30:45,123', 'Label': 'OK',
                     'FileName': 'frozen-voltage.wav'}]


@pytest.mark.parametrize('has_failure_field', [True, False])
@pytest.mark.parametrize('results,label', [({'SPL': (True, 0.)}, 'OK'),
                                          ({'SPL': (False, 1.)}, 'NG'),
                                          ({}, 'not_labeled')])
def test_tcp_sender_preserves_complete_and_legacy_payloads(has_failure_field, results, label):
    data = SimpleNamespace(analysis_result_dict=results)
    if has_failure_field:
        data.analysis_prerequisite_failures = {}
    window, sent = tcp_sender_window(data)

    window._send_tcp_analysis_result_callback()

    assert sent == [{'TimeStamp': '2026-11-02 12:30:45,123', 'Label': label,
                     'FileName': 'frozen-voltage.wav'}]
