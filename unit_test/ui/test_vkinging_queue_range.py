import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PyQt5.QtWidgets import QApplication
from base.acquisition_recording_defaults import normalize_record_only_detail

@pytest.fixture(scope='module')
def qapp(): return QApplication.instance() or QApplication([])

def mic():
    return dict(name='daq', hardware_id='A', hostapi_name='vkinging', samplerate=51200, bit_depth=32)


def test_range_editor_preserves_queue_state_and_disables_unsupported_modes(qapp):
    from ui.acquisition_config_window import RecordConfigWindow
    a = RecordConfigWindow({'vkinging_range_index': 2}, mic=mic())
    b = RecordConfigWindow({'vkinging_range_index': 5}, mic=mic())
    assert a.vkinging_range_combo.currentData() == 2
    assert a.vkinging_range_combo.count() == 7
    a.vkinging_range_combo.setCurrentIndex(6)
    assert a._collect_record_detail()['vkinging_range_index'] == 6
    assert b._collect_record_detail()['vkinging_range_index'] == 5
    assert not a.monitor_checkbox.isEnabled()
    assert not a.streaming_recording_checkbox.isEnabled()
    reopened = RecordConfigWindow(a._collect_record_detail(), mic=mic())
    assert reopened.vkinging_range_combo.currentData() == 6
    for w in (a,b,reopened): w.close()


def test_old_missing_range_and_soundcard_edit_preserves_explicit_value(qapp):
    from ui.acquisition_config_window import RecordConfigWindow
    old = RecordConfigWindow({}, mic=mic())
    assert old._collect_record_detail()['vkinging_range_index'] == 0
    soundcard = RecordConfigWindow({'vkinging_range_index': 4}, mic=dict(name='card', samplerate=48000))
    assert soundcard._collect_record_detail()['vkinging_range_index'] == 4
    old.close(); soundcard.close()

@pytest.mark.parametrize('invalid', [True, 2.0, '2', None, -1, 7])
def test_invalid_range_is_preserved_and_rejected(qapp, invalid):
    from ui.acquisition_config_window import RecordConfigWindow
    window = RecordConfigWindow({'vkinging_range_index': invalid}, mic=mic())
    with pytest.raises(ValueError, match='vkinging_range_index'):
        window._collect_record_detail()
    window.close()

def test_queue_json_copy_and_default_write_failure(qapp, tmp_path, monkeypatch):
    import copy
    import json
    from base.data_struct.sequence_data import SequenceData
    from base.acquisition_recording_defaults import save_acquisition_default, load_acquisition_defaults
    from ui import acquisition_config_window as module
    a=SequenceData('seq1'); a.mode='RECORD_ONLY'; a.detail={'vkinging_range_index':2}
    b=copy.deepcopy(a); b.detail={'vkinging_range_index':5}
    queue_a=tmp_path/'a.json';queue_b=tmp_path/'b.json'
    queue_a.write_text(json.dumps([a.config_info]));queue_b.write_text(json.dumps([b.config_info]))
    detail_a=json.loads(queue_a.read_text())[0]['seq1']['acq']['detail']
    detail_b=json.loads(queue_b.read_text())[0]['seq1']['acq']['detail']
    window=module.RecordConfigWindow(detail_a,mic=mic())
    defaults=tmp_path/'defaults.json';assert save_acquisition_default('RECORD_ONLY',detail_b,path=defaults)
    window.vkinging_range_combo.setCurrentIndex(6)
    monkeypatch.setattr(module,'save_acquisition_default',lambda *a,**k:False)
    reported=[];monkeypatch.setattr(window,'_show_default_save_result',reported.append)
    window.on_default_btn_clicked()
    assert reported==[False]
    assert load_acquisition_defaults(defaults)['RECORD_ONLY']['vkinging_range_index']==5
    assert detail_a['vkinging_range_index']==2
    assert detail_b['vkinging_range_index']==5
    window.close()

def _queue_save_window(tmp_path, detail, mode='RECORD_ONLY', selected_mic=None):
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    from base.data_struct.sequence_data import SequenceData
    from base.vkinging_input import is_vkinging_device, validate_range_index
    tree=ast.parse(Path('ui/operation_sequence.py').read_text(encoding='utf-8'))
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='AnalysisModelSelect')
    names={'save_btn_clicked','ok_btn_clicked','_validate_queue_input'}
    methods=[node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name in names]
    calls=[];warnings=[]
    def call(name, result=None):
        return lambda *args,**kwargs:(calls.append(name),result)[1]
    namespace=dict(
        DEFAULT_DIR=str(tmp_path),
        os=SimpleNamespace(path=os.path,makedirs=call('makedirs')),
        QFileDialog=SimpleNamespace(getSaveFileName=call('file_dialog',(str(tmp_path/'queue.json'),''))),
        MessageBox=SimpleNamespace(warning=lambda *args:warnings.append(args)),
        LoadUiConfig=SimpleNamespace(
            save_sequence_config_to_json=call('queue_write',True),
            append_sequence_config_registry_entry=call('registry_append'),
            _load_sequence_config_registry=call('registry_read',{}),
            ensure_sequence_config_registry_field=call('registry_ensure'),
            update_using_config_path=call('registry_update')),
        is_vkinging_device=is_vkinging_device,
        validate_range_index=validate_range_index)
    exec(compile(ast.Module(body=methods,type_ignores=[]),'queue-save','exec'),namespace)
    item=SequenceData('seq1');item.mode=mode;item.detail=dict(detail)
    window=SimpleNamespace(mic=mic() if selected_mic is None else selected_mic,
                           select_list=SimpleNamespace(config=[item]),
                           format_config_data=lambda config:[entry.config_info for entry in config],
                           _new_target_path_selected=False,using_config_path=str(tmp_path/'queue.json'),
                           config_saved=False,accept=call('accepted'),close=call('closed'))
    if '_validate_queue_input' in namespace:
        window._validate_queue_input=namespace['_validate_queue_input'].__get__(window)
    return window,namespace,calls,warnings


@pytest.mark.parametrize('method',['save_btn_clicked','ok_btn_clicked'])
@pytest.mark.parametrize('invalid',[True,2.0,-1,7])
def test_direct_queue_save_rejects_invalid_vkinging_range_before_writes(tmp_path, method, invalid):
    window,namespace,calls,warnings=_queue_save_window(tmp_path,{'vkinging_range_index':invalid})
    namespace[method](window)
    assert warnings
    assert calls==[]
    assert window.config_saved is False


@pytest.mark.parametrize('method',['save_btn_clicked','ok_btn_clicked'])
@pytest.mark.parametrize('detail,selected_mic', [({},None),({'vkinging_range_index':True},{'name':'soundcard'})])
def test_direct_queue_save_keeps_old_missing_range_and_ordinary_card_behavior(tmp_path, method, detail, selected_mic):
    window,namespace,calls,warnings=_queue_save_window(tmp_path,detail,selected_mic=selected_mic)
    namespace[method](window)
    assert not warnings
    assert 'queue_write' in calls
    assert window.select_list.config[0].detail==detail

@pytest.mark.parametrize('method',['save_btn_clicked','ok_btn_clicked'])
@pytest.mark.parametrize('mode,detail',[('PLAY_AND_RECORD',{}),('RECORD_ONLY',{'monitor_playback':True})])
def test_direct_queue_save_rejects_unsupported_vkinging_modes(tmp_path, method, mode, detail):
    window,namespace,calls,warnings=_queue_save_window(tmp_path,detail,mode=mode)
    namespace[method](window)
    assert warnings and not calls and not window.config_saved
