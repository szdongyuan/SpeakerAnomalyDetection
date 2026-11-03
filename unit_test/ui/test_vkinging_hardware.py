import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PyQt5.QtWidgets import QApplication, QDialog
from base.vkinging_service import VkingingEvent
from base.vkinging_discovery import DiscoveryResult
import ui.hardware_window as hardware_window_module

@pytest.fixture(scope='module')
def qapp():
    return QApplication.instance() or QApplication([])

class FakeService:
    def __init__(self):
        self.state = 'idle'
        self.events = []
        self.requests = []
        self.ticks = []
    def submit_discovery(self, request_id):
        if self.state != 'idle': return False
        self.requests.append(request_id)
        self.state = 'discovering'
        return True
    def poll_events(self):
        result, self.events = self.events, []
        return result
    def tick(self, now): self.ticks.append(now)
    def cancel(self, request_id): self.cancelled = request_id; return True
    def close(self): self.state = 'closed'


def device(name='renamed', channels=(0, 8, 15, 31)):
    return dict(hostapi_name='vkinging', hardware_id='machine-A', name=name,
                physical_channels=channels, max_input_channels=max(channels)+1,
                max_output_channels=0, samplerate=51200, bit_depth=32)


def test_bridge_request_subscription_and_reentrant_current_state(qapp):
    from ui.vkinging_bridge import VkingingBridge
    service = FakeService()
    bridge = VkingingBridge(service)
    owner = QDialog()
    seen = []
    first = bridge.request_discovery(owner, lambda e: (seen.append(e), bridge.request_discovery(owner)))
    service.state = 'idle'
    service.events = [VkingingEvent(first, 'succeeded', 'idle', 'releasing', value=DiscoveryResult((device(),)), resources_released=True)]
    bridge.poll()
    assert len(seen) == 1
    assert len(service.requests) == 2
    assert bridge.state == 'discovering'
    assert bridge.devices[0]['hardware_id'] == 'machine-A'
    assert bridge.timer.isActive()
    bridge.close()


def test_closed_owner_ignores_late_result_and_release_still_polled(qapp):
    from ui.vkinging_bridge import VkingingBridge
    service = FakeService(); bridge = VkingingBridge(service)
    owner = QDialog(); seen = []
    request_id = bridge.request_discovery(owner, seen.append)
    owner.reject()
    service.events = [VkingingEvent(request_id, 'succeeded', 'idle', 'releasing', value=DiscoveryResult((device(),)), resources_released=True)]
    service.state = 'idle'; bridge.poll()
    assert seen == []
    assert bridge.timer.isActive()
    service.state = 'unavailable'; bridge.poll()
    assert bridge.busy
    service.events = [VkingingEvent(request_id, 'released', 'idle', 'releasing', resources_released=True)]
    service.state = 'idle'; bridge.poll()
    assert not bridge.busy
    bridge.close()


def test_vkinging_match_uses_identity_and_preserves_output_on_api_change(qapp):
    from ui.hardware_window import HardwareSelectionController, HardwareSelectionModel, HardwareSelectionState
    asset = dict(device('old'), device_name='old')
    assert HardwareSelectionController._match_registered_asset(asset, {'vkinging': [device()]}) == ('ok', device())
    output = dict(name='speaker', hostapi_name='ASIO')
    model = HardwareSelectionModel(HardwareSelectionState(speaker_device=output))
    model.set_api('vkinging')
    assert model.state.speaker_device == output


def test_registration_restore_key_is_stable_machine_id():
    from ui.hardware_management_window import HardwareRegistrationDialog
    key = HardwareRegistrationDialog._device_restore_key(device('old'))
    assert HardwareRegistrationDialog._device_matches_restore_key(device('new'), key)

def test_capture_bridge_uses_same_lifecycle_subscription(qapp):
    from ui.vkinging_bridge import VkingingBridge
    from types import SimpleNamespace
    service = FakeService()
    service.start_capture = lambda request: service.requests.append(request.request_id) or True
    bridge = VkingingBridge(service); owner = QDialog(); seen=[]
    assert bridge.start_capture(owner, SimpleNamespace(request_id='capture-A'), seen.append)
    service.events=[VkingingEvent('capture-A', 'succeeded', 'idle', 'committing', value={'path':'x'}, resources_released=True)]
    bridge.poll()
    assert len(seen) == 1
    owner.reject(); bridge.close()


def test_vkinging_preflight_rejects_all_soundcard_routes(monkeypatch):
    # Match nearby calibration tests: execute the real methods without unrelated
    # HID/ML startup dependencies imported by the full application module.
    import ast
    import math
    from pathlib import Path
    from types import SimpleNamespace
    from base.vkinging_input import is_vkinging_device, resolve_vkinging_input_config
    tree=ast.parse(Path('ui/sequence/sequence_widget.py').read_text(encoding='utf-8'))
    helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_vkinging_preflight_error')
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SequenceWindow')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='checked_work_status_message')
    warnings=[]
    namespace=dict(math=math,is_vkinging_device=is_vkinging_device,resolve_vkinging_input_config=resolve_vkinging_input_config,
                   MessageBox=SimpleNamespace(warning=lambda *a:warnings.append(a)))
    exec(compile(ast.Module(body=[helper,method],type_ignores=[]),'preflight','exec'),namespace)
    for mode, detail in [('RECORD_ONLY', {}), ('PLAY_AND_RECORD', {}), ('RECORD_ONLY', {'monitor_playback':True})]:
        win=SimpleNamespace(sequence_config=[{'seq1':{'acq':{'mode':mode,'detail':detail}}}], mic=device(),
                            mic_channels=[0,8,15,31], audio_devices_available=True)
        assert namespace['checked_work_status_message'](win)
    assert len(warnings) == 3


def test_completed_empty_discovery_is_unavailable_not_pending(tmp_path, monkeypatch):
    from unit_test.base.test_vkinging_hardware import device as runtime, register
    from base.db_manager import DataSave
    from base.hardware_management import HardwareManagementRepository
    import base.sound_device_manager as module
    path=tmp_path/'system.db'; db=DataSave(str(path)); db.create_system_tables(); db.close()
    repo=HardwareManagementRepository(str(path)); asset=register(repo)
    monkeypatch.setattr(module, 'HardwareManagementRepository', lambda: repo)
    saved={'version':3,'mic':dict(asset,name='old'),'speaker':None,'mic_channels':[31]}
    monkeypatch.setattr(module.SoundDeviceManager, 'load_selected_devices', staticmethod(lambda:saved))
    manager=module.SoundDeviceManager()
    empty=manager.complete_vkinging_startup_discovery(())
    assert empty['device_available'] is False
    assert not empty.get('validation_pending',False)
    restored=manager.complete_vkinging_startup_discovery((runtime(name='renamed'),))
    assert restored['device_available'] is True
    assert restored['mic']['name']=='renamed'
    assert restored['mic_channels']==[31]


def test_registration_selection_sparse_channels_and_busy_mutations(qapp, tmp_path, monkeypatch):
    from base.db_manager import DataSave
    from base.hardware_management import HardwareManagementRepository
    import base.sound_device_manager as devices
    import ui.hardware_management_window as management
    from ui.hardware_window import HardwareSelectionModel, HardwareSelectionController, HardwareSelectionView, HardwareSelectionState
    from ui.vkinging_bridge import VkingingBridge
    path=tmp_path/'system.db'; db=DataSave(str(path)); db.create_system_tables(); db.close()
    repo=HardwareManagementRepository(str(path)); asset=repo.register_asset(device('old'), 'vkinging', 'DAQ', 51200)
    service=FakeService(); bridge=VkingingBridge(service)
    bridge.discovery_complete=True; bridge.devices=(device(),)
    monkeypatch.setattr(devices, 'AUDIO_DEVICE_CONFIG_PATH',str(tmp_path/'selected.json'))
    monkeypatch.setattr(devices.SoundDeviceManager,'change_default_device',staticmethod(lambda *a:pytest.fail('PortAudio default')))
    monkeypatch.setattr(devices.SoundDeviceManager,'change_default_input_device',staticmethod(lambda *a:pytest.fail('PortAudio input')))
    output=dict(hardware_type='audio_interface', name='speaker', device_name='speaker', display_name='Output', hostapi_name='ASIO', hardware_id='out', index=7, samplerate=48000, bit_depth=32, latency_ms=100)
    state=HardwareSelectionState(speaker_device=output,mic_device=asset,mic_channels=[0,31],api_name='vkinging')
    view=HardwareSelectionView(); model=HardwareSelectionModel(state,repository=repo)
    controller=HardwareSelectionController(model,view,vkinging_bridge=bridge)
    request=service.requests[-1]; service.state='idle';service.events=[VkingingEvent(request,'succeeded','idle','releasing',value=DiscoveryResult((device(),)),resources_released=True)]
    bridge.poll()
    assert model.channels_for_device(asset,'mic')==[0,8,15,31]
    assert view.mic_channel_table.model().rowCount()==4
    controller._on_ok_clicked()
    assert view.result()==QDialog.Accepted
    assert model.state.speaker_device==output
    assert model.state.mic_channels==[0,31]
    assert model.state.mic_device['name']=='renamed'
    table=management.HardwareManagementTableModel(repo,mutation_allowed=lambda:not bridge.busy)
    index=table.index(0,table.column_index('samplerate'))
    assert table.setData(index,64000)
    service.state='unavailable'
    assert not table.setData(index,51200)
    assert repo.get_asset('machine-A')['samplerate']==64000
    bridge.close(); view.close()

@pytest.mark.parametrize('channels', [tuple(range(32)), (0,8,15,31)])
def test_real_route_choices_and_output_survive_switching_driver(qapp, tmp_path, monkeypatch, channels):
    from base.db_manager import DataSave
    from base.hardware_management import HardwareManagementRepository
    from ui.hardware_window import HardwareSelectionModel, HardwareSelectionController, HardwareSelectionView, HardwareSelectionState
    path=tmp_path/'system.db'; db=DataSave(str(path)); db.create_system_tables(); db.close()
    repo=HardwareManagementRepository(str(path))
    daq=repo.register_asset(device(channels=channels),'vkinging','DAQ',51200)
    card=dict(name='card',index=1,max_input_channels=2,max_output_channels=2)
    output=repo.register_asset(card,'ASIO','speaker',48000)
    output=repo.get_asset(output['hardware_id'])
    state=HardwareSelectionState(speaker_device=output,api_name='ASIO')
    view=HardwareSelectionView();model=HardwareSelectionModel(state,repository=repo)
    controller=HardwareSelectionController(model,view)
    controller._on_api_changed('vkinging')
    assert model.state.speaker_device==output
    controller._on_mic_device_checked(daq)
    assert view.mic_channel_table.model().rowCount()==len(channels)
    controller._on_api_changed('ASIO')
    assert model.state.speaker_device is None
    assert view.speaker_device_table.checked_payload() is None
    view.close()


def test_service_accepts_native_project_logger():
    import logging
    from base.vkinging_service import VkingingService
    logger=logging.getLogger('task7-native-logger')
    service=VkingingService(logger=logger)
    try:
        assert service._logger is logger
    finally:
        service.close(); service._worker.join(timeout=2)
        assert not service._worker.is_alive()

# Reuse the existing application's dependency fakes, while exercising MainWindow.
from unit_test.ui.test_main_window_startup_device_recovery import main_window_module


def test_main_window_shares_owner_and_restores_pending_input(qapp, main_window_module, monkeypatch):
    from ui.vkinging_bridge import VkingingBridge
    from types import SimpleNamespace
    service=FakeService(); bridge=VkingingBridge(service)
    output={'name':'speaker','hostapi_name':'ASIO','index':7}
    pending=device('old')
    restored=dict(pending,name='renamed')
    manager=SimpleNamespace(
        get_startup_devices=lambda:dict(mic=pending,speaker=output,mic_channels=[31],device_available=False,validation_pending=True),
        complete_vkinging_startup_discovery=lambda snapshots:dict(mic=restored,speaker=output,mic_channels=[31],device_available=bool(snapshots)))
    monkeypatch.setattr(main_window_module,'SoundDeviceManager',lambda:manager)
    monkeypatch.setattr(main_window_module.MainWindow,'init_ui',lambda self:None)
    window=main_window_module.MainWindow(vkinging_bridge=bridge)
    assert window.vkinging_service is service
    assert window.mic==pending and window.speaker==output
    window._schedule_startup_device_recovery_if_needed()
    request=service.requests[-1]
    service.state='idle';service.events=[VkingingEvent(request,'succeeded','idle','releasing',value=DiscoveryResult((device(),)),resources_released=True)]
    bridge.poll()
    assert window.device_workflow_available
    assert window.mic['name']=='renamed'
    assert window.speaker==output and window.mic_channels==[31]
    assert not window._vkinging_validation_pending
    bridge.close()


def test_qtimer_delivers_fake_discovery(qapp):
    from PyQt5.QtCore import QEventLoop, QTimer
    from ui.vkinging_bridge import VkingingBridge
    service=FakeService();bridge=VkingingBridge(service)
    owner=QDialog();loop=QEventLoop();seen=[]
    request=bridge.request_discovery(owner,lambda event:(seen.append(event),loop.quit()))
    def complete():
        service.state='idle'
        service.events=[VkingingEvent(request,'succeeded','idle','releasing',value=DiscoveryResult((device(),)),resources_released=True)]
    QTimer.singleShot(0,complete)
    deadline=QTimer();deadline.setSingleShot(True);deadline.timeout.connect(loop.quit);deadline.start(2000)
    loop.exec_()
    deadline.stop()
    assert len(seen)==1
    assert bridge.devices[0]['physical_channels']==(0,8,15,31)
    bridge.close();owner.close()

def test_async_registration_parameters_and_busy_delete(qapp, tmp_path, monkeypatch):
    from base.db_manager import DataSave
    from base.hardware_management import HardwareManagementRepository
    import base.sound_device_manager as devices
    import ui.hardware_management_window as module
    from ui.vkinging_bridge import VkingingBridge
    path=tmp_path/'system.db'; db=DataSave(str(path));db.create_system_tables();db.close()
    repo=HardwareManagementRepository(str(path))
    monkeypatch.setattr(devices,'AUDIO_DEVICE_CONFIG_PATH',str(tmp_path/'selected.json'))
    monkeypatch.setattr(module.HardwareRegistrationDialog,'_collect_devices_by_api',lambda self:{'ASIO':[]})
    warnings=[];monkeypatch.setattr(module.MessageBox,'warning',lambda *a:warnings.append(a))
    service=FakeService();bridge=VkingingBridge(service)
    dialog=module.HardwareRegistrationDialog(repository=repo,vkinging_bridge=bridge)
    assert not dialog.register_btn.isEnabled()
    request=service.requests[-1];service.state='idle'
    service.events=[VkingingEvent(request,'succeeded','idle','releasing',value=DiscoveryResult((device(channels=tuple(range(32))),)),resources_released=True)]
    bridge.poll();dialog.api_combo.setCurrentText('vkinging');dialog.device_combo.setCurrentIndex(1)
    assert dialog.samplerate_spin.minimum()==4000 and dialog.samplerate_spin.maximum()==102400
    dialog.samplerate_spin.setValue(64000);dialog.bit_depth_combo.setCurrentText('64')
    dialog.accept_registration()
    assert dialog.result()==QDialog.Accepted
    assert repo.get_asset('machine-A')['samplerate']==64000
    assert len(repo.list_channels('machine-A'))==32
    window=module.HardwareManagementWindow(repository=repo,vkinging_bridge=bridge)
    window.model.set_checked_hardware_id('machine-A')
    service.state='unavailable';window.delete_selected_asset()
    assert repo.get_asset('machine-A') is not None and warnings
    service.state='idle';monkeypatch.setattr(module.MessageBox,'question',lambda *a:module.MessageBox.Yes)
    window.delete_selected_asset()
    assert repo.get_asset('machine-A') is None
    bridge.close();dialog.close();window.close()

def test_new_subscription_generation_ignores_previous_discovery(qapp):
    from ui.vkinging_bridge import VkingingBridge
    service=FakeService();bridge=VkingingBridge(service);owner=QDialog();seen=[]
    old=bridge.request_discovery(owner,lambda event:pytest.fail('stale callback'))
    service.state='idle'
    current=bridge.request_discovery(owner,seen.append)
    service.events=[VkingingEvent(old,'succeeded','idle','releasing',value=DiscoveryResult((device('obsolete'),)),resources_released=True)]
    bridge.poll()
    assert not seen and not bridge.devices
    service.state='idle'
    service.events=[VkingingEvent(current,'succeeded','idle','releasing',value=DiscoveryResult((device('current'),)),resources_released=True)]
    bridge.poll()
    assert len(seen)==1 and bridge.devices[0]['name']=='current'
    bridge.close();owner.close()


def test_injected_native_logger_records_external_failure(caplog):
    import logging
    from base.vkinging_service import VkingingService
    logger=logging.getLogger('task7-native-failure')
    def unavailable_sdk(): raise OSError('fake SDK unavailable')
    service=VkingingService(logger=logger,sdk_factory=unavailable_sdk)
    try:
        with caplog.at_level(logging.ERROR,logger=logger.name):
            assert service.submit_discovery('logger-discovery')
            with service._condition:
                assert service._condition.wait_for(lambda:service.state=='idle',timeout=2)
        terminal=[e for e in service.poll_events() if e.kind=='failed']
        assert len(terminal)==1 and terminal[0].resources_released
        assert terminal[0].diagnostic_error is None
        assert 'logger-discovery' in caplog.text and 'fake SDK unavailable' in caplog.text
    finally:
        service.close();service._worker.join(timeout=2)
        assert not service._worker.is_alive()

def test_cancelled_vkinging_selection_preserves_saved_output(qapp, main_window_module, monkeypatch):
    from ui.vkinging_bridge import VkingingBridge
    from types import SimpleNamespace
    service=FakeService();bridge=VkingingBridge(service)
    output={'name':'speaker','hostapi_name':'ASIO','index':7}
    pending=device('old')
    manager=SimpleNamespace(get_startup_devices=lambda:dict(mic=pending,speaker=output,mic_channels=[31],device_available=False,validation_pending=True))
    monkeypatch.setattr(main_window_module,'SoundDeviceManager',lambda:manager)
    monkeypatch.setattr(main_window_module.MainWindow,'init_ui',lambda self:None)
    window=main_window_module.MainWindow(vkinging_bridge=bridge)
    window.on_hardware_window_init()
    assert window.speaker==output and window.mic==pending and window.mic_channels==[31]
    assert not window.device_workflow_available
    bridge.close()

@pytest.mark.parametrize('kind,reason', [('cancelled','cancelled'),('failed','read_stall')])
def test_unreleased_terminal_keeps_live_owner_until_release(qapp, kind, reason):
    from ui.vkinging_bridge import VkingingBridge
    service=FakeService();bridge=VkingingBridge(service);owner=QDialog();seen=[]
    request=bridge.request_discovery(owner,seen.append)
    service.state='stopping'
    service.events=[VkingingEvent(request,kind,'stopping','releasing',reason=reason,resources_released=False)]
    bridge.poll()
    assert [event.kind for event in seen]==[kind]
    assert request in bridge._subscriptions
    service.state='idle'
    service.events=[VkingingEvent(request,'released','idle','releasing',resources_released=True)]
    bridge.poll()
    assert [event.kind for event in seen]==[kind,'released']
    assert not bridge._subscriptions
    bridge.close();owner.close()


def test_closed_owner_suppresses_unreleased_terminal_and_late_release(qapp):
    from ui.vkinging_bridge import VkingingBridge
    service=FakeService();bridge=VkingingBridge(service);owner=QDialog();seen=[]
    request=bridge.request_discovery(owner,seen.append)
    owner.reject()
    service.state='stopping'
    service.events=[VkingingEvent(request,'cancelled','stopping','releasing',resources_released=False)]
    bridge.poll()
    service.state='idle'
    service.events=[VkingingEvent(request,'released','idle','releasing',resources_released=True)]
    bridge.poll()
    assert not seen and not bridge._subscriptions
    bridge.close()


def test_new_request_after_unreleased_terminal_suppresses_old_release(qapp):
    from ui.vkinging_bridge import VkingingBridge
    service=FakeService();bridge=VkingingBridge(service);owner=QDialog();old_events=[];new_events=[]
    previous=bridge.request_discovery(owner,old_events.append)
    service.state='stopping'
    service.events=[VkingingEvent(previous,'cancelled','stopping','releasing',resources_released=False)]
    bridge.poll()
    service.state='idle'
    current=bridge.request_discovery(owner,new_events.append)
    assert list(bridge._subscriptions)==[current]
    service.events=[VkingingEvent(previous,'released','idle','releasing',resources_released=True)]
    bridge.poll()
    assert [event.kind for event in old_events]==['cancelled']
    assert not new_events and list(bridge._subscriptions)==[current]
    bridge.close();owner.close()

def test_owner_closing_after_cancel_suppresses_release(qapp):
    from ui.vkinging_bridge import VkingingBridge
    service=FakeService();bridge=VkingingBridge(service);owner=QDialog();seen=[]
    request=bridge.request_discovery(owner,seen.append)
    service.state='stopping'
    service.events=[VkingingEvent(request,'cancelled','stopping','releasing',resources_released=False)]
    bridge.poll();owner.reject()
    assert not bridge._subscriptions
    service.state='idle'
    service.events=[VkingingEvent(request,'released','idle','releasing',resources_released=True)]
    bridge.poll()
    assert [event.kind for event in seen]==['cancelled']
    bridge.close()

def test_missing_registered_route_remains_visible_and_cannot_be_selected(qapp, tmp_path):
    from PyQt5.QtCore import Qt
    from base.db_manager import DataSave
    from base.hardware_management import HardwareManagementRepository
    from ui.hardware_window import HardwareSelectionModel, HardwareSelectionController, HardwareSelectionView, HardwareSelectionState
    from ui.vkinging_bridge import VkingingBridge
    path=tmp_path/'system.db'; db=DataSave(str(path));db.create_system_tables();db.close()
    repo=HardwareManagementRepository(str(path))
    asset=repo.register_asset(device(channels=(0,31)),'vkinging','DAQ',51200)
    before=repo.list_channels('machine-A')
    service=FakeService();bridge=VkingingBridge(service)
    state=HardwareSelectionState(mic_device=asset,mic_channels=[0,31],api_name='vkinging')
    view=HardwareSelectionView();model=HardwareSelectionModel(state,repository=repo)
    controller=HardwareSelectionController(model,view,vkinging_bridge=bridge)
    request=service.requests[-1];service.state='idle'
    service.events=[VkingingEvent(request,'succeeded','idle','releasing',value=DiscoveryResult((device(channels=(0,)),)),resources_released=True)]
    bridge.poll()
    rows=view.mic_channel_table.model()
    assert rows.rowCount()==2
    assert rows.item(0,0).data(Qt.UserRole)==0 and rows.item(0,0).isEnabled()
    missing=rows.item(1,0)
    assert missing.data(Qt.UserRole)==31 and '不可用' in missing.text()
    assert not missing.isEnabled() and not missing.isCheckable()
    assert missing.checkState()==Qt.Unchecked
    missing.setCheckState(Qt.Checked)
    assert view.mic_channel_table.checked_payloads()==[0]
    assert repo.list_channels('machine-A')==before
    bridge.close();view.close()

def test_registration_rate_presets_allow_arbitrary_valid_integer(qapp, monkeypatch):
    from types import SimpleNamespace
    import ui.hardware_management_window as module
    captured=[]
    monkeypatch.setattr(module.HardwareRegistrationDialog,'_collect_devices_by_api',lambda self:{'vkinging':[device()]})
    dialog=module.HardwareRegistrationDialog(repository=SimpleNamespace(register_asset=lambda *args:captured.append(args)))
    dialog.device_combo.setCurrentIndex(1)
    control=dialog.samplerate_spin
    assert [control.itemData(i) for i in range(control.count())]==[44100,48000,51200]
    control.setEditText('64000')
    dialog.accept_registration()
    assert dialog.result()==QDialog.Accepted and captured[0][3]==64000
    dialog.close()


@pytest.mark.parametrize('text',['3999','102401','51200.0','True',''])
def test_registration_rejects_invalid_typed_rate_without_clamping(qapp, monkeypatch, text):
    from types import SimpleNamespace
    import ui.hardware_management_window as module
    captured=[];warnings=[]
    monkeypatch.setattr(module.HardwareRegistrationDialog,'_collect_devices_by_api',lambda self:{'vkinging':[device()]})
    monkeypatch.setattr(module.MessageBox,'warning',lambda *args:warnings.append(args))
    dialog=module.HardwareRegistrationDialog(repository=SimpleNamespace(register_asset=lambda *args:captured.append(args)))
    dialog.device_combo.setCurrentIndex(1)
    dialog.samplerate_spin.setEditText(text)
    dialog.accept_registration()
    assert not captured and warnings and dialog.result()!=QDialog.Accepted
    assert dialog.samplerate_spin.currentText()==text
    dialog.close()


def test_hardware_rate_editor_presets_and_strict_custom_input(qapp, monkeypatch):
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QStandardItemModel, QStandardItem
    from PyQt5.QtWidgets import QWidget
    import ui.hardware_management_window as module
    parent=QWidget();model=QStandardItemModel(1,1);item=QStandardItem()
    item.setData(51200,Qt.EditRole);model.setItem(0,0,item);model.assets=[device()]
    index=model.index(0,0)
    delegate=module.HardwareSampleRateDelegate([(str(rate),rate) for rate in (44100,48000)],parent)
    editor=delegate.createEditor(parent,None,index)
    assert [editor.itemData(i) for i in range(editor.count())]==[44100,48000,51200]
    delegate.setEditorData(editor,index)
    assert editor.value()==51200
    captured=[];warnings=[]
    monkeypatch.setattr(model,'setData',lambda index,value,role:captured.append(value) or True)
    monkeypatch.setattr(module.MessageBox,'warning',lambda *args:warnings.append(args))
    for rate in (4000,64000,102400):
        editor.setEditText(str(rate));delegate.setModelData(editor,model,index)
        assert captured[-1]==rate
    assert captured==[4000,64000,102400]
    captured.clear()
    for invalid in ('3999','102401','51200.0','True',''):
        editor.setEditText(invalid);delegate.setModelData(editor,model,index)
        assert not captured and editor.currentText()==invalid
    assert len(warnings)==5
    parent.close()

@pytest.mark.parametrize('old_output_connected',[True,False])
def test_ordinary_driver_switch_saves_and_applies_only_visible_input(qapp, tmp_path, monkeypatch, old_output_connected):
    import json
    from unit_test.ui.test_hardware_window_device_selection import FakeRepository, _asset, _device
    import base.sound_device_manager as devices
    import ui.hardware_window as module
    output=_asset('out-A','Output A','speaker A',hostapi_name='API-A',outputs=2)
    mic=_asset('mic-B','Input B','mic B',hostapi_name='API-B',inputs=2)
    repository=FakeRepository([output,mic],channels={'mic-B':[dict(hardware_id='mic-B',channel_index=0,channel_label='In1',direction='input')]})
    runtime_mic=_device(7,'mic B',inputs=2)
    runtime_output=_device(9,'speaker A',outputs=2)
    runtime={'API-A':{'input':[],'output':[runtime_output] if old_output_connected else []},
             'API-B':{'input':[runtime_mic],'output':[]}}
    selected_path=tmp_path/'selected-devices.json'
    monkeypatch.setattr(devices,'AUDIO_DEVICE_CONFIG_PATH',str(selected_path))
    monkeypatch.setattr(module.SoundDeviceManager,'refresh_available_device',staticmethod(lambda:None))
    monkeypatch.setattr(module.SoundDeviceManager,'get_device_info',staticmethod(lambda:runtime))
    input_calls=[];duplex_calls=[];warnings=[]
    monkeypatch.setattr(module.SoundDeviceManager,'change_default_input_device',staticmethod(lambda index:input_calls.append(index)))
    monkeypatch.setattr(module.SoundDeviceManager,'change_default_device',staticmethod(lambda *indices:duplex_calls.append(indices)))
    monkeypatch.setattr(module.MessageBox,'warning',lambda *args:warnings.append(args))
    state=module.HardwareSelectionState(speaker_device=output,api_name='API-A')
    model=module.HardwareSelectionModel(state,repository=repository);view=module.HardwareSelectionView()
    controller=module.HardwareSelectionController(model,view)
    assert view.speaker_device_table.checked_payload()['hardware_id']=='out-A'
    view.driver_combo.setCurrentText('API-B')
    assert view.speaker_device_table.model().rowCount()==0
    assert view.speaker_device_table.checked_payload() is None
    assert model.state.speaker_device is None
    view.mic_device_table.set_checked_by_predicate(lambda payload:payload['hardware_id']=='mic-B')
    controller._restore_channels(view.mic_channel_table,[0])
    controller._on_ok_clicked()
    assert not warnings and view.result()==QDialog.Accepted
    saved=json.loads(selected_path.read_text(encoding='utf-8'))
    assert saved['speaker'] is None and saved['mic']['hardware_id']=='mic-B'
    assert saved['mic_channels']==[0]
    assert input_calls==[7] and duplex_calls==[]
    assert model.state.speaker_device is None
    selection=controller.get_selected_devices(return_status=True)
    assert selection.accepted
    assert selection.devices==(None,model.state.mic_device,[0])
    view.close()

@pytest.mark.parametrize('entrypoint',['on_hardware_window_init','_open_hardware_selection_for_recovery'])
def test_cancelled_real_controller_preserves_pending_availability_and_diagnostics(qapp, main_window_module, monkeypatch, entrypoint):
    from types import SimpleNamespace
    from unit_test.ui.test_main_window_startup_device_recovery import _window
    module=hardware_window_module
    pending=device('pending');output={'name':'speaker','hostapi_name':'ASIO','index':7}
    repository=SimpleNamespace(
        tables_exist=lambda:True,
        list_assets_for_selection=lambda:{'vkinging':{'input':[pending],'output':[]}},
        list_channels=lambda *a,**k:[dict(channel_index=31,channel_label='In32')])
    state=module.HardwareSelectionState(speaker_device=output,mic_device=pending,mic_channels=[31],api_name='vkinging')
    view=module.HardwareSelectionView();model=module.HardwareSelectionModel(state,repository=repository)
    controller=module.HardwareSelectionController(model,view);view.reject()
    legacy=controller.get_selected_devices()
    assert legacy==(output,pending,[31])
    def open_selection(**kwargs):
        return controller.get_selected_devices(return_status=kwargs.get('return_status',False)) if kwargs.get('return_status') else legacy
    monkeypatch.setattr(main_window_module,'open_hardware_selection_window',open_selection)
    window=_window(main_window_module)
    window.mic=pending;window.speaker=output;window.mic_channels=[31]
    window.device_workflow_available=False
    window.startup_device_error_reason='vkinging 尚未验证'
    window.startup_device_notice_message='请等待原生发现'
    window.startup_recovery_action='retry_vkinging'
    window._vkinging_validation_pending=True
    getattr(window,entrypoint)()
    assert not window.device_workflow_available
    assert window.mic==pending and window.speaker==output and window.mic_channels==[31]
    assert window.startup_device_error_reason=='vkinging 尚未验证'
    assert window.startup_device_notice_message=='请等待原生发现'
    assert window.startup_recovery_action=='retry_vkinging'
    assert window._vkinging_validation_pending
    assert window.sequence_window.available_calls==[]
    view.close()


def test_real_controller_rejected_status_keeps_legacy_tuple(qapp):
    from types import SimpleNamespace
    module=hardware_window_module
    pending=device();output={'name':'speaker','hostapi_name':'ASIO'}
    repository=SimpleNamespace(tables_exist=lambda:True,
        list_assets_for_selection=lambda:{'vkinging':{'input':[pending],'output':[]}},
        list_channels=lambda *a,**k:[dict(channel_index=31,channel_label='In32')])
    state=module.HardwareSelectionState(speaker_device=output,mic_device=pending,mic_channels=[31],api_name='vkinging')
    view=module.HardwareSelectionView();model=module.HardwareSelectionModel(state,repository=repository)
    controller=module.HardwareSelectionController(model,view);view.reject()
    assert controller.get_selected_devices()==(output,pending,[31])
    result=controller.get_selected_devices(return_status=True)
    assert not result.accepted and result.devices==(output,pending,[31])
    view.close()

@pytest.mark.parametrize('available',[False,True])
@pytest.mark.parametrize('target',['mic','speaker'])
@pytest.mark.parametrize('field,value',[('display_name','Edited asset'),('samplerate',64000)])
def test_metadata_edits_preserve_availability_and_propagated_diagnostics(qapp, main_window_module, monkeypatch, available, target, field, value):
    from unit_test.ui.test_main_window_startup_device_recovery import _window
    from ui.vkinging_bridge import VkingingBridge
    window=_window(main_window_module)
    service=FakeService();bridge=VkingingBridge(service)
    bridge.discovery_complete=True
    assert bridge.state=='idle' and bridge.devices==()
    window.vkinging_bridge=bridge
    window.mic=device('disconnected retained input')
    window.speaker=dict(hardware_id='out',name='Speaker',hostapi_name='ASIO',samplerate=48000,bit_depth=32)
    window.mic_channels=[31]
    window.device_workflow_available=available
    window.startup_device_error_reason='原生输入仍不可用' if not available else None
    window.startup_device_notice_message='请连接设备后重新发现' if not available else None
    window.startup_recovery_action='retry_vkinging' if not available else None
    window.startup_can_retry_saved_devices=True
    window._vkinging_validation_pending=False
    diagnostics=(window.startup_device_error_reason,window.startup_device_notice_message,window.startup_recovery_action)
    message='发现完成但设备已断开，请重新连接 (machine-A)' if not available else ''
    window.sequence_window.audio_devices_unavailable_message=message
    window.sequence_window.audio_devices_available=available
    def publish(flag, text=''):
        window.sequence_window.available_calls.append((flag,text))
        window.sequence_window.audio_devices_available=flag
        window.sequence_window.audio_devices_unavailable_message=text
    window.sequence_window.set_audio_devices_available=publish
    before_mic=dict(window.mic);before_output=dict(window.speaker)
    selected=window.mic if target=='mic' else window.speaker
    if target=='speaker' and field=='samplerate':
        value=44100  # Ordinary outputs retain their existing supported-rate policy.
    saved=[]
    monkeypatch.setattr(main_window_module.SoundDeviceManager,'save_selected_devices',
        staticmethod(lambda mic,speaker,channels:saved.append((dict(mic),dict(speaker),list(channels)))))
    window.on_registered_audio_hardware_updated(selected['hardware_id'],dict(hardware_id=selected['hardware_id'],**{field:value}))
    assert window.device_workflow_available is available
    assert window.sequence_window.audio_devices_available is available
    assert window.sequence_window.audio_devices_unavailable_message==message
    assert window.sequence_window.available_calls[-1]==(available,message)
    assert (window.startup_device_error_reason,window.startup_device_notice_message,window.startup_recovery_action)==diagnostics
    assert window.startup_can_retry_saved_devices and not window._vkinging_validation_pending
    expected_mic=dict(before_mic,**({field:value} if target=='mic' else {}))
    expected_output=dict(before_output,**({field:value} if target=='speaker' else {}))
    assert window.mic==expected_mic and window.speaker==expected_output
    assert window.mic_channels==[31]
    assert saved==[(expected_mic,expected_output,[31])]
    bridge.close()
