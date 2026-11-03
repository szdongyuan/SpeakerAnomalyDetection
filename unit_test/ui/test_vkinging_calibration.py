import os
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
from types import SimpleNamespace
import pytest
from PyQt5.QtWidgets import QApplication
import ui.calibration_window as module
from consts.calibration_consts import INPUT_CALIBRATION_MODE_MANUAL

@pytest.fixture(scope="module")
def qapp(): return QApplication.instance() or QApplication([])

class Bridge:
    busy=False
    def start_capture(self,owner,request,callback):
        self.request=request; self.callback=callback; return True
    def cancel(self,rid): self.cancelled=rid
    def unsubscribe(self,owner): self.unsubscribed=True

@pytest.fixture
def wnd(qapp,monkeypatch):
    monkeypatch.setattr(module,"load_mic_channel_v2pa_factors",lambda **kw:{})
    monkeypatch.setattr(module.MessageBox,"warning",lambda *a:None)
    bridge=Bridge()
    device=dict(hostapi_name="vkinging",hardware_id="A",name="dev",physical_channels=(15,31),samplerate=4000,bit_depth=64)
    window=module.InputCalibration(vkinging_bridge=bridge,selected_mic=device,mic_channels=[15,31],acq_config={"mode":"RECORD_ONLY","detail":{"vkinging_range_index":2}})
    window.show(); yield window,bridge
    window.close()

def test_request_and_cancel_late_result(wnd,monkeypatch):
    window,bridge=wnd; saves=[]
    monkeypatch.setattr(module,"save_mic_channel_v2pa_factor",lambda *a,**kw:saves.append((a,kw)))
    assert window.clicked_calibration()
    req=bridge.request
    assert (req.channels,req.target_samples,req.discard_initial_samples,req.range_index,req.calibration_target_spl)==((15,),40000,0,2,94)
    window.stop_active_streaming_capture()
    bridge.callback(SimpleNamespace(request_id=req.request_id,kind="succeeded",resources_released=True,value={"factor":2.123456789,"standard_spl":94},reason=None))
    assert saves==[]
    assert bridge.cancelled==req.request_id

def test_success_full_precision_conditions_and_next_channel(wnd,monkeypatch):
    window,bridge=wnd; saves=[]
    monkeypatch.setattr(module,"save_mic_channel_v2pa_factor",lambda *a,**kw:saves.append((a,kw)))
    assert window.clicked_calibration()
    bridge.callback(SimpleNamespace(request_id=bridge.request.request_id,kind="succeeded",resources_released=True,value={"factor":2.123456789,"standard_spl":94},reason=None))
    assert saves[0][0]==(15,2.123456789)
    assert saves[0][1]["input_conditions"]==dict(input_mode="IEPE",unit="V",range_min=-2.5,range_max=2.5)
    assert window.current_channel==31

def test_manual_precision_and_failure_restores(wnd,monkeypatch):
    window,bridge=wnd
    window.calibration_mode=INPUT_CALIBRATION_MODE_MANUAL
    window._refresh_current_channel_display()
    window.v2pa_factor_lineedit.setText("0.123456789012345")
    window._manual_factor_dirty=True
    saves=[]
    monkeypatch.setattr(module,"save_manual_mic_channel_v2pa_factor",lambda *a,**kw:saves.append((a,kw)))
    window._manual_factor_editing_finished()
    assert saves[0][0][1]==.123456789012345
    assert window.v2pa_factor_lineedit.text()=="0.123456789012345"
    previous=window.v2pa_factor_lineedit.text()
    def fail(*a,**kw): raise ValueError("DB failure")
    monkeypatch.setattr(module,"save_manual_mic_channel_v2pa_factor",fail)
    window.v2pa_factor_lineedit.setText("4.5"); window._manual_factor_dirty=True
    window._manual_factor_editing_finished()
    assert window.v2pa_factor_lineedit.text()==previous


@pytest.mark.parametrize("kind",["busy","reset_failure","success_no_release","hide","save_failure"])
def test_mutation_guards_and_reset_scope(wnd,monkeypatch,kind):
    window,bridge=wnd
    window.session_channel_factors[15]=3.123456789
    window._refresh_current_channel_display()
    calls=[]
    def save(*a,**kw):
        calls.append((a,kw))
        if kind in ("reset_failure","save_failure"): raise ValueError("database failure")
    monkeypatch.setattr(module,"save_mic_channel_v2pa_factor",save)
    monkeypatch.setattr(module,"clear_mic_channel_v2pa_factors",save)
    if kind=="busy":
        bridge.busy=True
        window.reset_btn_clicked()
        assert not window.clicked_calibration()
        assert calls==[]
    elif kind=="reset_failure":
        before=window.v2pa_factor_lineedit.text()
        window.reset_btn_clicked()
        assert calls==[((),dict(hardware_id="A",channel_indices=[15,31]))]
        assert window.v2pa_factor_lineedit.text()==before
    else:
        assert window.clicked_calibration()
        rid=bridge.request.request_id
        if kind=="hide": window.hide()
        bridge.callback(SimpleNamespace(request_id=rid,kind="succeeded",resources_released=kind!="success_no_release",value={"factor":8.5},reason=None))
        assert window.session_channel_factors[15]==3.123456789
        assert len(calls)==(1 if kind=="save_failure" else 0)


def test_snapshot_range_and_device_are_detached(wnd):
    window,bridge=wnd
    assert window.clicked_calibration()
    window._injected_mic["samplerate"]=8000
    window._acq_config["detail"]["vkinging_range_index"]=0
    assert bridge.request.sample_rate==4000
    assert bridge.request.range_index==2
    assert window._vkinging_conditions["range_max"]==2.5


def test_mainwindow_injects_committed_queue_and_same_bridge():
    import ast
    import copy
    from pathlib import Path
    source=ast.parse(Path("main_window.py").read_text(encoding="utf-8"))
    cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=="MainWindow")
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="on_calibration_window_init")
    created=[]
    class Dialog:
        output_calibration_changed=False
        input_calibration_flag=False
        def __init__(self,**kw): created.append(kw)
        def exec(self): pass
    ns=dict(is_vkinging_device=module.is_vkinging_device,CalibrationWindow=Dialog,MessageBox=module.MessageBox)
    exec(compile(ast.Module(body=[method],type_ignores=[]),"main_window.py","exec"),ns)
    acq=dict(mode="RECORD_ONLY",detail={"vkinging_range_index":3})
    bridge=Bridge()
    owner=SimpleNamespace(mic=dict(hostapi_name="vkinging"),speaker=None,mic_channels=[31],vkinging_bridge=bridge,sequence_window=SimpleNamespace(sequence_config=[{"seq1":{"acq":acq}}]))
    ns["on_calibration_window_init"](owner)
    assert created[0]["vkinging_bridge"] is bridge
    assert created[0]["acq_config"]==acq
    assert created[0]["acq_config"] is not acq



def test_real_database_saves_distinct_channel_conditions_and_reset(wnd,tmp_path,monkeypatch):
    import json
    from base.db_manager import DataSave
    from base.hardware_management import HardwareManagementRepository
    from base import soundcard_calibration_manager as manager
    from consts import error_code
    path=str(tmp_path/"hardware.db")
    database=DataSave(path)
    try:
        assert database.create_system_tables()[0]==error_code.OK
    finally:
        database.close()
    repo=HardwareManagementRepository(path)
    window,bridge=wnd
    repo.register_asset(window.selected_input_device,"vkinging","calibration asset",4000)
    monkeypatch.setattr(module,"load_mic_channel_v2pa_factors",lambda **kw:manager.load_mic_channel_v2pa_factors(db_path=path,**kw))
    monkeypatch.setattr(module,"save_manual_mic_channel_v2pa_factor",lambda *a,**kw:manager.save_manual_mic_channel_v2pa_factor(*a,db_path=path,**kw))
    monkeypatch.setattr(module,"save_mic_channel_v2pa_factor",lambda *a,**kw:manager.save_mic_channel_v2pa_factor(*a,db_path=path,**kw))
    monkeypatch.setattr(module,"clear_mic_channel_v2pa_factors",lambda **kw:manager.clear_mic_channel_v2pa_factors(db_path=path,**kw))
    assert window.clicked_calibration()
    bridge.callback(SimpleNamespace(request_id=bridge.request.request_id,kind="succeeded",resources_released=True,value={"factor":1.23456789012345},reason=None))
    assert window.current_channel==31
    window._acq_config["detail"]["vkinging_range_index"]=0
    window.calibration_mode=INPUT_CALIBRATION_MODE_MANUAL
    window._refresh_current_channel_display()
    window.v2pa_factor_lineedit.setText("0.123456789012345")
    window._manual_factor_dirty=True
    window._manual_factor_editing_finished()
    assert window._persist_complete_selected_channel_factors_if_ready()
    rows={r["channel_index"]:r for r in repo.list_channels("A")}
    assert rows[15]["standard_spl"]==94
    assert rows[15]["factor_value"]==1.23456789012345
    assert rows[31]["standard_spl"] is None
    assert rows[31]["factor_value"]==.123456789012345
    assert json.loads(rows[15]["coefficients_json"])["range_max"]==2.5
    assert json.loads(rows[31]["coefficients_json"])["range_max"]==10
    window.selected_input_channels=[31]
    window.reset_btn_clicked()
    rows={r["channel_index"]:r for r in repo.list_channels("A")}
    assert rows[31]["factor_value"] is None
    assert rows[15]["factor_value"]==1.23456789012345



def test_start_resolves_defaults_once_for_request_and_conditions(wnd,monkeypatch):
    window,bridge=wnd
    window._acq_config=None
    calls=[]
    def defaults():
        calls.append(True)
        return {"RECORD_ONLY":{"vkinging_range_index":2 if len(calls)==1 else 0}}
    monkeypatch.setattr(module,"load_acquisition_defaults",defaults)
    assert window._start_vkinging_calibration()
    assert bridge.request.range_index==2
    assert window._vkinging_conditions["range_max"]==2.5


@pytest.mark.parametrize("kind,reason,cleanup",[("failed","read_stall",None),("failed","read failed","clear_task failed"),("cancelled","cancelled",None)])
def test_terminal_failure_is_reported_before_release_and_never_saves(wnd,monkeypatch,kind,reason,cleanup):
    from base.vkinging_service import VkingingEvent
    window,bridge=wnd
    warnings=[]; saves=[]
    monkeypatch.setattr(module.MessageBox,"warning",lambda *a:warnings.append(a[-1]))
    monkeypatch.setattr(module,"save_mic_channel_v2pa_factor",lambda *a,**kw:saves.append((a,kw)))
    assert window.clicked_calibration()
    rid=bridge.request.request_id
    bridge.busy=True
    terminal=VkingingEvent(rid,kind,"unavailable","capturing",reason=reason,cleanup_error=cleanup,resources_released=False)
    bridge.callback(terminal)
    assert len(warnings)==1
    assert reason in warnings[0]
    assert "重启" in warnings[0]
    assert "等待释放" in window.channel_status_label.text()
    assert "录制中" not in window.channel_status_label.text()
    assert window._vkinging_request.request_id==rid
    assert not window.channel_combo_box.isEnabled()
    assert not window._vkinging_available()
    # A duplicate terminal and a stale success must never publish or reprompt.
    bridge.callback(terminal)
    bridge.callback(VkingingEvent(rid,"succeeded","unavailable","releasing",value={"factor":2.5},resources_released=True))
    assert saves==[]
    assert len(warnings)==1
    if cleanup is None:
        bridge.busy=False
        bridge.callback(VkingingEvent(rid,"released","idle","releasing",resources_released=True))
        assert window._vkinging_request is None
        assert reason in window.channel_status_label.toolTip()
        assert "失败" in window.channel_status_label.text() or "取消" in window.channel_status_label.text()
        assert len(warnings)==1
    else:
        # No acknowledgement exists for permanent cleanup failure.
        assert window._vkinging_request.request_id==rid
        assert cleanup in warnings[0]


@pytest.mark.parametrize("operation",["manual","reset"])
@pytest.mark.parametrize("db_fails",[False,True])
def test_successful_operation_clears_old_failure_but_db_failure_preserves_it(wnd,monkeypatch,operation,db_fails):
    from base.vkinging_service import VkingingEvent
    window,bridge=wnd
    assert window.clicked_calibration()
    rid=bridge.request.request_id
    bridge.callback(VkingingEvent(rid,"failed","unavailable","capturing",reason="read_stall",resources_released=False))
    bridge.callback(VkingingEvent(rid,"released","idle","releasing",resources_released=True))
    diagnosis=window.channel_status_label.toolTip()
    def save(*a,**kw):
        if db_fails: raise ValueError("database failure")
    if operation=="manual":
        monkeypatch.setattr(module,"save_manual_mic_channel_v2pa_factor",save)
        window.calibration_mode=INPUT_CALIBRATION_MODE_MANUAL
        window._refresh_current_channel_display()
        window.v2pa_factor_lineedit.setText("2.5")
        window._manual_factor_dirty=True
        window._manual_factor_editing_finished()
    else:
        monkeypatch.setattr(module,"clear_mic_channel_v2pa_factors",save)
        window.reset_btn_clicked()
    if db_fails:
        assert window._vkinging_terminal is not None
        assert window.channel_status_label.toolTip()==diagnosis
        assert "失败" in window.channel_status_label.text()
    else:
        assert window._vkinging_terminal is None
        assert window.channel_status_label.toolTip()==""
        assert "失败" not in window.channel_status_label.text()
        assert "已校准" in window.channel_status_label.text() if operation=="manual" else "未校准" in window.channel_status_label.text()


def test_reset_of_other_channel_preserves_failure_context(wnd,monkeypatch):
    from base.vkinging_service import VkingingEvent
    window,bridge=wnd
    assert window.clicked_calibration()
    rid=bridge.request.request_id
    bridge.callback(VkingingEvent(rid,"cancelled","unavailable","releasing",reason="cancelled",resources_released=False))
    bridge.callback(VkingingEvent(rid,"released","idle","releasing",resources_released=True))
    diagnosis=window._vkinging_terminal
    window.selected_input_channels=[31]
    monkeypatch.setattr(module,"clear_mic_channel_v2pa_factors",lambda **kw:None)
    window.reset_btn_clicked()
    assert window._vkinging_terminal is diagnosis
    assert window.current_channel==31
    assert window.channel_status_label.toolTip()==""
    window.current_channel=15
    window._refresh_current_channel_display()
    assert window.channel_status_label.toolTip()==diagnosis["message"]


@pytest.mark.parametrize("outcome",["success","db_failure","no_release","stale","cancelled","failed","hidden"])
def test_internal_calibration_artifacts_removed_only_after_valid_db_commit(wnd,tmp_path,monkeypatch,outcome):
    from base.vkinging_service import VkingingEvent
    window,bridge=wnd
    owned=[tmp_path/"raw.partial.wav",tmp_path/"assembled.partial.wav"]
    unrelated=tmp_path/"unrelated.wav"
    canonical=tmp_path/"canonical.wav"
    for path in [*owned,unrelated,canonical]: path.write_bytes(b"voltage data")
    committed=[]
    def save(*a,**kw):
        assert all(path.exists() for path in owned)
        if outcome=="db_failure": raise ValueError("DB failure")
        committed.append(a)
    monkeypatch.setattr(module,"save_mic_channel_v2pa_factor",save)
    assert window.clicked_calibration()
    rid=bridge.request.request_id
    if outcome=="hidden": window.hide()
    event=VkingingEvent("old" if outcome=="stale" else rid,
        outcome if outcome in ("cancelled","failed") else "succeeded","idle","finalizing",
        value={"factor":2.5,"path":str(canonical),"owned_paths":[str(p) for p in owned]},
        resources_released=outcome!="no_release")
    bridge.callback(event)
    assert all(path.exists()==(outcome!="success") for path in owned)
    assert unrelated.exists() and canonical.exists()
    assert bool(committed)==(outcome=="success")


def test_cleanup_failure_keeps_coefficient_and_attempts_remaining_owned_files(wnd,tmp_path,monkeypatch):
    import os
    from base.vkinging_service import VkingingEvent
    window,bridge=wnd
    blocked=tmp_path/"blocked.partial.wav"
    removable=tmp_path/"removable.partial.wav"
    unrelated=tmp_path/"unrelated.wav"
    for path in (blocked,removable,unrelated): path.write_bytes(b"voltage data")
    committed=[]; attempts=[]; warnings=[]
    monkeypatch.setattr(module,"save_mic_channel_v2pa_factor",lambda *a,**kw:committed.append(a))
    monkeypatch.setattr(window.default_logger,"warning",lambda *a:warnings.append(a))
    def unlink(path):
        assert committed
        attempts.append(path)
        if path==str(blocked): raise PermissionError(13,"file busy " + "x"*3000,path)
        os.unlink(path)
    monkeypatch.setattr(module,"unlink",unlink,raising=False)
    assert window.clicked_calibration()
    bridge.callback(VkingingEvent(bridge.request.request_id,"succeeded","idle","finalizing",
        value={"factor":2.5,"owned_paths":[str(blocked),str(removable)]},resources_released=True))
    assert committed==[(15,2.5)]
    assert window.session_channel_factors[15]==2.5
    assert 15 in window.calibrated_channels
    assert attempts==[str(blocked),str(removable)]
    assert blocked.exists() and not removable.exists() and unrelated.exists()
    assert len(warnings)==1
    assert all(len(argument)<=512 for argument in warnings[0][1:])



def test_cleanup_logging_io_failure_keeps_committed_callback_and_remaining_cleanup(wnd,tmp_path,monkeypatch):
    import os
    from base.vkinging_service import VkingingEvent
    window,bridge=wnd
    blocked=tmp_path/"blocked.partial.wav"
    removable=tmp_path/"removable.partial.wav"
    for path in (blocked,removable): path.write_bytes(b"voltage data")
    committed=[]; attempts=[]; logging_attempts=[]
    monkeypatch.setattr(module,"save_mic_channel_v2pa_factor",lambda *a,**kw:committed.append(a))
    def unlink(path):
        assert committed
        attempts.append(path)
        if path==str(blocked): raise OSError("file busy " + "x"*3000)
        os.unlink(path)
    def warning(*a):
        logging_attempts.append(a)
        raise OSError("log storage unavailable " + "y"*3000)
    monkeypatch.setattr(module,"unlink",unlink)
    monkeypatch.setattr(window.default_logger,"warning",warning)
    assert window.clicked_calibration()
    bridge.callback(VkingingEvent(bridge.request.request_id,"succeeded","idle","finalizing",
        value={"factor":2.5,"owned_paths":[str(blocked),str(removable)]},resources_released=True))
    assert committed==[(15,2.5)]
    assert window.session_channel_factors[15]==2.5
    assert attempts==[str(blocked),str(removable)]
    assert blocked.exists() and not removable.exists()
    assert len(logging_attempts)==1
    assert window._vkinging_request is None and window.active_capture_channel is None
    assert window.channel_combo_box.isEnabled()
    assert window.current_channel==31
    diagnostics=window._vkinging_cleanup_diagnostics
    assert len(diagnostics)<=4
    assert all(len(message)<=1024 for message in diagnostics)
    assert "file busy" in " ".join(diagnostics)
    assert "log storage unavailable" in " ".join(diagnostics)
