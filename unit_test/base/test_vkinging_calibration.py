import numpy as np
import pytest
from base.vkinging_calibration import calculate_calibration_factor
from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis

@pytest.mark.parametrize("target", [94,114])
def test_preserves_existing_algorithm_and_precision(target):
    rate=4000
    raw=.03+ .25*np.sin(2*np.pi*997*np.arange(rate*10)/rate)
    original=raw.copy()
    smooth=AudioThdFrequencyResponseAnalysis().spl_calculation(raw,method="rms",window_size=1201)
    center=len(smooth)//2; step=len(raw)//3
    expected=10**((target-np.mean(smooth[center-step:center+step]))/20)
    assert calculate_calibration_factor(raw,rate,target)==pytest.approx(expected,rel=1e-14)
    np.testing.assert_array_equal(raw,original)

@pytest.mark.parametrize("kind", ["zero","dc","nan","short","stereo"])
def test_rejects_invalid_capture(kind):
    raw=np.sin(np.arange(40000))
    if kind=="zero": raw[:]=0
    if kind=="dc": raw[:]=2
    if kind=="nan": raw[100]=np.nan
    if kind=="short": raw=raw[:-1]
    if kind=="stereo": raw=np.column_stack([raw,raw])
    with pytest.raises(ValueError): calculate_calibration_factor(raw,4000,94)


def test_production_routing_runs_calculation_after_sdk_close_on_same_owner(tmp_path,monkeypatch):
    import threading
    from base.vkinging_input import VkingingCaptureRequest
    from base.vkinging_service import VkingingService
    from unit_test.base.vkinging_fakes import FakeSDK
    import base.vkinging_calibration as module
    raw=.2*np.sin(np.arange(40000))
    reads=[(raw[i:i+400].tolist(),400) for i in range(0,len(raw),400)]
    sdk=FakeSDK(devices=(("addr","dev"),),attributes={("dev","MachineId"):"id",("dev","Model"):""},channels={"dev":("dev/AIN32",)},reads=reads)
    sdk.verify_actual_sample_rate=lambda *a:None
    req=VkingingCaptureRequest("cal","calibration",dict(hostapi_name="vkinging",hardware_id="id",name="dev",physical_channels=(31,),samplerate=4000,bit_depth=64),(31,),4000,64,2,40000,output_path=str(tmp_path/"cal.wav"),calibration_target_spl=114)
    called=threading.Event(); seen=[]; original=module.calculate_calibration_factor
    def calculate(*args):
        seen.append((sdk.closed,threading.get_ident()))
        called.set()
        return original(*args)
    monkeypatch.setattr(module,"calculate_calibration_factor",calculate)
    service=VkingingService(sdk_factory=lambda:sdk)
    try:
        assert service.start_capture(req)
        assert called.wait(3)
        with service._condition:
            assert service._condition.wait_for(lambda: service.state == "idle", timeout=3)
        events=service.poll_events()
        success=next(e for e in events if e.kind=="succeeded")
        assert success.resources_released
        assert success.value["factor"]==pytest.approx(original(raw,4000,114))
    finally:
        service.close(); service._worker.join(3)
    assert seen==[(True,service._worker.ident)]
    assert not (tmp_path/"cal.wav").exists()


def test_cancel_immediately_after_file_finalization_proves_release(monkeypatch):
    from types import SimpleNamespace
    from base.vkinging_service import CaptureResult
    import base.vkinging_calibration as module
    class Control:
        cancelled=False
        released=False
        def confirm_released(self): self.released=True
    control=Control()
    def finalize(control):
        control.cancelled=True
        return CaptureResult({"path":"unused"},resources_released=True)
    monkeypatch.setattr(module,"capture_vkinging",lambda *a,**kw:CaptureResult(resources_released=True,finalize=finalize))
    result=module.capture_calibration(None,SimpleNamespace(calibration_target_spl=94),control)
    with pytest.raises(RuntimeError,match="cancelled"): result.finalize(control)
    assert control.released


@pytest.mark.parametrize("target",[True,94.0,93,115,"94"])
def test_request_rejects_invalid_frozen_target(target):
    from base.vkinging_input import VkingingCaptureRequest
    with pytest.raises(ValueError,match="calibration_target_spl"):
        VkingingCaptureRequest("cal","calibration",dict(hostapi_name="vkinging",hardware_id="id",name="dev",physical_channels=(31,),samplerate=4000,bit_depth=64),(31,),4000,64,2,40000,calibration_target_spl=target)


@pytest.mark.parametrize("read_fails,close_kind", [(True,"normal"),(False,"normal"),(True,"broken_text"),(True,None)])
def test_calibration_reader_preserves_primary_and_close_diagnostics(monkeypatch,read_fails,close_kind):
    from types import SimpleNamespace
    import soundfile as sf
    import base.vkinging_calibration as module
    from base.vkinging_service import CaptureResult
    primary=ValueError("primary read failed")
    class BrokenText(OSError):
        def __str__(self): raise RuntimeError("format failed")
    secondary=BrokenText() if close_kind=="broken_text" else OSError("secondary close failed")
    class Reader:
        samplerate=4000
        channels=1
        def __len__(self): return 40000
        def blocks(self,**kw):
            if read_fails: raise primary
            yield np.sin(np.arange(40000))
        def close(self):
            if close_kind is not None: raise secondary
    class Control:
        cancelled=False
        released=False
        cleanup=[]
        def confirm_released(self): self.released=True
        def report_cleanup_failure(self,reason): self.cleanup.append(reason)
    control=Control(); control.cleanup=[]
    descriptor={"path":"owned-calibration.partial.wav","owned_paths":["owned-calibration.partial.wav"]}
    monkeypatch.setattr(module,"capture_vkinging",lambda *a,**kw:CaptureResult(resources_released=True,finalize=lambda c:CaptureResult(descriptor,resources_released=True)))
    monkeypatch.setattr(sf,"SoundFile",lambda *a,**kw:Reader())
    req=SimpleNamespace(calibration_target_spl=94,sample_rate=4000,target_samples=40000)
    result=module.capture_calibration(None,req,control)
    with pytest.raises(Exception) as caught: result.finalize(control)
    assert caught.value is (primary if read_fails else secondary)
    if close_kind is not None:
        assert not control.released
        text=" ".join(control.cleanup)
        assert "owned-calibration.partial.wav" in text
        assert ("exception text unavailable" if close_kind=="broken_text" else "secondary close failed") in text
        if read_fails: assert control.cleanup[0] in caught.value.__notes__
    else:
        assert control.released
        assert not control.cleanup


@pytest.mark.parametrize("unknown",[False,True])
def test_constructor_failure_release_proof(tmp_path,monkeypatch,unknown):
    from types import SimpleNamespace
    import soundfile as sf
    import base.vkinging_calibration as module
    from base.vkinging_service import CaptureResult
    class Control:
        cancelled=False
        released=False
        def __init__(self): self.cleanup=[]
        def confirm_released(self): self.released=True
        def report_cleanup_failure(self,reason): self.cleanup.append(reason)
    control=Control()
    descriptor={"path":str(tmp_path/"missing.wav"),"owned_paths":[str(tmp_path/"missing.wav")]}
    monkeypatch.setattr(module,"capture_vkinging",lambda *a,**kw:CaptureResult(resources_released=True,finalize=lambda c:CaptureResult(descriptor,resources_released=True)))
    if unknown:
        def fail(*a,**kw): raise RuntimeError("unexpected open failure")
        monkeypatch.setattr(sf,"SoundFile",fail)
    result=module.capture_calibration(None,SimpleNamespace(calibration_target_spl=94),control)
    with pytest.raises(RuntimeError if unknown else sf.LibsndfileError): result.finalize(control)
    assert control.released is not unknown
    assert bool(control.cleanup)==unknown
    if unknown:
        assert "unexpected open failure" in control.cleanup[0]
        assert descriptor["path"] in control.cleanup[0]
