import pytest

from base.output_calibration import (
    OutputCalibrationCache,
    OutputCalibrationPreparation,
    resolve_output_amplitude,
)
from base.soundcard_calibration_manager import SoundcardCalibrationManager
from unit_test.base.test_hardware_management import runtime_device
from unit_test.ui.test_speaker_calibration_db_consumers import (
    _registered_speaker_db,
    _select_speaker,
)


@pytest.mark.parametrize("voltage_type", ["RMS", "Peak"])
def test_resolves_latest_calibration_for_explicit_speaker(tmp_path, monkeypatch, voltage_type):
    repo, selected = _registered_speaker_db(tmp_path, monkeypatch)
    global_speaker = repo.register_asset(
        runtime_device(name="Global Speaker", inputs=0, outputs=1, index=8),
        "Windows WASAPI", "Global Speaker", 48000,
    )
    repo.update_output_amplitude_calibration(global_speaker["hardware_id"], [0.9, 0.0], max_voltage=2.0)
    _select_speaker(monkeypatch, global_speaker["hardware_id"])
    repo.update_output_amplitude_calibration(selected["hardware_id"], [0.5, 0.0], max_voltage=2.0)
    repo.update_output_amplitude_calibration(selected["hardware_id"], [0.25, 0.0], max_voltage=2.0)

    result = resolve_output_amplitude(selected, 1.0, voltage_type)

    assert result == OutputCalibrationPreparation(amplitude=0.25, max_voltage=2.0, cacheable=True)


def test_equal_to_max_succeeds_with_one_calibration_read(tmp_path, monkeypatch):
    repo, speaker = _registered_speaker_db(tmp_path, monkeypatch)
    repo.update_output_amplitude_calibration(speaker["hardware_id"], [0.25, 0.0], max_voltage=2.0)
    reads = []
    original = SoundcardCalibrationManager.load_output_amplitude_calibration

    def read(manager):
        reads.append(manager.speaker_hardware_id)
        return original(manager)

    monkeypatch.setattr(SoundcardCalibrationManager, "load_output_amplitude_calibration", read)

    result = resolve_output_amplitude(speaker, 2.0, "RMS")

    assert result == OutputCalibrationPreparation(amplitude=0.5, max_voltage=2.0, cacheable=True)
    assert reads == [speaker["hardware_id"]]


def test_above_max_rejects_before_prediction(tmp_path, monkeypatch):
    repo, speaker = _registered_speaker_db(tmp_path, monkeypatch)
    repo.update_output_amplitude_calibration(speaker["hardware_id"], [0.25, 0.0], max_voltage=2.0)

    def unexpected_prediction(*args):
        pytest.fail("Out-of-range voltage must not be extrapolated")

    monkeypatch.setattr(SoundcardCalibrationManager, "predict_amplitude", unexpected_prediction)

    result = resolve_output_amplitude(speaker, 3.0, "RMS")

    assert result.amplitude is None
    assert result.max_voltage == 2.0
    assert "3 V" in result.error
    assert "2 V" in result.error
    assert "调整" in result.error
    assert "重新校准" in result.error
    assert not result.cacheable


def test_missing_calibration_returns_noncacheable_zero(tmp_path, monkeypatch):
    _, speaker = _registered_speaker_db(tmp_path, monkeypatch)

    result = resolve_output_amplitude(speaker, 1.0, "RMS")

    assert result == OutputCalibrationPreparation(amplitude=0.0, max_voltage=None)


@pytest.mark.parametrize("voltage", [None, "invalid", float("nan"), float("inf")])
def test_invalid_target_has_a_diagnostic_and_no_cacheable_amplitude(tmp_path, monkeypatch, voltage):
    repo, speaker = _registered_speaker_db(tmp_path, monkeypatch)
    repo.update_output_amplitude_calibration(speaker["hardware_id"], [0.25, 0.0], max_voltage=2.0)

    result = resolve_output_amplitude(speaker, voltage, "RMS")

    assert result.amplitude is None
    assert result.error
    assert not result.cacheable


def test_nonfinite_prediction_is_not_successfully_prepared(tmp_path, monkeypatch):
    repo, speaker = _registered_speaker_db(tmp_path, monkeypatch)
    repo.update_output_amplitude_calibration(speaker["hardware_id"], [1e308, 0.0], max_voltage=2.0)

    with pytest.warns(RuntimeWarning):
        result = resolve_output_amplitude(speaker, 2.0, "RMS")

    assert result.amplitude is None
    assert result.max_voltage == 2.0
    assert result.error
    assert not result.cacheable


@pytest.mark.parametrize("speaker", [None, {}, {"hardware_id": ""}, {"hardware_id": "unregistered"}])
def test_unregistered_speaker_never_uses_global_calibration(tmp_path, monkeypatch, speaker):
    repo, global_speaker = _registered_speaker_db(tmp_path, monkeypatch)
    repo.update_output_amplitude_calibration(global_speaker["hardware_id"], [0.5, 0.0], max_voltage=2.0)
    _select_speaker(monkeypatch, global_speaker["hardware_id"])

    assert resolve_output_amplitude(speaker, 1.0, "RMS") == OutputCalibrationPreparation(
        amplitude=0.0, max_voltage=None,
    )


def test_cache_reuses_same_key_and_rebuilds_on_change_or_invalidation():
    cache = OutputCalibrationCache()
    calls = []

    def resolve():
        calls.append(True)
        return OutputCalibrationPreparation(amplitude=0.25, max_voltage=2.0, cacheable=True)

    first = cache.get_or_resolve(("speaker", 1.0, "RMS", "config", 0), resolve)
    assert cache.get_or_resolve(("speaker", 1.0, "RMS", "config", 0), resolve) is first
    assert len(calls) == 1
    cache.get_or_resolve(("speaker", 2.0, "RMS", "config", 0), resolve)
    assert len(calls) == 2
    cache.invalidate()
    cache.get_or_resolve(("speaker", 2.0, "RMS", "config", 0), resolve)
    assert len(calls) == 3
    OutputCalibrationCache().get_or_resolve(("speaker", 2.0, "RMS", "config", 0), resolve)
    assert len(calls) == 4


@pytest.mark.parametrize("cacheable", [False, True])
def test_failed_resolution_removes_old_success_and_is_retried(cacheable):
    cache = OutputCalibrationCache()
    success = OutputCalibrationPreparation(amplitude=0.25, max_voltage=2.0, cacheable=True)
    failed = OutputCalibrationPreparation(amplitude=None, max_voltage=0.5, error="range error", cacheable=cacheable)
    calls = []

    def resolve(result):
        calls.append(result)
        return result

    cache.get_or_resolve("old", lambda: resolve(success))
    assert cache.get_or_resolve("new", lambda: resolve(failed)) is failed
    assert cache.get_or_resolve("new", lambda: resolve(failed)) is failed
    assert cache.get_or_resolve("old", lambda: resolve(success)) is success
    assert calls == [success, failed, failed, success]


def test_noncacheable_fallback_is_retried_and_can_recover():
    cache = OutputCalibrationCache()
    fallback = OutputCalibrationPreparation(amplitude=0.0, max_voltage=None)
    success = OutputCalibrationPreparation(amplitude=0.25, max_voltage=2.0, cacheable=True)
    results = iter([fallback, fallback, success])
    calls = []

    def resolve():
        calls.append(True)
        return next(results)

    assert cache.get_or_resolve("key", resolve) is fallback
    assert cache.get_or_resolve("key", resolve) is fallback
    assert cache.get_or_resolve("key", resolve) is success
    assert cache.get_or_resolve("key", resolve) is success
    assert len(calls) == 3


def test_resolver_exception_cannot_leave_previous_cache_usable():
    cache = OutputCalibrationCache()
    success = OutputCalibrationPreparation(amplitude=0.25, max_voltage=2.0, cacheable=True)
    cache.get_or_resolve("old", lambda: success)

    def fail():
        raise ValueError("preparation failed")

    with pytest.raises(ValueError, match="preparation failed"):
        cache.get_or_resolve("new", fail)
    replacement = OutputCalibrationPreparation(amplitude=0.1, max_voltage=2.0, cacheable=True)
    assert cache.get_or_resolve("old", lambda: replacement) is replacement
