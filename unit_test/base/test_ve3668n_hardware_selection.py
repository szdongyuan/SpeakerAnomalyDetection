"""Hardware selection integration, with all files under system pytest TEMP."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from base import hardware_selection as selection
from base.ve3668n_stores import VEInputProfileStore, VECalibrationStore
from unit_test.base.ve3668n_fakes import device_info
from unit_test.base.test_ve3668n_stores import save_measurement


def patch_audio_defaults(monkeypatch):
    """Exercise real manager setters without touching PortAudio or resolving input."""
    from base import sound_device_manager

    class Pair:
        def __init__(self):
            self.raw = [None, 9]
            self.reads, self.writes = [], []

        def __getitem__(self, slot):
            self.reads.append(slot)
            return 101 if self.raw[slot] is None else self.raw[slot]

        def __setitem__(self, slot, value):
            self.writes.append((slot, value))
            self.raw[slot] = value

    pair, calls = Pair(), []

    class Defaults:
        @property
        def device(self):
            return pair

        @device.setter
        def device(self, values):
            calls.append(values)
            pair[0], pair[1] = values

    monkeypatch.setattr(sound_device_manager, "sd", SimpleNamespace(default=Defaults()))
    return SimpleNamespace(pair=pair, calls=calls)


def fake_soundcards(monkeypatch):
    """Two real-looking API buckets and a distinct system fallback, no PA I/O."""
    from base.sound_device_manager import SoundDeviceManager
    from consts import error_code

    defaults = patch_audio_defaults(monkeypatch)
    mic = {"name": "Old mic", "index": 1, "hostapi": 0, "max_input_channels": 2}
    speaker = {"name": "Old output", "index": 2, "hostapi": 0, "max_output_channels": 2}
    asio_mic = {**mic, "name": "ASIO mic", "index": 3, "hostapi": 1}
    asio_speaker = {**speaker, "name": "ASIO output", "index": 4, "hostapi": 1}
    system_mic = {**mic, "name": "System mic", "index": 9, "hostapi": 2}
    devices = {"MME": {"input": [mic], "output": [speaker]},
               "ASIO": {"input": [asio_mic], "output": [asio_speaker]}}
    queries = []

    class FakeSDM(SoundDeviceManager):
        @staticmethod
        def refresh_available_device():
            return None

        @staticmethod
        def get_device_info():
            return deepcopy(devices)

        @staticmethod
        def get_api_info(index):
            return {"name": ("MME", "ASIO", "WASAPI")[index]}

        @staticmethod
        def get_default_device(kind, refresh=False):
            queries.append(kind)
            return error_code.OK, deepcopy(system_mic if kind == "mic" else speaker)

    monkeypatch.setattr(selection, "SoundDeviceManager", FakeSDM)
    return SimpleNamespace(sdm=FakeSDM, mic=mic, speaker=speaker, asio_mic=asio_mic,
        asio_speaker=asio_speaker, devices=devices, queries=queries, defaults=defaults)


@pytest.fixture
def audio_defaults(monkeypatch):
    return patch_audio_defaults(monkeypatch)


@pytest.fixture
def hardware(tmp_path, monkeypatch, audio_defaults):
    mic = {"name": "Old mic", "index": 1, "hostapi": 0, "max_input_channels": 2}
    speaker = {"name": "Old output", "index": 2, "hostapi": 0, "max_output_channels": 2}
    devices = {"MME": {"input": [mic], "output": [speaker]}}
    defaults, applied = [], []
    monkeypatch.setattr(selection.SoundDeviceManager, "refresh_available_device", lambda: None)
    monkeypatch.setattr(selection, "_enumerate_devices", lambda: devices)
    monkeypatch.setattr(selection.SoundDeviceManager, "get_api_info", lambda index: {"name": "MME"})
    monkeypatch.setattr(selection, "_os_default_device", lambda kind: defaults.append(kind) or None)
    monkeypatch.setattr(selection.SoundDeviceManager, "change_default_device", lambda *args: applied.append(args))
    path = tmp_path / "hardware.json"
    monkeypatch.setattr(selection, "_HARDWARE_SELECTION_PATH", str(path))
    return path, mic, speaker, defaults, applied


def saved_record(**overrides):
    return {"schema_version": 1, "backend": "vkinging", "machine_id": "test-machine-1",
            "physical_channels": [7, 1], **overrides}


def write_saved(path, record):
    path.write_text(json.dumps({"api_name": "MME", "speaker_name": "Old output",
                               "speaker_channels": [], "mic_name": "Old mic", "mic_channels": [1],
                               "input_selection": record}), encoding="utf-8")


def test_saved_ve_is_unavailable_without_default_mic_fallback(hardware):
    path, old_mic, speaker, defaults, applied = hardware
    write_saved(path, saved_record())
    mic, output, channels, output_channels = selection.restore_or_default()
    assert mic.get("backend") == "vkinging"
    assert mic["machine_id"] == "test-machine-1"
    assert mic["available"] is False
    assert mic["diagnostic"]
    assert channels == [7, 1]
    assert output == speaker and output_channels == []
    assert "mic" not in defaults and applied == []
    assert "index" not in mic and "hostapi" not in mic


@pytest.mark.parametrize("record", [saved_record(schema_version=2), saved_record(machine_id=""),
    saved_record(physical_channels=[True]), saved_record(sample_rate=51200), None])
def test_corrupt_explicit_selection_never_recovers_as_soundcard(hardware, record):
    path, _, _, defaults, applied = hardware
    write_saved(path, record)
    mic, _, _, _ = selection.restore_or_default()
    assert mic["backend"] == "vkinging" and not mic["available"]
    assert mic["selection_error"]
    assert "mic" not in defaults and not applied


def test_truncated_explicit_ve_json_cannot_fall_back(hardware):
    path, _, _, defaults, applied = hardware
    path.write_text('{"input_selection": {"backend": "vkinging",', encoding="utf-8")
    mic, _, _, _ = selection.restore_or_default()
    assert mic["backend"] == "vkinging" and not mic["available"]
    assert "mic" not in defaults and not applied


def test_legacy_fields_restore_unchanged(hardware):
    path, mic, speaker, defaults, applied = hardware
    path.write_text(json.dumps({"api_name": "MME", "mic_name": "Old mic", "mic_channels": [1],
                               "speaker_name": "Old output", "speaker_channels": []}), encoding="utf-8")
    assert selection.restore_or_default() == (mic, speaker, [1], [])
    assert applied == [(1, 2)] and not defaults


@pytest.mark.parametrize("change", [{}, {"name": "Replugged"}, {"machine_id": "other"},
                                   {"model": "VE3668N-other"}, {"physical_channels": [1]},
                                   {"available": False}])
def test_discovery_confirms_exact_identity_model_and_routes(hardware, tmp_path, change):
    path, _, _, defaults, applied = hardware
    write_saved(path, saved_record())
    mic, _, channels, _ = selection.restore_or_default()
    profiles = VEInputProfileStore(tmp_path / "profiles.json")
    calibrations = VECalibrationStore(tmp_path / "calibrations.json")
    profiles.set_sample_rate(device_info(), 48000, calibrations)
    resolved = selection.resolve_ve_input(mic, channels, [device_info(**change)],
        profile_store=profiles, calibration_store=calibrations)
    expected = not any(key in change for key in ("machine_id", "model", "physical_channels", "available"))
    assert resolved["available"] is expected
    assert resolved["machine_id"] == mic["machine_id"]
    if expected:
        assert resolved["name"] == change.get("name", "Dev1")
        assert resolved["input_config"]["sample_rate"] == 48000
    else:
        assert resolved["diagnostic"]
    assert not defaults and not applied


def test_unknown_saved_rate_is_unavailable_not_defaulted(hardware, tmp_path):
    path, *_ = hardware
    write_saved(path, saved_record())
    mic, _, channels, _ = selection.restore_or_default()
    profiles = VEInputProfileStore(tmp_path / "profiles.json")
    calibrations = VECalibrationStore(tmp_path / "calibrations.json")
    profiles.path.write_text(json.dumps({"schema_version": 1, "devices": {
        mic["machine_id"]: {**device_info()["input_config"], "sample_rate": 102400}}}), encoding="utf-8")
    before = profiles.path.read_bytes()
    resolved = selection.resolve_ve_input(mic, channels, [device_info()],
        profile_store=profiles, calibration_store=calibrations)
    assert not resolved["available"] and "sample_rate" in resolved["diagnostic"]
    assert resolved["input_config"] is None
    assert profiles.path.read_bytes() == before


def test_strict_ve_save_retains_old_mic_fields_and_no_duplicate_rate(hardware, tmp_path):
    path, mic, speaker, defaults, applied = hardware
    assert selection.save_if_changed(mic, speaker, [1], [])
    old = json.loads(path.read_text(encoding="utf-8"))
    profiles = VEInputProfileStore(tmp_path / "profiles.json")
    calibrations = VECalibrationStore(tmp_path / "calibrations.json")
    device = device_info()
    saved = selection.save_ve_selection(device, None, [7, 1], [], profile_store=profiles,
        calibration_store=calibrations, path=path, sample_rate=44100, api_name="MME")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["input_selection"] == saved_record()
    assert payload["mic_name"] == old["mic_name"] and payload["mic_channels"] == old["mic_channels"]
    assert payload["speaker_name"] is None
    assert saved["input_config"]["sample_rate"] == 44100
    assert device["input_config"]["sample_rate"] == 51200
    assert not defaults and not applied
    # Explicit soundcard switch recovers old mic, without writing anything.
    restored = selection.restore_or_default(path=path, soundcard_only=True)
    assert restored[0] == mic and restored[2] == [1]


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["profile", "selection"])
def test_failed_save_keeps_profile_selection_and_calibration_bytes(hardware, tmp_path, monkeypatch, existing, failure):
    path, mic, speaker, *_ = hardware
    selection.save_if_changed(mic, speaker, [1], [])
    profiles = VEInputProfileStore(tmp_path / "profiles.json")
    calibrations = VECalibrationStore(tmp_path / "calibrations.json")
    device = device_info()
    save_measurement(calibrations, device, 7)
    if existing:
        profiles.set_sample_rate(device, 51200, calibrations)
    before_profile = profiles.path.read_bytes() if existing else None
    before_hardware, before_calibration = path.read_bytes(), calibrations.path.read_bytes()
    before_device = deepcopy(device)
    if failure == "profile":
        def fail(*args):
            raise OSError("profile write denied")
        monkeypatch.setattr(profiles, "set_sample_rate", fail)
    else:
        monkeypatch.setattr(selection, "_atomic_write_json", lambda *args: False)
    with pytest.raises(OSError, match="profile|selection"):
        selection.save_ve_selection(device, speaker, [7, 1], [], profile_store=profiles,
            calibration_store=calibrations, path=path, sample_rate=48000)
    assert (profiles.path.read_bytes() if profiles.path.exists() else None) == before_profile
    assert path.read_bytes() == before_hardware
    assert calibrations.path.read_bytes() == before_calibration
    assert device == before_device


def test_successful_unchanged_selection_is_not_a_write_failure(hardware, tmp_path, monkeypatch):
    path, *_ = hardware
    profiles = VEInputProfileStore(tmp_path / "profiles.json")
    calibrations = VECalibrationStore(tmp_path / "calibrations.json")
    device = device_info()
    for channel in (7, 1):
        save_measurement(calibrations, device, channel)
    before = calibrations.path.read_bytes()
    for rate in (44100, 48000, 51200, 44100):
        saved = selection.save_ve_selection(device, None, [7, 1], [], profile_store=profiles,
            calibration_store=calibrations, path=path, sample_rate=rate)
        assert saved["input_config"]["sample_rate"] == rate
        assert calibrations.path.read_bytes() == before
        # No new hardware write is required; False here would mean failure if called.
        monkeypatch.setattr(selection, "_atomic_write_json", lambda *args: False)


def test_profiles_stay_independent_between_machine_ids(hardware, tmp_path):
    path, *_ = hardware
    profiles = VEInputProfileStore(tmp_path / "profiles.json")
    calibrations = VECalibrationStore(tmp_path / "calibrations.json")
    for machine_id, rate in (("one", 44100), ("two", 48000), ("one", 51200)):
        selection.save_ve_selection(device_info(machine_id=machine_id), None, [7, 1], [],
            profile_store=profiles, calibration_store=calibrations, path=path, sample_rate=rate)
    assert profiles.load(device_info(machine_id="one"), calibrations)["sample_rate"] == 51200
    assert profiles.load(device_info(machine_id="two"), calibrations)["sample_rate"] == 48000


def test_legacy_save_api_handles_ve_without_overwriting_soundcard_fields(hardware):
    path, mic, speaker, *_ = hardware
    assert selection.save_if_changed(mic, speaker, [1], [])
    assert selection.save_if_changed(device_info(), speaker, [7, 1], [])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["input_selection"] == saved_record()
    assert payload["mic_name"] == mic["name"] and payload["mic_channels"] == [1]
    assert selection.save_if_changed(device_info(), speaker, [7, 1], []) is False


@pytest.mark.parametrize("input_default", [None, -1, 101])
def test_output_only_setter_preserves_raw_input_default(audio_defaults, input_default):
    audio_defaults.pair.raw[0] = input_default
    selection.SoundDeviceManager.change_default_output_device(2)
    assert audio_defaults.pair.raw == [input_default, 2]
    assert audio_defaults.pair.writes == [(1, 2)]
    assert audio_defaults.pair.reads == [] and audio_defaults.calls == []


def test_ve_startup_restore_applies_selected_output_only(hardware, audio_defaults):
    path, _, speaker, defaults, applied = hardware
    write_saved(path, saved_record())
    mic, output, channels, _ = selection.restore_or_default()
    assert not mic["available"] and channels == [7, 1] and output == speaker
    assert audio_defaults.pair.raw == [None, 2]
    assert audio_defaults.pair.writes == [(1, 2)]
    assert not audio_defaults.pair.reads and not applied and not defaults


def test_ve_restore_without_output_does_not_apply_defaults(hardware, audio_defaults, monkeypatch):
    path, *_ = hardware
    write_saved(path, saved_record())
    monkeypatch.setattr(selection, "_enumerate_devices", lambda: {})
    assert selection.restore_or_default()[1] is None
    assert audio_defaults.pair.raw == [None, 9]
    assert not audio_defaults.pair.writes and not audio_defaults.calls


def test_soundcard_staging_restore_is_non_mutating_but_default_api_still_applies(hardware, audio_defaults):
    path, mic, speaker, _, applied = hardware
    write_saved(path, saved_record())
    before = path.read_bytes()
    assert selection.restore_or_default(soundcard_only=True, apply_defaults=False) == (mic, speaker, [1], [])
    assert not applied and not audio_defaults.pair.writes
    assert path.read_bytes() == before
    assert selection.restore_or_default(soundcard_only=True) == (mic, speaker, [1], [])
    assert applied == [(1, 2)]


@pytest.mark.parametrize("root, recoverable", [
    ([{"input_selection": saved_record()}], True),
    ([saved_record()], True),
    ([[{"input_selection": saved_record()}]], True),
    (json.dumps({"input_selection": saved_record()}), False),
    ([{"input_selection": saved_record()}, {"input_selection": saved_record(machine_id="other")}], False),
])
def test_recognizable_ve_in_invalid_root_retains_unavailable_provenance(hardware, monkeypatch, root, recoverable):
    path, old_mic, speaker, defaults, applied = hardware
    path.write_text(json.dumps(root), encoding="utf-8")
    monkeypatch.setattr(selection, "_os_default_device", lambda kind: defaults.append(kind) or
                        (old_mic if kind == "mic" else speaker))
    mic, _, channels, _ = selection.restore_or_default()
    assert mic.get("backend") == "vkinging" and not mic["available"]
    assert mic["selection_error"] and mic["diagnostic"]
    assert "mic" not in defaults and not applied
    assert "index" not in mic and "hostapi" not in mic
    if recoverable:
        assert mic["machine_id"] == "test-machine-1" and channels == [7, 1]
    else:
        assert mic["machine_id"] is None and channels == []


@pytest.mark.parametrize("root", [[], ["legacy"], "legacy", 1, None])
def test_unmarked_legacy_invalid_root_keeps_opportunistic_defaults(hardware, monkeypatch, root):
    path, mic, speaker, defaults, applied = hardware
    path.write_text(json.dumps(root), encoding="utf-8")
    monkeypatch.setattr(selection, "_os_default_device", lambda kind: defaults.append(kind) or
                        (mic if kind == "mic" else speaker))
    assert selection.restore_or_default() == (mic, speaker, [0], [0, 1])
    assert defaults == ["mic", "speaker"] and applied == [(1, 2)]


@pytest.mark.parametrize("save_api", ["strict", "legacy"])
def test_cross_api_repeated_ve_saves_restore_prior_soundcard_configuration(tmp_path, monkeypatch, save_api):
    audio = fake_soundcards(monkeypatch)
    path = tmp_path / "hardware.json"
    selection.save_if_changed(audio.mic, audio.speaker, [1], [1, 0], path=path)
    old = json.loads(path.read_text(encoding="utf-8"))
    assert "input_selection" not in old  # Unmarked old configurations need no migration.
    for index, (machine_id, output) in enumerate([
            ("test-machine-1", audio.asio_speaker), ("two", audio.speaker),
            ("test-machine-1", audio.asio_speaker)]):
        device = device_info(machine_id=machine_id)
        # Recreate stores on every save as well as restoring from disk.
        profiles = VEInputProfileStore(tmp_path / "profiles.json")
        calibrations = VECalibrationStore(tmp_path / "calibrations.json")
        if save_api == "strict":
            legacy = dict(legacy_mic_device=audio.mic, legacy_mic_channels=[1]) if index == 0 else {}
            selection.save_ve_selection(device, output, [7, 1], [], path=path,
                profile_store=profiles, calibration_store=calibrations,
                api_name=audio.sdm.get_api_info(output["hostapi"])["name"], **legacy)
        else:
            assert selection.save_if_changed(device, output, [7, 1], [], path=path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["api_name"] == audio.sdm.get_api_info(output["hostapi"])["name"]
        assert payload["soundcard_selection"] == old
        assert payload["input_selection"] == saved_record(machine_id=machine_id)
        assert payload["mic_name"] == old["mic_name"] and payload["mic_channels"] == [1]
        assert "sample_rate" not in path.read_text(encoding="utf-8")
        mic, speaker, channels, out_channels = selection.restore_or_default(path=path)
        assert mic["machine_id"] == machine_id and not mic["available"]
        assert (speaker, channels, out_channels) == (output, [7, 1], [])
        assert audio.defaults.pair.raw == [None, output["index"]]
        assert not audio.defaults.calls and not audio.defaults.pair.reads and not audio.queries
        before = path.read_bytes(), list(audio.defaults.pair.writes)
        assert selection.restore_or_default(path=path, soundcard_only=True, apply_defaults=False) == (
            audio.mic, audio.speaker, [1], [1, 0])
        assert (path.read_bytes(), audio.defaults.pair.writes) == before
        assert not audio.queries
        assert not selection.save_if_changed(device, output, [7, 1], [], path=path)
        assert path.read_bytes() == before[0]


@pytest.mark.parametrize("layout", ["bare", "nested", "nested_bare", "multiple"])
def test_wrong_dictionary_layout_retains_ve_provenance_without_input_defaults(tmp_path, monkeypatch, layout):
    audio = fake_soundcards(monkeypatch)
    roots = {"bare": saved_record(), "nested": {"hardware": {"input_selection": saved_record()}},
             "nested_bare": {"hardware": saved_record()},
             "multiple": {"one": saved_record(), "two": saved_record(machine_id="two")}}
    path = tmp_path / "hardware.json"
    path.write_text(json.dumps(roots[layout]), encoding="utf-8")
    before = path.read_bytes()
    result = selection.restore_or_default(path=path)
    assert len(result) == 4
    mic, speaker, channels, output_channels = result
    assert mic.get("backend") == "vkinging" and not mic["available"]
    assert mic["selection_error"] and mic["diagnostic"]
    assert "root must be an object" not in mic["diagnostic"]
    assert "index" not in mic and "hostapi" not in mic
    assert mic["machine_id"] == (None if layout == "multiple" else "test-machine-1")
    assert channels == ([] if layout == "multiple" else [7, 1])
    assert (speaker, output_channels) == (audio.speaker, [0, 1])
    assert audio.queries == ["speaker"]
    assert audio.defaults.pair.raw == [None, 2] and audio.defaults.pair.writes == [(1, 2)]
    assert not audio.defaults.pair.reads and not audio.defaults.calls
    assert path.read_bytes() == before


@pytest.mark.parametrize("canonical", [False, True])
def test_dictionary_recovery_respects_canonical_and_unmarked_legacy_choices(tmp_path, monkeypatch, canonical):
    audio = fake_soundcards(monkeypatch)
    path = tmp_path / "hardware.json"
    selection.save_if_changed(audio.mic, audio.speaker, [1], [], path=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if canonical:
        payload.update(input_selection=saved_record(), hardware={"input_selection": saved_record(machine_id="other")})
    else:
        payload["hardware"] = {"api_name": "ASIO", "mic_name": "ASIO mic"}
    path.write_text(json.dumps(payload), encoding="utf-8")
    mic, speaker, channels, _ = selection.restore_or_default(path=path)
    if canonical:
        assert mic["machine_id"] == "test-machine-1" and not mic.get("selection_error")
        assert channels == [7, 1] and not audio.defaults.calls
        assert audio.defaults.pair.writes == [(1, 2)]
    else:
        assert mic == audio.mic and channels == [1]
        assert audio.defaults.calls == [(1, 2)] and audio.defaults.pair.raw == [1, 2]
    assert speaker == audio.speaker and not audio.queries
