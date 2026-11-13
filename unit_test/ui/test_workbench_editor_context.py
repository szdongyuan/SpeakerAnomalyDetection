"""Independent editor queues never borrow the displayed product configuration."""
from copy import deepcopy
import json
import logging
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from base.vkinging_stores import VECalibrationStore, VEInputProfileStore, VEStoreFormatError
from ui.sequence.workbench_editor_context import (
    WorkbenchEditorContext, resolve_editor_ve_device,
)
from unit_test.base.vkinging_fakes import device_info


def queue(detail=None):
    return [{"seq1": {
        "acq": {"mode": "RECORD_ONLY", "detail": {} if detail is None else detail},
        "analysis_list": {"display_sequence": ["spl"],
                          "spl": {"type": "SPL", "analysis_channel": 0,
                                  "analysis_channels": [0]}},
    }}]


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def files(tmp_path, monkeypatch):
    registry = tmp_path / "registry.json"
    monkeypatch.setattr("base.load_config.SEQUENCE_CONFIG_REGISTRY_PATH", str(registry))
    return SimpleNamespace(root=tmp_path, registry=registry,
                           logger=logging.getLogger("editor-context-test"))


def test_load_resolves_registry_relative_paths_from_another_cwd(files, monkeypatch):
    data = queue({"sample_rate": 48000, "ve_range_index": 5})
    path = write(files.root / "chosen.json", data)
    write(files.registry, {"Chosen": path.name, "using_config_path": path.name})
    other = files.root / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    before = files.registry.read_bytes(), path.read_bytes()

    context = WorkbenchEditorContext.load(files.logger)

    assert context.selected_path == str(path)
    assert context.queue_data == data
    assert context.test_full_scale_peak_volts() == .1
    assert before == (files.registry.read_bytes(), path.read_bytes())


def test_explicit_none_selection_does_not_fall_back(files):
    path = write(files.root / "valid.json", queue())
    write(files.registry, {"Saved": path.name, "默认配置": path.name, "using_config_path": None})
    before = files.registry.read_bytes()
    for _ in range(2):
        context = WorkbenchEditorContext.load(files.logger)
        assert context.selected_path is None and context.queue_data == []
        assert context.test_full_scale_peak_volts() is None
    assert files.registry.read_bytes() == before


@pytest.mark.parametrize("selection", ["missing.json", "invalid.json", 3, ""])
def test_invalid_selection_falls_back_to_sorted_user_entry_without_writes(files, selection):
    z = write(files.root / "z.json", queue({"sample_rate": 96000}))
    a = write(files.root / "a.json", queue({"sample_rate": 48000}))
    default = write(files.root / "default.json", queue())
    write(files.root / "invalid.json", [{"seq1": {"acq": None}}])
    write(files.registry, {"Z": z.name, "A": a.name, "默认配置": default.name,
                           "using_config_path": selection})
    before = {path: path.read_bytes() for path in files.root.glob("*.json")}

    context = WorkbenchEditorContext.load(files.logger)

    assert context.selected_path == str(a) and context.queue_data == queue({"sample_rate": 48000})
    assert all(path.read_bytes() == content for path, content in before.items())


def test_default_fallback_is_used_only_after_valid_user_entries(files):
    path = write(files.root / "default.json", queue())
    write(files.registry, {"Broken": "missing.json", "默认配置": path.name})
    before = files.registry.read_bytes()
    context = WorkbenchEditorContext.load(files.logger)
    assert context.selected_path == str(path) and context.queue_data == queue()
    assert files.registry.read_bytes() == before


@pytest.mark.parametrize("content", [None, "{", [], {}, [{"seq1": {}}],
    [{"seq1": {"acq": {"detail": {}}, "analysis_list": {"display_sequence": ["missing"]}}}],
    [{"seq1": {"acq": {"detail": {}}, "analysis_list": {"spl": {"type": 2}}}}],
    queue() + [{"seq2": {"acq": {"detail": []}}}],
])
def test_missing_invalid_or_display_only_queue_leaves_empty_editor_context(files, content):
    path = files.root / "candidate.json"
    if content == "{":
        path.write_text(content, encoding="utf-8")
    elif content is not None:
        write(path, content)
    write(files.registry, {"Saved": path.name, "using_config_path": path.name})
    before = files.registry.read_bytes()
    context = WorkbenchEditorContext.load(files.logger)
    assert context.selected_path is None and context.queue_data == []
    assert context.update_channels([7, 1]) is True
    assert files.registry.read_bytes() == before


@pytest.mark.parametrize("registry_content", [None, "{", "[]"])
def test_missing_or_invalid_registry_does_not_block_workbench(files, registry_content):
    if registry_content is not None:
        files.registry.write_text(registry_content, encoding="utf-8")
    context = WorkbenchEditorContext.load(files.logger)
    assert context.selected_path is None and context.queue_data == []
    if registry_content is None:
        assert not files.registry.exists()
    else:
        assert files.registry.read_text(encoding="utf-8") == registry_content


def test_channel_updates_only_change_owned_memory_and_preserve_queue_rules(files):
    data = queue()
    stimulus = deepcopy(data[0]["seq1"])
    stimulus["acq"]["mode"] = "PLAY_AND_RECORD"
    wav = deepcopy(data[0]["seq1"])
    wav["analysis_channel_basis"] = "wav"
    data += [{"seq2": stimulus, "seq3": wav}]
    path = write(files.root / "queue.json", data)
    write(files.registry, {"using_config_path": path.name})
    before = path.read_bytes(), files.registry.read_bytes()
    first = WorkbenchEditorContext.load(files.logger)
    second = WorkbenchEditorContext.load(files.logger)

    assert first.update_channels([7, 1, 7]) is True

    spl = first.queue_data[0]["seq1"]["analysis_list"]["spl"]
    assert spl == {"type": "SPL", "analysis_channel": 1, "analysis_channels": [1, 7]}
    assert first.queue_data[1] == data[1]
    assert second.queue_data == data
    assert before == (path.read_bytes(), files.registry.read_bytes())


@pytest.mark.parametrize("channels", [[], [-1], [True], [1.5], "1"])
def test_invalid_channels_report_failure_without_mutation(files, channels, caplog):
    path = write(files.root / "queue.json", queue())
    write(files.registry, {"using_config_path": path.name})
    context = WorkbenchEditorContext.load(files.logger)
    before = deepcopy(context.queue_data)
    with caplog.at_level(logging.WARNING):
        assert context.update_channels(channels) is False
    assert context.queue_data == before
    assert "Independent analysis channel update failed" in caplog.text


def test_invalid_in_memory_queue_is_diagnosed_without_partial_channel_updates(files, caplog):
    path = write(files.root / "queue.json", queue())
    write(files.registry, {"using_config_path": path.name})
    context = WorkbenchEditorContext.load(files.logger)
    context.queue_data.append({"seq2": {"acq": None}})
    before = deepcopy(context.queue_data)
    with caplog.at_level(logging.WARNING):
        assert context.update_channels([7]) is False
    assert context.queue_data == before
    assert "invalid runtime structure" in caplog.text


@pytest.mark.parametrize("kind,same", [("queue", True), ("queue", False), ("product", True)])
def test_deletion_clears_only_selected_queue_and_never_recreates_it(files, kind, same):
    path = write(files.root / "queue.json", queue({"ve_range_index": 5}))
    write(files.registry, {"using_config_path": path.name})
    context = WorkbenchEditorContext.load(files.logger)
    before = deepcopy(context.queue_data)
    target_path = os.path.join(str(files.root), ".", path.name if same else "other.json")
    if kind == "queue" and same:
        path.unlink()
    context.configuration_deleted(SimpleNamespace(kind=kind, path=target_path))
    if kind == "queue" and same:
        assert context.selected_path is None and context.queue_data == []
        assert context.test_full_scale_peak_volts() is None
        assert context.update_channels([7]) is True
        assert not path.exists()
    else:
        assert context.selected_path == str(path) and context.queue_data == before


def test_explicit_queue_rate_and_range_do_not_read_stores(files):
    data = queue({"sample_rate": 96000, "ve_range_index": 5})
    path = write(files.root / "queue.json", data)
    write(files.registry, {"using_config_path": path.name})
    context = WorkbenchEditorContext.load(files.logger)
    device = device_info()
    original = deepcopy(device)
    stores = Mock(side_effect=AssertionError("explicit queue rate must not read stores"))
    profiles = SimpleNamespace(load=stores)

    resolved = context.resolve_ve_device(device, profiles, stores)

    assert resolved["input_config"] == {"sample_rate": 96000, "input_mode": "IEPE",
                                         "unit": "V", "range_min": -.1, "range_max": .1}
    assert context.test_full_scale_peak_volts() == .1
    assert device == original and context.queue_data == data
    stores.assert_not_called()


def test_missing_rate_uses_injected_profile_but_queue_owns_range(files):
    profiles = VEInputProfileStore(files.root / "profile.json")
    calibrations = VECalibrationStore(files.root / "calibration.json")
    device = device_info()
    profiles.set_sample_rate(device, 48000, calibrations)
    before = profiles.path.read_bytes()
    resolved = resolve_editor_ve_device(queue({"ve_range_index": 5}), device, profiles, calibrations)
    assert resolved["input_config"]["sample_rate"] == 48000
    assert resolved["input_config"]["range_max"] == .1
    assert profiles.path.read_bytes() == before and not calibrations.path.exists()


@pytest.mark.parametrize("data", [[], queue()])
def test_missing_profile_uses_runtime_default_without_creating_file(files, data):
    profiles = VEInputProfileStore(files.root / "profile.json")
    calibrations = VECalibrationStore(files.root / "calibration.json")
    resolved = resolve_editor_ve_device(data, device_info(), profiles, calibrations)
    assert resolved["input_config"]["sample_rate"] == 51200
    assert not profiles.path.exists() and not calibrations.path.exists()


@pytest.mark.parametrize("profiles,calibrations", [(None, None), (None, object()), (object(), None)])
def test_missing_rate_requires_both_shared_stores(profiles, calibrations):
    with pytest.raises(ValueError, match="VE recording requires the shared profile and calibration stores"):
        resolve_editor_ve_device(queue(), device_info(), profiles, calibrations)


@pytest.mark.parametrize("rate", [None, True, 48000.5, -1, "48000"])
def test_explicit_invalid_rate_is_not_replaced_by_profile(rate):
    profiles = SimpleNamespace(load=Mock(side_effect=AssertionError("unexpected profile read")))
    with pytest.raises(ValueError):
        resolve_editor_ve_device(queue({"sample_rate": rate}), device_info(), profiles, object())
    profiles.load.assert_not_called()


@pytest.mark.parametrize("overrides", [{"machine_id": ""}, {"physical_channels": [-1]},
                                       {"physical_channels": [True]}, {"available": "yes"}])
def test_invalid_device_and_physical_channels_keep_validation_errors(overrides):
    with pytest.raises(ValueError):
        resolve_editor_ve_device(queue({"sample_rate": 48000}), device_info(**overrides), None, None)


@pytest.mark.parametrize("failure", ["format", "io"])
def test_profile_errors_propagate_without_rewriting_saved_configuration(files, failure):
    profiles = VEInputProfileStore(files.root / "profile.json")
    calibrations = VECalibrationStore(files.root / "calibration.json")
    if failure == "format":
        profiles.path.write_text("{", encoding="utf-8")
        expected = VEStoreFormatError
    else:
        profiles.path.mkdir()
        expected = OSError
    with pytest.raises(expected):
        resolve_editor_ve_device(queue(), device_info(), profiles, calibrations)
    if failure == "format":
        assert profiles.path.read_text(encoding="utf-8") == "{"
    assert not calibrations.path.exists()


@pytest.mark.parametrize("index,expected", list(enumerate((10., 5., 2.5, 1., .5, .1, .02)))
                         + [(None, None), (-1, None), (True, None), (7, None)])
def test_calibration_scale_uses_same_independent_queue_without_resolving_device(files, index, expected):
    data = queue({"ve_range_index": index, "sample_rate": "invalid"})
    path = write(files.root / "queue.json", data)
    write(files.registry, {"using_config_path": path.name})
    context = WorkbenchEditorContext.load(files.logger)
    assert context.test_full_scale_peak_volts() == expected
    assert context.queue_data == data


@pytest.mark.parametrize("entry", ["signature", "prewarm"])
def test_main_window_resolves_loaded_editor_queue_through_new_helper(monkeypatch, entry):
    from ui.sequence import workbench_editor_context
    from unit_test.ui.test_vkinging_prewarm_trigger import _WindowHarness

    window = _WindowHarness()
    window.editor_context.queue_data = queue({"sample_rate": 48000, "ve_range_index": 5})
    device = device_info()
    resolve = Mock(wraps=resolve_editor_ve_device)
    monkeypatch.setattr(workbench_editor_context, "resolve_editor_ve_device", resolve)
    if entry == "signature":
        signature = window._ve_signature(device, [7, 1])
    else:
        assert window._try_start_ve_prewarm(device, [7, 1], "test") == "accepted"
        signature = window.recording_bridge.calls[0].signature
    assert signature[-5:] == (48000, "IEPE", "V", -.1, .1)
    resolve.assert_called_once_with(window.editor_context.queue_data, device,
                                    window.ve_profile_store, window.ve_calibration_store)
