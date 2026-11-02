"""The real opening method snapshots only the queue's display voltage range."""
from copy import deepcopy
from types import SimpleNamespace
from unittest import mock

import pytest

from unit_test.base.ve3668n_fakes import device_info
from unit_test.test_input_calibration_runtime import DEVICE, ROOT, _load_method


def _queue(detail):
    return [{"seq1": {"acq": {"detail": detail}}}]


@pytest.fixture
def opening(monkeypatch):
    from base import ve3668n_calibration_settings as settings
    from base import ve3668n_recording_config as recording
    from ui import calibration_window as calibration
    from ui.sequence.sequence_widget_config_ops import SequenceWidgetConfigOpsMixin

    forbidden = mock.Mock(side_effect=AssertionError("range handoff performed extra I/O"))
    for module, name in (
        (settings, "load_ve_calibration_settings"),
        (calibration, "load_ve_calibration_settings"),
        (recording, "resolve_ve_recording_config"),
        (recording, "resolve_ve_recording_device"),
        (SequenceWidgetConfigOpsMixin, "_current_ve_recording_device"),
    ):
        monkeypatch.setattr(module, name, forbidden)
    opened = []

    class FakeCalibrationWindow:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.input_calibration_flag = True
            self.exec = mock.Mock()
            opened.append(self)

    messages = SimpleNamespace(warning=mock.Mock())
    host = SimpleNamespace(
        mic=device_info(input_config={"range_min": -123, "range_max": 123}),
        mic_channels=[7, 1],
        recording_bridge=object(),
        ve_profile_store=SimpleNamespace(load=forbidden),
        ve_calibration_store=mock.Mock(),
        sequence_window=SimpleNamespace(sequence_config=[], update_v2pa_factor=mock.Mock()),
        _calibration_admission_available=mock.Mock(return_value=True),
        default_logger=mock.Mock(),
    )
    open_dialog = _load_method(
        ROOT / "main_window.py", "MainWindow", "on_calibration_window_init",
        {"CalibrationWindow": FakeCalibrationWindow, "QMessageBox": messages,
         "ve_failure_text": lambda _: "unavailable"},
    )
    yield SimpleNamespace(host=host, open=lambda: open_dialog(host), opened=opened,
                          messages=messages)
    forbidden.assert_not_called()
    assert host.ve_calibration_store.mock_calls == []


@pytest.mark.parametrize("index,expected", enumerate((10., 5., 2.5, 1., .5, .1, .02)))
def test_vk_handoff_uses_current_queue_range_without_io(opening, index, expected):
    host = opening.host
    host.sequence_window.sequence_config = _queue(
        {"ve_range_index": index, "sample_rate": "broken"})
    before_queue = deepcopy(host.sequence_window.sequence_config)
    before_device = deepcopy(host.mic)

    opening.open()

    dialog, = opening.opened
    assert dialog.kwargs == {
        "input_device": host.mic, "input_channels": host.mic_channels,
        "recording_bridge": host.recording_bridge,
        "ve_profile_store": host.ve_profile_store,
        "ve_calibration_store": host.ve_calibration_store,
        "test_full_scale_peak_volts": expected,
    }
    assert dialog.ve_profile_store is host.ve_profile_store
    assert dialog.ve_calibration_store is host.ve_calibration_store
    dialog.exec.assert_called_once_with()
    host.sequence_window.update_v2pa_factor.assert_not_called()
    opening.messages.warning.assert_not_called()
    assert host.sequence_window.sequence_config == before_queue
    assert host.mic == before_device


@pytest.mark.parametrize("queue,expected", [
    (_queue({}), 10.), (None, None), ([], None), ({}, None), ((), None),
    ([None], None), ([{}], None), ([{"seq1": None}], None),
    ([{"seq1": {}}], None), ([{"seq1": {"acq": []}}], None),
    ([{"seq1": {"acq": {}}}], None), (_queue(None), None), (_queue([]), None),
] + [(_queue({"ve_range_index": value}), None)
     for value in (True, False, "2", 2., -1, 7, 99, None)])
def test_vk_unavailable_range_still_opens_without_warning(opening, queue, expected):
    opening.host.sequence_window.sequence_config = queue
    before = deepcopy(queue)

    opening.open()

    dialog, = opening.opened
    assert dialog.kwargs["test_full_scale_peak_volts"] == expected
    dialog.exec.assert_called_once_with()
    opening.messages.warning.assert_not_called()
    assert opening.host.sequence_window.sequence_config == before


@pytest.mark.parametrize("missing", ["sequence_window", "sequence_config"])
def test_vk_missing_sequence_context_is_unavailable(opening, missing):
    if missing == "sequence_window":
        del opening.host.sequence_window
    else:
        del opening.host.sequence_window.sequence_config
    opening.open()
    assert opening.opened[0].kwargs["test_full_scale_peak_volts"] is None
    opening.messages.warning.assert_not_called()


def test_reopening_takes_new_snapshot_without_changing_existing_dialog(opening):
    queue = _queue({"ve_range_index": 2})
    opening.host.sequence_window.sequence_config = queue
    opening.open()
    queue[0]["seq1"]["acq"]["detail"]["ve_range_index"] = 5
    opening.open()
    assert [dialog.kwargs["test_full_scale_peak_volts"]
            for dialog in opening.opened] == [2.5, .1]


def test_soundcard_opening_preserves_original_kwargs_and_success_refresh(opening):
    opening.host.mic = DEVICE.copy()
    opening.host.sequence_window.sequence_config = _queue({"ve_range_index": 2})
    opening.open()
    dialog, = opening.opened
    assert dialog.kwargs == {
        "input_device": opening.host.mic, "input_channels": opening.host.mic_channels,
        "recording_bridge": opening.host.recording_bridge,
    }
    opening.host.sequence_window.update_v2pa_factor.assert_called_once_with()


@pytest.mark.parametrize("blocked", ["busy", "unavailable"])
def test_existing_admission_blocks_dialog_creation(opening, blocked):
    if blocked == "busy":
        opening.host._calibration_admission_available.return_value = False
    else:
        opening.host.mic["available"] = False
    opening.open()
    assert opening.opened == []
    opening.messages.warning.assert_called_once()
