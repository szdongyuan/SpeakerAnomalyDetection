import os
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from base.soundcard_calibration_manager import (
    get_mic_v2pa_factor,
    save_mic_input_calibration,
)
from ui.calibration_window import CalibrationWindow, InputCalibration


DEVICE = {
    "index": 7,
    "name": "Test Microphone",
    "hostapi": 3,
    "max_input_channels": 2,
}


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def isolated_logger():
    logger = SimpleNamespace(
        info=mock.Mock(),
        warning=mock.Mock(),
        error=mock.Mock(),
    )
    with mock.patch(
        "ui.calibration_window.LogManager.set_log_handler",
        return_value=logger,
    ):
        yield


class _FakeProcessor:
    def __init__(self, data=None, *, target_samples=4, error_message=""):
        self.data = np.asarray(
            data if data is not None else [0.1, 0.1, 0.1, 0.1],
            dtype=np.float32,
        )
        self.target_samples = target_samples
        self.sample_rate = 2
        self.is_recording = False
        self.error_occurred = bool(error_message)
        self.error_message = error_message
        self.process_calls = []
        self.stop_calls = 0

    def process_queue(self, emit_signal=True):
        self.process_calls.append(emit_signal)

    def get_recorded_data(self):
        return self.data

    def stop_streaming(self):
        self.stop_calls += 1
        self.is_recording = False


@pytest.mark.parametrize(
    "input_device,input_channels",
    [
        (None, [0]),
        (DEVICE, []),
        (DEVICE, [0, 1]),
        (DEVICE, [-1]),
    ],
)
def test_input_calibration_requires_one_valid_channel(
    qapp,
    input_device,
    input_channels,
):
    widget = InputCalibration(input_device, input_channels)
    widget.calibration_popup = mock.Mock()

    with mock.patch(
        "ui.calibration_window.stream_record_without_play"
    ) as start_recording:
        started = widget.clicked_calibration()

    assert started is False
    start_recording.assert_not_called()
    widget.calibration_popup.assert_called_once()


def test_calibration_uses_selected_device_and_physical_channel(qapp):
    widget = InputCalibration(DEVICE, [1])
    processor = _FakeProcessor()

    with mock.patch(
        "ui.calibration_window.stream_record_without_play",
        return_value=(processor, 44100),
    ) as start_recording:
        started = widget.clicked_calibration()

    assert started is True
    recorded_dict = start_recording.call_args.args[0]
    assert recorded_dict["device"] is DEVICE
    assert recorded_dict["input_channels"] == [1]
    assert recorded_dict["num_frames"] == 441000
    widget.cancel_calibration()


def test_streaming_poll_uses_normalized_queue_processing(qapp):
    widget = InputCalibration(DEVICE, [1])
    processor = _FakeProcessor()
    processor.is_recording = True
    widget.streaming_processor = processor

    widget._poll_streaming_queue()

    assert processor.process_calls == [False]
    assert widget.streaming_processor is processor


def test_completed_calibration_saves_before_emitting_success(qapp):
    widget = InputCalibration(DEVICE, [1])
    processor = _FakeProcessor()
    widget.streaming_processor = processor
    widget._calculate_spl_from_data = mock.Mock(return_value=90.0)
    widget.calibration_popup = mock.Mock()
    emitted = []
    widget.calibration_finished.connect(emitted.append)

    with mock.patch(
        "ui.calibration_window.save_mic_input_calibration",
        return_value=(True, "saved"),
    ) as save:
        widget._on_streaming_complete()

    save.assert_called_once_with(
        v2pa_factor=pytest.approx(10 ** (4.0 / 20.0)),
        input_device=DEVICE,
        input_channel=1,
        standard_spl_db=94.0,
        sample_rate_hz=2,
        duration_seconds=2.0,
    )
    assert emitted == [True]
    assert widget.streaming_processor is None
    assert float(widget.v2pa_factor_lineedit.text()) > 0.0
    widget.calibration_popup.assert_called_once_with(success_flag=True)


def test_completed_calibration_round_trips_through_runtime_resolver(qapp, tmp_path):
    calibration_path = tmp_path / "mic_input_calibration.json"
    widget = InputCalibration(DEVICE, [1])
    widget.streaming_processor = _FakeProcessor()
    widget._calculate_spl_from_data = mock.Mock(return_value=90.0)
    widget.calibration_popup = mock.Mock()

    def save_to_temporary_path(**kwargs):
        return save_mic_input_calibration(
            **kwargs,
            calibration_path=str(calibration_path),
            calibrated_at="2026-08-12T10:00:00+08:00",
        )

    with mock.patch(
        "base.soundcard_calibration_manager.SoundDeviceManager.get_api_info",
        return_value={"name": "Test API"},
    ), mock.patch(
        "ui.calibration_window.save_mic_input_calibration",
        side_effect=save_to_temporary_path,
    ):
        widget._on_streaming_complete()
        factor = get_mic_v2pa_factor(
            DEVICE,
            [1],
            str(calibration_path),
        )

    assert factor == pytest.approx(10 ** (4.0 / 20.0))
    assert calibration_path.exists()


@pytest.mark.parametrize(
    "processor,save_result",
    [
        (_FakeProcessor(data=[0.1, 0.1], target_samples=4), (True, "saved")),
        (_FakeProcessor(error_message="device lost"), (True, "saved")),
        (_FakeProcessor(), (False, "read only")),
    ],
)
def test_failure_never_emits_success_or_replaces_result(
    qapp,
    processor,
    save_result,
):
    widget = InputCalibration(DEVICE, [1])
    widget.streaming_processor = processor
    widget._calculate_spl_from_data = mock.Mock(return_value=90.0)
    widget.calibration_popup = mock.Mock()
    emitted = []
    widget.calibration_finished.connect(emitted.append)

    with mock.patch(
        "ui.calibration_window.save_mic_input_calibration",
        return_value=save_result,
    ):
        widget._on_streaming_complete()

    assert emitted == [False]
    assert widget.streaming_processor is None
    assert widget.v2pa_factor_lineedit.text() == ""
    assert processor.stop_calls == 1
    assert widget.calibration_popup.call_args.kwargs["success_flag"] is False


def test_dialog_marks_success_only_after_finished_signal(qapp):
    dialog = CalibrationWindow(input_device=DEVICE, input_channels=[1])

    assert dialog.input_calibration_flag is False
    dialog._on_input_calibration_finished(False)
    assert dialog.input_calibration_flag is False
    dialog._on_input_calibration_finished(True)
    assert dialog.input_calibration_flag is True
    dialog.reject()


def test_cancel_stops_active_recording_without_reporting_failure(qapp):
    widget = InputCalibration(DEVICE, [1])
    processor = _FakeProcessor()
    processor.is_recording = True
    widget.streaming_processor = processor
    emitted = []
    widget.calibration_finished.connect(emitted.append)

    widget.cancel_calibration()

    assert processor.stop_calls == 1
    assert widget.streaming_processor is None
    assert emitted == []
