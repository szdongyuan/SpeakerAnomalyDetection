"""Task6 native and shared capture tests; every acquisition uses an injected SDK."""
from dataclasses import FrozenInstanceError
import logging
import os
import re
import threading
import time

import numpy as np
import pytest
import soundfile as sf

from base.ve3668n_capture import Ve3668nInputStream
from base.ve3668n_capture_timing import VeCaptureDeadline
from base.recording_capture import RecordingCapture
from base.recording_process_protocol import RecordingResult, RecordingFailure
from base.wav_calibration_metadata import (
    WavCalibrationMetadataAppendResult, append_wav_calibration_metadata_result,
    inspect_wav_calibration_metadata,
)
from consts.recording_preview_consts import (
    MAIN_RECORDING_LIVE_WINDOW_SECONDS,
    PREVIEW_TIME_LOWER_BOUND_TOLERANCE,
    PREVIEW_TIME_MODE_CUMULATIVE,
    PREVIEW_TIME_MODE_RELATIVE_LATEST,
)
from unit_test.base.recording_process_fakes import (
    ControlledMetadataAppender, ControlledWriter, FakeStatus, MetadataFileFaults,
)
from unit_test.base.ve3668n_fakes import CaptureSDK, DiscoveryClock, capture_request, wav_metadata


@pytest.fixture(autouse=True)
def forbid_real_native_load(monkeypatch):
    from base.vkinging_sdk import VkDaqClient
    def forbidden(*args, **kwargs):
        pytest.fail("Task6 tests must never load the native SDK")
    monkeypatch.setattr(VkDaqClient, "_load", forbidden)


def make_stream(tmp_path, sdk=None, **request_options):
    sdk = sdk or CaptureSDK()
    clock = DiscoveryClock()
    stop = threading.Event()
    blocks, failures = [], []
    stream = Ve3668nInputStream(
        capture_request(tmp_path / "native.wav", **request_options),
        lambda data, frames, time_info, status: blocks.append(data.copy()),
        lambda stage, message: failures.append((stage, message)), stop, lambda: sdk, clock,
    )
    return stream, sdk, clock, blocks, failures


def finish_stream(stream):
    assert stream.done.wait(2)
    stream.close()


def cleanup_log_records(caplog):
    return [record for record in caplog.records
            if record.name == "base.ve3668n_capture"
            and record.getMessage().startswith("[VE cleanup]")]


def test_cleanup_diagnostic_success_is_ordered_flushed_and_deterministically_timed(
    tmp_path, monkeypatch, caplog,
):
    stream, sdk, clock, _, failures = make_stream(tmp_path)
    durations = {"stop_task": .125, "clear_task": .25, "close": .375}
    for operation, duration in durations.items():
        sdk.hooks[operation] = lambda *args, duration=duration, **kwargs: clock.advance(duration)
    printed = []
    monkeypatch.setattr("builtins.print",
                        lambda *args, **kwargs: printed.append((args, kwargs)))
    caplog.set_level(logging.INFO, logger="base.ve3668n_capture")

    assert stream.start()
    finish_stream(stream)

    expected = []
    for operation, elapsed in (("stop_task", .125), ("clear_task", .25),
                               ("close_sdk", .375)):
        prefix = f"[VE cleanup] task={stream._task}"
        expected.extend([
            f"{prefix} event=start operation={operation} elapsed_s=0.000000",
            f"{prefix} event=completed operation={operation} elapsed_s={elapsed:.6f}",
        ])
    assert [args[0] for args, _ in printed] == expected
    assert all(len(args) == 1 and kwargs == {"flush": True} for args, kwargs in printed)
    records = cleanup_log_records(caplog)
    assert [record.getMessage() for record in records] == expected
    assert [record.levelno for record in records] == [logging.INFO] * len(expected)
    assert not failures and stream.handles_released


@pytest.mark.parametrize("operation,sdk_operation,error_text", [
    ("stop_task", "stop_task", "injected stop\r\ncontinued"),
    ("clear_task", "clear_task", "injected clear"),
    ("close_sdk", "close", "injected close"),
])
def test_cleanup_diagnostic_failure_is_single_line_and_preserves_cleanup_semantics(
    tmp_path, monkeypatch, caplog, operation, sdk_operation, error_text,
):
    stream, sdk, clock, _, failures = make_stream(tmp_path)
    def fail_cleanup(*args, **kwargs):
        clock.advance(.625)
        raise RuntimeError(error_text)
    sdk.hooks[sdk_operation] = fail_cleanup
    printed = []
    monkeypatch.setattr("builtins.print",
                        lambda *args, **kwargs: printed.append((args, kwargs)))
    caplog.set_level(logging.INFO, logger="base.ve3668n_capture")

    assert stream.start()
    finish_stream(stream)

    sanitized = error_text.replace("\r", " ").replace("\n", " ")
    failed_line = (f"[VE cleanup] task={stream._task} event=failed operation={operation} "
                   f"elapsed_s=0.625000 error_type=RuntimeError error={sanitized}")
    output = [args[0] for args, _ in printed
              if args and isinstance(args[0], str) and args[0].startswith("[VE cleanup]")]
    assert failed_line in output
    assert all("\r" not in line and "\n" not in line for line in output)
    failed_records = [record for record in cleanup_log_records(caplog)
                      if record.getMessage() == failed_line]
    assert len(failed_records) == 1
    assert failed_records[0].levelno == logging.ERROR
    assert failed_records[0].exc_info is not None
    assert failed_records[0].exc_info[0] is RuntimeError
    assert [row["operation"] for row in sdk.trace][-3:] == ["stop_task", "clear_task", "close"]
    assert failures == [(operation, error_text)]
    assert not stream.handles_released


@pytest.mark.parametrize("sdk_operation,expected_stage", [
    ("stop_task", "stop_task"),
    ("clear_task", "clear_task"),
    ("close", "close_sdk"),
    ("read_task_data", "none"),
])
def test_hung_owner_cleanup_diagnostic_names_active_stage(
    tmp_path, sdk_operation, expected_stage,
):
    entered, release = threading.Event(), threading.Event()
    def block(*args, **kwargs):
        entered.set()
        assert release.wait(3)
    stream, _, _, _, failures = make_stream(
        tmp_path, CaptureSDK(hooks={sdk_operation: block}))
    stream._join_timeout = .03
    try:
        assert stream.start()
        assert entered.wait(1)
        stream.close()
        assert failures == [(
            "close_stream",
            f"VE native owner did not exit (cleanup_stage={expected_stage})",
        )]
        assert not stream.handles_released
    finally:
        release.set()
        finish_stream(stream)


def test_cleanup_diagnostic_task_creation_failure_emits_only_close_sdk(
    tmp_path, monkeypatch,
):
    stream, _, _, _, failures = make_stream(tmp_path, CaptureSDK(failures=["create_task"]))
    printed = []
    monkeypatch.setattr("builtins.print",
                        lambda *args, **kwargs: printed.append((args, kwargs)))

    assert not stream.start()
    finish_stream(stream)

    assert [args[0].split(" operation=")[1].split()[0] for args, _ in printed] == [
        "close_sdk", "close_sdk",
    ]
    assert [args[0].split(" event=")[1].split()[0] for args, _ in printed] == [
        "start", "completed",
    ]
    assert len(failures) == 1 and failures[0][0] == "create_task"


def test_cleanup_diagnostic_sdk_construction_failure_emits_nothing(
    tmp_path, monkeypatch, caplog,
):
    stream, _, _, _, failures = make_stream(tmp_path)
    stream._sdk_factory = lambda: (_ for _ in ()).throw(
        OSError("injected SDK construction failure"))
    printed = []
    monkeypatch.setattr("builtins.print",
                        lambda *args, **kwargs: printed.append((args, kwargs)))
    caplog.set_level(logging.INFO, logger="base.ve3668n_capture")

    assert not stream.start()
    finish_stream(stream)

    assert printed == []
    assert cleanup_log_records(caplog) == []
    assert failures == [("device", "injected SDK construction failure")]


@pytest.mark.parametrize("sink", ["formatter", "print", "logger"])
def test_cleanup_diagnostic_output_failure_is_secondary_and_does_not_change_cleanup(
    tmp_path, monkeypatch, sink,
):
    stream, sdk, _, _, failures = make_stream(tmp_path)
    if sink == "formatter":
        def fail_formatter(*args, **kwargs):
            raise RuntimeError("injected formatter failure")
        monkeypatch.setattr(stream, "_format_cleanup_diagnostic", fail_formatter, raising=False)
    elif sink == "print":
        def fail_print(*args, **kwargs):
            raise RuntimeError("injected print failure")
        monkeypatch.setattr("builtins.print", fail_print)
    else:
        class FailingLogger:
            def info(self, *args, **kwargs):
                raise RuntimeError("injected logger failure")
            def error(self, *args, **kwargs):
                raise RuntimeError("injected logger failure")
        monkeypatch.setattr(stream, "_logger", FailingLogger(), raising=False)

    assert stream.start()
    finish_stream(stream)

    cleanup_operations = [row["operation"] for row in sdk.trace][-3:]
    assert cleanup_operations == ["stop_task", "clear_task", "close"]
    output_failures = [message for message in stream.diagnostics
                       if message.startswith(f"diagnostic_output: {sink} failed:")]
    assert len(output_failures) == 6
    assert not failures and stream.handles_released


@pytest.mark.parametrize("rate", [44100, 48000, 51200])
def test_owner_lifecycle_is_one_thread_with_fresh_routes_and_frozen_rate(tmp_path, rate):
    stream, sdk, clock, blocks, failures = make_stream(tmp_path, sample_rate=rate)
    assert stream.start()
    finish_stream(stream)
    operations = [row["operation"] for row in sdk.trace]
    assert operations == ["get_devices", "get_device_attribute", "get_device_attribute",
                          "get_channels", "get_device_attribute", "create_task",
                          "create_iepe_voltage_channel", "configure_sample_clock", "start_task",
                          "verify_actual_sample_rate", "read_task_data", "stop_task", "clear_task", "close"]
    assert {row["pid"] for row in sdk.trace} == {os.getpid()}
    owner_ids = {row["thread_id"] for row in sdk.trace}
    assert len(owner_ids) == 1 and threading.get_ident() not in owner_ids
    assert sdk.trace[6]["args"][1] == "FreshDev/AIN8,FreshDev/AIN2"
    assert sdk.trace[7]["args"][1] == rate
    assert sdk.trace[9]["args"] == ("FreshDev", rate)
    task = sdk.trace[5]["args"][0]
    assert re.fullmatch(r"VE_[0-9a-f]{24}", task)
    assert len(task.encode("ascii")) == 27
    assert len(task.encode("utf-8")) <= 32
    assert all(row["args"][0] == task for row in sdk.trace if row["operation"] in
               {"create_task", "create_iepe_voltage_channel", "configure_sample_clock", "start_task",
                "read_task_data", "stop_task", "clear_task"})
    np.testing.assert_array_equal(blocks[0][:, (7, 1)], np.tile([8.25, 2.5], (9, 1)))
    assert not failures and stream.handles_released and sdk.closed
    assert stream.started.is_set() and stream.started_at == 100.0
    assert stream.progress_snapshot().frames == 9
    with pytest.raises((AttributeError, FrozenInstanceError)):
        stream.progress_snapshot().frames = 100
    with pytest.raises(AttributeError):
        stream.started_at = 0


@pytest.mark.parametrize("operation", ["get_devices", "create_task", "create_iepe_voltage_channel",
                                       "configure_sample_clock", "start_task", "verify_actual_sample_rate",
                                       "read_task_data", "stop_task", "clear_task", "close"])
def test_native_failure_is_diagnostic_and_only_owned_tasks_are_cleaned(tmp_path, operation):
    stream, sdk, _, _, failures = make_stream(tmp_path, CaptureSDK(failures=[operation]))
    stream.start()
    finish_stream(stream)
    assert len(failures) == 1 and operation in failures[0][1]
    operations = [row["operation"] for row in sdk.trace]
    created = operation not in ("get_devices", "create_task")
    assert ("clear_task" in operations) is created
    assert ("stop_task" in operations) is created
    assert stream.started.is_set() is (operation in ("read_task_data", "stop_task", "clear_task", "close"))
    assert stream.handles_released is (operation not in ("stop_task", "clear_task", "close"))


def test_first_error_wins_and_all_cleanup_diagnostics_survive(tmp_path):
    stream, sdk, _, _, failures = make_stream(tmp_path, CaptureSDK(
        failures=["read_task_data", "stop_task", "clear_task", "close"]))
    stream.start()
    finish_stream(stream)
    assert len(failures) == 1 and "read_task_data" in failures[0][1]
    assert all(any(operation in message for message in stream.diagnostics)
               for operation in ("read_task_data", "stop_task", "clear_task", "close"))
    assert not stream.handles_released


@pytest.mark.parametrize("phase", ["before", "get_devices", "create_task", "configure_sample_clock", "start_task"])
def test_cancel_never_starts_next_native_phase_and_cleans_only_owned_task(tmp_path, phase):
    stream, sdk, _, _, failures = make_stream(tmp_path)
    if phase == "before":
        stream.stop()
    else:
        sdk.hooks[phase] = lambda *args, **kwargs: stream.stop_event.set()
    assert not stream.start()
    finish_stream(stream)
    operations = [row["operation"] for row in sdk.trace]
    assert not stream.started.is_set() and not failures and stream.handles_released
    if phase == "before":
        assert not operations
    else:
        assert operations[-1] == "close"
        assert ("clear_task" in operations) is (phase not in ("get_devices",))
        assert "read_task_data" not in operations


def test_native_started_timestamp_precedes_slow_rate_verification_and_public_ready(tmp_path):
    stream, sdk, clock, _, failures = make_stream(tmp_path)
    observed = []
    def verify(*args):
        observed.append((stream.started_at, stream.started.is_set()))
        clock.advance(1)
    sdk.hooks["verify_actual_sample_rate"] = verify
    stream.start()
    finish_stream(stream)
    assert observed == [(100.0, False)]
    assert stream.started_at == 100.0 and not failures


def test_task_names_are_unique_even_for_identical_request_ids(tmp_path):
    names = []
    for _ in range(2):
        stream, sdk, _, _, _ = make_stream(tmp_path)
        stream.start()
        finish_stream(stream)
        names.append(next(row["args"][0] for row in sdk.trace if row["operation"] == "create_task"))
    assert names[0] != names[1]
    assert all(re.fullmatch(r"VE_[0-9a-f]{24}", name) for name in names)
    assert all(len(name.encode("utf-8")) <= 32 for name in names)


def start_capture(tmp_path, sdk=None, *, request_options=None, **dependencies):
    sdk = sdk or CaptureSDK()
    clock = DiscoveryClock()
    streams = []
    def factory(**kwargs):
        stream = Ve3668nInputStream(**kwargs, sdk_factory=lambda: sdk, clock=clock)
        streams.append(stream)
        return stream
    capture = RecordingCapture(capture_request(tmp_path / "capture.wav", **(request_options or {})),
                               ve_stream_factory=factory, **dependencies)
    capture.start()
    return capture, sdk, streams, clock


def start_persistent_capture(tmp_path, sdk=None, **dependencies):
    from base.ve3668n_resource import VeResourceController

    sdk = sdk or CaptureSDK()
    clock = DiscoveryClock()
    controller = VeResourceController(
        sdk_factory=lambda: sdk, clock=clock, bind_timeout=.5, detach_timeout=.2)
    capture = RecordingCapture(
        capture_request(tmp_path / "persistent.wav"),
        ve_stream_factory=controller.stream,
        **dependencies,
    )
    capture.start()
    return capture, controller, sdk, clock


def test_ve_capture_slot_releases_before_metadata_finalizer(tmp_path):
    finalizer = ControlledMetadataAppender()
    writer = ControlledWriter()
    capture, controller, sdk, clock = start_persistent_capture(
        tmp_path, writer_factory=writer, metadata_appender=finalizer)
    try:
        assert finalizer.entered.wait(3)
        assert capture.capture_slot_released.wait(1)
        slot = capture.capture_slot
        assert slot.target_reached_at == clock()
        assert slot.raw_frames == capture.request.target_samples == capture.raw_frames
        assert slot.adapter_released and slot.writer_released
        assert writer.closed and capture.queued_frames == 0
        assert not capture.done.is_set() and capture.outcome is None
        assert sdk.calls("stop_task") == sdk.calls("clear_task") == sdk.calls("close") == 0
    finally:
        finalizer.release.set()
    assert isinstance(capture.wait(3), RecordingResult)
    assert controller.release(.5).success


def test_ve_write_failure_with_confirmed_writer_release_opens_slot(tmp_path):
    writer = ControlledWriter(fail_at="write")
    capture, controller, _, _ = start_persistent_capture(tmp_path, writer_factory=writer)
    outcome = capture.wait(3)
    try:
        assert isinstance(outcome, RecordingFailure) and outcome.stage == "write"
        assert capture.capture_slot_released.is_set()
        assert capture.capture_slot.raw_frames == capture.request.target_samples
        assert capture.capture_slot.adapter_released
        assert capture.capture_slot.writer_released and writer.closed
    finally:
        assert controller.release(.5).success


def test_ve_writer_release_uncertainty_keeps_capture_slot_closed(tmp_path):
    writer = ControlledWriter(fail_at="close")
    capture, controller, _, _ = start_persistent_capture(tmp_path, writer_factory=writer)
    outcome = capture.wait(3)
    try:
        assert isinstance(outcome, RecordingFailure) and outcome.stage == "close_wav"
        assert not capture.capture_slot_released.is_set() and capture.capture_slot is None
        assert capture._writer is writer
        assert (capture.request.path, writer) in capture._unreleased_finalization_handles
        assert not outcome.handles_released
    finally:
        assert controller.release(.5).success


def test_ve_adapter_release_uncertainty_keeps_capture_slot_closed(tmp_path):
    from base.ve3668n_capture_timing import VeCaptureProgress

    adapters = []

    class UncertainAdapter:
        def __init__(self, *, request, callback, fail, stop_event):
            self.request = request
            self.callback = callback
            self.stop_event = stop_event
            self.stop_calls = 0
            self.close_calls = 0
            self._progress = VeCaptureProgress(50.0, request.target_samples, 51.0)
            adapters.append(self)

        def start(self):
            data = np.ones((self.request.target_samples, 8), dtype=np.float32)
            self.callback(data, len(data), None, FakeStatus())
            return True

        def stop(self):
            self.stop_calls += 1
            return False

        def close(self):
            self.close_calls += 1
            raise RuntimeError("uncertain adapter waited for detach a second time")

        @property
        def handles_released(self):
            return False

        @property
        def diagnostics(self):
            return ("detach confirmation timed out",)

        @property
        def started_at(self):
            return self._progress.started_at

        def progress_snapshot(self):
            return self._progress

    capture = RecordingCapture(
        capture_request(tmp_path / "uncertain.wav"), ve_stream_factory=UncertainAdapter)
    capture.start()
    outcome = capture.wait(3)
    assert isinstance(outcome, RecordingFailure) and outcome.stage == "close_stream"
    assert adapters[0].stop_calls == 1 and adapters[0].close_calls == 0
    assert not capture.capture_slot_released.is_set() and capture.capture_slot is None
    assert not outcome.handles_released


@pytest.mark.parametrize("rate", [44100, 48000, 51200])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("preview_time_mode", [
    PREVIEW_TIME_MODE_RELATIVE_LATEST,
    PREVIEW_TIME_MODE_CUMULATIVE,
])
def test_shared_capture_exact_trim_float_voltage_with_no_portaudio(
        tmp_path, monkeypatch, rate, streaming, preview_time_mode):
    from base import recording_capture
    def forbidden(*args, **kwargs):
        pytest.fail("VE must not use sounddevice or default device")
    monkeypatch.setattr(recording_capture, "sounddevice_backend", forbidden)
    monkeypatch.setattr(RecordingCapture, "_validate_device", forbidden)
    raw = rate // 10 + 17
    trim = rate // 100
    capture, sdk, streams, _ = start_capture(tmp_path, CaptureSDK(counts=[0, 2, 1]), request_options={
        "sample_rate": rate, "target_samples": raw, "trim_samples": trim, "streaming": streaming,
        "preview_time_mode": preview_time_mode,
    }, backend=object())
    outcome = capture.wait(3)
    assert isinstance(outcome, RecordingResult), outcome
    audio, actual = sf.read(outcome.path, dtype="float32", always_2d=True)
    assert (actual, outcome.raw_frames, outcome.final_frames) == (rate, raw, raw - trim)
    np.testing.assert_array_equal(audio, np.tile([8.25, 2.5], (raw - trim, 1)))
    assert sf.info(outcome.path).subtype == "FLOAT"
    assert outcome.metadata_appended and outcome.handles_released
    assert inspect_wav_calibration_metadata(outcome.path).metadata == capture.request.calibration_metadata.to_dict()
    assert capture.started.is_set() and streams[0].handles_released
    assert capture.started_at == 100.0
    assert capture.progress_snapshot().frames == raw  # still available after _stream is closed
    preview = capture.snapshot(generation=1, sequence=1)
    if streaming:
        assert preview.sample_stop == raw - trim
        assert preview.time_mode == preview_time_mode
        for waveform in preview.waveforms:
            if preview_time_mode == PREVIEW_TIME_MODE_RELATIVE_LATEST:
                assert waveform.time[-1] == 0.0
                assert waveform.time[0] >= (
                    -MAIN_RECORDING_LIVE_WINDOW_SECONDS - PREVIEW_TIME_LOWER_BOUND_TOLERANCE
                )
            else:
                assert waveform.time[0] == 0.0
                assert waveform.time[-1] > 0.0
                assert np.all(np.diff(waveform.time) > 0.0)
        assert max(preview.waveforms[0].amplitude) == 8.25
        assert max(preview.waveforms[1].amplitude) == 2.5
    else:
        assert preview is None


@pytest.mark.parametrize("phase", ["get_devices", "create_task", "start_task",
                                   "verify_actual_sample_rate", "read_task_data", "stop_task", "clear_task", "close"])
def test_blocked_native_boundary_has_bounded_wrappers_and_sticky_uncertainty(tmp_path, phase):
    entered, release = threading.Event(), threading.Event()
    def block(*args, **kwargs):
        entered.set()
        assert release.wait(3)
    stream, sdk, _, _, failures = make_stream(tmp_path, CaptureSDK(hooks={phase: block}))
    stream._ready_timeout = .03
    stream._join_timeout = .03
    try:
        before = time.monotonic()
        stream.start()
        assert entered.wait(1)
        stream.close()
        assert time.monotonic() - before < .7
        assert not stream.handles_released
        assert len(failures) == 1
        if phase in ("get_devices", "create_task", "start_task", "verify_actual_sample_rate"):
            assert not stream.started.is_set()
        if phase == "verify_actual_sample_rate":
            assert stream.started_at == 100.0
    finally:
        release.set()
        finish_stream(stream)
    assert not stream.handles_released  # later return never restores reusability


@pytest.mark.parametrize("actual", [None, True, False, 51200.0, 48000])
def test_injected_invalid_actual_rate_never_publishes_started(tmp_path, actual):
    stream, sdk, _, _, failures = make_stream(tmp_path)
    sdk.verify_actual_sample_rate = lambda *args: actual
    assert not stream.start()
    finish_stream(stream)
    assert failures and "sample rate" in failures[0][1]
    assert stream.started_at == 100.0 and not stream.started.is_set()


def test_compact_short_reads_zero_two_one_remaining_never_replay_tail(tmp_path):
    stream, sdk, _, blocks, failures = make_stream(tmp_path, CaptureSDK(counts=[0, 2, 1]))
    stream.start()
    finish_stream(stream)
    assert not failures
    assert [len(block) for block in blocks] == [2, 1, 6]
    np.testing.assert_array_equal(np.concatenate(blocks)[:, (7, 1)], np.tile([8.25, 2.5], (9, 1)))
    assert [row["kwargs"]["samples_per_channel"] for row in sdk.trace
            if row["operation"] == "read_task_data"] == [9, 9, 7, 6]
    assert all(row["kwargs"]["timeout_seconds"] <= .2 for row in sdk.trace
               if row["operation"] == "read_task_data")


@pytest.mark.parametrize("returned", [-1, -10028, 10, 2**31, True, None, 1.0])
def test_bad_read_count_fails_before_accessing_buffer(tmp_path, returned):
    stream, sdk, _, blocks, failures = make_stream(tmp_path)
    sdk.read_task_data = lambda *args, **kwargs: (object(), returned)
    stream.start()
    finish_stream(stream)
    assert not blocks and len(failures) == 1 and "count" in failures[0][1]
    assert stream.progress_snapshot().frames == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1e300])
def test_native_nonfinite_or_float32_overflow_cannot_be_delivered(tmp_path, value):
    sdk = CaptureSDK()
    sdk.values = (value, 2.5)
    stream, _, _, blocks, failures = make_stream(tmp_path, sdk)
    stream.start()
    finish_stream(stream)
    assert not blocks and len(failures) == 1 and "non-finite" in failures[0][1]
    assert stream.progress_snapshot().frames == 0


def test_wrong_capacity_stride_hits_unwritten_nan_and_fails(tmp_path):
    sdk = CaptureSDK(counts=[2])
    sdk.layout = "capacity"
    stream, _, _, blocks, failures = make_stream(tmp_path, sdk)
    stream.start()
    finish_stream(stream)
    assert not blocks and "non-finite" in failures[0][1]


def test_zero_read_never_accesses_buffer_and_uses_cancellable_wait(tmp_path):
    stream, sdk, _, blocks, failures = make_stream(tmp_path)
    waits = []
    original_wait = stream.stop_event.wait
    def wait(seconds):
        waits.append(seconds)
        stream.stop_event.set()
        return original_wait(seconds)
    stream.stop_event.wait = wait
    sdk.read_task_data = lambda *args, **kwargs: (object(), 0)
    stream.start()
    finish_stream(stream)
    assert not failures and not blocks and waits and 0 < waits[0] <= .05


def test_malformed_native_buffer_shape_is_not_audio(tmp_path):
    import ctypes
    stream, sdk, _, blocks, failures = make_stream(tmp_path)
    sdk.read_task_data = lambda *args, **kwargs: ((ctypes.c_double * 8)(*([1] * 8)), 9)
    stream.start()
    finish_stream(stream)
    assert not blocks and "buffer" in failures[0][1]


@pytest.mark.parametrize("duration", [1, 10, 60, 600])
@pytest.mark.parametrize("rate", [44100, 48000, 51200])
def test_deadline_formula_uses_raw_target_including_trim_and_remaining_bound(rate, duration):
    clock = DiscoveryClock()
    target = rate * duration + 7
    deadline = VeCaptureDeadline(rate, target, clock(), clock=clock)
    assert deadline.duration == target / rate
    assert deadline.capture_deadline == 100 + target / rate + max(5, target / rate * .1)
    assert deadline.block_frames == 2048
    assert deadline.requested_frames == 2048
    deadline.observe(target - 3)
    assert deadline.requested_frames == 3
    deadline.observe(target)
    assert deadline.requested_frames == 0 and deadline.complete
    clock.advance(10000)
    deadline.check()


def test_deadline_empty_duplicate_progress_does_not_refresh_stall_timer():
    clock = DiscoveryClock()
    deadline = VeCaptureDeadline(51200, 512000, clock(), clock=clock)
    clock.advance(4.999)
    deadline.observe(0)
    deadline.check()
    assert deadline.snapshot().last_frame_at == 100
    clock.advance(.001)
    with pytest.raises(TimeoutError, match="no progress"):
        deadline.check()
    deadline.observe(1)
    assert deadline.snapshot().last_frame_at == 105
    clock.advance(4)
    deadline.observe(1)
    deadline.check()
    clock.advance(1)
    with pytest.raises(TimeoutError, match="no progress"):
        deadline.check()


@pytest.mark.parametrize("frames", [True, -1, 10, 1.0])
def test_deadline_rejects_untrusted_counts(frames):
    clock = DiscoveryClock()
    deadline = VeCaptureDeadline(51200, 9, clock(), clock=clock)
    with pytest.raises(ValueError, match="frames"):
        deadline.observe(frames)


@pytest.mark.parametrize("rate", [44100, 48000, 51200])
@pytest.mark.parametrize("seconds,expected", [(4.999, None), (5, "no progress")])
def test_native_empty_read_recovers_only_before_five_seconds(tmp_path, seconds, expected, rate):
    stream, sdk, clock, blocks, failures = make_stream(tmp_path, CaptureSDK(counts=[0]), sample_rate=rate)
    advances = iter([seconds, 0])
    sdk.hooks["read_task_data"] = lambda *args, **kwargs: clock.advance(next(advances, 0))
    stream.start()
    finish_stream(stream)
    if expected is None:
        assert not failures and sum(map(len, blocks)) == 9
    else:
        assert not blocks and expected in failures[0][1]


@pytest.mark.parametrize("rate", [44100, 48000, 51200])
def test_native_sparse_positive_frames_cannot_extend_total_deadline(tmp_path, rate):
    stream, sdk, clock, blocks, failures = make_stream(tmp_path, CaptureSDK(counts=[1] * 9), sample_rate=rate)
    sdk.hooks["read_task_data"] = lambda *args, **kwargs: clock.advance(2)
    stream.start()
    finish_stream(stream)
    assert "total deadline" in failures[0][1]
    assert sum(map(len, blocks)) == 2
    assert stream.progress_snapshot().last_frame_at == 104


def test_rate_verification_delay_cannot_reset_native_capture_deadline(tmp_path):
    stream, sdk, clock, blocks, failures = make_stream(tmp_path)
    sdk.hooks["verify_actual_sample_rate"] = lambda *args: clock.advance(5)
    assert not stream.start()
    finish_stream(stream)
    assert stream.started_at == 100
    assert not blocks and "no progress" in failures[0][1]


@pytest.mark.parametrize("mode", ["false", "missing", "throw", "unreleased", "retained", "lie", "tamper",
                                  "gain", "frames", "rate", "channels", "subtype"])
def test_main_metadata_is_required_and_cannot_damage_raw_audio(tmp_path, mode):
    def appender(path, metadata, **kwargs):
        if mode == "false":
            return False
        if mode == "missing":
            return None
        if mode == "throw":
            raise OSError("injected metadata failure")
        if mode == "unreleased":
            return WavCalibrationMetadataAppendResult(True, False, close_errors=("unreleased metadata",))
        if mode == "retained":
            append_wav_calibration_metadata_result(path, metadata, **kwargs)
            return WavCalibrationMetadataAppendResult(True, True, retained_handles=((path, object()),))
        if mode == "lie":
            return True
        if mode == "tamper":
            metadata["acquisition"]["machine_id"] = "different-frozen-snapshot"
        else:
            audio, rate = sf.read(path, dtype="float32", always_2d=True)
            if mode == "gain":
                audio *= .1
            elif mode == "frames":
                audio = audio[:-1]
            elif mode == "rate":
                rate = 48000
            elif mode == "channels":
                audio = audio[:, :1]
            sf.write(path, audio, rate, subtype="PCM_16" if mode == "subtype" else "FLOAT")
        return append_wav_calibration_metadata_result(path, metadata, **kwargs)
    capture, _, _, _ = start_capture(tmp_path, metadata_appender=appender)
    outcome = capture.wait(3)
    assert isinstance(outcome, RecordingFailure), outcome
    assert outcome.stage == "metadata"
    assert outcome.handles_released is (mode not in ("unreleased", "retained"))


def test_quality_gate_uses_only_temporary_tenth_scale_and_leaves_raw_voltage(tmp_path, monkeypatch):
    from base import recording_capture
    inspected = []
    def validate(audio, thresholds):
        inspected.append(audio.copy())
        return True, "", ""
    monkeypatch.setattr(recording_capture, "validate_recorded_audio", validate)
    capture, _, _, _ = start_capture(tmp_path, request_options={"validation_thresholds": {"enabled": True}})
    outcome = capture.wait(3)
    assert isinstance(outcome, RecordingResult)
    expected = np.tile(np.array([8.25, 2.5], dtype=np.float32), (7, 1))
    np.testing.assert_array_equal(inspected[0], expected / 10)
    np.testing.assert_array_equal(sf.read(outcome.path, dtype="float32")[0], expected)


@pytest.mark.parametrize("purpose", ["main", "calibration"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1e300])
def test_shared_callback_rejects_nonfinite_even_with_quality_disabled(tmp_path, purpose, value):
    options = {} if purpose == "main" else dict(purpose=purpose, channels=(7,), target_samples=512000,
                                                trim_samples=0, calibration_metadata=None)
    capture = RecordingCapture(capture_request(tmp_path / "bad.wav", **options))
    data = np.ones((1, 8), dtype=np.float64)
    data[0, 7] = value
    capture._input_callback(data, 1, None, None)
    assert capture._failure is not None and "non-finite" in capture._failure[1]
    assert capture.raw_frames == 0 and capture._wake.is_set()


def test_shared_overflow_diagnostic_uses_ve_identity_not_portaudio_index(tmp_path):
    capture = RecordingCapture(capture_request(tmp_path / "bad.wav"))
    capture._input_callback(np.ones((1, 8), dtype=np.float32), 1, None, FakeStatus(input_overflow=True))
    assert "overflow" in capture._failure[1] and "test-machine-1" in capture._failure[1]


@pytest.mark.parametrize("purpose", ["main", "calibration"])
def test_readback_nonfinite_fails_even_without_product_quality_gate(tmp_path, purpose):
    writer = ControlledWriter()
    original_finalize = writer.finalize
    def corrupt():
        original_finalize()
        path = writer.writer.file_path
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        audio[0, 0] = np.nan
        sf.write(path, audio, rate, subtype="FLOAT")
    writer.finalize = corrupt
    options = {"trim_samples": 0}
    if purpose == "calibration":
        options.update(purpose=purpose, channels=(7,), target_samples=512000, calibration_metadata=None)
    capture, _, _, _ = start_capture(tmp_path, request_options=options, writer_factory=writer, queue_seconds=11)
    outcome = capture.wait(4)
    assert isinstance(outcome, RecordingFailure), outcome
    assert "non-finite" in outcome.message


def test_queue_overflow_wakes_capture_and_never_publishes_success(tmp_path):
    writer = ControlledWriter(pause=True)
    sdk = CaptureSDK()
    reads = []
    def next_read(*args, **kwargs):
        if reads:
            assert writer.entered.wait(2)
        reads.append(1)
    sdk.hooks["read_task_data"] = next_read
    capture, _, _, _ = start_capture(tmp_path, sdk, request_options={"target_samples": 10000},
                                    writer_factory=writer, queue_seconds=.01)
    try:
        assert writer.entered.wait(2)
        assert capture._stop_requested.wait(2)
    finally:
        writer.release.set()
    outcome = capture.wait(3)
    assert isinstance(outcome, RecordingFailure) and "queue" in outcome.message
    assert outcome.raw_frames == outcome.written_frames == 4096
    assert outcome.handles_released


@pytest.mark.parametrize("fail_at", ["write", "close"])
def test_ve_writer_failure_cannot_become_success(tmp_path, fail_at):
    capture, _, _, _ = start_capture(tmp_path, writer_factory=ControlledWriter(fail_at=fail_at))
    outcome = capture.wait(3)
    assert isinstance(outcome, RecordingFailure) and "injected" in outcome.message
    assert outcome.handles_released is (fail_at == "write")


@pytest.mark.parametrize("stage", ["source", "temporary", "validation"])
@pytest.mark.parametrize("close_fails", [True, False])
def test_ve_actual_metadata_file_faults_are_failures_with_owned_handles(tmp_path, monkeypatch, stage, close_fails):
    faults = MetadataFileFaults(stage, close_fails=close_fails)
    faults.install(monkeypatch)
    try:
        capture, _, _, _ = start_capture(tmp_path)
        outcome = capture.wait(3)
        assert isinstance(outcome, RecordingFailure), outcome
        assert outcome.handles_released is not close_fails
        assert outcome.stage == "metadata"
        if close_fails:
            assert capture._unreleased_finalization_handles
            assert outcome.cleanup_paths == tuple(faults.temporary_paths)
    finally:
        faults.release_all()


@pytest.mark.parametrize("operation", ["get_devices", "create_task", "create_iepe_voltage_channel",
                                       "configure_sample_clock", "start_task", "verify_actual_sample_rate",
                                       "read_task_data", "stop_task", "clear_task", "close"])
def test_shared_capture_native_failure_wakes_owner_and_preserves_release_state(tmp_path, operation):
    capture, _, _, _ = start_capture(tmp_path, CaptureSDK(failures=[operation]))
    outcome = capture.wait(3)
    assert isinstance(outcome, RecordingFailure) and operation in outcome.message
    assert outcome.handles_released is (operation not in ("stop_task", "clear_task", "close"))
    assert capture.started.is_set() is (operation in ("read_task_data", "stop_task", "clear_task", "close"))


def test_read_failure_beats_cancel_and_cleanup_errors_in_shared_capture(tmp_path):
    sdk = CaptureSDK(failures=["read_task_data", "stop_task", "clear_task", "close"])
    capture = RecordingCapture(capture_request(tmp_path / "v.wav"), ve_stream_factory=lambda **kwargs:
                               Ve3668nInputStream(**kwargs, sdk_factory=lambda: sdk, clock=DiscoveryClock()))
    sdk.hooks["read_task_data"] = lambda *args, **kwargs: capture.cancel()
    capture.start()
    outcome = capture.wait(3)
    assert isinstance(outcome, RecordingFailure) and "read_task_data" in outcome.message
    assert not outcome.handles_released
    assert len(capture.native_diagnostics) == 4


@pytest.mark.parametrize("phase", ["read_task_data", "stop_task", "clear_task", "close"])
def test_shared_capture_hung_owner_never_reports_released_or_success(tmp_path, phase):
    entered, release = threading.Event(), threading.Event()
    def block(*args, **kwargs):
        entered.set()
        assert release.wait(4)
    capture, _, streams, _ = start_capture(tmp_path, CaptureSDK(hooks={phase: block}))
    try:
        assert entered.wait(2)
        capture.cancel()
        outcome = capture.wait(2)
        assert isinstance(outcome, RecordingFailure) and not outcome.handles_released
        assert capture.progress_snapshot().frames == (0 if phase == "read_task_data" else 9)
    finally:
        release.set()
        finish_stream(streams[0])


@pytest.mark.parametrize("purpose", ["main", "calibration"])
@pytest.mark.parametrize("shape,frames", [((1, 2), 1), ((1, 8), 2), ((1, 8), True),
                                         ((0, 8), 0), ((2,), 2)])
def test_shared_callback_structural_checks_are_unconditional(tmp_path, purpose, shape, frames):
    options = {} if purpose == "main" else dict(purpose=purpose, channels=(7,), target_samples=512000,
                                                trim_samples=0, calibration_metadata=None)
    capture = RecordingCapture(capture_request(tmp_path / "bad.wav", **options))
    capture._input_callback(np.ones(shape, dtype=np.float32), frames, None, None)
    assert capture._failure is not None and "shape" in capture._failure[1]
    assert capture.raw_frames == 0


@pytest.mark.parametrize("rate", [44100, 48000, 51200])
def test_calibration_captures_exact_ten_seconds_raw_voltage_without_metadata(tmp_path, rate):
    def forbidden(*args, **kwargs):
        pytest.fail("calibration must not append a main recording snapshot")
    capture, _, _, _ = start_capture(tmp_path, request_options=dict(
        purpose="calibration", sample_rate=rate, channels=(7,), target_samples=rate * 10,
        trim_samples=0, streaming=True, calibration_metadata=None,
        validation_thresholds={"enabled": True}), metadata_appender=forbidden, queue_seconds=11)
    outcome = capture.wait(4)
    assert isinstance(outcome, RecordingResult), outcome
    assert outcome.raw_frames == outcome.final_frames == rate * 10
    audio, actual = sf.read(outcome.path, dtype="float32")
    assert actual == rate and len(audio) == rate * 10
    np.testing.assert_array_equal(audio, np.full(rate * 10, 8.25, dtype=np.float32))
    assert sf.info(outcome.path).subtype == "FLOAT"
    assert not outcome.metadata_appended and capture.snapshot(generation=1, sequence=1) is None


def test_compact_short_read_ignores_old_finite_tail(tmp_path):
    sdk = CaptureSDK(counts=[2, 1])
    read = sdk.read_task_data
    def dirty_tail(*args, **kwargs):
        buffer, count = read(*args, **kwargs)
        for index in range(count * kwargs["channel_count"], len(buffer)):
            buffer[index] = -999
        return buffer, count
    sdk.read_task_data = dirty_tail
    stream, _, _, blocks, failures = make_stream(tmp_path, sdk)
    stream.start()
    finish_stream(stream)
    assert not failures
    np.testing.assert_array_equal(np.concatenate(blocks)[:, (7, 1)], np.tile([8.25, 2.5], (9, 1)))


@pytest.mark.parametrize("target", ["initial_reader", "trim_writer", "metadata_reader"])
@pytest.mark.parametrize("processing_fails", [False, True])
def test_ve_finalization_retains_failed_close_without_masking_primary_error(
    tmp_path, monkeypatch, target, processing_fails,
):
    real_soundfile = sf.SoundFile
    opened = []
    read_count = 0
    class Boundary:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.close_attempts = 0
        def __getattr__(self, name):
            return getattr(self.wrapped, name)
        def __len__(self):
            return len(self.wrapped)
        def read(self, *args, **kwargs):
            if processing_fails:
                raise OSError("first processing failure")
            return self.wrapped.read(*args, **kwargs)
        def write(self, *args, **kwargs):
            if processing_fails:
                raise OSError("first processing failure")
            return self.wrapped.write(*args, **kwargs)
        def close(self):
            self.close_attempts += 1
            raise OSError("secondary close failure")
    def open_file(path, *args, **kwargs):
        nonlocal read_count
        wrapped = real_soundfile(path, *args, **kwargs)
        mode = kwargs.get("mode", args[0] if args else "r")
        if mode == "r":
            read_count += 1
            current = "initial_reader" if read_count == 1 else "metadata_reader"
        else:
            current = "trim_writer" if os.path.basename(str(path)).startswith(".recording-trim-") else "writer"
        if current == target:
            wrapped = Boundary(wrapped)
            opened.append(wrapped)
        return wrapped
    monkeypatch.setattr(sf, "SoundFile", open_file)
    try:
        capture, _, _, _ = start_capture(tmp_path)
        outcome = capture.wait(3)
        assert isinstance(outcome, RecordingFailure) and not outcome.handles_released
        assert "first processing failure" in outcome.message if processing_fails else "close failure" in outcome.message
        assert len(opened) == 1 and opened[0].close_attempts == 1
        assert capture._unreleased_finalization_handles
        assert bool(outcome.cleanup_paths) is (target == "trim_writer")
    finally:
        for boundary in opened:
            boundary.wrapped.close()


def test_measured_snapshot_never_applies_pa_gain_to_capture_or_preview(tmp_path):
    metadata = wav_metadata(sample_rate=44100)
    metadata["acquisition"]["machine_id"] = "test-machine-1"
    capture, _, _, _ = start_capture(tmp_path, request_options=dict(
        sample_rate=44100, streaming=True, calibration_metadata=metadata))
    outcome = capture.wait(3)
    assert isinstance(outcome, RecordingResult), outcome
    np.testing.assert_array_equal(sf.read(outcome.path, dtype="float32")[0], np.tile([8.25, 2.5], (7, 1)))
    preview = capture.snapshot(generation=1, sequence=1)
    assert preview.time_mode == PREVIEW_TIME_MODE_RELATIVE_LATEST
    assert preview.waveforms[0].time[-1] == 0.0
    assert preview.waveforms[0].time[0] >= -MAIN_RECORDING_LIVE_WINDOW_SECONDS
    assert max(preview.waveforms[0].amplitude) == 8.25
    assert inspect_wav_calibration_metadata(outcome.path).metadata == metadata
    assert metadata["recorded_channels"][0]["calibration"]["sample_rate"] == 51200


def test_sdk_factory_itself_runs_on_the_owner_thread_and_failure_opens_no_task(tmp_path):
    stream, sdk, _, _, failures = make_stream(tmp_path)
    calls = []
    def factory():
        calls.append((os.getpid(), threading.get_ident()))
        raise OSError("injected SDK construction failure")
    stream._sdk_factory = factory
    assert not stream.start()
    finish_stream(stream)
    assert calls[0][0] == os.getpid() and calls[0][1] != threading.get_ident()
    assert not sdk.trace and stream.handles_released
    assert "construction failure" in failures[0][1]


@pytest.mark.parametrize("phase", ["get_channels", "create_iepe_voltage_channel", "verify_actual_sample_rate", "read_task_data"])
def test_cancel_at_additional_native_boundaries_drops_returned_data(tmp_path, phase):
    stream, sdk, _, blocks, failures = make_stream(tmp_path)
    sdk.hooks[phase] = lambda *args, **kwargs: stream.stop_event.set()
    stream.start()
    finish_stream(stream)
    assert not failures and not blocks and stream.handles_released
    assert stream.progress_snapshot().frames == 0


@pytest.mark.parametrize("field", ["ready_timeout", "join_timeout"])
@pytest.mark.parametrize("value", [None, True, 0, -1, float("inf"), float("nan")])
def test_native_control_waits_cannot_be_configured_unbounded(tmp_path, field, value):
    with pytest.raises(ValueError, match=field):
        Ve3668nInputStream(capture_request(tmp_path / "v.wav"), lambda *args: None,
                          lambda *args: None, threading.Event(), lambda: CaptureSDK(),
                          DiscoveryClock(), **{field: value})


def test_mandatory_metadata_reader_close_failure_transfers_open_handle(tmp_path, monkeypatch):
    from unit_test.base.ve3668n_fakes import MetadataOwnershipFaults
    faults = MetadataOwnershipFaults(tmp_path / "capture.wav", readback_open=2,
                                     close_failures=("readback",))
    faults.install(monkeypatch)
    try:
        capture, _, _, _ = start_capture(tmp_path)
        outcome = capture.wait(3)
        readers = [item for item in faults.files if item.stage == "readback"]
        assert len(readers) == 1 and readers[0].close_attempts == 1
        assert not readers[0].wrapped.closed
        assert isinstance(outcome, RecordingFailure) and outcome.stage == "metadata"
        assert not outcome.handles_released
        assert (outcome.path, readers[0]) in capture._unreleased_finalization_handles
        assert "SECONDARY-READBACK-CLOSE-ERROR" in outcome.message
        assert outcome.cleanup_paths == ()
    finally:
        faults.release_all()


@pytest.mark.parametrize("processing,close_stages,first", [
    ("temporary", (), "PRIMARY-WRITE-ERROR"),
    ("temporary", ("temporary",), "PRIMARY-WRITE-ERROR"),
    ("temporary", ("source", "temporary"), "PRIMARY-WRITE-ERROR"),
    ("validation", ("validation",), "PRIMARY-VALIDATION-READ-ERROR"),
    ("readback", (), "PRIMARY-READBACK-READ-ERROR"),
    ("readback", ("readback",), "PRIMARY-READBACK-READ-ERROR"),
    (None, ("temporary", "source"), "SECONDARY-TEMPORARY-CLOSE-ERROR"),
])
def test_metadata_primary_fault_reaches_capture_with_separate_cleanup_diagnostics(
    tmp_path, monkeypatch, caplog, processing, close_stages, first,
):
    from unit_test.base.ve3668n_fakes import MetadataOwnershipFaults
    faults = MetadataOwnershipFaults(tmp_path / "capture.wav", readback_open=2,
                                     processing_failure=processing, close_failures=close_stages,
                                     read_after_backend=processing == "readback")
    faults.install(monkeypatch)
    try:
        capture, _, _, _ = start_capture(tmp_path)
        outcome = capture.wait(3)
        assert isinstance(outcome, RecordingFailure) and outcome.stage == "metadata"
        assert outcome.handles_released is (not close_stages)
        assert first in outcome.message
        if processing is not None:
            assert "SECONDARY-" not in outcome.message
        for stage in close_stages:
            boundary = next(item for item in faults.files if item.stage == stage)
            assert not boundary.wrapped.closed
            assert (boundary.name, boundary) in capture._unreleased_finalization_handles
            assert any(f"SECONDARY-{stage.upper()}-CLOSE-ERROR" in item for item in capture._warnings)
            assert f"SECONDARY-{stage.upper()}-CLOSE-ERROR" in caplog.text
        assert all(item.close_attempts == 1 for item in faults.files)
    finally:
        faults.release_all()


@pytest.mark.parametrize("close_fails", [False, True])
@pytest.mark.parametrize("error_type", [OSError, ValueError, RuntimeError, MemoryError])
def test_legacy_metadata_write_fault_keeps_warning_only_when_handles_release(
    tmp_path, monkeypatch, caplog, close_fails, error_type,
):
    from base.recording_process_protocol import RecordingRequest
    from unit_test.base.recording_process_fakes import FakeBackend, device_info, known_audio
    from unit_test.base.ve3668n_fakes import MetadataOwnershipFaults
    path = tmp_path / "legacy.wav"
    metadata = {"recorded_channels": [
        {"wav_channel_index": 0, "physical_input_channel": 0, "calibrated": False},
        {"wav_channel_index": 1, "physical_input_channel": 2, "calibrated": False},
    ]}
    faults = MetadataOwnershipFaults(path, processing_failure="temporary",
                                     close_failures=("temporary",) if close_fails else (),
                                     processing_error_type=error_type)
    faults.install(monkeypatch)
    backend = FakeBackend()
    capture = RecordingCapture(RecordingRequest(
        request_id="legacy-metadata", purpose="main", sample_rate=100, target_samples=9,
        channels=(0, 2), device=device_info(), path=str(path), streaming=False,
        trim_samples=0, monitor={}, calibration_metadata=metadata,
        validation_thresholds={"enabled": False}), backend=backend)
    try:
        capture.start()
        assert capture.started.wait(3)
        backend.stream.feed(known_audio())
        outcome = capture.wait(3)
        assert isinstance(outcome, RecordingFailure if close_fails else RecordingResult)
        assert outcome.handles_released is not close_fails
        assert "PRIMARY-WRITE-ERROR" in caplog.text
        if close_fails:
            assert outcome.stage == "metadata" and "PRIMARY-WRITE-ERROR" in outcome.message
            assert "SECONDARY-" not in outcome.message
            assert capture._unreleased_finalization_handles
        else:
            assert not outcome.metadata_appended
            assert "WAV calibration metadata was not appended" in outcome.warnings
            assert not capture._unreleased_finalization_handles
        np.testing.assert_array_equal(sf.read(path, dtype="float32")[0], known_audio()[:9, [0, 2]])
    finally:
        capture.cancel()
        capture.wait(3)
        faults.release_all()


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, MemoryError])
@pytest.mark.parametrize("stage", ["source", "temporary", "validation", "readback"])
@pytest.mark.parametrize("close_fails", [False, True])
def test_metadata_ordinary_exception_cannot_bypass_capture_ownership_handoff(
    tmp_path, monkeypatch, caplog, error_type, stage, close_fails,
):
    from unit_test.base.ve3668n_fakes import MetadataOwnershipFaults
    close_order = (("temporary", "source") if stage == "temporary" else (stage,)) if close_fails else ()
    faults = MetadataOwnershipFaults(tmp_path / "capture.wav", readback_open=2,
                                     processing_failure=stage, close_failures=close_order,
                                     processing_error_type=error_type)
    faults.install(monkeypatch)
    try:
        capture, _, _, _ = start_capture(tmp_path)
        outcome = capture.wait(3)
        assert isinstance(outcome, RecordingFailure) and outcome.stage == "metadata"
        first = "PRIMARY-WRITE-ERROR" if stage == "temporary" else f"PRIMARY-{stage.upper()}-READ-ERROR"
        assert outcome.message == first
        expected = tuple(next(item for item in faults.files if item.stage == name) for name in close_order)
        assert all(not item.wrapped.closed for item in expected)
        assert outcome.handles_released is not close_fails
        assert capture._unreleased_finalization_handles == [(item.name, item) for item in expected]
        for detail, name in zip(capture._warnings, close_order):
            assert f"SECONDARY-{name.upper()}-CLOSE-ERROR" in detail
            assert detail in caplog.text
        assert len(capture._warnings) == len(expected)
        assert all(item.close_attempts == 1 for item in faults.files)
        # Successful append already replaced its temporary file before readback.
        pending_temporaries = tuple(faults.temporary_paths) if close_fails and stage != "readback" else ()
        assert outcome.cleanup_paths == pending_temporaries
    finally:
        faults.release_all()


@pytest.mark.parametrize("streaming", [False, True])
def test_uncalibrated_main_capture_preserves_voltage_and_cumulative_preview(tmp_path, streaming):
    request = capture_request(tmp_path / "uncalibrated.wav", streaming=streaming, preview_time_mode=PREVIEW_TIME_MODE_CUMULATIVE)
    sdk = CaptureSDK()
    def stream_factory(**kwargs):
        return Ve3668nInputStream(**kwargs, sdk_factory=lambda: sdk)
    capture = RecordingCapture(request, ve_stream_factory=stream_factory)
    capture.start()
    outcome = capture.wait(3)
    assert isinstance(outcome, RecordingResult), outcome
    assert outcome.handles_released and outcome.metadata_appended
    assert inspect_wav_calibration_metadata(outcome.path).metadata == request.calibration_metadata.to_dict()
    assert outcome.raw_frames == request.target_samples
    assert outcome.final_frames == request.target_samples - request.trim_samples
    audio, rate = sf.read(request.path, dtype="float32", always_2d=True)
    assert rate == request.sample_rate
    np.testing.assert_array_equal(audio, np.tile(np.array(sdk.values, dtype=np.float32),
                                                (outcome.final_frames, 1)))
    preview = capture.snapshot(generation=1, sequence=1)
    if streaming:
        assert preview is not None
        assert preview.sample_stop == outcome.final_frames
        assert preview.channels == request.channels
        for column, waveform in enumerate(preview.waveforms):
            assert len(waveform.time) > 0
            assert waveform.time[0] >= 0 and np.all(np.diff(waveform.time) > 0)
            np.testing.assert_array_equal(waveform.amplitude,
                                          np.full_like(waveform.amplitude, sdk.values[column]))
    else:
        assert preview is None
    assert [row["operation"] for row in sdk.trace][-3:] == ["stop_task", "clear_task", "close"]
