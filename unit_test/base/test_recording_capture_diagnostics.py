"""Evidence for real child operations, including operations still blocked."""
import json
import logging
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from base.log_manager import LogManager
from base.recording_capture import RecordingCapture
from base.recording_diagnostics import RecordingDiagnostics
from base.recording_process_protocol import RecordingFailure, RecordingResult
from base.streaming_file_writer import StreamingWavWriter
from base.wav_pcm24 import quantize_pcm24
from unit_test.base.test_ve3668n_capture import start_persistent_capture
from unit_test.base.ve3668n_fakes import capture_request


CATEGORIES = ("consume", "write", "waveform_lock_wait", "waveform_lock_hold",
              "snapshot", "close_stream", "close_wav", "final_drain")


def diagnostics(monkeypatch, **kwargs):
    records = []
    logger = logging.Logger("capture-diagnostics", logging.INFO)

    class Handler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger.addHandler(Handler())
    monkeypatch.setattr(LogManager, "request_flush", lambda: None)
    return RecordingDiagnostics(logger, categories=CATEGORIES, generation=7, **kwargs), records


def events(records, stage):
    return [json.loads(message.split(" details=", 1)[1].split(" summary=", 1)[0])
            for message in list(records) if f" stage={stage} " in message]


@pytest.mark.parametrize("fail", [False, True])
def test_finalize_blocked_evidence_and_once_only_release_proof(tmp_path, monkeypatch, fail):
    diag, records = diagnostics(monkeypatch)
    entered, release = threading.Event(), threading.Event()

    class Writer(StreamingWavWriter):
        def finalize(self):
            entered.set()
            assert release.wait(3)
            super().finalize()
            if fail:
                raise OSError("finalize failed")

    capture, controller, _, _ = start_persistent_capture(
        tmp_path, writer_factory=Writer, diagnostics=diag)
    try:
        assert entered.wait(2)
        before = time.perf_counter_ns()
        assert events(records, "finalizing")
        assert len(events(records, "close_stream_begin")) == 1
        assert events(records, "close_stream_end")[0]["adapter_released"] is True
        assert len(events(records, "final_drain_begin")) == 1
        assert len(events(records, "final_drain_end")) == 1
        assert len(events(records, "close_wav_begin")) == 1
        assert not events(records, "close_wav_end")
        assert not capture.capture_slot_released.is_set()
        time.sleep(.11)
        diag.sample_once()
        active, = events(records, "close_wav")
        assert active["status"] == "in_progress"
        assert active["start_ns"] < before
        assert active["elapsed_ns"] >= 100_000_000
    finally:
        release.set()
        outcome = capture.wait(3)
        assert capture.join(3)
        assert controller.release(.5).success
    capture._close_writer()
    capture._close_stream()
    capture._publish_capture_slot()
    assert len(events(records, "close_wav_begin")) == 1
    end, = events(records, "close_wav_end")
    assert end["writer_released"] is (not fail)
    assert end["status"] == ("error" if fail else "success")
    assert isinstance(outcome, RecordingFailure if fail else RecordingResult)
    published = events(records, "capture_slot_published")
    assert len(published) == (0 if fail else 1)
    if not fail:
        assert published[0]["target_reached_at"] == capture.capture_slot.target_reached_at
        assert published[0]["raw_frames"] == capture.request.target_samples
        assert published[0]["queued_frames"] == 0
        assert published[0]["writer_released"] and published[0]["adapter_released"]


def test_snapshot_and_consumer_lock_wait_report_before_lock_released(tmp_path, monkeypatch):
    diag, records = diagnostics(monkeypatch)
    capture = RecordingCapture(capture_request(tmp_path / "lock.wav", streaming=True), diagnostics=diag)
    entered, release = threading.Event(), threading.Event()
    snapshot_thread = None
    consumer = None

    class Waveforms:
        def snapshots(self):
            entered.set()
            assert release.wait(3)
            return {}  # The existing optional-preview boundary handles this fault.

        def append(self, block):
            pass

    capture._waveforms = Waveforms()
    capture._writer = SimpleNamespace(write_chunk=quantize_pcm24)
    capture._diagnostic_blocks = 1  # This next block must not be a detailed sample.
    diag.start_sampler()
    try:
        snapshot_thread = threading.Thread(target=lambda: capture.snapshot(generation=7, sequence=1))
        snapshot_thread.start()
        assert entered.wait(1)
        consumer = threading.Thread(target=lambda: capture._consume(np.ones((4, 2), dtype=np.float32)))
        consumer.start()
        deadline = time.monotonic() + 2
        while not events(records, "waveform_lock_wait") and time.monotonic() < deadline:
            time.sleep(.01)
        assert snapshot_thread.is_alive() and consumer.is_alive()
        wait, = events(records, "waveform_lock_wait")
        hold, = events(records, "waveform_lock_hold")
        assert wait["status"] == hold["status"] == "in_progress"
        assert hold["start_ns"] <= wait["start_ns"]
        assert not diag.snapshot()["categories"]["snapshot"]["count"]
        assert not diag.snapshot()["categories"]["consume"]["count"]
    finally:
        release.set()
        if snapshot_thread:
            snapshot_thread.join(2)
        if consumer:
            consumer.join(2)
        diag.close(timeout=.2)
    assert not diag.sampler_thread.is_alive()
    assert diag.snapshot()["categories"]["snapshot"]["count"] == 1
    assert diag.snapshot()["categories"]["consume"]["count"] == 1


def test_tail_summary_is_request_local_and_callback_does_not_log(tmp_path, monkeypatch):
    diag, records = diagnostics(monkeypatch)
    diag.observe("consume", 3_000_000_000, request="old-request", started_ns=10)
    capture = RecordingCapture(capture_request(tmp_path / "tail.wav"), diagnostics=diag)
    capture._thread = threading.current_thread()
    capture._writer = SimpleNamespace(write_chunk=quantize_pcm24)
    capture._input_callback(np.ones((9, 8), dtype=np.float32), 9, None, None)
    assert records == []
    capture._tail_started_ns = time.perf_counter_ns()
    capture._consume(capture._pop_block())
    capture._diagnostic_tail_summary()
    tail, = events(records, "capture_tail")
    assert tail["scope"] == "request_tail"
    assert tail["consume_count"] == 1
    assert tail["consume_max_ns"] < 3_000_000_000
    assert tail["consume_max_start_ns"] >= capture._tail_started_ns
    assert diag.snapshot()["categories"]["consume"]["max_request"] == "old-request"


def test_cancel_records_close_proof_without_publishing_incomplete_capture(tmp_path, monkeypatch):
    from base.recording_process_protocol import RecordingCancelled

    diag, records = diagnostics(monkeypatch)
    capture = RecordingCapture(capture_request(tmp_path / "cancel.wav"), diagnostics=diag)
    capture.cancel()
    capture.start()
    assert isinstance(capture.wait(2), RecordingCancelled)
    assert capture.join(2)
    assert len(events(records, "finalizing")) == 1
    assert not events(records, "close_wav_begin")
    assert not events(records, "capture_slot_published")


def test_normal_blocks_count_consumes_but_label_write_detail_as_sampled(tmp_path, monkeypatch):
    diag, records = diagnostics(monkeypatch)
    capture = RecordingCapture(capture_request(tmp_path / "sampled.wav"), diagnostics=diag)
    capture._writer = SimpleNamespace(write_chunk=quantize_pcm24)
    for _ in range(6):
        capture._consume(np.ones((4, 2), dtype=np.float32))
    stats = diag.snapshot()["categories"]
    assert stats["consume"]["observed_count"] == 6
    assert stats["write"]["sampled_count"] == 1
    assert "observed_count" not in stats["write"]
    assert not records


def test_unmeasured_write_keeps_slow_interval_without_sampler(tmp_path, monkeypatch):
    from unit_test.base.test_recording_diagnostics import Clock

    clock = Clock()
    diag, records = diagnostics(monkeypatch, perf_ns=clock.perf, monotonic=clock.monotonic)
    capture = RecordingCapture(capture_request(tmp_path / "slow.wav"), diagnostics=diag)
    capture._writer = SimpleNamespace(write_chunk=lambda block: (clock.advance(110_000_000), quantize_pcm24(block))[1])
    capture._diagnostic_blocks = 1  # Not one of the detailed samples.
    capture._consume(np.ones((4, 2), dtype=np.float32))
    write, = events(records, "write")
    assert write["status"] == "completed"
    assert write["start_ns"] == 1_000_000_000
    assert write["elapsed_ns"] == 110_000_000
    assert write["monotonic_source"] == "qpc_aligned"
    assert diag.snapshot()["categories"]["write"]["slow_count"] == 1
