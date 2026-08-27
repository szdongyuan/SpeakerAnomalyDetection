import os
import sys
from datetime import datetime
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile

import record_vkinging_continuous as recorder
from record_vkinging_continuous import SegmentAccumulator, WavPublisher


class FakeStream:
    def __init__(self, events, chunks):
        self.events = events
        self.chunks = iter(chunks)

    def __enter__(self):
        self.events.append("stream_enter")
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.events.append("stream_exit")

    def __iter__(self):
        return self

    def __next__(self):
        chunk = next(self.chunks)
        if isinstance(chunk, BaseException):
            raise chunk
        return chunk


class FakeClient:
    def __init__(self, events, *, devices=("device zero",), channels=("ai1", "ai0"), chunks=()):
        self.events = events
        self.devices = devices
        self.channels = channels
        self.chunks = chunks
        self.selected = None
        self.stream_arguments = None

    def __enter__(self):
        self.events.append("client_enter")
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.events.append("client_exit")

    def list_devices(self):
        return self.devices

    def select_device(self, selector):
        self.selected = selector
        return self.devices[selector]

    def list_channels(self):
        return self.channels

    def stream(self, **arguments):
        self.stream_arguments = arguments
        return FakeStream(self.events, self.chunks)


class TrackingPublisher:
    def __init__(self, events, *, interrupt_on_call=None):
        self.events = events
        self.interrupt_on_call = interrupt_on_call
        self.calls = 0
        self.saved = []

    def publish_pending(self, accumulator, *, sample_rate, timestamp):
        head = accumulator.peek_complete()
        if head is None:
            return None
        self.calls += 1
        self.events.append(f"publish_{self.calls}")
        if self.calls == self.interrupt_on_call:
            raise KeyboardInterrupt
        accumulator.acknowledge(head)
        path = Path(f"saved-{self.calls}.wav")
        self.saved.append(head.copy())
        return path


def chunk(*channels):
    return SimpleNamespace(samples=channels)


def install_small_accumulator(monkeypatch, events=None, *, interrupt_add=False):
    requested_sizes = []
    real_accumulator = recorder.SegmentAccumulator

    class SmallAccumulator(real_accumulator):
        def __init__(self, *, channel_count, frames_per_segment):
            requested_sizes.append(frames_per_segment)
            super().__init__(channel_count=channel_count, frames_per_segment=3)

        def add_channel_major(self, channel_major):
            super().add_channel_major(channel_major)
            if interrupt_add:
                if events is not None:
                    events.append("conversion_interrupt")
                raise KeyboardInterrupt

    monkeypatch.setattr(recorder, "SegmentAccumulator", SmallAccumulator)
    return requested_sizes


def test_accumulator_splits_uneven_channel_major_chunks_without_loss():
    accumulator = SegmentAccumulator(channel_count=2, frames_per_segment=5)

    accumulator.add_channel_major(((1, 2, 3), (11, 12, 13)))
    accumulator.add_channel_major(((4, 5, 6, 7), (14, 15, 16, 17)))

    assert accumulator.peek_complete().dtype == np.float32
    np.testing.assert_array_equal(
        accumulator.peek_complete(),
        [[1, 11], [2, 12], [3, 13], [4, 14], [5, 15]],
    )
    first = accumulator.peek_complete()
    accumulator.acknowledge(first)
    accumulator.queue_tail()
    np.testing.assert_array_equal(accumulator.peek_complete(), [[6, 16], [7, 17]])


def test_accumulator_queues_multiple_segments_from_oversized_chunk_and_keeps_tail():
    accumulator = SegmentAccumulator(channel_count=2, frames_per_segment=3)

    accumulator.add_channel_major((range(1, 9), range(11, 19)))

    first = accumulator.peek_complete()
    np.testing.assert_array_equal(first, [[1, 11], [2, 12], [3, 13]])
    accumulator.acknowledge(first)
    second = accumulator.peek_complete()
    np.testing.assert_array_equal(second, [[4, 14], [5, 15], [6, 16]])
    accumulator.acknowledge(second)
    accumulator.queue_tail()
    np.testing.assert_array_equal(accumulator.peek_complete(), [[7, 17], [8, 18]])


def test_accumulator_rejects_mismatched_channel_lengths():
    accumulator = SegmentAccumulator(channel_count=2, frames_per_segment=5)

    with pytest.raises(ValueError, match="same number of frames"):
        accumulator.add_channel_major(((1, 2), (11,)))


def test_accumulator_rejects_wrong_channel_count():
    accumulator = SegmentAccumulator(channel_count=2, frames_per_segment=5)

    with pytest.raises(ValueError, match="expected 2 channels"):
        accumulator.add_channel_major(((1, 2),))


def test_accumulator_acknowledge_rejects_non_head_object():
    accumulator = SegmentAccumulator(channel_count=1, frames_per_segment=2)
    accumulator.add_channel_major(((1, 2, 3, 4),))
    head = accumulator.peek_complete()
    equal_but_distinct = head.copy()

    with pytest.raises(ValueError, match="head"):
        accumulator.acknowledge(equal_but_distinct)

    assert accumulator.peek_complete() is head


def test_accumulator_empty_queue_has_no_tail_segment():
    accumulator = SegmentAccumulator(channel_count=2, frames_per_segment=5)

    assert accumulator.queue_tail() is None
    assert accumulator.peek_complete() is None


class InterruptingAccumulator(SegmentAccumulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interrupt_phase = None

    def _commit_state(self, next_state):
        if self.interrupt_phase == "before":
            raise KeyboardInterrupt
        super()._commit_state(next_state)
        if self.interrupt_phase == "after":
            raise KeyboardInterrupt


@pytest.mark.parametrize("interrupt_phase", ["before", "after"])
def test_accumulator_add_interrupt_exposes_only_old_or_complete_new_state(
    interrupt_phase,
):
    accumulator = InterruptingAccumulator(channel_count=2, frames_per_segment=3)
    accumulator.add_channel_major(((1,), (11,)))
    accumulator.interrupt_phase = interrupt_phase

    with pytest.raises(KeyboardInterrupt):
        accumulator.add_channel_major(((2, 3, 4), (12, 13, 14)))

    accumulator.interrupt_phase = None
    accumulator.queue_tail()
    if interrupt_phase == "before":
        np.testing.assert_array_equal(accumulator.peek_complete(), [[1, 11]])
    else:
        first = accumulator.peek_complete()
        np.testing.assert_array_equal(first, [[1, 11], [2, 12], [3, 13]])
        accumulator.acknowledge(first)
        np.testing.assert_array_equal(accumulator.peek_complete(), [[4, 14]])


@pytest.mark.parametrize("interrupt_phase", ["before", "after"])
def test_accumulator_acknowledge_interrupt_is_transactional(interrupt_phase):
    accumulator = InterruptingAccumulator(channel_count=1, frames_per_segment=2)
    accumulator.add_channel_major(((1, 2, 3, 4),))
    first = accumulator.peek_complete()
    accumulator.interrupt_phase = interrupt_phase

    with pytest.raises(KeyboardInterrupt):
        accumulator.acknowledge(first)

    if interrupt_phase == "before":
        assert accumulator.peek_complete() is first
    else:
        np.testing.assert_array_equal(accumulator.peek_complete(), [[3], [4]])


@pytest.mark.parametrize("interrupt_phase", ["before", "after"])
def test_accumulator_queue_tail_interrupt_never_loses_or_duplicates_tail(
    interrupt_phase,
):
    accumulator = InterruptingAccumulator(channel_count=1, frames_per_segment=3)
    accumulator.add_channel_major(((1, 2),))
    accumulator.interrupt_phase = interrupt_phase

    with pytest.raises(KeyboardInterrupt):
        accumulator.queue_tail()

    accumulator.interrupt_phase = None
    if interrupt_phase == "before":
        assert accumulator.peek_complete() is None
        accumulator.queue_tail()
    else:
        assert accumulator.queue_tail() is None
    np.testing.assert_array_equal(accumulator.peek_complete(), [[1], [2]])


@pytest.mark.parametrize("channel_count", [0, -1, 1.5, True])
def test_accumulator_requires_positive_integer_channel_count(channel_count):
    with pytest.raises(ValueError, match="channel_count"):
        SegmentAccumulator(channel_count=channel_count, frames_per_segment=5)


@pytest.mark.parametrize("frames_per_segment", [0, -1, 1.5, True])
def test_accumulator_requires_positive_integer_segment_size(frames_per_segment):
    with pytest.raises(ValueError, match="frames_per_segment"):
        SegmentAccumulator(channel_count=2, frames_per_segment=frames_per_segment)


def pending_accumulator():
    accumulator = SegmentAccumulator(channel_count=2, frames_per_segment=3)
    accumulator.add_channel_major(((1.25, 2.5, 3.75), (-1.25, -2.5, -3.75)))
    return accumulator


def test_wav_publisher_writes_float32_multichannel_samples_at_requested_rate(tmp_path):
    accumulator = pending_accumulator()
    publisher = WavPublisher(tmp_path)

    published = publisher.publish_pending(
        accumulator,
        sample_rate=48_000,
        timestamp=datetime(2026, 8, 25, 12, 34, 56),
    )

    assert published == tmp_path / "recording_20260825_123456.wav"
    rate, samples = wavfile.read(published)
    assert rate == 48_000
    assert samples.dtype == np.float32
    np.testing.assert_array_equal(
        samples,
        [[1.25, -1.25], [2.5, -2.5], [3.75, -3.75]],
    )
    assert accumulator.peek_complete() is None


def test_wav_publisher_selects_incrementing_suffix_without_overwriting(tmp_path):
    timestamp = datetime(2026, 8, 25, 12, 34, 56)
    original = tmp_path / "recording_20260825_123456.wav"
    first_collision = tmp_path / "recording_20260825_123456_1.wav"
    original.write_bytes(b"original recording")
    first_collision.write_bytes(b"first collision")

    published = WavPublisher(tmp_path).publish_pending(
        pending_accumulator(), sample_rate=48_000, timestamp=timestamp
    )

    assert published == tmp_path / "recording_20260825_123456_2.wav"
    assert original.read_bytes() == b"original recording"
    assert first_collision.read_bytes() == b"first collision"


def test_publish_collision_during_rename_reuses_completed_temp_wav(tmp_path):
    timestamp = datetime(2026, 8, 25, 12, 34, 56)
    first_candidate = tmp_path / "recording_20260825_123456.wav"
    write_calls = []
    rename_calls = []

    def tracking_writer(path, rate, samples):
        write_calls.append((path, rate, samples.copy()))
        wavfile.write(path, rate, samples)

    def colliding_rename(source, destination):
        rename_calls.append((source, destination))
        if len(rename_calls) == 1:
            destination.write_bytes(b"concurrent winner")
            raise FileExistsError("destination appeared concurrently")
        os.rename(source, destination)

    accumulator = pending_accumulator()
    published = WavPublisher(
        tmp_path, wav_writer=tracking_writer, rename=colliding_rename
    ).publish_pending(accumulator, sample_rate=48_000, timestamp=timestamp)

    assert published == tmp_path / "recording_20260825_123456_1.wav"
    assert first_candidate.read_bytes() == b"concurrent winner"
    assert len(write_calls) == 1
    assert len(rename_calls) == 2
    assert rename_calls[0][0] == rename_calls[1][0]
    rate, samples = wavfile.read(published)
    assert rate == 48_000
    np.testing.assert_array_equal(samples, write_calls[0][2])
    assert accumulator.peek_complete() is None


@pytest.mark.parametrize("failure_stage", ["write", "rename"])
def test_wav_publish_failure_cleans_temp_and_retains_pending_head(
    tmp_path, failure_stage
):
    accumulator = pending_accumulator()
    head = accumulator.peek_complete()

    def failing_writer(path, rate, samples):
        if failure_stage == "write":
            raise OSError("write failed")
        wavfile.write(path, rate, samples)

    def failing_rename(source, destination):
        raise OSError("rename failed")

    publisher = WavPublisher(
        tmp_path,
        wav_writer=failing_writer,
        rename=failing_rename if failure_stage == "rename" else os.rename,
    )

    with pytest.raises(OSError, match=f"{failure_stage} failed"):
        publisher.publish_pending(
            accumulator,
            sample_rate=48_000,
            timestamp=datetime(2026, 8, 25, 12, 34, 56),
        )

    assert accumulator.peek_complete() is head
    assert list(tmp_path.iterdir()) == []


def test_interrupt_after_successful_rename_acknowledges_once_without_republish(tmp_path):
    accumulator = pending_accumulator()
    head = accumulator.peek_complete()
    acknowledge_calls = []
    real_acknowledge = accumulator.acknowledge

    def tracking_acknowledge(segment):
        acknowledge_calls.append(segment)
        real_acknowledge(segment)

    accumulator.acknowledge = tracking_acknowledge

    def rename_then_interrupt(source, destination):
        os.rename(source, destination)
        raise KeyboardInterrupt

    timestamp = datetime(2026, 8, 25, 12, 34, 56)
    publisher = WavPublisher(tmp_path, rename=rename_then_interrupt)

    with pytest.raises(KeyboardInterrupt):
        publisher.publish_pending(
            accumulator, sample_rate=48_000, timestamp=timestamp
        )

    final_path = tmp_path / "recording_20260825_123456.wav"
    assert final_path.exists()
    assert acknowledge_calls == [head]
    assert accumulator.peek_complete() is None
    assert (
        WavPublisher(tmp_path).publish_pending(
            accumulator, sample_rate=48_000, timestamp=timestamp
        )
        is None
    )
    assert list(tmp_path.iterdir()) == [final_path]


@pytest.mark.parametrize("interrupt_stage", ["write", "rename"])
def test_interrupt_before_successful_rename_cleans_temp_and_retains_head(
    tmp_path, interrupt_stage
):
    accumulator = pending_accumulator()
    head = accumulator.peek_complete()

    def interrupt_during_write(path, rate, samples):
        raise KeyboardInterrupt

    def interrupt_before_rename(source, destination):
        raise KeyboardInterrupt

    publisher = WavPublisher(
        tmp_path,
        wav_writer=interrupt_during_write if interrupt_stage == "write" else None,
        rename=interrupt_before_rename if interrupt_stage == "rename" else None,
    )

    with pytest.raises(KeyboardInterrupt):
        publisher.publish_pending(
            accumulator,
            sample_rate=48_000,
            timestamp=datetime(2026, 8, 25, 12, 34, 56),
        )

    assert accumulator.peek_complete() is head
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("interrupt_point", ["path_conversion", "descriptor_close"])
def test_interrupt_after_mkstemp_closes_descriptor_removes_temp_and_retains_head(
    tmp_path, monkeypatch, interrupt_point
):
    accumulator = pending_accumulator()
    head = accumulator.peek_complete()
    publisher = WavPublisher(tmp_path)
    created = {}
    real_mkstemp = tempfile.mkstemp
    real_close = os.close

    def tracking_mkstemp(*args, **kwargs):
        descriptor, name = real_mkstemp(*args, **kwargs)
        created.update(descriptor=descriptor, name=name)
        return descriptor, name

    def interrupt_during_path_conversion(value):
        if interrupt_point == "path_conversion" and value == created.get("name"):
            raise KeyboardInterrupt
        return Path(value)

    close_interrupted = False

    def interrupt_during_first_close(descriptor):
        nonlocal close_interrupted
        if (
            interrupt_point == "descriptor_close"
            and descriptor == created.get("descriptor")
            and not close_interrupted
        ):
            close_interrupted = True
            raise KeyboardInterrupt
        real_close(descriptor)

    monkeypatch.setattr(recorder.tempfile, "mkstemp", tracking_mkstemp)
    monkeypatch.setattr(recorder, "Path", interrupt_during_path_conversion)
    monkeypatch.setattr(recorder.os, "close", interrupt_during_first_close)

    with pytest.raises(KeyboardInterrupt):
        publisher.publish_pending(
            accumulator,
            sample_rate=48_000,
            timestamp=datetime(2026, 8, 25, 12, 34, 56),
        )

    descriptor = created["descriptor"]
    temp_name = created["name"]
    try:
        with pytest.raises(OSError):
            os.fstat(descriptor)
        assert not Path(temp_name).exists()
        assert accumulator.peek_complete() is head
    finally:
        try:
            real_close(descriptor)
        except OSError:
            pass
        Path(temp_name).unlink(missing_ok=True)


def test_recorder_selects_first_device_all_channels_and_exact_stream_settings(
    monkeypatch, capsys
):
    events = []
    client = FakeClient(events, chunks=(KeyboardInterrupt(),))
    publisher = TrackingPublisher(events)
    requested_sizes = install_small_accumulator(monkeypatch)

    status = recorder.run_recorder(lambda: client, publisher=publisher)

    assert status == 0
    assert client.selected == 0
    assert client.stream_arguments == {
        "channels": ("ai1", "ai0"),
        "mode": "voltage",
        "sample_rate": 48_000,
        "samples_per_chunk": 48_000,
        "min_value": -10.0,
        "max_value": 10.0,
        "terminal": "single_ended",
        "timeout": 2.0,
    }
    assert client.stream_arguments["timeout"] > 0
    assert requested_sizes == [48_000 * 60]
    assert requested_sizes == [2_880_000]
    output = capsys.readouterr().out
    assert "Selected device: device zero" in output
    assert "Selected channels: ai1, ai0" in output
    assert "Recording started" in output
    assert "Recording stopped" in output


@pytest.mark.parametrize(
    ("devices", "channels", "message"),
    [
        ((), ("ai0",), "No Vkinging devices found"),
        (("device zero",), (), "No channels found"),
    ],
)
def test_recorder_rejects_missing_device_or_channel_before_stream_creation(
    devices, channels, message
):
    events = []
    client = FakeClient(events, devices=devices, channels=channels)

    with pytest.raises(RuntimeError, match=message):
        recorder.run_recorder(lambda: client, publisher=TrackingPublisher(events))

    assert client.stream_arguments is None


def test_recorder_publishes_complete_segments_while_one_stream_stays_open(
    monkeypatch, capsys
):
    events = []
    client = FakeClient(
        events,
        chunks=(
            chunk((1, 2, 3, 4, 5, 6), (11, 12, 13, 14, 15, 16)),
            KeyboardInterrupt(),
        ),
    )
    publisher = TrackingPublisher(events)
    install_small_accumulator(monkeypatch)

    assert recorder.run_recorder(lambda: client, publisher=publisher) == 0

    assert events == [
        "client_enter",
        "stream_enter",
        "publish_1",
        "publish_2",
        "stream_exit",
        "client_exit",
    ]
    assert len(publisher.saved) == 2
    output = capsys.readouterr().out
    assert output.count("Saved: saved-1.wav") == 1
    assert output.count("Saved: saved-2.wav") == 1


@pytest.mark.parametrize("interrupt_stage", ["iteration", "conversion", "publication"])
def test_first_interrupt_closes_native_contexts_before_finalizing_complete_and_tail(
    monkeypatch, interrupt_stage
):
    events = []
    chunks = [chunk((1, 2, 3, 4), (11, 12, 13, 14))]
    interrupt_add = interrupt_stage == "conversion"
    if interrupt_stage == "iteration":
        chunks.append(KeyboardInterrupt())
    elif interrupt_stage == "publication":
        chunks.append(KeyboardInterrupt())
    publisher = TrackingPublisher(
        events, interrupt_on_call=1 if interrupt_stage == "publication" else None
    )
    client = FakeClient(events, chunks=chunks)
    install_small_accumulator(
        monkeypatch, events=events, interrupt_add=interrupt_add
    )

    assert recorder.run_recorder(lambda: client, publisher=publisher) == 0

    stream_exit = events.index("stream_exit")
    client_exit = events.index("client_exit")
    finalization_publications = [
        index
        for index, event in enumerate(events)
        if event.startswith("publish_") and index > client_exit
    ]
    assert stream_exit < client_exit < min(finalization_publications)
    assert [saved.shape[0] for saved in publisher.saved] == [3, 1]


def test_recorder_reports_post_rename_interrupt_path_once_then_finalizes(
    monkeypatch, tmp_path, capsys
):
    events = []
    client = FakeClient(
        events,
        chunks=(chunk((1, 2, 3, 4), (11, 12, 13, 14)),),
    )
    destinations = []

    def interrupt_after_first_rename(source, destination):
        os.rename(source, destination)
        destinations.append(destination)
        events.append(f"rename_{len(destinations)}")
        if len(destinations) == 1:
            raise KeyboardInterrupt

    publisher = WavPublisher(tmp_path, rename=interrupt_after_first_rename)
    install_small_accumulator(monkeypatch)

    assert recorder.run_recorder(lambda: client, publisher=publisher) == 0

    first_path = destinations[0]
    output_lines = capsys.readouterr().out.splitlines()
    assert output_lines.count(f"Saved: {first_path}") == 1
    assert len(destinations) == 2
    assert all(path.exists() for path in destinations)
    assert events.index("rename_1") < events.index("stream_exit")
    assert events.index("client_exit") < events.index("rename_2")


def test_recorder_does_not_publish_empty_tail(monkeypatch):
    events = []
    client = FakeClient(
        events,
        chunks=(chunk((1, 2, 3), (11, 12, 13)), KeyboardInterrupt()),
    )
    publisher = TrackingPublisher(events)
    install_small_accumulator(monkeypatch)

    assert recorder.run_recorder(lambda: client, publisher=publisher) == 0

    assert publisher.calls == 1


def test_second_interrupt_during_finalization_cleans_temp_and_returns_130(
    tmp_path, capsys
):
    events = []
    client = FakeClient(
        events,
        channels=("ai0",),
        chunks=(chunk((1, 2, 3)), KeyboardInterrupt()),
    )

    def interrupting_writer(path, rate, samples):
        raise KeyboardInterrupt

    publisher = WavPublisher(tmp_path, wav_writer=interrupting_writer)

    status = recorder.run_recorder(lambda: client, publisher=publisher)

    assert status == 130
    assert events[-2:] == ["stream_exit", "client_exit"]
    assert list(tmp_path.iterdir()) == []
    assert "buffered data was not saved" in capsys.readouterr().err.lower()


def test_injected_client_factory_keeps_vendor_binding_lazy(monkeypatch):
    monkeypatch.delitem(sys.modules, "vkinging_daq", raising=False)
    events = []
    client = FakeClient(events, chunks=(KeyboardInterrupt(),))

    assert recorder.run_recorder(
        lambda: client, publisher=TrackingPublisher(events)
    ) == 0
    assert "vkinging_daq" not in sys.modules


def test_recorder_imports_vendor_client_only_when_factory_is_omitted(monkeypatch):
    events = []
    client = FakeClient(events, chunks=(KeyboardInterrupt(),))
    calls = []

    def vendor_factory():
        calls.append("constructed")
        return client

    fake_binding = SimpleNamespace(VkDaqClient=vendor_factory)
    monkeypatch.setitem(sys.modules, "vkinging_daq", fake_binding)

    assert recorder.run_recorder(publisher=TrackingPublisher(events)) == 0
    assert calls == ["constructed"]


def test_main_prints_one_concise_cli_error_and_returns_nonzero(monkeypatch, capsys):
    def fail():
        raise OSError("SDK unavailable")

    monkeypatch.setattr(recorder, "run_recorder", fail)

    assert recorder.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("SDK unavailable") == 1
