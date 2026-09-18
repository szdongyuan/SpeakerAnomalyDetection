from contextlib import contextmanager
from types import SimpleNamespace
import hashlib
import json

import av
from base.video.capture_diagnostics import CaptureDiagnostics
from base.video.config import VideoConfig
from base.video.usb_capture import CapturePump, enumerate_cameras, open_camera, parse_devices
import pytest


class InputPacket:
    pts = dts = 1
    time_base = "1/30"
    is_keyframe = True
    is_corrupt = False

    def __init__(self, data=b"input", frames=(), error=None):
        self.data, self.frames, self.error = data, frames, error
        self.size = len(data)

    def __bytes__(self):
        return self.data

    def decode(self):
        if self.error:
            raise self.error
        return self.frames


class InputSource:
    def __init__(self, packets, error=None):
        self.packets, self.error, self.closed = packets, error, False
        # Generic mocked packets exercise the unchanged non-MJPEG lifecycle.
        # MJPEG framing is covered with real compressed bytes in dedicated tests.
        codec = SimpleNamespace(name="rawvideo", width=1280, height=720, format=None,
                                thread_count=0, thread_type="SLICE")
        self.streams = SimpleNamespace(video=[SimpleNamespace(
            codec_context=codec, average_rate="30", time_base="1/30",
        )])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def demux(self, **kwargs):
        yield from self.packets
        if self.error:
            raise self.error


class RetryEvent:
    def __init__(self):
        self.stopped = False
        self.waits = []

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True

    def wait(self, delay):
        self.waits.append(delay)
        assert len(self.waits) <= 3, "capture failed to stop after recovery"
        return self.stopped


def test_dshow_device_names_use_alternative_identity_not_display_name():
    logs = [(32, "dshow", '"USB Camera" (video)\n  Alternative name "@device_pnp_one"\n'
            '"Microphone" (audio)\n  Alternative name "audio-id"\n'
            '"USB Camera" (video)\n  Alternative name "@device_pnp_two"\n')]
    devices = parse_devices(logs)
    assert [item.device_id for item in devices] == ["@device_pnp_one", "@device_pnp_two"]
    assert devices[0].name == devices[1].name


@pytest.mark.parametrize("chunk_size", [1, 7, 32])
def test_fragmented_device_logs_preserve_names_and_stable_identities(chunk_size):
    listing = (
        '"USB Cam" (video)\n  Alternative name "@device_pnp_one"\n'
        '"Microphone" (audio)\n  Alternative name "audio-id"\n'
        '"USB Cam" (video)\n  Alternative name "@device_pnp_two"\n'
    )
    logs = []
    for offset in range(0, len(listing), chunk_size):
        logs.append((32, "dshow", listing[offset:offset + chunk_size]))
        logs.append((32, "other", 'unrelated log\n'))
    devices = parse_devices(logs)
    assert [(device.name, device.device_id) for device in devices] == [
        ("USB Cam", "@device_pnp_one"), ("USB Cam", "@device_pnp_two"),
    ]


def test_real_usb_cam_callback_boundaries():
    logs = [(32, "dshow", fragment) for fragment in (
        '"USB Cam"', ' (video', ')', '\n',
        '  Alternative name "@device_pnp_usb_cam"\n',
    )]
    devices = parse_devices(logs)
    assert len(devices) == 1
    assert devices[0].name == "USB Cam"
    assert devices[0].device_id == "@device_pnp_usb_cam"


@pytest.mark.parametrize("listing", [
    "", '"Microphone" (audio)\n  Alternative name "audio-id"\n',
    '"USB Cam" (video)\n',
])
def test_no_complete_video_identity_is_not_a_camera(listing):
    assert parse_devices([(32, "dshow", listing)]) == ()


def test_missing_identity_never_falls_back_to_default_camera(monkeypatch):
    monkeypatch.setattr("base.video.usb_capture.enumerate_cameras", lambda: ())
    with pytest.raises(OSError, match="未找到"):
        open_camera(VideoConfig(device_id="missing"))


def test_installed_dshow_enumerator_returns_structured_devices():
    # Enumerates only; does not activate a camera or record any footage.
    assert isinstance(enumerate_cameras(), tuple)


def test_burst_capture_uses_precise_clock_but_watchdog_keeps_monotonic(monkeypatch):
    stamps = [1000.0, 1000.001, 1000.002]
    clock_values = iter(stamps)
    monkeypatch.setattr("base.video.usb_capture.time", SimpleNamespace(
        monotonic=lambda: 42.0, perf_counter=lambda: next(clock_values),
    ))
    config = VideoConfig()
    frame = SimpleNamespace(width=config.width, height=config.height)

    @contextmanager
    def opener(config):
        yield InputSource([InputPacket(frames=[frame] * 3)])

    received, progress = [], []

    def on_frame(frame, stamp):
        received.append(stamp)
        progress.append(pump.last_progress)
        if len(received) == 3:
            pump.stop()

    pump = CapturePump(config, on_frame, lambda *args: None, opener=opener)
    pump._run()
    assert received == stamps
    assert progress == [42.0] * 3


@pytest.mark.parametrize("stage", ["open", "demux", "decode", "frame_dispatch"])
def test_failure_stage_is_logged_and_same_capture_can_recover(stage, tmp_path, caplog):
    config = VideoConfig(recording_root=str(tmp_path))
    frame = SimpleNamespace(width=config.width, height=config.height)
    error = av.error.InvalidDataError(1094995529, "injected input failure")
    calls, connections, sources = [], [], []

    def opener(config):
        calls.append(True)
        if len(calls) == 1 and stage == "open":
            raise OSError(5, "injected open failure")
        bad = len(calls) == 1
        packet = InputPacket(data=b"original-failed-packet", frames=[frame],
                             error=error if bad and stage == "decode" else None)
        source = InputSource([] if bad and stage == "demux" else [packet],
                             error if bad and stage == "demux" else None)
        sources.append(source)
        return source

    def on_frame(frame, stamp):
        if len(calls) == 1 and stage == "frame_dispatch":
            raise ValueError("injected preview conversion failure")
        pump.stop()

    caplog.set_level("INFO", logger="core.video")
    pump = CapturePump(config, on_frame, lambda ready, detail: connections.append(ready), opener=opener)
    pump.stop_event = RetryEvent()
    pump._run()
    assert len(calls) == 2
    assert connections[-1] is True and False in connections
    assert all(source.closed for source in sources)
    assert all(delay == 2 for delay in pump.stop_event.waits)
    assert f"stage={stage}" in caplog.text
    assert "attempt=2" in caplog.text and "outage_seconds=" in caplog.text
    reports = list(tmp_path.rglob("result.json"))
    assert len(reports) == (1 if stage == "decode" else 0)
    if reports:
        report = json.loads(reports[0].read_text(encoding="utf-8"))
        item = report["saved_packets"][0]
        assert (reports[0].parent / item["file"]).read_bytes() == b"original-failed-packet"
        assert item["sha256"] == hashlib.sha256(b"original-failed-packet").hexdigest()


@pytest.mark.parametrize("byte_limit, packet_limit, expected", [(10, 16, 2), (100, 2, 2)])
def test_raw_evidence_has_byte_and_packet_caps(monkeypatch, byte_limit, packet_limit, expected):
    monkeypatch.setattr("base.video.capture_diagnostics.PACKET_BYTE_LIMIT", byte_limit)
    monkeypatch.setattr("base.video.capture_diagnostics.PACKET_COUNT_LIMIT", packet_limit)
    evidence = CaptureDiagnostics(VideoConfig())
    for index in range(10):
        evidence.remember(InputPacket(data=bytes([index]) * 4))
    assert len(evidence.history) == expected
    assert evidence.history_bytes <= byte_limit
    assert evidence.history[-1][1] == b"\x09" * 4
    evidence.remember(InputPacket(data=b"x" * (byte_limit + 1)))
    assert evidence.history_bytes == 0
    assert evidence.packet["size"] == byte_limit + 1


def test_sample_disk_limit_survives_new_capture_instances(tmp_path, caplog):
    for _ in range(4):
        evidence = CaptureDiagnostics(VideoConfig(recording_root=str(tmp_path)))
        evidence.begin_attempt()
        evidence.remember(InputPacket())
        evidence.stage = "decode"
        evidence.failed(ValueError("injected"))
    assert len(list(tmp_path.rglob("result.json"))) == 3
    assert "evidence limit reached" in caplog.text


def test_evidence_write_failure_does_not_prevent_reconnect(tmp_path, monkeypatch, caplog):
    config = VideoConfig(recording_root=str(tmp_path))
    frame = SimpleNamespace(width=config.width, height=config.height)
    sources = [InputSource([InputPacket(error=av.error.InvalidDataError(1094995529, "bad"))]),
               InputSource([InputPacket(frames=[frame])])]
    pending = iter(sources)

    def cannot_write(self, exc):
        assert sources[0].closed
        raise PermissionError("injected disk permission error")

    monkeypatch.setattr(CaptureDiagnostics, "_save_sample", cannot_write)
    pump = CapturePump(config, lambda *args: pump.stop(), lambda *args: None,
                       opener=lambda config: next(pending))
    pump.stop_event = RetryEvent()
    pump._run()
    assert pump.frames == 1 and sources[1].closed
    assert "evidence write failed" in caplog.text


def test_packet_iteration_matches_pyav_decode_for_real_jpeg():
    import io
    import numpy as np

    encoder = av.CodecContext.create("mjpeg", "w")
    encoder.width, encoder.height, encoder.pix_fmt = 160, 90, "yuvj420p"
    frame = av.VideoFrame.from_ndarray(np.full((90, 160, 3), 80, dtype=np.uint8), format="rgb24")
    data = b"".join(bytes(packet) for packet in encoder.encode(frame) + encoder.encode(None))
    with av.open(io.BytesIO(data), format="mjpeg") as source:
        expected = [frame.to_ndarray(format="rgb24") for frame in source.decode(video=0)]
    pump = CapturePump(VideoConfig(), lambda *args: None, lambda *args: None)
    with av.open(io.BytesIO(data), format="mjpeg") as source:
        received = [frame.to_ndarray(format="rgb24") for frame in pump._decode(source)]
    assert len(expected) == len(received) == 1
    np.testing.assert_array_equal(received[0], expected[0])


def test_failure_evidence_does_not_attribute_other_thread_logs_to_decoder(tmp_path, caplog):
    import threading

    class NoisyPacket(InputPacket):
        def decode(self):
            thread = threading.Thread(target=lambda: av.logging.log(
                av.logging.ERROR, "writer", "unrelated writer operation\n",
            ))
            thread.start()
            thread.join()
            av.logging.log(av.logging.ERROR, "mjpeg", "local input decoder failure\n")
            raise av.error.InvalidDataError(
                1094995529, "bad input", log=(av.logging.ERROR, "h264", "stale writer suffix"),
            )

    previous_level = av.logging.get_level()
    av.logging.set_level(av.logging.INFO)
    pump = CapturePump(VideoConfig(recording_root=str(tmp_path)), lambda *args: None, lambda *args: None)
    try:
        with pytest.raises(av.error.InvalidDataError) as failure:
            list(pump._decode(InputSource([NoisyPacket()])))
        pump.diagnostics.failed(failure.value)
    finally:
        av.logging.set_level(previous_level)
    report = json.loads(next(tmp_path.rglob("result.json")).read_text(encoding="utf-8"))
    assert report["message"] == "bad input"
    assert "local input decoder failure" in str(report["local_ffmpeg_logs"])
    assert "unrelated writer operation" not in str(report)
    assert "stale writer suffix" not in str(report)
