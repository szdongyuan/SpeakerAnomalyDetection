"""Isolated technical PoC, not imported by the application or a production backend.

Only immutable parameters/packet bytes cross the two worker threads. All media
is synthetic and written under pytest's temporary directory; no camera imports.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
import io
import os
import queue
import threading
import time

import av
import numpy as np
import pytest


TIME_BASE = Fraction(1, 90000)
RATE = 30
BYTE_LIMIT = 64 * 1024**2


@dataclass(frozen=True)
class Parameters:
    codec: str
    width: int
    height: int
    extradata: bytes


@dataclass(frozen=True)
class EncodedPacket:
    payload: bytes
    pts: int
    dts: int
    duration: int
    time_base: Fraction
    keyframe: bool
    side_data: tuple

    @classmethod
    def capture(cls, packet):
        return cls(bytes(packet), packet.pts, packet.dts, packet.duration,
                   packet.time_base, packet.is_keyframe,
                   tuple((side.data_type, bytes(side)) for side in packet.iter_sidedata()))

    @property
    def cost(self):
        return len(self.payload) + sum(len(data) for _, data in self.side_data) + 512

    def restore(self):
        packet = av.Packet(self.payload)
        packet.pts, packet.dts = self.pts, self.dts
        packet.duration, packet.time_base = self.duration, self.time_base
        packet.is_keyframe = self.keyframe
        for name, data in self.side_data:
            side = av.packet.PacketSideData(av.packet.packet_sidedata_type_from_literal(name), len(data))
            side.update(data)
            side.to_packet(packet)
        assert tuple((s.data_type, bytes(s)) for s in packet.iter_sidedata()) == self.side_data
        return packet


class TestTransport:
    """Minimal test accounting; not the production queue or its full contract."""

    __test__ = False

    def __init__(self):
        self.items = queue.Queue(maxsize=8192)
        self.lock = threading.Lock()
        self.abort = threading.Event()
        self.bytes = self.peak = self.packet_count = self.peak_packets = 0

    def put(self, message):
        cost = message.cost if isinstance(message, EncodedPacket) else 0
        with self.lock:
            if self.bytes + cost > BYTE_LIMIT:
                raise AssertionError("PoC compressed-byte budget exceeded")
            self.bytes += cost
            self.packet_count += bool(cost)
            self.peak = max(self.peak, self.bytes)
            self.peak_packets = max(self.peak_packets, self.packet_count)
        self.items.put_nowait(message)

    def release(self, packet):
        with self.lock:
            self.bytes -= packet.cost
            self.packet_count -= 1

    def get(self):
        while not self.abort.is_set():
            try:
                return self.items.get(timeout=.1)
            except queue.Empty:
                pass
        raise RuntimeError("other PoC worker failed")


def synthetic_frame(index, width, height, background):
    pixels = np.roll(background, index * 3, axis=1).copy()
    block = width // 16
    for bit in range(16):
        pixels[:48, bit * block:(bit + 1) * block] = 235 if index & (1 << bit) else 16
    return av.VideoFrame.from_ndarray(pixels, format="rgb24")


def run_poc(directory, codec_name, *, width=320, height=180, frames=90,
            segment_frames=45, stall_seconds=0, paced=False, b_frames=0):
    transport = TestTransport()
    stats = {"encoded": 0, "written": 0, "tail_packets": 0, "stall_encoded": 0}
    files = []

    def encode():
        stats["encoder_thread"] = threading.get_ident()
        background = np.random.default_rng(17).integers(16, 236, (height, width, 3), dtype=np.uint8)
        start = time.perf_counter()
        for first in range(0, frames, segment_frames):
            codec = av.CodecContext.create(codec_name, "w")
            codec.width, codec.height, codec.pix_fmt = width, height, "yuv420p"
            codec.time_base, codec.framerate = TIME_BASE, Fraction(RATE)
            codec.bit_rate, codec.thread_count = 4_000_000, 2
            codec.gop_size, codec.max_b_frames = RATE, b_frames
            codec.flags |= av.codec.context.Flags.global_header
            codec.options = {"preset": "veryfast"}
            # The production mode has no B frames. Delayed-output coverage uses
            # normal lookahead; x265 rejects B frames with zerolatency lookahead.
            if not b_frames:
                codec.options["tune"] = "zerolatency"
            if codec_name == "libx265":
                codec.options["x265-params"] = "pools=1:frame-threads=1:log-level=error"
            codec.open()
            transport.put(Parameters(codec_name, width, height, bytes(codec.extradata)))
            for index in range(first, min(first + segment_frames, frames)):
                if paced and transport.abort.wait(max(0, start + index / RATE - time.perf_counter())):
                    raise RuntimeError("PoC cancelled")
                if transport.abort.is_set():
                    raise RuntimeError("PoC cancelled")
                frame = synthetic_frame(index, width, height, background)
                frame.pts, frame.time_base = (index - first) * 3000, TIME_BASE
                for packet in codec.encode(frame):
                    transport.put(EncodedPacket.capture(packet))
                stats["encoded"] += 1
            for packet in codec.encode(None):
                stats["tail_packets"] += 1
                transport.put(EncodedPacket.capture(packet))
            transport.put("segment_end")
            del codec
        transport.put("end")

    class PausedFile:
        def __init__(self, stream):
            self.stream = stream

        def write(self, data):
            if stall_seconds and stats["written"] >= 15 and not stats.get("stalled"):
                stats["stalled"] = True
                before = stats["encoded"]
                started = time.perf_counter()
                if transport.abort.wait(stall_seconds):
                    raise RuntimeError("PoC cancelled during write")
                stats["stall_seconds"] = time.perf_counter() - started
                stats["stall_encoded"] = stats["encoded"] - before
            return self.stream.write(data)

        def __getattr__(self, name):
            return getattr(self.stream, name)

    def write():
        stats["writer_thread"] = threading.get_ident()
        container = stream = output = None
        sync_due = time.monotonic() + 1
        try:
            while True:
                message = transport.get()
                if isinstance(message, Parameters):
                    assert container is None
                    path = directory / f"segment_{len(files) + 1:03d}.mp4"
                    files.append(path)
                    output = path.open("xb", buffering=0)
                    container = av.open(PausedFile(output), "w", format="mp4", options={
                        "movflags": "empty_moov+default_base_moof+frag_keyframe",
                        "frag_duration": "1000000", "flush_packets": "1",
                    })
                    # PyAV 16.1 has no add_mux_stream. Build a decoder template
                    # locally from bounded header bytes, never from an active file.
                    fmt = "h264" if message.codec == "libx264" else "hevc"
                    with av.open(io.BytesIO(message.extradata), format=fmt) as header:
                        stream = container.add_stream_from_template(header.streams.video[0], opaque=True)
                    context = stream.codec_context
                    assert context.is_decoder
                    context.width, context.height = message.width, message.height
                    context.pix_fmt, context.extradata = "yuv420p", message.extradata
                    stream.time_base = TIME_BASE
                    container.start_encoding()
                    stats.setdefault("header_sizes", []).append(len(context.extradata))
                elif isinstance(message, EncodedPacket):
                    try:
                        packet = message.restore()
                        packet.stream = stream
                        container.mux(packet)
                        stats["written"] += 1
                        if time.monotonic() >= sync_due:
                            output.flush()
                            os.fsync(output.fileno())
                            sync_due = time.monotonic() + 1
                    finally:
                        transport.release(message)
                elif message == "segment_end":
                    container.close()
                    container = None
                    output.flush()
                    os.fsync(output.fileno())
                    output.close()
                    output = None
                else:
                    assert message == "end" and container is None
                    break
        finally:
            try:
                if container is not None:
                    container.close()
            finally:
                if output is not None:
                    output.close()

    def guarded(operation):
        try:
            operation()
        except BaseException:
            transport.abort.set()
            raise

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="VideoPoC") as pool:
        encoder = pool.submit(guarded, encode)
        writer = pool.submit(guarded, write)
        try:
            encoder.result(timeout=45)
            writer.result(timeout=45)
        finally:
            transport.abort.set()
    assert stats["encoder_thread"] != stats["writer_thread"]
    assert transport.bytes == transport.packet_count == 0
    stats.update(peak_bytes=transport.peak, peak_packets=transport.peak_packets)
    return files, stats


def verify_files(files, expected_frames, *, b_frames=0):
    decoded_ids = []
    for path in files:
        with av.open(str(path)) as container:
            times = []
            for frame in container.decode(video=0):
                if not times:
                    assert frame.key_frame
                luma = np.frombuffer(frame.planes[0], np.uint8).reshape(frame.height, frame.planes[0].line_size)
                block = frame.width // 16
                identity = sum(1 << bit for bit in range(16)
                               if luma[12:36, bit * block + block // 4:bit * block + 3 * block // 4].mean() > 128)
                decoded_ids.append(identity)
                times.append(float(frame.time))
            assert times
            if not b_frames:
                assert times[0] == 0
            assert np.allclose(np.array(times) - times[0], np.arange(len(times)) / RATE, atol=1 / 90000)
        assert b"moof" in path.read_bytes()
    assert decoded_ids == list(range(expected_frames))


@pytest.mark.parametrize("codec", ["libx264", "libx265"])
@pytest.mark.parametrize("b_frames", [0, 2])
def test_independent_encoding_and_muxing(tmp_path, codec, b_frames):
    files, stats = run_poc(tmp_path, codec, b_frames=b_frames)
    verify_files(files, 90, b_frames=b_frames)
    assert len(files) == 2
    assert stats["encoded"] == stats["written"] == 90
    if b_frames:
        assert stats["tail_packets"] > 0
    print(f"PoC {codec} b_frames={b_frames}: {stats}")


@pytest.mark.parametrize("stall_seconds", [2, 5, 10])
def test_720p_recovers_from_write_stall(tmp_path, stall_seconds):
    frames = (stall_seconds + 3) * RATE
    files, stats = run_poc(tmp_path, "libx264", width=1280, height=720,
                           frames=frames, segment_frames=frames, stall_seconds=stall_seconds, paced=True)
    verify_files(files, frames)
    assert stats["stall_seconds"] >= stall_seconds
    assert stats["stall_encoded"] >= stall_seconds * RATE * .7
    assert stats["peak_bytes"] <= BYTE_LIMIT
    assert stats["encoded"] == stats["written"] == frames
    print(f"PoC 720p write stall={stall_seconds}s: {stats}")
