"""Actual JPEG bytes exercise framing, corruption, bounds and capture continuity."""

from types import SimpleNamespace

import av
import numpy as np
import pytest

from base.video.config import VideoConfig
from base.video.mjpeg_decoder import MjpegDecoder
from base.video.usb_capture import CapturePump


@pytest.fixture
def jpeg():
    encoder = av.CodecContext.create("mjpeg", "w")
    encoder.width, encoder.height, encoder.pix_fmt = 160, 90, "yuvj420p"
    pixels = np.random.default_rng(7).integers(0, 256, (90, 160, 3), dtype=np.uint8)
    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
    return b"".join(bytes(p) for p in encoder.encode(frame) + encoder.encode(None))


def decoder(**kwargs):
    return MjpegDecoder(av.CodecContext.create("mjpeg", "r"),
                        kwargs.pop("config", VideoConfig(width=160, height=90)), **kwargs)


@pytest.mark.parametrize("split", [1, 2, 77, 2036, -2])
def test_split_jpeg_is_reassembled_without_partial_frames(jpeg, split):
    dec = decoder()
    assert dec.feed(av.Packet(jpeg[:split])) == []
    assert dec.feed(av.Packet(jpeg[split:])) == []
    frames = dec.feed(av.Packet(jpeg)) + dec.finish()
    assert len(frames) == 2 and dec.discarded == 0
    expected = av.CodecContext.create("mjpeg", "r").decode(av.Packet(jpeg))[0]
    for frame in frames:
        np.testing.assert_array_equal(frame.to_ndarray(), expected.to_ndarray())
    assert dec.pending_bytes == 0


def test_multiple_images_in_one_input_and_empty_eof_packet(jpeg):
    dec = decoder()
    frames = dec.feed(av.Packet(jpeg * 3))
    assert dec.feed(av.Packet(b"")) == []
    frames += dec.finish()
    assert len(frames) == 3 and dec.discarded == 0


@pytest.mark.parametrize("bad", [b"\xff\xd8\xff\xd9", b"garbage-no-image"])
def test_invalid_image_does_not_prevent_next_valid_image(jpeg, bad):
    dec = decoder()
    frames = dec.feed(av.Packet(bad)) + dec.feed(av.Packet(jpeg)) + dec.finish()
    assert len(frames) == 1 and dec.discarded >= 1


def test_trailing_junk_does_not_discard_complete_image(jpeg):
    dec = decoder()
    frames = dec.feed(av.Packet(jpeg + b"garbage")) + dec.feed(av.Packet(jpeg)) + dec.finish()
    assert len(frames) == 2 and dec.discarded == 1


def test_decoder_error_with_complete_markers_recovers_on_next_frame(jpeg):
    malformed = bytearray(jpeg)
    frame_header = malformed.index(b"\xff\xc0")
    malformed[frame_header + 5:frame_header + 9] = b"\x00" * 4  # Invalid dimensions.
    dec = decoder()
    frames = dec.feed(av.Packet(malformed)) + dec.feed(av.Packet(jpeg)) + dec.finish()
    assert len(frames) == 1 and dec.discarded == 1


@pytest.mark.parametrize("metadata", [b"text\xff\xd9text", b"\xff\xd8thumbnail\xff\xd9"])
def test_jpeg_markers_inside_app_metadata_are_not_frame_boundaries(jpeg, metadata):
    app = b"\xff\xe1" + (len(metadata) + 2).to_bytes(2, "big") + metadata
    annotated = jpeg[:2] + app + jpeg[2:]
    dec = decoder()
    frames = dec.feed(av.Packet(annotated)) + dec.finish()
    assert len(frames) == 1 and dec.discarded == 0
    dec = decoder()
    assert dec.feed(av.Packet(annotated[:-2])) + dec.finish() == []
    assert dec.discarded == 1


def test_truncated_image_is_not_published_when_next_image_starts(jpeg):
    dec = decoder()
    assert dec.feed(av.Packet(jpeg[:len(jpeg) // 2])) == []
    assert dec.feed(av.Packet(jpeg)) == []
    assert len(dec.finish()) == 1 and dec.discarded == 1


def test_pending_budget_resets_without_blocking_following_images(jpeg):
    dec = decoder()
    dec.max_bytes = len(jpeg) + 16
    assert dec.feed(av.Packet(jpeg[:-2])) == []
    assert dec.feed(av.Packet(jpeg)) == []
    assert dec.pending_bytes <= dec.max_bytes
    assert len(dec.finish()) == 1 and dec.discarded == 1


def test_oversized_input_never_enters_native_parser(jpeg):
    dec = decoder()
    dec.max_bytes = len(jpeg) - 1
    assert dec.feed(av.Packet(jpeg)) == []
    assert dec.pending_bytes == 0
    assert dec.finish() == [] and dec.discarded == 1


def test_stale_partial_is_replaced_by_fresh_image(jpeg):
    now = [0.0]
    dec = decoder(clock=lambda: now[0])
    dec.feed(av.Packet(jpeg[:100]))
    now[0] = 1.1
    dec.feed(av.Packet(jpeg))
    assert len(dec.finish()) == 1 and dec.discarded == 1


def test_continuous_invalid_data_eventually_requests_reconnect():
    now = [0.0]
    dec = decoder(clock=lambda: now[0])
    dec.feed(av.Packet(b"bad"))
    now[0] = 2.1
    with pytest.raises(OSError, match="持续未输出"):
        dec.feed(av.Packet(b"still-bad"))


def test_one_fps_stream_is_not_treated_as_stale(jpeg):
    now = [0.0]
    dec = decoder(clock=lambda: now[0], config=VideoConfig(width=160, height=90, fps_num=1))
    dec.feed(av.Packet(jpeg))
    now[0] = 1.1
    assert len(dec.feed(av.Packet(jpeg))) == 1
    assert dec.discarded == 0


def test_incomplete_final_image_is_not_flushed_as_a_frame(jpeg):
    dec = decoder()
    dec.feed(av.Packet(jpeg[:-2]))
    assert dec.finish() == [] and dec.discarded == 1


def test_split_input_keeps_same_camera_connection_and_stop_closes_it(jpeg, tmp_path):
    config = VideoConfig(width=160, height=90, recording_root=str(tmp_path))
    opens, connections, frames = [], [], []

    class Source:
        closed = False

        def __init__(self):
            codec = av.CodecContext.create("mjpeg", "r")
            codec.width, codec.height = config.width, config.height
            self.streams = SimpleNamespace(video=[SimpleNamespace(
                codec_context=codec, average_rate="30", time_base="1/30")])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def demux(self, **kwargs):
            yield from map(av.Packet, (jpeg[:2036], jpeg[2036:], jpeg))
            pytest.fail("stop must not read another packet")

    source = Source()

    def opener(config):
        opens.append(True)
        assert len(opens) == 1, "fragmentation must not reopen the device"
        return source

    def on_frame(frame, stamp):
        frames.append(frame)
        pump.stop()

    pump = CapturePump(config, on_frame, lambda ready, detail: connections.append(ready), opener=opener)
    pump._run()
    assert connections == [True] and len(frames) == 1
    assert source.closed and pump.diagnostics.attempt == 1
    assert not list(tmp_path.rglob("result.json"))
