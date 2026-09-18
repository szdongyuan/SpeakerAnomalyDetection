"""Bounded MJPEG framing before decode; USB samples need not be whole images."""

import logging
import time

logger = logging.getLogger("core.video")


def _jpeg_end(data):
    """Locate EOI outside length-delimited headers and stuffed scan bytes."""
    position, in_scan, saw_scan = 2, False, False
    while position < len(data):
        if in_scan:
            position = data.find(b"\xff", position)
            if position < 0:
                return None
        elif data[position] != 0xff:
            return None
        while position < len(data) and data[position] == 0xff:
            position += 1
        if position >= len(data):
            return None
        marker = data[position]
        position += 1
        if in_scan and (marker == 0 or 0xd0 <= marker <= 0xd7):
            continue
        if marker == 0xd9:
            return position if saw_scan else None
        if marker == 0x01:  # Standalone TEM marker.
            continue
        if marker == 0xd8 or position + 2 > len(data):
            return None
        size = int.from_bytes(data[position:position + 2], "big")
        if size < 2 or position + size > len(data):
            return None
        position += size
        in_scan = marker == 0xda or (in_scan and marker == 0xdc)
        saw_scan |= marker == 0xda
    return None


class MjpegDecoder:
    def __init__(self, codec, config, *, clock=time.monotonic):
        import av

        self.codec = codec
        self.clock = clock
        self.max_bytes = max(4 * 1024**2, config.width * config.height * 3)
        self.pending_timeout = max(1.0, 2 * config.fps_den / config.fps_num)
        self.invalid_timeout = max(2.0, 3 * config.fps_den / config.fps_num)
        self.parser = av.CodecContext.create("mjpeg", "r")
        self.pending_bytes = 0
        self.pending_since = None
        self.last_valid = clock()
        self.fragmented_inputs = 0
        self.discarded = 0
        self.last_log = float("-inf")
        self.logged_counts = (0, 0)

    def _reset_parser(self):
        import av

        self.parser = av.CodecContext.create("mjpeg", "r")
        self.pending_bytes = 0
        self.pending_since = None

    def _log(self, now, *, force=False):
        counts = (self.fragmented_inputs, self.discarded)
        if counts != self.logged_counts and (force or now - self.last_log >= 5):
            logger.log(logging.WARNING if self.discarded > self.logged_counts[1] else logging.INFO,
                       "Video MJPEG framing: fragmented_inputs=%s discarded_units=%s pending_bytes=%s "
                       "byte_limit=%s; isolated input damage does not reconnect camera",
                       self.fragmented_inputs, self.discarded, self.pending_bytes, self.max_bytes)
            self.logged_counts, self.last_log = counts, now

    def _decode_packets(self, packets, now):
        import av

        frames = []
        for packet in packets:
            data = bytes(packet)
            self.pending_bytes = max(0, self.pending_bytes - len(data))
            # Native parsing may retain leading/trailing junk, or emit an
            # incomplete image at the next SOI. Never publish a concealed half-frame.
            start = data.find(b"\xff\xd8")
            end = _jpeg_end(data[start:]) if start >= 0 else None
            if end is None:
                self.discarded += 1
                continue
            if start or start + end != len(data):
                self.discarded += 1
                packet = av.Packet(data[start:start + end])
            try:
                decoded = self.codec.decode(packet)
            except av.error.InvalidDataError:
                self.discarded += 1
                self.codec.flush_buffers()
                continue
            for frame in decoded:
                if frame.is_corrupt:
                    self.discarded += 1
                else:
                    frames.append(frame)
        if frames:
            self.last_valid = now
        return frames

    def feed(self, packet):
        now = self.clock()
        data = bytes(packet)
        if data and (not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9")):
            self.fragmented_inputs += 1
        if self.pending_bytes and (
            self.pending_bytes + len(data) > self.max_bytes
            or now - self.pending_since > self.pending_timeout
        ):
            self.discarded += 1
            self._reset_parser()
        if len(data) > self.max_bytes:
            self.discarded += 1
            frames = []
        else:
            if data and not self.pending_bytes:
                self.pending_since = now
            self.pending_bytes += len(data)
            packets = self.parser.parse(data) if data else []
            frames = self._decode_packets(packets, now)
            if packets:
                self.pending_since = now if self.pending_bytes else None
        self._log(now)
        if now - self.last_valid > self.invalid_timeout:
            self._log(now, force=True)
            raise OSError("摄像头持续未输出可解码的完整MJPEG图像")
        return frames

    def finish(self):
        now = self.clock()
        frames = self._decode_packets(self.parser.parse(b""), now)
        self._reset_parser()
        self._log(now, force=True)
        return frames
