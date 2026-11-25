import re
from collections.abc import Iterable


_HEX_BYTE_PATTERN = re.compile(r"^[0-9A-Fa-f]{2}$")


def normalize_hex_frame(value, *, minimum_bytes=2):
    """Return a canonical upper-case, space-separated hexadecimal frame."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        frame_bytes = bytes(value)
        tokens = [f"{byte:02X}" for byte in frame_bytes]
    else:
        text = str(value or "").strip()
        if not text:
            raise ValueError("完整状态报文不能为空")

        tokens = text.split()
        if len(tokens) == 1 and len(tokens[0]) > 2:
            compact = tokens[0]
            if len(compact) % 2:
                raise ValueError("完整状态报文必须由成对的十六进制字符组成")
            tokens = [compact[index : index + 2] for index in range(0, len(compact), 2)]

        if not tokens or any(_HEX_BYTE_PATTERN.fullmatch(token) is None for token in tokens):
            raise ValueError("完整状态报文必须使用两位十六进制字节，例如 FE 02 01 03 D1 9D")

    if len(tokens) < int(minimum_bytes):
        raise ValueError(f"完整状态报文至少需要 {int(minimum_bytes)} 个字节")
    return " ".join(token.upper() for token in tokens)


def normalize_frame_candidates(values: Iterable):
    normalized = []
    seen = set()
    for value in values or ():
        frame = normalize_hex_frame(value)
        if frame in seen:
            raise ValueError(f"完整状态报文重复：{frame}")
        seen.add(frame)
        normalized.append(frame)

    frame_bytes = [bytes.fromhex(frame) for frame in normalized]
    for index, frame in enumerate(frame_bytes):
        for other_index, other in enumerate(frame_bytes):
            if index == other_index:
                continue
            if other.startswith(frame):
                raise ValueError(
                    f"完整状态报文存在前缀冲突：{normalized[index]} / {normalized[other_index]}"
                )
    return tuple(normalized)


class SerialFullFrameMatcher:
    """Extract configured full frames from an arbitrary serial byte stream."""

    def __init__(self, frames=()):
        self._buffer = bytearray()
        self.discarded_byte_count = 0
        self.set_frames(frames)

    @property
    def frames(self):
        return self._frame_hexes

    @property
    def buffered_bytes(self):
        return bytes(self._buffer)

    def set_frames(self, frames):
        frame_hexes = normalize_frame_candidates(frames)
        self._frame_hexes = frame_hexes
        self._frames = tuple(bytes.fromhex(frame) for frame in frame_hexes)
        self._max_frame_length = max((len(frame) for frame in self._frames), default=0)
        self.clear()

    def clear(self):
        self._buffer.clear()
        self._read_boundaries = []

    def feed(self, data):
        return [frame for frame, _, _ in self.feed_with_timestamps(data, None, None)]

    def feed_with_timestamps(self, data, received_monotonic, received_perf_ns):
        """Return frames with the read timestamps of their final bytes."""
        chunk = bytes(data or b"")
        if not chunk:
            return []

        if not self._frames:
            self.discarded_byte_count += len(chunk)
            return []

        self._buffer.extend(chunk)
        self._read_boundaries.append((len(self._buffer), received_monotonic, received_perf_ns))
        read_index = 0
        matched = []
        offset = 0
        while offset < len(self._buffer):
            frame = next(
                (candidate for candidate in self._frames if self._buffer.startswith(candidate, offset)),
                None,
            )
            if frame is not None:
                offset += len(frame)
                while self._read_boundaries[read_index][0] < offset:
                    read_index += 1
                _, final_monotonic, final_perf_ns = self._read_boundaries[read_index]
                matched.append((frame, final_monotonic, final_perf_ns))
                continue

            if len(self._buffer) - offset < self._max_frame_length:
                # A short frame inside this prefix may be payload of the earlier frame.
                suffix = self._buffer[offset:]
                if any(candidate.startswith(suffix) for candidate in self._frames):
                    break
            self.discarded_byte_count += 1
            offset += 1

        del self._buffer[:offset]
        if offset:
            self._read_boundaries = [
                (end - offset, monotonic, perf_ns)
                for end, monotonic, perf_ns in self._read_boundaries[read_index:]
                if end > offset
            ]
        return matched
