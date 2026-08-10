import pytest

from base.hardware_trigger.serial_full_frame_matcher import (
    SerialFullFrameMatcher,
    normalize_frame_candidates,
    normalize_hex_frame,
)


FRAME_6000 = "FE 02 01 01 50 5C"
FRAME_7000 = "FE 02 01 02 91 9C"
FRAME_8000 = "FE 02 01 03 D1 9D"


def test_normalize_hex_frame_accepts_spaced_and_compact_input():
    assert normalize_hex_frame(" fe  02 01 03 d1 9d ") == FRAME_8000
    assert normalize_hex_frame("fe020103d19d") == FRAME_8000


@pytest.mark.parametrize("value", ["", "0", "GG", "FE 2", "01"])
def test_normalize_hex_frame_rejects_invalid_or_single_byte_input(value):
    with pytest.raises(ValueError):
        normalize_hex_frame(value)


def test_candidates_reject_duplicates_and_prefix_conflicts():
    with pytest.raises(ValueError, match="重复"):
        normalize_frame_candidates([FRAME_6000, FRAME_6000.lower()])
    with pytest.raises(ValueError, match="前缀冲突"):
        normalize_frame_candidates(["FE 02", FRAME_6000])


def test_matcher_extracts_a_complete_frame():
    matcher = SerialFullFrameMatcher([FRAME_6000])
    assert matcher.feed(bytes.fromhex(FRAME_6000)) == [bytes.fromhex(FRAME_6000)]
    assert matcher.buffered_bytes == b""


@pytest.mark.parametrize("split_index", range(1, 6))
def test_matcher_keeps_every_possible_half_frame_split(split_index):
    frame = bytes.fromhex(FRAME_7000)
    matcher = SerialFullFrameMatcher([FRAME_7000])

    assert matcher.feed(frame[:split_index]) == []
    assert matcher.feed(frame[split_index:]) == [frame]


def test_matcher_extracts_sticky_frames_and_preserves_transport_duplicates():
    first = bytes.fromhex(FRAME_6000)
    second = bytes.fromhex(FRAME_7000)
    matcher = SerialFullFrameMatcher([FRAME_6000, FRAME_7000])

    assert matcher.feed(first + second + second) == [first, second, second]


def test_matcher_discards_noise_and_recovers_at_the_next_known_frame():
    frame = bytes.fromhex(FRAME_8000)
    matcher = SerialFullFrameMatcher([FRAME_8000])

    assert matcher.feed(b"\x00\xFF" + frame[:3]) == []
    assert matcher.feed(frame[3:]) == [frame]
    assert matcher.discarded_byte_count == 2


def test_matcher_does_not_emit_a_corrupted_frame():
    valid = bytes.fromhex(FRAME_6000)
    corrupted = valid[:-1] + bytes([valid[-1] ^ 0xFF])
    matcher = SerialFullFrameMatcher([FRAME_6000])

    assert matcher.feed(corrupted) == []
    assert matcher.feed(valid) == [valid]


def test_replacing_candidates_clears_partial_old_frame():
    old_frame = bytes.fromhex(FRAME_6000)
    new_frame = bytes.fromhex(FRAME_7000)
    matcher = SerialFullFrameMatcher([FRAME_6000])

    assert matcher.feed(old_frame[:3]) == []
    matcher.set_frames([FRAME_7000])

    assert matcher.buffered_bytes == b""
    assert matcher.feed(old_frame[3:] + new_frame) == [new_frame]
