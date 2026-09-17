import multiprocessing

import pytest

from base.video.preview import PreviewMailbox


def test_latest_frame_only_fixed_capacity_and_owned_copy():
    mailbox = PreviewMailbox(multiprocessing.get_context("spawn"), 8, 4)
    assert mailbox.latest() is None
    for index in range(100):
        assert mailbox.publish(bytes([index]) * 96)
    frame = mailbox.latest()
    assert frame.sequence == 100
    assert frame.rgb == bytes([99]) * 96
    assert mailbox.latest(frame.sequence) is None
    mailbox.publish(bytes([200]) * 96)
    assert frame.rgb == bytes([99]) * 96
    assert mailbox.capacity == 96


def test_busy_mailbox_never_blocks_reader_or_writer():
    mailbox = PreviewMailbox(multiprocessing.get_context("spawn"), 2, 2)
    mailbox._lock.acquire()
    try:
        assert mailbox.latest() is None
        assert not mailbox.publish(bytes(12))
    finally:
        mailbox._lock.release()
    with pytest.raises(ValueError):
        mailbox.publish(bytes(10))
