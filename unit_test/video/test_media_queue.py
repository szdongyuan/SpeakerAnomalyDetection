import queue
import threading

import pytest

from base.video.recording import MediaQueue, MediaQueueClosed


def test_inflight_bytes_packets_and_control_reserve():
    channel = MediaQueue(byte_limit=10, packet_limit=2, control_limit=1)
    channel.put_nowait("a", size=10, packets=2)
    item = channel.get(timeout=.1)
    with pytest.raises(queue.Full):
        channel.put_nowait("b", size=1, packets=1)
    channel.put_nowait("stop")
    with pytest.raises(queue.Full):
        channel.put_nowait("duplicate-stop")
    channel.release(item)
    channel.put_nowait("b", size=1, packets=1)
    end = channel.get(timeout=.1)
    assert end.message == "stop"
    channel.release(end)
    last = channel.get(timeout=.1)
    assert last.message == "b"
    channel.release(last)
    assert channel.snapshot()["bytes"] == 0
    with pytest.raises(ValueError):
        channel.release(last)


def test_close_wakes_waiter_and_discard_does_not_release_inflight():
    channel = MediaQueue(byte_limit=10)
    channel.put_nowait("active", size=4, packets=1)
    active = channel.get(timeout=.1)
    channel.put_nowait("queued", size=6, packets=1)
    channel.close(discard=True)
    assert channel.snapshot()["bytes"] == 4
    channel.release(active)
    assert channel.snapshot()["bytes"] == 0
    with pytest.raises(MediaQueueClosed):
        channel.put_nowait("new")
    stopped = threading.Event()

    def wait():
        with pytest.raises(MediaQueueClosed):
            channel.get()
        stopped.set()

    thread = threading.Thread(target=wait)
    thread.start()
    thread.join(1)
    assert stopped.is_set()


def test_boundary_rejections_do_not_charge_budget():
    channel = MediaQueue(byte_limit=10, packet_limit=2)
    for size, packets in [(11, 1), (1, 3)]:
        with pytest.raises(queue.Full):
            channel.put_nowait("large", size=size, packets=packets)
    assert channel.snapshot()["bytes"] == 0
    channel.put_nowait("exact", size=10, packets=2)
    channel.close()
    item = channel.get()
    assert item.message == "exact"
    channel.release(item)
    with pytest.raises(MediaQueueClosed):
        channel.get()
