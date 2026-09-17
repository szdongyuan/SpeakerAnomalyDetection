from contextlib import contextmanager
from types import SimpleNamespace

from base.video.config import VideoConfig
from base.video.usb_capture import CapturePump, enumerate_cameras, open_camera, parse_devices
import pytest


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
        yield SimpleNamespace(decode=lambda **kwargs: iter([frame] * 3))

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
