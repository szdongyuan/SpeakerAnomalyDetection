"""Windows DirectShow capture by device identity, never by a persistent index."""

from dataclasses import dataclass
import re
import threading
import time

from base.video.capture_diagnostics import CaptureDiagnostics
from base.video.mjpeg_decoder import MjpegDecoder


@dataclass(frozen=True)
class CameraDevice:
    device_id: str
    name: str


def parse_devices(logs):
    devices = []
    name = None
    # FFmpeg can emit a single device line across several log callbacks.
    # Preserve its own newlines instead of treating callbacks as complete lines.
    listing = "".join(message for _, component, message in logs if component == "dshow")
    for line in listing.splitlines():
        match = re.search(r'^"(.*)" \(video\)', line.strip())
        if match:
            name = match.group(1)
        elif re.search(r'^".*" \(audio\)', line.strip()):
            name = None
        alternative = re.search(r'Alternative name "(.*)"', line)
        if alternative and name is not None:
            devices.append(CameraDevice(alternative.group(1), name))
            name = None
    return tuple(devices)


def enumerate_cameras():
    import av

    av.logging.set_level(av.logging.INFO)
    with av.logging.Capture(local=True) as logs:
        try:
            av.open("dummy", format="dshow", options={"list_devices": "true"})
        except av.error.ExitError:
            pass  # DirectShow deliberately exits after enumerating its devices.
    return parse_devices(logs)


def probe_worker(channel):
    try:
        channel.send((enumerate_cameras(), ""))
    except Exception as exc:
        channel.send(((), str(exc)))
    finally:
        channel.close()


def open_camera(config):
    import av

    matches = [device for device in enumerate_cameras() if device.device_id == config.device_id]
    if len(matches) != 1:
        raise OSError("指定USB摄像头未找到或身份不唯一，请检查连接和设置")
    options = {
        "video_size": f"{config.width}x{config.height}",
        "framerate": f"{config.fps_num}/{config.fps_den}",
        "rtbufsize": str(32 * 1024**2),
    }
    if config.input_format == "mjpeg":
        options["vcodec"] = "mjpeg"
    elif config.input_format != "auto":
        options["pixel_format"] = {"yuy2": "yuyv422", "nv12": "nv12"}[config.input_format]
    return av.open(f"video={config.device_id}", format="dshow", options=options, timeout=(4, 4))


class CapturePump:
    def __init__(self, config, on_frame, on_connection, *, opener=open_camera):
        self.config = config
        self.on_frame = on_frame
        self.on_connection = on_connection
        self.opener = opener
        self.stop_event = threading.Event()
        self.last_progress = time.monotonic()
        self.frames = 0
        self.diagnostics = CaptureDiagnostics(config)
        self.thread = threading.Thread(target=self._run, name="VideoCapture", daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        # The owning thread closes the device; never close an AV container concurrently.
        self.stop_event.set()

    def _run(self):
        import av

        last_error = ""
        while not self.stop_event.is_set():
            self.diagnostics.begin_attempt()
            try:
                self.last_progress = time.monotonic()
                with av.logging.Capture(local=True) as logs:
                    try:
                        source = self.opener(self.config)
                    finally:
                        self.diagnostics.add_logs(logs)
                with source as source:
                    self.diagnostics.opened(source)
                    ready = False
                    for frame in self._decode(source):
                        if self.stop_event.is_set():
                            return
                        self.diagnostics.stage = "frame_dispatch"
                        if (frame.width, frame.height) != (self.config.width, self.config.height):
                            raise ValueError("摄像头实际分辨率与设置不一致，拒绝静默降级")
                        # On Windows/Python 3.12 monotonic may have ~15 ms resolution.
                        # Keep media timestamps precise, but watchdogs in their own clock domain.
                        stamp = time.perf_counter()
                        self.last_progress = time.monotonic()
                        if not ready:
                            self.diagnostics.first_frame()
                            self.on_connection(True, "")
                            ready, last_error = True, ""
                        self.frames += 1
                        self.on_frame(frame, stamp)
                    if not self.stop_event.is_set():
                        self.diagnostics.stage = "end_of_stream"
                        raise OSError("摄像头停止输出画面")
            except (OSError, ValueError, av.error.FFmpegError) as exc:
                message = str(exc)[:1000]
                if message != last_error:
                    self.on_connection(False, message)
                    last_error = message
                self.last_progress = time.monotonic()
                self.diagnostics.failed(exc)
            self.last_progress = time.monotonic()
            if not self.stop_event.is_set():
                self.diagnostics.retry()
            self.stop_event.wait(2)

    def _decode(self, source):
        """Preserve input evidence and reassemble MJPEG before decoding images."""
        import av

        codec = source.streams.video[0].codec_context
        mjpeg = MjpegDecoder(codec, self.config) if codec.name == "mjpeg" else None
        packets = iter(source.demux(video=0))
        while not self.stop_event.is_set():
            self.diagnostics.stage = "demux"
            with av.logging.Capture(local=True) as logs:
                try:
                    packet = next(packets, None)
                finally:
                    self.diagnostics.add_logs(logs)
            if packet is None:
                if mjpeg is not None and not self.stop_event.is_set():
                    self.diagnostics.stage = "decode"
                    with av.logging.Capture(local=True) as logs:
                        try:
                            frames = mjpeg.finish()
                        finally:
                            self.diagnostics.add_logs(logs)
                    yield from frames
                return
            self.diagnostics.remember(packet)
            self.diagnostics.stage = "decode"
            with av.logging.Capture(local=True) as logs:
                try:
                    frames = mjpeg.feed(packet) if mjpeg is not None else packet.decode()
                finally:
                    self.diagnostics.add_logs(logs)
            yield from frames
