"""Versioned USB video settings, independent of audio/product configuration."""

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoConfig:
    schema_version: int = 1
    enabled: bool = False
    device_id: str = ""
    device_name: str = ""
    width: int = 1280
    height: int = 720
    fps_num: int = 25
    fps_den: int = 1
    input_format: str = "auto"

    def __post_init__(self):
        integer_fields = (
            "schema_version", "width", "height", "fps_num", "fps_den",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"视频配置项“{name}”必须为正整数。")
        for name in ("device_id", "device_name", "input_format"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"视频配置项“{name}”必须为文本。")
        if type(self.enabled) is not bool:
            raise ValueError("摄像头启用状态必须为开启或关闭。")
        if self.schema_version != 1:
            raise ValueError("视频配置版本不受支持，请检查配置文件。")
        if self.width > 7680 or self.height > 4320:
            raise ValueError("分辨率不能超过7680×4320。")
        if self.fps_num > 240 * self.fps_den:
            raise ValueError("帧率不能超过240帧/秒。")
        if self.input_format not in {"auto", "mjpeg", "yuy2", "nv12"}:
            raise ValueError("摄像头输入格式不受支持。")
        if self.enabled:
            self.validate_capture()

    @property
    def preview_size(self):
        """Bound the preview buffer by capture size so it never enlarges pixels."""
        return min(640, self.width), min(360, self.height)

    def validate_capture(self):
        """Validate the device selected for live preview."""
        if not self.device_id.strip():
            raise ValueError("请选择摄像头；如果列表为空，请连接USB摄像头后点击“刷新”。")


def load_config(path):
    path = Path(path)
    if not path.exists():
        return VideoConfig()
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("视频配置文件格式无效，内容必须为JSON对象。")
    # Legacy files contain both camera preferences and video output settings.
    for field in ("codec", "target_bitrate_bps", "recording_root", "segment_duration_seconds",
                  "min_free_bytes", "preview_width", "preview_height"):
        data.pop(field, None)
    try:
        return VideoConfig(**data)
    except TypeError as exc:
        raise ValueError("视频配置包含无法识别的设置项，请检查配置文件。") from exc


def migrate_config(source, destination):
    """Copy validated legacy settings once; never remove source or overwrite destination."""
    source, destination = Path(source), Path(destination)
    if destination.exists() or not source.exists():
        return False
    config = load_config(source)
    try:
        save_config(destination, config, overwrite=False)
    except OSError as exc:
        if isinstance(exc, FileExistsError) and destination.exists():
            return False  # Another instance installed settings while migration was preparing.
        raise OSError(f"视频配置迁移失败，旧文件已保留；请检查新配置目录是否可写：{destination.parent}") from exc
    return True


def save_config(path, config, *, overwrite=True):
    """Replace atomically; a failed write must not truncate the previous settings."""
    if not isinstance(config, VideoConfig):
        raise TypeError("视频配置对象类型无效。")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(asdict(config), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary_path, path)
        elif os.name == "nt":
            # Windows rename atomically publishes the file but rejects an existing target.
            os.rename(temporary_path, path)
        else:
            os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
