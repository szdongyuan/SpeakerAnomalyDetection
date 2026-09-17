import json
import os

import pytest

from base.video.config import VideoConfig, load_config, migrate_config, save_config


def test_missing_config_is_disabled_and_does_not_write(tmp_path):
    path = tmp_path / "configs" / "video_settings.json"
    assert load_config(path) == VideoConfig()
    assert not path.parent.exists()


def test_roundtrip_unicode_and_fixed_segment_duration(tmp_path):
    path = tmp_path / "设置" / "video_settings.json"
    config = VideoConfig(
        enabled=True, device_id="usb:stable-id", device_name="测试摄像头",
        recording_root=str(tmp_path / "录像"),
    )
    save_config(path, config)
    assert load_config(path) == config
    assert config.segment_duration_seconds == 7200
    assert not (tmp_path / "录像").exists()
    assert "device_id" in json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("changes", [
    {"schema_version": 2}, {"enabled": 1}, {"enabled": True},
    {"width": 0}, {"height": 9000}, {"fps_num": True}, {"fps_den": 0},
    {"fps_num": 1000}, {"target_bitrate_bps": -1}, {"min_free_bytes": 0},
    {"recording_root": "relative/path"}, {"segment_duration_seconds": 600},
    {"segment_duration_seconds": 18000},
    {"codec": "unknown"}, {"input_format": "unknown"}, {"device_id": 5},
])
def test_invalid_configuration_rejected(changes):
    with pytest.raises(ValueError):
        VideoConfig(**changes)


def test_replace_failure_preserves_previous_settings(tmp_path, monkeypatch):
    path = tmp_path / "video_settings.json"
    save_config(path, VideoConfig())
    previous = path.read_bytes()

    def denied(*args):
        raise PermissionError("injected replace failure")

    monkeypatch.setattr("base.video.config.os.replace", denied)
    with pytest.raises(PermissionError):
        save_config(path, VideoConfig(width=640))
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("content", [
    "[]", '{"unexpected": 1}', '{"schema_version": 9}', "bad-json",
    '{"segment_duration_seconds": 18000.0}', '{"segment_duration_seconds": "18000"}',
    '{"segment_duration_seconds": 600}',
])
def test_bad_settings_do_not_silently_fall_back(tmp_path, content):
    path = tmp_path / "video_settings.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize("duration", [None, 18000, 7200])
def test_load_segment_duration_preserves_other_settings_and_saves_two_hours(tmp_path, duration):
    path = tmp_path / "video_settings.json"
    expected = VideoConfig(
        enabled=True, device_id="stable-usb-id", fps_num=30, codec="h265",
        target_bitrate_bps=6_000_000, recording_root=str(tmp_path / "videos"),
    )
    data = dict(vars(expected))
    if duration is None:
        data.pop("segment_duration_seconds")
    else:
        data["segment_duration_seconds"] = duration
    path.write_text(json.dumps(data), encoding="utf-8")
    original = path.read_bytes()
    loaded = load_config(path)
    assert loaded == expected
    assert loaded.segment_duration_seconds == 7200
    assert path.read_bytes() == original  # Loading does not rewrite local configuration.
    save_config(path, loaded)
    assert json.loads(path.read_text(encoding="utf-8"))["segment_duration_seconds"] == 7200


def test_migrate_preserves_all_settings_and_old_backup(tmp_path):
    source = tmp_path / "configs" / "video_settings.json"
    target = tmp_path / "ui" / "ui_config" / "video_settings.json"
    config = VideoConfig(
        enabled=True, device_id="stable-usb-id", device_name="测试摄像头", fps_num=30,
        input_format="mjpeg", codec="h265", target_bitrate_bps=6_000_000,
        recording_root=str(tmp_path / "录像"),
    )
    save_config(source, config)
    legacy = json.loads(source.read_text(encoding="utf-8"))
    legacy["segment_duration_seconds"] = 18000
    source.write_text(json.dumps(legacy), encoding="utf-8")
    original = source.read_bytes()
    assert migrate_config(source, target)
    assert load_config(target) == config
    assert json.loads(target.read_text(encoding="utf-8"))["segment_duration_seconds"] == 7200
    assert source.read_bytes() == original
    assert not migrate_config(source, target)


def test_existing_destination_wins_even_if_legacy_is_invalid(tmp_path):
    source, target = tmp_path / "old.json", tmp_path / "new.json"
    source.write_text("invalid-json", encoding="utf-8")
    save_config(target, VideoConfig(fps_num=30))
    original = target.read_bytes()
    assert not migrate_config(source, target)
    assert target.read_bytes() == original


def test_missing_legacy_does_not_create_directories_or_default_settings(tmp_path):
    target = tmp_path / "ui" / "ui_config" / "video_settings.json"
    assert not migrate_config(tmp_path / "old.json", target)
    assert not target.parent.exists()


def test_invalid_legacy_is_not_copied(tmp_path):
    source, target = tmp_path / "old.json", tmp_path / "new.json"
    source.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        migrate_config(source, target)
    assert not target.exists()
    assert source.read_text(encoding="utf-8") == "[]"


def test_migration_failure_keeps_backup_and_no_partial_target(tmp_path, monkeypatch):
    source, target = tmp_path / "old.json", tmp_path / "ui" / "video.json"
    save_config(source, VideoConfig())
    original = source.read_bytes()

    def deny(*args):
        raise PermissionError("read-only-directory")

    monkeypatch.setattr("base.video.config.os." + ("rename" if os.name == "nt" else "link"), deny)
    with pytest.raises(OSError, match="迁移失败.*旧文件已保留"):
        migrate_config(source, target)
    assert source.read_bytes() == original
    assert not target.exists()
    assert not list(target.parent.iterdir())


def test_concurrent_new_settings_are_not_overwritten_by_migration(tmp_path, monkeypatch):
    source, target = tmp_path / "old.json", tmp_path / "new.json"
    save_config(source, VideoConfig(fps_num=25))
    publish = os.rename if os.name == "nt" else os.link

    def another_instance_wins(temporary, destination):
        save_config(target, VideoConfig(fps_num=30))
        publish(temporary, destination)

    monkeypatch.setattr(
        "base.video.config.os." + ("rename" if os.name == "nt" else "link"), another_instance_wins,
    )
    assert not migrate_config(source, target)
    assert load_config(target).fps_num == 30
    assert load_config(source).fps_num == 25
    assert sorted(path.name for path in tmp_path.iterdir()) == ["new.json", "old.json"]


def test_blocked_destination_directory_is_reported_not_treated_as_completed_migration(tmp_path):
    source = tmp_path / "old.json"
    save_config(source, VideoConfig())
    blocked = tmp_path / "ui_config"
    blocked.write_text("occupied", encoding="utf-8")
    with pytest.raises(OSError, match="迁移失败"):
        migrate_config(source, blocked / "video_settings.json")
    assert blocked.read_text(encoding="utf-8") == "occupied"
    assert load_config(source) == VideoConfig()
