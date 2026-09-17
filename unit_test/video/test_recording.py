import logging
import multiprocessing
import os
from pathlib import Path
from datetime import datetime, timedelta

import av
import numpy as np
import pytest

from base.video.config import VideoConfig
from base.video.recording import (
    EncodedFrame, FrameEncoder, RecordingRootLease, RecordingSession, VideoDiagnostics, TimedVideoFile,
)


def config_for(root, **changes):
    return VideoConfig(
        enabled=True, device_id="test-device", width=160, height=90, fps_num=10,
        recording_root=str(root), min_free_bytes=1, **changes,
    )


def frame(index):
    pixels = np.full((90, 160, 3), index * 8 % 240, dtype=np.uint8)
    return av.VideoFrame.from_ndarray(pixels, format="rgb24")


def write_segment(session, samples):
    """Feed one explicit segment through the production parameter/packet interface."""
    encoder = FrameEncoder(session.config, session.diagnostics)
    session.next_segment(encoder.parameters)
    for index, timestamp in samples:
        session.write(encoder.write(frame(index), timestamp), timestamp)
    session.write_tail(encoder.finish())


def decoded(path):
    with av.open(str(path)) as container:
        return [(float(item.time), float(item.to_ndarray(format="rgb24").mean())) for item in container.decode(video=0)]


def test_diagnostics_preserves_inflight_overflow_across_next_session(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("base.video.recording.time.perf_counter", lambda: clock[0])
    diagnostics = VideoDiagnostics()
    diagnostics.begin("old")
    with diagnostics.stage("frame_total"):
        with diagnostics.stage("disk_sync"):
            clock[0] += 2.3
            diagnostics.overflow(59, 60)  # One frame is still being processed.
    diagnostics.begin("new")
    report = diagnostics.take_report(0, 60)
    assert report["session"] == "old"
    assert report["active_stage"] == "disk_sync"
    assert report["active_ms"] == 2300
    assert report["queue_items"] == 59
    assert diagnostics.take_report(0, 60) is None


def test_diagnostics_nested_metrics_preserve_exceptions_and_rate_limit(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("base.video.recording.time.perf_counter", lambda: clock[0])
    diagnostics = VideoDiagnostics()
    diagnostics.begin("timings")
    with diagnostics.stage("frame_total"):
        with diagnostics.stage("encode"):
            clock[0] += .01
        assert diagnostics.take_report(0, 60) is None
        with pytest.raises(OSError, match="write failed"):
            with diagnostics.stage("mux_write"):
                clock[0] += .3
                raise OSError("write failed")
        report = diagnostics.take_report(1, 60)
        assert report["active_stage"] == "frame_total"
        assert report["stages"]["encode"] == {"count": 1, "mean_ms": 10, "max_ms": 10}
        assert report["stages"]["mux_write"] == {"count": 1, "mean_ms": 300, "max_ms": 300}
    assert diagnostics.take_report(48, 60) is None
    clock[0] += 29.99
    assert diagnostics.take_report(48, 60) is None
    clock[0] += .02
    assert diagnostics.take_report(48, 60)["active_stage"] == "idle"
    # Queue-full evidence bypasses the ordinary 30-second throttle exactly once.
    diagnostics.overflow(60, 60)
    assert diagnostics.take_report(60, 60)["reason"] == "queue_full"
    assert diagnostics.take_report(60, 60) is None


def test_diagnostics_arrival_history_is_bounded_and_normal_path_is_quiet(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("base.video.recording.time.perf_counter", lambda: clock[0])
    diagnostics = VideoDiagnostics()
    diagnostics.begin("burst")
    for _ in range(500):
        diagnostics.arrival()
        with diagnostics.stage("encode"):
            clock[0] += .0001
    assert diagnostics.take_report(0, 60) is None
    diagnostics.overflow(60, 60)
    report = diagnostics.take_report(60, 60)
    assert report["arrivals_last_second"] == report["arrival_sample_limit"] == 120
    assert report["stages"]["encode"]["count"] == 500
    assert len(diagnostics._arrivals) == 120
    assert len(diagnostics._stats) == 1
    clock[0] += 2
    diagnostics.overflow(60, 60)
    assert diagnostics.take_report(60, 60)["arrivals_last_second"] == 0


def test_real_encoder_diagnostics_cover_stages_without_changing_frames(tmp_path):
    session = RecordingSession(config_for(tmp_path), "metrics")
    write_segment(session, [(index, 100 + index / 10) for index in range(15)])
    session.finish()
    session.diagnostics.overflow(0, 60)
    stats = session.diagnostics.take_report(0, 60)["stages"]
    for name in ("pixel_convert", "encode", "mux_write"):
        assert stats[name]["count"] == 15
        assert stats[name]["max_ms"] >= stats[name]["mean_ms"] >= 0
    for name in ("open_encoder", "open_segment", "check_space", "disk_sync", "encoder_flush",
                 "mux_close", "file_write", "file_close", "rename_media"):
        assert stats[name]["count"] >= 1
    assert stats["open_encoder"]["count"] == 1
    assert stats["open_segment"]["count"] == 1
    assert "validate_media" not in stats
    assert len(decoded(next(session.directory.glob("*.mp4")))) == 15


def test_file_boundary_completes_short_writes_and_does_not_count_failed_progress():
    class ShortFile:
        data = b""
        fail = False

        def write(self, data):
            if self.fail:
                return 0
            self.data += bytes(data[:3])
            return min(3, len(data))

    diagnostics = VideoDiagnostics()
    raw = ShortFile()
    file = TimedVideoFile(raw, diagnostics)
    assert file.write(b"12345678") == 8
    assert raw.data == b"12345678"
    progress = diagnostics.progress_snapshot()[0]
    raw.fail = True
    with pytest.raises(OSError, match="未取得进展"):
        file.write(b"9")
    assert diagnostics.progress_snapshot()[0] == progress


def test_session_never_writes_json_even_when_replace_is_denied(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("JSON replacement denied")

    monkeypatch.setattr(os, "replace", denied)
    session = RecordingSession(config_for(tmp_path), "no-json")
    write_segment(session, [(1, 100)])
    session._last_sync = 0
    session.maintain()
    session.finish()
    assert not list(tmp_path.rglob("*.json"))
    assert len(decoded(next(session.directory.glob("*.mp4")))) == 1


def test_progress_log_every_ten_minutes_without_catchup_burst(tmp_path, monkeypatch, caplog):
    clock = [1000.0]
    monkeypatch.setattr("base.video.recording.time.monotonic", lambda: clock[0])
    caplog.set_level(logging.INFO, logger="core.video")
    session = RecordingSession(config_for(tmp_path), "log-interval")
    write_segment(session, [(i, 100 + i / 10) for i in range(15)])
    assert session.segment_writer.has_written_media
    try:
        def progress():
            return [r for r in caplog.records if "Video recording progress:" in r.message]

        for now in (1001, 1060, 1599.999):
            clock[0] = now
            session.maintain()
        assert not progress()
        clock[0] = 1600
        session.maintain()
        assert len(progress()) == 1
        assert "frames=15" in progress()[0].message
        assert "elapsed_seconds=1.400" in progress()[0].message
        assert f"bytes={session.segment_writer.path.stat().st_size}" in progress()[0].message
        session.maintain()
        assert len(progress()) == 1
        clock[0] = 3400
        session.maintain()
        assert len(progress()) == 2
    finally:
        session.finish()
    clock[0] += 600
    session.maintain()
    assert len(progress()) == 2


def test_real_h264_fragments_split_without_missing_or_duplicate_frames(tmp_path):
    session = RecordingSession(config_for(tmp_path), "test-1")
    for start, end in [(0, 10), (10, 20), (20, 25)]:
        write_segment(session, [(i, 100 + i / 10) for i in range(start, end)])
    session.finish()
    files = sorted(session.directory.glob("*.mp4"))
    assert len(files) == 3
    videos = [decoded(path) for path in files]
    assert [len(video) for video in videos] == [10, 10, 5]
    values = [value for video in videos for _, value in video]
    assert all(abs(value - index * 8) <= 4 for index, value in enumerate(values))
    for path, video in zip(files, videos):
        assert b"moof" in path.read_bytes()
        assert video[0][0] == 0
        assert all(a[0] < b[0] for a, b in zip(video, video[1:]))
    assert session.summary["state"] == "completed"
    assert session.summary["frames"] == 25
    assert session.summary["segments"] == 3
    assert not (session.directory / "session.json").exists()
    assert not (session.directory / "events.jsonl").exists()
    assert not list(session.directory.glob("*.recording.mp4"))


def test_finished_segment_stop_produces_no_empty_tail(tmp_path):
    session = RecordingSession(config_for(tmp_path), "exact")
    write_segment(session, [(i, i / 10) for i in range(10)])
    session.finish()
    session.finish()
    assert len(list(session.directory.glob("*.mp4"))) == 1


def test_media_sync_stays_one_second_and_sync_errors_are_not_hidden(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("base.video.recording.time.monotonic", lambda: clock[0])
    session = RecordingSession(config_for(tmp_path), "sync")
    write_segment(session, [(1, 100)])
    calls = []
    real_sync = session.segment_writer.sync

    def sync():
        calls.append(clock[0])
        real_sync()

    monkeypatch.setattr(session.segment_writer, "sync", sync)
    clock[0] = 1000.99
    session.maintain()
    assert calls == []
    clock[0] = 1001
    session.maintain()
    assert calls == [1001]

    def denied():
        raise PermissionError("MP4 sync denied")

    monkeypatch.setattr(session.segment_writer, "sync", denied)
    clock[0] = 1002
    try:
        with pytest.raises(PermissionError, match="MP4 sync denied"):
            session.maintain()
    finally:
        monkeypatch.setattr(session.segment_writer, "sync", real_sync)
        session.finish()


def test_empty_session_has_no_periodic_log_and_stop_logs_final_counts(tmp_path, monkeypatch, caplog):
    clock = [1000.0]
    monkeypatch.setattr("base.video.recording.time.monotonic", lambda: clock[0])
    caplog.set_level(logging.INFO, logger="core.video")
    session = RecordingSession(config_for(tmp_path), "idle")
    clock[0] += 3600
    session.maintain()
    session.finish()
    session.finish()
    assert "Video recording progress:" not in caplog.text
    assert "Video recording started:" not in caplog.text
    assert caplog.text.count("Video session ended:") == 1
    assert "frames=0 segments=0 gaps=0" in caplog.text
    assert not list(session.directory.iterdir())


def test_variable_capture_timestamps_preserved(tmp_path):
    session = RecordingSession(config_for(tmp_path), "vfr")
    write_segment(session, enumerate([20, 20.1, 20.5, 20.6]))
    session.finish()
    video = decoded(next(session.directory.glob("*.mp4")))
    assert [stamp for stamp, _ in video] == pytest.approx([0, .1, .5, .6], abs=.0001)


@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_sub_tick_frame_intervals_preserve_every_frame_and_increasing_pts(tmp_path, codec):
    session = RecordingSession(config_for(tmp_path, codec=codec), "burst")
    stamps = [100.0, 100.000001, 100.000002, 100.1]
    try:
        write_segment(session, enumerate(stamps))
        session.finish()
        video = decoded(next(session.directory.glob("*.mp4")))
        assert len(video) == len(stamps)
        assert [stamp for stamp, _ in video] == pytest.approx([0, 1 / 90_000, 2 / 90_000, .1], abs=1e-7)
        assert session.summary["elapsed_seconds"] == pytest.approx(.1)
    finally:
        if not session.closed:
            session.finish()


@pytest.mark.parametrize("bad_stamp", [9.9, 10.0])
@pytest.mark.parametrize("stage", ["encoder", "writer"])
def test_nonincreasing_capture_time_is_still_rejected_in_chinese(tmp_path, bad_stamp, stage):
    session = RecordingSession(config_for(tmp_path), "invalid-time")
    encoder = FrameEncoder(session.config)
    session.prepare(encoder.parameters)
    session.write(encoder.write(frame(1), 10.0), 10.0)
    try:
        with pytest.raises(ValueError, match="采集时间戳未递增"):
            if stage == "encoder":
                encoder.write(frame(2), bad_stamp)
            else:
                session.write(encoder.write(frame(2), 10.1), bad_stamp)
        assert session.summary["frames"] == 1
    finally:
        session.write_tail(encoder.finish())
        session.finish()


@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_explicit_two_hour_segments_and_decoder(tmp_path, codec):
    session = RecordingSession(config_for(tmp_path, codec=codec), "two-hour-boundary")
    # The runtime owns the timed boundary; the writer follows explicit parameters.
    write_segment(session, [(0, 0), (1, 7_199.9)])
    write_segment(session, [(2, 7_200), (3, 7_200.1)])
    session.finish()
    files = sorted(session.directory.glob("*.mp4"))
    assert [len(decoded(path)) for path in files] == [2, 2]


def test_stop_before_frames_creates_no_fake_video(tmp_path):
    session = RecordingSession(config_for(tmp_path), "empty")
    session.finish()
    assert not list(session.directory.glob("*.mp4"))
    assert session.summary["frames"] == 0


def test_prepare_opens_file_without_starting_media_and_empty_stop_is_clean(tmp_path):
    session = RecordingSession(config_for(tmp_path), "prepared")
    frame_encoder = FrameEncoder(session.config)
    session.prepare(frame_encoder.parameters)
    segment_writer = session.segment_writer
    assert segment_writer is not None
    assert segment_writer.first_time is None
    assert session.summary["state"] == "starting"
    session.prepare(frame_encoder.parameters)
    assert session.segment_writer is segment_writer
    session.write_tail(frame_encoder.finish())
    session.finish()
    assert session.summary["state"] == "completed"
    assert session.summary["segments"] == 0
    assert session.summary["current_segment"] is None
    assert not list(session.directory.glob("*.mp4"))


def test_writer_saves_external_packets_without_creating_an_encoder(tmp_path, monkeypatch):
    session = RecordingSession(config_for(tmp_path), "external-encoder")
    encoder = FrameEncoder(session.config)
    batches = [encoder.write(frame(i), 10 + i / 10) for i in range(3)]
    tail = encoder.finish()

    def unexpected_encoder(*args, **kwargs):
        pytest.fail("the file writer must not create a frame encoder")

    monkeypatch.setattr("base.video.recording.FrameEncoder", unexpected_encoder)
    try:
        session.prepare(encoder.parameters)
        for i, batch in enumerate(batches):
            session.write(batch, 10 + i / 10)
        session.write_tail(tail)
    finally:
        session.finish()
    video = decoded(next(session.directory.glob("*.mp4")))
    assert len(video) == session.summary["frames"] == 3
    assert [stamp for stamp, _ in video] == pytest.approx([0, .1, .2])


def test_compressed_frame_without_file_header_is_rejected(tmp_path):
    session = RecordingSession(config_for(tmp_path), "missing-header")
    try:
        with pytest.raises(RuntimeError, match="压缩帧缺少对应的录像文件头"):
            session.write(EncodedFrame(()), 10)
        assert session.summary["frames"] == 0
        assert not list(session.directory.iterdir())
    finally:
        session.finish()


def test_gap_closes_segment_and_marks_session_interrupted(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="core.video")
    session = RecordingSession(config_for(tmp_path), "gap")
    write_segment(session, [(i, 10 + i / 10) for i in range(15)])
    session.gap("USB disconnected")
    write_segment(session, [(i, 15 + i / 10) for i in range(15)])
    session.finish()
    assert session.summary["state"] == "interrupted"
    assert session.summary["segments"] == 2
    assert session.summary["gaps"] == 1
    assert "USB disconnected" in caplog.text
    assert "Video capture resumed:" in caplog.text


def test_readable_names_same_second_collision_and_cross_day_session(tmp_path, monkeypatch):
    stamp = datetime(2026, 9, 9, 10, 23, 30).astimezone()

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return stamp if tz is None else stamp.astimezone(tz)

    monkeypatch.setattr("base.video.recording.datetime", FixedDatetime)
    first = RecordingSession(config_for(tmp_path), "unique1")
    second = RecordingSession(config_for(tmp_path), "unique2")
    assert first.directory.name == "2026-09-09_10-23-30"
    assert second.directory.name == "2026-09-09_10-23-30_02"
    write_segment(first, [(1, 100)])
    partial = first.directory / "2026-09-09_10-23-30_001.recording.mp4"
    assert partial.exists()
    stamp += timedelta(days=1)
    write_segment(first, [(2, 101)])
    first.finish()
    write_segment(second, [(3, 100)])
    second.finish()
    assert first.directory.parent == second.directory.parent == tmp_path / "video"
    assert not (tmp_path / "2026-09-09").exists()
    assert sorted(path.name for path in first.directory.glob("*.mp4")) == [
        "2026-09-09_10-23-30_001.mp4", "2026-09-09_10-23-30_002.mp4",
    ]
    assert (second.directory / "2026-09-09_10-23-30_02_001.mp4").exists()
    assert not partial.exists()


def test_existing_video_directory_is_reused_without_overwriting_files(tmp_path):
    video = tmp_path / "video"
    video.mkdir()
    existing = video / "existing.mp4"
    existing.write_bytes(b"existing-recording")
    session = RecordingSession(config_for(tmp_path), "reuse")
    write_segment(session, [(1, 10)])
    session.finish()
    assert session.directory.parent == video
    assert not (video / "video").exists()
    assert existing.read_bytes() == b"existing-recording"
    assert len(decoded(next(session.directory.glob("*.mp4")))) == 1


def test_video_name_occupied_by_file_fails_without_changing_it(tmp_path):
    occupied = tmp_path / "video"
    occupied.write_bytes(b"not-a-directory")
    with pytest.raises(FileExistsError):
        RecordingSession(config_for(tmp_path), "occupied")
    assert occupied.read_bytes() == b"not-a-directory"
    assert list(tmp_path.iterdir()) == [occupied]





def test_stop_while_disconnected_logs_detection_without_false_resume(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="core.video")
    session = RecordingSession(config_for(tmp_path), "unplugged")
    write_segment(session, [(1, 10)])
    session.gap("USB disconnected", disconnected_at="2026-09-09T02:00:00+00:00")
    session.finish()
    assert "detected_at=2026-09-09T02:00:00+00:00" in caplog.text
    assert "Video capture resumed:" not in caplog.text
    assert session.summary["state"] == "interrupted"


def test_repeated_gaps_keep_only_current_counters_in_memory(tmp_path):
    session = RecordingSession(config_for(tmp_path), "bounded")
    keys = set(session.summary)
    for i in range(20):
        write_segment(session, [(i, float(i))])
        session.gap("disconnected")
    session.finish()
    assert set(session.summary) == keys
    assert not any(isinstance(v, (list, dict)) for v in session.summary.values())
    assert session.summary["gaps"] == session.summary["segments"] == 20
    assert len(list(session.directory.glob("*.mp4"))) == 20


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_failed_finalization_keeps_partial_file_and_logs_error(tmp_path, monkeypatch, caplog, cleanup_fails):
    caplog.set_level(logging.INFO, logger="core.video")
    session = RecordingSession(config_for(tmp_path), "finalize-failed")
    write_segment(session, [(1, 10)])
    path = session.segment_writer.path
    calls = []
    real_abandon = session.segment_writer.abandon

    def fail():
        calls.append("finish")
        raise PermissionError("模拟视频改名拒绝访问")

    def abandon():
        calls.append("abandon")
        real_abandon()
        if cleanup_fails:
            raise OSError("simulated cleanup failure")

    monkeypatch.setattr(session.segment_writer, "finish", fail)
    monkeypatch.setattr(session.segment_writer, "abandon", abandon)
    with pytest.raises(OSError, match="模拟视频改名拒绝访问"):
        session.finish()
    assert path.exists()
    assert session.summary["state"] == "failed"
    assert session.summary["frames"] == 1
    assert "模拟视频改名拒绝访问" in caplog.text
    assert "Video session ended:" in caplog.text
    assert not list(session.directory.glob("*.json"))
    assert session.segment_writer is None
    session.finish()
    assert calls == ["finish", "abandon"]
    assert "模拟视频改名拒绝访问" in session.summary["error"]
    assert "simulated cleanup failure" not in session.summary["error"]
    assert ("Video resource cleanup failed:" in caplog.text) == cleanup_fails


def test_disk_full_keeps_existing_segments(tmp_path, monkeypatch):
    session = RecordingSession(config_for(tmp_path), "space")
    write_segment(session, [(1, 1)])
    session.gap("gap")

    def full(*args):
        raise OSError("disk full")

    monkeypatch.setattr("base.video.recording.check_space", full)
    with pytest.raises(OSError, match="disk full"):
        write_segment(session, [(2, 2)])
    with pytest.raises(OSError):
        session.finish(error="disk full")
    assert len(list(session.directory.glob("*.mp4"))) == 1
    assert session.summary["state"] == "failed"


def test_new_recording_does_not_read_or_change_old_manifests_or_partial_video(tmp_path, monkeypatch):
    stamp = datetime(2026, 9, 9, 10, 23, 30).astimezone()

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return stamp if tz is None else stamp.astimezone(tz)

    monkeypatch.setattr("base.video.recording.datetime", FixedDatetime)
    existing = tmp_path / "video" / "2026-09-09_10-23-30"
    existing.mkdir(parents=True)
    old_json = existing / "session.json"
    old_json.write_bytes(b"invalid-old-json")
    partial = existing / "2026-09-09_10-23-30_001.recording.mp4"
    partial.write_bytes(b"unverified-partial-video")
    legacy = tmp_path / "2026-09-09" / "legacy"
    legacy.mkdir(parents=True)
    legacy_json = legacy / "session.json"
    legacy_json.write_bytes(b"legacy-unreadable-json")
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.suffix in {".json", ".jsonl"}:
            raise PermissionError("No recording JSON access allowed")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", guarded_open)
        session = RecordingSession(config_for(tmp_path), "new-session")
        write_segment(session, [(1, 10)])
        session.finish()
    assert session.directory.name == "2026-09-09_10-23-30_02"
    assert old_json.read_bytes() == b"invalid-old-json"
    assert legacy_json.read_bytes() == b"legacy-unreadable-json"
    assert partial.read_bytes() == b"unverified-partial-video"
    assert not list(session.directory.glob("*.json"))


def test_invalid_identity_rejected_and_directory_collision_never_overwrites(tmp_path):
    with pytest.raises(ValueError):
        RecordingSession(config_for(tmp_path), "../escape")
    first = RecordingSession(config_for(tmp_path), "identity")
    second = RecordingSession(config_for(tmp_path), "identity")
    assert first.directory != second.directory
    first.finish()
    second.finish()


def test_root_lease_prevents_marking_another_process_active_session_interrupted(tmp_path):
    with RecordingRootLease(tmp_path):
        with pytest.raises(OSError, match="另一视频进程"):
            with RecordingRootLease(tmp_path):
                pytest.fail("second owner must not acquire the root")
    with RecordingRootLease(tmp_path):
        pass


def crash_during_recording(config):
    session = RecordingSession(config, "crashed-writer")
    encoder = FrameEncoder(config)
    session.prepare(encoder.parameters)
    for index in range(35):
        session.write(encoder.write(frame(index), index / 10), index / 10)
    session.segment_writer.sync()
    os._exit(27)  # Deliberately skip encoder.close(), normal finalization and Python cleanup.


def test_fragmented_file_readable_after_abrupt_child_exit_without_footer(tmp_path):
    context = multiprocessing.get_context("spawn")
    child = context.Process(target=crash_during_recording, args=(config_for(tmp_path),))
    child.start()
    child.join(10)
    try:
        assert child.exitcode == 27
        path = next(tmp_path.glob("*/*/*.recording.mp4"))
        before = path.read_bytes()
        assert len(decoded(path)) >= 20  # The incomplete final fragment may be lost.
        assert not list(tmp_path.rglob("*.json"))
        assert path.read_bytes() == before
    finally:
        if child.is_alive():
            child.terminate()
            child.join(3)
        child.close()
