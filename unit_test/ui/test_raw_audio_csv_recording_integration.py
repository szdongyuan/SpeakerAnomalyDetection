from datetime import datetime
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from base.analysis_artifact_paths import AnalysisStorageContext
from consts.product_test_project_consts import EXPORT_RAW_AUDIO_CSV_KEY
from ui.sequence import sequence_widget_streaming_ops as streaming_ops_module
from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin
from ui.sequence.recording_process_context import RecordingProcessContext


class _SignalRecorder:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)


class _ImmediateThread:
    def __init__(self, *, target, name, daemon):
        self.target = target
        self.name = name
        self.daemon = daemon

    def start(self):
        self.target()


def _storage_context(tmp_path):
    return AnalysisStorageContext(
        result_root_directory=str(tmp_path.resolve()),
        project_name="项目A",
        product_model="型号01",
        sample_number="样本001",
        port_name="端口1",
        test_round=1,
        condition_name="档位1",
        recording_started_at=datetime(2026, 9, 4, 10, 11, 12),
    )


def _host(tmp_path, *, enabled=True):
    storage_context = _storage_context(tmp_path)
    wav_path = (
        tmp_path
        / "项目A"
        / "型号01"
        / "样本001"
        / "audio"
        / "wav"
        / "recording.wav"
    )
    return SimpleNamespace(
        product_test_project_context={EXPORT_RAW_AUDIO_CSV_KEY: enabled},
        recorded_path=str(wav_path),
        recorded_signal_info={
            "analysis_storage": storage_context.to_metadata(),
        },
        raw_audio_csv_export_succeeded=_SignalRecorder(),
        raw_audio_csv_export_failed=_SignalRecorder(),
        _raw_audio_csv_export_lock=threading.Lock(),
        _raw_audio_csv_export_threads=set(),
    )


def test_raw_audio_csv_export_is_not_scheduled_in_wav_only_mode(
    tmp_path,
    monkeypatch,
):
    host = _host(tmp_path, enabled=False)
    export_calls = []
    monkeypatch.setattr(
        streaming_ops_module,
        "export_raw_audio_csv",
        lambda *args: export_calls.append(args),
    )

    scheduled = SequenceWidgetStreamingOpsMixin._schedule_raw_audio_csv_export(
        host,
        (0, 2),
    )

    assert scheduled is False
    assert export_calls == []


def test_raw_audio_csv_export_uses_same_wav_stem_and_physical_channels(
    tmp_path,
    monkeypatch,
):
    host = _host(tmp_path)
    export_calls = []
    monkeypatch.setattr(streaming_ops_module.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(
        streaming_ops_module,
        "export_raw_audio_csv",
        lambda *args: export_calls.append(args),
    )

    scheduled = SequenceWidgetStreamingOpsMixin._schedule_raw_audio_csv_export(
        host,
        (0, 2),
    )

    assert scheduled is True
    assert len(export_calls) == 1
    wav_path, csv_path, raw_channels = export_calls[0]
    assert wav_path.endswith("wav\\recording.wav")
    assert csv_path == (
        tmp_path
        / "项目A"
        / "型号01"
        / "样本001"
        / "audio"
        / "raw_csv"
        / "recording.csv"
    )
    assert raw_channels == (0, 2)
    assert host.raw_audio_csv_export_succeeded.calls == [(str(csv_path),)]
    assert host.raw_audio_csv_export_failed.calls == []
    assert host._raw_audio_csv_export_threads == set()


def test_background_csv_failure_is_reported_without_raising_to_recording_flow(
    tmp_path,
    monkeypatch,
):
    host = _host(tmp_path)
    monkeypatch.setattr(streaming_ops_module.threading, "Thread", _ImmediateThread)

    def fail_export(*_args):
        raise PermissionError("target is locked")

    monkeypatch.setattr(
        streaming_ops_module,
        "export_raw_audio_csv",
        fail_export,
    )

    scheduled = SequenceWidgetStreamingOpsMixin._schedule_raw_audio_csv_export(
        host,
        (0,),
    )

    assert scheduled is True
    assert host.raw_audio_csv_export_succeeded.calls == []
    assert len(host.raw_audio_csv_export_failed.calls) == 1
    assert "target is locked" in host.raw_audio_csv_export_failed.calls[0][1]
    assert host._raw_audio_csv_export_threads == set()


def test_csv_failure_handler_logs_and_shows_explicit_wav_safe_message(
    monkeypatch,
):
    messages = []
    logs = []
    host = SimpleNamespace(
        default_logger=SimpleNamespace(error=logs.append),
    )
    monkeypatch.setattr(
        streaming_ops_module.QMessageBox,
        "warning",
        lambda _parent, title, message: messages.append((title, message)),
    )

    SequenceWidgetStreamingOpsMixin._on_raw_audio_csv_export_failed(
        host,
        "D:/results/recording.wav",
        "target is locked",
    )

    assert "raw_audio_csv_export_failed" in logs[0]
    assert messages == [
        (
            "原始音频 CSV 保存失败",
            "WAV 已保存，但原始 CSV 保存失败。\ntarget is locked",
        )
    ]


@pytest.mark.parametrize("request_owned_only", [True, False])
@pytest.mark.parametrize("enabled", [True, False])
def test_request_completion_exports_own_audio_once_across_business_retry(
    tmp_path, monkeypatch, request_owned_only, enabled,
):
    host = _host(tmp_path, enabled=enabled)
    old_path = host.recorded_path
    context = RecordingProcessContext(
        request=SimpleNamespace(path=old_path, channels=(0, 2), request_id="A"),
        direction="forward", tcp_completion=None, preview_enabled=False,
        recorded_signal_info=host.recorded_signal_info.copy(),
    )
    # B has taken the UI; A's completion must still export A's source and folder.
    host.recorded_path = str(tmp_path / "newer-B.wav")
    host.recorded_signal_info = {"analysis_storage": {"newer": "B"}}
    host._recording_context_owns_active_workflow = lambda _context: request_owned_only
    host._schedule_raw_audio_csv_export = (
        SequenceWidgetStreamingOpsMixin._schedule_raw_audio_csv_export.__get__(host))
    host._complete_detached_recording_context = (
        SequenceWidgetStreamingOpsMixin._complete_detached_recording_context.__get__(host))
    host.default_logger = SimpleNamespace(error=lambda _message: None)

    def fail_business(*_args, **_kwargs):
        raise RuntimeError("business publication will retry")

    host._cache_condition_record = fail_business
    export_calls = []
    monkeypatch.setattr(streaming_ops_module.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(
        streaming_ops_module, "export_raw_audio_csv",
        lambda *args: export_calls.append(args))
    audio = np.ones((4, 2), dtype=np.float32)
    for _attempt in range(2):
        assert SequenceWidgetStreamingOpsMixin._on_streaming_complete(
            host, recorded_mono=audio.mean(axis=1), recorded_multi=audio,
            sample_rate=48000, prefinalized=True, recording_context=context,
            request_owned_only=request_owned_only,
        ) is False
        assert "business publication will retry" in context.business_failure

    assert len(export_calls) == int(enabled)
    if enabled:
        wav_path, csv_path, channels = export_calls[0]
        assert wav_path == old_path
        assert csv_path == (
            tmp_path / "项目A" / "型号01" / "样本001"
            / "audio" / "raw_csv" / "recording.csv")
        assert channels == (0, 2)
    assert context.recorded_signal_info["sample_rate"] == 48000
    assert host.recorded_signal_info == {"analysis_storage": {"newer": "B"}}


def test_request_csv_path_failure_uses_signal_for_background_completion(
    tmp_path,
):
    host = _host(tmp_path)
    context = SimpleNamespace(
        request=SimpleNamespace(path="A.wav"), recorded_signal_info={})

    assert SequenceWidgetStreamingOpsMixin._schedule_raw_audio_csv_export(
        host, (0,), recording_context=context,
    ) is False

    assert len(host.raw_audio_csv_export_failed.calls) == 1
    assert host.raw_audio_csv_export_failed.calls[0][0] == "A.wav"
