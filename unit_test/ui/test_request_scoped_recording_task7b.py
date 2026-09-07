import csv
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from openpyxl import load_workbook


def _request(tmp_path, *, backend="soundcard", factors=(1.0,), channels=None):
    if channels is None:
        channels = tuple(range(len(factors)))
    metadata = {
        "recorded_channels": [
            {"v2pa_factor": factor} for factor in factors
        ]
    }
    return SimpleNamespace(
        request_id="task7b-request",
        path=str(tmp_path / "recording.wav"),
        channels=tuple(channels),
        device={"backend": backend},
        calibration_metadata=metadata,
    )


def test_bounded_request_executor_runs_off_caller_and_delivers_once():
    from ui.sequence.request_scoped_recording_executor import (
        RequestScopedRecordingExecutor,
    )

    caller = threading.get_ident()
    dispatched = []
    release = threading.Event()
    executor = RequestScopedRecordingExecutor(
        dispatch=dispatched.append, capacity=2, max_workers=1)
    try:
        first = executor.submit(
            "A", lambda: (release.wait(2), threading.get_ident())[1],
            lambda outcome: dispatched.append(("A", outcome)))
        second = executor.submit(
            "B", lambda: threading.get_ident(),
            lambda outcome: dispatched.append(("B", outcome)))
        assert first is True and second is True
        assert executor.submit("C", lambda: None, lambda _outcome: None) is False
        assert executor.submit("A", lambda: None, lambda _outcome: None) is False
        release.set()

        deadline = threading.Event()
        for _index in range(200):
            if len(dispatched) >= 2:
                break
            deadline.wait(.01)
        assert len(dispatched) == 2
        callbacks = list(dispatched)
        dispatched.clear()
        for callback in callbacks:
            callback()
        assert [item[0] for item in dispatched] == ["A", "B"]
        assert all(item[1].error is None for item in dispatched)
        assert all(item[1].value != caller for item in dispatched)
        assert executor.pending_count == 0
    finally:
        executor.shutdown(wait=True)


def test_request_executor_close_does_not_leave_non_daemon_long_job_or_deliver_late():
    from ui.sequence.request_scoped_recording_executor import (
        RequestScopedRecordingExecutor,
    )

    started = threading.Event()
    release = threading.Event()
    dispatched = []
    executor = RequestScopedRecordingExecutor(
        dispatch=dispatched.append, capacity=2, max_workers=1)
    executor.submit(
        "long-A", lambda: (started.set(), release.wait(5))[1],
        lambda outcome: dispatched.append(outcome))
    assert started.wait(2)
    workers = [thread for thread in threading.enumerate()
               if thread.name.startswith("recording-result")]
    assert workers and all(thread.daemon for thread in workers)
    before = time.monotonic()
    executor.shutdown(wait=False)
    assert time.monotonic() - before < .1
    release.set()
    time.sleep(.05)
    assert dispatched == []


def test_request_executor_invalidates_callback_already_queued_before_shutdown():
    from ui.sequence.request_scoped_recording_executor import (
        RequestScopedRecordingExecutor,
    )

    queued = []
    delivered = []
    executor = RequestScopedRecordingExecutor(
        dispatch=queued.append, capacity=2, max_workers=1)
    executor.submit("queued-A", lambda: "complete", delivered.append)
    for _index in range(200):
        if queued:
            break
        time.sleep(.01)
    assert len(queued) == 1
    executor.shutdown(wait=True)
    queued[0]()
    assert delivered == []


def test_directional_publication_uses_frozen_group_not_repeated_barcode(
        tmp_path, monkeypatch):
    from unit_test.ui.test_recording_process_integration import main_host, _late_context
    from ui.sequence import request_scoped_count_publisher as counts

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    published = []
    host.count_board = SimpleNamespace(mode="B-mode")
    monkeypatch.setattr(counts, "increment_mark_result", published.append)
    monkeypatch.setattr(counts, "increment_shared_result", lambda _label: None)

    def context(request_id, direction, group_id, label):
        value = _late_context(
            host, request_id, tmp_path / f"{request_id}.wav", object(),
            direction=direction, session_id=request_id, tcp=None)
        value.directional_cycle_active = True
        value.count_mode = "mark"
        value.barcode = "REPEATED-SN"
        value.publication_group_id = group_id
        value.analysis_label = label
        value.recorded_signal_info["labels"] = label
        return value

    for group in ("run-1", "run-2"):
        host._publish_request_scoped_recording_business(
            context(f"{group}-forward", "forward", group, "OK"))
        host._publish_request_scoped_recording_business(
            context(f"{group}-reverse", "reverse", group, "NG"))

    assert published == ["NG", "NG"]


def test_publication_retries_only_uncommitted_effect_and_counts_test_mode(
        tmp_path, monkeypatch):
    from unit_test.ui.test_recording_process_integration import main_host, _late_context
    from ui.sequence import request_scoped_count_publisher as counts

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    calls = []

    def flaky(label):
        calls.append(label)
        if len(calls) == 1:
            raise OSError("durable count failed")

    host.count_board = SimpleNamespace(mode="B-mode")
    monkeypatch.setattr(counts, "increment_shared_result", flaky)
    context = _late_context(
        host, "retry-A", tmp_path / "retry.wav", object(),
        direction="", session_id="retry-session", tcp=None)
    context.count_mode = "test"
    context.analysis_label = "OK"
    context.recorded_signal_info["labels"] = "OK"

    with pytest.raises(OSError, match="durable count failed"):
        host._publish_request_scoped_recording_business(context)
    assert host._publish_request_scoped_recording_business(context) == {
        "test_count_label": "OK"}
    assert calls == ["OK", "OK"]
    assert host._publish_request_scoped_recording_business(context) == {
        "test_count_label": "OK"}
    assert calls == ["OK", "OK"]


def test_mark_count_retry_after_shared_failure_never_repeats_mark_increment(
        tmp_path, monkeypatch):
    from unit_test.ui.test_recording_process_integration import main_host, _late_context
    from ui.sequence import request_scoped_count_publisher as counts

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    context = _late_context(
        host, "split-count", tmp_path / "split.wav", object(),
        direction="", session_id="split-session", tcp=None)
    context.count_mode = "mark"
    context.publication_group_id = "unique-run"
    context.barcode = "REPEATED"
    context.analysis_label = "OK"
    context.recorded_signal_info["labels"] = "OK"
    calls = {"mark": 0, "shared": 0}

    def mark(label):
        calls["mark"] += 1
        return label

    def shared(label):
        calls["shared"] += 1
        if calls["shared"] == 1:
            raise OSError("shared count failed")
        return label

    monkeypatch.setattr(counts, "increment_mark_result", mark)
    monkeypatch.setattr(counts, "increment_shared_result", shared)
    with pytest.raises(OSError, match="shared count failed"):
        host._publish_request_scoped_recording_business(context)
    assert host._publish_request_scoped_recording_business(context) == {
        "mark_count_label": "OK"}
    assert calls == {"mark": 1, "shared": 2}
    assert context.business_effect_attempts[
        "count:mark:unique-run:mark-file"] == 1
    assert context.business_effect_attempts[
        "count:mark:unique-run:shared"] == 2


@pytest.mark.parametrize("order", [("c1", "c2"), ("c2", "c1")])
def test_manual_group_counts_only_after_every_explicit_completed_sibling(
        tmp_path, monkeypatch, order):
    from unit_test.ui.test_recording_process_integration import main_host, _late_context
    from ui.sequence import request_scoped_count_publisher as counts

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    published = []
    monkeypatch.setattr(counts, "increment_mark_result", published.append)
    monkeypatch.setattr(counts, "increment_shared_result", lambda _label: None)
    host.recent_test_session_by_id = {}

    def make(key, label, group_id="group-1"):
        context = _late_context(
            host, f"request-{group_id}-{key}", tmp_path / f"{key}.wav",
            object(), direction=key, session_id=f"session-{group_id}-{key}",
            tcp=None)
        context.manual_product_cycle_active = True
        context.count_mode = "mark"
        context.product_group_id = group_id
        context.publication_group_id = group_id
        context.product_condition_key = key
        context.product_condition_keys = ("c1", "c2")
        context.analysis_label = label
        context.recorded_signal_info["labels"] = label
        host.recent_test_session_by_id[context.recent_session_id] = {
            "session_id": context.recent_session_id,
            "group_id": group_id,
            "condition_key": key,
            "product_recording_state": "completed",
            "analysis_report_state": "completed",
            "business_completion_state": "completed",
            "recorded_signal_info": {"labels": label},
        }
        return context

    contexts = {"c1": make("c1", "OK"), "c2": make("c2", "NG")}
    first, second = (contexts[key] for key in order)
    assert host._publish_request_scoped_recording_business(first) == {}
    assert published == []
    assert host._publish_request_scoped_recording_business(second) == {
        "mark_count_label": "NG"}
    assert published == ["NG"]
    assert host._publish_request_scoped_recording_business(first) == {}
    assert host._publish_request_scoped_recording_business(second) == {
        "mark_count_label": "NG"}
    assert published == ["NG"]

    # Same barcode is irrelevant; a new frozen group counts independently.
    next_c1 = make("c1", "OK", group_id="group-2")
    next_c2 = make("c2", "OK", group_id="group-2")
    next_c1.barcode = next_c2.barcode = "REPEATED"
    host._publish_request_scoped_recording_business(next_c1)
    host._publish_request_scoped_recording_business(next_c2)
    assert published == ["NG", "OK"]


def test_manual_group_pending_placeholder_does_not_count_or_fail(tmp_path, monkeypatch):
    from unit_test.ui.test_recording_process_integration import main_host, _late_context
    from ui.sequence import request_scoped_count_publisher as counts

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    published = []
    monkeypatch.setattr(counts, "increment_mark_result", published.append)
    monkeypatch.setattr(counts, "increment_shared_result", lambda _label: None)
    context = _late_context(
        host, "pending-A", tmp_path / "pending.wav", object(),
        direction="c1", session_id="pending-session", tcp=None)
    context.manual_product_cycle_active = True
    context.count_mode = "mark"
    context.product_group_id = context.publication_group_id = "pending-group"
    context.product_condition_key = "c1"
    context.product_condition_keys = ("c1", "c2")
    context.analysis_label = "OK"
    host.recent_test_session_by_id = {
        "pending-session": {
            "group_id": "pending-group", "condition_key": "c1",
            "product_recording_state": "completed",
            "analysis_report_state": "completed",
            "business_completion_state": "completed",
            "recorded_signal_info": {"labels": "OK"},
        },
        "placeholder": {
            "group_id": "pending-group", "condition_key": "c2",
            "product_recording_state": "pending",
            "analysis_report_state": "pending",
            "business_completion_state": "pending",
            "recorded_signal_info": {"labels": "NG"},
        },
    }
    assert host._publish_request_scoped_recording_business(context) == {}
    assert published == []


def test_publication_retry_keeps_durable_excel_and_retries_only_failed_pdf(
        tmp_path, monkeypatch):
    from unit_test.ui.test_recording_process_integration import main_host, _late_context
    from ui.sequence import sequence_widget_analysis_ops as analysis

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    context = _late_context(
        host, "partial-A", tmp_path / "partial.wav", object(),
        direction="c1", session_id="partial-session", tcp=None)
    context.analysis_required = True
    context.product_group_id = "frozen-group"
    context.publication_group_id = "frozen-run"
    context.product_condition_key = "c1"
    context.product_condition_keys = ("c1",)
    context.product_report_config = {"enabled": True}
    context.recent_session_config_snapshot = {"analysis_config": {
        "excel": {"type": "Excel", "fast_mode": True},
    }}
    context.analysis_items_data = {"ai": {"type": "AI", "label": "OK"}}
    context.analysis_result_dict = {"ai": (True, 0.2)}
    host.recent_test_session_by_id = {"partial-session": {
        "session_id": "partial-session", "group_id": "frozen-group",
        "condition_key": "c1", "product_recording_state": "completed",
        "analysis_report_state": "completed",
        "recorded_signal_info": {"labels": "OK"},
    }}
    calls = {"excel": 0, "pdf": 0}

    def export_excel(*_args, **_kwargs):
        calls["excel"] += 1
        return SimpleNamespace(ok=True, message="durable")

    def export_pdf(*_args, **_kwargs):
        calls["pdf"] += 1
        if calls["pdf"] == 1:
            return SimpleNamespace(ok=False, message="disk full", file_path="")
        return SimpleNamespace(ok=True, message="durable", file_path="group.pdf")

    monkeypatch.setattr(analysis, "export_analysis_to_csv_spool", export_excel)
    monkeypatch.setattr(analysis, "export_product_test_pdf", export_pdf)

    with pytest.raises(RuntimeError, match="disk full"):
        host._publish_request_scoped_recording_business(context)
    assert calls == {"excel": 1, "pdf": 1}
    assert host._publish_request_scoped_recording_business(context) == {
        "excel": ("excel",), "product_report": "group.pdf"}
    assert calls == {"excel": 1, "pdf": 2}
    assert host._publish_request_scoped_recording_business(context) == {
        "excel": ("excel",), "product_report": "group.pdf"}
    assert calls == {"excel": 1, "pdf": 2}


def test_600_second_spectrogram_retention_plan_is_ci_safe_and_bounded():
    from ui.sequence.request_scoped_recording_analysis import (
        _spectrogram_retention_plan,
    )
    from base.spectrogram_analysis_service import cqt_resource_plan

    started = time.monotonic()
    plan = _spectrogram_retention_plan(
        frame_count=600 * 48000, sample_rate=48000,
        n_fft=2048, hop_length=256, channel_count=8)
    assert time.monotonic() - started < .1
    assert plan["source_samples"] == 28_800_000
    assert plan["retained_time_bins"] <= 1024
    assert plan["retained_frequency_bins"] <= 512
    assert plan["estimated_numeric_bytes"] <= 4 * 1024 * 1024
    assert plan["source_numeric_bytes"] == 600 * 48000 * 8 * 4
    assert len(plan["frame_indices"]) == plan["retained_time_bins"]

    # Warm librosa's lazily imported filter helpers; the timed assertion is
    # about plan size scaling, not one-time module initialization.
    cqt_resource_plan(
        frame_count=48000, sample_rate=48000,
        n_fft=2048, hop_length=256, channel_count=1)
    log_started = time.monotonic()
    log_plan = cqt_resource_plan(
        frame_count=600 * 48000, sample_rate=48000,
        n_fft=2048, hop_length=256, channel_count=8)
    assert time.monotonic() - log_started < .1
    assert log_plan["retained_time_bins"] <= 1024
    assert log_plan["retained_frequency_bins"] <= 512
    assert log_plan["chunk_count"] == 32
    assert log_plan["max_chunk_samples"] < 1_200_000
    assert log_plan["estimated_peak_numeric_bytes"] < 64 * 1024 * 1024


def test_detached_spl_uses_shared_production_trace_service(tmp_path):
    from base.spl_analysis_service import compute_spl_trace
    from ui.sequence.request_scoped_recording_analysis import _analyze_spl

    sample_rate = 8000
    signal = (.01 * np.sin(
        2 * np.pi * 440 * np.arange(4096) / sample_rate)).astype(np.float32)
    config = {"analysis_channel": 0, "weighting": "Z",
              "smooth_checked": True, "show_overall_spl": True,
              "limit_checked": False}
    expected = compute_spl_trace(
        signal, sample_rate, config, v2pa_factor=1.0)
    judgement, detail = _analyze_spl(
        _request(tmp_path), signal[:, None], sample_rate, "spl", config)

    assert judgement == (None, 0.0)
    np.testing.assert_allclose(detail["signal_duration"], expected.signal_duration)
    np.testing.assert_allclose(detail["signal_spl"], expected.signal_spl)
    assert detail["overall_spl"] == pytest.approx(expected.overall_spl)


def test_request_scoped_spl_sparse_csv_limits_do_not_bridge_missing_rows():
    from ui.sequence.request_scoped_recording_analysis import _curve_limits
    from ui.signal_analysis_window import _interpolate_spl_limit_curves

    target = np.asarray([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    limit_data = (
        [2.0, 0.0, 1.0],
        [10.0, 10.0, np.nan],
        [2.0, np.nan, 0.0],
    )
    upper, lower = _curve_limits(
        {
            "limit_mode": "csv",
            "limit_data": limit_data,
        },
        target,
    )
    production_upper, production_lower = _interpolate_spl_limit_curves(
        target,
        *limit_data,
    )

    np.testing.assert_allclose(
        upper,
        [np.nan, 10.0, np.nan, np.nan, np.nan, 10.0, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        lower,
        [np.nan, np.nan, np.nan, 0.0, 1.0, 2.0, np.nan],
        equal_nan=True,
    )
    np.testing.assert_array_equal(upper, production_upper)
    np.testing.assert_array_equal(lower, production_lower)


def test_active_detached_and_standalone_spl_share_pure_interpolation_helper():
    from base.pre_processing import spl_runtime_config
    from ui import signal_analysis_window
    from ui.sequence import request_scoped_recording_analysis
    from ui.ui_analysis_config import analysis_compat

    shared = analysis_compat.interpolate_spl_limit_curves
    assert signal_analysis_window.interpolate_spl_limit_curves is shared
    assert request_scoped_recording_analysis.interpolate_spl_limit_curves is shared
    assert spl_runtime_config.interpolate_spl_limit_curves is shared


def test_request_scoped_fba_custom_bands_match_config_parser_for_legal_syntax():
    from ui.sequence.request_scoped_recording_analysis import _parse_custom_bands
    from ui.ui_analysis_config.fba_config_dialog import FbaConfigWindow

    text = """
        # comment and blank lines are ignored

        1000, 2000, Presence Band
        200 1000 Mid Band
        20-200
        2000\t4000\tHigh Band
    """
    expected = [
        (20.0, 200.0, None),
        (200.0, 1000.0, "Mid Band"),
        (1000.0, 2000.0, "Presence Band"),
        (2000.0, 4000.0, "High Band"),
    ]

    assert FbaConfigWindow._parse_custom_bands_text(text) == expected
    assert _parse_custom_bands(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "invalid",
        "0, 20, zero",
        "200, 20, reversed",
        "20, 300, Low\n200, 1000, Mid",
    ],
)
def test_request_scoped_fba_custom_band_errors_match_config_parser(text):
    from ui.sequence.request_scoped_recording_analysis import _parse_custom_bands
    from ui.ui_analysis_config.fba_config_dialog import FbaConfigWindow

    with pytest.raises(ValueError) as config_error:
        FbaConfigWindow._parse_custom_bands_text(text)
    with pytest.raises(ValueError) as request_error:
        _parse_custom_bands(text)

    assert str(request_error.value) == str(config_error.value)


def test_log_cqt_active_and_detached_share_numeric_shape_axes_and_ve_calibration(
        tmp_path):
    from PyQt5.QtWidgets import QApplication
    from ui.signal_analysis_window import Spectrogram
    from ui.sequence.request_scoped_recording_analysis import _analyze_spec

    app = QApplication.instance() or QApplication([])
    sample_rate = 48000
    time_axis = np.arange(16384, dtype=float) / sample_rate
    signal = (.01 * np.sin(2 * np.pi * 1000 * time_axis)).astype(np.float32)
    config = {"analysis_channel": 0, "n_fft": 2048,
              "hop_length": 256, "window_func": "hann",
              "freq_scale_type": "log", "custom_limit": False}
    widget = Spectrogram("cqt")
    try:
        widget.data_struct.store_wave_data = signal
        widget.data_struct.store_wave_data_multi = signal[:, None]
        widget.data_struct.sample_rate = sample_rate
        widget.analysis_config = dict(config)
        widget.v2pa_factor = 2.0
        active = widget.calculate_spec()
        _judgement, detached = _analyze_spec(
            _request(tmp_path, backend="vkinging", factors=(2.0,)),
            signal[:, None], sample_rate, config)

        assert active["mode"] == detached["mode"] == "log"
        assert np.asarray(active["spectrogram_db"]).shape == np.asarray(
            detached["spectrogram_db"]).shape
        np.testing.assert_allclose(
            active["frequency_bins"], detached["frequency_bins"])
        np.testing.assert_allclose(active["time_s"], detached["time_s"])
        np.testing.assert_allclose(
            active["spectrogram_db"], detached["spectrogram_db"],
            rtol=1e-7, atol=1e-7)

        uncalibrated = dict(config)
        widget.v2pa_factor = 1.0
        widget.analysis_config = uncalibrated
        raw = widget.calculate_spec()
        from base.pre_processing.audio_thd_frequency_response_analysis import (
            AudioThdFrequencyResponseAnalysis,
        )
        direct_cqt, direct_frequency, direct_time = (
            AudioThdFrequencyResponseAnalysis.compute_cqt(
                y=signal, sr=sample_rate, hop_length=256, n_fft=2048,
                fmin=None))
        import librosa
        direct_db = librosa.amplitude_to_db(
            np.abs(direct_cqt), ref=20e-6)
        np.testing.assert_allclose(raw["frequency_bins"], direct_frequency)
        np.testing.assert_allclose(raw["time_s"], direct_time)
        np.testing.assert_allclose(
            raw["spectrogram_db"], direct_db, rtol=1e-7, atol=1e-7)
        finite = np.isfinite(active["spectrogram_db"]) & np.isfinite(
            raw["spectrogram_db"])
        np.testing.assert_allclose(
            np.asarray(active["spectrogram_db"])[finite]
            - np.asarray(raw["spectrogram_db"])[finite],
            20.0 * np.log10(2.0), atol=1e-4)
    finally:
        widget.close()
        app.processEvents()
