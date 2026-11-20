"""Product integration: real algorithms, stored artifacts, reports and required verdicts."""
import csv
from dataclasses import replace
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from base.analysis.contracts import AnalysisInstance
from base.analysis.executor import execute_task
from base.analysis.runtime_protocol import failed_task
from base.analysis_report import _prepare_record
from base.analysis_report_source import AnalysisItemIdentity, ReportCandidate, _discover_analysis_items
from base.audio_label_policy import aggregate_judgements, audio_label_for_results, judgement_summary
from base.audio_label_sync import AudioLabelJob, AudioLabelSyncService
from consts.acoustic_analysis.common_consts import SOUND_QUALITY_ANALYSES
from unit_test.base.analysis.test_sound_quality_backend import quality_task


def item(key, verdict="OK", **changes):
    return {"config_key": key, "analysis_type": "SHRP", "channel_index": 0,
            "judgement_required": True, "judgement_status": "completed", "judgement": verdict,
            "execution_status": "分析完成", **changes}


@pytest.mark.parametrize("mode,expected", [
    ("success", "OK"), ("missing", "结果不完整"), ("calculation", "结果不完整"),
    ("judgement", "结果不完整"), ("duplicate", "结果不完整"),
    ("ng_missing", "NG（结果不完整）"), ("optional_failed", "OK"),
    ("wrong_channel", "结果不完整"),
])
def test_required_completeness(mode, expected):
    items = [item("a"), item("b")]
    required = [dict(config_key=i["config_key"], analysis_type="SHRP", channel_index=0) for i in items]
    if mode in {"missing", "ng_missing"}:
        items.pop()
    if mode == "ng_missing":
        items[0]["judgement"] = "NG"
    if mode == "calculation":
        items[1].update(execution_status="分析失败", judgement_status="not_executed", judgement=None)
    if mode == "judgement":
        items[1].update(judgement_status="failed", judgement=None)
    if mode == "duplicate":
        items.append(items[1].copy())
    if mode == "wrong_channel":
        items[1]["channel_index"] = 1
    if mode == "optional_failed":
        required.pop()
        items[1].update(execution_status="分析失败", judgement=None, judgement_required=False)
    summary = judgement_summary(items, required)
    assert summary["status"] == expected
    task = {"instance_results": items, "judgement_summary": summary}
    assert audio_label_for_results(task) == ("NG" if expected.startswith("NG") else "OK" if expected == "OK" else "not_labeled")
    assert aggregate_judgements(["OK", expected]) == expected


def test_no_required_items_does_not_invent_ok():
    assert judgement_summary([item("a")], [])["status"] == "--"


@pytest.mark.parametrize("field", ["execution_status", "judgement_status"])
def test_frozen_required_item_needs_explicit_success_status(field):
    result = item("a")
    result.pop(field)
    assert judgement_summary([result], [item("a")])["status"] == "结果不完整"


@pytest.mark.parametrize("kind", ["SHRP", "ROUGH"])
def test_quality_csv_report_and_full_curve(quality_task, kind, monkeypatch):
    spec = SOUND_QUALITY_ANALYSES[kind]
    config = {"limit_checked": True, "limit_metric": spec["default_metric"],
              "scalar_upper_enabled": True, "scalar_upper_value": 99.0}
    task = replace(quality_task, instances=tuple(AnalysisInstance("测量项目", kind, ch, config) for ch in [0, 1]))
    result = execute_task(task)
    assert result["judgement_summary"]["status"] == "OK"
    directory = Path(task.analysis_result_directory) / "csv" / Path(task.wav_path).stem
    discovered = _discover_analysis_items(directory, directory / "absent-images")
    assert len(discovered) == 1 and discovered[0].identity == AnalysisItemIdentity("测量项目", kind)
    assert discovered[0].has_scalar_values and discovered[0].has_charts
    candidate = ReportCandidate(task.wav_path, "project", "model", "sample", analysis_items=discovered)
    prepared = _prepare_record(candidate, discovered[0].identity, "values_and_charts")
    assert not prepared.issues and not prepared.chart_issues
    assert prepared.unit == spec["unit"]
    assert len(prepared.charts) == 2 and set(prepared.scalar_map()) == {"CH1", "CH2"}
    for instance in result["instance_results"]:
        curve_file = next(a["path"] for a in instance["artifacts"] if a["kind"] == "curve_csv")
        with open(curve_file, encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream))
        assert spec["unit"] in rows[0][1]
        assert len(rows) == len(instance["display_payload"]["x"]) + 1
        np.testing.assert_allclose([float(r[1]) if r[1] else np.nan for r in rows[1:]],
                                   np.asarray(instance["display_payload"]["y"], dtype=float), rtol=1e-9, equal_nan=True)
    # Both saves may fail, while computation and a valid verdict remain usable.
    from base.analysis import storage
    monkeypatch.setattr(storage, "save_sound_quality_summary", lambda *a: (_ for _ in ()).throw(OSError("denied")))
    failed_save = execute_task(task)
    assert failed_save["judgement_summary"]["status"] == "OK"
    assert all(i["artifacts"][0]["status"] == "failed" and i["artifacts"][1]["status"] == "saved"
               for i in failed_save["instance_results"])


def test_silent_sharpness_keeps_missing_values_and_failed_judgement(quality_task):
    import soundfile as sf
    sf.write(quality_task.wav_path, np.zeros((48000, 2)), 48000, subtype="FLOAT")
    task = replace(quality_task, instances=(AnalysisInstance("SHRP", "SHRP", 0, {
        "limit_checked": True, "limit_metric": "s_mean_acum", "scalar_upper_enabled": True,
        "scalar_upper_value": 99.0}),))
    result = execute_task(task)
    assert result["judgement_summary"]["status"] == "结果不完整"
    record = result["instance_results"][0]
    assert record["metrics"]["s_mean_acum"] is None
    with open(record["artifacts"][0]["path"], encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["平均尖锐度acum"] == "" and row["判定结果"] == "判定失败"
    directory = Path(record["artifacts"][0]["path"]).parent
    discovered = _discover_analysis_items(directory, directory / "absent-images")
    candidate = ReportCandidate(task.wav_path, "project", "model", "sample", analysis_items=discovered)
    prepared = _prepare_record(candidate, discovered[0].identity, "values_only")
    assert prepared.scalar_map()["CH1"].value == "—"
    assert "CH1 测量值缺失" in prepared.issues


def test_status_saved_by_existing_label_worker_and_read_after_restart(tmp_path):
    from base.analysis_report_source import _build_candidate
    root = tmp_path / "project"
    wav = root / "model" / "sample" / "wav" / "audio.wav"
    wav.parent.mkdir(parents=True)
    wav.touch()
    summary = judgement_summary([item("a")], [item("a"), item("b")])
    calls, finished = [], Event()
    class Repository:
        def update(self, path, label):
            calls.append((path, label))
            return "updated"
    service = AudioLabelSyncService(Repository(), lambda *args: finished.set())
    try:
        service.submit(AudioLabelJob("task", str(wav), "not_labeled", str(wav.parent.parent), summary))
        assert finished.wait(5)
    finally:
        service.begin_shutdown()
        service._thread.join(5)
    candidate = _build_candidate(root, wav)
    # Even a stale OK database field cannot erase the frozen completeness record.
    assert replace(candidate, label="OK").result_text == "结果不完整"
    assert candidate.data_status == "结果不完整"
    assert calls == [(str(wav), "not_labeled")]


def test_process_failure_uses_frozen_requirements(quality_task):
    task = replace(quality_task, instances=(AnalysisInstance("SHRP", "SHRP", 0, {"limit_checked": True}),))
    result = failed_task(task, {"type": "WorkerExit", "message": "exited"})
    assert result["judgement_summary"]["status"] == "结果不完整"
    assert result["judgement_summary"]["required_count"] == 1
    assert audio_label_for_results(result) == "not_labeled"


def test_quality_artifact_names_do_not_strip_user_channel_suffix():
    from base.analysis_artifact_names import parse_csv_artifact
    assert parse_csv_artifact(Path("custom_CH1_尖锐度汇总.csv")) == ("custom_CH1", "SHRP", "尖锐度汇总")
    assert parse_csv_artifact(Path("custom_CH1_尖锐度汇总_CH2.csv")) == ("custom_CH1", "SHRP", "尖锐度汇总")


def test_status_save_failure_does_not_cancel_label_commit(tmp_path, monkeypatch):
    import base.audio_label_sync as module
    done, results, calls = Event(), [], []
    class Repository:
        def update(self, path, label):
            calls.append(label)
            return "committed"
    def denied(*args):
        raise OSError("read only")
    monkeypatch.setattr(module, "write_csv_atomic", denied)
    def complete(*args):
        results.append(args)
        done.set()
    service = AudioLabelSyncService(Repository(), complete)
    try:
        service.submit(AudioLabelJob("id", "audio.wav", "OK", str(tmp_path), judgement_summary([item("a")])))
        assert done.wait(5)
    finally:
        service.begin_shutdown()
        service._thread.join(5)
    assert calls == ["OK"] and results[0][1] == "committed" and isinstance(results[0][2], OSError)


def test_daily_incomplete_count_preserves_ng_and_pass_rate():
    from datetime import datetime
    from base.motor_daily_wav_records import DailyWavRecordStore
    now = datetime(2026, 11, 17, 10)
    store = DailyWavRecordStore()
    store.replace([dict(completed_at=now, station_id=1, file_name=f"{i}.wav", condition="正转", result=status)
                   for i, status in enumerate(["OK", "NG", "结果不完整", "NG（结果不完整）", "未标记"])], now.date())
    assert (store.summary.total, store.summary.ok, store.summary.ng, store.summary.incomplete,
            store.summary.unmarked, store.summary.pass_rate) == (5, 1, 2, 2, 1, "20.0%")


def test_report_counts_and_status_preserve_incomplete_ng():
    from datetime import datetime
    from base.analysis_report import _build_summary_html
    candidates = [ReportCandidate("x.wav", "project", "model", "sample", label="OK", judgement_status=status)
                  for status in ["OK", "结果不完整", "NG（结果不完整）"]]
    html = _build_summary_html(candidates, [], "values_only", datetime(2026, 11, 17), 2)
    assert "OK 1" in html and "NG 1" in html and "结果不完整 2" in html
    assert candidates[1].result_text == "结果不完整" and candidates[2].data_status == "结果不完整"
