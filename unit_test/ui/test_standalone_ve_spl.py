import json
import os
from contextlib import contextmanager
from pathlib import Path
import threading
import time

import numpy as np
import pytest
from scipy.io import wavfile

from base.pre_processing.standalone_spl_analysis import StandaloneSplResult
from base.wav_calibration_metadata import append_wav_calibration_metadata


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp():
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def _pump_until(app, predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        app.processEvents()
        if time.monotonic() >= deadline:
            pytest.fail("Qt SPL operation did not finish")
        time.sleep(0.001)
    app.processEvents()


def _config(**updates):
    value = {
        "analysis_time_range_enabled": False,
        "analysis_start_time_sec": 0.0,
        "analysis_end_time_sec": 0.0,
        "weighting": "Z",
        "smooth_checked": False,
        "limit_checked": False,
        "limit_metric": "overall_spl",
    }
    value.update(updates)
    return value


def _metadata(channels=(0, 2), *, calibrated=True):
    recorded = []
    for wav_index, physical in enumerate(channels):
        recorded.append({
            "wav_channel_index": wav_index,
            "physical_input_channel": physical,
            "factor_source": "measured" if calibrated else "none",
            "calibrated": calibrated,
            "v2pa_factor": float(wav_index + 2) if calibrated else None,
            "calibration": ({
                "standard_spl": 94.0,
                "calibrated_at": "2026-09-01T10:00:00+08:00",
                "sample_rate": 51200,
                "duration_seconds": 10.0,
            } if calibrated else None),
        })
    return {
        "schema_version": 1,
        "backend": "vkinging",
        "acquisition": {
            "model": "VE3668N",
            "machine_id": "machine-1",
            "input_mode": "IEPE",
            "unit": "V",
            "range_min": -10.0,
            "range_max": 10.0,
            "sample_rate": 51200,
        },
        "recorded_channels": recorded,
    }


def _wav(tmp_path, *, channels=(0, 2), calibrated=True, frames=1600):
    path = tmp_path / "current.wav"
    signal = np.arange(frames * len(channels), dtype=np.float32).reshape(
        frames, len(channels)
    )
    wavfile.write(path, 51200, signal)
    assert append_wav_calibration_metadata(
        path, _metadata(channels, calibrated=calibrated)
    )
    return path, signal


def test_spl_config_missing_uses_memory_default_and_accept_is_atomic(tmp_path):
    from ui.standalone_ve_spl import StandaloneSplConfigOwner

    path = tmp_path / "standalone.json"
    owner = StandaloneSplConfigOwner(path)
    assert owner.active_config is not None
    assert not path.exists()

    changed = _config(weighting="A")
    assert owner.accept_config(changed)
    assert json.loads(path.read_text(encoding="utf-8")) == {"SPL": changed}
    assert owner.active_config["weighting"] == "A"
    assert not list(tmp_path.glob("*.tmp"))


def test_spl_config_corrupt_cancel_and_failed_recovery_preserve_state(tmp_path, monkeypatch):
    from ui.standalone_ve_spl import StandaloneSplConfigOwner

    path = tmp_path / "standalone.json"
    path.write_text("{broken", encoding="utf-8")
    owner = StandaloneSplConfigOwner(path)
    assert owner.active_config is None
    owner.cancel_config()
    assert path.read_text(encoding="utf-8") == "{broken"

    monkeypatch.setattr(owner, "_atomic_write", lambda _payload: (_ for _ in ()).throw(OSError("disk")))
    assert not owner.accept_config(_config())
    assert owner.active_config is None
    assert path.read_text(encoding="utf-8") == "{broken"


def test_spl_config_failed_write_retains_prior_active_config(tmp_path, monkeypatch):
    from ui.standalone_ve_spl import StandaloneSplConfigOwner

    path = tmp_path / "standalone.json"
    owner = StandaloneSplConfigOwner(path)
    before = owner.active_config
    monkeypatch.setattr(
        owner,
        "_atomic_write",
        lambda _payload: (_ for _ in ()).throw(OSError("read only")),
    )

    assert not owner.accept_config(_config(weighting="C"))
    assert owner.active_config is before
    assert not path.exists()


def test_spl_config_huge_json_time_is_recoverable_and_file_is_preserved(tmp_path):
    from ui.standalone_ve_spl import StandaloneSplConfigOwner

    path = tmp_path / "standalone.json"
    huge = 10 ** 1000
    path.write_text(
        json.dumps({"SPL": _config(
            analysis_time_range_enabled=True,
            analysis_start_time_sec=huge,
        )}),
        encoding="utf-8",
    )
    before = path.read_bytes()

    owner = StandaloneSplConfigOwner(path)

    assert owner.active_config is None
    assert not owner.analyze_available
    assert "recovery" in owner.diagnostic
    assert path.read_bytes() == before


def test_spl_config_accepted_huge_time_preserves_prior_or_none(tmp_path):
    from ui.standalone_ve_spl import StandaloneSplConfigOwner

    huge_config = _config(
        analysis_time_range_enabled=True,
        analysis_start_time_sec=10 ** 1000,
    )
    missing_owner = StandaloneSplConfigOwner(tmp_path / "missing.json")
    prior = missing_owner.active_config
    assert not missing_owner.accept_config(huge_config)
    assert missing_owner.active_config is prior
    assert not missing_owner.path.exists()

    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text("{bad", encoding="utf-8")
    corrupt_owner = StandaloneSplConfigOwner(corrupt_path)
    assert not corrupt_owner.accept_config(huge_config)
    assert corrupt_owner.active_config is None
    assert not corrupt_owner.analyze_available
    assert corrupt_path.read_text(encoding="utf-8") == "{bad"


def _huge_threshold_configs():
    huge = 10 ** 1000
    return {
        "overall": _config(
            limit_checked=True,
            limit_metric="overall_spl",
            scalar_upper_enabled=True,
            scalar_upper_value=huge,
            scalar_lower_enabled=False,
        ),
        "manual-curve": _config(
            limit_checked=True,
            limit_metric="curve_y",
            limit_mode="manual",
            manual_input_mode="segments",
            manual_upper_enabled=True,
            manual_lower_enabled=False,
            manual_upper_segments=[{
                "start_x": 0.0,
                "start_y": huge,
                "end_x": 1.0,
                "end_y": huge,
            }],
        ),
        "csv-curve": _config(
            limit_checked=True,
            limit_metric="curve_y",
            limit_mode="csv",
            limit_data=[[0.0, 1.0], [huge, huge], [None, None]],
        ),
    }


def _invalid_csv_curve_configs():
    return {
        "nonfinite-x": _config(
            limit_checked=True,
            limit_metric="curve_y",
            limit_mode="csv",
            limit_data=[
                [0.0, float("inf")],
                [50.0, 50.0],
                [float("nan"), float("nan")],
            ],
        ),
        "infinite-bound": _config(
            limit_checked=True,
            limit_metric="curve_y",
            limit_mode="csv",
            limit_data=[
                [0.0, 1.0],
                [50.0, float("inf")],
                [float("nan"), float("nan")],
            ],
        ),
        "both-nan-row": _config(
            limit_checked=True,
            limit_metric="curve_y",
            limit_mode="csv",
            limit_data=[[0.0, 1.0], [50.0, float("nan")], [None, float("nan")]],
        ),
    }


@pytest.mark.parametrize("name", ["overall", "manual-curve", "csv-curve"])
def test_spl_config_huge_threshold_startup_is_recoverable_and_preserves_bytes(
    tmp_path, name
):
    from ui.standalone_ve_spl import StandaloneSplConfigOwner

    path = tmp_path / f"{name}.json"
    path.write_text(
        json.dumps({"SPL": _huge_threshold_configs()[name]}),
        encoding="utf-8",
    )
    before = path.read_bytes()

    owner = StandaloneSplConfigOwner(path)

    assert owner.active_config is None
    assert not owner.analyze_available
    assert "invalid" in owner.diagnostic.lower()
    assert path.read_bytes() == before


@pytest.mark.parametrize("name", ["overall", "manual-curve", "csv-curve"])
@pytest.mark.parametrize("prior_state", ["valid", "corrupt"])
def test_spl_config_huge_threshold_accept_and_save_default_preserve_prior_state(
    tmp_path, name, prior_state
):
    from ui.standalone_ve_spl import StandaloneSplConfigOwner

    path = tmp_path / f"{name}-{prior_state}.json"
    if prior_state == "valid":
        path.write_text(json.dumps({"SPL": _config(weighting="C")}), encoding="utf-8")
    else:
        path.write_bytes(b"{broken")
    before = path.read_bytes()
    owner = StandaloneSplConfigOwner(path)
    prior_active = owner.active_config
    invalid = _huge_threshold_configs()[name]

    assert not owner.save_default_config("SPL", invalid)
    assert not owner.accept_config(invalid)
    assert owner.active_config is prior_active
    assert path.read_bytes() == before
    assert "invalid" in owner.diagnostic.lower()


@pytest.mark.parametrize("name", ["overall", "manual-curve", "csv-curve"])
def test_analysis_admission_huge_threshold_is_normalized_before_worker(
    qapp, tmp_path, name
):
    from ui.standalone_ve_spl import StandaloneSplController

    path = tmp_path / "placeholder.wav"
    path.write_bytes(b"not read because configuration is rejected first")

    class Owner:
        active_config = _huge_threshold_configs()[name]
        analyze_available = True

    workers = []
    failures = []
    controller = StandaloneSplController(
        config_owner=Owner(),
        worker_factory=lambda admission: workers.append(admission),
    )
    controller.analysis_failed.connect(failures.append)
    controller.set_current_run(path, (0,))

    assert not controller.start_analysis()
    assert workers == []
    assert len(failures) == 1
    assert "active SPL configuration is invalid" in failures[0]


@pytest.mark.parametrize(
    "name", ["nonfinite-x", "infinite-bound", "both-nan-row"]
)
def test_spl_config_invalid_csv_curve_startup_recovers_without_changing_disk(
    tmp_path, name
):
    from ui.standalone_ve_spl import StandaloneSplConfigOwner

    path = tmp_path / f"startup-{name}.json"
    path.write_text(
        json.dumps({"SPL": _invalid_csv_curve_configs()[name]}),
        encoding="utf-8",
    )
    before = path.read_bytes()

    owner = StandaloneSplConfigOwner(path)

    assert owner.active_config is None
    assert not owner.analyze_available
    assert "recovery" in owner.diagnostic.lower()
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "name", ["nonfinite-x", "infinite-bound", "both-nan-row"]
)
@pytest.mark.parametrize("prior_state", ["valid", "corrupt"])
def test_spl_config_invalid_csv_curve_accept_preserves_active_and_disk(
    tmp_path, name, prior_state
):
    from ui.standalone_ve_spl import StandaloneSplConfigOwner

    path = tmp_path / f"accept-{prior_state}-{name}.json"
    if prior_state == "valid":
        path.write_text(json.dumps({"SPL": _config(weighting="C")}), encoding="utf-8")
    else:
        path.write_bytes(b"{broken")
    before = path.read_bytes()
    owner = StandaloneSplConfigOwner(path)
    prior_active = owner.active_config

    assert not owner.accept_config(_invalid_csv_curve_configs()[name])
    assert owner.active_config is prior_active
    assert path.read_bytes() == before
    assert "not activated" in owner.diagnostic.lower()


@pytest.mark.parametrize(
    "name", ["nonfinite-x", "infinite-bound", "both-nan-row"]
)
def test_analysis_admission_invalid_csv_curve_fails_before_worker(
    qapp, tmp_path, name
):
    from ui.standalone_ve_spl import StandaloneSplController

    path = tmp_path / "placeholder.wav"
    path.write_bytes(b"configuration fails before WAV parsing")

    class Owner:
        active_config = _invalid_csv_curve_configs()[name]
        analyze_available = True

    workers = []
    failures = []
    controller = StandaloneSplController(
        config_owner=Owner(),
        worker_factory=lambda admission: workers.append(admission),
    )
    controller.analysis_failed.connect(failures.append)
    controller.set_current_run(path, (0,))

    assert not controller.start_analysis()
    assert workers == []
    assert len(failures) == 1
    assert "active SPL configuration is invalid" in failures[0]


@pytest.mark.parametrize("prior_state", ["valid", "corrupt", "absent"])
def test_spl_config_post_commit_readback_failure_restores_exact_prior_state(
    tmp_path, monkeypatch, prior_state
):
    from ui.standalone_ve_spl import StandaloneSplConfigOwner

    path = tmp_path / f"{prior_state}.json"
    if prior_state == "valid":
        path.write_bytes(b'{"SPL":{"weighting":"C"}}\r\n')
    elif prior_state == "corrupt":
        path.write_bytes(b"\xff{broken\r\n")
    before_exists = path.exists()
    before_bytes = path.read_bytes() if before_exists else None
    owner = StandaloneSplConfigOwner(path)
    prior_active = owner.active_config

    monkeypatch.setattr(
        owner,
        "_readback",
        lambda: (_ for _ in ()).throw(ValueError("primary readback verification failed")),
    )

    assert not owner.accept_config(_config(weighting="A"))
    assert owner.active_config is prior_active
    assert path.exists() is before_exists
    if before_exists:
        assert path.read_bytes() == before_bytes
    assert "primary readback verification failed" in owner.diagnostic
    assert not list(tmp_path.glob(".*.tmp"))

    restarted = StandaloneSplConfigOwner(path)
    if prior_state == "valid":
        assert restarted.active_config["weighting"] == "C"
    elif prior_state == "corrupt":
        assert restarted.active_config is None
    else:
        assert restarted.active_config["weighting"] != "A"


def test_spl_config_dialog_has_no_available_channel_selector(tmp_path):
    from ui.standalone_ve_spl import StandaloneSplConfigOwner

    calls = []
    class Dialog:
        Accepted = 1
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))
        def exec_(self): return 0

    owner = StandaloneSplConfigOwner(tmp_path / "config.json", dialog_factory=Dialog)
    assert not owner.open_dialog()
    assert len(calls[0][0]) == 2
    assert "available_channels" not in calls[0][1]


def test_spl_config_real_parented_dialog_stays_top_level_modal_without_selector(
    qapp, tmp_path
):
    from PyQt5 import QtCore, QtWidgets
    from ui.ui_analysis_config.spl_config_dialog import SplConfigWindow
    from ui.standalone_ve_spl import StandaloneSplConfigOwner

    dialogs = []
    observed = []
    def factory(*args, **kwargs):
        dialog = SplConfigWindow(*args, **kwargs)
        dialogs.append(dialog)
        return dialog

    owner = StandaloneSplConfigOwner(tmp_path / "spl.json", dialog_factory=factory)
    parent = QtWidgets.QWidget()

    def inspect_and_reject():
        dialog = dialogs[0]
        observed.append((
            dialog.isWindow(),
            dialog.isModal(),
            dialog.parent() is parent,
            dialog.show_channel_selector,
            hasattr(dialog, "channel_selector"),
        ))
        dialog.reject()

    QtCore.QTimer.singleShot(0, inspect_and_reject)
    assert not owner.open_dialog(parent=parent)

    assert observed == [(True, True, True, False, False)]
    parent.close()


def test_analysis_admission_is_file_local_and_lists_all_uncalibrated(tmp_path):
    from ui.standalone_ve_spl import AnalysisAdmissionError, admit_standalone_spl_analysis

    path, _ = _wav(tmp_path, channels=(0, 2), calibrated=False)
    with pytest.raises(AnalysisAdmissionError, match="In1.*In3"):
        admit_standalone_spl_analysis(path, (0, 2), _config())


def test_analysis_admission_uses_actual_duration_for_strict_range(tmp_path):
    from ui.standalone_ve_spl import AnalysisAdmissionError, admit_standalone_spl_analysis

    path, _ = _wav(tmp_path, channels=(0,), frames=1600)
    with pytest.raises(AnalysisAdmissionError, match="duration.*configured interval"):
        admit_standalone_spl_analysis(
            path,
            (0,),
            _config(
                analysis_time_range_enabled=True,
                analysis_start_time_sec=0.04,
                analysis_end_time_sec=0.0,
            ),
        )


def test_analysis_admission_huge_finite_start_fails_before_worker(
    qapp, tmp_path
):
    from ui.standalone_ve_spl import StandaloneSplController

    path, _ = _wav(tmp_path, channels=(0,))

    class Owner:
        active_config = _config(
            analysis_time_range_enabled=True,
            analysis_start_time_sec=1.0e308,
            analysis_end_time_sec=0.0,
        )
        analyze_available = True

    workers = []
    failures = []
    controller = StandaloneSplController(
        config_owner=Owner(),
        worker_factory=lambda admission: workers.append(admission),
    )
    controller.analysis_failed.connect(failures.append)
    controller.set_current_run(path, (0,))

    assert not controller.start_analysis()
    assert workers == []
    assert len(failures) == 1
    assert "actual duration" in failures[0]
    assert "configured interval" in failures[0]


def test_analysis_admission_huge_finite_end_clamps_to_actual_eof(tmp_path):
    from ui.standalone_ve_spl import admit_standalone_spl_analysis

    path, _ = _wav(tmp_path, channels=(0,), frames=1600)
    admission = admit_standalone_spl_analysis(
        path,
        (0,),
        _config(
            analysis_time_range_enabled=True,
            analysis_start_time_sec=0.005,
            analysis_end_time_sec=1.0e308,
        ),
    )

    assert admission.range_start_sample == 256
    assert admission.range_stop_sample == 1600


@pytest.mark.parametrize(
    ("rate", "dtype", "channel_count", "match"),
    [
        (48000, np.float32, 1, "51200"),
        (51200, np.int16, 1, "float32"),
        (51200, np.float32, 2, "channel count"),
    ],
)
def test_analysis_admission_rejects_wrong_wav_contract_before_analysis(
    tmp_path, rate, dtype, channel_count, match
):
    from ui.standalone_ve_spl import AnalysisAdmissionError, admit_standalone_spl_analysis

    path = tmp_path / "wrong.wav"
    wavfile.write(path, rate, np.ones((1600, channel_count), dtype=dtype))
    # Metadata appending is deliberately unnecessary: WAV contract rejection
    # must happen before metadata or worker construction.
    with pytest.raises(AnalysisAdmissionError, match=match):
        admit_standalone_spl_analysis(path, (0,), _config())


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("short-riff", b"RIFF\x00\x00\x00\x00WAVE"),
        (
            "truncated-format",
            b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x03\x00",
        ),
    ],
)
def test_analysis_admission_normalizes_malformed_riff_parser_errors_before_worker(
    qapp, tmp_path, name, content
):
    from ui.standalone_ve_spl import StandaloneSplController

    path = tmp_path / f"{name}.wav"
    path.write_bytes(content)

    class Owner:
        active_config = _config()
        analyze_available = True

    workers = []
    failures = []
    controller = StandaloneSplController(
        config_owner=Owner(),
        worker_factory=lambda admission: workers.append(admission),
    )
    controller.analysis_failed.connect(failures.append)
    controller.set_current_run(path, (0,))

    assert not controller.start_analysis()
    assert workers == []
    assert len(failures) == 1
    assert failures[0].startswith("current WAV cannot be read:")


def test_analysis_admission_rejects_invalid_metadata_and_channel_order(tmp_path):
    from ui.standalone_ve_spl import AnalysisAdmissionError, admit_standalone_spl_analysis

    path = tmp_path / "no-metadata.wav"
    wavfile.write(path, 51200, np.ones((1600, 2), dtype=np.float32))
    with pytest.raises(AnalysisAdmissionError, match="metadata"):
        admit_standalone_spl_analysis(path, (0, 2), _config())

    path, _ = _wav(tmp_path, channels=(2, 0))
    with pytest.raises(AnalysisAdmissionError, match="channel order"):
        admit_standalone_spl_analysis(path, (0, 2), _config())


def test_analysis_admission_rejects_metadata_acquisition_rate_before_worker(
    qapp, tmp_path, monkeypatch
):
    from base.wav_calibration_metadata import (
        WavCalibrationMetadataReadResult,
        WavCalibrationMetadataReadStatus,
    )
    from ui.standalone_ve_spl import StandaloneSplController

    path = tmp_path / "mismatched-metadata-rate.wav"
    wavfile.write(path, 51200, np.ones((1600, 2), dtype=np.float32))
    mismatched = _metadata((0, 2))
    mismatched["acquisition"]["sample_rate"] = 48000
    monkeypatch.setattr(
        "ui.standalone_ve_spl.inspect_wav_calibration_metadata",
        lambda _path: WavCalibrationMetadataReadResult(
            WavCalibrationMetadataReadStatus.VALID,
            mismatched,
            declared_backend="vkinging",
        ),
    )

    class Owner:
        active_config = _config()
        analyze_available = True

    worker_calls = []
    controller = StandaloneSplController(
        config_owner=Owner(),
        worker_factory=lambda admission: worker_calls.append(admission),
    )
    failures = []
    controller.analysis_failed.connect(failures.append)
    controller.set_current_run(path, (0, 2))

    assert not controller.start_analysis()
    assert worker_calls == []
    assert len(failures) == 1
    assert "metadata" in failures[0].lower()
    assert "48000" in failures[0]
    assert "51200" in failures[0]


def test_analysis_admission_rejects_invalid_active_config(tmp_path):
    from ui.standalone_ve_spl import AnalysisAdmissionError, admit_standalone_spl_analysis

    path, _ = _wav(tmp_path, channels=(0,))
    with pytest.raises(AnalysisAdmissionError, match="active SPL configuration"):
        admit_standalone_spl_analysis(path, (0,), _config(weighting="bad"))


@pytest.mark.parametrize("current", [None, Path("missing.wav")])
def test_analysis_admission_rejects_missing_current_path_before_worker(current):
    from ui.standalone_ve_spl import AnalysisAdmissionError, admit_standalone_spl_analysis

    with pytest.raises(AnalysisAdmissionError):
        admit_standalone_spl_analysis(current, (0,), _config())


def test_analysis_worker_is_sequential_ordered_and_continues_after_failure(tmp_path):
    from ui.standalone_ve_spl import StandaloneSplAnalysisWorker, admit_standalone_spl_analysis

    path, source = _wav(tmp_path)
    admission = admit_standalone_spl_analysis(
        path,
        (0, 2),
        _config(
            analysis_time_range_enabled=True,
            analysis_start_time_sec=0.005,
            analysis_end_time_sec=0.02,
        ),
    )
    lifecycle = []

    @contextmanager
    def reader(_path, index, start, stop, **_kwargs):
        lifecycle.append(("open", index, start, stop))
        try:
            yield source[start:stop, index]
        finally:
            lifecycle.append(("close", index))

    def analyzer(voltage, **kwargs):
        index = len([event for event in lifecycle if event[0] == "open"]) - 1
        lifecycle.append((
            "analyze",
            index,
            kwargs["v2pa_factor"],
            id(kwargs["config"]),
            len(voltage),
            kwargs["source_start_sample"],
            kwargs["config"]["analysis_time_range_enabled"],
        ))
        if index == 0:
            raise RuntimeError("first failed")
        values = np.asarray([1.0, 2.0])
        values.setflags(write=False)
        return StandaloneSplResult(values, values, None, None, None, "dB")

    worker = StandaloneSplAnalysisWorker(
        admission,
        analyzer=analyzer,
        channel_reader=reader,
        max_plot_points=10,
    )
    successes, failures, finished = [], [], []
    worker.channel_succeeded.connect(lambda item: successes.append(item))
    worker.channel_failed.connect(lambda label, message: failures.append((label, message)))
    worker.finished.connect(lambda summary: finished.append(summary))
    worker.run()

    assert lifecycle[0] == ("open", 0, 256, 1024)
    assert lifecycle[1] == (
        "analyze", 0, 2.0, lifecycle[1][3], 768, 256, False
    )
    assert lifecycle[2] == ("close", 0)
    assert lifecycle[3][0:2] == ("open", 1)
    assert lifecycle[4][3] == lifecycle[1][3]
    assert lifecycle[4][4:] == (768, 256, False)
    assert lifecycle[-1] == ("close", 1)
    assert failures == [("In1", "first failed")]
    assert [item.label for item in successes] == ["In3"]
    assert finished[0].failures == (("In1", "first failed"),)


def test_analysis_worker_default_mapping_is_bounded_and_released_at_channel_boundary(tmp_path):
    from ui.standalone_ve_spl import mapped_wav_channel

    path = tmp_path / "mapped.wav"
    wavfile.write(path, 51200, np.ones((32, 2), dtype=np.float32))
    with mapped_wav_channel(path, 1, 7, 19) as voltage:
        assert voltage.shape == (12,)
        assert isinstance(voltage.base, np.memmap)

    # Windows replacement fails while scipy's mmap handle remains open.
    replacement = tmp_path / "released.wav"
    path.replace(replacement)
    assert replacement.is_file()


def test_analysis_worker_cancellation_before_first_work_emits_one_summary(tmp_path):
    from ui.standalone_ve_spl import (
        StandaloneSplAnalysisWorker,
        admit_standalone_spl_analysis,
    )

    path, _ = _wav(tmp_path)
    admission = admit_standalone_spl_analysis(path, (0, 2), _config())
    reader_calls = []
    worker = StandaloneSplAnalysisWorker(
        admission,
        channel_reader=lambda *_args, **_kwargs: reader_calls.append(True),
    )
    summaries = []
    worker.finished.connect(summaries.append)

    worker.request_cancel()
    worker.run()

    assert reader_calls == []
    assert len(summaries) == 1
    assert summaries[0].cancelled
    assert summaries[0].succeeded_labels == ()


def test_analysis_worker_cancellation_between_channels_releases_current_mapping(tmp_path):
    from ui.standalone_ve_spl import (
        StandaloneSplAnalysisWorker,
        admit_standalone_spl_analysis,
    )

    path, _ = _wav(tmp_path)
    admission = admit_standalone_spl_analysis(path, (0, 2), _config())
    lifecycle = []
    worker_box = {}

    @contextmanager
    def reader(_path, index, _start, _stop, **_kwargs):
        lifecycle.append(("open", index))
        try:
            yield np.ones(1401)
        finally:
            lifecycle.append(("close", index))

    def analyzer(_voltage, **_kwargs):
        worker_box["worker"].request_cancel()
        x = np.asarray([0.0, 1.0])
        return StandaloneSplResult(x, x, None, None, None, "dB")

    worker = StandaloneSplAnalysisWorker(
        admission,
        analyzer=analyzer,
        channel_reader=reader,
    )
    worker_box["worker"] = worker
    summaries = []
    worker.finished.connect(summaries.append)
    worker.run()

    assert lifecycle == [("open", 0), ("close", 0)]
    assert len(summaries) == 1
    assert summaries[0].cancelled
    assert summaries[0].succeeded_labels == ("In1",)


def test_analysis_worker_controller_cancel_during_blocked_channel_is_nonblocking_and_retires(
    qapp, tmp_path
):
    from ui.standalone_ve_spl import (
        StandaloneSplAnalysisWorker,
        StandaloneSplController,
    )

    path, _ = _wav(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    lifecycle = []

    @contextmanager
    def reader(_path, index, _start, _stop, **_kwargs):
        lifecycle.append(("open", index))
        entered.set()
        assert release.wait(3.0)
        try:
            yield np.ones(1401)
        finally:
            lifecycle.append(("close", index))

    def analyzer(_voltage, **_kwargs):
        x = np.asarray([0.0, 1.0])
        return StandaloneSplResult(x, x, None, None, None, "dB")

    workers = []
    def worker_factory(admission):
        worker = StandaloneSplAnalysisWorker(
            admission,
            analyzer=analyzer,
            channel_reader=reader,
        )
        workers.append(worker)
        return worker

    class Owner:
        active_config = _config()
        analyze_available = True

    class Window:
        def show(self): pass
        def close(self): pass

    controller = StandaloneSplController(
        config_owner=Owner(),
        worker_factory=worker_factory,
        result_window_factory=lambda *_args, **_kwargs: Window(),
    )
    availability = []
    summaries = []
    controller.availability_changed.connect(availability.append)
    controller.analysis_finished.connect(summaries.append)
    controller.set_current_run(path, (0, 2))
    assert controller.start_analysis()
    _pump_until(qapp, entered.is_set)

    started = time.monotonic()
    controller.request_cancel()
    assert time.monotonic() - started < 0.1
    assert controller.analyzing
    assert controller._worker is workers[0]
    assert availability[-1] is False

    release.set()
    _pump_until(qapp, lambda: not controller.analyzing)

    assert lifecycle == [("open", 0), ("close", 0)]
    assert controller._worker is None
    assert controller._thread is None
    assert len(summaries) == 1
    assert summaries[0].cancelled
    assert summaries[0].succeeded_labels == ("In1",)
    assert availability == [True, False, True]


@pytest.mark.parametrize("weighting", ["A", "C"])
def test_analysis_worker_weighted_nonzero_range_matches_full_source_causal_filter(
    tmp_path, monkeypatch, weighting
):
    import ui.standalone_ve_spl as standalone
    from base.pre_processing.standalone_spl_analysis import analyze_standalone_spl
    from ui.standalone_ve_spl import (
        StandaloneSplAnalysisWorker,
        admit_standalone_spl_analysis,
    )

    frames = 6000
    time_axis = np.arange(frames, dtype=float) / 51200.0
    signal = (
        0.2 * np.sin(2 * np.pi * 1000 * time_axis)
        + 0.05 * np.sin(2 * np.pi * 80 * time_axis)
    ).astype(np.float32)
    path = tmp_path / f"weighted-{weighting}.wav"
    wavfile.write(path, 51200, signal)
    assert append_wav_calibration_metadata(path, _metadata((0,)))
    config = _config(
        weighting=weighting,
        show_overall_spl=True,
        analysis_time_range_enabled=True,
        analysis_start_time_sec=0.02,
        analysis_end_time_sec=0.10,
        limit_checked=True,
        limit_metric="curve_y",
        limit_mode="manual",
        manual_input_mode="constant",
        constant_upper_enabled=True,
        constant_lower_enabled=True,
        constant_upper_value=200.0,
        constant_lower_value=-200.0,
    )
    expected = analyze_standalone_spl(
        signal,
        sample_rate=51200,
        v2pa_factor=2.0,
        config=config,
        max_plot_points=12000,
    )
    admission = admit_standalone_spl_analysis(path, (0,), config)

    real_lfilter = standalone.lfilter
    chunk_sizes = []
    def tracked_lfilter(b, a, values, **kwargs):
        chunk_sizes.append(len(values))
        return real_lfilter(b, a, values, **kwargs)
    monkeypatch.setattr(standalone, "lfilter", tracked_lfilter)

    worker = StandaloneSplAnalysisWorker(
        admission,
        filter_chunk_frames=257,
        max_plot_points=12000,
    )
    successes = []
    worker.channel_succeeded.connect(successes.append)
    worker.run()

    assert len(successes) == 1
    actual = successes[0].result
    np.testing.assert_allclose(actual.time_seconds, expected.time_seconds)
    np.testing.assert_allclose(actual.spl_db, expected.spl_db, rtol=1e-10, atol=1e-10)
    assert actual.overall_spl == pytest.approx(expected.overall_spl)
    assert actual.judged_ok is expected.judged_ok
    assert actual.deviation_db == pytest.approx(expected.deviation_db)
    assert chunk_sizes
    assert max(chunk_sizes) <= 257
    assert sum(chunk_sizes) == admission.range_stop_sample
    assert max(chunk_sizes) < admission.range_start_sample


def test_result_window_factory_wiring_is_lazy(monkeypatch, tmp_path):
    import record_vkinging_continuous as recorder

    sentinel = object()
    monkeypatch.setattr(
        "ui.standalone_ve_spl.StandaloneSplController",
        lambda **kwargs: (sentinel, kwargs),
    )
    result = recorder.create_standalone_spl_controller(
        config_path=tmp_path / "spl.json", parent="parent"
    )
    assert result[0] is sentinel
    assert result[1]["parent"] == "parent"


def test_result_window_uses_precomputed_bounded_result_only(qapp, monkeypatch):
    from ui.signal_analysis_window import Spl
    from ui.standalone_ve_spl import (
        StandaloneSplChannel,
        StandaloneSplChannelResult,
        StandaloneSplResultWindow,
    )

    monkeypatch.setattr(
        Spl,
        "calculate_spl",
        lambda *_args, **_kwargs: pytest.fail("must not recalculate SPL"),
    )
    time_values = np.asarray([0.0, 0.5, 1.0])
    spl_values = np.asarray([40.0, 42.0, 41.0])
    time_values.setflags(write=False)
    spl_values.setflags(write=False)
    bounded = StandaloneSplResult(
        time_values, spl_values, 41.5, True, 1.0, "dBA"
    )
    channel = StandaloneSplChannel(0, 2, "In3", 2.0)

    window = StandaloneSplResultWindow(
        StandaloneSplChannelResult(channel, bounded), _config(weighting="A")
    )
    try:
        assert window.windowTitle() == "SPL - In3"
        curve = window.analysis_plot.listDataItems()[0]
        np.testing.assert_array_equal(curve.xData, time_values)
        np.testing.assert_array_equal(curve.yData, spl_values)
    finally:
        window.close()


def test_result_window_honors_nested_ranges_from_immutable_config_owner(
    qapp, tmp_path
):
    from ui.standalone_ve_spl import (
        StandaloneSplChannel,
        StandaloneSplChannelResult,
        StandaloneSplConfigOwner,
        StandaloneSplResultWindow,
    )

    owner = StandaloneSplConfigOwner(tmp_path / "spl.json")
    configured = _config(
        display={
            "plot_view": {
                "x_enabled": True,
                "x_min": 0.25,
                "x_max": 0.75,
                "y_enabled": True,
                "y_min": 35.0,
                "y_max": 45.0,
            }
        }
    )
    assert owner.accept_config(configured)
    assert not isinstance(owner.active_config["display"], dict)
    x = np.asarray([0.0, 1.0])
    y = np.asarray([40.0, 41.0])
    x.setflags(write=False)
    y.setflags(write=False)
    result = StandaloneSplResult(x, y, None, None, None, "dB")
    item = StandaloneSplChannelResult(
        StandaloneSplChannel(0, 0, "In1", 2.0), result
    )

    window = StandaloneSplResultWindow(item, owner.active_config)
    try:
        x_range, y_range = window.analysis_plot.getViewBox().viewRange()
        assert x_range == pytest.approx([0.25, 0.75])
        assert y_range == pytest.approx([35.0, 45.0])
    finally:
        window.close()


@pytest.mark.parametrize(
    ("show_overall", "judge_overall", "expect_value", "expect_title"),
    [
        (True, False, True, True),
        (False, True, True, False),
        (False, False, False, False),
    ],
)
def test_result_window_overall_title_follows_display_config_not_internal_value(
    qapp,
    tmp_path,
    show_overall,
    judge_overall,
    expect_value,
    expect_title,
):
    from base.pre_processing.standalone_spl_analysis import analyze_standalone_spl
    from ui.standalone_ve_spl import (
        StandaloneSplChannel,
        StandaloneSplChannelResult,
        StandaloneSplConfigOwner,
        StandaloneSplResultWindow,
    )

    config = _config(
        show_overall_spl=show_overall,
        limit_checked=judge_overall,
        limit_metric="overall_spl",
        scalar_upper_enabled=True,
        scalar_upper_value=200.0,
        scalar_lower_enabled=False,
        scalar_lower_value=0.0,
    )
    owner = StandaloneSplConfigOwner(tmp_path / f"overall-{show_overall}-{judge_overall}.json")
    assert owner.accept_config(config)
    result = analyze_standalone_spl(
        np.ones(1401),
        sample_rate=51200,
        v2pa_factor=1.0,
        config=owner.active_config,
        max_plot_points=100,
    )
    assert (result.overall_spl is not None) is expect_value
    if judge_overall:
        assert result.judged_ok is True
    item = StandaloneSplChannelResult(
        StandaloneSplChannel(0, 0, "In1", 1.0), result
    )

    window = StandaloneSplResultWindow(item, owner.active_config)
    try:
        title_text = window.analysis_plot.plotItem.titleLabel.text
        assert bool(title_text) is expect_title
        if expect_title:
            assert "Overall SPL" in title_text
    finally:
        window.close()


@pytest.mark.parametrize(
    ("metric", "judged_ok", "deviation", "unit", "verdict"),
    [
        ("overall_spl", True, 4.25, "dBA", "OK"),
        ("overall_spl", False, 2.50, "dBC", "NG"),
        ("curve_y", True, 3.75, "dB", "OK"),
        ("curve_y", False, 1.25, "dBD", "NG"),
    ],
)
def test_result_window_shows_threshold_verdict_and_deviation(
    qapp, tmp_path, metric, judged_ok, deviation, unit, verdict
):
    from ui.standalone_ve_spl import (
        StandaloneSplChannel,
        StandaloneSplChannelResult,
        StandaloneSplConfigOwner,
        StandaloneSplResultWindow,
    )

    if metric == "overall_spl":
        config = _config(
            limit_checked=True,
            limit_metric=metric,
            scalar_upper_enabled=True,
            scalar_upper_value=200.0,
            scalar_lower_enabled=False,
        )
    else:
        config = _config(
            limit_checked=True,
            limit_metric=metric,
            limit_mode="manual",
            manual_input_mode="constant",
            constant_upper_enabled=True,
            constant_upper_value=200.0,
            constant_lower_enabled=False,
        )
    owner = StandaloneSplConfigOwner(tmp_path / f"{metric}-{judged_ok}.json")
    assert owner.accept_config(config)
    x = np.asarray([0.0, 1.0])
    y = np.asarray([40.0, 41.0])
    result = StandaloneSplResult(x, y, 40.5, judged_ok, deviation, unit)
    item = StandaloneSplChannelResult(
        StandaloneSplChannel(0, 0, "In1", 2.0), result
    )

    window = StandaloneSplResultWindow(item, owner.active_config)
    try:
        assert window.windowTitle() == "SPL - In1"
        assert window.judgment_label is not None
        text = window.judgment_label.text()
        assert verdict in text
        assert f"{deviation:.2f}" in text
        assert unit in text
    finally:
        window.close()


def test_result_window_without_threshold_judgment_has_no_verdict(qapp):
    from ui.standalone_ve_spl import (
        StandaloneSplChannel,
        StandaloneSplChannelResult,
        StandaloneSplResultWindow,
    )

    x = np.asarray([0.0, 1.0])
    result = StandaloneSplResult(x, x, None, None, None, "dB")
    item = StandaloneSplChannelResult(
        StandaloneSplChannel(0, 0, "In1", 2.0), result
    )

    window = StandaloneSplResultWindow(item, _config())
    try:
        assert window.judgment_label is None
        assert "OK" not in window.windowTitle()
        assert "NG" not in window.windowTitle()
    finally:
        window.close()


def test_analysis_worker_ng_judgment_remains_successful_batch_result(tmp_path):
    from ui.standalone_ve_spl import (
        StandaloneSplAnalysisWorker,
        admit_standalone_spl_analysis,
    )

    path, _ = _wav(tmp_path, channels=(0, 2))
    admission = admit_standalone_spl_analysis(path, (0, 2), _config())

    def analyzer(_voltage, **_kwargs):
        x = np.asarray([0.0, 1.0])
        return StandaloneSplResult(x, x, 40.0, False, 2.0, "dB")

    worker = StandaloneSplAnalysisWorker(admission, analyzer=analyzer)
    successes = []
    failures = []
    summaries = []
    worker.channel_succeeded.connect(successes.append)
    worker.channel_failed.connect(lambda *args: failures.append(args))
    worker.finished.connect(summaries.append)

    worker.run()

    assert [item.label for item in successes] == ["In1", "In3"]
    assert all(item.result.judged_ok is False for item in successes)
    assert failures == []
    assert summaries[0].succeeded_labels == ("In1", "In3")
    assert summaries[0].failures == ()


def test_result_window_with_parent_is_top_level_and_delete_on_close(qapp):
    from PyQt5 import QtCore, QtWidgets, sip
    from ui.standalone_ve_spl import (
        StandaloneSplChannel,
        StandaloneSplChannelResult,
        StandaloneSplResultWindow,
    )

    parent = QtWidgets.QWidget()
    x = np.asarray([0.0, 1.0])
    y = np.asarray([40.0, 41.0])
    result = StandaloneSplResult(x, y, None, None, None, "dB")
    item = StandaloneSplChannelResult(
        StandaloneSplChannel(0, 0, "In1", 2.0), result
    )
    window = StandaloneSplResultWindow(item, _config(), parent=parent)
    destroyed = []
    window.destroyed.connect(lambda: destroyed.append(True))

    assert window.isWindow()
    assert window.testAttribute(QtCore.Qt.WA_DeleteOnClose)
    window.show()
    window.close()
    qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    qapp.processEvents()
    assert destroyed == [True]
    assert sip.isdeleted(window)
    parent.close()


def test_result_window_controller_retires_previous_real_windows(qapp, tmp_path):
    from PyQt5 import QtCore, QtWidgets, sip
    from ui.standalone_ve_spl import (
        StandaloneSplChannel,
        StandaloneSplChannelResult,
        StandaloneSplController,
        StandaloneSplResultWindow,
    )

    parent = QtWidgets.QWidget()
    class Owner:
        active_config = _config()
        analyze_available = True

    controller = StandaloneSplController(config_owner=Owner(), parent=parent)
    x = np.asarray([0.0, 1.0])
    y = np.asarray([40.0, 41.0])
    result = StandaloneSplResult(x, y, None, None, None, "dB")
    items = []
    for physical in (0, 2):
        items.append(StandaloneSplChannelResult(
            StandaloneSplChannel(physical, physical, f"In{physical + 1}", 2.0),
            result,
        ))
        controller._show_result(items[-1])
    old_windows = tuple(controller.result_windows)

    controller.close_result_windows()
    qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    qapp.processEvents()

    assert controller.result_windows == []
    assert all(sip.isdeleted(window) for window in old_windows)
    controller._show_result(items[0])
    assert len(controller.result_windows) == 1
    assert controller.result_windows[0].isWindow()
    assert not sip.isdeleted(controller.result_windows[0])
    controller.close_result_windows()
    qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    qapp.processEvents()
    parent.close()


@pytest.mark.parametrize("failure_stage", ["factory", "show"])
def test_result_window_presentation_failure_is_reconciled_once(
    qapp, failure_stage
):
    from ui.standalone_ve_spl import (
        StandaloneSplBatchSummary,
        StandaloneSplChannel,
        StandaloneSplChannelResult,
        StandaloneSplController,
    )

    class Owner:
        active_config = _config()
        analyze_available = True

    partial = []
    class BrokenWindow:
        def __init__(self):
            self.close_calls = 0
            self.delete_calls = 0
        def show(self):
            raise RuntimeError("show exploded")
        def close(self):
            self.close_calls += 1
        def deleteLater(self):
            self.delete_calls += 1

    def factory(*_args, **_kwargs):
        if failure_stage == "factory":
            raise RuntimeError("factory exploded")
        window = BrokenWindow()
        partial.append(window)
        return window

    controller = StandaloneSplController(
        config_owner=Owner(),
        result_window_factory=factory,
    )
    failures = []
    finals = []
    controller.analysis_failed.connect(failures.append)
    controller.analysis_finished.connect(finals.append)
    x = np.asarray([0.0, 1.0])
    result = StandaloneSplResult(x, x, None, None, None, "dB")
    item = StandaloneSplChannelResult(
        StandaloneSplChannel(0, 2, "In3", 2.0), result
    )

    controller._show_result(item)
    controller._retain_summary(StandaloneSplBatchSummary(("In3",), ()))
    controller._thread_finished()

    assert failures == [f"In3: {failure_stage} exploded"]
    assert len(finals) == 1
    assert finals[0].succeeded_labels == ()
    assert finals[0].failures == (("In3", f"{failure_stage} exploded"),)
    assert controller.result_windows == []
    if partial:
        assert partial[0].close_calls == 1
        assert partial[0].delete_calls == 1


def test_result_window_controller_uses_fresh_thread_and_closes_previous_batch(
    qapp, tmp_path
):
    from PyQt5 import QtCore
    from ui.standalone_ve_spl import (
        StandaloneSplAnalysisWorker,
        StandaloneSplController,
    )

    path, _ = _wav(tmp_path, channels=(0, 2))

    class Owner:
        active_config = _config()
        analyze_available = True

    workers = []
    def analyzer(_voltage, **_kwargs):
        x = np.asarray([0.0, 1.0])
        y = np.asarray([40.0, 41.0])
        x.setflags(write=False)
        y.setflags(write=False)
        return StandaloneSplResult(x, y, None, None, None, "dB")

    def worker_factory(admission):
        worker = StandaloneSplAnalysisWorker(admission, analyzer=analyzer)
        workers.append(worker)
        return worker

    gui_thread = qapp.thread()
    windows = []
    class Window:
        def __init__(self, channel_result):
            assert QtCore.QThread.currentThread() is gui_thread
            self.label = channel_result.label
            self.show_calls = 0
            self.close_calls = 0
        def show(self): self.show_calls += 1
        def close(self): self.close_calls += 1

    def window_factory(channel_result, _config, parent=None):
        window = Window(channel_result)
        windows.append(window)
        return window

    controller = StandaloneSplController(
        config_owner=Owner(),
        worker_factory=worker_factory,
        result_window_factory=window_factory,
    )
    controller.set_current_run(path, (0, 2))

    assert controller.start_analysis()
    first_thread = controller._thread
    _pump_until(qapp, lambda: not controller.analyzing)
    first_windows = tuple(windows)
    assert [window.label for window in first_windows] == ["In1", "In3"]
    assert [window.show_calls for window in first_windows] == [1, 1]

    assert controller.start_analysis()
    second_thread = controller._thread
    assert second_thread is not first_thread
    assert [window.close_calls for window in first_windows] == [1, 1]
    _pump_until(qapp, lambda: not controller.analyzing)
    assert len(workers) == 2
