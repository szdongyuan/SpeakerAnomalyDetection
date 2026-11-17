"""Stable recorded WAV identity and frozen channel provenance."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from base.record_analysis_reference import RecordAnalysisReference
from base.workbench_recording_protocol import StationWavChannel


def channel(index):
    return StationWavChannel(f"channel-{index}", f"Mic {index}", "sound", "device", index + 4, index)


def test_reference_keeps_full_path_identity_and_frozen_channel_order(tmp_path, monkeypatch):
    def forbidden_io(*args, **kwargs):
        raise AssertionError("reference validation must not inspect the filesystem")

    for name in ("exists", "resolve", "stat", "open"):
        monkeypatch.setattr(Path, name, forbidden_io)
    channels = [channel(0), channel(1)]
    path = str(tmp_path / "not-created" / "same.wav")
    reference = RecordAnalysisReference("request-a", "source-z", "condition-b", path, channels)
    channels.reverse()
    channels.clear()
    assert reference.wav_path == path
    assert reference.record_key == ("request-a", "source-z")
    assert reference.channels == (channel(0), channel(1))
    assert isinstance(reference.channels, tuple)
    with pytest.raises(FrozenInstanceError):
        reference.wav_path = "changed.wav"
    with pytest.raises(FrozenInstanceError):
        reference.channels[0].channel_id = "changed"
    other = replace(reference, request_id="request-b", wav_path=str(tmp_path / "other" / "same.wav"))
    assert other.record_key != reference.record_key
    assert other.wav_path != reference.wav_path
    assert replace(reference, station_id="source-y").record_key != reference.record_key


@pytest.mark.parametrize("field,value", [
    ("request_id", ""), ("request_id", "  "), ("request_id", 1),
    ("station_id", None), ("station_id", "\t"),
    ("condition_id", ""), ("condition_id", 2),
    ("wav_path", "same.wav"), ("wav_path", ""), ("wav_path", None),
    ("wav_path", "C:relative.wav"), ("wav_path", "\x00bad.wav"),
    ("channels", None), ("channels", []), ("channels", "channels"),
    ("channels", [object()]), ("channels", [channel(1)]),
    ("channels", [channel(0), channel(0)]),
    ("channels", [channel(0), channel(2)]),
])
def test_invalid_reference_reports_field(tmp_path, field, value):
    values = dict(request_id="request", station_id="source", condition_id="condition",
                  wav_path=str(tmp_path / "record.wav"), channels=[channel(0)])
    values[field] = value
    with pytest.raises(ValueError, match=field):
        RecordAnalysisReference(**values)
