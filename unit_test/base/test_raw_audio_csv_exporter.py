import csv

import numpy as np
import pytest
import soundfile as sf

from base.raw_audio_csv_exporter import export_raw_audio_csv


def test_export_raw_audio_csv_preserves_physical_channels_and_samples(tmp_path):
    wav_path = tmp_path / "wav" / "recording.wav"
    csv_path = tmp_path / "csv" / "raw_csv" / "recording.csv"
    wav_path.parent.mkdir()
    samples = np.array(
        [
            [0.0, 0.25],
            [-0.5, 1.0 / 3.0],
            [0.75, -1.0],
        ],
        dtype=np.float32,
    )
    sf.write(wav_path, samples, 4, subtype="FLOAT")

    result = export_raw_audio_csv(
        wav_path,
        csv_path,
        raw_channels=(0, 2),
        block_frames=2,
    )

    assert result == csv_path
    assert csv_path.read_bytes().startswith(b"\xef\xbb\xbf")
    with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.reader(csv_file))
    assert rows[0] == ["sample_index", "time_s", "CH1", "CH3"]
    assert [row[:2] for row in rows[1:]] == [
        ["0", "0.000000000"],
        ["1", "0.250000000"],
        ["2", "0.500000000"],
    ]
    np.testing.assert_allclose(
        np.asarray([[float(value) for value in row[2:]] for row in rows[1:]]),
        samples,
        rtol=0,
        atol=1e-8,
    )


def test_export_raw_audio_csv_rejects_channel_mapping_mismatch(tmp_path):
    wav_path = tmp_path / "recording.wav"
    sf.write(wav_path, np.zeros((2, 2), dtype=np.float32), 48000, subtype="FLOAT")

    with pytest.raises(ValueError, match="channel count"):
        export_raw_audio_csv(
            wav_path,
            tmp_path / "raw.csv",
            raw_channels=(0,),
        )


def test_export_raw_audio_csv_keeps_existing_target_when_replace_fails(
    tmp_path,
    monkeypatch,
):
    wav_path = tmp_path / "recording.wav"
    csv_path = tmp_path / "raw_csv" / "recording.csv"
    sf.write(wav_path, np.array([0.25], dtype=np.float32), 48000, subtype="FLOAT")
    csv_path.parent.mkdir()
    csv_path.write_bytes(b"existing")

    def fail_replace(_source, _target):
        raise PermissionError("target is locked")

    monkeypatch.setattr("base.raw_audio_csv_exporter.os.replace", fail_replace)

    with pytest.raises(PermissionError, match="locked"):
        export_raw_audio_csv(wav_path, csv_path, raw_channels=(0,))

    assert csv_path.read_bytes() == b"existing"
    assert list(csv_path.parent.glob("*.tmp")) == []
