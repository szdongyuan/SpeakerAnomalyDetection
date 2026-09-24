import csv
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from base.raw_audio_csv_exporter import export_raw_audio_csv


@pytest.fixture
def export_paths(tmp_path):
    wav = tmp_path / "recording.wav"
    sf.write(wav, np.array([0.25], dtype=np.float32), 48000, subtype="FLOAT")
    return wav, tmp_path / "recording.csv", tmp_path / ".recording.task.tmp"


def test_export_uses_controlled_temporary_path(export_paths, monkeypatch):
    wav, target, temporary = export_paths
    replace = os.replace
    published = []

    def observe_replace(source, destination):
        published.append((Path(source), Path(destination)))
        replace(source, destination)

    monkeypatch.setattr("base.raw_audio_csv_exporter.os.replace", observe_replace)
    assert export_raw_audio_csv(wav, target, (0,), temporary_path=temporary) == target
    assert published == [(temporary, target)]
    assert not temporary.exists()


@pytest.mark.parametrize("invalid", ["outside", "target", "source", "directory"])
def test_export_rejects_unsafe_temporary_path(export_paths, invalid):
    wav, target, temporary = export_paths
    choices = {"outside": target.parent / "other" / "task.tmp", "target": target,
               "source": wav, "directory": target.parent}
    with pytest.raises(ValueError, match="temporary"):
        export_raw_audio_csv(wav, target, (0,), temporary_path=choices[invalid])
    assert wav.exists()
    assert not target.exists()


def test_controlled_temporary_collision_preserves_existing_files(export_paths):
    wav, target, temporary = export_paths
    temporary.write_bytes(b"belongs to another task")
    target.write_bytes(b"old csv")
    with pytest.raises(FileExistsError):
        export_raw_audio_csv(wav, target, (0,), temporary_path=temporary)
    assert temporary.read_bytes() == b"belongs to another task"
    assert target.read_bytes() == b"old csv"


@pytest.mark.parametrize("failure_stage", ["open", "fdopen", "write", "replace"])
def test_controlled_temporary_failure_closes_and_cleans_owned_file(
    export_paths, monkeypatch, failure_stage
):
    wav, target, temporary = export_paths
    target.write_bytes(b"old csv")
    descriptors = []
    real_open = os.open
    real_writer = csv.writer

    def observe_open(*args, **kwargs):
        descriptor = real_open(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor

    def fail(*args, **kwargs):
        raise OSError("injected " + failure_stage)

    class PartialWriter:
        def __init__(self, *args, **kwargs):
            self.writer = real_writer(*args, **kwargs)

        def writerow(self, row):
            return self.writer.writerow(row)

        def writerows(self, rows):
            self.writer.writerow(next(iter(rows)))
            fail()

    monkeypatch.setattr("base.raw_audio_csv_exporter.os.open", observe_open)
    boundary = {"open": "os.open", "fdopen": "os.fdopen", "write": "csv.writer",
                "replace": "os.replace"}[failure_stage]
    monkeypatch.setattr("base.raw_audio_csv_exporter." + boundary,
                        PartialWriter if failure_stage == "write" else fail)
    with pytest.raises(OSError, match="injected"):
        export_raw_audio_csv(wav, target, (0,), temporary_path=temporary)
    assert target.read_bytes() == b"old csv"
    assert not temporary.exists()
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.parametrize("callback_raises", [False, True])
def test_cleanup_failure_reports_exact_path_without_hiding_export_error(
    export_paths, monkeypatch, callback_raises
):
    wav, target, temporary = export_paths
    cleanup_error = PermissionError("cannot unlink owned temporary")
    diagnostics = []

    def fail_replace(*args):
        raise OSError("publish failed")

    def fail_unlink(path, *args, **kwargs):
        assert path == temporary
        raise cleanup_error

    def on_cleanup(path, error):
        diagnostics.append((path, error))
        if callback_raises:
            raise RuntimeError("diagnostic consumer failed")

    monkeypatch.setattr("base.raw_audio_csv_exporter.os.replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises(OSError, match="publish failed"):
        export_raw_audio_csv(wav, target, (0,), temporary_path=temporary,
                             cleanup_failed=on_cleanup)
    assert diagnostics == [(temporary, cleanup_error)]
    assert temporary.exists()


def test_temporary_created_acknowledges_owned_file_and_closes_on_callback_failure(export_paths):
    wav, target, temporary = export_paths
    observed = []

    def created(path, identity):
        observed.append((path, identity))
        info = temporary.stat()
        assert identity == (info.st_dev, info.st_ino)
        raise RuntimeError("acknowledgment failed")

    with pytest.raises(RuntimeError, match="acknowledgment failed"):
        export_raw_audio_csv(wav, target, (0,), temporary_path=temporary, temporary_created=created)
    assert len(observed) == 1
    assert observed[0][0] == temporary
    assert not temporary.exists()
    wav.unlink()


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
    assert rows[0] == ["time_s", "CH1", "CH3"]
    assert [row[0] for row in rows[1:]] == [
        "0.000000000",
        "0.250000000",
        "0.500000000",
    ]
    np.testing.assert_allclose(
        np.asarray([[float(value) for value in row[1:]] for row in rows[1:]]),
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


def test_product_pcm24_export_preserves_low_amplitude_and_sub_lsb_floor(tmp_path):
    from base.save_data import save_audio_simple
    source = np.array([[0.5, 2.5], [2**-25, -2**-25],
                       [0.001953125, -3]], dtype=np.float32)
    expected_text = [["0.5", "0.999999881"], ["0", "-1.1920929e-07"],
                     ["0.001953125", "-1"]]
    wav, target = tmp_path / "pcm24.wav", tmp_path / "pcm24.csv"
    save_audio_simple(str(wav), source, 8000)
    export_raw_audio_csv(wav, target, (7, 1), block_frames=1)
    with target.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ["time_s", "CH8", "CH2"]
    assert [row[1:] for row in rows[1:]] == expected_text
    decoded, _ = sf.read(wav, dtype="float32", always_2d=True)
    np.testing.assert_array_equal(np.array(expected_text, dtype=np.float32), decoded)
