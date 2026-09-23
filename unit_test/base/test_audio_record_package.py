from pathlib import Path
from zipfile import ZipFile

import pytest

from base.audio_record_package import (
    PACKAGE_KINDS, collect_audio_package_files, select_audio_package_files,
)
from base.file_ops import FileOps


def make_recording(root, project="ProjectA", sample="003", legacy=False):
    stem = f"{project}_Model1_{sample}_Port1_R0002_High_20260915-093000"
    directory = root / project / "Model1" / sample
    files = {
        "wav": directory / ("wav" if legacy else "audio/wav") / f"{stem}.wav",
        "raw_csv": directory / "audio/raw_csv" / f"{stem}.csv",
        "images": directory / "images" / stem / "SPL_CH1.png",
        "analysis_csv": directory / "csv" / stem / "SPL_CH1.csv",
    }
    for kind, path in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{project}/{sample}/{kind}".encode())
    return files


@pytest.mark.parametrize("legacy,sample", [(False, "003"), (True, "003"), (True, "audio")])
def test_exact_companions_and_actual_zip_preserve_tree_and_contents(tmp_path, legacy, sample):
    root = tmp_path / "results"
    selected = make_recording(root, legacy=legacy, sample=sample)
    make_recording(root, project="ProjectB", sample=sample)
    make_recording(root, sample="004")
    selected["images"].with_name("notes.txt").write_text("not an image")
    selected["analysis_csv"].with_name("unfinished.csv.tmp").write_text("incomplete")
    groups = collect_audio_package_files([str(selected["wav"])])
    assert {kind: len(files) for kind, files in groups.items()} == dict.fromkeys(PACKAGE_KINDS, 1)
    entries = select_audio_package_files(groups, set(PACKAGE_KINDS))
    progress = []
    output = tmp_path / "selected.zip"
    FileOps.create_zip_with_files(
        [entry.source for entry in entries], str(output),
        archive_names={entry.source: entry.archive_name for entry in entries},
        progress_callback=lambda current, total: progress.append((current, total)),
    )
    with ZipFile(output) as archive:
        assert set(archive.namelist()) == {p.relative_to(root).as_posix() for p in selected.values()}
        for path in selected.values():
            assert archive.read(path.relative_to(root).as_posix()) == path.read_bytes()
    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_csv_types_are_independent_and_same_named_results_do_not_collide(tmp_path):
    first = make_recording(tmp_path, sample="003")
    second = make_recording(tmp_path, sample="004")
    groups = collect_audio_package_files([str(first["wav"]), str(second["wav"])])
    raw = select_audio_package_files(groups, {"raw_csv"})
    analysis = select_audio_package_files(groups, {"analysis_csv"})
    assert {Path(entry.source) for entry in raw} == {first["raw_csv"], second["raw_csv"]}
    assert {Path(entry.source) for entry in analysis} == {first["analysis_csv"], second["analysis_csv"]}
    assert len({entry.archive_name for entry in analysis}) == 2


def test_no_generated_companions_or_duplicate_sources(tmp_path):
    files = make_recording(tmp_path)
    for kind in ("raw_csv", "images", "analysis_csv"):
        files[kind].unlink()
    groups = collect_audio_package_files([str(files["wav"])] * 2)
    assert len(groups["wav"]) == 1
    assert all(not groups[kind] for kind in PACKAGE_KINDS if kind != "wav")
    assert select_audio_package_files(groups, {"analysis_csv"}) == ()


def test_relative_old_wav_and_missing_wav_keep_available_companions(tmp_path):
    old = tmp_path / "stored" / "OK" / "old.wav"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"RIFF")
    groups = collect_audio_package_files(["stored/OK/old.wav"], application_root=tmp_path)
    assert groups["wav"][0].archive_name == "stored/OK/old.wav"
    files = make_recording(tmp_path)
    files["wav"].unlink()
    groups = collect_audio_package_files([str(files["wav"])])
    assert groups["wav"] == ()
    assert len(groups["analysis_csv"]) == 1


def test_same_project_name_in_different_roots_cannot_silently_merge(tmp_path):
    first = make_recording(tmp_path / "one")
    second = make_recording(tmp_path / "two")
    groups = collect_audio_package_files([str(first["wav"]), str(second["wav"])])
    with pytest.raises(ValueError, match="打包路径相同"):
        select_audio_package_files(groups, {"images"})


@pytest.mark.parametrize('retain_csv', [False, True])
@pytest.mark.parametrize('stem_variant', ['ordinary', 'spaces', 'long'])
def test_exact_raw_zip_reuses_producer_name_and_preserves_nested_bytes(tmp_path, retain_csv, stem_variant):
    from base.analysis_artifact_paths import sanitize_path_component
    root = tmp_path / '中文 results'
    files = make_recording(root)
    wav = files['wav']
    stem = wav.stem
    if stem_variant == 'spaces':
        stem = stem.replace('High', 'High speed')
    elif stem_variant == 'long':
        stem = stem.replace('High', 'H' * 180)
    renamed = wav.with_name(stem + '.wav')
    wav.rename(renamed)
    files['raw_csv'].unlink()
    csv = files['raw_csv'].with_name(sanitize_path_component(stem, max_length=220) + '.csv')
    archive = Path(str(csv) + '.zip')
    with ZipFile(archive, 'w') as inner:
        inner.writestr(csv.name, b'original csv bytes')
    if retain_csv:
        csv.write_bytes(b'original csv bytes')
    neighbor = archive.with_name('neighbor.csv.zip')
    neighbor.write_bytes(b'leave alone')
    grouped = collect_audio_package_files([str(renamed)])
    entries = grouped['raw_csv']
    assert {Path(entry.source) for entry in entries} == ({csv, archive} if retain_csv else {archive})
    output = tmp_path / 'outer.zip'
    FileOps.create_zip_with_files([entry.source for entry in entries], str(output),
        archive_names={entry.source: entry.archive_name for entry in entries})
    with ZipFile(output) as outer:
        assert outer.read(archive.relative_to(root).as_posix()) == archive.read_bytes()



def test_zip_companion_directory_cannot_escape_sample(tmp_path, monkeypatch):
    files = make_recording(tmp_path)
    archive = Path(str(files['raw_csv']) + '.zip')
    archive.write_bytes(b'zip')
    original = Path.resolve
    def resolve(path, *args, **kwargs):
        if path == archive.parent:
            return tmp_path / 'outside'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'resolve', resolve)
    with pytest.raises(ValueError, match='超出录音样本目录'):
        collect_audio_package_files([str(files['wav'])])
    assert archive.read_bytes() == b'zip'
