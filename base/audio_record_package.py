"""Collect existing files belonging to explicitly selected recordings."""

from dataclasses import dataclass
import os
from pathlib import Path

from base.audio_record_filter import parse_audio_filter_metadata
from base.analysis_artifact_paths import sanitize_path_component
from base.raw_audio_csv_zip import raw_csv_zip_path
from consts.running_consts import DEFAULT_DIR


PACKAGE_KINDS = ("wav", "raw_csv", "images", "analysis_csv")


@dataclass(frozen=True)
class AudioPackageFile:
    source: str
    archive_name: str


def _raw_csv_companion(wav_path):
    metadata = parse_audio_filter_metadata(str(wav_path))
    if metadata.project_key is None:
        return None
    parts = wav_path.relative_to(Path(metadata.project_key)).parts
    project_dir = wav_path.parents[len(parts) - 1]
    sample_dir = project_dir / parts[0] / parts[1]
    stem = sanitize_path_component(wav_path.stem, max_length=220)
    return sample_dir / "audio" / "raw_csv" / f"{stem}.csv"


def audio_record_mutation_paths(wav_paths, *, application_root=DEFAULT_DIR):
    """Claim expected companions even before they have been published."""
    paths = set()
    for raw_path in wav_paths:
        wav = Path(os.path.abspath(Path(application_root) / raw_path))
        paths.add(str(wav))
        csv = _raw_csv_companion(wav)
        if csv is not None:
            paths.update((str(csv), str(raw_csv_zip_path(csv))))
    return tuple(sorted(paths))


def collect_audio_package_files(wav_paths, *, application_root=DEFAULT_DIR):
    """Inspect only the selected WAVs and their exact companion locations."""
    application_root = Path(application_root).resolve()
    grouped = {kind: {} for kind in PACKAGE_KINDS}

    def add(kind, path, archive_name):
        if path.is_file():
            grouped[kind][os.path.normcase(str(path))] = AudioPackageFile(
                str(path), archive_name,
            )

    for raw_path in dict.fromkeys(wav_paths):
        wav_path = Path(os.path.abspath(application_root / raw_path))
        metadata = parse_audio_filter_metadata(str(wav_path))
        if metadata.project_key is None:
            try:
                archive_name = wav_path.relative_to(application_root).as_posix()
            except ValueError:
                archive_name = f"wav/{wav_path.name}"
            add("wav", wav_path, archive_name)
            continue

        project_dir = Path(metadata.project_key)
        relative_parts = wav_path.relative_to(project_dir).parts
        # The filter key is case-normalized; keep original directory spelling in ZIPs.
        project_dir = wav_path.parents[len(relative_parts) - 1]
        sample_dir = project_dir / relative_parts[0] / relative_parts[1]
        archive_root = project_dir.parent
        add("wav", wav_path, wav_path.relative_to(archive_root).as_posix())
        raw_csv = _raw_csv_companion(wav_path)
        for companion in (raw_csv, raw_csv_zip_path(raw_csv)):
            if not companion.parent.resolve().is_relative_to(sample_dir.resolve()):
                raise ValueError(f"配套文件路径超出录音样本目录：{companion}")
            add("raw_csv", companion, companion.relative_to(archive_root).as_posix())
        for kind, directory, suffixes in (
            ("images", sample_dir / "images" / wav_path.stem, {".png", ".jpg", ".jpeg"}),
            ("analysis_csv", sample_dir / "csv" / wav_path.stem, {".csv"}),
        ):
            if directory.is_dir():
                for path in sorted(directory.iterdir()):
                    if path.suffix.lower() in suffixes:
                        add(kind, path, path.relative_to(archive_root).as_posix())
    return {kind: tuple(files.values()) for kind, files in grouped.items()}


def select_audio_package_files(grouped, selected_kinds):
    """Deduplicate sources and reject ambiguous ZIP destinations before writing."""
    selected = {}
    destinations = {}
    for kind in PACKAGE_KINDS:
        if kind not in selected_kinds:
            continue
        for entry in grouped[kind]:
            source_key = os.path.normcase(entry.source)
            destination_key = entry.archive_name.casefold()
            previous = destinations.get(destination_key)
            if previous is not None and previous != source_key:
                raise ValueError(f"不同文件的打包路径相同：{entry.archive_name}")
            destinations[destination_key] = source_key
            selected[source_key] = entry
    return tuple(selected.values())
