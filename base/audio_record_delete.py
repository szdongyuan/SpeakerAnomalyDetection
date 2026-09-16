"""Plan and delete only the confirmed files of selected audio records."""

from dataclasses import dataclass
import os
from pathlib import Path

from base.audio_record_filter import parse_audio_filter_metadata
from base.audio_record_package import PACKAGE_KINDS, collect_audio_package_files
from consts import error_code
from consts.running_consts import DEFAULT_DIR


@dataclass(frozen=True)
class AudioRecordDeletion:
    record_id: str | int
    wav_path: str
    files: dict[str, tuple[str, ...]]
    sample_directory: str | None


def _validate_companion_path(filename, sample_directory):
    if not Path(filename).parent.resolve().is_relative_to(Path(sample_directory).resolve()):
        raise ValueError(f"配套文件路径超出录音样本目录：{filename}")


def plan_audio_record_deletion(rows, *, application_root=DEFAULT_DIR):
    plans = []
    for row in rows:
        wav_path = os.path.abspath(os.path.join(application_root, row[1]))
        grouped = collect_audio_package_files([wav_path], application_root=application_root)
        files = {kind: tuple(entry.source for entry in entries) for kind, entries in grouped.items()}
        metadata = parse_audio_filter_metadata(wav_path)
        sample_directory = None
        if metadata.project_key is not None:
            project = Path(metadata.project_key)
            parts = Path(wav_path).relative_to(project).parts
            sample_directory = str(project / parts[0] / parts[1])
            for kind in ("raw_csv", "images", "analysis_csv"):
                for filename in files[kind]:
                    _validate_companion_path(filename, sample_directory)
        plans.append(AudioRecordDeletion(row[0], wav_path, files, sample_directory))
    return tuple(plans)


def count_audio_deletion_files(plans):
    return {
        kind: len({os.path.normcase(path) for plan in plans for path in plan.files[kind]})
        for kind in PACKAGE_KINDS
    }


def delete_audio_recordings(plans, recording_manager):
    """Keep a record for retry when any file or its database deletion fails."""
    deleted_ids = []
    errors = []
    for plan in plans:
        try:
            for kind in ("raw_csv", "images", "analysis_csv", "wav"):
                for filename in plan.files[kind]:
                    if kind != "wav":
                        _validate_companion_path(filename, plan.sample_directory)
                    Path(filename).unlink(missing_ok=True)
        except (OSError, ValueError) as error:
            errors.append(f"{Path(plan.wav_path).name}：{error}")
            continue
        code, message = recording_manager.delete_audio_at_id_list([plan.record_id])
        if code == error_code.OK:
            deleted_ids.append(plan.record_id)
        else:
            errors.append(f"{Path(plan.wav_path).name}：数据库记录删除失败，{message}")
    return deleted_ids, errors
