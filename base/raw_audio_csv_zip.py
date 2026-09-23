"""Bounded, verified ZIP publication for a completed raw-audio CSV."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import sys
from time import perf_counter
import zipfile


@dataclass(frozen=True)
class CsvZipResult:
    archive_path: str
    csv_bytes: int
    archive_bytes: int
    csv_retained: bool
    cleanup_diagnostics: tuple[str, ...]
    zip_write_seconds: float
    zip_verify_seconds: float
    zip_publish_seconds: float
    csv_cleanup_seconds: float


def raw_csv_zip_path(csv_path) -> Path:
    return Path(str(csv_path) + '.zip')


def _identity(info):
    return info.st_dev, info.st_ino


def _regular_file(info):
    return (stat.S_ISREG(info.st_mode)
            and not getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _file_state(info):
    # Windows fstat and lstat expose different ctime semantics on Python 3.12.
    return _identity(info), info.st_size, info.st_mtime_ns


def _cleanup_csv(source, original_info):
    try:
        current = source.lstat()
        if not _regular_file(current) or _file_state(current) != _file_state(original_info):
            return True, (f'CSV cleanup skipped: identity or state changed: {source}',)
        source.unlink()
    except FileNotFoundError:
        return False, ()
    except OSError as error:
        return True, (f'CSV cleanup failed: {source}: {error}',)
    return False, ()


def _cleanup_temporary(temporary, identity, primary_error):
    """Reclaim only an acknowledged identity; keep the original failure diagnosable."""
    diagnostic = None
    try:
        current = temporary.lstat()
        if identity is None or not _regular_file(current) or _identity(current) != identity:
            diagnostic = f'ZIP temporary cleanup skipped: identity changed or unknown: {temporary}'
        else:
            temporary.unlink()
    except FileNotFoundError:
        return
    except OSError as error:
        diagnostic = f'ZIP temporary cleanup failed: {temporary}: {error}'
    if diagnostic is not None:
        if primary_error is None:
            raise OSError(diagnostic)
        primary_error.add_note(diagnostic)


def _verify_archive(temporary, member_name, csv_bytes, digest, block_bytes):
    with zipfile.ZipFile(temporary, 'r') as archive:
        entries = archive.infolist()
        if len(entries) != 1 or entries[0].filename != member_name:
            raise ValueError('ZIP must contain only the original CSV member')
        if entries[0].file_size != csv_bytes:
            raise ValueError('ZIP member byte count does not match CSV')
        verified_digest = hashlib.sha256()
        verified_bytes = 0
        with archive.open(entries[0]) as member:
            while chunk := member.read(block_bytes):
                verified_digest.update(chunk)
                verified_bytes += len(chunk)
        if verified_bytes != csv_bytes or verified_digest.digest() != digest:
            raise ValueError('ZIP member content does not match CSV')


def archive_raw_audio_csv(
    csv_path, *, temporary_path, temporary_created=None, stage_changed=None,
) -> CsvZipResult:
    source = Path(csv_path)
    target = raw_csv_zip_path(source)
    temporary = Path(temporary_path)
    if (temporary.resolve().parent != target.parent.resolve()
            or temporary.resolve() in (source.resolve(), target.resolve())
            or temporary.is_dir()):
        raise ValueError('temporary path must be a distinct file in the target directory')
    block_bytes = 1024 * 1024
    owned = False
    temporary_identity = None
    try:
        started = perf_counter()
        if stage_changed is not None:
            stage_changed('zip_write')
        with source.open('rb') as csv_file:
            csv_info = os.fstat(csv_file.fileno())
            fd = os.open(temporary, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            owned = True
            try:
                temporary_identity = _identity(os.fstat(fd))
                if temporary_created is not None:
                    temporary_created(temporary, temporary_identity)
                output = os.fdopen(fd, 'w+b')
                fd = None  # output owns and closes the descriptor from here.
                digest = hashlib.sha256()
                csv_bytes = 0
                with output:
                    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED,
                                         compresslevel=1, allowZip64=True) as archive:
                        with archive.open(source.name, 'w', force_zip64=True) as member:
                            while chunk := csv_file.read(block_bytes):
                                member.write(chunk)
                                digest.update(chunk)
                                csv_bytes += len(chunk)
                    output.flush()
                    os.fsync(output.fileno())
            finally:
                if fd is not None:
                    os.close(fd)
        write_seconds = perf_counter() - started
        started = perf_counter()
        if stage_changed is not None:
            stage_changed('zip_verify')
        verification_info = temporary.lstat()
        if not _regular_file(verification_info) or _identity(verification_info) != temporary_identity:
            raise ValueError(f'ZIP temporary identity changed: {temporary}')
        _verify_archive(temporary, source.name, csv_bytes, digest.digest(), block_bytes)
        verify_seconds = perf_counter() - started
        started = perf_counter()
        if stage_changed is not None:
            stage_changed('zip_publish')
        archive_info = temporary.lstat()
        if (not _regular_file(archive_info)
                or _file_state(archive_info) != _file_state(verification_info)):
            raise ValueError(f'ZIP temporary identity or state changed: {temporary}')
        archive_bytes = archive_info.st_size
        os.replace(temporary, target)
        owned = False
        publish_seconds = perf_counter() - started
        started = perf_counter()
        if stage_changed is not None:
            stage_changed('csv_cleanup')
        retained, diagnostics = _cleanup_csv(source, csv_info)
        cleanup_seconds = perf_counter() - started
        return CsvZipResult(
            archive_path=str(target), csv_bytes=csv_bytes, archive_bytes=archive_bytes,
            csv_retained=retained, cleanup_diagnostics=diagnostics,
            zip_write_seconds=write_seconds, zip_verify_seconds=verify_seconds,
            zip_publish_seconds=publish_seconds, csv_cleanup_seconds=cleanup_seconds,
        )
    finally:
        if owned:
            _cleanup_temporary(temporary, temporary_identity, sys.exception())
