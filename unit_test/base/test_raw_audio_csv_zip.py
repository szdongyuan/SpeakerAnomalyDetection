from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import struct
import zipfile

import pytest

from base.raw_audio_csv_zip import archive_raw_audio_csv, raw_csv_zip_path


@pytest.mark.parametrize('original', [
    b'', b'\xef\xbb\xbftime_s,CH1\n',
    b'\xef\xbb\xbftime_s,CH1\n0.000000000,0.5\n',
    b'0.000000000,0.5\n' * 160000,
], ids=['empty', 'header', 'one-row', 'cross-block'])
def test_zip_appends_extension_and_preserves_member(tmp_path, original):
    source = tmp_path / '中文 example.v2.csv'
    source.write_bytes(original)
    temporary = tmp_path / '.owned.zip.tmp'
    result = archive_raw_audio_csv(source, temporary_path=temporary)
    target = tmp_path / '中文 example.v2.csv.zip'
    assert raw_csv_zip_path(source) == target
    assert result.archive_path == str(target)
    assert result.csv_bytes == len(original)
    assert result.archive_bytes == target.stat().st_size
    assert not source.exists()
    assert not temporary.exists()
    assert not result.csv_retained
    assert result.cleanup_diagnostics == ()
    for phase in ('zip_write', 'zip_verify', 'zip_publish', 'csv_cleanup'):
        assert getattr(result, phase + '_seconds') >= 0
    with pytest.raises(FrozenInstanceError):
        result.csv_retained = True
    with zipfile.ZipFile(target) as archive:
        assert archive.namelist() == [source.name]
        assert archive.read(source.name) == original
        assert archive.getinfo(source.name).compress_type == zipfile.ZIP_DEFLATED


@pytest.fixture
def paths(tmp_path):
    source = tmp_path / 'example.csv'
    source.write_bytes(b'time_s,CH1\n0,0.5\n')
    target = raw_csv_zip_path(source)
    target.write_bytes(b'old zip')
    witness = tmp_path / 'old-zip-witness'
    os.link(target, witness)
    return source, target, tmp_path / '.owned.zip.tmp', witness


@pytest.mark.parametrize('choice', ['source', 'target', 'directory', 'outside'])
def test_invalid_temporary_path(paths, choice):
    source, target, temporary, witness = paths
    bad = {'source': source, 'target': target, 'directory': source.parent,
           'outside': source.parent / 'outside' / temporary.name}[choice]
    with pytest.raises(ValueError, match='temporary'):
        archive_raw_audio_csv(source, temporary_path=bad)
    assert source.exists()
    assert target.read_bytes() == witness.read_bytes() == b'old zip'


def test_collision_is_not_owned_or_deleted(paths):
    source, target, temporary, witness = paths
    temporary.write_bytes(b'neighbor task')
    with pytest.raises(FileExistsError):
        archive_raw_audio_csv(source, temporary_path=temporary)
    assert temporary.read_bytes() == b'neighbor task'
    assert source.exists()
    assert target.read_bytes() == witness.read_bytes() == b'old zip'


@pytest.mark.parametrize('boundary', [
    'create', 'fdopen', 'read', 'write', 'fsync', 'central_directory', 'verify', 'replace',
    'ownership_callback',
])
def test_failure_preserves_csv_and_old_zip_and_closes_handles(paths, monkeypatch, boundary):
    source, target, temporary, witness = paths
    original = source.read_bytes()
    handles = []
    descriptors = []
    real_open = Path.open
    real_fdopen = os.fdopen
    real_os_open = os.open

    def fail(*args, **kwargs):
        raise OSError('injected ' + boundary)

    class Reader:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def fileno(self):
            return self.handle.fileno()

        def read(self, size):
            fail()

    def observe_open(path, *args, **kwargs):
        handle = real_open(path, *args, **kwargs)
        handles.append(handle)
        return Reader(handle) if path == source and boundary == 'read' else handle

    def observe_fdopen(*args, **kwargs):
        if boundary == 'fdopen':
            fail()
        handle = real_fdopen(*args, **kwargs)
        handles.append(handle)
        return handle

    def observe_os_open(*args, **kwargs):
        if boundary == 'create':
            fail()
        fd = real_os_open(*args, **kwargs)
        descriptors.append(fd)
        return fd

    with monkeypatch.context() as patch:
        patch.setattr(Path, 'open', observe_open)
        patch.setattr(os, 'open', observe_os_open)
        patch.setattr(os, 'fdopen', observe_fdopen)
        if boundary == 'write':
            patch.setattr(zipfile._ZipWriteFile, 'write', fail)
        if boundary == 'central_directory':
            patch.setattr(zipfile.ZipFile, '_write_end_record', fail)
        if boundary == 'verify':
            patch.setattr(zipfile.ZipExtFile, 'read', fail)
        if boundary == 'fsync':
            patch.setattr(os, 'fsync', fail)
        if boundary == 'replace':
            patch.setattr(os, 'replace', fail)
        with pytest.raises(OSError, match='injected ' + boundary):
            archive_raw_audio_csv(source, temporary_path=temporary,
                                  temporary_created=fail if boundary == 'ownership_callback' else None)
    assert all(handle.closed for handle in handles)
    for fd in descriptors:
        with pytest.raises(OSError):
            os.fstat(fd)
    assert source.read_bytes() == original
    assert target.read_bytes() == witness.read_bytes() == b'old zip'
    assert not temporary.exists()


@pytest.mark.parametrize('change', ['replace', 'deleted', 'unlink_error', 'modified'])
def test_csv_cleanup_after_publication_is_warning_success(paths, monkeypatch, change):
    source, target, temporary, witness = paths
    original = source.read_bytes()
    unlink = Path.unlink

    def on_stage(stage):
        if stage != 'csv_cleanup':
            return
        with zipfile.ZipFile(target) as archive:
            assert archive.read(source.name) == original
        if change == 'replace':
            replacement = source.with_suffix('.replacement')
            replacement.write_bytes(b'new csv')
            os.replace(replacement, source)
        elif change == 'deleted':
            source.unlink()
        elif change == 'modified':
            source.write_bytes(b'changed in place')

    def denied(path, *args, **kwargs):
        if path == source:
            raise PermissionError('CSV busy')
        return unlink(path, *args, **kwargs)

    if change == 'unlink_error':
        monkeypatch.setattr(Path, 'unlink', denied)
    result = archive_raw_audio_csv(source, temporary_path=temporary, stage_changed=on_stage)
    assert result.csv_retained == (change != 'deleted')
    assert bool(result.cleanup_diagnostics) == (change != 'deleted')
    if result.cleanup_diagnostics:
        assert str(source) in result.cleanup_diagnostics[0]
    assert source.exists() == result.csv_retained
    assert witness.read_bytes() == b'old zip'
    assert not temporary.exists()


@pytest.mark.parametrize('damage', ['extra', 'name', 'size', 'hash', 'truncated', 'crc'])
def test_corrupt_zip_is_never_published(paths, damage):
    source, target, temporary, witness = paths
    original = source.read_bytes()

    def corrupt(stage):
        if stage != 'zip_verify':
            return
        if damage == 'truncated':
            temporary.write_bytes(temporary.read_bytes()[:20])
            return
        if damage == 'crc':
            data = bytearray(temporary.read_bytes())
            central = data.index(b'PK\x01\x02')
            data[central + 16] ^= 1
            temporary.write_bytes(data)
            return
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
            name = 'wrong.csv' if damage == 'name' else source.name
            content = original + b'x' if damage == 'size' else original
            content = b'x' * len(original) if damage == 'hash' else content
            archive.writestr(name, content)
            if damage == 'extra':
                archive.writestr('extra', b'')

    with pytest.raises((ValueError, zipfile.BadZipFile)):
        archive_raw_audio_csv(source, temporary_path=temporary, stage_changed=corrupt)
    assert source.read_bytes() == original
    assert target.read_bytes() == witness.read_bytes() == b'old zip'
    assert not temporary.exists()


def test_temporary_cleanup_failure_does_not_mask_primary_error(paths, monkeypatch):
    source, target, temporary, witness = paths
    primary = OSError('publish failure')
    unlink = Path.unlink

    def fail_publish(*args):
        raise primary

    def fail_cleanup(path, *args, **kwargs):
        if path == temporary:
            raise PermissionError('temporary busy')
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, 'replace', fail_publish)
    monkeypatch.setattr(Path, 'unlink', fail_cleanup)
    with pytest.raises(OSError) as caught:
        archive_raw_audio_csv(source, temporary_path=temporary)
    assert caught.value is primary
    assert any(str(temporary) in note and 'temporary busy' in note
               for note in caught.value.__notes__)
    assert temporary.exists()
    assert source.exists()
    assert target.read_bytes() == witness.read_bytes() == b'old zip'


def test_replaced_temporary_is_not_deleted(paths):
    source, target, temporary, witness = paths

    def replace_owned(stage):
        if stage == 'zip_verify':
            replacement = temporary.with_suffix('.neighbor')
            replacement.write_bytes(b'neighbor')
            os.replace(replacement, temporary)
            raise OSError('verification interrupted')

    with pytest.raises(OSError, match='verification interrupted') as caught:
        archive_raw_audio_csv(source, temporary_path=temporary, stage_changed=replace_owned)
    assert temporary.read_bytes() == b'neighbor'
    assert any('identity' in note for note in caught.value.__notes__)


def test_stages_ownership_bounded_io_and_zip64(paths, monkeypatch):
    source, target, temporary, witness = paths
    original = b'0,0.5\n' * 400000
    source.write_bytes(original)
    stages = []
    acknowledged = []
    read_sizes = []
    verify_sizes = []
    write_sizes = []
    handles = []
    real_path_open = Path.open
    real_read = zipfile.ZipExtFile.read
    real_write = zipfile._ZipWriteFile.write
    real_compressor = zipfile._get_compressor
    compressor_settings = []

    class BoundedReader:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def fileno(self):
            return self.handle.fileno()

        def read(self, size=-1):
            assert 0 < size <= 1024 * 1024
            read_sizes.append(size)
            return self.handle.read(size)

    def observe_open(path, *args, **kwargs):
        handle = real_path_open(path, *args, **kwargs)
        handles.append(handle)
        return BoundedReader(handle) if path == source else handle

    def observe_read(member, size=-1):
        assert 0 < size <= 1024 * 1024
        verify_sizes.append(size)
        return real_read(member, size)

    def observe_write(member, data):
        write_sizes.append(len(data))
        assert len(data) <= 1024 * 1024
        return real_write(member, data)

    def compressor(kind, level):
        compressor_settings.append((kind, level))
        return real_compressor(kind, level)

    def created(path, identity):
        info = path.stat()
        assert identity == (info.st_dev, info.st_ino)
        acknowledged.append(path)

    def stage_changed(stage):
        stages.append(stage)
        if stage != 'zip_write':
            assert all(handle.closed for handle in handles)
        if stage == 'zip_publish':
            # Windows opening exclusively by renaming proves verifier closed its reader.
            moved = temporary.with_suffix('.closed')
            temporary.rename(moved)
            moved.rename(temporary)

    with monkeypatch.context() as patch:
        patch.setattr(Path, 'open', observe_open)
        patch.setattr(zipfile.ZipExtFile, 'read', observe_read)
        patch.setattr(zipfile._ZipWriteFile, 'write', observe_write)
        patch.setattr(zipfile, '_get_compressor', compressor)
        # Exercise real large-member branching without allocating gigabytes.
        patch.setattr(zipfile, 'ZIP64_LIMIT', 1024)
        result = archive_raw_audio_csv(source, temporary_path=temporary,
                                       temporary_created=created, stage_changed=stage_changed)
    assert stages == ['zip_write', 'zip_verify', 'zip_publish', 'csv_cleanup']
    assert acknowledged == [temporary]
    assert len(read_sizes) >= 4 and len(verify_sizes) >= 4
    assert sum(write_sizes) == len(original) == result.csv_bytes
    assert compressor_settings == [(zipfile.ZIP_DEFLATED, 1)]
    data = target.read_bytes()
    assert data[:4] == b'PK\x03\x04'
    assert struct.unpack_from('<II', data, 18) == (0xffffffff, 0xffffffff)
    assert b'PK\x06\x06' in data  # Real ZIP64 end of central directory.
    with zipfile.ZipFile(target) as archive:
        assert archive.read(source.name) == original
    assert witness.read_bytes() == b'old zip'


def test_zip_changed_after_verification_is_not_published(paths):
    source, target, temporary, witness = paths

    def change(stage):
        if stage == 'zip_publish':
            temporary.write_bytes(b'no longer a verified zip')

    with pytest.raises(ValueError, match='changed'):
        archive_raw_audio_csv(source, temporary_path=temporary, stage_changed=change)
    assert source.exists()
    assert target.read_bytes() == witness.read_bytes() == b'old zip'
    assert not temporary.exists()


def test_verification_counts_actual_stream_bytes(paths, monkeypatch):
    source, target, temporary, witness = paths
    real_read = zipfile.ZipExtFile.read

    def incomplete_read(member, size):
        return real_read(member, size)[1:]

    monkeypatch.setattr(zipfile.ZipExtFile, 'read', incomplete_read)
    with pytest.raises(ValueError, match='content'):
        archive_raw_audio_csv(source, temporary_path=temporary)
    assert source.exists()
    assert target.read_bytes() == witness.read_bytes() == b'old zip'
    assert not temporary.exists()


def test_unknown_temporary_identity_preserved_with_diagnostic(paths, monkeypatch):
    source, target, temporary, witness = paths
    real_fstat = os.fstat
    count = 0

    def fail_identity(fd):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError('temporary identity unavailable')
        return real_fstat(fd)

    monkeypatch.setattr(os, 'fstat', fail_identity)
    with pytest.raises(OSError, match='temporary identity unavailable') as caught:
        archive_raw_audio_csv(source, temporary_path=temporary)
    assert temporary.exists()
    assert any(str(temporary) in note and 'unknown' in note for note in caught.value.__notes__)
    temporary.unlink()  # All descriptors must have closed even before ownership acknowledgment.
    assert source.exists()


def test_csv_stat_failure_returns_cleanup_warning(paths, monkeypatch):
    source, target, temporary, witness = paths
    real_lstat = Path.lstat

    def deny_stat(path):
        if path == source:
            raise PermissionError('cannot inspect CSV')
        return real_lstat(path)

    def stage(stage):
        if stage == 'csv_cleanup':
            monkeypatch.setattr(Path, 'lstat', deny_stat)

    result = archive_raw_audio_csv(source, temporary_path=temporary, stage_changed=stage)
    assert result.csv_retained
    assert 'cannot inspect CSV' in result.cleanup_diagnostics[0]
    assert source.exists() and target.exists()


def test_handles_close_before_fsync_verification_publication_and_cleanup(paths, monkeypatch):
    source, target, temporary, witness = paths
    archives = []
    members = []
    outputs = []
    events = []
    real_archive = zipfile.ZipFile
    real_fdopen = os.fdopen
    real_fsync = os.fsync

    class ObservedArchive(real_archive):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            archives.append(self)

        def open(self, *args, **kwargs):
            member = super().open(*args, **kwargs)
            members.append(member)
            return member

        def _write_end_record(self):
            assert members[0].closed
            super()._write_end_record()
            events.append('central_directory')

    def fdopen(*args, **kwargs):
        output = real_fdopen(*args, **kwargs)
        outputs.append(output)
        return output

    def fsync(fd):
        assert events == ['central_directory']
        assert archives[0].fp is None
        assert not outputs[0].closed
        real_fsync(fd)
        events.append('fsync')

    def stage(stage):
        if stage != 'zip_write':
            assert events == ['central_directory', 'fsync']
            assert all(output.closed for output in outputs)
            assert all(archive.fp is None for archive in archives)
            assert all(member.closed for member in members)

    monkeypatch.setattr(zipfile, 'ZipFile', ObservedArchive)
    monkeypatch.setattr(os, 'fdopen', fdopen)
    monkeypatch.setattr(os, 'fsync', fsync)
    archive_raw_audio_csv(source, temporary_path=temporary, stage_changed=stage)
    assert len(archives) == len(members) == 2
