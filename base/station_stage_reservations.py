"""Filesystem operations scoped to the supervisor's verified staging directory."""

from contextlib import contextmanager
import os
import stat


class StageCreationError(OSError):
    """Setup failed after exclusive creation and cleanup still needs attention."""

    def __init__(self, path, error, *, identity=None, rollback=None):
        self.identity = identity
        detail = f"recording stage setup failed after creating {path}: {error}"
        if rollback is not None:
            detail += f" (rollback failed: {rollback})"
        super().__init__(detail)


if os.name == "nt":
    import ctypes
    from ctypes import wintypes
    import msvcrt

    # Win32 opens the directory; the NT file API can use that handle as the
    # root for an exclusive file creation even if the directory is renamed.
    _FILE_LIST_DIRECTORY = 0x0001
    _FILE_READ_DATA = 0x0001
    _FILE_WRITE_DATA = 0x0002
    _FILE_READ_ATTRIBUTES = 0x0080
    _DELETE = 0x00010000
    _SYNCHRONIZE = 0x00100000
    _FILE_SHARE_READ = 0x0001
    _FILE_SHARE_READ_WRITE = 0x0003
    _FILE_SHARE_ALL = 0x0007  # read, write, and delete
    _OPEN_EXISTING = 3
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    _OBJ_CASE_INSENSITIVE = 0x0040
    _FILE_CREATE = 2
    _FILE_OPEN = 1
    _FILE_DIRECTORY_FILE = 0x0001
    _FILE_OPEN_REPARSE_POINT = 0x00200000
    _FILE_NON_DIRECTORY_FILE = 0x0040
    _FILE_SYNCHRONOUS_IO_NONALERT = 0x0020
    _FILE_RENAME_INFORMATION = 10
    _FILE_DISPOSITION_INFO = 4
    _FILE_BASIC_INFO = 0

    class _UnicodeString(ctypes.Structure):
        _fields_ = [("length", wintypes.USHORT), ("capacity", wintypes.USHORT),
                    ("buffer", wintypes.LPWSTR)]

    class _ObjectAttributes(ctypes.Structure):
        _fields_ = [("length", wintypes.ULONG), ("root_directory", wintypes.HANDLE),
                    ("name", ctypes.POINTER(_UnicodeString)), ("attributes", wintypes.ULONG),
                    ("security_descriptor", wintypes.LPVOID), ("security_qos", wintypes.LPVOID)]

    class _IoStatusBlock(ctypes.Structure):
        _fields_ = [("status_or_pointer", wintypes.LPVOID), ("information", ctypes.c_size_t)]

    class _RenameInformation(ctypes.Structure):
        _fields_ = [("replace", ctypes.c_ubyte), ("root_directory", wintypes.HANDLE),
                    ("length", wintypes.ULONG), ("name", ctypes.c_wchar * 1)]

    class _FileBasicInformation(ctypes.Structure):
        _fields_ = [("creation_time", ctypes.c_int64), ("last_access_time", ctypes.c_int64),
                    ("last_write_time", ctypes.c_int64), ("change_time", ctypes.c_int64),
                    ("attributes", wintypes.DWORD)]

    _kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    _open_directory_handle = _kernel.CreateFileW
    _open_directory_handle.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                       wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    _open_directory_handle.restype = wintypes.HANDLE
    _close_handle = _kernel.CloseHandle
    _close_handle.argtypes = [wintypes.HANDLE]
    _close_handle.restype = wintypes.BOOL
    _set_file_information = _kernel.SetFileInformationByHandle
    _set_file_information.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    _set_file_information.restype = wintypes.BOOL
    _get_file_information = _kernel.GetFileInformationByHandleEx
    _get_file_information.argtypes = _set_file_information.argtypes
    _get_file_information.restype = wintypes.BOOL

    _nt = ctypes.WinDLL("ntdll")
    _create_relative_file = _nt.NtCreateFile
    _create_relative_file.argtypes = [
        ctypes.POINTER(wintypes.HANDLE), wintypes.ULONG, ctypes.POINTER(_ObjectAttributes),
        ctypes.POINTER(_IoStatusBlock), wintypes.LPVOID, wintypes.ULONG, wintypes.ULONG,
        wintypes.ULONG, wintypes.ULONG, wintypes.LPVOID, wintypes.ULONG,
    ]
    _create_relative_file.restype = wintypes.LONG
    _set_nt_information = _nt.NtSetInformationFile
    _set_nt_information.argtypes = [wintypes.HANDLE, ctypes.POINTER(_IoStatusBlock),
                                    wintypes.LPVOID, wintypes.ULONG, ctypes.c_int]
    _set_nt_information.restype = wintypes.LONG
    _status_to_winerror = _nt.RtlNtStatusToDosError
    _status_to_winerror.argtypes = [wintypes.LONG]
    _status_to_winerror.restype = wintypes.ULONG

    def _windows_error(code, path):
        return OSError(0, ctypes.FormatError(code).strip(), path, code)

    def _take_handle(handle, flags):
        try:
            return msvcrt.open_osfhandle(handle, flags | os.O_BINARY)
        except OSError:
            _close_handle(handle)
            raise

    def _open_directory(path, *, delete=False, pin=False):
        access = _FILE_LIST_DIRECTORY | _FILE_READ_ATTRIBUTES | (_DELETE if delete else 0)
        share = _FILE_SHARE_READ_WRITE if pin else _FILE_SHARE_ALL
        handle = _open_directory_handle(
            path, access, share,
            None, _OPEN_EXISTING, _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT, None,
        )
        if handle == ctypes.c_void_p(-1).value:
            raise _windows_error(ctypes.get_last_error(), path)
        return _take_handle(handle, os.O_RDONLY)

    def _open_relative_handle(directory, path, access, disposition, kind, share):
        name = os.path.basename(path)
        buffer = ctypes.create_unicode_buffer(name)
        capacity = ctypes.sizeof(buffer)
        unicode_name = _UnicodeString(capacity - ctypes.sizeof(ctypes.c_wchar), capacity,
                                      ctypes.cast(buffer, wintypes.LPWSTR))
        attributes = _ObjectAttributes(ctypes.sizeof(_ObjectAttributes), msvcrt.get_osfhandle(directory),
                                       ctypes.pointer(unicode_name), _OBJ_CASE_INSENSITIVE, None, None)
        handle, io_status = wintypes.HANDLE(), _IoStatusBlock()
        status = _create_relative_file(
            ctypes.byref(handle), access | _FILE_READ_ATTRIBUTES | _SYNCHRONIZE,
            ctypes.byref(attributes), ctypes.byref(io_status), None, 0, share, disposition,
            _FILE_OPEN_REPARSE_POINT | kind | _FILE_SYNCHRONOUS_IO_NONALERT,
            None, 0,
        )
        if status < 0:
            raise _windows_error(_status_to_winerror(status), path)
        return handle.value

    def _open_relative_entry(directory, path, access, disposition, kind, *, share=_FILE_SHARE_ALL):
        handle = _open_relative_handle(directory, path, access, disposition, kind, share)
        return _take_handle(handle, os.O_WRONLY if access == _FILE_WRITE_DATA else os.O_RDONLY)

    def _open_relative(directory, path):
        return _open_relative_entry(directory, path, _FILE_WRITE_DATA, _FILE_CREATE,
                                    _FILE_NON_DIRECTORY_FILE)

    def _create_directory_entry(directory, path):
        handle = _open_relative_handle(directory, path, _FILE_LIST_DIRECTORY | _DELETE,
                                       _FILE_CREATE, _FILE_DIRECTORY_FILE, _FILE_SHARE_ALL)
        try:
            return msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
        except OSError as exc:
            rollback_error = None
            try:
                _mark_delete(handle, path)
            except OSError as rollback:
                rollback_error = rollback
            if not _close_handle(handle) and rollback_error is None:
                rollback_error = _windows_error(ctypes.get_last_error(), path)
            if rollback_error is not None:
                raise StageCreationError(path, exc, rollback=rollback_error) from exc
            raise

    def _open_file_entry(directory, path, *, delete=False):
        access = _DELETE if delete else _FILE_READ_DATA
        share = _FILE_SHARE_ALL if delete else _FILE_SHARE_READ
        return _open_relative_entry(directory, path, access, _FILE_OPEN, _FILE_NON_DIRECTORY_FILE,
                                    share=share)

    def _rename_entry(fd, parent, target):
        name = os.path.basename(target).encode("utf-16-le")
        offset = _RenameInformation.name.offset
        size = max(ctypes.sizeof(_RenameInformation), offset + len(name))
        buffer = ctypes.create_string_buffer(size)
        info = _RenameInformation.from_buffer(buffer)
        info.root_directory = msvcrt.get_osfhandle(parent)
        info.length = len(name)
        ctypes.memmove(ctypes.addressof(buffer) + offset, name, len(name))
        io_status = _IoStatusBlock()
        status = _set_nt_information(msvcrt.get_osfhandle(fd), ctypes.byref(io_status),
                                     buffer, size, _FILE_RENAME_INFORMATION)
        if status < 0:
            raise _windows_error(_status_to_winerror(status), target)

    def rename_stage_directory(directory, parent, target):
        """Rename this directory handle to a new sibling without replacement."""
        _rename_entry(directory, parent, target)

    def _mark_delete(handle, path):
        delete = ctypes.c_ubyte(1)
        if not _set_file_information(handle, _FILE_DISPOSITION_INFO,
                                     ctypes.byref(delete), ctypes.sizeof(delete)):
            raise _windows_error(ctypes.get_last_error(), path)

    def _delete_entry_handle(directory, path):
        _mark_delete(msvcrt.get_osfhandle(directory), path)

    def _file_change_time(fd, path):
        info = _FileBasicInformation()
        if not _get_file_information(msvcrt.get_osfhandle(fd), _FILE_BASIC_INFO,
                                     ctypes.byref(info), ctypes.sizeof(info)):
            raise _windows_error(ctypes.get_last_error(), path)
        return info.change_time

    @contextmanager
    def verified_wav_source(directory, path, identity):
        """Hold the verified source so its move can be recorded before close."""
        if identity is None:
            raise OSError(f"missing recording WAV identity: {path}")
        fd = _open_file_entry(directory, path, delete=True)
        try:
            _verify_entry(fd, path, is_directory=False, identity=identity)
            yield fd
        finally:
            os.close(fd)

    def publish_verified_wav(source_fd, parent, target):
        """Rename the verified WAV handle relative to its destination directory."""
        _rename_entry(source_fd, parent, target)

else:
    def _open_directory(path, *, delete=False, pin=False):
        return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)

    def _open_relative(directory, path):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        return os.open(os.path.basename(path), flags, mode=0o666, dir_fd=directory)

    def _create_directory_entry(directory, path):
        name = os.path.basename(path)
        os.mkdir(name, dir_fd=directory)
        try:
            return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
        except OSError as exc:
            raise StageCreationError(path, exc) from exc

    def _open_file_entry(directory, path, *, delete=False):
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        return os.open(os.path.basename(path), flags, dir_fd=directory)


def _verify_entry(fd, path, *, is_directory, identity=None):
    info = os.fstat(fd)
    right_type = stat.S_ISDIR(info.st_mode) if is_directory else (
        stat.S_ISREG(info.st_mode) and info.st_nlink == 1)
    actual = (info.st_dev, info.st_ino)
    if (not right_type or not info.st_ino
            or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
            or (identity is not None and actual != identity)):
        raise OSError(f"recording stage entry identity changed: {path}")
    return info


@contextmanager
def verified_stage_directory(path, identity, *, delete=False, pin=False):
    directory = _open_directory(path, delete=delete, pin=pin)
    try:
        _verify_entry(directory, path, is_directory=True, identity=identity)
        yield directory
    finally:
        os.close(directory)


def create_stage_directory(stage, parent_identity):
    identity = None
    try:
        with verified_stage_directory(os.path.dirname(stage), parent_identity) as parent:
            directory = _create_directory_entry(parent, stage)
            try:
                info = _verify_entry(directory, stage, is_directory=True)
                identity = info.st_dev, info.st_ino
            except OSError as exc:
                if os.name != "nt":
                    raise StageCreationError(stage, exc) from exc
                try:
                    _delete_entry_handle(directory, stage)
                except OSError as rollback:
                    raise StageCreationError(stage, exc, rollback=rollback) from exc
                raise
            finally:
                try:
                    os.close(directory)
                except OSError as exc:
                    raise StageCreationError(stage, exc, identity=identity) from exc
    except OSError as exc:
        if identity is not None and not isinstance(exc, StageCreationError):
            raise StageCreationError(stage, exc, identity=identity) from exc
        raise
    return identity


def verified_stage_file_stamp(directory, path, identity):
    fd = _open_file_entry(directory, path)
    try:
        return _verified_file_stamp(fd, path, identity)
    finally:
        os.close(fd)


def _verified_file_stamp(fd, path, identity):
    info = _verify_entry(fd, path, is_directory=False, identity=identity)
    # Windows st_ctime describes creation time. Native ChangeTime records
    # changes even if the file's last-write timestamp has been restored.
    change_time = _file_change_time(fd, path) if os.name == "nt" else info.st_ctime_ns
    return info.st_size, info.st_mtime_ns, change_time


def delete_stage_file(directory, path, identity):
    fd = _open_file_entry(directory, path, delete=True)
    try:
        _verify_entry(fd, path, is_directory=False, identity=identity)
        if os.name == "nt":
            _delete_entry_handle(fd, path)
        else:
            os.unlink(os.path.basename(path), dir_fd=directory)
    finally:
        os.close(fd)


def delete_stage_directory(directory, path, parent_identity):
    if os.name == "nt":
        _delete_entry_handle(directory, path)
    else:
        with verified_stage_directory(os.path.dirname(path), parent_identity) as parent:
            name = os.path.basename(path)
            info = os.stat(name, dir_fd=parent, follow_symlinks=False)
            opened = os.fstat(directory)
            if (info.st_dev, info.st_ino) != (opened.st_dev, opened.st_ino):
                raise OSError(f"recording stage entry identity changed: {path}")
            os.rmdir(name, dir_fd=parent)


@contextmanager
def stage_reservation_opener(stage, identity):
    """Provide a binary exclusive-create opener anchored to the owned stage."""
    with verified_stage_directory(stage, identity) as directory:
        def open_reservation(path, _flags):
            return _open_relative(directory, path)

        yield open_reservation
