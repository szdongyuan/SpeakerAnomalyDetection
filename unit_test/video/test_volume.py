import ctypes
import os
from pathlib import PureWindowsPath
from unittest.mock import Mock

import pytest

from base.video.config import VideoConfig
from base.video.recording import validate_volume


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows volume validation")


@pytest.fixture
def volume_api(monkeypatch):
    kernel = Mock()

    def volume_path(root, output, size):
        output.value = PureWindowsPath(root).anchor
        return 1

    def volume_information(root, label, size, serial, length, flags, output, count):
        output.value = "NTFS"
        return 1

    kernel.GetVolumePathNameW.side_effect = volume_path
    kernel.GetVolumeInformationW.side_effect = volume_information
    kernel.GetDriveTypeW.side_effect = lambda root: 3 if root == "C:\\" else 4
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: kernel)
    return kernel


@pytest.mark.parametrize("root", [
    r"\\nas-example\share\录像",
    "//nas-example/share/录像",
    r"Z:\录像",
    r"C:\录像",
])
def test_local_and_network_recording_roots_are_accepted(root, volume_api):
    config = VideoConfig(recording_root=root)
    validate_volume(config.recording_root)
    assert volume_api.GetVolumeInformationW.call_args.args[0] == PureWindowsPath(root).anchor


@pytest.mark.parametrize("filesystem", ["FAT", "FAT32"])
@pytest.mark.parametrize("root", [r"C:\录像", r"\\nas-example\share\录像", r"Z:\录像"])
def test_large_file_protection_remains_for_all_roots(root, filesystem, volume_api):
    def volume_information(root, label, size, serial, length, flags, output, count):
        output.value = filesystem
        return 1

    volume_api.GetVolumeInformationW.side_effect = volume_information
    with pytest.raises(ValueError, match="FAT/FAT32"):
        validate_volume(root)


@pytest.mark.parametrize("operation, error", [
    ("GetVolumePathNameW", 53),
    ("GetVolumeInformationW", 5),
])
def test_network_volume_query_errors_are_preserved(operation, error, volume_api):
    def fail(*args):
        ctypes.set_last_error(error)
        return 0

    getattr(volume_api, operation).side_effect = fail
    with pytest.raises(OSError) as caught:
        validate_volume(r"\\nas-example\share\录像")
    assert caught.value.winerror == error
