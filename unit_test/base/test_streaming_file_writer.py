from unittest import mock

import pytest

from base.streaming_file_writer import StreamingWavWriter


def _writer_with_close_failure():
    writer = StreamingWavWriter.__new__(StreamingWavWriter)
    writer.file_path = "unused.wav"
    writer.sample_rate = 48_000
    writer.channels = 1
    writer.logger = mock.Mock()
    writer.use_soundfile = True
    writer.sf_file = mock.Mock()
    writer.sf_file.close.side_effect = RuntimeError("close failed")
    writer.wave_file = None
    writer.total_frames = 0
    writer.is_open = True
    return writer


def test_failed_finalize_is_terminal_and_destructor_does_not_retry():
    writer = _writer_with_close_failure()

    with pytest.raises(RuntimeError, match="close failed"):
        writer.finalize()

    assert writer._terminal_attempted is True
    assert writer.is_open is False
    writer.finalize()
    writer.__del__()
    writer.sf_file.close.assert_called_once_with()


def test_failed_context_exit_is_terminal_and_does_not_retry():
    writer = _writer_with_close_failure()

    with pytest.raises(RuntimeError, match="close failed"):
        writer.__exit__(None, None, None)

    writer.__exit__(None, None, None)
    writer.sf_file.close.assert_called_once_with()


def test_destructor_swallows_first_close_failure_and_does_not_retry():
    writer = _writer_with_close_failure()

    writer.__del__()
    writer.__del__()

    assert writer._terminal_attempted is True
    assert writer.is_open is False
    writer.sf_file.close.assert_called_once_with()
