from unittest import mock
import logging

import numpy as np
import soundfile as sf

import pytest

from base.streaming_file_writer import StreamingWavWriter


@pytest.mark.parametrize("deferred", [False, True])
def test_successful_close_logs_once_with_optional_explicit_deferral(tmp_path, deferred):
    records = []

    class Handler(logging.Handler):
        def emit(self, record):
            records.append(record)

    writer = StreamingWavWriter(str(tmp_path / "writer.wav"), channels=1)
    writer.logger = logging.Logger("writer-success", logging.INFO)
    writer.logger.addHandler(Handler())
    if deferred:
        writer.defer_finalization_log()
    writer.write_chunk(np.array([.25, .5], dtype=np.float32))
    writer.finalize()
    assert writer.sf_file.closed and not writer.is_open
    np.testing.assert_array_equal(sf.read(writer.file_path, dtype="float32")[0], [.25, .5])
    assert len(records) == (0 if deferred else 1)
    writer.emit_finalization_log()
    writer.emit_finalization_log()
    writer.finalize()
    record, = records
    assert record.getMessage() == "StreamingWavWriter finalized. Total frames: 2"
    assert record.levelno == logging.INFO and record.filename == "streaming_file_writer.py"


def test_deferred_real_close_failure_never_emits_success():
    writer = _writer_with_close_failure()
    writer.defer_finalization_log()
    with pytest.raises(RuntimeError, match="close failed"):
        writer.finalize()
    writer.emit_finalization_log()
    writer.logger.info.assert_not_called()
    writer.sf_file.close.assert_called_once_with()


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
