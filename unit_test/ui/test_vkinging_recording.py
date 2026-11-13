"""Current editor resolution into frozen VE capture requests, with a fake SDK."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from base.recording_calibration_snapshot import build_recording_wav_calibration_metadata
from base.recording_process_protocol import RecordingResult
from base.wav_calibration_metadata import inspect_wav_calibration_metadata, resolve_wav_channel_v2pa_factor
from ui.sequence.workbench_editor_context import resolve_editor_ve_device
from unit_test.base.test_vkinging_capture import start_capture
from unit_test.base.vkinging_fakes import CaptureSDK
from unit_test.ui.vkinging_editor_fixtures import editor_factory


@pytest.mark.parametrize('rate', [44100, 48000, 51200])
@pytest.mark.parametrize('streaming', [False, True])
def test_resolved_queue_parameters_and_calibration_are_frozen_in_capture(editor_factory, tmp_path, rate, streaming):
    host = editor_factory()
    factor = 0.12345678901234567
    host.ve_calibration_store.save_factor(host.mic, 7, v2pa_factor=factor,
                                          calibrated_at='2026-10-07T15:00:00+08:00')
    detail = host.queue_data[0]['seq1']['acq']['detail']
    detail.update(sample_rate=rate, ve_range_index=5)
    device = resolve_editor_ve_device(host.queue_data, host.mic,
                                     host.ve_profile_store, host.ve_calibration_store)
    metadata = build_recording_wav_calibration_metadata(
        host.mic_channels, device, ve_calibration_store=host.ve_calibration_store)
    sdk = CaptureSDK()
    sdk.values_by_physical_channel = {7: .5, 1: .25}
    capture, _, _, _ = start_capture(tmp_path, sdk, request_options={
        'device': device, 'sample_rate': rate, 'streaming': streaming,
        'channels': (7, 1), 'calibration_metadata': metadata,
    })
    detail.update(sample_rate=96000, ve_range_index=1)
    host.ve_calibration_store.save_factor(host.mic, 7, v2pa_factor=99,
                                          calibrated_at='2026-10-07T16:00:00+08:00')
    try:
        outcome = capture.wait(5)
        assert isinstance(outcome, RecordingResult), outcome
        assert capture.join(3)
        assert outcome.handles_released and outcome.metadata_appended
        assert capture.request.sample_rate == rate
        assert capture.request.device['input_config']['range_max'] == .1
        saved, actual_rate = sf.read(outcome.path, dtype='float32', always_2d=True)
        assert actual_rate == rate
        np.testing.assert_array_equal(saved, np.tile([.5, .25], (7, 1)))
        original = Path(outcome.path).read_bytes()
        inspected = inspect_wav_calibration_metadata(outcome.path)
        assert inspected.metadata['schema_version'] == 2
        assert [row['physical_input_channel'] for row in inspected.metadata['recorded_channels']] == [7, 1]
        resolution = resolve_wav_channel_v2pa_factor(inspected, 0)
        assert resolution.factor == factor and resolution.used_file_metadata
        assert Path(outcome.path).read_bytes() == original
    finally:
        capture.cancel()
        assert capture.join(3)
