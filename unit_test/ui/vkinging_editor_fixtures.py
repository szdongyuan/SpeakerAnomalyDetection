"""Shared editor fixtures with real VE stores and no retired recording host."""
import logging
from types import SimpleNamespace

import pytest

from base.vkinging_stores import VECalibrationStore, VEInputProfileStore
from unit_test.base.vkinging_fakes import device_info, input_config


@pytest.fixture
def editor_factory(tmp_path):
    def make(rate=51200):
        device = device_info(input_config=input_config(rate))
        profiles = VEInputProfileStore(tmp_path / 'profiles.json')
        calibrations = VECalibrationStore(tmp_path / 'calibrations.json')
        profiles.set_sample_rate(device, rate, calibrations)
        return SimpleNamespace(
            mic=device, mic_channels=[7, 1], ve_profile_store=profiles,
            ve_calibration_store=calibrations, default_logger=logging.getLogger(__name__),
            queue_data=[{'seq1': {'acq': {'detail': {
                'total_time': .04, 'startup_trim_ms': 10,
                'use_streaming_recording': False, 'audio_validation': {'enabled': False},
            }}}}],
        )
    return make
