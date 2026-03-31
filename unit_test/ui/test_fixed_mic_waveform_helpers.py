import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ui.sequence.fixed_mic.waveform_helpers import (
    build_fixed_mic_plot_data,
    get_fixed_mic_channel_titles,
    get_fixed_mic_downsample_step,
    get_fixed_mic_page_channel_indices,
)


class TestFixedMicWaveformHelpers(object):

    def test_get_fixed_mic_page_channel_indices_returns_current_page_slice(self):
        assert get_fixed_mic_page_channel_indices(1, 0) == [0]
        assert get_fixed_mic_page_channel_indices(6, 0) == [0, 1, 2, 3]
        assert get_fixed_mic_page_channel_indices(6, 1) == [4, 5]
        assert get_fixed_mic_page_channel_indices(6, 99) == [4, 5]

    def test_get_fixed_mic_downsample_step_depends_on_visible_channels(self):
        assert get_fixed_mic_downsample_step(0) == 1
        assert get_fixed_mic_downsample_step(1) == 4
        assert get_fixed_mic_downsample_step(2) == 4
        assert get_fixed_mic_downsample_step(3) == 8
        assert get_fixed_mic_downsample_step(4) == 8

    def test_build_fixed_mic_plot_data_downsamples_all_channels_with_shared_time_axis(self):
        audio = np.arange(24, dtype=np.float32).reshape(12, 2)

        time_axis, display_audio = build_fixed_mic_plot_data(
            audio,
            sample_rate=12,
            visible_channel_count=2,
        )

        assert np.allclose(time_axis, [0.0, 4.0 / 12.0, 8.0 / 12.0, 11.0 / 12.0])
        assert display_audio.shape == (4, 2)
        assert list(display_audio[:, 0]) == [0.0, 8.0, 16.0, 22.0]
        assert list(display_audio[:, 1]) == [1.0, 9.0, 17.0, 23.0]

    def test_build_fixed_mic_plot_data_uses_8x_downsampling_for_three_or_more_visible_channels(self):
        audio = np.arange(48, dtype=np.float32).reshape(16, 3)

        time_axis, display_audio = build_fixed_mic_plot_data(
            audio,
            sample_rate=16,
            visible_channel_count=3,
        )

        assert np.allclose(time_axis, [0.0, 8.0 / 16.0, 15.0 / 16.0])
        assert display_audio.shape == (3, 3)
        assert list(display_audio[:, 0]) == [0.0, 24.0, 45.0]

    def test_get_fixed_mic_channel_titles_prefers_labels_and_falls_back(self):
        channel_config = [
            {"label": "Left"},
            {"label": ""},
        ]

        titles = get_fixed_mic_channel_titles(4, channel_config)

        assert titles == ["Left", "Mic2", "Mic3", "Mic4"]
