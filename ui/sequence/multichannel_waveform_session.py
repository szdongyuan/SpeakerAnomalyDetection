from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ui.sequence.streaming_waveform_accumulator import (
    StreamingWaveformAccumulator,
    StreamingWaveformSnapshot,
)


class MultichannelWaveformSession:
    """Bounded display-only waveform envelopes for one recording run."""

    def __init__(self, *, max_points: int):
        self._max_points = max_points
        self._channels: tuple[int, ...] = ()
        self._accumulators: dict[int, StreamingWaveformAccumulator] = {}

    @property
    def channels(self) -> tuple[int, ...]:
        return self._channels

    def begin(
        self,
        *,
        channels: tuple[int, ...],
        sample_rate: float,
        startup_trim_samples: int,
    ) -> None:
        channel_snapshot = tuple(channels)
        if not channel_snapshot:
            raise ValueError("channels must contain at least one physical channel")
        if any(
            isinstance(channel, (bool, np.bool_))
            or not isinstance(channel, (int, np.integer))
            or channel < 0
            for channel in channel_snapshot
        ):
            raise ValueError("channels must contain non-negative integers")
        if len(set(channel_snapshot)) != len(channel_snapshot):
            raise ValueError("channels must not contain duplicates")

        self.clear()
        new_channels = tuple(int(channel) for channel in channel_snapshot)
        new_accumulators = {}
        for channel in new_channels:
            accumulator = StreamingWaveformAccumulator(
                max_points=self._max_points,
                retain_raw=False,
            )
            accumulator.begin(
                sample_rate=sample_rate,
                startup_trim_samples=startup_trim_samples,
            )
            new_accumulators[channel] = accumulator
        self._channels = new_channels
        self._accumulators = new_accumulators

    def append(self, multi_chunk: np.ndarray) -> None:
        if not self._channels:
            raise RuntimeError("begin must be called before append")

        actual_shape = self._actual_shape(multi_chunk)
        expected_channels = len(self._channels)
        if (
            not isinstance(multi_chunk, np.ndarray)
            or multi_chunk.dtype != np.float32
            or multi_chunk.size == 0
        ):
            raise self._shape_error(expected_channels, actual_shape)

        normalized = multi_chunk
        if normalized.ndim == 1 and expected_channels == 1:
            normalized = normalized.reshape(-1, 1)
        if normalized.ndim != 2 or normalized.shape[1] != expected_channels:
            raise self._shape_error(expected_channels, actual_shape)

        for column, channel in enumerate(self._channels):
            self._accumulators[channel].append(normalized[:, column])

    def snapshots(self) -> Mapping[int, StreamingWaveformSnapshot]:
        return {
            channel: self._accumulators[channel].snapshot()
            for channel in self._channels
        }

    def clear(self) -> None:
        for accumulator in self._accumulators.values():
            accumulator.clear()
        self._accumulators.clear()
        self._channels = ()

    @staticmethod
    def _actual_shape(chunk) -> tuple:
        if isinstance(chunk, np.ndarray):
            return tuple(chunk.shape)
        return tuple(np.shape(chunk))

    @staticmethod
    def _shape_error(expected_channels: int, actual_shape: tuple) -> ValueError:
        return ValueError(
            "multichannel chunk must be a non-empty float32 array with "
            f"expected {expected_channels} channels; actual shape {actual_shape}"
        )
