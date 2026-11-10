"""Per-station writers for a single already-trimmed capture stream."""

from dataclasses import dataclass
import os
import stat

import numpy as np

from base.recording_process_protocol import FrozenConfig
from base.recording_quality_accumulator import RecordingQualityAccumulator
from base.streaming_file_writer import StreamingWavWriter
from base.workbench_recording_protocol import StationWav


@dataclass
class _StationFile:
    descriptor: StationWav
    columns: tuple[int, ...]
    quality: RecordingQualityAccumulator
    metadata: FrozenConfig | None
    writer: StreamingWavWriter | None = None

    @property
    def label(self):
        return f"station {self.descriptor.station_id!r}: "


class StationWavOutput:
    def __init__(self, request, range_max):
        self.directory = request.path
        self.sample_rate = request.sample_rate
        column_by_physical = {physical: column for column, physical in enumerate(request.channels)}
        self.files = tuple(
            _StationFile(
                StationWav(plan.station_id, os.path.join(self.directory, plan.filename), plan.channels),
                tuple(column_by_physical[physical] for physical in plan.physical_channels),
                RecordingQualityAccumulator(len(plan.channels), range_max),
                plan.calibration_metadata,
            )
            for plan in request.station_group.stations
        )

    def open(self, writer_factory, cancelled, *, reserved=False):
        if not stat.S_ISDIR(os.stat(self.directory).st_mode):
            raise NotADirectoryError(f"station staging path is not a directory: {self.directory}")
        for station in self.files:
            if cancelled():
                return
            path = station.descriptor.path
            try:
                if not reserved:
                    self._require_absent(path)
                if not cancelled():
                    station.writer = writer_factory(path, sample_rate=self.sample_rate,
                                                    channels=len(station.columns))
            except Exception as exc:
                # The injected file backend may raise backend-specific errors.
                # Earlier writers remain registered for the capture owner's close pass.
                raise OSError(f"{station.label}{exc}") from exc

    @staticmethod
    def _require_absent(path):
        # The capture owner has the staging directory exclusively. Ask the
        # filesystem about this target, including links and Windows short names.
        try:
            os.lstat(path)
        except FileNotFoundError:
            return
        raise FileExistsError(f"WAV target already exists: {path}")

    def write_chunk(self, block):
        for station in self.files:
            selected = block[:, station.columns]
            try:
                saved = station.writer.write_chunk(selected)
            except Exception as exc:
                # Some earlier stations may already have written this block.
                # The owner reports failure and keeps the last common frame count.
                raise OSError(f"{station.label}{exc}") from exc
            if (not isinstance(saved, np.ndarray) or saved.dtype != np.float32
                    or saved.shape != selected.shape):
                raise ValueError(f"{station.label}writer must return saved float32 samples with the retained shape")
            station.quality.append(saved)

    def finalize(self, close_writer):
        released = True
        for station in self.files:
            if station.writer is not None:
                if close_writer(station.writer, station.descriptor.path, station.label):
                    station.writer = None
                else:
                    released = False
        return released
