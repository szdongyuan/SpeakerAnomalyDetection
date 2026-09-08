"""Chunked export of finalized WAV samples to an atomic raw-audio CSV."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import tempfile

import soundfile as sf


DEFAULT_BLOCK_FRAMES = 8192


def export_raw_audio_csv(
    wav_path,
    csv_path,
    raw_channels,
    *,
    block_frames=DEFAULT_BLOCK_FRAMES,
):
    """Export a finalized WAV without exposing a partially written CSV."""
    source_path = Path(wav_path)
    target_path = Path(csv_path)
    channels = tuple(raw_channels)
    if type(block_frames) is not int or block_frames <= 0:
        raise ValueError("block_frames must be a positive integer")

    temp_path = None
    with sf.SoundFile(str(source_path), mode="r") as source:
        if len(channels) != source.channels:
            raise ValueError(
                "raw channel count does not match WAV channel count: "
                f"{len(channels)} != {source.channels}"
            )
        if any(type(channel) is not int or channel < 0 for channel in channels):
            raise ValueError("raw channels must be non-negative integers")
        if len(set(channels)) != len(channels):
            raise ValueError("raw channels must be unique")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temp_name = tempfile.mkstemp(
            dir=str(target_path.parent),
            prefix=f".{target_path.name}.",
            suffix=".tmp",
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(
                file_descriptor,
                mode="w",
                encoding="utf-8-sig",
                newline="",
            ) as csv_file:
                writer = csv.writer(csv_file, lineterminator="\n")
                writer.writerow(
                    [
                        "time_s",
                        *(f"CH{channel + 1}" for channel in channels),
                    ]
                )

                sample_index = 0
                while True:
                    samples = source.read(
                        frames=block_frames,
                        dtype="float32",
                        always_2d=True,
                    )
                    if not len(samples):
                        break
                    writer.writerows(
                        [
                            f"{(sample_index + offset) / source.samplerate:.9f}",
                            *(f"{float(value):.9g}" for value in row),
                        ]
                        for offset, row in enumerate(samples)
                    )
                    sample_index += len(samples)

                csv_file.flush()
                os.fsync(csv_file.fileno())

            os.replace(temp_path, target_path)
            temp_path = None
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    return target_path
