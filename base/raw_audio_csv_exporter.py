"""Chunked export of finalized WAV samples to an atomic raw-audio CSV."""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
import sys
import tempfile

import soundfile as sf


DEFAULT_BLOCK_FRAMES = 8192


def export_raw_audio_csv(
    wav_path,
    csv_path,
    raw_channels,
    *,
    block_frames=DEFAULT_BLOCK_FRAMES,
    temporary_path=None,
    cleanup_failed=None,
    temporary_created=None,
):
    """Export a finalized WAV without exposing a partially written CSV."""
    source_path = Path(wav_path)
    target_path = Path(csv_path)
    channels = tuple(raw_channels)
    if type(block_frames) is not int or block_frames <= 0:
        raise ValueError("block_frames must be a positive integer")
    controlled_path = Path(temporary_path) if temporary_path is not None else None
    if controlled_path is not None:
        normalized = controlled_path.resolve()
        if (
            normalized.parent != target_path.parent.resolve()
            or normalized in (target_path.resolve(), source_path.resolve())
            or not controlled_path.name
            or controlled_path.is_dir()
        ):
            raise ValueError("temporary path must be a distinct file in the target directory")

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
        if controlled_path is None:
            file_descriptor, temp_name = tempfile.mkstemp(
                dir=str(target_path.parent),
                prefix=f".{target_path.name}.",
                suffix=".tmp",
            )
            temp_path = Path(temp_name)
        else:
            file_descriptor = os.open(
                controlled_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            # Ownership begins only after exclusive creation succeeds.
            temp_path = controlled_path
        try:
            if temporary_created is not None:
                info = os.fstat(file_descriptor)
                temporary_created(temp_path, (info.st_dev, info.st_ino))
            csv_file = os.fdopen(
                file_descriptor,
                mode="w",
                encoding="utf-8-sig",
                newline="",
            )
            file_descriptor = None  # csv_file now owns the descriptor.
            with csv_file:
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
            export_error = sys.exception()
            if file_descriptor is not None:
                os.close(file_descriptor)
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except OSError as cleanup_error:
                    if cleanup_failed is None:
                        logging.getLogger(__name__).warning(
                            "CSV temporary cleanup failed: %s", temp_path, exc_info=True
                        )
                    else:
                        try:
                            cleanup_failed(temp_path, cleanup_error)
                        finally:
                            # Diagnostics must not replace the export error. A
                            # callback failure remains in the exception chain.
                            if export_error is not None:
                                raise export_error

    return target_path
