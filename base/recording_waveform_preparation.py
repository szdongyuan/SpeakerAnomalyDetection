"""Qt-free, exact peak-preserving waveform display preparation."""
import numpy as np


def prepare_waveform_display_data(
    waveform,
    sample_rate,
    *,
    max_points=None,
):
    """Build a peak-preserving display copy without changing the full waveform."""
    if max_points is None:
        point_limit = 4000
    elif isinstance(max_points, (bool, np.bool_)) or not isinstance(
        max_points,
        (int, np.integer),
    ) or max_points < 4:
        raise ValueError("max_points must be an integer >= 4")
    else:
        point_limit = int(max_points)

    sample_count = waveform.shape[0]
    if sample_count <= point_limit:
        sample_indices = np.arange(sample_count, dtype=np.int64)
    else:
        peak_bucket_count = (point_limit - 2) // 2
        bucket_size = (sample_count + peak_bucket_count - 1) // peak_bucket_count
        full_block_count = sample_count // bucket_size
        full_sample_count = full_block_count * bucket_size
        blocks = waveform[:full_sample_count].reshape(full_block_count, bucket_size)
        block_starts = np.arange(full_block_count, dtype=np.int64) * bucket_size
        min_indices = block_starts + np.argmin(blocks, axis=1)
        max_indices = block_starts + np.argmax(blocks, axis=1)
        ordered_indices = np.empty(full_block_count * 2, dtype=np.int64)
        ordered_indices[0::2] = np.minimum(min_indices, max_indices)
        ordered_indices[1::2] = np.maximum(min_indices, max_indices)
        peak_indices = [ordered_indices]

        if full_sample_count < sample_count:
            tail = waveform[full_sample_count:]
            tail_min = full_sample_count + int(np.argmin(tail))
            tail_max = full_sample_count + int(np.argmax(tail))
            peak_indices.append(
                np.array([min(tail_min, tail_max), max(tail_min, tail_max)], dtype=np.int64)
            )

        sample_indices = np.unique(
            np.concatenate(
                [
                    np.array([0], dtype=np.int64),
                    *peak_indices,
                    np.array([sample_count - 1], dtype=np.int64),
                ]
            )
        )

    time_axis = sample_indices.astype(np.float64) / float(sample_rate or 1.0)
    return time_axis, waveform[sample_indices]
