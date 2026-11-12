"""Validate display matrices and derive bin edges without recomputing a transform."""

import numpy as np


def validate_spectrogram(payload):
    if not isinstance(payload, dict) or payload.get('kind') != 'spectrogram':
        raise ValueError('没有时频矩阵')
    if payload.get('matrix_axes') != ['frequency', 'time']:
        raise ValueError('时频矩阵轴顺序必须为 frequency、time')
    time = np.array(payload.get('time_s'), dtype=float, copy=True)
    frequency = np.array(payload.get('frequency_hz'), dtype=float, copy=True)
    values = np.array(payload.get('values_db'), dtype=float, copy=True)
    for axis, name in ((time, '时间'), (frequency, '频率')):
        if (axis.ndim != 1 or axis.size < 2 or not np.isfinite(axis).all()
                or axis[0] < 0 or not np.all(np.diff(axis) > 0)):
            raise ValueError(f'{name}轴必须包含至少两个非负、严格递增的有限数值')
    if not np.allclose(np.diff(time), time[1] - time[0], rtol=1e-5, atol=1e-9):
        raise ValueError('当前时频展示要求 STFT/CQT 等间距时间帧')
    if values.shape != (frequency.size, time.size):
        raise ValueError('时频矩阵形状必须为频率数 × 时间点数')
    if np.isinf(values).any() or not np.isfinite(values).any():
        raise ValueError('时频矩阵没有有效数值或含无穷值')
    return dict(time_s=time, frequency_hz=frequency, values_db=values)


def bin_edges(centres):
    """Samples are bin centres, not image endpoints; never shift peaks by half a bin."""
    middle = (centres[:-1] + centres[1:]) / 2
    return np.r_[centres[0] - (middle[0] - centres[0]), middle,
                 centres[-1] + (centres[-1] - middle[-1])]
