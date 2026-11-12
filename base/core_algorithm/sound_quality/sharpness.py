"""
DIN 45692:2009 sharpness from specific loudness.

This module implements DIN 45692 sharpness on top of the project's loudness
chain (see ``loudness.py``). Specific loudness ``N'(z, t)`` -- 中文项目术语
**特征响度** -- is consumed as the only physical input; the time-domain signal
is never re-processed here. The discretization scheme follows the public
references MoSQITo and SQAT (Riemann sum with ``dz = 0.1 Bark`` on the
ISO 532-1 / DIN 45631 standard Bark grid ``np.linspace(0.1, 24.0, 240)``).

The full algorithm specification, including the exact formula references and
verification plan, lives in
``docs/docs_NVH/Sharpness_尖锐度算法实现说明.md``. This module is a faithful
implementation of §5 D1-D6 and §9 of that document.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .loudness import LoudnessMethod
from consts.acoustic_analysis.specific_consts.sound_quality_common_consts import (
    BARK_AXIS_24_BARK_0_1_STEP as _BARK_AXIS_REFERENCE,
)
from consts.acoustic_analysis.specific_consts.sharpness_consts import (
    DEFAULT_K_DIN45692,
    DEFAULT_N_FLOOR_SONE,
    DEFAULT_TIME_SKIP_S,
    SHARPNESS_DIN45692_G_WEIGHT as _G_WEIGHT_REFERENCE,
    SHARPNESS_DZ_BARK as _DZ_BARK,
)

__all__ = [
    "SharpnessResult",
    "SharpnessSummary",
    "compute_sharpness_din45692",
    "DEFAULT_K_DIN45692",
    "DEFAULT_N_FLOOR_SONE",
    "DEFAULT_TIME_SKIP_S",
]


@dataclass(frozen=True)
class SharpnessSummary:
    """Aggregate statistics for a sharpness time series.

    All values are in ``acum``. ``s_stationary`` is filled even for time-varying
    inputs by re-running the DIN 45692 core on the time-averaged specific
    loudness, which makes it directly comparable across modes.
    """

    s_max: float
    s_mean: float
    s_stationary: float


@dataclass(frozen=True)
class SharpnessResult:
    """Output of :func:`compute_sharpness_din45692`.

    ``sharpness_acum`` is always 1D so PER_SEGMENT / TIME_VARYING
    consumers can share the same plotting and statistics code path.
    """

    time_s: np.ndarray
    sharpness_acum: np.ndarray
    summary: SharpnessSummary
    metadata: Dict[str, object]


def _compute_sharpness_values(
    n_prime_2d: np.ndarray,
    z: np.ndarray,
    *,
    k: float = DEFAULT_K_DIN45692,
    n_floor_sone: float = DEFAULT_N_FLOOR_SONE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure DIN 45692 numerator/denominator/ratio kernel (§5 D2-D4).

    Parameters
    ----------
    n_prime_2d
        Specific loudness, shape ``(240, T_eff)``, expected non-negative.
    z
        Bark axis, shape ``(240,)``. Must equal ``np.linspace(0.1, 24.0, 240)``.
    k
        DIN 45692 calibration constant (default 0.11).
    n_floor_sone
        Frames whose denominator falls below this threshold are returned as
        ``NaN`` (see §5 D4).

    Returns
    -------
    sharpness_acum
        Shape ``(T_eff,)`` array in ``acum``; silent frames are ``NaN``.
    n_total
        Shape ``(T_eff,)`` array, the Bark-domain integral of ``N'`` (sone).
    valid_mask
        Boolean mask of frames where ``n_total >= n_floor_sone``.

    The denominator and the validity mask are returned as well so the public
    API can build ``S_stationary`` from valid frames without re-evaluating the
    integral, see §5 D6.
    """

    weight = (_G_WEIGHT_REFERENCE * z)[:, np.newaxis]

    numerator = np.sum(n_prime_2d * weight, axis=0) * _DZ_BARK
    n_total = np.sum(n_prime_2d, axis=0) * _DZ_BARK

    sharpness = np.full(n_total.shape, np.nan, dtype=np.float64)
    valid_mask = n_total >= n_floor_sone
    sharpness[valid_mask] = float(k) * numerator[valid_mask] / n_total[valid_mask]
    return sharpness, n_total, valid_mask


def compute_sharpness_din45692(
    specific_loudness: np.ndarray,
    bark_axis: np.ndarray,
    time_s: Optional[np.ndarray],
    *,
    method: LoudnessMethod = LoudnessMethod.TIME_VARYING_ISO532_1,
    weighting: str = "din45692_default",
    k: float = DEFAULT_K_DIN45692,
    n_floor_sone: float = DEFAULT_N_FLOOR_SONE,
    time_skip: float = DEFAULT_TIME_SKIP_S,
) -> SharpnessResult:
    """Compute DIN 45692 sharpness from specific loudness.

    See ``docs/docs_NVH/Sharpness_尖锐度算法实现说明.md`` §5 / §9.2 for the
    full specification (formulas, validation, mode handling).

    Parameters
    ----------
    specific_loudness
        ``N'(z, t)`` in ``sone/Bark``. Accepted shapes:
        ``(240, T)`` for ``TIME_VARYING_ISO532_1``,
        ``(240, N_seg)`` for ``PER_SEGMENT``.
    bark_axis
        Bark axis, must equal ``np.linspace(0.1, 24.0, 240)``.
    time_s
        Per-frame timestamps in seconds. Required for
        ``TIME_VARYING_ISO532_1`` and ``PER_SEGMENT`` modes.
    method
        Loudness mode the input was produced under. Only
        ``TIME_VARYING_ISO532_1`` triggers ``time_skip`` and the
        ``S_stationary`` time-average path; the other modes treat
        ``sharpness_acum`` as already-static values.
    weighting
        Weighting curve identifier. Currently only ``"din45692_default"``
        (the standard ``g(z)``) is supported; the parameter exists for future
        extensions to the strong / weak weightings.
    k
        DIN 45692 calibration constant. Defaults to 0.11.
    n_floor_sone
        Total-loudness floor. Frames below this become ``NaN`` in
        ``sharpness_acum``.
    time_skip
        Initial transient (in seconds) skipped before computing summary
        statistics. Only applied when ``method`` is
        ``TIME_VARYING_ISO532_1``.
    """

    if weighting != "din45692_default":
        raise NotImplementedError(
            f"weighting={weighting!r} is reserved for future DIN 45692 strong / weak "
            "variants and not implemented yet"
        )
    if not np.isfinite(k):
        raise ValueError("k must be a finite float")
    if not np.isfinite(n_floor_sone) or n_floor_sone < 0.0:
        raise ValueError("n_floor_sone must be a non-negative finite float")
    if not np.isfinite(time_skip) or time_skip < 0.0:
        raise ValueError("time_skip must be a non-negative finite float")

    z = np.asarray(bark_axis, dtype=np.float64)
    if z.shape != _BARK_AXIS_REFERENCE.shape or not np.all(np.isfinite(z)):
        raise ValueError(
            "DIN 45692 sharpness expects a finite Bark axis of shape (240,) equal to "
            "np.linspace(0.1, 24.0, 240)"
        )
    if not np.allclose(z, _BARK_AXIS_REFERENCE, rtol=0.0, atol=1e-12):
        raise ValueError(
            "DIN 45692 sharpness expects bark_axis == np.linspace(0.1, 24.0, 240); "
            "the engine's Bark grid disagrees with the standard 240-point reference."
        )

    n_raw = np.asarray(specific_loudness, dtype=np.float64)
    if n_raw.ndim not in (1, 2):
        raise ValueError("specific_loudness must have shape (240,) or (240, T_eff)")
    if n_raw.shape[0] != _BARK_AXIS_REFERENCE.size:
        raise ValueError("specific_loudness first dimension must be 240 Bark bins")
    if not np.all(np.isfinite(n_raw)):
        raise ValueError("specific_loudness must not contain NaN or inf")

    n_prime = np.maximum(n_raw, 0.0)
    if n_prime.ndim == 1:
        n_prime = n_prime[:, np.newaxis]
    t_eff = n_prime.shape[1]
    if t_eff <= 0:
        raise ValueError("specific_loudness must contain at least one frame")

    if time_s is not None:
        time_s_arr = np.asarray(time_s, dtype=np.float64)
        if time_s_arr.ndim != 1 or time_s_arr.shape[0] != t_eff:
            raise ValueError("time_s must be a 1D array with length matching specific_loudness")
        if not np.all(np.isfinite(time_s_arr)):
            raise ValueError("time_s must not contain NaN or inf")
    else:
        time_s_arr = None
        if t_eff != 1:
            raise ValueError(
                "time_s is required for PER_SEGMENT and TIME_VARYING_ISO532_1 sharpness"
            )

    sharpness, n_total, valid_mask = _compute_sharpness_values(
        n_prime, z, k=k, n_floor_sone=n_floor_sone
    )

    is_time_varying = method == LoudnessMethod.TIME_VARYING_ISO532_1
    if is_time_varying and time_s_arr is not None:
        skip_frames = int(np.searchsorted(time_s_arr, time_skip, side="left"))
    else:
        skip_frames = 0
    skip_frames = min(skip_frames, t_eff)

    stat_window = sharpness[skip_frames:]
    stat_values = stat_window[np.isfinite(stat_window)]

    if stat_values.size == 0:
        s_max = s_mean = float("nan")
    else:
        s_max = float(np.max(stat_values))
        s_mean = float(np.mean(stat_values))

    if is_time_varying:
        valid_after_skip = valid_mask[skip_frames:]
        if valid_after_skip.any():
            n_window = n_prime[:, skip_frames:][:, valid_after_skip]
            n_stationary = np.mean(n_window, axis=1)
            stationary_values, _, _ = _compute_sharpness_values(
                n_stationary[:, np.newaxis], z, k=k, n_floor_sone=n_floor_sone
            )
            s_stationary = float(stationary_values[0]) if np.isfinite(stationary_values[0]) else float("nan")
        else:
            s_stationary = float("nan")
    elif stat_values.size == 0:
        s_stationary = float("nan")
    elif stat_values.size == 1:
        s_stationary = float(stat_values[0])
    else:
        s_stationary = float(np.mean(stat_values))

    if time_s_arr is not None:
        time_axis = time_s_arr.copy()
    else:
        time_axis = np.array([0.0], dtype=np.float64)

    sharpness_out = np.ascontiguousarray(sharpness, dtype=np.float64)
    sharpness_out.setflags(write=False)
    time_axis.setflags(write=False)

    summary = SharpnessSummary(
        s_max=s_max,
        s_mean=s_mean,
        s_stationary=s_stationary,
    )

    method_value = method.value if isinstance(method, LoudnessMethod) else str(method)
    metadata: Dict[str, object] = {
        "standard": "DIN 45692:2009",
        "method": method_value,
        "weighting": weighting,
        "k": float(k),
        "n_floor_sone": float(n_floor_sone),
        "time_skip_s": float(time_skip) if is_time_varying else 0.0,
        "skip_frames_for_summary": int(skip_frames) if is_time_varying else 0,
        "dz_bark": _DZ_BARK,
        "n_bark_bins": int(_BARK_AXIS_REFERENCE.size),
        "implementation": "project_din45692_riemann_sum_from_specific_loudness",
        "reference": (
            "DIN 45692:2009; cross-checked against MoSQITo "
            "sharpness_din_from_loudness and SQAT Sharpness_DIN45692_from_loudness"
        ),
        "input_shape": tuple(np.asarray(specific_loudness).shape),
    }

    return SharpnessResult(
        time_s=time_axis,
        sharpness_acum=sharpness_out,
        summary=summary,
        metadata=metadata,
    )
