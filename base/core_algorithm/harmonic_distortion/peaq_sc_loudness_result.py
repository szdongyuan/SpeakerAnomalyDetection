from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from base.core_algorithm.harmonic_distortion.peaq_sc_utils import hz_to_bark_peaq, sones_to_phons


@dataclass(frozen=True)
class PEAQLoudnessResult:
    """
    Result container used for both "loudness" (Eq.9-12) and "partial loudness" (Eq.13-15).

    The paper's naming:
      - N_total / TotalNL correspond to `n_total_sones`
      - N[k] / NL[k] correspond to `n_specific_sones_per_band` (per band index k)
    """

    n_total_sones: np.ndarray  # (T,)
    n_total_phons: np.ndarray  # (T,)
    n_specific_sones_per_band: np.ndarray  # (Z, T)
    band_center_hz: np.ndarray  # (Z,)
    band_center_bark: np.ndarray  # (Z,)

    def interpolate_specific_sones(
        self,
        target_freqs_hz: np.ndarray,
        *,
        out_of_range: Literal["zero", "edge"] = "zero",
    ) -> np.ndarray:
        """
        Interpolate the per-band specific loudness values to target frequencies.

        - Uses linear interpolation on the Bark axis (bands are equally spaced in Bark).
        - `n_specific_sones_per_band` is a *density-like* quantity (sones/Bark); the paper integrates it via
          (24/Z) * sum_k to obtain total loudness.

        target_freqs_hz:
          - shape (M,): same target frequencies for all frames
          - shape (M, T): per-frame target frequencies (e.g., harmonics of a time-varying f0)

        Returns: shape (M, T)
        """
        y = np.asarray(self.n_specific_sones_per_band, dtype=np.float64)
        x = np.asarray(self.band_center_bark, dtype=np.float64).reshape(-1)
        if y.ndim != 2 or x.ndim != 1 or y.shape[0] != x.size:
            raise ValueError(
                "Invalid loudness result shapes: "
                f"n_specific_sones_per_band={y.shape}, band_center_bark={x.shape}"
            )

        target_hz = np.asarray(target_freqs_hz, dtype=np.float64)
        if not np.all(np.isfinite(target_hz)):
            raise ValueError("target_freqs_hz must be finite")
        if np.any(target_hz < 0.0):
            raise ValueError("target_freqs_hz must be >= 0")

        t_frames = int(y.shape[1])
        if target_hz.ndim == 0:
            target_hz = target_hz.reshape(1)
        if target_hz.ndim == 1:
            # Broadcast to all frames.
            target_bark = hz_to_bark_peaq(target_hz).reshape(-1)
            m = int(target_bark.size)
            idx = np.searchsorted(x, target_bark, side="left")
            # Clamp to interior so i0/i1 are valid; we'll mask oob later.
            i1 = np.clip(idx, 1, x.size - 1)
            i0 = i1 - 1
            x0 = x[i0]
            x1 = x[i1]
            w = (target_bark - x0) / np.maximum(x1 - x0, 1e-300)
            y0 = y[i0, :]  # (M, T)
            y1 = y[i1, :]
            out = y0 * (1.0 - w.reshape(m, 1)) + y1 * w.reshape(m, 1)

            oob = (target_bark < x[0]) | (target_bark > x[-1])
            if np.any(oob):
                if out_of_range == "zero":
                    out[oob, :] = 0.0
                elif out_of_range == "edge":
                    out[target_bark < x[0], :] = y[0:1, :]
                    out[target_bark > x[-1], :] = y[-1:, :]
                else:
                    raise ValueError(f"Unsupported out_of_range={out_of_range!r}")
            return out

        if target_hz.ndim == 2:
            if target_hz.shape[1] != t_frames:
                raise ValueError(
                    f"target_freqs_hz second dim must be T={t_frames}, got {target_hz.shape}"
                )
            m = int(target_hz.shape[0])
            target_bark = hz_to_bark_peaq(target_hz.reshape(-1)).reshape(-1)
            idx = np.searchsorted(x, target_bark, side="left")
            i1 = np.clip(idx, 1, x.size - 1)
            i0 = i1 - 1
            x0 = x[i0]
            x1 = x[i1]
            w = (target_bark - x0) / np.maximum(x1 - x0, 1e-300)

            # Select y[i0, t] and y[i1, t] for each flattened element.
            t_idx = np.tile(np.arange(t_frames, dtype=np.int64), m)
            y0 = y[i0, t_idx]
            y1 = y[i1, t_idx]
            out_flat = y0 * (1.0 - w) + y1 * w

            oob = (target_bark < x[0]) | (target_bark > x[-1])
            if np.any(oob):
                if out_of_range == "zero":
                    out_flat[oob] = 0.0
                elif out_of_range == "edge":
                    out_flat[target_bark < x[0]] = y[0, t_idx[target_bark < x[0]]]
                    out_flat[target_bark > x[-1]] = y[-1, t_idx[target_bark > x[-1]]]
                else:
                    raise ValueError(f"Unsupported out_of_range={out_of_range!r}")

            return out_flat.reshape(m, t_frames)

        raise ValueError(f"target_freqs_hz must be scalar, 1D, or 2D, got shape {target_hz.shape}")

    def interpolate_specific_phons_equiv(
        self,
        target_freqs_hz: np.ndarray,
        *,
        out_of_range: Literal["zero", "edge"] = "zero",
    ) -> np.ndarray:
        """
        Convenience wrapper: interpolate in sones-domain, then convert each interpolated value to phons.

        This is a "phon-equivalent" for a *specific* loudness sample and is not ISO 532 loudness level.
        """
        return sones_to_phons(self.interpolate_specific_sones(target_freqs_hz, out_of_range=out_of_range))

