from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral
from typing import Dict, Tuple

import numpy as np
from scipy import signal as scipy_signal


def validate_selected_harmonic_orders(harmonic_orders: Iterable, max_harmonic_order: int = 35) -> Tuple[int, ...]:
    max_order = int(max_harmonic_order)
    if harmonic_orders is None:
        raise ValueError("harmonic_orders must be an iterable of integer orders")

    selected = []
    seen = set()
    for order in harmonic_orders:
        if isinstance(order, (bool, np.bool_)) or not isinstance(order, Integral):
            raise ValueError(f"harmonic order must be an integer from 2 to {max_order}, got {order}")
        order_int = int(order)
        if order_int < 2 or order_int > max_order:
            raise ValueError(f"harmonic order must be in the range 2 through {max_order}, got {order}")
        if order_int not in seen:
            selected.append(order_int)
            seen.add(order_int)
    return tuple(selected)


class SynchronousHarmonicDetector:
    def __init__(self, max_harmonic_order: int = 35, fundamental_floor: float = 1e-12):
        self.max_harmonic_order = int(max_harmonic_order)
        self.fundamental_floor = float(fundamental_floor)

    def analyze(
        self,
        segment,
        *,
        f0: float,
        sample_rate: int,
        harmonic_orders: Iterable,
        stft_window_type="hann",
    ) -> Tuple[Dict[int, float], float]:
        selected_orders = self._validate_selected_orders(harmonic_orders)
        amplitudes = self.estimate_amplitudes(
            segment,
            f0=f0,
            sample_rate=sample_rate,
            stft_window_type=stft_window_type,
        )
        return amplitudes, self.compute_distortion_percent(amplitudes, selected_orders)

    def estimate_amplitudes(
        self,
        segment,
        *,
        f0: float,
        sample_rate: int,
        stft_window_type="hann",
    ) -> Dict[int, float]:
        x = np.asarray(segment, dtype=np.float64)
        if x.ndim != 1:
            x = np.ravel(x)

        fs = float(sample_rate)
        f0 = float(f0)
        if not np.isfinite(f0) or f0 <= 0.0 or f0 >= fs / 2.0:
            raise ValueError(f"f0 must satisfy 0 < f0 < Nyquist, got {f0}")

        effective_orders = self._effective_orders(f0, fs)
        window = self._window(stft_window_type, int(x.size))
        design = self._design_matrix(int(x.size), f0, fs, effective_orders)
        weighted_design = design * window[:, None]
        weighted_signal = x * window
        rcond = np.finfo(np.float64).eps * max(weighted_design.shape)
        coeffs, *_ = np.linalg.lstsq(weighted_design, weighted_signal, rcond=rcond)
        return self._amplitudes_from_coeffs(coeffs, effective_orders)

    def compute_distortion_percent(self, amplitudes: Dict[int, float], selected_orders: Iterable) -> float:
        selected_orders = self._validate_selected_orders(selected_orders)
        harmonic_power = 0.0
        for order in selected_orders:
            if order in amplitudes:
                amplitude = float(amplitudes[order])
                harmonic_power += amplitude * amplitude
        fundamental = max(float(amplitudes.get(1, 0.0)), self.fundamental_floor)
        return float(100.0 * np.sqrt(harmonic_power) / fundamental)

    def _effective_orders(self, f0: float, fs: float) -> Tuple[int, ...]:
        return tuple(
            order
            for order in range(1, self.max_harmonic_order + 1)
            if order * f0 < fs / 2.0
        )

    def _window(self, stft_window_type, sample_count: int) -> np.ndarray:
        if isinstance(stft_window_type, (str, tuple)):
            window = scipy_signal.get_window(stft_window_type, sample_count, fftbins=True)
        else:
            window = np.asarray(stft_window_type, dtype=np.float64)
            if window.ndim != 1:
                raise ValueError("Explicit window must be one-dimensional")
            if int(window.size) != sample_count:
                raise ValueError(
                    f"Explicit window length must match segment length {sample_count}, got {window.size}"
                )

        window = np.asarray(window, dtype=np.float64)
        if window.ndim != 1:
            raise ValueError("Window must be one-dimensional")
        if int(window.size) != sample_count:
            raise ValueError(f"Window length must match segment length {sample_count}, got {window.size}")
        if not np.all(np.isfinite(window)):
            raise ValueError("Window values must be finite")
        if not np.any(window != 0.0):
            raise ValueError("Window must contain at least one non-zero value")
        return window

    @staticmethod
    def _design_matrix(
        sample_count: int,
        f0: float,
        fs: float,
        effective_orders: Tuple[int, ...],
    ) -> np.ndarray:
        n = np.arange(sample_count, dtype=np.float64)
        columns = [np.ones(sample_count, dtype=np.float64)]
        for order in effective_orders:
            phase = 2.0 * np.pi * order * f0 * n / fs
            columns.append(np.cos(phase))
            columns.append(np.sin(phase))
        return np.column_stack(columns)

    @staticmethod
    def _amplitudes_from_coeffs(coeffs: np.ndarray, effective_orders: Tuple[int, ...]) -> Dict[int, float]:
        amplitudes = {}
        coeff_idx = 1
        for order in effective_orders:
            cos_coeff = float(coeffs[coeff_idx])
            sin_coeff = float(coeffs[coeff_idx + 1])
            amplitudes[int(order)] = float(np.sqrt(cos_coeff * cos_coeff + sin_coeff * sin_coeff))
            coeff_idx += 2
        return amplitudes

    def _validate_selected_orders(self, harmonic_orders: Iterable) -> Tuple[int, ...]:
        return validate_selected_harmonic_orders(harmonic_orders, self.max_harmonic_order)
