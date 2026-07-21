"""Generic digital-domain Mel spectrogram analysis."""

from __future__ import annotations

import librosa
import numpy as np

from consts.acoustic_analysis.specific_consts import spec_consts


class InvalidMelBandConfigurationError(ValueError):
    """Raised when the FFT grid cannot support the requested Mel bands."""


class MelSpectrogramAnalyzer:
    """Compute an uncalibrated log-Mel power spectrogram."""

    def analyze(
        self,
        signal: np.ndarray,
        fs: int,
        n_fft: int = spec_consts.DEFAULT_SPEC_N_FFT,
        hop_length: int = spec_consts.DEFAULT_SPEC_HOP_LENGTH,
        n_mels: int = spec_consts.DEFAULT_MEL_BAND_COUNT,
        fmin_hz: float = spec_consts.DEFAULT_MEL_FMIN_HZ,
        fmax_hz: float | None = None,
        window: str = spec_consts.DEFAULT_SPEC_WINDOW,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(signal, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("Mel 频谱输入必须是一维非空信号")
        if not np.all(np.isfinite(values)):
            raise ValueError("Mel 频谱输入包含非有限数值")

        fs = self._positive_int("采样率", fs)
        n_fft = self._positive_int("FFT 点数", n_fft)
        hop_length = self._positive_int("时间步长", hop_length)
        n_mels = self._positive_int("Mel 频带数量", n_mels)

        try:
            fmin_hz = float(fmin_hz)
        except (TypeError, ValueError) as exc:
            raise ValueError("频率下限必须为数字") from exc
        if not np.isfinite(fmin_hz) or fmin_hz < 0:
            raise ValueError("频率下限必须是非负有限数值")

        nyquist_hz = fs / 2.0
        if fmax_hz is None:
            effective_fmax_hz = nyquist_hz
        else:
            try:
                effective_fmax_hz = float(fmax_hz)
            except (TypeError, ValueError) as exc:
                raise ValueError("频率上限必须为数字") from exc
        if not np.isfinite(effective_fmax_hz) or effective_fmax_hz <= fmin_hz:
            raise ValueError("频率上限必须大于频率下限")
        if effective_fmax_hz > nyquist_hz:
            raise ValueError(
                f"频率上限不能超过 Nyquist 频率 {nyquist_hz:g} Hz"
            )

        stft_matrix = librosa.stft(
            y=values,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            pad_mode="constant",
        )
        power_spectrum = np.abs(stft_matrix) ** 2
        mel_basis = librosa.filters.mel(
            sr=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin_hz,
            fmax=effective_fmax_hz,
            htk=False,
            norm="slaney",
            dtype=np.float64,
        )
        if np.any(np.sum(mel_basis, axis=1) <= 0):
            raise InvalidMelBandConfigurationError(
                "部分 Mel 频带没有对应的 FFT 频点，请减少频带数量或增大 FFT 点数"
            )

        mel_power = mel_basis @ power_spectrum
        mel_power_db = librosa.power_to_db(
            mel_power,
            ref=spec_consts.MEL_POWER_DB_REFERENCE,
            amin=spec_consts.MEL_POWER_DB_MINIMUM,
            top_db=None,
        )
        frame_indices = np.arange(mel_power_db.shape[1])
        times_s = librosa.frames_to_time(
            frame_indices,
            sr=fs,
            hop_length=hop_length,
        )
        mel_band_edges_hz = librosa.mel_frequencies(
            n_mels=n_mels + 2,
            fmin=fmin_hz,
            fmax=effective_fmax_hz,
            htk=False,
        )
        mel_frequencies_hz = mel_band_edges_hz[1:-1]

        return (
            np.asarray(times_s, dtype=np.float64),
            np.asarray(mel_frequencies_hz, dtype=np.float64),
            np.asarray(mel_power_db, dtype=np.float64),
        )

    @staticmethod
    def _positive_int(name, value):
        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}必须为正整数") from exc
        if value <= 0:
            raise ValueError(f"{name}必须为正整数")
        return value
