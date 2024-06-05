import librosa.core.spectrum as spectrum
import librosa.feature.spectral as spectral
import numpy as np
import librosa
from scipy.fftpack import fft
import matplotlib.pyplot as plt


class AudioFeatureExtraction(object):

    @staticmethod
    def spectrogram(signal, sr, **kwargs):
        """
        - preprocess_method: "spectrogram"
          preprocess_param:
            extraction_kwargs:
              n_fft: 2048
              hop_length: 512
        :param signal:
        :param sr:
        :param kwargs:
        :return:
        """
        extraction_kwargs = kwargs.get("extraction_kwargs", {})
        spec = np.abs(spectrum.stft(y=signal, **extraction_kwargs))
        if kwargs.get("time_series_first", True):
            spec = spec.T
        if kwargs.get("flatten", False):
            spec = spec.reshape((1, spec.shape[0] * spec.shape[1]))[0]
        return spec

    @staticmethod
    def mfcc(signal, sr, **kwargs):
        """
        - preprocess_method: "mfcc"
          preprocess_param:
            extraction_kwargs:
              n_mfcc: 20
              n_fft: 2048
              hop_length: 512
        :param signals:
        :param sr:
        :param kwargs:
        :return:
        """
        extraction_kwargs = kwargs.get("extraction_kwargs", {})
        mfcc = spectral.mfcc(y=signal, sr=sr, **extraction_kwargs)
        if kwargs.get("time_series_first", True):
            mfcc = mfcc.T
        if kwargs.get("flatten", False):
            mfcc = mfcc.reshape((1, mfcc.shape[0] * mfcc.shape[1]))[0]
        return mfcc

    @staticmethod
    def mel_spec(signal, sr, **kwargs):
        """
        - preprocess_method: "mel_spec"
          preprocess_param:
            extraction_kwargs:
              n_fft: 2048
              hop_length: 256
        :param signals:
        :param sr:
        :param kwargs:
        :return:
        """
        extraction_kwargs = kwargs.get("extraction_kwargs", {})
        mel_spec = spectral.melspectrogram(y=signal, sr=sr, **extraction_kwargs)
        if kwargs.get("time_series_first", True):
            mel_spec = mel_spec.T
        if kwargs.get("flatten", False):
            mel_spec = mel_spec.reshape((1, mel_spec.shape[0] * mel_spec.shape[1]))[0]
        return mel_spec

    @staticmethod
    def zero_crossing_rate(signal, sr, **kwargs):
        """
        - preprocess_method: "zero_crossing_rate"
          preprocess_param:
            extraction_kwargs:
              frame_length: 2048
              hop_length: 256
        :param signals:
        :param sr:
        :param kwargs:
        :return:
        """
        extraction_kwargs = kwargs.get("extraction_kwargs", {})
        zcr = spectral.zero_crossing_rate(y=signal, **extraction_kwargs)
        if kwargs.get("time_series_first", True):
            zcr = zcr.T
        if kwargs.get("flatten", False):
            zcr = zcr.reshape((1, zcr.shape[0] * zcr.shape[1]))[0]
        return zcr

    @staticmethod
    def spectral_flatness(signal, sr, **kwargs):
        """
        - preprocess_method: "spectral_flatness"
          preprocess_param:
            extraction_kwargs:
              n_fft: 2048
              hop_length: 256
        :param signals:
        :param sr:
        :param kwargs:
        :return:
        """
        extraction_kwargs = kwargs.get("extraction_kwargs", {})
        spectral_flatness = spectral.spectral_flatness(y=signal, **extraction_kwargs)
        if kwargs.get("time_series_first", True):
            spectral_flatness = spectral_flatness.T
        if kwargs.get("flatten", False):
            spectral_flatness = spectral_flatness.reshape((1, spectral_flatness.shape[0] * spectral_flatness.shape[1]))[0]
        return spectral_flatness

    @staticmethod
    def calculate_thd(signal, sr, **kwargs):
        window_size = kwargs.get("window_size", 2048)
        step_size = kwargs.get("step_size", 256)
        num_windows = (len(signal) - window_size) // step_size + 1
        windows = np.array([signal[i * step_size:i * step_size + window_size] for i in range(num_windows)])

        thd_list = []

        for window in windows:
            N = len(window)

            yf = fft(window)

            yf = np.abs(yf[:N // 2])
            xf = np.fft.fftfreq(N, 1 / sr)[:N // 2]

            fundamental_idx = np.argmax(yf)
            fundamental_freq = xf[fundamental_idx]

            harmonics = [fundamental_freq * (i + 1) for i in range(1, 5)]

            fundamental_amplitude = yf[fundamental_idx]
            harmonic_amplitudes = []
            for harmonic in harmonics:
                harmonic_idx = np.argmin(np.abs(xf - harmonic))
                harmonic_amplitudes.append(yf[harmonic_idx])

            thd = (np.sqrt(sum([amp ** 2 for amp in harmonic_amplitudes])) / fundamental_amplitude) * 100
            thd_list.append(thd)

        return np.array(thd_list)

    @staticmethod
    def calculate_hohd(signal, sr, **kwargs):
        window_size = kwargs.get("window_size", 2048)
        step_size = kwargs.get("step_size", 256)
        num_windows = (len(signal) - window_size) // step_size + 1
        windows = np.array([signal[i * step_size:i * step_size + window_size] for i in range(num_windows)])

        hohd_list = []

        for window in windows:
            N = len(window)
            yf = np.fft.fft(window)
            xf = np.fft.fftfreq(N, 1 / sr)

            yf = np.abs(yf[:N // 2])
            xf = xf[:N // 2]

            fundamental_freq = xf[np.argmax(yf)]
            fundamental_amplitude = yf[np.argmax(yf)]

            harmonics = [fundamental_freq * (i + 1) for i in range(9, 35)]  # 选择第10到第35次谐波
            harmonic_amplitudes = []

            for harmonic in harmonics:
                harmonic_idx = np.argmin(np.abs(xf - harmonic))
                harmonic_amplitudes.append(yf[harmonic_idx])

            hohd = (np.sqrt(sum([amp ** 2 for amp in harmonic_amplitudes])) / fundamental_amplitude) * 100
            hohd_list.append(hohd)

        return np.array(hohd_list)

    @staticmethod
    def calculate_nthd(signal, sr, **kwargs):
        window_size = kwargs.get("window_size", 2048)
        step_size = kwargs.get("step_size", 256)
        num_windows = (len(signal) - window_size) // step_size + 1
        windows = np.array([signal[i * step_size:i * step_size + window_size] for i in range(num_windows)])

        nthd_list = []

        for window in windows:
            N = len(window)
            yf = np.fft.fft(window)
            xf = np.fft.fftfreq(N, 1 / sr)

            yf = np.abs(yf[:N // 2])
            xf = xf[:N // 2]

            fundamental_index = np.argmax(yf)
            fundamental_freq = xf[fundamental_index]

            harmonics = [fundamental_freq * (i + 1) for i in range(1, 35)]

            harmonic_indices = []
            for harmonic in harmonics:
                harmonic_idx = np.argmin(np.abs(xf - harmonic))
                harmonic_indices.append(harmonic_idx)

            clean_spectrum = np.copy(yf)
            clean_spectrum[fundamental_index] = 0
            for idx in harmonic_indices:
                clean_spectrum[idx] = 0

            nthd = (np.sqrt(np.sum(clean_spectrum ** 2)) / np.sqrt(np.sum(yf ** 2))) * 100
            nthd_list.append(nthd)

        return np.array(nthd_list)

    @staticmethod
    def calculate_spl(signal):
        rms = np.sqrt(np.mean(signal ** 2))
        p0 = 20e-6
        spl = 20 * np.log10(rms / p0)
        return spl

    @staticmethod
    def calculate_frequency_response(signal, sr, output_db="db", **kwargs):
        window_size = kwargs.get("window_size", 2048)
        step_size = kwargs.get("step_size", 256)
        num_windows = (len(signal) - window_size) // step_size + 1
        windows = np.array([signal[i * step_size:i * step_size + window_size] for i in range(num_windows)])

        xf_list = []

        for window in windows:

            N = len(window)

            if output_db == "db":

                yf = np.fft.fft(window)
                yf = np.abs(yf[:N // 2])
                yf_db = 20 * np.log10(yf)
                xf_list.append(yf_db)
            else:
                xf = np.fft.fftfreq(N, 1 / sr)
                xf = xf[:N // 2]
                xf_list.append(xf)

        return np.array(xf_list)
