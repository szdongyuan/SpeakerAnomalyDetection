import librosa.core.spectrum as spectrum
import librosa.feature.spectral as spectral
import librosa.util

import numpy as np
from scipy.fftpack import fft


class AudioFeatureExtraction(object):

    @staticmethod
    def spectrogram(signal, sr, **kwargs):
        """
            preprocess_method: "spectrogram", which calculates the spectrogram of a given audio signal data.

            Args:
            - signal: array
                The audio signal data.
            - sr: int
                The sample rate of the audio signal.
            - **kwargs: Additional parameters
                - extraction_kwargs: dictionary
                    Parameters for STFT operations.(n_fft: 2048, hop_length: 512)
                - time_series_first: bool
                    Whether to return a time series first or not, the default is True.
                - flatten：bool
                    Whether to flatten the spectral graph to 1 dimension or not, the default value is False.

            Returns:
            - spec: array
                The spectrogram of the signal.
        """
        extraction_kwargs = kwargs.get("extraction_kwargs", {})
        spec = np.abs(spectrum.stft(y=signal, **extraction_kwargs))  # The short-time Fourier transform(STFT)

        if kwargs.get("time_series_first", True):
            spec = spec.T
        if kwargs.get("flatten", False):
            spec = spec.reshape((1, spec.shape[0] * spec.shape[1]))[0]
        return spec

    @staticmethod
    def mfcc(signal, sr, **kwargs):
        """
            preprocess_method: "mfcc", which calculates the Mel-frequency cepstrum coefficient (MFCCs) for the given audio signal.
            Args:
            - signal: array
                The audio signal data.
            - sr: int
                The sample rate of the audio signal.
            - **kwargs: Additional parameters
                - extraction_kwargs: dictionary
                    Parameters for mfcc operations.(n_mfcc: 20, n_fft: 2048, hop_length: 512)
                - time_series_first: bool
                    Whether to return a time series first or not, the default is True.
                - flatten：bool
                    Whether to flatten the spectral graph to 1 dimension or not, the default value is False.

            Returns:
            - mfcc: array
                The MFCCs of the original audio signal.
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
            preprocess_method: "mel_spec", which computes the Mel-spectrogram of the given audio signal.

            Args:
            - signal: array
                The audio signal data.
            - sr: int
                The sample rate of the audio signal.
            - **kwargs: Additional parameters
                - extraction_kwargs: dictionary
                    Parameters for mel-spectrogram operations.(n_fft: 2048, hop_length: 256)
                - time_series_first: bool
                    Whether to return a time series first or not, the default is True.
                - flatten：bool
                    Whether to flatten the spectral graph to 1 dimension or not, the default value is False.

            Returns:
            - mel_spec: array
                The Mel-spectrogram of the original audio signal.
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
            preprocess_method: "zero_crossing_rate", which calculates the zero-crossing rate of the given audio signal.

            Args:
            - signal: array
                The audio signal data.
            - sr: int
                The sample rate of the audio signal.
            - **kwargs: Additional parameters
                - extraction_kwargs: dictionary
                    Parameters for zero_crossing_rate operations.(frame_length: 2048, hop_length: 256)
                - time_series_first: bool
                    Whether to return a time series first or not, the default is True.
                - flatten：bool
                    Whether to flatten the spectral graph to 1 dimension or not, the default value is False.

            Returns:
            - zcr: array
                The zero-crossing rate of the original audio signal.
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
            preprocess_method: "spectral_flatness", which calculates the spectral_flatness of the given audio signal.

            Args:
            - signal: array
                The audio signal data.
            - sr: int
                The sample rate of the audio signal.
            - **kwargs: Additional parameters
                - extraction_kwargs: dictionary
                    Parameters for zero_crossing_rate operations.(n_fft: 2048, hop_length: 256)
                - time_series_first: bool
                    Whether to return a time series first or not, the default is True.
                - flatten：bool
                    Whether to flatten the spectral graph to 1 dimension or not, the default value is False.

            Returns:
            - spectral_flatness: array
                The spectral flatness of the original audio signal.
        """
        extraction_kwargs = kwargs.get("extraction_kwargs", {})
        spectral_flatness = spectral.spectral_flatness(y=signal, **extraction_kwargs)
        if kwargs.get("time_series_first", True):
            spectral_flatness = spectral_flatness.T
        if kwargs.get("flatten", False):
            spectral_shape = spectral_flatness.shape
            spectral_flatness = spectral_flatness.reshape((1, spectral_shape[0] * spectral_shape[1]))[0]
        return spectral_flatness

    @staticmethod
    def sound_pressure_level(signal):
        """
            Calculates the sound pressure level (SPL) of the given audio signal.

            Args:
            - signal: array
                The audio signal data.
            Returns:
            - spl: int
                The sound pressure level of the given audio signal.
        """
        rms = np.sqrt(np.mean(signal ** 2))
        p0 = 20e-6
        spl = 20 * np.log10(rms / p0)
        return spl

    @staticmethod
    def data_normalize(signal, sr, **kwargs):
        """To normalize audio raw data to [-1, 1] """
        signal_normalized = librosa.util.normalize(signal)
        return signal_normalized

    def total_harmonic_distortion(self, signal, sr, **kwargs):
        """
            Calculates the total harmonic distortion (THD) of the given audio signal.

            Args:
            - signal: array
                The audio signal data.
            - sr: int
                The sample rate of the audio signal.
            - **kwargs: Additional parameters
                - extraction_kwargs: dictionary
                    Parameters for spectrogram operations.
                - time_series_first: bool
                    Whether to return a time series first or not, the default is True.
                - flatten：bool
                    Whether to flatten the spectral graph to 1 dimension or not, the default value is False.
                - window_size: int
                    FFT calculates the size of the window
                - step_size: int
                    The step size of the window slide.

            Returns:
                An array is used to save the total harmonic distortion (THD) values for each window of the given signal.
        """

        spec_kwargs = {"extraction_kwargs": kwargs.get("extraction_kwargs", {}),
                       "time_series_first": True,
                       "flatten": False}
        spec = self.spectrogram(signal, sr, **spec_kwargs)

        xf = np.fft.fftfreq(len(spec[0]), 1 / sr)[:len(spec[0]) // 2]
        fundamental_idx = np.argmax(spec)

        window_size = kwargs.get("window_size", 2048)
        step_size = kwargs.get("step_size", 256)
        num_windows = (len(signal) - window_size) // step_size + 1
        windows = np.array([signal[i * step_size:i * step_size + window_size] for i in range(num_windows)])

        thd_list = []

        for window in windows:
            N = len(window)

            yf = fft(window)

            yf = np.abs(yf[:N // 2])
            yf = yf * 2 / N
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
    def high_order_harmonic_distortion(signal, sr, **kwargs):
        """
            Calculates the High Order Harmonic Distortion (HOHD) of the given audio signal.

            Args:
            - signal: array
                The audio signal data.
            - sr: int
                The sample rate of the audio signal.
            - **kwargs: Additional parameters
                - window_size: int
                    FFT calculates the size of the window
                - step_size: int
                    The step size of the window slide.

            Returns:
                An array is used to save the High Order Harmonic Distortion (HOHD) values for each window of the given signal.
        """

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
            yf = yf * 2 / N
            xf = xf[:N // 2]

            fundamental_freq = xf[np.argmax(yf)]
            fundamental_amplitude = yf[np.argmax(yf)]

            harmonics = [fundamental_freq * (i + 1) for i in range(9, 35)]  # Select the 10th through 35th harmonics
            harmonic_amplitudes = []

            for harmonic in harmonics:
                harmonic_idx = np.argmin(np.abs(xf - harmonic))
                harmonic_amplitudes.append(yf[harmonic_idx])

            hohd = (np.sqrt(sum([amp ** 2 for amp in harmonic_amplitudes])) / fundamental_amplitude) * 100
            hohd_list.append(hohd)

        return np.array(hohd_list)

    @staticmethod
    def intermodulation_distortion(signal, sr, **kwargs):
        """
            Calculates the intermodulation distortion (IMD) of the given audio signal.

            Args:
            - signal: array
                The audio signal data.
            - sr: int
                The sample rate of the audio signal.
            - **kwargs: Additional parameters
                - window_size: int
                    FFT calculates the size of the window
                - step_size: int
                    The step size of the window slide.

            Returns:
                An array is used to save the intermodulation distortion (IMD) for each window of the given signal.
        """
        window_size = kwargs.get("window_size", 2048)
        step_size = kwargs.get("step_size", 256)
        num_windows = (len(signal) - window_size) // step_size + 1
        windows = np.array([signal[i * step_size:i * step_size + window_size] for i in range(num_windows)])

        imd_list = []

        for window in windows:
            N = len(window)
            yf = np.fft.fft(window)
            xf = np.fft.fftfreq(N, 1 / sr)  # Returns the discrete Fourier transform sampling frequency.

            yf = np.abs(yf[:N // 2])
            yf = yf * 2 / N
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

            imd = (np.sqrt(np.sum(clean_spectrum ** 2)) / np.sqrt(np.sum(yf ** 2))) * 100
            imd_list.append(imd)

        return np.array(imd_list)

    @staticmethod
    def frequency_response(signal, sr, output_db="db", **kwargs):
        """
            Calculates the frequency response of a given audio signal.

            Args:
            - signal: array
                The audio signal data.
            - sr: int
                The sample rate of the audio signal.
            - output_db: string
                Specifies the db output format.
            - **kwargs: Additional parameters
                - window_size: int
                    FFT calculates the size of the window
                - step_size: int
                    The step size of the window slide.

            Returns:
                An array is used to save the frequency response for each window of the given signal.
        """
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
                yf_db = 20 * np.log10(yf * 2 / N)
                xf_list.append(yf_db)
            else:
                xf = np.fft.fftfreq(N, 1 / sr)
                xf = xf[:N // 2]
                xf_list.append(xf)

        return np.array(xf_list)
