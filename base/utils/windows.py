import numpy as np
import librosa
from scipy.signal.windows import flattop, kaiser, gaussian, nuttall, tukey


class Add_Windows(object):
    def __init__(self, audio_data, sr=44100):
        self.sr = sr
        self.audio_data = audio_data

    def apply_window(self, window_size, hop_size, window_type='hamming'):
        r"""
            The purpose of adding Windows is to reduce frequency leakage.

            Parameters
            ----------
            in1 : window_size
            in2 : hop_size

            window_type : {'hamming', 'hann', 'blackman', 'rectangular',
            'bartlett','flattop', 'kaiser', 'gaussian', 'nuttall', 'tukey'}
                A string indicating the size of the output:

                ``hamming``
                    Provides smooth transitions and better sidelobe suppression.
                ``hann``
                    Similar to Hamming Windows, but with a higher main lobe and a lower side lobe.
                ``blackman``
                    It has a lower sidelobe than Hann and Hamming Windows and is often used for accurate spectral analysis.
                ``rectangular``
                    The simplest window function is equivalent to no window.
                ``bartlett``
                    It has a lower sidelobe than rectangular Windows,
                    but the main lobe is wider than Hamming and Hann Windows,
                    which is suitable for signal processing where smoothness is required.
                ``flattop``
                    Provides a very flat frequency response for accurate amplitude measurement.
                ``kaiser``
                   The main lobe width and sidelobe suppression can be balanced by
                   adjusting the parameter β, which is a very flexible window function.
                ``gaussian``
                   The shape of the window is adjusted by the parameter σ.
                   Gaussian Windows are often used for applications that
                   require smoothing in both the time and frequency domains.
                ``nuttall``
                   The window function of multiple cosine terms is synthesized to provide
                   better sidelobe suppression effect, which is suitable for high-precision spectrum analysis.
                ``tukey``
                   It can vary between rectangular Windows and Hanning Windows by adjusting the alpha parameter,
                   and is suitable for applications that require flexible control of the main lobe and side lobe.

            Returns
            -------
                Returns the windowed signal and window function.
            """

        if window_type == 'hamming':
            window_func = np.hamming
        elif window_type == 'hann':
            window_func = np.hanning
        elif window_type == 'blackman':
            window_func = np.blackman
        elif window_type == 'rectangular':
            window_func = np.ones
        elif window_type == 'bartlett':
            window_func = np.bartlett
        elif window_type == 'flattop':
            window_func = lambda size: flattop(size)
        elif window_type == 'kaiser':
            window_func = lambda size: kaiser(size, beta=14)
        elif window_type == 'gaussian':
            window_func = lambda size: gaussian(size, std=size / 6)
        elif window_type == 'nuttall':
            window_func = lambda size: nuttall(size)
        elif window_type == 'tukey':
            window_func = lambda size: tukey(size, alpha=0.5)
        else:
            raise ValueError("Unsupported window type")

        window = window_func(window_size)

        frames = librosa.util.frame(self.audio_data, frame_length=window_size, hop_length=hop_size).T

        # apply window
        windowed_frames = np.array([frame * window for frame in frames])

        # new windowed signal
        windowed_signal = np.zeros(len(self.audio_data))
        for i, frame in enumerate(windowed_frames):
            start = i * hop_size
            end = start + window_size
            windowed_signal[start:end] += frame

        return windowed_signal, window
