import librosa
import numpy as np
from scipy.signal import correlate

class data_alignment(object):
    def __init__(self, audio_data, input_data, sr=44100):
        self.sr = sr
        self.audio_data = audio_data
        self.input_data = input_data


    def onset_detect(self):
        """
        This algorithm is used for the calculation of long time delay of music signals,
        may not be suitable for loudspeaker signals
        """
        onsets_frames = librosa.onset.onset_detect(y=self.audio_data, sr=self.sr)
        onsets_times = librosa.frames_to_time(onsets_frames, sr=self.sr)
        first_onset_time = onsets_times[0] if len(onsets_times) > 0 else None

        music_delay_frames = int(first_onset_time * self.sr) if first_onset_time is not None else 0

        return music_delay_frames, first_onset_time, onsets_times

    def cross_correlation(self):
        cross_correlation = correlate(self.input_data, self.audio_data, mode="full")
        delay = np.argmax(cross_correlation) - (len(self.audio_data) - 1)
        delay_frames = np.abs(delay)
        delay_seconds = delay_frames / self.sr
        return delay_frames, delay_seconds

    def phase_correlation(self, input_data, audio_data):
        min_len = min(len(input_data), len(audio_data))
        x_trimmed = input_data[:min_len]
        y_trimmed = audio_data[:min_len]

        # fft
        X = np.fft.fft(x_trimmed)
        Y = np.fft.fft(y_trimmed)

        # Phase difference
        R = X * np.conj(Y)  # frequency multiply
        R /= np.abs(R)  # normalization

        # ifft
        r = np.fft.ifft(R).real

        delay_index = np.argmax(r)
        if delay_index > min_len // 2:
            delay_index -= min_len

        delay_time = delay_index

        return delay_time, r

