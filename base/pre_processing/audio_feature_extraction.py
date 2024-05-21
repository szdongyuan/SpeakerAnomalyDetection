import librosa.core.spectrum as spectrum
import librosa.feature.spectral as spectral
import numpy as np


class AudioFeatureExtraction(object):

    @staticmethod
    def spectrogram(signal, sr, **kwargs):
        extraction_kwargs = kwargs.get("extraction_kwargs", {})
        spec = np.abs(spectrum.stft(y=signal, **extraction_kwargs))
        if kwargs.get("time_series_first", True):
            spec = spec.T
        if kwargs.get("flatten", False):
            spec = spec.reshape((1, spec.shape[0] * spec.shape[1]))[0]
        return spec

    @staticmethod
    def mfcc(signal, sr, **kwargs):
        extraction_kwargs = kwargs.get("extraction_kwargs", {})
        mfcc = spectral.mfcc(y=signal, sr=sr, **extraction_kwargs)
        if kwargs.get("time_series_first", True):
            mfcc = mfcc.T
        if kwargs.get("flatten", False):
            mfcc = mfcc.reshape((1, mfcc.shape[0] * mfcc.shape[1]))[0]
        return mfcc

    @staticmethod
    def mel_spec(signal, sr, **kwargs):
        extraction_kwargs = kwargs.get("extraction_kwargs", {})
        mel_spec = spectral.melspectrogram(y=signal, sr=sr, **extraction_kwargs)
        if kwargs.get("time_series_first", True):
            mel_spec = mel_spec.T
        if kwargs.get("flatten", False):
            mel_spec = mel_spec.reshape((1, mel_spec.shape[0] * mel_spec.shape[1]))[0]
        return mel_spec

    @staticmethod
    def zero_crossing_rate(signal, sr, **kwargs):
        extraction_kwargs = kwargs.get("extraction_kwargs", {})
        zcr = spectral.zero_crossing_rate(y=signal, **extraction_kwargs)
        if kwargs.get("time_series_first", True):
            zcr = zcr.T
        if kwargs.get("flatten", False):
            zcr = zcr.reshape((1, zcr.shape[0] * zcr.shape[1]))[0]
        return zcr

    @staticmethod
    def spectral_flatness(signal, sr, **kwargs):
        extraction_kwargs = kwargs.get("extraction_kwargs", {})
        spectral_flatness = spectral.spectral_flatness(y=signal, **extraction_kwargs)
        if kwargs.get("time_series_first", True):
            spectral_flatness = spectral_flatness.T
        if kwargs.get("flatten", False):
            spectral_flatness = spectral_flatness.reshape((1, spectral_flatness.shape[0] * spectral_flatness.shape[1]))[0]
        return spectral_flatness
