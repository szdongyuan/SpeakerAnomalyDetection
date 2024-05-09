import librosa.feature.spectral as spectral
import numpy as np


class AudioFeatureExtraction(object):

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
