import numpy as np

from base.pre_processing.audio_feature_extraction import AudioFeatureExtraction
from base.pre_processing.data_alignment import DataAlignment


class PreprocessingManager(object):

    @classmethod
    def get_processor(cls, process_method):
        process_mapping = {
            "spectrogram": AudioFeatureExtraction.spectrogram,
            "mfcc": AudioFeatureExtraction.mfcc,
            "mel_spec": AudioFeatureExtraction.mel_spec,
            "zero_crossing_rate": AudioFeatureExtraction.zero_crossing_rate,
            "spectral_flatness": AudioFeatureExtraction.spectral_flatness,
            "data_padding": DataAlignment.data_padding,
            "sequence_process": cls.sequence_process,
            "stack_process": cls.stack_process,
        }
        return process_mapping.get(process_method)

    def process(self, signal, sr, **kwargs):
        process_method = kwargs.get("preprocess_method")
        if not process_method:
            return signal

        process_kwargs = kwargs.get("preprocess_param", {})
        process_handler = self.get_processor(process_method)
        if not process_handler:
            return signal
        return process_handler(signal, sr, **process_kwargs)

    @staticmethod
    def sequence_process(signal, sr, **kwargs):
        for processor_kwargs in kwargs.get("processor_list", []):
            signal = PreprocessingManager().process(signal, sr, **processor_kwargs)
        return signal

    @staticmethod
    def stack_process(signal, sr, **kwargs):
        stacked_result = []
        for processor_kwargs in kwargs.get("processor_list", []):
            stacked_result.append(PreprocessingManager().process(signal, sr, **processor_kwargs))
        return np.hstack(stacked_result)
