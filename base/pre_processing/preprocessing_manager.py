import numpy as np

from base.pre_processing.audio_feature_extraction import AudioFeatureExtraction
from base.pre_processing.data_alignment import DataAlignment


class PreprocessingManager(object):

    @classmethod
    def get_processor(cls, process_method):
        """
            Returns the corresponding function based on the specified method name.

            Args:
            - process_method: string
                The method name for audio data preprocessing.

            Returns:
                The corresponding function to the given method name.
        """
        process_mapping = {
            "spectrogram": AudioFeatureExtraction.spectrogram,
            "mfcc": AudioFeatureExtraction.mfcc,
            "mel_spec": AudioFeatureExtraction.mel_spec,
            "zero_crossing_rate": AudioFeatureExtraction.zero_crossing_rate,
            "data_normalize": AudioFeatureExtraction.data_normalize,
            "spectral_flatness": AudioFeatureExtraction.spectral_flatness,
            "data_padding": DataAlignment.data_padding,
            "sequence_process": cls.sequence_process,
            "stack_process": cls.stack_process,
        }
        return process_mapping.get(process_method)

    def process(self, signal, sr, **kwargs):
        """
            The original audio signal is processed using the specified preprocessing method.

            Args:
            - signal: array
                The original audio signal data.
            - sr: int
                The sample rate of original audio signal data.
            - **kwargs: dictionary
                Additional parameters of the preprocessing method.

            Returns:
                Return the preprocessed audio signal data if the specified preprocessing method can be found
                otherwise return the original signal.
        """
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
        """
            Apply all specified preprocessing methods to the raw audio signal data.

            Args:
            - signal: array
                The original audio signal data.
            - sr: int
                The sample rate of original audio signal data.
            - **kwargs: dictionary
                A dictionary containing a list of parameters for each preprocessing method.

            Returns:
                An audio signal that has been processed by all specified preprocessing methods.
        """
        for processor_kwargs in kwargs.get("processor_list", []):
            signal = PreprocessingManager().process(signal, sr, **processor_kwargs)
        return signal

    @staticmethod
    def stack_process(signal, sr, **kwargs):
        """
            Apply all specified preprocessing methods to the raw audio signal data
            and stack the preprocessing results.

            Args:
            - signal: array
                The original audio signal data.
            - sr: int
                The sample rate of original audio signal data.
            - **kwargs: dictionary
                A dictionary containing a list of parameters for each preprocessing method.

            Returns:
                An array contains all the audio signals obtained by the specified preprocessing method.
        """
        stacked_result = []
        for processor_kwargs in kwargs.get("processor_list", []):
            stacked_result.append(PreprocessingManager().process(signal, sr, **processor_kwargs))
        return np.hstack(stacked_result)
