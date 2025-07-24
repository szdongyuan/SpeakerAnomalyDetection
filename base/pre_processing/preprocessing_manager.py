import numpy as np

from base.pre_processing.audio_equalizer import AudioEqualizer
from base.pre_processing.audio_feature_extraction import AudioFeatureExtraction
from base.pre_processing.data_alignment import DataAlignment
from base.pre_processing.emphasis import Emphasis
from base.pre_processing.split_repeat_signal import SplitRepeatSignal


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

            Categories:：
                1. Feature Extraction:
                    - "spectrogram"         : Extracts spectrogram features.
                    - "mfcc"                : Extracts MFCC (Mel-frequency cepstral coefficients).
                    - "mel_spec"            : Extracts Mel-spectrogram features.
                    - "zero_crossing_rate"  : Computes the zero crossing rate.
                    - "spectral_flatness"   : Measures the spectral flatness.

                2. Preprocessing:
                    - "data_normalize"      : Normalizes the audio data.
                    - "data_padding"        : Pads audio data to uniform length.
                    - "split_repeat_signal" : Splits and pads the audio signal.

                3. Augmentation:
                    - "apply_equalizer"     : Applies equalizer to modify frequency response.
                    - "random_fluctuation"  : Applies random fluctuations for augmentation.

                4. Pipeline / Structuring:
                    - "sequence_process"    : Sequentially applies multiple processing steps.
                    - "stack_process"       : Stacks data for multiple processing steps.
        """
        process_mapping = {
            "spectrogram": AudioFeatureExtraction.spectrogram,
            "mfcc": AudioFeatureExtraction.mfcc,
            "mel_spec": AudioFeatureExtraction.mel_spec,
            "zero_crossing_rate": AudioFeatureExtraction.zero_crossing_rate,
            "data_normalize": AudioFeatureExtraction.data_normalize,
            "spectral_flatness": AudioFeatureExtraction.spectral_flatness,
            "data_padding": DataAlignment.data_padding,
            "apply_equalizer": AudioEqualizer.apply_equalizer,
            "random_fluctuation": Emphasis.random_fluctuation,
            "split_repeat_signal": SplitRepeatSignal.split_repeat_signal,
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

        if signal.ndim == 1:
            return process_handler(signal, sr, **process_kwargs)

        process_segments = []
        for segment in signal:
            process_segment = process_handler(segment, sr, **process_kwargs)
            process_segments.append(process_segment)
        return np.stack(process_segments, axis=-1)


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
