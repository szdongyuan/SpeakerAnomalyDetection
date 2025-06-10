import numpy as np


class DataAlignment(object):

    @staticmethod
    def data_padding(signal, sr, **kwargs):
        """
            Pads or truncates a given audio signal to a specified length.

            Args:
            - signal: array
                The audio signal data.
            - sr: float
                The sample rate of the audio signal.
            - **kwargs: Additional keyword arguments
                - dtype: string
                    The data type of the numpy array returned.
                - maxlen: int
                    This parameter is the maximum length of the sequence. Sequences greater than this length will be truncated,
                    and sequences less than this length will be followed by a 0.
                - padding: string
                    'pre' or 'post', which determines whether 0 should be filled at the beginning or the end of the sequence.
                - truncating: string
                    'pre 'or' post ', which determines whether to truncate the sequence from the beginning or the end when it needs to..be truncated

            Returns:
            - padded_inputs: array
                A numpy array contains padded or truncated signal data.
        """
        dtype = kwargs.get("dtype", "float32")
        maxlen = kwargs.get("maxlen", 66150)
        padding = kwargs.get("padding", "post")
        truncating = kwargs.get("truncating", "post")
        signal_length = len(signal)
        if signal_length > maxlen:
            if truncating == "post":
                padded_inputs = signal[:maxlen]
            else:
                padded_inputs = signal[-maxlen:]
        elif signal_length < maxlen:
            padding_length = maxlen - signal_length
            if padding == "post":
                padded_inputs = np.pad(signal, (0, padding_length), mode='constant', constant_values=0)
            else:
                padded_inputs = np.pad(signal, (padding_length, 0), mode='constant', constant_values=0)
        else:
            padded_inputs = signal
        return np.array(padded_inputs, dtype=dtype)

    @staticmethod
    def chop_data(raw_inputs, chop_head=0, chop_tail=None):
        """
            Extract a specific part of the data.

            Args:
            - raw_inputs: array
                Data that needs to be chopped.
            - chop_head: int
                The starting index of the chop(included).
            - chop_tail: int or None, optional
                The ending index for the chop (excluded). If None, it chops to the end of the data.

            Returns:
            - chopped_data: array
                The selected section of data.
        """

        chopped_data = raw_inputs[:, chop_head:chop_tail]
        return chopped_data
