import numpy as np


class DataAlignment(object):

    @staticmethod
    def _normalize_maxlen(maxlen, default=66150):
        try:
            value = int(maxlen)
        except (TypeError, ValueError):
            value = int(default)
        return max(1, value)

    @staticmethod
    def _resolve_offset_start(signal_length: int, maxlen: int, sr, **kwargs):
        limit = DataAlignment._normalize_maxlen(maxlen)
        total = max(0, int(signal_length))
        max_start = max(0, total - limit)

        offset_samples = kwargs.get("truncating_offset_samples", None)
        if offset_samples not in (None, ""):
            try:
                start = int(offset_samples)
            except (TypeError, ValueError):
                start = 0
            return max(0, min(start, max_start))

        offset_sec = kwargs.get("truncating_offset_sec", None)
        if offset_sec not in (None, ""):
            try:
                start = int(round(float(offset_sec) * float(sr or 0)))
            except (TypeError, ValueError):
                start = 0
            return max(0, min(start, max_start))

        return 0

    @staticmethod
    def _resolve_truncation_slice(signal_length: int, maxlen: int, truncating: str = "post", sr=None, **kwargs):
        total = max(0, int(signal_length))
        limit = DataAlignment._normalize_maxlen(maxlen)
        if total <= limit:
            return 0, total

        mode = str(truncating or "post").strip().lower()
        if mode == "post":
            return 0, limit
        if mode == "center":
            start = max(0, (total - limit) // 2)
            return start, start + limit
        if mode == "offset":
            start = DataAlignment._resolve_offset_start(total, limit, sr, **kwargs)
            return start, start + limit

        # Keep the legacy fallback behavior: any non-"post" value keeps the tail.
        return total - limit, total

    @staticmethod
    def resolve_padded_signal_length(signal_length: int, **kwargs) -> int:
        maxlen = DataAlignment._normalize_maxlen(kwargs.get("maxlen", 66150))
        total = max(0, int(signal_length))
        if total == 0:
            return maxlen
        return maxlen if total != maxlen else total

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
        maxlen = DataAlignment._normalize_maxlen(kwargs.get("maxlen", 66150))
        padding = kwargs.get("padding", "post")
        truncating = kwargs.get("truncating", "post")
        signal_length = len(signal)
        if signal_length > maxlen:
            truncation_kwargs = dict(kwargs)
            truncation_kwargs.pop("truncating", None)
            truncation_kwargs.pop("maxlen", None)
            start, stop = DataAlignment._resolve_truncation_slice(
                signal_length,
                maxlen,
                truncating=truncating,
                sr=sr,
                **truncation_kwargs,
            )
            padded_inputs = signal[start:stop]
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
