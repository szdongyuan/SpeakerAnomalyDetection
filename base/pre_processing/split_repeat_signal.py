import math
import numpy as np


class SplitRepeatSignal(object):

    @staticmethod
    def split_repeat_signal(audio_data, sr, **kwargs):
        """
        Split the audio into `repeat_times` segments of equal length.
        If the length is not divisible, the last segment is taken directly
        from the end of the original audio to match the segment length.

        Returns:
            A 2D ndarray of shape (repeat_times, segment_length)
        """
        repeat_times = kwargs.get("repeat_times", 1)

        if not repeat_times or repeat_times <= 0:
            return np.empty([[]], dtype=np.float32)

        if repeat_times == 1:
            return np.array([audio_data], dtype=np.float32)

        total_len = len(audio_data)
        seg_len = math.ceil(total_len / repeat_times)

        if seg_len == 0:
            return np.array([[]], dtype=np.float32)

        segments = []
        for i in range(repeat_times - 1):
            start = i * seg_len
            end = start + seg_len
            segment = audio_data[start:end]
            segments.append(segment)
        segments.append(audio_data[-seg_len:])
        return np.array(segments, dtype=np.float32)
