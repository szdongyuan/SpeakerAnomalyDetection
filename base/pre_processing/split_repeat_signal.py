import numpy as np


class SplitRepeatSignal(object):

    @staticmethod
    def split_repeat_signal(audio_data, sr, **kwargs):
        """
        Divide the repetitive audio into multiple segments evenly,
        with all segments having the same length.
        """

        repeat_times = kwargs.get("repeat_times", 1)
        if not repeat_times or repeat_times <= 0:
            return np.array([], dtype=np.float32)

        if repeat_times == 1:
            return np.array(audio_data, dtype=np.float32)

        segments = np.array_split(audio_data, repeat_times)
        if not segments or len(segments[0]) == 0:
            return np.array([], dtype=np.float32)

        max_len = max(len(seg) for seg in segments)
        padded_segments = []
        for seg in segments:
            seg_len = len(seg)
            if seg_len < max_len:
                pad_len = max_len - seg_len
                padding = seg[:pad_len]
                padded_seg = np.concatenate([seg, padding])
            else:
                padded_seg = seg
            padded_segments.append(padded_seg)

        return np.array(padded_segments, dtype=np.float32)