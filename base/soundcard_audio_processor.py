import numpy as np

from base.log_manager import LogManager
from base.save_data import save_audio_simple
from base.sound_device_manager import sd
from consts import error_code


def alignment_reference_from_stimulus(stimulus_dict):
    data = np.asarray(stimulus_dict.get("data", []))
    count = stimulus_dict.get("alignment_sample_count")
    if count is None:
        return data
    try:
        count = int(count)
    except (TypeError, ValueError):
        return data
    if count <= 0 or count > data.shape[0]:
        return data
    return data[:count]


def bounded_aligned_recording_slice(recorded_signal, start_frame, sample_count):
    recorded_signal = np.asarray(recorded_signal)
    sample_count = int(sample_count)
    if sample_count <= 0:
        return recorded_signal[:0]

    try:
        start_frame = int(start_frame)
    except (TypeError, ValueError):
        start_frame = 0

    start_frame = max(0, min(start_frame, recorded_signal.shape[0]))
    end_frame = min(start_frame + sample_count, recorded_signal.shape[0])
    copied_count = max(0, end_frame - start_frame)
    aligned_data = np.zeros(sample_count, dtype=recorded_signal.dtype)
    if copied_count:
        aligned_data[:copied_count] = recorded_signal[start_frame:end_frame]
    return aligned_data


class SoundcardAudioProcessor(object):

    def __init__(self):
        self.logger = LogManager.set_log_handler("soundcard_core")

    @staticmethod
    def _coerce_nonnegative_frames(value, default=0):
        if isinstance(value, bool):
            return default
        try:
            frames = int(value)
        except (TypeError, ValueError):
            return default
        return frames if frames >= 0 else default

    @staticmethod
    def _device_index(device):
        if isinstance(device, dict):
            return int(device["index"])
        return device

    @classmethod
    def _playrec_device_selector(cls, record_dict):
        input_device = record_dict.get("input_device") or record_dict.get("device")
        output_device = record_dict.get("output_device")
        input_index = cls._device_index(input_device)
        output_index = cls._device_index(output_device)

        if input_index is not None and output_index is not None:
            if input_index == output_index:
                return input_index
            return input_index, output_index
        if input_index is not None:
            return input_index, None
        if output_index is not None:
            return None, output_index
        return None

    def sd_play_rec(self, record_dict, stimulus_dict, recording_path):
        data = np.asarray(stimulus_dict.get("data")) * stimulus_dict.get("amplitude")
        alignment_reference = alignment_reference_from_stimulus(
            {
                "data": data,
                "alignment_sample_count": stimulus_dict.get("alignment_sample_count"),
            }
        )
        prepare_frames = self._coerce_nonnegative_frames(record_dict.get("prepare_frames", 1000), 1000)
        prolong_frames = self._coerce_nonnegative_frames(record_dict.get("prolong_frames", 10000), 10000)
        delay_frames = self._coerce_nonnegative_frames(
            record_dict.get("recording_start_delay_frames", 0), 0
        )
        prolong_data = np.concatenate(
            [
                np.zeros(delay_frames + prepare_frames, dtype=data.dtype),
                data,
                np.zeros(prolong_frames, dtype=data.dtype),
            ]
        )
        sr = stimulus_dict.get("sr")
        device = self._playrec_device_selector(record_dict)
        if device is None:
            rec_data = sd.playrec(prolong_data, samplerate=sr, channels=1, blocking=True).T[0]
        else:
            rec_data = sd.playrec(prolong_data, samplerate=sr, channels=1, blocking=True, device=device).T[0]
        if delay_frames > 0:
            rec_data = rec_data[delay_frames:]
        align_frames = self.calculate_alignment(alignment_reference, rec_data)
        aligned_data = bounded_aligned_recording_slice(
            rec_data, align_frames, len(alignment_reference)
        )
        save_audio_simple(recording_path, aligned_data, sr)
        return error_code.OK, aligned_data

    @staticmethod
    def sd_play(stimulus_params):
        try:
            data = stimulus_params.get("data") * stimulus_params.get("amplitude")
            output_channels = stimulus_params.get("output_channels")
            if (
                isinstance(output_channels, int)
                and not isinstance(output_channels, bool)
                and output_channels >= 2
            ):
                data = np.asarray(data)
                if data.ndim == 1 or (data.ndim == 2 and data.shape[1] == 1):
                    data = np.tile(data.reshape(-1, 1), (1, output_channels))
            sr = stimulus_params.get("sr")
            blocking = stimulus_params.get("blocking", True)
            device = stimulus_params.get("device", None)
            sd.play(data, samplerate=sr, device=device, blocking=blocking)
            return error_code.OK, "play successfully"
        except Exception as e:
            err_msg = "Failed to play audio. [%s]" % (str(e)[:50])
            return error_code.INVALID_PLAY, err_msg

    @staticmethod
    def sd_rec(recorded_dict):
        num_frames = SoundcardAudioProcessor._coerce_nonnegative_frames(
            recorded_dict.get("num_frames", 441000), 441000
        )
        sample_rate = recorded_dict.get("sample_rate", recorded_dict.get("sr", 44100))
        channels = recorded_dict.get("channels", 1)
        blocking = recorded_dict.get("blocking", True)
        prolong_frames = SoundcardAudioProcessor._coerce_nonnegative_frames(
            recorded_dict.get("prolong_frames", 0), 0
        )
        delay_frames = SoundcardAudioProcessor._coerce_nonnegative_frames(
            recorded_dict.get("recording_start_delay_frames", 0), 0
        )
        device = recorded_dict.get("device") or recorded_dict.get("input_device")
        device = SoundcardAudioProcessor._device_index(device)
        input_channels = recorded_dict.get("input_channels")
        selected_channels = []
        if input_channels is not None:
            if isinstance(input_channels, int):
                selected_channels = [input_channels]
            else:
                selected_channels = list(input_channels)

        record_channels = max(selected_channels) + 1 if selected_channels else channels
        recorded_data = sd.rec(
            frames=num_frames + delay_frames,
            samplerate=sample_rate,
            channels=record_channels,
            device=device,
            blocking=blocking,
        )
        recorded_data = np.asarray(recorded_data)
        trim_frames = delay_frames + prolong_frames
        if trim_frames > 0:
            recorded_data = recorded_data[trim_frames:, ...]

        if recorded_data.ndim == 1:
            return error_code.OK, recorded_data

        if selected_channels:
            if selected_channels == [0]:
                recorded_data = recorded_data[:, 0]
            else:
                recorded_data = recorded_data[:, selected_channels]
        elif channels == 1:
            recorded_data = recorded_data[:, 0]

        return error_code.OK, recorded_data

    @staticmethod
    def gcc_phat(stimulus_signal, recorded_signal):
        """计算GCC-PHAT互相关函数并返回延迟。"""
        n = len(recorded_signal) + len(stimulus_signal)
        n_11 = len(stimulus_signal) // 11
        SIG = np.fft.rfft(recorded_signal, n=n)
        REF = np.fft.rfft(stimulus_signal, n=n)
        R = SIG * np.conj(REF)
        max_shift = n // 2
        corr_func_r = np.fft.irfft(R)
        corr_func_shifted_r = np.fft.fftshift(corr_func_r)
        new_delay_samples_r = 0
        tmp_max = 0
        for i in range(n // 3, n - len(stimulus_signal) // 12, len(stimulus_signal) // 12):
            max_min_diff = max(corr_func_shifted_r[i: i + n_11]) - min(corr_func_shifted_r[i: i + n_11])
            if max_min_diff >= tmp_max:
                tmp_max = max_min_diff
                new_delay_samples_r = i + np.argmax(np.abs(corr_func_shifted_r[i: i + n_11]))
        new_delay_samples_r -= max_shift
        return new_delay_samples_r, corr_func_shifted_r, max_shift

    @staticmethod
    def calculate_alignment(stimulus_signal, recorded_signal):
        """
        使用GCC-PHAT对齐两个信号。

        Args:
            stimulus_signal (np.ndarray): 激励信号。
            recorded_signal (np.ndarray): 录音信号。
        Returns:
            int: 计算出的对齐帧数（延迟）。
        """
        align_frames, corr_func, max_shift = SoundcardAudioProcessor.gcc_phat(stimulus_signal, recorded_signal)
        return align_frames
