import numpy as np

from base.log_manager import LogManager
from base.save_data import save_audio_simple
from base.wav_pcm24 import quantize_pcm24
from base.sound_device_manager import SoundDeviceManager, sd
from consts import error_code


class SoundcardAudioProcessor(object):

    def __init__(self):
        self.logger = LogManager.set_log_handler("soundcard_core")

    def sd_play_rec(self, record_dict, stimulus_dict, recording_path):
        data = stimulus_dict.get("data") * stimulus_dict.get("amplitude")
        prepare_frames = record_dict.get("prepare_frames", 1000)
        prolong_frames = record_dict.get("prolong_frames", 10000)
        prolong_data = [0] * prepare_frames + list(data) + [0] * prolong_frames
        sr = stimulus_dict.get("sr")

        in_sel = record_dict.get("input_channels")
        if in_sel is None:
            channels = int(record_dict.get("channels", 1) or 1)
            in_sel = list(range(max(1, channels)))
        try:
            in_sel = sorted({int(i) for i in in_sel if int(i) >= 0})
        except Exception:
            in_sel = [0]
        if not in_sel:
            in_sel = [0]

        in_num = max(in_sel) + 1

        code, output = SoundDeviceManager().get_default_device("speaker", refresh=False)
        if code != error_code.OK or not output:
            return error_code.INVALID_PLAY, "No system default output is available."
        input_device = record_dict.get("device")
        if input_device is None:
            input_device = record_dict.get("input_device")
        if isinstance(input_device, dict):
            input_device = input_device.get("index")
        with SoundDeviceManager.convenience_operation() as acquired:
            if not acquired:
                return error_code.INVALID_PLAY, "Audio backend is busy with another operation."
            if SoundDeviceManager.output_would_interrupt():
                return error_code.INVALID_PLAY, "System default output cannot replace another audio operation."
            try:
                rec_raw = sd.playrec(prolong_data, samplerate=sr, channels=in_num,
                                     device=(input_device, output["index"]), blocking=True)
            except Exception as exc:
                # PortAudio/driver boundary: reject incompatible or unavailable
                # input/output combinations without retrying another output or
                # stopping streams owned by other operations.
                return error_code.INVALID_PLAY, (
                    f"Failed to play/record with selected input and system default output: {str(exc)[:80]}")
        rec_raw = np.asarray(rec_raw, dtype=np.float32)
        if rec_raw.ndim == 1:
            rec_raw = rec_raw.reshape(-1, 1)

        rec_sel = rec_raw[:, in_sel] if rec_raw.shape[1] > 1 else rec_raw[:, [0]]
        rec_mono = rec_sel.mean(axis=1).astype(np.float32, copy=False)

        align_frames = self.calculate_alignment(data, rec_mono)
        if align_frames < 0:
            align_frames = 0

        end_frame = align_frames + len(data)
        if end_frame > rec_sel.shape[0]:
            end_frame = rec_sel.shape[0]

        aligned_multi = rec_sel[align_frames:end_frame, :].astype(np.float32, copy=False)
        if aligned_multi.shape[0] < len(data):
            shortfall = len(data) - aligned_multi.shape[0]
            aligned_multi = np.concatenate(
                [aligned_multi, np.zeros((shortfall, aligned_multi.shape[1]), dtype=np.float32)], axis=0
            )

        # Save multi-channel aligned data. keep mono return for compatibility.
        aligned_multi = quantize_pcm24(aligned_multi)
        save_audio_simple(recording_path, aligned_multi, sr)
        record_dict["_recorded_multi"] = aligned_multi
        aligned_mono = aligned_multi.mean(axis=1).astype(np.float32, copy=False)
        return error_code.OK, aligned_mono

    @staticmethod
    def sd_play(stimulus_params):
        """Play on the system output; legacy stimulus device is ignored."""
        data = stimulus_params.get("data") * stimulus_params.get("amplitude")
        sr = stimulus_params.get("sr")
        blocking = stimulus_params.get("blocking", True)
        code, output = SoundDeviceManager().get_default_device("speaker", refresh=False)
        if code != error_code.OK or not output:
            return error_code.INVALID_PLAY, "No system default output is available."
        with SoundDeviceManager.convenience_operation() as acquired:
            if not acquired:
                return error_code.INVALID_PLAY, "Audio backend is busy with another operation."
            if SoundDeviceManager.output_would_interrupt():
                return error_code.INVALID_PLAY, "System default output cannot replace another audio operation."
            try:
                sd.play(data, samplerate=sr, device=output["index"], blocking=blocking)
            except Exception as e:
                # Normalize backend open/play errors at the audio API boundary.
                err_msg = "Failed to play audio on system default output. [%s]" % (str(e)[:80])
                return error_code.INVALID_PLAY, err_msg
        return error_code.OK, "play successfully"

    @staticmethod
    def sd_rec(recorded_dict):
        num_frames = recorded_dict.get("num_frames", 441000)
        sample_rate = recorded_dict.get("sample_rate", 44100)
        channels = int(recorded_dict.get("channels", 1) or 1)
        blocking = recorded_dict.get("blocking", True)
        prolong_frames = recorded_dict.get("prolong_frames", 0)
        device = recorded_dict.get("device")
        if device is None:
            device = recorded_dict.get("input_device")
        if isinstance(device, dict):
            device = device.get("index")

        in_sel = recorded_dict.get("input_channels")
        if in_sel is None:
            in_sel = list(range(max(1, channels)))
        try:
            in_sel = sorted({int(i) for i in in_sel if int(i) >= 0})
        except Exception:
            in_sel = [0]
        if not in_sel:
            in_sel = [0]

        in_num = max(in_sel) + 1
        with SoundDeviceManager.convenience_operation() as acquired:
            if not acquired:
                return error_code.INVALID_RECORD, "Audio backend is busy with another operation."
            if SoundDeviceManager.output_would_interrupt():
                return error_code.INVALID_RECORD, "Recording cannot replace another audio operation."
            rec_raw = sd.rec(
                frames=num_frames,
                samplerate=sample_rate,
                channels=in_num,
                device=device,
                blocking=blocking,
            )
        rec_raw = np.asarray(rec_raw, dtype=np.float32)
        if rec_raw.ndim == 1:
            rec_raw = rec_raw.reshape(-1, 1)

        rec_sel = rec_raw[:, in_sel] if rec_raw.shape[1] > 1 else rec_raw[:, [0]]
        if prolong_frames > 0:
            rec_sel = rec_sel[int(prolong_frames):, :]

        try:
            recorded_dict["_recorded_multi"] = rec_sel.astype(np.float32, copy=False)
        except Exception:
            pass

        rec_mono = rec_sel.mean(axis=1).astype(np.float32, copy=False)
        return error_code.OK, rec_mono

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
