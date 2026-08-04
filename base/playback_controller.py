import os
import threading
import time

import numpy as np

from base.sound_device_manager import sd
from consts import error_code


class PlaybackController:

    def __init__(self, monitor_interval_sec: float = 0.08):
        self._playback_lock = threading.RLock()
        self._playback_current_file = None
        self._playback_is_running = False
        self._playback_session_id = 0
        self._playback_stream = None
        self._monitor_interval_sec = float(monitor_interval_sec)

    @staticmethod
    def _close_stream_safely(stream):
        if stream is None:
            return
        try:
            if not getattr(stream, "closed", True):
                stream.close(ignore_errors=True)
        except Exception:
            pass

    @staticmethod
    def _load_playback_audio(file_path):
        try:
            import soundfile as sf

            audio_data, sample_rate = sf.read(file_path, dtype="float32", always_2d=True)
            return np.asarray(audio_data, dtype=np.float32), int(sample_rate)
        except Exception:
            import librosa

            audio_data, sample_rate = librosa.load(file_path, sr=None, mono=False)
            audio_data = np.asarray(audio_data, dtype=np.float32)
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(-1, 1)
            elif audio_data.ndim == 2:
                audio_data = audio_data.T
            else:
                raise RuntimeError("Unsupported audio shape")
            return audio_data, int(sample_rate)

    @staticmethod
    def _get_output_max_channels(device=None):
        try:
            if device is None:
                device_info = sd.query_devices(kind="output")
            else:
                device_info = sd.query_devices(device, "output")
            return int(device_info.get("max_output_channels") or 0)
        except Exception:
            return 0

    @staticmethod
    def _normalize_playback_audio_shape(audio_data):
        audio_data = np.asarray(audio_data, dtype=np.float32)
        if audio_data.ndim == 1:
            return audio_data.reshape(-1, 1)
        if audio_data.ndim == 2:
            return audio_data
        raise RuntimeError("Unsupported audio shape")

    @classmethod
    def _downmix_playback_audio(cls, audio_data, target_channels):
        audio_data = cls._normalize_playback_audio_shape(audio_data)
        target_channels = max(1, int(target_channels or 1))

        mono_data = np.mean(audio_data, axis=1, dtype=np.float32).astype(np.float32).reshape(-1, 1)
        if target_channels >= 2:
            return np.repeat(mono_data, 2, axis=1)
        return mono_data

    @classmethod
    def _prepare_playback_audio(cls, audio_data, output_max_channels=None):
        audio_data = cls._normalize_playback_audio_shape(audio_data)
        source_channels = int(audio_data.shape[1])
        if source_channels <= 0:
            raise RuntimeError("Unsupported audio shape")

        try:
            max_channels = int(output_max_channels or 0)
        except Exception:
            max_channels = 0

        # Recording may be multi-channel for analysis, while normal speakers usually accept only
        # mono/stereo playback. Keep the saved file unchanged and adapt only the playback buffer.
        playback_channel_limit = min(max_channels, 2) if max_channels > 0 else 2
        target_channels = max(1, playback_channel_limit)
        if source_channels <= target_channels and source_channels <= 2:
            return audio_data

        return cls._downmix_playback_audio(audio_data, target_channels)

    def _reset_playback_state_if_session(self, session_id):
        with self._playback_lock:
            if session_id != self._playback_session_id:
                return
            stream = self._playback_stream
            self._playback_current_file = None
            self._playback_is_running = False
            self._playback_stream = None

        self._close_stream_safely(stream)

    def _monitor_playback_done(self, session_id, stream):
        while True:
            with self._playback_lock:
                if session_id != self._playback_session_id:
                    return
                if self._playback_stream is not stream:
                    return

            try:
                stream_active = (not getattr(stream, "closed", True)) and bool(stream.active)
            except Exception:
                stream_active = False

            if not stream_active:
                self._reset_playback_state_if_session(session_id)
                return

            time.sleep(self._monitor_interval_sec)

    def start_audio_playback(self, file_path: str, device=None):
        if not file_path:
            return error_code.INVALID_PATH, "Missing audio file path."

        abs_path = os.path.abspath(file_path)
        if not os.path.isfile(abs_path):
            return error_code.INVALID_PATH, f"Audio file does not exist: {abs_path}"

        try:
            audio_data, sample_rate = self._load_playback_audio(abs_path)
        except Exception as e:
            return error_code.INVALID_FILE, f"Failed to decode audio: {str(e)[:80]}"

        if audio_data.size == 0:
            return error_code.INVALID_FILE, "Audio file is empty."

        output_max_channels = self._get_output_max_channels(device)
        try:
            playback_data = self._prepare_playback_audio(audio_data, output_max_channels=output_max_channels)
        except Exception as e:
            return error_code.INVALID_FILE, f"Failed to prepare playback audio: {str(e)[:80]}"

        with self._playback_lock:
            self._playback_session_id += 1
            session_id = self._playback_session_id
            self._playback_current_file = abs_path
            self._playback_is_running = True
            self._playback_stream = None

        stream = None
        try:
            sd.play(
                playback_data,
                samplerate=sample_rate,
                device=device,
                blocking=False,
            )
            stream = sd.get_stream()
        except Exception as e:
            if playback_data.ndim == 2 and playback_data.shape[1] > 1:
                try:
                    playback_data = self._downmix_playback_audio(playback_data, 1)
                    sd.play(
                        playback_data,
                        samplerate=sample_rate,
                        device=device,
                        blocking=False,
                    )
                    stream = sd.get_stream()
                except Exception as fallback_e:
                    self._reset_playback_state_if_session(session_id)
                    return error_code.INVALID_PLAY, f"Failed to start playback: {str(fallback_e)[:80]}"
            else:
                self._reset_playback_state_if_session(session_id)
                return error_code.INVALID_PLAY, f"Failed to start playback: {str(e)[:80]}"

        with self._playback_lock:
            if session_id != self._playback_session_id:
                self._close_stream_safely(stream)
                return error_code.INVALID_PLAY, "Playback session was replaced."
            self._playback_stream = stream

        threading.Thread(target=self._monitor_playback_done, args=(session_id, stream), daemon=True).start()

        return error_code.OK, "Audio playback started."

    def stop_audio_playback(self):
        with self._playback_lock:
            self._playback_session_id += 1
            was_running = bool(self._playback_is_running)
            stream = self._playback_stream
            self._playback_current_file = None
            self._playback_is_running = False
            self._playback_stream = None

        stop_error = None
        if stream is not None:
            try:
                if not getattr(stream, "closed", True) and bool(stream.active):
                    stream.stop(ignore_errors=True)
            except Exception as e:
                stop_error = e

            try:
                if not getattr(stream, "closed", True):
                    stream.close(ignore_errors=True)
            except Exception as e:
                if stop_error is None:
                    stop_error = e

        if stop_error is not None and was_running:
            return error_code.INVALID_PLAY, f"Failed to stop playback: {str(stop_error)[:80]}"

        if was_running:
            return error_code.OK, "Audio playback stopped."
        return error_code.INVALID_PLAY, "No active audio playback."

    def is_audio_playing(self) -> bool:
        with self._playback_lock:
            running = bool(self._playback_is_running)
            stream = self._playback_stream
            session_id = self._playback_session_id

        if not running:
            return False

        if stream is None:
            return True

        try:
            active = (not getattr(stream, "closed", True)) and bool(stream.active)
        except Exception:
            active = False

        if not active:
            self._reset_playback_state_if_session(session_id)
            return False

        return True

    def get_current_playing_file(self):
        with self._playback_lock:
            return self._playback_current_file
