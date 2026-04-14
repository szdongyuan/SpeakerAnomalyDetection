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

        with self._playback_lock:
            self._playback_session_id += 1
            session_id = self._playback_session_id
            self._playback_current_file = abs_path
            self._playback_is_running = True
            self._playback_stream = None

        stream = None
        try:
            sd.play(
                audio_data,
                samplerate=sample_rate,
                device=device,
                blocking=False,
            )
            stream = sd.get_stream()
        except Exception as e:
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
