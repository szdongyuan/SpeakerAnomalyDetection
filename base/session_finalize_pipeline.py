import os

from base.log_manager import LogManager
from base.play_and_record import get_recorded_info
from base.recording_management import RecordingManager
from base.save_data import save_audio_simple
from consts import error_code


class SessionFinalizePipeline(object):
    def __init__(self):
        self.default_logger = LogManager.set_log_handler("core")

    @staticmethod
    def apply_fixed_mic_channel_suffix(recorded_path, recorded_signal_info, session):
        selected_channel = getattr(session, "selected_channel", None)
        if not selected_channel:
            return recorded_path, recorded_signal_info

        base_path, ext = os.path.splitext(recorded_path)
        suffixed_path = "%s_ch%s%s" % (base_path, selected_channel, ext or "")
        updated_signal_info = recorded_signal_info.copy()
        updated_signal_info["file_path"] = suffixed_path
        updated_signal_info["selected_channel"] = selected_channel
        updated_signal_info["effective_channel_count"] = getattr(session, "effective_channel_count", None)
        return suffixed_path, updated_signal_info

    def save_fixed_mic_session(self, session, product_model, recorded_count):
        if session is None:
            return error_code.INVALID_DATA_LOADING, "session is None", None, None
        if session.audio_clip is None:
            return error_code.INVALID_DATA_LOADING, "session audio clip is empty", None, None

        recorded_path, recorded_signal_info = get_recorded_info(
            product_model,
            recorded_count,
            session.vehicle_barcode,
            "not_labeled",
        )
        recorded_path, recorded_signal_info = self.apply_fixed_mic_channel_suffix(
            recorded_path,
            recorded_signal_info,
            session,
        )
        sample_rate = session.metadata.get("sample_rate", 44100)

        try:
            save_audio_simple(recorded_path, session.audio_clip, sample_rate)
            recorded_signal_info["sample_rate"] = sample_rate
            save_code, save_msg = RecordingManager().save_signal_info_to_db(recorded_signal_info, None)
            if save_code != error_code.OK:
                return save_code, save_msg, recorded_path, recorded_signal_info

            session.metadata["recorded_path"] = recorded_path
            session.metadata["recorded_signal_info"] = recorded_signal_info.copy()
            return error_code.OK, "fixed mic session saved", recorded_path, recorded_signal_info
        except Exception as e:
            err_msg = "Failed to finalize fixed mic session. %s" % (str(e)[:60])
            self.default_logger.error(err_msg)
            return error_code.INVALID_SAVE, err_msg, recorded_path, recorded_signal_info
