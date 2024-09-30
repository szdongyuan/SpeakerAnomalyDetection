import numpy as np
import shutil
import os
import uuid
from scipy.io import wavfile

from base.db_manager import DataSave
from consts import error_code, model_consts


class RecordingManager(object):
    def __init__(self):
        DEFAULT_DIR = os.path.split(os.path.realpath(__file__))[0].replace("\\", "/") + "/../"
        self.db_path = DEFAULT_DIR + model_consts.DATABASE_PATH

    def save_recording_to_wav(self, audio_info: dict, stimulus_parameter: dict):
        try:
            if not audio_info["file_path"]:
                return error_code.INVALID_PATH, "missing file path."
            dir_path = os.path.dirname(audio_info["file_path"])
            filename = os.path.basename(audio_info["file_path"])
            if not isinstance(audio_info["recorded_signal"], np.ndarray):
                return error_code.INVALID_TYPE_DATA, "invalid recorded signal data."
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            if not filename.endswith('.wav'):
                filename = os.path.splitext(filename)[0]
                filename += '.wav'
                audio_info["file_path"] = dir_path + '/' + filename
            if filename in os.listdir(dir_path):
                return error_code.INVALID_PATH, "The file already exists."
            wavfile.write(audio_info["file_path"], audio_info["sample_rate"],
                          audio_info["recorded_signal"].astype(np.int16))
            self.save_signal_info_to_db(audio_info, stimulus_parameter)
            return (error_code.OK,
                    f"Recorded signal {filename} has been saved and its stimulus and recording information to database.")
        except Exception as e:
            err_msg = "Failed to save the recorded signal file. %s" % (str(e))
            return error_code.INVALID_SAVE, err_msg

    def save_signal_info_to_db(self, audio_info: dict, stimulus_parameter: dict):
        try:
            with DataSave(self.db_path) as database:
                stimulus_data, flag = self.get_stimulus_info_to_db(stimulus_parameter, database)
                audio_data = self.get_audio_info_to_db(audio_info, stimulus_data, database)
                database.insert_audio_files_info('audio_data_table', model_consts.AUDIO_DATA_TABLE_COLUMNS,
                                                 [audio_data])
                if flag:
                    database.insert_audio_files_info('stimulus_signal_table',
                                                     model_consts.STIMULUS_SIGNAL_TABLE_COLUMNS, stimulus_data)
                    return error_code.OK, "Successfully saved the recording and stimulus signals to the database."
                else:
                    return error_code.OK, "Successfully saved the recording signals to the database."
        except Exception as e:
            err_msg = "Failed to save the recording and stimulus signals to the database. %s" % (str(e))
            return error_code.INVALID_SAVE, err_msg

    @staticmethod
    def get_stimulus_info_to_db(stimulus_parameter: dict, database):
        flag = False
        stimulus_data = tuple(
            stimulus_parameter[key] for key in model_consts.STIMULUS_COLUMNS if key in stimulus_parameter)
        result = database.check_database_info_equal([stimulus_data], "stimulus_signal_table",
                                                    model_consts.STIMULUS_COLUMNS, model_consts.DB_STIMULUS_COLUMNS)
        if result:
            stimulus_data = result
        else:
            flag = True
            stimulus_data = database.get_data_id([stimulus_data], 0)
        return stimulus_data, flag

    @staticmethod
    def get_audio_info_to_db(audio_info: dict, stimulus_data, database):
        audio_data = tuple(audio_info[key] for key in model_consts.AUDIO_COLUMNS if key in audio_info)
        audio_data = audio_data + (stimulus_data[0][0],)
        result = database.check_database_info_equal([audio_data], "audio_data_table", model_consts.AUDIO_COLUMNS,
                                                    ['audio_data_id'])
        if result:
            audio_data_id = result[0][0]
        else:
            audio_data_id = str(uuid.uuid1())
        audio_data = (audio_data_id,) + audio_data
        return audio_data

    def rename_audio(self, file_path, new_name):
        try:
            if not os.path.exists(file_path):
                return error_code.INVALID_PATH, "The old path is invalid."
            dir_path = os.path.dirname(file_path)
            new_path = dir_path + '/' + new_name
            if not new_path.endswith('.wav'):
                new_path += '.wav'
            if os.path.exists(new_path):
                return error_code.INVALID_PATH, "The new file path already exists."
            os.rename(file_path, new_path)
            update_data = {
                "new_data": {"file_path": new_path},
                "old_data": {"file_path": file_path},
            }
            with DataSave(self.db_path) as database:
                database.update_audio_files_info("audio_data_table", update_data)
            return error_code.OK, "The rename operation successful and the database information updated."
        except Exception as e:
            err_msg = "The rename operation failed. %s" % (str(e))
            return error_code.INVALID_RENAME, err_msg

    def move_audio(self, file_path, new_dir_path):
        try:
            if not os.path.isdir(new_dir_path):
                return error_code.INVALID_PATH, "The directory path is invalid."
            filename = os.path.basename(file_path)
            if filename in os.listdir(new_dir_path):
                return error_code.INVALID_MOVE, "The file with the same name already exists."
            shutil.move(file_path, new_dir_path)
            new_file_path = new_dir_path + '/' + filename
            update_data = {
                "new_data": {"file_path": new_file_path},
                "old_data": {"file_path": file_path},
            }
            with DataSave(self.db_path) as database:
                database.update_audio_files_info("audio_data_table", update_data)
            return error_code.OK, f"The move operation succeeded."
        except Exception as e:
            err_msg = "The move operation failed. %s" % (str(e))
            return error_code.INVALID_MOVE, err_msg

    def delete_audio(self, file_path):
        try:
            if not os.path.exists(file_path):
                return error_code.INVALID_PATH, "The file does not exist."
            os.remove(file_path)
            delete_condition = {"file_path": file_path}
            with DataSave(self.db_path) as database:
                database.delete_with_condition("audio_data_table", delete_condition)
            return error_code.OK, "The file is deleted successfully."
        except Exception as e:
            err_msg = "The delete operation failed. %s" % (str(e))
            return error_code.INVALID_DELETE, err_msg

    def query_signal_info(self, file_path):
        try:
            if os.path.exists(file_path):
                query_clause_data = {"file_path": file_path}
                with DataSave(self.db_path) as database:
                    query_result, msg = database.query("audio_data_table", model_consts.SELECT_COLUMNS,
                                                     query_clause_data, FK_related=True)
                return query_result, "Query success."
            return error_code.INVALID_PATH, "The query file does not exist."
        except Exception as e:
            err_msg = "The query operation failed. %s" % (str(e))
            return error_code.INVALID_QUERY, err_msg
