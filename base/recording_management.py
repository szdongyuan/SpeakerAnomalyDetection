import numpy as np
import shutil
import os
import uuid

from base.db_manager import DataSave
from base.save_data import save_audio_simple
from consts import error_code, model_consts, running_consts


class RecordingManager(object):
    def __init__(self):
        self.db_path = model_consts.DATABASE_PATH

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
            if not filename.endswith(".wav"):
                filename = os.path.splitext(filename)[0]
                filename += ".wav"
                audio_info["file_path"] = dir_path + "/" + filename
            if filename in os.listdir(dir_path):
                return error_code.INVALID_PATH, "The file already exists."
            save_audio_simple(audio_info["file_path"], audio_info["recorded_signal"], audio_info["sample_rate"])
            self.save_signal_info_to_db(audio_info, stimulus_parameter)
            return (
                error_code.OK,
                f"Recorded signal {filename} has been saved and its stimulus and recording information to database.",
            )
        except Exception as e:
            err_msg = "Failed to save the recorded signal file. %s" % (str(e)[:40])
            return error_code.INVALID_SAVE, err_msg

    def save_signal_info_to_db(self, audio_info: dict, stimulus_parameter: dict):
        try:
            with DataSave(self.db_path) as database:
                if stimulus_parameter:
                    stimulus_data, flag = self.get_stimulus_info_to_db(stimulus_parameter, database)
                else:
                    stimulus_data = None
                    flag = False
                audio_data = self.get_audio_info_to_db(audio_info, stimulus_data, database)
                ret_msg = "Successfully saved the recording signals to the database."
                if flag:
                    name = stimulus_data[0][1] + "-" + stimulus_data[0][2] + "-" + stimulus_data[0][0].split("-")[0]
                    stimulus_data[0] += (name,)
                    database.insert_data_into_db(
                        "stimulus_signal_table", model_consts.DB_STIMULUS_COLUMNS, stimulus_data
                    )
                    ret_msg = "Successfully saved the recording and stimulus signals to the database."
                database.insert_data_into_db("audio_data_table", model_consts.DB_AUDIO_COLUMNS, [audio_data])
                return error_code.OK, ret_msg
        except Exception as e:
            err_msg = "Failed to save the recording and stimulus signals to the database. %s" % (str(e)[:40])
            return error_code.INVALID_SAVE, err_msg

    @staticmethod
    def get_stimulus_info_to_db(stimulus_parameter: dict, database):
        flag = False
        normalized_parameter = stimulus_parameter.copy()
        normalized_parameter.setdefault("num_steps", None)
        normalized_parameter.setdefault("voltage_type", "RMS")
        normalized_parameter["voltage"] = float(normalized_parameter.get("voltage", 1.0))
        stimulus_data = tuple(normalized_parameter.get(key) for key in model_consts.STIMULUS_CONFIG_COLUMNS)
        result = database.query_matching_data(
            [stimulus_data],
            "stimulus_signal_table",
            model_consts.STIMULUS_CONFIG_COLUMNS,
            model_consts.DB_STIMULUS_COLUMNS,
        )
        if result:
            stimulus_data = result
        else:
            is_default = database.set_default("stimulus_signal_table")
            stimulus_data += (is_default,)
            flag = True
            stimulus_data = database.get_data_id([stimulus_data], 0)
        return stimulus_data, flag

    @staticmethod
    def get_audio_info_to_db(audio_info: dict, stimulus_data: list[tuple], database):
        audio_info["file_path"] = audio_info["file_path"].replace(running_consts.DEFAULT_DIR, "")
        audio_data = tuple(audio_info[key] for key in model_consts.AUDIO_COLUMNS if key in audio_info)
        if stimulus_data:
            audio_data = audio_data + (stimulus_data[0][0],)
        else:
            audio_data = audio_data + (None,)
        result = database.query_matching_data(
            [audio_data], "audio_data_table", model_consts.AUDIO_COLUMNS, ["audio_data_id"]
        )
        if result:
            audio_data_id = result[0][0]
        else:
            audio_data_id = str(uuid.uuid1())
        audio_data = (audio_data_id,) + audio_data
        return audio_data

    def update_audio_label(self, recorded_signal_info, old_file_path):
        try:
            update_data = {"file_path": recorded_signal_info["file_path"], "labels": recorded_signal_info["labels"]}
            condition_field = {"file_path": old_file_path}
            with DataSave(self.db_path) as database:
                database.update_table_data("audio_data_table", update_data, condition_field)
            return error_code.OK, "Database information updated."
        except Exception as e:
            err_msg = "The update operation failed. %s" % (str(e)[:40])
            return error_code.INVALID_RENAME, err_msg

    def rename_audio(self, file_path, new_name):
        try:
            if not os.path.exists(file_path):
                return error_code.INVALID_PATH, "The old path is invalid."
            dir_path = os.path.dirname(file_path)
            new_path = dir_path + "/" + new_name
            if not new_path.endswith(".wav"):
                new_path += ".wav"
            if os.path.exists(new_path):
                return error_code.INVALID_PATH, "The new file path already exists."
            os.rename(file_path, new_path)
            update_data = {"file_path": new_path}
            condition_field = {"file_path": file_path}
            with DataSave(self.db_path) as database:
                database.update_table_data("audio_data_table", update_data, condition_field)
            return error_code.OK, "The rename operation successful and the database information updated."
        except Exception as e:
            err_msg = "The rename operation failed. %s" % (str(e)[:40])
            return error_code.INVALID_RENAME, err_msg

    def move_audio(self, file_path, new_dir_path):
        try:
            if not os.path.isdir(new_dir_path):
                return error_code.INVALID_PATH, "The directory path is invalid."
            filename = os.path.basename(file_path)
            if filename in os.listdir(new_dir_path):
                return error_code.INVALID_MOVE, "The file with the same name already exists."
            shutil.move(file_path, new_dir_path)
            new_file_path = new_dir_path + "/" + filename
            update_data = {
                "new_data": {"file_path": new_file_path},
                "old_data": {"file_path": file_path},
            }
            with DataSave(self.db_path) as database:
                database.update_table_data("audio_data_table", update_data)
            return error_code.OK, f"The move operation succeeded."
        except Exception as e:
            err_msg = "The move operation failed. %s" % (str(e)[:40])
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
            err_msg = "The delete operation failed. %s" % (str(e)[:40])
            return error_code.INVALID_DELETE, err_msg

    def query_signal_info(self, file_path):
        try:
            if os.path.exists(file_path):
                query_clause_data = {"file_path": file_path}
                with DataSave(self.db_path) as database:
                    query_code, query_result = database.query(
                        "audio_data_table", model_consts.SELECT_COLUMNS, query_clause_data, FK_related=True
                    )
                    return query_code, query_result
            return error_code.INVALID_PATH, "The query file does not exist."
        except Exception as e:
            err_msg = "The query operation failed. %s" % (str(e)[:40])
            return error_code.INVALID_QUERY, err_msg

    def get_record_audio_data(self):
        try:
            with DataSave(self.db_path) as database:
                query_code, query_result = database.query("audio_data_table", model_consts.DB_AUDIO_COLUMNS)
                if query_code == error_code.OK and query_result:
                    return error_code.OK, query_result
                else:
                    return error_code.INVALID_QUERY, "Failed to query audio data."
        except Exception as e:
            err_msg = "The query operation failed. %s" % (str(e)[:40])
            return error_code.INVALID_QUERY, err_msg

    def delete_audio_at_id_list(self, id_list: list, progress_callback=None):
        if not id_list:
            return error_code.INVALID_DELETE, "Field list is empty, cannot perform delete operation."
        try:
            with DataSave(self.db_path) as database:
                return database.delete_by_fields("audio_data_table", "audio_data_id", id_list, progress_callback)
        except Exception as e:
            err_msg = "Failed to delete the file. Error: {}".format(e)
            return error_code.INVALID_DELETE, err_msg

    def query_stimulus_name_and_id(self):
        try:
            with DataSave(self.db_path) as database:
                code, result = database.query("stimulus_signal_table", ["stimulus_id", "stimulus_name"])
                if code == error_code.OK:
                    queru_result = dict()
                    for i in result:
                        queru_result[i[0]] = i[1]
                    return error_code.OK, queru_result
                else:
                    return error_code.INVALID_QUERY, "Failed to query the stimulus name."
        except Exception as e:
            err_msg = "Failed to query the stimulus name. Error:  %s" % (str(e)[:40])
            return error_code.INVALID_QUERY, err_msg
