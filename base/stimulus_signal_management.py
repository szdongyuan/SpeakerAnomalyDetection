import json
import os

from base.db_manager import DataSave, ensure_audio_database_ready
from consts import model_consts, error_code, running_consts


class StimulusSignalManagement(object):
    @staticmethod
    def update_stimulus_default(stimulus_id, is_default):
        try:
            ensure_audio_database_ready()
            with DataSave(model_consts.AUDIO_DATABASE_PATH) as database:
                query_code, query_data = database.query("stimulus_signal_table", ["is_default"],
                                                        {"stimulus_id": stimulus_id})
                if query_code != error_code.OK or not query_data:
                    return error_code.INVALID_QUERY, f"Stimulus ID {stimulus_id} not found or query failed."
                (current_default,) = query_data[0]
                if current_default == is_default:
                    return error_code.OK, "Stimulus default settings are updated successfully."
                else:
                    update_code, _ = database.update_table_data("stimulus_signal_table",
                                                                {"is_default": is_default},
                                                                {"stimulus_id": stimulus_id})
                    if update_code != error_code.OK:
                        return error_code.INVALID_UPDATE, f"Failed to update stimulus_id {stimulus_id}."
                    update_other_code, _ = database.update_table_data("stimulus_signal_table",
                                                                      {"is_default": 0},
                                                                      {"stimulus_id": {"!=": stimulus_id}})
                    if update_other_code != error_code.OK:
                        return error_code.INVALID_UPDATE, f"Failed to reset other records' is_default."
                    return error_code.OK, "Stimulus default settings are updated successfully."
        except Exception as e:
            err_msg = "Failed to update the stimulus default settings. %s" % (str(e)[:40])
            return error_code.INVALID_UPDATE, err_msg

    @staticmethod
    def query_default_stimulus_info():
        try:
            ensure_audio_database_ready()
            with DataSave(model_consts.AUDIO_DATABASE_PATH) as database:
                query_code, query_data = database.query("stimulus_signal_table", model_consts.DB_STIMULUS_COLUMNS,
                                                        {"is_default": 1})
            if query_code == error_code.OK and query_data:
                return error_code.OK, query_data
            else:
                return error_code.INVALID_QUERY, "Failed to query the default stimulus signal settings."
        except Exception as e:
            err_msg = "Failed to query the default stimulus signal. %s" % (str(e)[:40])
            return error_code.INVALID_QUERY, err_msg

    @staticmethod
    def query_all_stimulus_info():
        try:
            ensure_audio_database_ready()
            with DataSave(model_consts.AUDIO_DATABASE_PATH) as database:
                query_code, query_data = database.query("stimulus_signal_table", model_consts.DB_STIMULUS_COLUMNS)
            if query_code == error_code.OK and query_data:
                return error_code.OK, query_data
            else:
                return error_code.INVALID_QUERY, "Failed to query stimulus signal info or no stimulus signal info."
        except Exception as e:
            err_msg = "Failed to query stimulus signal. %s" % (str(e)[:40])
            return error_code.INVALID_QUERY, err_msg

    @staticmethod
    def save_stimulus_info_to_db(stimulus_info: dict):
        try:
            ensure_audio_database_ready()
            with DataSave(model_consts.AUDIO_DATABASE_PATH) as database:
                normalized_info = stimulus_info.copy()
                normalized_info.setdefault('num_steps', None)
                normalized_info.setdefault('voltage_type', 'RMS')
                normalized_info['voltage'] = float(normalized_info.get("voltage", 1.0))
                stimulus_info.setdefault('num_steps', normalized_info['num_steps'])
                stimulus_info.setdefault('voltage_type', normalized_info['voltage_type'])
                stimulus_info['voltage'] = normalized_info['voltage']

                name_result = database.query_matching_data(
                    [[stimulus_info.get("stimulus_name")]],
                    "stimulus_signal_table",
                    ['stimulus_name'],
                    ['stimulus_id']
                )
                if name_result: 
                    return error_code.INVALID_NAME, "This stimulus signals name info already exists."

                is_default = database.set_default("stimulus_signal_table")
                stimulus_info["is_default"] = is_default
                normalized_info["is_default"] = is_default
                insert_stimulus_config = tuple(normalized_info.get(key) for key in model_consts.INERT_STIMULUS_CONFIG_COLUMNS)
                insert_stimulus_config = database.get_data_id([insert_stimulus_config], 0)
                insert_code, msg = database.insert_data_into_db("stimulus_signal_table",
                                                                model_consts.DB_STIMULUS_COLUMNS, insert_stimulus_config)
                if insert_code == error_code.OK:
                    return error_code.OK, "Successfully saved stimulus signals to the database."
                else:
                    return error_code.INVALID_INSERT, msg

        except Exception as e:
            err_msg = "Failed to save stimulus signals to the database. %s" % (str(e)[:40])
            return error_code.INVALID_SAVE, err_msg
        
    @staticmethod
    def delete_stimulus_info_from_db(stimulus_name: str):
        delete_condition = {"stimulus_name": stimulus_name}
        try:
            ensure_audio_database_ready()
            with DataSave(model_consts.AUDIO_DATABASE_PATH) as database:
                delete_code, msg = database.delete_with_condition("stimulus_signal_table", delete_condition)
                return delete_code, msg
        except Exception as e:
            err_msg = "Delete stimulus info from database error: %s" % str(e)
            return error_code.INVALID_DELETE, err_msg
        
    @staticmethod
    def update_stimulus_info_to_db(stimulus_info: dict):
        try:
            ensure_audio_database_ready()
            with DataSave(model_consts.AUDIO_DATABASE_PATH) as database:
                result = database.query_matching_data([(stimulus_info.get("stimulus_id"),)], "stimulus_signal_table", ["stimulus_id"],
                                                      ['stimulus_id'])
                if result:
                    update_data = {"stimulus_name": stimulus_info["new_name"]}
                    condition_field = {"stimulus_id": stimulus_info["stimulus_id"]}
                    database.update_table_data("stimulus_signal_table", update_data, condition_field)
                    return error_code.OK, "The stimulus info has been updated."
                else:
                    return error_code.INVALID_UPDATE, "The stimulus name does not exist."
        except Exception as e:
            err_msg = "Failed to update the stimulus info to the database. %s" % str(e)
            return error_code.INVALID_UPDATE, err_msg

    @staticmethod
    def update_stimulus_params_to_db(stimulus_id: int, update_params: dict):
        """
        Update editable stimulus parameters for a given stimulus_id.

        Only allows updating of configurable numeric parameters consistent with UI constraints.
        The following fields are supported (all optional):
        - repeat_times (int)
        - start_freq (int)
        - stop_freq (int)
        - sample_rate (int)
        - total_time (float)
        - num_steps (int)
        - voltage (float)

        Args:
            stimulus_id: int, primary key in stimulus_signal_table
            update_params: dict, subset of supported fields to update

        Returns:
            (code, message) tuple
        """
        if not isinstance(update_params, dict) or not update_params:
            return error_code.INVALID_UPDATE, "No parameters provided to update."
        # Whitelist supported fields to avoid accidental updates
        allowed_fields = {
            "repeat_times",
            "start_freq",
            "stop_freq",
            "sample_rate",
            "total_time",
            "num_steps",
            "voltage",
        }
        # Filter update_params by allowed fields
        update_data = {k: v for k, v in update_params.items() if k in allowed_fields}
        if not update_data:
            return error_code.INVALID_UPDATE, "No valid parameters to update."
        try:
            ensure_audio_database_ready()
            with DataSave(model_consts.AUDIO_DATABASE_PATH) as database:
                # Ensure record exists
                result = database.query_matching_data(
                    [(stimulus_id,)], "stimulus_signal_table", ["stimulus_id"], ["stimulus_id"]
                )
                if not result:
                    return error_code.INVALID_UPDATE, "Stimulus ID does not exist."
                update_code, msg = database.update_table_data(
                    "stimulus_signal_table", update_data, {"stimulus_id": stimulus_id}
                )
                if update_code == error_code.OK:
                    return error_code.OK, "Stimulus parameters updated."
                else:
                    return error_code.INVALID_UPDATE, msg
        except Exception as e:
            err_msg = "Failed to update the stimulus parameters. %s" % str(e)
            return error_code.INVALID_UPDATE, err_msg

