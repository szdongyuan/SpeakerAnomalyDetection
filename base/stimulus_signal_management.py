from base.db_manager import DataSave
from consts import model_consts, error_code


class StimulusSignalManagement(object):
    @staticmethod
    def update_stimulus_default(stimulus_id, is_default):
        """
            Update the 'is_default' field of the stimulus signal with the given stimulus_id.
            If the stimulus is marked as the default, all other stimuli will have their 'is_default' field set to 0.

            Args:
                - stimulus_id: str
                    The ID of the stimulus to be updated.
                - is_default : int
                    The value to set for 'is_default' (1 for default, 0 for not default).

            Returns:
                - tuple: A tuple containing an error code and a message.

        """
        try:
            with DataSave(model_consts.DATABASE_PATH) as database:
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
        """
            Query the database for all default stimulus signal information, i.e., records where 'is_default' is 1.

            Returns:
                - tuple: A tuple containing an error code and a message or query data.

        """
        try:
            with DataSave(model_consts.DATABASE_PATH) as database:
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
        """
            Query the database for all stimulus signal information.

            Returns:
                - tuple: A tuple containing an error code and a message or query data.

        """
        try:
            with DataSave(model_consts.DATABASE_PATH) as database:
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
        """
            Save the provided stimulus information to the database if it doesn't already exist.

            Args:
                - stimulus_info: dict
                    Dictionary containing the stimulus signal information to be saved.

                Returns:
                    - tuple: A tuple containing an error code and a message or query data.

        """
        stimulus_config = tuple(stimulus_info[key] for key in model_consts.STIMULUS_COLUMNS if key in stimulus_info)
        try:
            with DataSave(model_consts.DATABASE_PATH) as database:
                is_default = database.set_default("stimulus_signal_table")
                stimulus_config += (is_default,)
                result = database.query_matching_data([stimulus_config], "stimulus_signal_table",
                                                      model_consts.STIMULUS_COLUMNS, ['stimulus_id'])
                if not result:
                    stimulus_config = database.get_data_id([stimulus_config], 0)
                    insert_code, msg = database.insert_data_into_db("stimulus_signal_table",
                                                                    model_consts.DB_STIMULUS_COLUMNS, stimulus_config)
                    if insert_code == error_code.OK:
                        return error_code.OK, "Successfully saved stimulus signals to the database."
                    else:
                        return error_code.INVALID_INSERT, msg
                else:
                    return error_code.INVALID_SAVE, "This stimulus signals info already exists."
        except Exception as e:
            err_msg = "Failed to save stimulus signals to the database. %s" % (str(e)[:40])
            return error_code.INVALID_SAVE, err_msg
