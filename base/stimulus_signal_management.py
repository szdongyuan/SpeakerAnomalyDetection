from base.db_manager import DataSave
from consts import model_consts, error_code


class StimulusSignalManagement(object):
    @staticmethod
    def update_stimulus_default(stimulus_id, is_default):
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
        try:
            with DataSave(model_consts.DATABASE_PATH) as database:
                stimulus_config = tuple(stimulus_info[key] for key in model_consts.STIMULUS_CONFIG_COLUMNS if key in stimulus_info)
                result = database.query_matching_data([stimulus_config], "stimulus_signal_table",
                                                      model_consts.STIMULUS_CONFIG_COLUMNS, ['stimulus_id'])
                name_result = database.query_matching_data([[stimulus_config[-1]]], "stimulus_signal_table",['stimulus_name'], ['stimulus_id'])
                if name_result: 
                    return error_code.INVALID_NAME, "This stimulus signals name info already exists."
                if not result:
                    is_default = database.set_default("stimulus_signal_table")
                    stimulus_info["is_default"] = is_default
                    insert_stimulus_config = tuple(stimulus_info[key] for key in model_consts.INERT_STIMULUS_CONFIG_COLUMNS if key in stimulus_info)
                    insert_stimulus_config = database.get_data_id([insert_stimulus_config], 0)
                    insert_code, msg = database.insert_data_into_db("stimulus_signal_table",
                                                                    model_consts.DB_STIMULUS_COLUMNS, insert_stimulus_config)
                    if insert_code == error_code.OK:
                        return error_code.OK, "Successfully saved stimulus signals to the database."
                    else:
                        return error_code.INVALID_INSERT, msg
                else:
                    return error_code.INVALID_INSERT, "This stimulus signals info already exists."
        except Exception as e:
            err_msg = "Failed to save stimulus signals to the database. %s" % (str(e)[:40])
            return error_code.INVALID_SAVE, err_msg
        
    @staticmethod
    def delete_stimulus_info_from_db(stimulus_name: str):
        delete_condition = {"stimulus_name": stimulus_name}
        try:
            with DataSave(model_consts.DATABASE_PATH) as database:
                delete_code, msg = database.delete_with_condition("stimulus_signal_table", delete_condition)
                return delete_code, msg
        except Exception as e:
            err_msg = "Delete stimulus info from database error: %s" % str(e)
            return error_code.INVALID_DELETE, err_msg
        
    @staticmethod
    def update_stimulus_info_to_db(stimulus_info: dict):
        try:
            with DataSave(model_consts.DATABASE_PATH) as database:
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
