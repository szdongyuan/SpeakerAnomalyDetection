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
