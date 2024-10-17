from base.db_manager import DataSave
from consts import error_code, model_consts


class UsersManagement(object):
    def __init__(self):
        self.db_path = model_consts.DATABASE_PATH

    def create_user(self, register_user_info: dict):
        if not register_user_info:
            return error_code.INVALID_DATA_LOADING, "Missing registration information."
        try:
            with DataSave(self.db_path) as database:
                result = database.query_matching_data([(register_user_info["user_name"],)],
                                                      "users_table", ["user_name"],
                                                      ["user_id"])
                if result:
                    return error_code.INVALID_USER_REGISTER_INFO, "This user name already exists."
                user_info = tuple(register_user_info[key] for key in model_consts.DB_USERS_COLUMNS
                                  if key in register_user_info)
                insert_code, msg = database.insert_audio_files_info("users_table",
                                                                    model_consts.USERS_COLUMNS, [user_info])
                return insert_code, msg
        except Exception as e:
            err_msg = "Failed to create user. %s" % (str(e)[:40])
            return error_code.INVALID_INSERT, err_msg

    def delete_user(self, user_name: str):
        if not user_name or not isinstance(user_name, str):
            return error_code.INVALID_TYPE_DATA, "The model name is empty or invalid."
        delete_condition = {"user_name": user_name}
        try:
            with DataSave(self.db_path) as database:
                delete_code, msg = database.delete_with_condition("users_table", delete_condition)
            return delete_code, msg
        except Exception as e:
            err_msg = "The delete operation failed. %s" % (str(e)[:40])
            return error_code.INVALID_DELETE, err_msg

    def verify_login(self, user_name: str, password: str):
        if not password:
            return False
        with DataSave(self.db_path) as database:
            query_data = database.query_matching_data([(user_name,)], "users_table",
                                                          ["user_name"], ["password"])
        if query_data:
            (query_password,) = query_data[0]
            return query_password == password
        return False

    def reset_access_level(self, user_name, access_level: str):
        if access_level not in ['Engineer', 'Technician', 'Operator']:
            return error_code.INVALID_DATA_LOADING, "Invalid access_level data."
        update_data = {"access_level": access_level}
        try:
            with DataSave(self.db_path) as database:
                update_code, msg = database.update_audio_files_info("users_table", update_data,
                                                                    {"user_name": user_name},
                                                                    update_time=True)
                if update_code != error_code.OK:
                    return update_code, msg
                return error_code.OK, "Access level reset succeeded."
        except Exception as e:
            err_msg = "The delete operation failed. %s" % (str(e)[:40])
            return error_code.INVALID_UPDATE, err_msg

    def reset_password(self, user_name: str, new_password: str):
        try:
            with DataSave(self.db_path) as database:
                result = database.query_matching_data([(user_name,)], "users_table",
                                                      ["user_name"], ["password"])
                if not result:
                    return error_code.INVALID_RESET, "The information of the user is not found."
                new_password_data = {"password": new_password}
                update_code, msg = database.update_audio_files_info("users_table", new_password_data,
                                                 {"user_name": user_name}, update_time=True)
                if update_code != error_code.OK:
                    return update_code, msg
                return error_code.OK, "Password reset succeeded."
        except Exception as e:
            err_msg = "Failed to reset password. %s" % (str(e)[:40])
            return error_code.INVALID_RESET, err_msg

    def query_user_access_level(self, user_name: str):
        query_clause_data = {"user_name": user_name}
        try:
            with DataSave(self.db_path) as database:
                query_code, query_data = database.query("users_table", ["access_level"], query_clause_data)
            if query_code == error_code.OK:
                if not query_data:
                    return error_code.OK, None
                (access_level,) = query_data[0]
                return error_code.OK, access_level
        except Exception as e:
            err_msg = "Failed to query the user's access level. %s" % (str(e)[:40])
            return error_code.INVALID_QUERY, err_msg

    def query_user_info(self, user_name: str, query_column: str):
        if query_column not in model_consts.DB_USERS_COLUMNS:
            return error_code.INVALID_QUERY, "Invalid query column."
        query_clause_data = {"user_name": user_name}
        try:
            with DataSave(self.db_path) as database:
                query_code, query_data = database.query("users_table", [f"{query_column}"], query_clause_data)
            if query_code == error_code.OK:
                if query_data:
                    (query_column_data,) = query_data[0]
                    return error_code.OK, query_column_data
                else:
                    return error_code.INVALID_QUERY, "The user does not exist."
        except Exception as e:
            err_msg = "Failed to query the user's info. %s" % (str(e)[:40])
            return error_code.INVALID_QUERY, err_msg

    def query_user_list(self, access_level=None):
        query_clause_data = {}
        if access_level is not None:
            query_clause_data["access_level"] = access_level
        try:
            with DataSave(self.db_path) as database:
                query_code, query_data = database.query("users_table", model_consts.USERS_COLUMNS, query_clause_data)
                return query_code, query_data
        except Exception as e:
            err_msg = "Failed to query user list. %s" % (str(e)[:40])
            return error_code.INVALID_QUERY, err_msg
