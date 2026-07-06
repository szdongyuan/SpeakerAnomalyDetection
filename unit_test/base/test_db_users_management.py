import mock
import pytest

from base.db_users_management import UsersManagement
from consts import error_code


@pytest.fixture(autouse=True)
def bypass_system_database_ready():
    with mock.patch("base.db_users_management.ensure_system_database_ready"):
        yield


class TestUsersManagement(object):
    @pytest.mark.parametrize("database_ret, query_match_set, insert_set, register_user_info, result_set", [
        (mock.Mock(), [], [], {}, (error_code.INVALID_DATA_LOADING, "Missing registration information.")),
        (mock.Mock(), [(3,)], [],
         {"user_name": 'zz', "password": 'zz123', "access_level": 'Operator'},
         (error_code.INVALID_USER_REGISTER_INFO, "This user name already exists.")
         ),
        (mock.Mock(), [], [(error_code.OK, "Insert data successfully.")],
         {"user_name": 'bb', "password": 'zz123', "access_level": 'Operator'},
         (error_code.OK, "Insert data successfully.")
         ),
        (mock.Mock(), [], Exception('xxx'),
         {"user_name": 'cc', "password": 'zz123', "access_level": 'Operator'},
         (error_code.INVALID_INSERT, "Failed to create user. xxx")
         )
    ])
    @mock.patch("base.db_users_management.DataSave")
    def test_create_user(self, mock_database, database_ret, query_match_set, insert_set, register_user_info, result_set):
        mock_database.return_value.__enter__.return_value = database_ret
        mock_database.return_value.__enter__.return_value.query_matching_data.return_value = query_match_set
        mock_database.return_value.__enter__.return_value.insert_data_into_db.side_effect = insert_set
        result = UsersManagement().create_user(register_user_info)
        assert result == result_set

    @pytest.mark.parametrize("database_set, delete_set, user_name, result_ret", [
        (mock.Mock(), [], "", (error_code.INVALID_TYPE_DATA, "The model name is empty or invalid.")),
        (mock.Mock(), [], 123, (error_code.INVALID_TYPE_DATA, "The model name is empty or invalid.")),
        (mock.Mock(), [(error_code.OK, "Delete the data that meets the condition.")], "zz",
         (error_code.OK, "Delete the data that meets the condition.")),
        (mock.Mock(), [(error_code.INVALID_DELETE, "No data matched the condition. No data was deleted.")], "bb",
         (error_code.INVALID_DELETE, "No data matched the condition. No data was deleted.")),
        (mock.Mock(), Exception('xxx'), "zz",
         (error_code.INVALID_DELETE, "The delete operation failed. xxx")),
    ])
    @mock.patch("base.db_users_management.DataSave")
    def test_delete_user(self, mock_database, database_set, delete_set, user_name, result_ret):
        mock_database.return_value.__enter__.return_value = database_set
        mock_database.return_value.__enter__.return_value.delete_with_condition.side_effect = delete_set
        result = UsersManagement().delete_user(user_name)
        assert result == result_ret

    @pytest.mark.parametrize("database_set, query_match_set, user_name, password, result_ret", [
        (mock.Mock(), mock.Mock(), "zz", "", False),
        (mock.Mock(), [('123',)], "zz", "123", True),
        (mock.Mock(), [('123',)], "zz", "zz123", False),
        (mock.Mock(), [], "bb", "123", False),
    ])
    @mock.patch("base.db_users_management.DataSave")
    def test_verify_login(self, mock_database, database_set, query_match_set, user_name, password, result_ret):
        mock_database.return_value.__enter__.return_value = database_set
        mock_database.return_value.__enter__.return_value.query_matching_data.return_value = query_match_set
        result = UsersManagement().verify_login(user_name, password)
        assert result == result_ret

    @pytest.mark.parametrize("database_set, update_set, user_name, access_level, result_ret", [
        (mock.Mock(), [mock.Mock(), mock.Mock()], "zz", "op",
         (error_code.INVALID_DATA_LOADING, "Invalid access_level data."),
         ),
        (mock.Mock(), [(error_code.OK, "Update data successfully.")], "zz", "Technician",
         (error_code.OK, "Access level reset succeeded."),
         ),
        (mock.Mock(), Exception('xxx'), "zz", "Technician",
         (error_code.INVALID_UPDATE, "The delete operation failed. xxx"),
         ),
    ])
    @mock.patch("base.db_users_management.DataSave")
    def test_reset_access_level(self, mock_database, database_set, update_set, user_name, access_level, result_ret):
        mock_database.return_value.__enter__.return_value = database_set
        mock_database.return_value.__enter__.return_value.update_table_data.side_effect = update_set
        result = UsersManagement().reset_access_level(user_name, access_level)
        assert result == result_ret

    @pytest.mark.parametrize("input_set, result_ret", [
            ({"database_set": mock.Mock(),
              "query_match_set": [],
              "update_set": [],
              "user_name": "bb",
              "new_password": "123"},
             (error_code.INVALID_RESET, "The information of the user is not found."),
             ),
            ({"database_set": mock.Mock(),
              "query_match_set": [('123',)],
              "update_set": [(error_code.OK, "Update data successfully.")],
              "user_name": "zz",
              "new_password": "zz123"},
             (error_code.OK, "Password reset succeeded.")
             ),
            ({"database_set": mock.Mock(),
              "query_match_set": mock.Mock(),
              "update_set": Exception('xxx'),
              "user_name": "zz",
              "new_password": "zz123"},
             (error_code.INVALID_RESET, "Failed to reset password. xxx"),
             ),
        ])
    @mock.patch("base.db_users_management.DataSave")
    def test_reset_password(self, mock_database, input_set, result_ret):
        mock_database.return_value.__enter__.return_value = input_set["database_set"]
        mock_database.return_value.__enter__.return_value.query_matching_data.return_value = input_set["query_match_set"]
        mock_database.return_value.__enter__.return_value.update_table_data.side_effect = input_set["update_set"]
        result = UsersManagement().reset_password(input_set["user_name"], input_set["new_password"])
        assert result == result_ret

    @pytest.mark.parametrize("database_set, query_set, user_name, result_set", [
        (mock.Mock(), [(error_code.OK, [('Operator',)])], "zz", (error_code.OK, "Operator")),
        (mock.Mock(), [(error_code.OK, [])], "bb", (error_code.OK, None)),
        (mock.Mock(), Exception('xxx'), "zz",
         (error_code.INVALID_QUERY, "Failed to query the user's access level. xxx")),
    ])
    @mock.patch("base.db_users_management.DataSave")
    def test_query_user_access_level(self, mock_database, database_set, query_set, user_name, result_set):
        mock_database.return_value.__enter__.return_value = database_set
        mock_database.return_value.__enter__.return_value.query.side_effect = query_set
        result = UsersManagement().query_user_access_level(user_name)
        assert result == result_set

    @pytest.mark.parametrize("database_set, query_set, user_name, query_column, result_ret", [
        (mock.Mock(), [], "zz", "user_status",
         (error_code.INVALID_QUERY, "Invalid query column.")
         ),
        (mock.Mock(), [(error_code.OK, [(3,)])], "zz", "user_id",
         (error_code.OK, 3)
         ),
        (mock.Mock(), [(error_code.OK, [])], "bb", "user_id",
         (error_code.INVALID_QUERY, "The user does not exist."),
         ),
        (mock.Mock(), Exception('xxx'), "bb", "user_id",
         (error_code.INVALID_QUERY, "Failed to query the user's info. xxx")
         ),
    ])
    @mock.patch("base.db_users_management.DataSave")
    def test_query_user_info(self, mock_database, database_set, query_set, user_name, query_column, result_ret):
        mock_database.return_value.__enter__.return_value = database_set
        mock_database.return_value.__enter__.return_value.query.side_effect = query_set
        result = UsersManagement().query_user_info(user_name, query_column)
        assert result == result_ret

    @pytest.mark.parametrize("database_set, query_set, access_level_set, result_set", [
        (mock.Mock(), [(error_code.OK, [('wlz', 'wlz123', 'Technician'), ('zz', 'zz123', 'Operator')])],
         None,
         (error_code.OK, [('wlz', 'wlz123', 'Technician'), ('zz', 'zz123', 'Operator')])
         ),
        (mock.Mock(), [(error_code.OK, [('zz', 'zz123', 'Operator')])],
         "Operator",
         (error_code.OK, [('zz', 'zz123', 'Operator')])
         ),
        (mock.Mock(), Exception('xxx'),
         None,
         (error_code.INVALID_QUERY, "Failed to query user list. xxx")
         )
    ])
    @mock.patch("base.db_users_management.DataSave")
    def test_query_user_list(self, mock_database, database_set, query_set, access_level_set, result_set):
        mock_database.return_value.__enter__.return_value = database_set
        mock_database.return_value.__enter__.return_value.query.side_effect = query_set
        result = UsersManagement().query_user_list(access_level_set)
        assert result == result_set
