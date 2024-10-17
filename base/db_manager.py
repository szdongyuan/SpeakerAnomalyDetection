import os
import sqlite3
import uuid
import wave
from datetime import datetime

from base.load_config import load_config
from base.log_manager import LogManager
from consts import model_consts, error_code


class DataSave(object):
    def __init__(self, db_name):
        self.db_name = db_name
        self.connection = None
        self.cursor = None
        self.logger = LogManager("db_core")

    def connect(self):
        try:
            self.connection = sqlite3.connect(self.db_name)
            self.cursor = self.connection.cursor()
            self.logger.info("Successfully connect to database.")
            return error_code.OK, "Successfully connect to database."
        except Exception as e:
            err_msg = "Failed to connect to the database %s" % (str(e)[:40])
            self.logger.error(err_msg)
            return error_code.INVALID_CONNECT_DATABASE, err_msg

    def create_table(self):
        try:
            self.connection = sqlite3.connect(self.db_name)
            self.cursor = self.connection.cursor()
            self.logger.info("...Creating tables...")
            create_audio_data_table_sql = '''
            CREATE TABLE IF NOT EXISTS audio_data_table(
                audio_data_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL UNIQUE,
                product_model TEXT NOT NULL,
                sample_rate INTEGER NOT NULL CHECK (sample_rate > 0),
                record_date DATETIME NOT NULL,
                labels BLOB,
                stimulus_id TEXT,
                FOREIGN KEY (stimulus_id) REFERENCES stimulus_signal_table (stimulus_id) ON DELETE NO ACTION ON UPDATE NO ACTION   
            );
            '''
            create_stimulus_signal_table_sql = '''
            CREATE TABLE IF NOT EXISTS stimulus_signal_table(
                stimulus_id TEXT PRIMARY KEY,
                sweep_method TEXT NOT NULL, 
                sweep_type TEXT NOT NULL, 
                repeats INTEGER NOT NULL CHECK (repeats > 0), 
                start_feq INTEGER NOT NULL CHECK (start_feq >= 0), 
                end_feq INTEGER NOT NULL CHECK (end_feq > start_feq), 
                sample_rate INTEGER NOT NULL CHECK (sample_rate > 0), 
                sweep_duration INTEGER NOT NULL CHECK (sweep_duration > 0)
            );
           '''
            create_training_model_table_sql = '''
            CREATE TABLE IF NOT EXISTS training_model_table(
                model_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL UNIQUE,
                model_path TEXT NOT NULL UNIQUE,
                config_path TEXT NOT NULL,
                input_dim TEXT NOT NULL,   
                output_dim INTEGER NOT NULL, 
                accuracy REAL CHECK (accuracy >= 0 and accuracy <= 1),
                update_date DATETIME NOT NULL,
                model_description TEXT 
            );
            '''
            self.cursor.execute(create_audio_data_table_sql)
            self.cursor.execute(create_stimulus_signal_table_sql)
            self.cursor.execute(create_training_model_table_sql)
            self.connection.commit()
            self.logger.info("Table creation success.")
            return error_code.OK, "Table creation success."
        except Exception as e:
            err_msg = "Failed to create table. %s" % (str(e)[:40])
            self.logger.error(err_msg)
            return error_code.INVALID_CREATE_TABLE, err_msg

    def get_audio_data_list(self, source_dir_list, label):
        data_list = []
        n_file = 0
        for source_dir in source_dir_list:
            source_dir_path = os.path.join(model_consts.STORED_SAMPLE_PATH, source_dir).replace("\\", "/")
            sub_folder_path = source_dir_path + "/" + label
            source_dir_str = source_dir.split("/")
            if not os.path.exists(sub_folder_path):
                continue
            for index, audio_file in enumerate(os.listdir(sub_folder_path)):
                audio_data_id = str(uuid.uuid1())
                file_path = os.path.join(source_dir, label, audio_file).replace("\\", "/")
                product_model = source_dir_str[2].split("_")[0]
                sample_rate = model_consts.SAMPLE_RATE
                record_date = (datetime.strptime(source_dir_str[3], "%Y%m%d")).strftime("%Y-%m-%d")
                sample_stimulus_data = self.get_audio_data_stimulus_info(sub_folder_path + "/" + audio_file)
                result = self.check_database_info_equal([sample_stimulus_data], "stimulus_signal_table",
                                                        model_consts.STIMULUS_COLUMNS,
                                                        ['stimulus_signal_table.stimulus_id'])
                stimulus_id = result[0][0] if result else None
                sample_data = (audio_data_id, file_path, product_model, sample_rate, record_date, label, stimulus_id)
                data_list.append(sample_data)
                n_file += 1
        self.logger.info(f"{n_file} audio samples were successfully inserted.")
        return data_list

    def stimulus_signal_file_list(self, source_dir_list, label):
        stimulus_data = []
        for source_dir in source_dir_list:
            source_dir_path = os.path.join(model_consts.STORED_SAMPLE_PATH, source_dir).replace("\\", "/")
            sub_folder_path = source_dir_path + "/" + label
            if not os.path.exists(sub_folder_path):
                continue
            for audio_file in os.listdir(sub_folder_path):
                sample_stimulus_data = self.get_audio_data_stimulus_info(sub_folder_path + "/" + audio_file)
                stimulus_data.append(sample_stimulus_data)
        stimulus_data = list(set(stimulus_data))
        result = self.check_database_info_equal(stimulus_data, "stimulus_signal_table", model_consts.STIMULUS_COLUMNS,
                                                model_consts.STIMULUS_COLUMNS)
        stimulus_data_list = [item for item in stimulus_data if item not in set(result)]
        return self.get_data_id(stimulus_data_list, 0)

    def check_database_info_equal(self, data_list, table_name, check_column, select_column):
        result = []
        base_sql = ' AND '.join([f"{column} = ?" for column in check_column])
        for data_item in data_list:
            sql_select = f"SELECT {', '.join(select_column)} FROM {table_name} WHERE {base_sql}"
            self.cursor.execute(sql_select, data_item)
            fet_result = self.cursor.fetchall()
            if fet_result:
                result.extend(row for row in fet_result)
        return result

    def get_audio_data_stimulus_info(self, file_path):
        audio_stimulus_data = ()
        if file_path:
            relpath = os.path.relpath(file_path, model_consts.DEFAULT_DIR).replace("\\", "/")
            relpath_str = relpath.split("/")
            sweep_type = relpath_str[3].split("_")[0]
            sweep_method = relpath_str[3].split("_")[1]
            repeats = relpath_str[3].split("_")[2]
            start_feq = relpath_str[4].split("_")[1]
            end_feq = relpath_str[4].split("_")[2]
            sample_rate = model_consts.SAMPLE_RATE
            sweep_duration = self.get_wav_duration(file_path)
            audio_stimulus_data = (
                sweep_method, sweep_type, int(repeats), int(start_feq), int(end_feq), sample_rate, sweep_duration)
        return audio_stimulus_data

    @staticmethod
    def get_data_id(data_list, id_index: int):
        for i, item in enumerate(data_list):
            data_id = str(uuid.uuid1())
            data_list[i] = item[:id_index] + (data_id,) + item[id_index:]
        return data_list

    @staticmethod
    def get_wav_duration(filepath):
        with wave.open(filepath, 'r') as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            duration = frames / float(rate)
        return int(duration)

    def insert_audio_files_info(self, table_name, columns, data):
        try:
            if len(data) == 0:
                self.logger.info("data empty.")
                return error_code.OK, "data empty."
            values_num = ','.join(['?'] * len(data[0]))
            sql = f'INSERT INTO {table_name} ({columns}) VALUES ({values_num});'
            self.cursor.executemany(sql, data)
            self.connection.commit()
            self.logger.info("Insert data successfully.")
            return error_code.OK, "Insert data successfully."
        except Exception as e:
            err_msg = "Failed to insert data into the database. %s" % (str(e))
            self.logger.error(err_msg)
            return error_code.INVALID_INSERT, err_msg

    def update_audio_files_info(self, table_name, update_data: dict):
        try:
            if not isinstance(update_data["new_data"], dict) or not isinstance(update_data["old_data"], dict):
                return error_code.INVALID_TYPE_DATA, "The update_data format is incorrect."
            if not update_data["new_data"] or not update_data["old_data"]:
                return error_code.INVALID_TYPE_DATA, "new_data and old_data must contain at least one key."
            set_clause = ', '.join([f"{key} = ?" for key in update_data["new_data"].keys()])
            where_clause = ' AND '.join([f"{key} = ?" for key in update_data["old_data"].keys()])
            sql = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
            self.cursor.execute(sql, list(update_data["new_data"].values()) +
                                list(update_data["old_data"].values()))
            self.connection.commit()
            self.logger.info("Update data successfully.")
            return error_code.OK, "Update data successfully."
        except Exception as e:
            err_msg = "Failed to update data into the database. %s" % (str(e))
            self.logger.error(err_msg)
            return error_code.INVALID_UPDATE, err_msg

    def query(self, table_name, query_column, query_clause_data: dict, FK_related: bool):
        try:
            join_sql = ''
            if query_clause_data:
                where_clause = ' AND '.join([f"{key} = ?" for key in query_clause_data.keys()])
                if FK_related:
                    join_sql = (' INNER JOIN stimulus_signal_table ON audio_data_table.stimulus_id = '
                                'stimulus_signal_table.stimulus_id')
                sql_query = f"SELECT {query_column} FROM {table_name} {join_sql} WHERE {where_clause};"
                self.cursor.execute(sql_query, list(query_clause_data.values()))
                query_data = self.cursor.fetchall()
                self.logger.info("The conditional query succeeds.")
                return query_data, "The conditional query succeeds."
            self.logger.info("The query condition is empty.")
            return error_code.INVALID_TYPE_DATA, "The query condition is empty."
        except Exception as e:
            err_msg = "Failed to query data from the table according to the condition. %s" % (str(e)[:40])
            self.logger.error(err_msg)
            return error_code.INVALID_QUERY, err_msg

    def query_conditions(self):
        try:
            placeholders = ''
            query_conditions = []
            params = []
            condition_mapping = self.get_data_config("data_load")
            record_date_mapping = condition_mapping.get("record_date")
            if record_date_mapping is not None:
                for key, data_date_list in record_date_mapping.items():
                    data_date_list = [] if not data_date_list else data_date_list
                    for item in data_date_list:
                        params.append(item)
                        placeholders += '?'
                query_conditions.append(f"record_date IN ({', '.join(placeholders)})")
            for key, value in condition_mapping.items():
                if key == "record_date":
                    continue
                if isinstance(value, list) and value:
                    query_conditions.append(f"{key} IN ({', '.join(['?'] * len(value))})")
                    params.extend(value)
                elif value is not None:
                    query_conditions.append(f"{key} = ?")
                    params.append(value)
            if any(key in condition_mapping for key in model_consts.STIMULUS_COLUMNS):
                join_sql = ("INNER JOIN stimulus_signal_table ON audio_data_table.stimulus_id = "
                            "stimulus_signal_table.stimulus_id")
            base_sql = f'SELECT {model_consts.SELECT_COLUMNS} FROM audio_data_table '
            if query_conditions:
                query_sql = f'{base_sql}{join_sql} WHERE {" AND ".join(query_conditions)}'
            self.cursor.execute(query_sql, params)
            query_data = self.cursor.fetchall()
            return query_data
        except Exception as e:
            err_msg = "Failed to query data from the table according to the condition. %s" % (str(e))
            self.logger.error(err_msg)
            return error_code.INVALID_QUERY, err_msg

    @staticmethod
    def get_data_config(model_name):
        data_load_config = load_config(model_name)
        data_load_config_mapping = {
            "product_model": data_load_config.get("product_model"),
            "record_date": data_load_config.get("record_date"),
            "sample_rate": data_load_config.get("sample_rate"),
            "sweep_method": data_load_config.get("sweep_method"),
            "sweep_type": data_load_config.get("sweep_type"),
            "sweep_duration": data_load_config.get("sweep_duration"),
            "start_feq": data_load_config.get("start_feq"),
            "end_feq": data_load_config.get("end_feq"),
        }
        return data_load_config_mapping

    def delete_all(self, table_name):
        try:
            sql_delete = f'DELETE FROM {table_name}'
            self.cursor.execute(sql_delete)
            self.connection.commit()
            self.logger.info("Delete all information in the table.")
            return error_code.OK, "Delete all information in the table."
        except Exception as e:
            err_msg = "Failed to delete data from the table. %s" % (str(e)[:40])
            self.logger.error(err_msg)
            return error_code.INVALID_DELETE, err_msg

    def delete_with_condition(self, table_name, delete_condition):
        try:
            where_condition = ' AND '.join([f"{key} = ?" for key in delete_condition.keys()])
            sql_delete = f'DELETE FROM {table_name} WHERE {where_condition}'
            self.cursor.execute(sql_delete, list(delete_condition.values()))
            self.connection.commit()
            self.logger.info("Delete the data that meets the condition.")
            return error_code.OK, "Delete the data that meets the condition."
        except Exception as e:
            err_msg = "Failed to delete data from the table. %s" % (str(e)[:40])
            self.logger.error(err_msg)
            return error_code.INVALID_DELETE, err_msg

    def close(self):
        try:
            self.connection.close()
            self.logger.info("Database connection closed.")
            self.logger.shut_down()
            return error_code.OK, "Database connection closed."
        except Exception as e:
            err_msg = "Error closing the connection. %s" % (str(e)[:40])
            self.logger.error(err_msg)
            self.logger.shut_down()
            return error_code.INVALID_CLOSED, err_msg

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if exc_type is not None:
            print(f"DatabaseError: {exc_val}.")
