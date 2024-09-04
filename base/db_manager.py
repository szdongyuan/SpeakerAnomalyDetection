from datetime import datetime

import os
import sqlite3
import uuid
import wave

from base.load_config import load_config
from consts import model_consts, error_code


class DataSave(object):

    def __init__(self, db_name):
        self.db_name = db_name
        self.connection = None
        self.cursor = None
        self.connect()

    def connect(self):
        try:
            self.connection = sqlite3.connect(self.db_name)
            self.cursor = self.connection.cursor()
            return error_code.OK, "Successfully connect to database."
        except Exception as e:
            err_msg = "Failed to connect to the database %s" % (str(e)[:40])
            return error_code.INVALID_CONNECT_DATABASE, err_msg

    def create_table(self):
        try:
            self.connection = sqlite3.connect(self.db_name)
            self.cursor = self.connection.cursor()
            create_audio_data_table_sql = '''
            create table if not exists audio_data_table(
                audio_data_id char(100) primary key not null,
                file_path varchar(200) not null,
                product_model char(50) not null,
                sample_rate integer not null,
                record_date datetime not null,
                labels bit not null,
                sweep_id char(100),
                FOREIGN KEY (sweep_id) REFERENCES sweep_signal_table (sweep_id) 
            );
            '''
            create_sweep_signal_table_sql = '''
            create table if not exists sweep_signal_table(
                sweep_id char(100) primary key not null,
        		sweep_type char(100) not null,
        		start_feq integer not null,
        		end_feq integer not null,
        		sample_rate integer not null,
        		sweep_duration integer not null 
        	);
           '''
            self.cursor.execute(create_audio_data_table_sql)
            self.cursor.execute(create_sweep_signal_table_sql)
            self.connection.commit()
            return error_code.OK, "Table creation success."
        except Exception as e:
            err_msg = "Failed to create table. %s" % (str(e)[:40])
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
                sample_sweep_data = self.get_audio_data_sweep_info(sub_folder_path + "/" + audio_file)
                (result, ) = self.check_database_info_equal([sample_sweep_data], "sweep_signal_table", model_consts.SWEEP_COLUMNS,
                                                        ['sweep_signal_table.sweep_id'])
                (sweep_id, ) = result if result else None
                sample_data = (audio_data_id, file_path, product_model, sample_rate, record_date, label, sweep_id)
                data_list.append(sample_data)
                n_file += 1
        print(f"{n_file} audio samples were successfully inserted.")
        return data_list

    def sweep_signal_file_list(self, source_dir_list, label):
        sweep_data = []
        for source_dir in source_dir_list:
            source_dir_path = os.path.join(model_consts.STORED_SAMPLE_PATH, source_dir).replace("\\", "/")
            sub_folder_path = source_dir_path + "/" + label
            if not os.path.exists(sub_folder_path):
                continue
            for audio_file in os.listdir(sub_folder_path):
                sample_sweep_data = self.get_audio_data_sweep_info(sub_folder_path + "/" + audio_file)
                sweep_data.append(sample_sweep_data)
        sweep_data = list(set(sweep_data))
        result = self.check_database_info_equal(sweep_data, "sweep_signal_table", model_consts.SWEEP_COLUMNS,
                                                model_consts.SWEEP_COLUMNS)
        sweep_data_list = [item for item in sweep_data if item not in set(result)]
        return self.get_data_id(sweep_data_list, 0)

    def check_database_info_equal(self, data_list, table_name, check_column, select_column):
        result = []
        base_sql = ' AND '.join([f"{column} = ?" for column in check_column])
        for data_item in data_list:
            sql_select = f"select {', '.join(select_column)} from {table_name} where {base_sql}"
            self.cursor.execute(sql_select, data_item)
            fet_result = self.cursor.fetchall()
            if fet_result:
                result.extend(row for row in fet_result)
        return result

    def get_audio_data_sweep_info(self, file_path):
        audio_sweep_data = ()
        if file_path:
            file_path_dir = file_path.split("/")
            sweep_type = file_path_dir[11]
            start_feq = file_path_dir[12].split("_")[1]
            end_feq = file_path_dir[12].split("_")[2]
            sample_rate = model_consts.SAMPLE_RATE
            sweep_duration = self.get_wav_duration(file_path)
            audio_sweep_data = (sweep_type, int(start_feq), int(end_feq), sample_rate, sweep_duration)
        return audio_sweep_data

    @staticmethod
    def get_data_id(data_list, id_index: int):
        for i, item in enumerate(data_list):
            data_id = str(uuid.uuid1())
            data_list[i] = item[:id_index] + (data_id, ) + item[id_index:]
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
                return error_code.OK, "data empty."
            values_num = ','.join(['?'] * len(data[0]))
            sql = f'insert into {table_name} ({columns}) values ({values_num});'
            self.cursor.executemany(sql, data)
            self.connection.commit()
            return error_code.OK, "Insert data successfully."
        except Exception as e:
            err_msg = "Failed to insert data into the database. %s" % (str(e))
            return error_code.INVALID_INSERT, err_msg

    def query(self, sql_query: str):
        try:
            self.cursor.execute(sql_query)
            query_data = self.cursor.fetchall()
            return query_data, "The conditional query succeeds."
        except Exception as e:
            err_msg = "Failed to query data from the table according to the condition. %s" % (str(e)[:40])
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
            if any(key in condition_mapping for key in model_consts.SWEEP_COLUMNS):
                join_sql = "inner join sweep_signal_table on audio_data_table.sweep_id = sweep_signal_table.sweep_id"
            base_sql = f'select {model_consts.SELECT_COLUMNS} from audio_data_table '
            if query_conditions:
                query_sql = f'{base_sql}{join_sql} where {" AND ".join(query_conditions)}'
            self.cursor.execute(query_sql, params)
            query_data = self.cursor.fetchall()
            return query_data
        except Exception as e:
            err_msg = "Failed to query data from the table according to the condition. %s" % (str(e))
            return error_code.INVALID_QUERY, err_msg

    @staticmethod
    def get_data_config(model_name):
        data_load_config = load_config(model_name)
        data_load_config_mapping = {
            "product_model": data_load_config.get("product_model"),
            "record_date": data_load_config.get("record_date"),
            "sample_rate": data_load_config.get("sample_rate"),
            "sweep_type": data_load_config.get("sweep_type"),
            "sweep_duration": data_load_config.get("sweep_duration"),
            "start_feq": data_load_config.get("start_feq"),
            "end_feq": data_load_config.get("end_feq"),
        }
        return data_load_config_mapping

    def delete_all(self, table_name):
        try:
            sql_delete = f'delete from {table_name}'
            self.cursor.execute(sql_delete)
            self.connection.commit()
            return error_code.OK, "Delete all information in the table."
        except Exception as e:
            err_msg = "Failed to delete data from the table. %s" % (str(e)[:40])
            return error_code.INVALID_DELETE, err_msg

    def delete_with_condition(self, table_name, condition):
        try:
            sql_delete = f'delete from {table_name} where {condition}'
            self.cursor.execute(sql_delete)
            self.connection.commit()
            return error_code.OK, "Delete the data that meets the condition."
        except Exception as e:
            err_msg = "Failed to delete data from the table. %s" % (str(e)[:40])
            return error_code.INVALID_DELETE, err_msg

    def close(self):
        try:
            self.connection.close()
            return error_code.OK, "Database connection closed."
        except Exception as e:
            err_msg = "Error closing the connection. %s" % (str(e)[:40])
            return error_code.INVALID_CLOSED, err_msg


