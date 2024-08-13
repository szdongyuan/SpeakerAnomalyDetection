from datetime import datetime

import os
import sqlite3
import uuid
import wave

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

    @staticmethod
    def get_audio_data_list(source_dir_list, label):
        data_list = []
        n_file = 0
        for source_dir in source_dir_list:
            source_dir_path = os.path.join(model_consts.STORED_SAMPLE_PATH, source_dir).replace("\\", "/")
            sub_folder_path = source_dir_path + "/" + label
            if not os.path.exists(sub_folder_path):
                continue
            for audio_file in os.listdir(sub_folder_path):
                audio_data_id = str(uuid.uuid1())
                file_path = os.path.join(source_dir, label, audio_file).replace("\\", "/")
                product_model = source_dir.split("/")[2].split("_")[0]
                sample_rate = model_consts.SAMPLE_RATE
                recode_date_time = os.path.getmtime(os.path.join(sub_folder_path, audio_file))
                recode_date = datetime.fromtimestamp(int(recode_date_time))
                sample_data = (audio_data_id, file_path, product_model, sample_rate, recode_date, label)
                data_list.append(sample_data)
                n_file += 1
        print(f"{n_file} ok samples were successfully inserted.")
        return data_list

    def sweep_signal_list(self, source_dir_list):
        sweep_data_list = []
        for source_dir in source_dir_list:
            source_dir_path = model_consts.STORED_SAMPLE_PATH + "/" + source_dir
            source_dir_str = source_dir.split("/")
            sweep_type = source_dir_str[1]
            start_feq = source_dir_str[2].split("_")[1]
            end_feq = source_dir_str[2].split("_")[2]
            sample_rate = model_consts.SAMPLE_RATE
            filename = os.listdir(source_dir_path + "/NG")[0]
            filepath = source_dir_path + "/NG/" + filename
            sweep_duration = self.get_wav_duration(filepath)
            sweep_data = (sweep_type, start_feq, end_feq, sample_rate, sweep_duration)
            sweep_data_list.append(sweep_data)
        return self.remove_rep_data(sweep_data_list)

    @staticmethod
    def remove_rep_data(sweep_data_list):
        sweep_data_list = list(set(sweep_data_list))
        if sweep_data_list:
            for sweep in range(len(sweep_data_list)):
                sweep_id = str(uuid.uuid1())
                temp_list = list(sweep_data_list[sweep])
                temp_list.insert(0, sweep_id)
                sweep_data_list[sweep] = tuple(temp_list)
        return sweep_data_list

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
            err_msg = "Failed to insert data into the database. %s" % (str(e)[:40])
            return error_code.INVALID_INSERT, err_msg

    def query(self, sql_query: str):
        try:
            self.cursor.execute(sql_query)
            query_data = self.cursor.fetchall()
            return query_data, "The conditional query succeeds."
        except Exception as e:
            err_msg = "Failed to query data from the table according to the condition. %s" % (str(e)[:40])
            return error_code.INVALID_QUERY, err_msg

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


