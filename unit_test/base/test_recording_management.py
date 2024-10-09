import mock
import numpy as np
import pytest

from base.recording_management import RecordingManager
from consts import error_code


class TestRecordingManager(object):
    @pytest.mark.parametrize("signal_db_ret, audio_info, stimulus_parameter, ret", [
        (mock.Mock(), {"recorded_signal": np.array(range(132300)), "sample_rate": 44100, "product_model": 'S004-1',
                       "record_date": '2024-06-07', "labels": 1, "file_path": ""},
         {"sample_rate": 44100, "start_feq": 10, "end_feq": 2000, "sweep_duration": 3, "sweep_method": 'chirp',
          "sweep_type": 'linear', "repeats": 1, "amplitude": 0.5},
         (error_code.INVALID_PATH, "missing file path.")
         ),
        (mock.Mock(), {"recorded_signal": [(1, 2), (5, 6)], "sample_rate": 44100, "product_model": 'S004-1',
                       "record_date": '2024-06-07', "labels": 1, "file_path": "../../audio_data/stored_sample/"
                                                                              "real_product_line/linear_chirp_1/S004-1_80_2000/20240607/OK/20240941-01.wav"},
         {"sample_rate": 44100, "start_feq": 10, "end_feq": 2000, "sweep_duration": 3, "sweep_method": 'chirp',
          "sweep_type": 'linear', "repeats": 1, "amplitude": 0.5},
         (error_code.INVALID_TYPE_DATA, "invalid recorded signal data.")
         ),
        (mock.Mock(), {"recorded_signal": np.array(range(132300)), "sample_rate": 44100, "product_model": 'S004-1',
                       "record_date": '2024-06-07', "labels": 1, "file_path": "../../audio_data/stored_sample/"
                                                                              "real_product_line/linear_chirp_1/S004-1_80_2000/20240607/label/OK/20241002-01.wav"},
         {"sample_rate": 44100, "start_feq": 10, "end_feq": 2000, "sweep_duration": 3, "sweep_method": 'chirp',
          "sweep_type": 'linear', "repeats": 1, "amplitude": 0.5},
         (error_code.OK,
          f"Recorded signal 20241002-01.wav has been saved and its stimulus and recording information to database.")
         ),
        (mock.Mock(), {"recorded_signal": np.array(range(132300)), "sample_rate": 44100, "product_model": 'S004-1',
                       "record_date": '2024-06-07', "labels": 1, "file_path": "../../audio_data/stored_sample/"
                                                                              "real_product_line/linear_chirp_1/S004-1_80_2000/20240607/label/OK/20241005-01"},
         {"sample_rate": 44100, "start_feq": 10, "end_feq": 2000, "sweep_duration": 3, "sweep_method": 'chirp',
          "sweep_type": 'linear', "repeats": 1, "amplitude": 0.5},
         (error_code.OK,
          f"Recorded signal 20241005-01.wav has been saved and its stimulus and recording information to database.")
         ),
        (mock.Mock(), {"recorded_signal": np.array(range(132300)), "sample_rate": 44100, "product_model": 'S004-1',
                       "record_date": '2024-06-07', "labels": 1, "file_path": "../../audio_data/stored_sample/"
                                                                              "real_product_line/linear_chirp_1/S004-1_80_2000/20240607/label/OK/20241006-01.txt"},
         {"sample_rate": 44100, "start_feq": 10, "end_feq": 2000, "sweep_duration": 3, "sweep_method": 'chirp',
          "sweep_type": 'linear', "repeats": 1, "amplitude": 0.5},
         (error_code.OK,
          f"Recorded signal 20241006-01.wav has been saved and its stimulus and recording information to database.")
         ),

        (mock.Mock(), {"recorded_signal": np.array(range(132300)), "sample_rate": 44100, "product_model": 'S004-1',
                       "record_date": '2024-06-07', "labels": 1, "file_path": "../../audio_data/stored_sample/"
                                                                              "real_product_line/linear_chirp_1/S004-1_80_2000/20240607/OK/20240941-01.wav"},
         {"sample_rate": 44100, "start_feq": 10, "end_feq": 2000, "sweep_duration": 3, "sweep_method": 'chirp',
          "sweep_type": 'linear', "repeats": 1, "amplitude": 0.5},
         (error_code.INVALID_PATH, "The file already exists.")
         ),

        (Exception("xxx"), {"recorded_signal": np.array(range(132300)), "sample_rate": 44100, "product_model": 'S004-1',
                            "record_date": '2024-06-07', "labels": 1, "file_path": "../../audio_data/stored_sample/"
                                                                                   "real_product_line/linear_chirp_1/S004-1_80_2000/20240607/OK/20241012-01.wav"},
         {"sample_rate": 44100, "start_feq": 10, "end_feq": 2000, "sweep_duration": 3, "sweep_method": 'chirp',
          "sweep_type": 'linear', "repeats": 1, "amplitude": 0.5},
         (error_code.INVALID_SAVE, "Failed to save the recorded signal file. xxx")
         ),
    ])
    @mock.patch("base.recording_management.RecordingManager.save_signal_info_to_db")
    def test_save_recording_to_wav(self, mock_signal_db, signal_db_ret, audio_info, stimulus_parameter, ret):
        mock_signal_db.side_effect = signal_db_ret
        result = RecordingManager().save_recording_to_wav(audio_info, stimulus_parameter)
        assert result == ret

    # "database_set, stimulus_info_db_set, audio_info_db_set, audio_info, stimulus_parameter, ret", [
    @pytest.mark.parametrize("input_ret, result_ret", [
        ({"database_set": mock.Mock(), "stimulus_info_db_set": [(mock.Mock(), mock.Mock())],
          "audio_info_db_set": mock.Mock(),
          "audio_info": {"recorded_signal": np.array(range(132300)), "sample_rate": 44100, "product_model": 'S004-1',
                         "record_date": '2024-06-07', "labels": 1, "file_path": "../../audio_data/stored_sample/"
                                                                                "real_product_line/linear_chirp_1/S004-1_80_2000/20240607/OK/20240950-01.wav"
                         },
          "stimulus_parameter": {"sample_rate": 44100, "start_feq": 70, "end_feq": 2000,
                                 "sweep_duration": 3, "sweep_method": 'chirp', "sweep_type": 'linear',
                                 "repeats": 1, "amplitude": 0.5}},
         (error_code.OK, "Successfully saved the recording and stimulus signals to the database.")
         ),
        ({"database_set": mock.Mock(), "stimulus_info_db_set": [(mock.Mock(), mock.Mock())],
          "audio_info_db_set": mock.Mock(),
          "audio_info": {"recorded_signal": np.array(range(132300)), "sample_rate": 44100, "product_model": 'S004-1',
                         "record_date": '2024-06-07', "labels": 1, "file_path": "../../audio_data/stored_sample/"
                                                                                "real_product_line/linear_chirp_1/S004-1_80_2000/20240607/OK/20240950-01.wav"},
          "stimulus_parameter": {"sample_rate": 44100, "start_feq": 10, "end_feq": 2000, "sweep_duration": 3,
                                 "sweep_method": 'chirp', "sweep_type": 'linear', "repeats": 1, "amplitude": 0.5
                                 }},
         (error_code.OK, "Successfully saved the recording and stimulus signals to the database.")
         ),
        ({"database_set": Exception(), "stimulus_info_db_set": [(mock.Mock(), mock.Mock())],
          "audio_info_db_set": mock.Mock(),
          "audio_info": {"recorded_signal": np.array(range(132300)), "sample_rate": 44100, "product_model": 'S004-1',
                         "record_date": '2024-06-07', "labels": 1, "file_path": "../../audio_data/stored_sample/"
                                                                                "real_product_line/linear_chirp_1/S004-1_80_2000/20240607/OK/20240950-01.wav"},
          "stimulus_parameter": {"sample_rate": 44100, "start_feq": 10, "end_feq": 2000, "sweep_duration": 3,
                                 "sweep_method": 'chirp', "sweep_type": 'linear', "repeats": 1, "amplitude": 0.5}},
         (error_code.INVALID_SAVE, "Failed to save the recording and stimulus signals to the database. "
                                   "'Exception' object has no attribute 'insert_audio_files_info'")
         )
    ])
    @mock.patch("base.recording_management.RecordingManager.get_audio_info_to_db")
    @mock.patch("base.recording_management.RecordingManager.get_stimulus_info_to_db")
    @mock.patch("base.recording_management.DataSave")
    def test_save_signal_info_to_db(self, mock_database, mock_stimulus, mock_audio, input_ret, result_ret):
        mock_database.return_value.__enter__.return_value = input_ret["database_set"]
        mock_stimulus.side_effect = input_ret["stimulus_info_db_set"]
        mock_audio.return_value = input_ret["audio_info_db_set"]
        result = RecordingManager().save_signal_info_to_db(input_ret["audio_info"], input_ret["stimulus_parameter"])
        assert result == result_ret

    @pytest.mark.parametrize("input_ret, result_ret", [
        ({"database_ret": mock.Mock(),
          "check_db_ret": [('9212f2d3-7b1e-11ef-96ff-107c610bb999', 'chirp', 'linear', 1, 10, 2000, 44100, 3)],
          "get_id_ret": mock.Mock(),
          "stimulus_parameter": {"sample_rate": 44100, "start_feq": 10, "end_feq": 2000, "sweep_duration": 3,
                                 "sweep_method": 'chirp', "sweep_type": 'linear', "repeats": 1}},
         ([('9212f2d3-7b1e-11ef-96ff-107c610bb999', 'chirp', 'linear', 1, 10, 2000, 44100, 3)], False)
         ),
        ({"database_ret": mock.Mock(), "check_db_ret": [],
          "get_id_ret": [('a999', 'chirp', 'linear', 1, 100, 2000, 44100, 3)],
          "stimulus_parameter": {"sample_rate": 44100, "start_feq": 100, "end_feq": 2000, "sweep_duration": 3,
                                 "sweep_method": 'chirp', "sweep_type": 'linear', "repeats": 1}},
         ([('a999', 'chirp', 'linear', 1, 100, 2000, 44100, 3)], True)
         ),
        ({"database_ret": mock.Mock(), "check_db_ret": [],
          "get_id_ret": [],
          "stimulus_parameter": {}}, ([], True)
         ),
    ])
    @mock.patch("base.recording_management.DataSave")
    def test_get_stimulus_info_to_db(self, mock_database, input_ret, result_ret):
        mock_database.return_value.__enter__.return_value = input_ret["database_ret"]
        mock_database.return_value.__enter__.return_value.check_database_info_equal.return_value = input_ret[
            "check_db_ret"]
        mock_database.return_value.__enter__.return_value.get_data_id.return_value = input_ret["get_id_ret"]
        result = RecordingManager().get_stimulus_info_to_db(input_ret["stimulus_parameter"], input_ret["database_ret"])
        assert result == result_ret

    @pytest.mark.parametrize("input_ret, result_ret", [
        ({"database_ret": mock.Mock(), "check_db_ret": [('6f0d377c-7b1f-11ef-99d4-107c610bb999',)], "uuid_ret": [],
          "audio_info": {"recorded_signal": np.array(range(132300)), "sample_rate": 44100, "product_model": 'S004-1',
                         "record_date": '2024-06-07', "labels": 1,
                         "file_path": "../../audio_data/stored_sample/real_product_line/linear_chirp_1/S004-1_80_2000/"
                                      "20240607/OK/20240941-01.wav"},
          "stimulus_data": [('9212f2d3-7b1e-11ef-96ff-107c610bb999', 'chirp', 'linear', 1, 10, 2000, 44100, 3)]},
         ('6f0d377c-7b1f-11ef-99d4-107c610bb999',
          '../../audio_data/stored_sample/real_product_line/linear_chirp_1/S004-1_80_2000/'
          '20240607/OK/20240941-01.wav', 'S004-1', 44100, '2024-06-07', 1, '9212f2d3-7b1e-11ef-96ff-107c610bb999'),
         ),
        ({"database_ret": mock.Mock(), "check_db_ret": [], "uuid_ret": 'a111',
          "audio_info": {"recorded_signal": np.array(range(132300)), "sample_rate": 44100, "product_model": 'S004-1',
                         "record_date": '2024-06-07', "labels": 1,
                         "file_path": "../../audio_data/stored_sample/real_product_line/linear_chirp_1/"
                                      "S004-1_80_2000/20240607/OK/20241010-01.wav"},
          "stimulus_data": [('9212f2d3-7b1e-11ef-96ff-107c610bb999', 'chirp', 'linear', 1, 10, 2000, 44100, 3)]},
         ('a111', '../../audio_data/stored_sample/real_product_line/linear_chirp_1/S004-1_80_2000/20240607/OK/'
                  '20241010-01.wav', 'S004-1', 44100, '2024-06-07', 1, '9212f2d3-7b1e-11ef-96ff-107c610bb999')
         )
    ])
    @mock.patch("base.recording_management.uuid.uuid1")
    @mock.patch("base.recording_management.DataSave")
    def test_get_audio_info_to_db(self, mock_database, mock_uuid, input_ret, result_ret):
        mock_database.return_value.__enter__.return_value = input_ret["database_ret"]
        mock_database.return_value.__enter__.return_value.check_database_info_equal.return_value = input_ret[
            "check_db_ret"]
        mock_uuid.return_value = input_ret["uuid_ret"]
        result = RecordingManager().get_audio_info_to_db(input_ret["audio_info"], input_ret["stimulus_data"],
                                                         input_ret["database_ret"])
        assert result == result_ret

    @pytest.mark.parametrize("database_ret, file_path, new_name, ret", [
        (mock.Mock(), "test1.wav", "new_test1", (error_code.INVALID_PATH, "The old path is invalid.")),
        (mock.Mock(),
         "../../audio_data/stored_sample/real_product_line/linear_chirp_1/S004-1_80_2000/20240607/OK/20240941-01.wav",
         "20240941-02.wav",
         (error_code.OK, "The rename operation successful and the database information updated.")),
        (mock.Mock(),
         "../../audio_data/stored_sample/real_product_line/linear_chirp_1/S004-1_80_2000/20240607/OK/20240953-01.wav",
         "20240953-02",
         (error_code.OK, "The rename operation successful and the database information updated.")),
        (mock.Mock(),
         "../../audio_data/stored_sample/real_product_line/linear_chirp_1/S004-1_80_2000/20240607/OK/20240948-02.wav",
         "20240948-02.wav",
         (error_code.INVALID_PATH, "The new file path already exists.")),
        (Exception(),
         "../../audio_data/stored_sample/real_product_line/linear_chirp_1/S004-1_80_2000/20240607/OK/20240946-02.wav",
         "20240946-03.wav",
         (error_code.INVALID_RENAME,
          "The rename operation failed. 'Exception' object has no attribute 'update_audio_files_info'")),
    ])
    @mock.patch("base.recording_management.DataSave")
    def test_rename_audio(self, mock_database, database_ret, file_path, new_name, ret):
        mock_database.return_value.__enter__.return_value = database_ret
        result = RecordingManager().rename_audio(file_path, new_name)
        assert result == ret

    @pytest.mark.parametrize("input_ret, result_ret", [
        ({"move_ret": mock.Mock(), "database_ret": mock.Mock(),
          "file_path": "../../audio_data/stored_sample/real_product_line/linear_chirp_1/"
                       "S004-1_80_2000/20240607/OK/20240941-01.wav",
          "new_dir_path": "../../audio_data/stored_sample/real_product_line/linear_chirp_1/S004-1_80_2000/20240607/NG"
          }, (error_code.OK, "The move operation succeeded."),
         ),
        ({"move_ret": mock.Mock(), "database_ret": mock.Mock(),
          "file_path": "../../audio_data/stored_sample/real_product_line/linear_chirp_1/"
                       "S004-1_80_2000/20240607/OK/20240941-01.wav",
          "new_dir_path": "../../audio_data/stored_sample/real_product_line/linear_chirp_1/"
                          "S004-1_80_2000/20240607/NG/None",
          }, (error_code.INVALID_PATH, "The directory path is invalid."),
         ),
        ({"move_ret": mock.Mock(), "database_ret": mock.Mock(),
          "file_path": "../../audio_data/stored_sample/real_product_line/linear_chirp_1/"
                       "S004-1_80_2000/20240607/OK/20240941-01.wav",
          "new_dir_path": "../../audio_data/stored_sample/real_product_line/linear_chirp_1/S004-1_80_2000/20240607/OK",
          }, (error_code.INVALID_MOVE, "The file with the same name already exists."),
         ),
        ({"move_ret": mock.Mock(), "database_ret": Exception(),
          "file_path": "../../audio_data/stored_sample/real_product_line/linear_chirp_1/"
                       "S004-1_80_2000/20240607/OK/20240941-01.wav",
          "new_dir_path": "../../audio_data/stored_sample/real_product_line/linear_chirp_1/S004-1_80_2000/20240607/NG",
          }, (error_code.INVALID_MOVE,
              "The move operation failed. 'Exception' object has no attribute 'update_audio_files_info'"),
         ),
    ])
    @mock.patch("base.recording_management.DataSave")
    @mock.patch("base.recording_management.shutil.move")
    def test_move_audio(self, mock_move, mock_database, input_ret, result_ret):
        mock_move.return_value = input_ret["move_ret"]
        mock_database.return_value.__enter__.return_value = input_ret["database_ret"]
        result = RecordingManager().move_audio(input_ret["file_path"], input_ret["new_dir_path"])
        assert result == result_ret

    @pytest.mark.parametrize("database_ret, file_path, ret", [
        (mock.Mock(), "test1.wav", (error_code.INVALID_PATH, "The file does not exist.")),
        (mock.Mock(),
         "../../audio_data/stored_sample/real_product_line/linear_chirp_1/S004-1_80_2000/20240607/OK/20240941-01.wav",
         (error_code.OK, "The file is deleted successfully.")),
        (Exception(),
         "../../audio_data/stored_sample/real_product_line/linear_chirp_1/S004-1_80_2000/20240607/OK/20240932-01.wav",
         (error_code.INVALID_DELETE,
          "The delete operation failed. 'Exception' object has no attribute 'delete_with_condition'"))
    ])
    @mock.patch("base.recording_management.DataSave")
    def test_delete_audio(self, mock_database, database_ret, file_path, ret):
        mock_database.return_value.__enter__.return_value = database_ret
        result = RecordingManager().delete_audio(file_path)
        assert result == ret

    @pytest.mark.parametrize("database_ret, query_set, file_path, ret", [
        (mock.Mock(), [(mock.Mock(), mock.Mock())], "test1.wav",
         (error_code.INVALID_PATH, "The query file does not exist.")),
        (mock.Mock(),
         [([('D:/PyCharm Community Edition 2024.1.4/Python_project/szdongyuan/code1/SpeakerAnomalyDetection/base/../'
             'audio_data/stored_sample/real_product_line/linear_chirp_1/S004-1_80_2000/20240607/OK/20240941-01.wav',
             'S004-1', '2024-06-07', 'chirp', 'linear', 3, '9212f2d3-7b1e-11ef-96ff-107c610bb999', 1)],
           'Query success.')],
         '../../audio_data/stored_sample/real_product_line/linear_chirp_1/S004-1_80_2000/20240607/OK/20240941-01.wav',
         ([('D:/PyCharm Community Edition 2024.1.4/Python_project/szdongyuan/code1/SpeakerAnomalyDetection/base/../'
            'audio_data/stored_sample/real_product_line/linear_chirp_1/S004-1_80_2000/20240607/OK/20240941-01.wav',
            'S004-1', '2024-06-07', 'chirp', 'linear', 3, '9212f2d3-7b1e-11ef-96ff-107c610bb999', 1)], 'Query success.')
         ),
        (mock.Mock(), Exception("xxx"),
         "../../audio_data/stored_sample/real_product_line/linear_chirp_1/S004-1_80_2000/20240607/OK/20240941-01.wav",
         (error_code.INVALID_QUERY, "The query operation failed. xxx")
         )

    ])
    @mock.patch("base.recording_management.DataSave")
    def test_query_signal_info(self, mock_database, database_ret, query_set, file_path, ret):
        mock_database.return_value.__enter__.return_value = database_ret
        mock_database.return_value.__enter__.return_value.query.side_effect = query_set
        result = RecordingManager().query_signal_info(file_path)
        assert result == ret
