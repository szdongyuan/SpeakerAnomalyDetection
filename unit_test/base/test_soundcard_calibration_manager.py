import mock
import pytest

from base.soundcard_calibration_manager import SoundcardCalibrationManager
from consts import error_code, model_consts
from unit_test.compare_methods import assert_equal


class TestSoundcardCalibrationManager(object):

    test_path = "base.soundcard_calibration_manager.SoundcardCalibrationManager"
    @pytest.mark.parametrize("amplitude, voltage, result_set", [
        ([], 3, (error_code.INVALID_DATA_LOADING, "Input data cannot be None.")),
        (0.1, [], (error_code.INVALID_DATA_LOADING, "Input data cannot be None.")),
        ([0.1], 3, (error_code.INVALID_TYPE_DATA, "Input data must be numeric.")),
        (0.1, 3, (error_code.OK, "Successfully add data.")),
    ])
    def test_add_data(self, amplitude, voltage, result_set):
        result = SoundcardCalibrationManager().add_data(amplitude, voltage)
        assert result == result_set

    @pytest.mark.parametrize("save_json_ret, amplitudes_ret, voltages_ret, result_ret", [
        ([], [], [1, 2], (error_code.INVALID_DATA_LOADING, "Amplitudes and voltages must not be empty.")),
        ([], [0.1, 0.2], [], (error_code.INVALID_DATA_LOADING, "Amplitudes and voltages must not be empty.")),
        ([], [0.1], [1, 2], (error_code.INVALID_DATA_LOADING, "Amplitudes and voltages must have the same length.")),
        ([(error_code.OK, 'xxx')], [0.1, 0.2, 0.3], [2, 3.99, 5.97], (error_code.OK, [0.05037773, -0.00083921])),
        ([(error_code.INVALID_SAVE, 'xxx')], [0.1, 0.2, 0.3], [2, 3.99, 5.97], (error_code.INVALID_SAVE, 'xxx')),
    ])
    @mock.patch(test_path + ".save_coefficients_to_json")
    def test_fit(self, mock_save_json, save_json_ret, amplitudes_ret, voltages_ret, result_ret):
        mock_save_json.side_effect = save_json_ret
        sc = SoundcardCalibrationManager()
        setattr(sc, "amplitudes", amplitudes_ret)
        setattr(sc, "voltages", voltages_ret)
        result = sc.fit()
        if isinstance(result[1], str):
            assert result == result_ret
        else:
            assert result[0] == result_ret[0]
            assert_equal(result[1], result_ret[1])

    @pytest.mark.parametrize("coefficients, target_voltage, result_ret", [
        ([0.05038, -0.0008392], 3, 0.1503),
        ([0.05038, -0.0008392], [1, 2], [0.0495, 0.0999]),
        ([0.05038, -0.0008392], [], []),
    ])
    def test_predict_amplitude(self, coefficients, target_voltage, result_ret):
        result = SoundcardCalibrationManager().predict_amplitude(coefficients, target_voltage)
        assert_equal(result, result_ret)

    @pytest.mark.parametrize("load_json_set, predict_set, target_voltage, result_set", [
        ([(error_code.OK, {"calibration_coefficients": [0.05037773, -0.00083921],
                           "calibration_voltage": [2, 3.99, 5.97],
                           "max_voltage": 5.97})],
         0.1503, 8,
         (error_code.INVALID_DATA_LOADING, "Target voltage cannot be greater than max voltage at calibration.")),
        ([(error_code.OK, {"calibration_coefficients": [0.05037773, -0.00083921],
                           "calibration_voltage": [2, 3.99, 5.97],
                           "max_voltage": 5.97})],
         0.1503, 3, (error_code.OK, 0.1503)),
        ([(error_code.INVALID_DATA_LOADING, {"calibration_coefficients": [0.05037773, -0.00083921],
                                             "calibration_voltage": [2, 3.99, 5.97],
                                             "max_voltage": 5.97})],
         0.1503, 3,
         (error_code.INVALID_DATA_LOADING, "Failed to load coefficients, please calibrate first.")),
    ])
    @mock.patch(test_path + ".predict_amplitude")
    @mock.patch(test_path + ".load_data_from_json")
    def test_calibrate_amplitude(self, mock_load_json, mock_predict, load_json_set, predict_set, target_voltage,
                                 result_set):
        mock_load_json.side_effect = load_json_set
        mock_predict.return_value = predict_set
        result = SoundcardCalibrationManager().calibrate_amplitude(target_voltage)
        assert result == result_set

    @pytest.mark.parametrize("json_ret, coefficients, max_voltages, json_file_name, result_ret", [
        (mock.Mock(), (1, 2), 10, "calibration_coefficients.json",
         (error_code.INVALID_TYPE_DATA, "Coefficients must be a list or numpy array.")),
        (mock.Mock(), [1, 2], 10, "calibration_coefficients.json",
         (error_code.OK, f"Successfully save the coefficients to D:/PyCharm Community Edition "
                         f"2024.1.4/Python_project/szdongyuan/code1/SpeakerAnomalyDetection/consts/../"
                         f"storagecalibration_coefficients.json.")),
        (mock.Mock(), [1, 2], 10, "calibration_coefficients",
         (error_code.OK, f"Successfully save the coefficients to D:/PyCharm Community Edition "
                         f"2024.1.4/Python_project/szdongyuan/code1/SpeakerAnomalyDetection/consts/../"
                         f"storagecalibration_coefficients.json.")),
        (Exception('xxx'), [1, 2], 10, "calibration_coefficients",
         (error_code.INVALID_SAVE, 'Failed saving coefficients to json. xxx')),
    ])
    @mock.patch("json.dump")
    def test_save_coefficients_to_json(self, mock_json, json_ret, coefficients, max_voltages, json_file_name, result_ret):
        mock_json.side_effect = json_ret
        result = SoundcardCalibrationManager().save_coefficients_to_json(coefficients, max_voltages, json_file_name)
        assert result == result_ret

    @pytest.mark.parametrize("load_ret, json_file_name, result_ret", [
        ({"calibration_coefficients": [0.1, 0.2, 0.3]}, "calibration_coefficients.json",
         (error_code.OK, {"calibration_coefficients": [0.1, 0.2, 0.3]})),
        ({"calibration_coefficients": [0.1, 0.2, 0.3]}, "coefficients.json",
         (error_code.INVALID_DATA_LOADING, "This json file does not exist.")),
    ])
    @mock.patch("json.load")
    def test_load_data_from_json(self, mock_load, load_ret, json_file_name, result_ret):
        mock_load.return_value = load_ret
        result = SoundcardCalibrationManager().load_data_from_json(json_file_name)
        assert result == result_ret
