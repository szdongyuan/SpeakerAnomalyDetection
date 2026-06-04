import json
import math
import os

from base.db_manager import DataSave, ensure_audio_database_ready
from base.stimulus_signal.frequency_stepped import (
    resolve_frequency_stepped_schedule,
    valid_frequency_stepped_mode_value,
    validate_frequency_stepped_resolution,
)
from base.stimulus_signal.methods import normalize_stimulus_method
from consts.frequency_stepped_consts import FREQUENCY_STEPPED_CANONICAL_METADATA_KEYS
from consts import model_consts, error_code, running_consts


def _merge_frequency_stepped_noncanonical_metadata(authoritative_metadata, caller_metadata):
    metadata = dict(authoritative_metadata)
    for key, value in caller_metadata.items():
        if key in FREQUENCY_STEPPED_CANONICAL_METADATA_KEYS:
            continue
        metadata[key] = value
    metadata["stimulus_method"] = "frequency_stepped"
    metadata["frequency_mode"] = metadata.get("frequency_mode") or metadata.get("stimulus_type")
    metadata["stimulus_type"] = metadata["frequency_mode"]
    return metadata


def stimulus_row_to_dict(row):
    if isinstance(row, dict):
        result = dict(row)
    else:
        result = {}
        for index, column in enumerate(model_consts.DB_STIMULUS_COLUMNS):
            result[column] = row[index] if index < len(row) else None
    if result.get("voltage") is not None:
        result["voltage"] = float(result["voltage"])
    return result


def _authoritative_frequency_stepped_frequencies(metadata):
    frequencies = metadata.get("frequencies")
    if not isinstance(frequencies, (list, tuple)) or len(frequencies) == 0:
        raise ValueError("frequency_stepped requires non-empty authoritative frequencies")
    if any(isinstance(frequency, bool) for frequency in frequencies):
        raise ValueError("frequency_stepped frequencies must be numeric, not boolean")

    try:
        frequency_values = [float(frequency) for frequency in frequencies]
    except (TypeError, ValueError):
        raise ValueError("frequency_stepped frequencies must be numeric") from None

    if not frequency_values:
        raise ValueError("frequency_stepped requires non-empty authoritative frequencies")
    if any(not math.isfinite(frequency) or frequency <= 0 for frequency in frequency_values):
        raise ValueError("frequency_stepped frequencies must be finite positive values")
    return frequency_values


def _require_frequency_stepped_db_save_resolution(metadata):
    frequency_mode = metadata.get("frequency_mode") or metadata.get("stimulus_type")
    if frequency_mode != "octave":
        return
    metadata["resolution"] = validate_frequency_stepped_resolution(metadata.get("resolution"))


def _require_frequency_stepped_db_load_resolution(metadata):
    frequency_mode = valid_frequency_stepped_mode_value(metadata.get("frequency_mode"))
    stimulus_type = valid_frequency_stepped_mode_value(metadata.get("stimulus_type"))
    if (frequency_mode or stimulus_type) != "octave":
        return
    metadata["resolution"] = validate_frequency_stepped_resolution(metadata.get("resolution"))


def parse_frequency_stepped_row(row):
    parsed = dict(row)
    if normalize_stimulus_method(parsed.get("stimulus_method")) != "frequency_stepped":
        return parsed

    metadata_json = parsed.get("stimulus_metadata_json")
    try:
        metadata = json.loads(metadata_json) if metadata_json else None
        if not isinstance(metadata, dict):
            raise ValueError("frequency_stepped metadata must be a JSON object")
        if normalize_stimulus_method(metadata.get("stimulus_method")) != "frequency_stepped":
            raise ValueError("frequency_stepped metadata method mismatch")
        if "frequencies" in metadata:
            metadata["frequencies"] = _authoritative_frequency_stepped_frequencies(metadata)
        _require_frequency_stepped_db_load_resolution(metadata)
        sample_rate = int(metadata.get("sample_rate"))
        schedule = resolve_frequency_stepped_schedule(metadata, sample_rate)
    except Exception:
        parsed["step_sc_row_state"] = "invalid_metadata"
        parsed.pop("stimulus_payload", None)
        return parsed

    payload = _merge_frequency_stepped_noncanonical_metadata(schedule.metadata, metadata)
    for key in ("stimulus_id", "stimulus_name", "is_default", "voltage_type", "voltage"):
        if key in parsed:
            payload[key] = parsed[key]
    parsed["step_sc_row_state"] = "valid"
    parsed["stimulus_payload"] = payload
    return parsed


def _query_rows_to_dicts(query_data):
    return [parse_frequency_stepped_row(stimulus_row_to_dict(row)) for row in query_data]


def frequency_stepped_insert_values(stimulus_info, is_default):
    metadata = dict(stimulus_info)
    method = normalize_stimulus_method(metadata.get("stimulus_method"))
    if method != "frequency_stepped":
        raise ValueError("Not a frequency_stepped stimulus")
    metadata["frequencies"] = _authoritative_frequency_stepped_frequencies(metadata)
    _require_frequency_stepped_db_save_resolution(metadata)

    metadata["stimulus_method"] = "frequency_stepped"
    frequency_mode = metadata.get("frequency_mode") or metadata.get("stimulus_type")
    metadata["frequency_mode"] = frequency_mode
    metadata["stimulus_type"] = frequency_mode
    metadata.setdefault("voltage_type", "RMS")
    metadata["voltage"] = float(metadata.get("voltage", 1.0))
    metadata["is_default"] = is_default

    schedule = resolve_frequency_stepped_schedule(metadata, int(metadata["sample_rate"]))
    metadata = _merge_frequency_stepped_noncanonical_metadata(schedule.metadata, metadata)
    frequency_mode = metadata["frequency_mode"]
    frequency_values = [float(frequency) for frequency in metadata["frequencies"]]
    metadata["voltage_type"] = stimulus_info.get("voltage_type", metadata.get("voltage_type", "RMS"))
    metadata["voltage"] = float(stimulus_info.get("voltage", metadata.get("voltage", 1.0)))
    metadata["is_default"] = is_default

    try:
        metadata_json = json.dumps(metadata, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Failed to serialize frequency_stepped metadata: {exc}") from exc

    return tuple(
        {
            "stimulus_method": "frequency_stepped",
            "stimulus_type": frequency_mode,
            "repeat_times": int(metadata.get("repeat_times", 1)),
            "start_freq": int(round(frequency_values[0])),
            "stop_freq": int(round(frequency_values[-1])),
            "sample_rate": int(metadata["sample_rate"]),
            "total_time": metadata["total_time"],
            "num_steps": len(frequency_values),
            "voltage_type": metadata.get("voltage_type", "RMS"),
            "voltage": metadata["voltage"],
            "is_default": is_default,
            "stimulus_name": metadata.get("stimulus_name"),
            "stimulus_metadata_json": metadata_json,
        }.get(key)
        for key in model_consts.INERT_STIMULUS_RICH_CONFIG_COLUMNS
    )


class StimulusSignalManagement(object):
    @staticmethod
    def update_stimulus_default(stimulus_id, is_default):
        try:
            ensure_audio_database_ready()
            with DataSave(model_consts.AUDIO_DATABASE_PATH) as database:
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
            ensure_audio_database_ready()
            with DataSave(model_consts.AUDIO_DATABASE_PATH) as database:
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
            ensure_audio_database_ready()
            with DataSave(model_consts.AUDIO_DATABASE_PATH) as database:
                query_code, query_data = database.query("stimulus_signal_table", model_consts.DB_STIMULUS_COLUMNS)
            if query_code == error_code.OK and query_data:
                return error_code.OK, _query_rows_to_dicts(query_data)
            else:
                return error_code.INVALID_QUERY, "Failed to query stimulus signal info or no stimulus signal info."
        except Exception as e:
            err_msg = "Failed to query stimulus signal. %s" % (str(e)[:40])
            return error_code.INVALID_QUERY, err_msg

    @staticmethod
    def save_stimulus_info_to_db(stimulus_info: dict):
        try:
            ensure_audio_database_ready()
            with DataSave(model_consts.AUDIO_DATABASE_PATH) as database:
                normalized_info = stimulus_info.copy()
                normalized_info.setdefault('num_steps', None)
                normalized_info.setdefault('voltage_type', 'RMS')
                normalized_info['voltage'] = float(normalized_info.get("voltage", 1.0))
                stimulus_info.setdefault('num_steps', normalized_info['num_steps'])
                stimulus_info.setdefault('voltage_type', normalized_info['voltage_type'])
                stimulus_info['voltage'] = normalized_info['voltage']

                name_result = database.query_matching_data(
                    [[stimulus_info.get("stimulus_name")]],
                    "stimulus_signal_table",
                    ['stimulus_name'],
                    ['stimulus_id']
                )
                if name_result: 
                    return error_code.INVALID_NAME, "This stimulus signals name info already exists."

                is_default = database.set_default("stimulus_signal_table")
                if normalize_stimulus_method(stimulus_info.get("stimulus_method")) == "frequency_stepped":
                    insert_stimulus_config = frequency_stepped_insert_values(stimulus_info, is_default)
                    insert_columns = model_consts.DB_STIMULUS_COLUMNS
                else:
                    normalized_info = stimulus_info.copy()
                    normalized_info.setdefault('num_steps', None)
                    normalized_info.setdefault('voltage_type', 'RMS')
                    normalized_info['voltage'] = float(normalized_info.get("voltage", 1.0))
                    normalized_info["is_default"] = is_default
                    insert_stimulus_config = tuple(
                        normalized_info.get(key) for key in model_consts.INERT_STIMULUS_CONFIG_COLUMNS
                    )
                    insert_columns = model_consts.DB_STIMULUS_SCALAR_COLUMNS

                insert_stimulus_config = database.get_data_id([insert_stimulus_config], 0)
                insert_code, msg = database.insert_data_into_db(
                    "stimulus_signal_table",
                    insert_columns,
                    insert_stimulus_config,
                )
                if insert_code == error_code.OK:
                    return error_code.OK, "Successfully saved stimulus signals to the database."
                else:
                    return error_code.INVALID_INSERT, msg

        except Exception as e:
            err_msg = "Failed to save stimulus signals to the database. %s" % (str(e)[:40])
            return error_code.INVALID_SAVE, err_msg
        
    @staticmethod
    def delete_stimulus_info_from_db(stimulus_name: str):
        delete_condition = {"stimulus_name": stimulus_name}
        try:
            ensure_audio_database_ready()
            with DataSave(model_consts.AUDIO_DATABASE_PATH) as database:
                delete_code, msg = database.delete_with_condition("stimulus_signal_table", delete_condition)
                return delete_code, msg
        except Exception as e:
            err_msg = "Delete stimulus info from database error: %s" % str(e)
            return error_code.INVALID_DELETE, err_msg
        
    @staticmethod
    def update_stimulus_info_to_db(stimulus_info: dict):
        try:
            ensure_audio_database_ready()
            with DataSave(model_consts.AUDIO_DATABASE_PATH) as database:
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

    @staticmethod
    def update_stimulus_params_to_db(stimulus_id: int, update_params: dict):
        """
        Update editable stimulus parameters for a given stimulus_id.

        Only allows updating of configurable numeric parameters consistent with UI constraints.
        The following fields are supported (all optional):
        - repeat_times (int)
        - start_freq (int)
        - stop_freq (int)
        - sample_rate (int)
        - total_time (float)
        - num_steps (int)
        - voltage (float)

        Args:
            stimulus_id: int, primary key in stimulus_signal_table
            update_params: dict, subset of supported fields to update

        Returns:
            (code, message) tuple
        """
        if not isinstance(update_params, dict) or not update_params:
            return error_code.INVALID_UPDATE, "No parameters provided to update."
        # Whitelist supported fields to avoid accidental updates
        allowed_fields = {
            "repeat_times",
            "start_freq",
            "stop_freq",
            "sample_rate",
            "total_time",
            "num_steps",
            "voltage",
        }
        # Filter update_params by allowed fields
        update_data = {k: v for k, v in update_params.items() if k in allowed_fields}
        if not update_data:
            return error_code.INVALID_UPDATE, "No valid parameters to update."
        try:
            ensure_audio_database_ready()
            with DataSave(model_consts.AUDIO_DATABASE_PATH) as database:
                # Ensure record exists
                result = database.query_matching_data(
                    [(stimulus_id,)],
                    "stimulus_signal_table",
                    ["stimulus_id"],
                    ["stimulus_id", "stimulus_method"],
                )
                if not result:
                    return error_code.INVALID_UPDATE, "Stimulus ID does not exist."
                if normalize_stimulus_method(result[0][1]) == "frequency_stepped":
                    return (
                        error_code.INVALID_UPDATE,
                        "Direct scalar parameter edits are blocked for frequency_stepped stimuli.",
                    )
                update_code, msg = database.update_table_data(
                    "stimulus_signal_table", update_data, {"stimulus_id": stimulus_id}
                )
                if update_code == error_code.OK:
                    return error_code.OK, "Stimulus parameters updated."
                else:
                    return error_code.INVALID_UPDATE, msg
        except Exception as e:
            err_msg = "Failed to update the stimulus parameters. %s" % str(e)
            return error_code.INVALID_UPDATE, err_msg

