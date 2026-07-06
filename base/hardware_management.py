import json
import math
import os
import sqlite3
import uuid
from contextlib import contextmanager

from consts import model_consts
from consts.audio_consts import VALID_BIT_DEPTHS, VALID_SAMPLE_RATES


class HardwareManagementError(Exception):
    pass


class MissingHardwareTablesError(HardwareManagementError):
    pass


class HardwareValidationError(HardwareManagementError):
    pass


class HardwareRuntimeMatchError(HardwareManagementError):
    pass


def infer_hardware_type(max_input_channels, max_output_channels):
    if max_input_channels > 0 and max_output_channels > 0:
        return "audio_interface"
    if max_input_channels > 0:
        return "microphone"
    if max_output_channels > 0:
        return "speaker"
    return "other"


def build_channel_placeholders(hardware_id, max_input_channels, max_output_channels):
    placeholders = []
    for index in range(max_input_channels):
        placeholders.append(
            {
                "channel_id": str(uuid.uuid1()),
                "hardware_id": hardware_id,
                "direction": "input",
                "channel_index": index,
                "channel_label": f"In{index + 1}",
            }
        )
    for index in range(max_output_channels):
        placeholders.append(
            {
                "channel_id": str(uuid.uuid1()),
                "hardware_id": hardware_id,
                "direction": "output",
                "channel_index": index,
                "channel_label": f"Out{index + 1}",
            }
        )
    return placeholders


def match_runtime_device(asset, runtime_devices, get_hostapi_name):
    matches = [
        device
        for device in runtime_devices
        if device.get("name") == asset.get("device_name") and get_hostapi_name(device) == asset.get("hostapi_name")
    ]
    if not matches:
        raise HardwareRuntimeMatchError("registered hardware is not currently available")
    if len(matches) > 1:
        raise HardwareRuntimeMatchError("registered hardware matches multiple current devices")
    return matches[0]


def augment_runtime_device(runtime_device, asset):
    payload = dict(runtime_device)
    payload.update(
        {
            "hardware_id": asset["hardware_id"],
            "display_name": asset["display_name"],
            "device_name": asset["device_name"],
            "hardware_type": asset["hardware_type"],
            "hostapi_name": asset["hostapi_name"],
            "samplerate": asset["samplerate"],
            "bit_depth": asset["bit_depth"],
            "latency_ms": asset["latency_ms"],
        }
    )
    return payload


def build_selected_device_payload(mic, speaker, mic_channels):
    return {
        "version": 2,
        "mic": _selected_device_entry(mic),
        "speaker": _selected_device_entry(speaker),
        "mic_channels": [int(channel) for channel in mic_channels],
    }


def _selected_device_entry(device):
    return {
        "hardware_id": device["hardware_id"],
        "display_name": device["display_name"],
        "hardware_type": device["hardware_type"],
        "name": device["device_name"],
        "device_name": device["device_name"],
        "hostapi_name": device["hostapi_name"],
        "default_samplerate": device.get("default_samplerate"),
        "samplerate": device["samplerate"],
        "bit_depth": device["bit_depth"],
        "latency_ms": device["latency_ms"],
        "max_input_channels": device.get("max_input_channels"),
        "max_output_channels": device.get("max_output_channels"),
    }


class HardwareManagementRepository:
    def __init__(self, db_path=None):
        self.db_path = db_path or model_consts.SYSTEM_DATABASE_PATH

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.db_path)
        connection.execute("PRAGMA foreign_keys = ON;")
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def tables_exist(self):
        if not os.path.exists(self.db_path):
            return False
        with self._connection() as connection:
            table_names = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        return {
            model_consts.HARDWARE_ASSETS_TABLE,
            model_consts.HARDWARE_CHANNEL_CALIBRATIONS_TABLE,
        }.issubset(table_names)

    def _require_tables(self):
        if not self.tables_exist():
            raise MissingHardwareTablesError("硬件管理表不存在，请使用最新版数据库")

    def list_assets(self):
        self._require_tables()
        with self._connection() as connection:
            rows = connection.execute(
                f"SELECT {', '.join(model_consts.HARDWARE_ASSET_COLUMNS)} "
                f"FROM {model_consts.HARDWARE_ASSETS_TABLE} ORDER BY rowid"
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def list_assets_for_selection(self):
        grouped = {}
        for asset in self.list_assets():
            api_group = grouped.setdefault(asset["hostapi_name"], {"input": [], "output": []})
            if asset["max_input_channels"] > 0:
                api_group["input"].append(asset)
            if asset["max_output_channels"] > 0:
                api_group["output"].append(asset)
        return grouped

    def get_asset(self, hardware_id):
        self._require_tables()
        with self._connection() as connection:
            row = connection.execute(
                f"SELECT {', '.join(model_consts.HARDWARE_ASSET_COLUMNS)} "
                f"FROM {model_consts.HARDWARE_ASSETS_TABLE} WHERE hardware_id = ?",
                (hardware_id,),
            ).fetchone()
        return _row_to_dict(row) if row is not None else None

    def list_channels(self, hardware_id, direction=None):
        self._require_tables()
        params = [hardware_id]
        where_clause = "hardware_id = ?"
        if direction is not None:
            where_clause += " AND direction = ?"
            params.append(direction)
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT {', '.join(model_consts.HARDWARE_CHANNEL_CALIBRATION_COLUMNS)}
                FROM {model_consts.HARDWARE_CHANNEL_CALIBRATIONS_TABLE}
                WHERE {where_clause}
                ORDER BY
                    CASE direction WHEN 'input' THEN 0 WHEN 'output' THEN 1 ELSE 2 END,
                    channel_index,
                    channel_label
                """,
                params,
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def get_channel_calibration(self, hardware_id, direction, channel_index, calibration_type=None):
        self._require_tables()
        _validate_required_text(hardware_id, "hardware_id")
        normalized_direction = _validate_direction(direction)
        normalized_channel = _validate_channel_index(channel_index)
        params = [hardware_id, normalized_direction, normalized_channel]
        calibration_clause = ""
        if calibration_type is not None:
            _validate_required_text(calibration_type, "calibration_type")
            calibration_clause = " AND calibration_type = ?"
            params.append(calibration_type)
        with self._connection() as connection:
            row = connection.execute(
                f"""
                SELECT {', '.join(model_consts.HARDWARE_CHANNEL_CALIBRATION_COLUMNS)}
                FROM {model_consts.HARDWARE_CHANNEL_CALIBRATIONS_TABLE}
                WHERE hardware_id = ? AND direction = ? AND channel_index = ?
                {calibration_clause}
                """,
                params,
            ).fetchone()
        return _row_to_dict(row)

    def list_channel_calibrations(self, hardware_id, direction, calibration_type=None):
        self._require_tables()
        _validate_required_text(hardware_id, "hardware_id")
        normalized_direction = _validate_direction(direction)
        params = [hardware_id, normalized_direction]
        calibration_clause = "AND calibration_type IS NOT NULL"
        if calibration_type is not None:
            _validate_required_text(calibration_type, "calibration_type")
            calibration_clause = "AND calibration_type = ?"
            params.append(calibration_type)
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT {', '.join(model_consts.HARDWARE_CHANNEL_CALIBRATION_COLUMNS)}
                FROM {model_consts.HARDWARE_CHANNEL_CALIBRATIONS_TABLE}
                WHERE hardware_id = ? AND direction = ? {calibration_clause}
                ORDER BY channel_index
                """,
                params,
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def update_mic_channel_calibrations(self, hardware_id, channel_factors, channel_standard_spl):
        self._require_tables()
        _validate_required_text(hardware_id, "hardware_id")
        if not channel_factors:
            raise HardwareValidationError("channel_factors is required")
        if not isinstance(channel_standard_spl, dict):
            raise HardwareValidationError("channel_standard_spl is required")

        normalized_standard_spl_by_channel = {}
        for channel_index, standard_spl in channel_standard_spl.items():
            normalized_standard_spl_by_channel[_validate_channel_index(channel_index)] = standard_spl

        normalized_rows = []
        for channel_index, factor_value in channel_factors.items():
            normalized_channel = _validate_channel_index(channel_index)
            if normalized_channel not in normalized_standard_spl_by_channel:
                raise HardwareValidationError("standard_spl is required for mic_v2pa calibration")
            standard_spl = normalized_standard_spl_by_channel[normalized_channel]
            normalized_rows.append(
                (
                    normalized_channel,
                    _validate_finite_positive(factor_value, "factor_value"),
                    _validate_standard_spl_required(standard_spl),
                )
            )

        # 外层管理连接生命周期，退出时关闭连接。
        with self._connection() as connection:
            # 内层管理 sqlite 事务（为_connection的返回值），批量更新成功提交，异常回滚。
            with connection:
                for normalized_channel, factor_value, standard_spl in normalized_rows:
                    self._require_channel_placeholder(connection, hardware_id, "input", normalized_channel)
                    connection.execute(
                        f"""
                        UPDATE {model_consts.HARDWARE_CHANNEL_CALIBRATIONS_TABLE}
                        SET calibration_type = ?,
                            factor_value = ?,
                            standard_spl = ?,
                            max_voltage = NULL,
                            coefficients_json = NULL,
                            updated_at = DATETIME('now', '+8 hours')
                        WHERE hardware_id = ? AND direction = 'input' AND channel_index = ?
                        """,
                        (
                            "mic_v2pa",
                            factor_value,
                            standard_spl,
                            hardware_id,
                            normalized_channel,
                        ),
                    )

    def clear_mic_channel_calibrations(self, hardware_id, channel_indices=None):
        self._require_tables()
        _validate_required_text(hardware_id, "hardware_id")
        with self._connection() as connection:
            with connection:
                if channel_indices is None:
                    connection.execute(
                        f"""
                        UPDATE {model_consts.HARDWARE_CHANNEL_CALIBRATIONS_TABLE}
                        SET calibration_type = NULL,
                            factor_value = NULL,
                            standard_spl = NULL,
                            max_voltage = NULL,
                            coefficients_json = NULL,
                            updated_at = DATETIME('now', '+8 hours')
                        WHERE hardware_id = ? AND direction = 'input'
                        """,
                        (hardware_id,),
                    )
                    return
                for channel_index in channel_indices:
                    normalized_channel = _validate_channel_index(channel_index)
                    self._require_channel_placeholder(connection, hardware_id, "input", normalized_channel)
                    connection.execute(
                        f"""
                        UPDATE {model_consts.HARDWARE_CHANNEL_CALIBRATIONS_TABLE}
                        SET calibration_type = NULL,
                            factor_value = NULL,
                            standard_spl = NULL,
                            max_voltage = NULL,
                            coefficients_json = NULL,
                            updated_at = DATETIME('now', '+8 hours')
                        WHERE hardware_id = ? AND direction = 'input' AND channel_index = ?
                        """,
                        (hardware_id, normalized_channel),
                    )

    def update_output_amplitude_calibration(self, hardware_id, coefficients, max_voltage, channel_index=0):
        self._require_tables()
        _validate_required_text(hardware_id, "hardware_id")
        normalized_channel = _validate_channel_index(channel_index)
        normalized_coefficients = _validate_output_coefficients(coefficients)
        normalized_max_voltage = _validate_finite_positive(max_voltage, "max_voltage")
        coefficients_json = json.dumps({"calibration_coefficients": normalized_coefficients})

        with self._connection() as connection:
            with connection:
                self._require_channel_placeholder(connection, hardware_id, "output", normalized_channel)
                connection.execute(
                    f"""
                    UPDATE {model_consts.HARDWARE_CHANNEL_CALIBRATIONS_TABLE}
                    SET calibration_type = ?,
                        factor_value = NULL,
                        standard_spl = NULL,
                        max_voltage = ?,
                        coefficients_json = ?,
                        updated_at = DATETIME('now', '+8 hours')
                    WHERE hardware_id = ? AND direction = 'output' AND channel_index = ?
                    """,
                    (
                        "output_amplitude",
                        normalized_max_voltage,
                        coefficients_json,
                        hardware_id,
                        normalized_channel,
                    ),
                )

    def get_output_amplitude_calibration(self, hardware_id, channel_index=0):
        return self.get_channel_calibration(
            hardware_id,
            "output",
            channel_index,
            "output_amplitude",
        )

    def clear_output_amplitude_calibration(self, hardware_id, channel_index=0):
        self._require_tables()
        _validate_required_text(hardware_id, "hardware_id")
        normalized_channel = _validate_channel_index(channel_index)
        with self._connection() as connection:
            with connection:
                self._require_channel_placeholder(connection, hardware_id, "output", normalized_channel)
                connection.execute(
                    f"""
                    UPDATE {model_consts.HARDWARE_CHANNEL_CALIBRATIONS_TABLE}
                    SET calibration_type = NULL,
                        factor_value = NULL,
                        standard_spl = NULL,
                        max_voltage = NULL,
                        coefficients_json = NULL,
                        updated_at = DATETIME('now', '+8 hours')
                    WHERE hardware_id = ? AND direction = 'output' AND channel_index = ?
                    """,
                    (hardware_id, normalized_channel),
                )

    def _require_channel_placeholder(self, connection, hardware_id, direction, channel_index):
        row = connection.execute(
            f"""
            SELECT channel_id
            FROM {model_consts.HARDWARE_CHANNEL_CALIBRATIONS_TABLE}
            WHERE hardware_id = ? AND direction = ? AND channel_index = ?
            """,
            (hardware_id, direction, channel_index),
        ).fetchone()
        if row is None:
            raise HardwareValidationError(
                "channel calibration placeholder is missing; please re-register or repair the hardware record"
            )
        return row

    def register_asset(
        self,
        runtime_device,
        hostapi_name,
        display_name,
        samplerate,
        bit_depth=32,
        latency_ms=100,
    ):
        self._require_tables()
        _validate_required_text(hostapi_name, "hostapi_name")
        _validate_required_text(runtime_device.get("name"), "device_name")
        _validate_required_text(display_name, "display_name")
        _validate_samplerate(samplerate)
        _validate_bit_depth(bit_depth)
        _validate_latency(latency_ms)

        hardware_id = str(uuid.uuid1())
        max_input_channels = int(runtime_device.get("max_input_channels", 0) or 0)
        max_output_channels = int(runtime_device.get("max_output_channels", 0) or 0)
        asset = {
            "hardware_id": hardware_id,
            "hardware_type": infer_hardware_type(max_input_channels, max_output_channels),
            "display_name": display_name.strip(),
            "device_name": runtime_device.get("name"),
            "hostapi_name": hostapi_name.strip(),
            "samplerate": int(samplerate),
            "bit_depth": int(bit_depth),
            "latency_ms": int(latency_ms),
            "max_input_channels": max_input_channels,
            "max_output_channels": max_output_channels,
            "updated_at": None,
        }
        channels = build_channel_placeholders(hardware_id, max_input_channels, max_output_channels)

        with self._connection() as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO hardware_assets (
                        hardware_id,
                        hardware_type,
                        display_name,
                        device_name,
                        hostapi_name,
                        samplerate,
                        bit_depth,
                        latency_ms,
                        max_input_channels,
                        max_output_channels
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        asset["hardware_id"],
                        asset["hardware_type"],
                        asset["display_name"],
                        asset["device_name"],
                        asset["hostapi_name"],
                        asset["samplerate"],
                        asset["bit_depth"],
                        asset["latency_ms"],
                        asset["max_input_channels"],
                        asset["max_output_channels"],
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO hardware_channel_calibrations (
                        channel_id,
                        hardware_id,
                        direction,
                        channel_index,
                        channel_label
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            channel["channel_id"],
                            channel["hardware_id"],
                            channel["direction"],
                            channel["channel_index"],
                            channel["channel_label"],
                        )
                        for channel in channels
                    ],
                )
        return dict(asset)

    def register_hardware_asset(self, *args, **kwargs):
        return self.register_asset(*args, **kwargs)

    def update_asset_fields(self, hardware_id, fields):
        self._require_tables()
        if not fields:
            return False

        allowed_fields = {"display_name", "samplerate", "bit_depth", "latency_ms"}
        invalid_fields = sorted(set(fields) - allowed_fields)
        if invalid_fields:
            raise HardwareValidationError(f"Cannot update immutable fields: {', '.join(invalid_fields)}")

        update_fields = dict(fields)
        if "display_name" in update_fields:
            _validate_required_text(update_fields["display_name"], "display_name")
            update_fields["display_name"] = update_fields["display_name"].strip()
        if "samplerate" in update_fields:
            _validate_samplerate(update_fields["samplerate"])
            update_fields["samplerate"] = int(update_fields["samplerate"])
        if "bit_depth" in update_fields:
            _validate_bit_depth(update_fields["bit_depth"])
            update_fields["bit_depth"] = int(update_fields["bit_depth"])
        if "latency_ms" in update_fields:
            _validate_latency(update_fields["latency_ms"])
            update_fields["latency_ms"] = int(update_fields["latency_ms"])

        set_clause = ", ".join([f"{field} = ?" for field in update_fields])
        params = list(update_fields.values()) + [hardware_id]
        with self._connection() as connection:
            with connection:
                cursor = connection.execute(
                    f"""
                    UPDATE hardware_assets
                    SET {set_clause}, updated_at = DATETIME('now', '+8 hours')
                    WHERE hardware_id = ?
                    """,
                    params,
                )
        return cursor.rowcount > 0

    def delete_asset(self, hardware_id):
        self._require_tables()
        with self._connection() as connection:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM hardware_assets WHERE hardware_id = ?",
                    (hardware_id,),
                )
        return cursor.rowcount > 0


def _row_to_dict(row):
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def _validate_required_text(value, field_name):
    if value is None or str(value).strip() == "":
        raise HardwareValidationError(f"{field_name} is required")


def _validate_direction(value):
    if value not in {"input", "output"}:
        raise HardwareValidationError("direction must be input or output")
    return value


def _validate_channel_index(value):
    if isinstance(value, bool):
        raise HardwareValidationError("channel_index must be a non-negative integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise HardwareValidationError("channel_index must be a non-negative integer") from exc
    if normalized < 0:
        raise HardwareValidationError("channel_index must be a non-negative integer")
    return normalized


def _validate_finite_positive(value, field_name):
    if isinstance(value, bool):
        raise HardwareValidationError(f"{field_name} must be a finite positive number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise HardwareValidationError(f"{field_name} must be a finite positive number") from exc
    if not math.isfinite(normalized) or normalized <= 0:
        raise HardwareValidationError(f"{field_name} must be a finite positive number")
    return normalized


def _validate_standard_spl_required(value):
    if value is None or isinstance(value, bool):
        raise HardwareValidationError("standard_spl is required and must be finite")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise HardwareValidationError("standard_spl is required and must be finite") from exc
    if not math.isfinite(normalized):
        raise HardwareValidationError("standard_spl is required and must be finite")
    if normalized.is_integer():
        return int(normalized)
    return normalized


def _validate_output_coefficients(coefficients):
    if not isinstance(coefficients, (list, tuple)):
        raise HardwareValidationError("output coefficients must be a list of two finite numbers")
    if len(coefficients) != 2:
        raise HardwareValidationError("output coefficients must contain exactly two values")
    normalized = []
    for coefficient in coefficients:
        if isinstance(coefficient, bool):
            raise HardwareValidationError("output coefficients must be finite numbers")
        try:
            numeric = float(coefficient)
        except (TypeError, ValueError) as exc:
            raise HardwareValidationError("output coefficients must be finite numbers") from exc
        if not math.isfinite(numeric):
            raise HardwareValidationError("output coefficients must be finite numbers")
        normalized.append(numeric)
    return normalized


def _validate_samplerate(value):
    if value not in VALID_SAMPLE_RATES:
        raise HardwareValidationError("samplerate must be 44100 or 48000")


def _validate_bit_depth(value):
    if value not in VALID_BIT_DEPTHS:
        raise HardwareValidationError("bit_depth must be 8, 16, 24, or 32")


def _validate_latency(value):
    if not isinstance(value, int) or value < 0 or value > 1000:
        raise HardwareValidationError("latency_ms must be an integer from 0 through 1000")
