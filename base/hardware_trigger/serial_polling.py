"""Shared polling configuration for listening and connection tests."""

MIN_POLLING_INTERVAL_MS = 1
MAX_POLLING_INTERVAL_MS = 1200
DEFAULT_POLLING_INTERVAL_MS = 50


def parse_polling_settings(settings):
    if not isinstance(settings, dict):
        raise ValueError("polling_settings 必须是对象")
    enabled = settings.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("polling_settings.enabled 必须为 true 或 false")
    interval_ms = settings.get("interval_ms", DEFAULT_POLLING_INTERVAL_MS)
    if (type(interval_ms) is not int
            or not MIN_POLLING_INTERVAL_MS <= interval_ms <= MAX_POLLING_INTERVAL_MS):
        raise ValueError("polling_settings.interval_ms 必须是 1～1200 的整数（毫秒）")
    query = b""
    if enabled:
        raw_hex = settings.get("query_command_hex", "")
        if not isinstance(raw_hex, str):
            raise ValueError("启用轮询时 query_command_hex 必须是 HEX 字符串")
        try:
            query = bytes.fromhex(raw_hex)
        except ValueError as error:
            raise ValueError("query_command_hex 必须是完整的 HEX 字节") from error
        if not query:
            raise ValueError("启用轮询时 query_command_hex 不能为空")
    return query, interval_ms / 1000.0
