"""Pure TCP transport boundary shared by formal and compatibility analysis paths."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any


class SequenceAnalysisTransportService:
    """Serialize and attempt one analysis-result TCP delivery."""

    def __init__(
        self,
        *,
        tcp_enabled_provider: Callable[[], bool],
        tcp_server_provider: Callable[[], Any],
        logger: Any = None,
    ) -> None:
        self.tcp_enabled_provider = tcp_enabled_provider
        self.tcp_server_provider = tcp_server_provider
        self.logger = logger

    @staticmethod
    def _bounded_error(error: BaseException) -> str:
        try:
            return (str(error) or type(error).__name__)[:384]
        except BaseException:
            return type(error).__name__[:384]

    def _log(self, level: str, message: str) -> bool:
        try:
            callback = getattr(self.logger, level, None)
            if callable(callback):
                callback(str(message)[:560])
        except BaseException:
            return False
        return True

    def send_payload(self, payload: Any) -> bool:
        """Attempt one send; False leaves any formal authorization untouched."""
        try:
            enabled = self.tcp_enabled_provider()
            if not enabled:
                return True
        except BaseException as error:
            self._log(
                "error",
                f"tcp_callback_config_error: {self._bounded_error(error)}",
            )
            return False
        if not isinstance(payload, Mapping):
            return False
        try:
            server = self.tcp_server_provider()
            if server is None:
                self._log("warning", "tcp_callback_skip: no tcp server")
                return False
            message = json.dumps(dict(payload), ensure_ascii=False)
            if server.send_to_current_client(message) is not True:
                self._log("warning", "tcp_callback_skip: no active tcp client")
                return False
            self._log("info", f"tcp_callback_sent: {message[:512]}")
            return True
        except BaseException as error:
            self._log(
                "error",
                f"tcp_callback_send_error: {self._bounded_error(error)}",
            )
            return False
