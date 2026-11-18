"""Validated, atomic persistence for the workbench's daily home summary."""

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from base.motor_daily_records import DailyRecordSummary
from consts.running_consts import DEFAULT_DIR


_SCHEMA_VERSION = 1


def _reject_non_json_constant(token: str) -> None:
    raise ValueError(f"invalid JSON constant: {token}")


@dataclass(frozen=True)
class WorkbenchSummarySnapshot:
    day: date
    ok: int
    ng: int
    unmarked: int

    def __post_init__(self):
        if type(self.day) is not date:
            raise ValueError("date must be a local calendar date")
        for field, count in (("ok", self.ok), ("ng", self.ng), ("unmarked", self.unmarked)):
            if type(count) is not int or count < 0:
                raise ValueError(f"{field} must be a non-negative integer")

    @property
    def summary(self) -> DailyRecordSummary:
        return DailyRecordSummary.from_counts(self.ok, self.ng, self.unmarked)


class WorkbenchSummaryStateError(Exception):
    """The daily summary could not be loaded or committed."""


class WorkbenchSummaryStore:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else self.default_path()

    @staticmethod
    def default_path() -> Path:
        return Path(DEFAULT_DIR) / "ui/ui_config/workbench_summary_state.json"

    def load(self) -> WorkbenchSummarySnapshot | None:
        error_message = f"Cannot load workbench summary state at {self.path}"
        try:
            try:
                stream = self.path.open(encoding="utf-8")
            except FileNotFoundError:
                return None
            with stream:
                text = stream.read()
        except (OSError, UnicodeError) as exc:
            raise WorkbenchSummaryStateError(f"{error_message}: {exc}") from exc

        try:
            data = json.loads(text, parse_constant=_reject_non_json_constant)
        except (ValueError, RecursionError) as exc:
            raise WorkbenchSummaryStateError(f"{error_message}: {exc}") from exc

        try:
            if not isinstance(data, dict):
                raise ValueError("state must be a JSON object")
            try:
                version = data["schema_version"]
                day = data["date"]
                ok, ng, unmarked = data["ok"], data["ng"], data["unmarked"]
            except KeyError as exc:
                raise WorkbenchSummaryStateError(f"{error_message}: {exc}") from exc

            if type(version) is not int or version != _SCHEMA_VERSION:
                raise ValueError("unsupported schema_version")

            if not isinstance(day, str):
                raise ValueError("date must be an ISO calendar date")
            parsed_day = date.fromisoformat(day)
            if parsed_day.isoformat() != day:
                raise ValueError("date must use YYYY-MM-DD format")

            return WorkbenchSummarySnapshot(parsed_day, ok, ng, unmarked)
        except ValueError as exc:
            raise WorkbenchSummaryStateError(f"{error_message}: {exc}") from exc

    def save(self, snapshot: WorkbenchSummarySnapshot) -> None:
        if not isinstance(snapshot, WorkbenchSummarySnapshot):
            raise TypeError("snapshot must be a WorkbenchSummarySnapshot")
        error_message = f"Cannot save workbench summary state at {self.path}"
        state = {
            "schema_version": _SCHEMA_VERSION,
            "date": snapshot.day.isoformat(),
            "ok": snapshot.ok,
            "ng": snapshot.ng,
            "unmarked": snapshot.unmarked,
        }
        try:
            text = json.dumps(state, ensure_ascii=False, indent=2)
        except ValueError as exc:
            # Python can reject conversion of integers above its digit limit.
            raise WorkbenchSummaryStateError(f"{error_message}: {exc}") from exc

        temporary_path = None
        diagnostics = []
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            stream = tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent,
                prefix=f".{self.path.name}.", suffix=".tmp", delete=False,
            )
            try:
                temporary_path = Path(stream.name)
                stream.write(text)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                # Close before cleanup while preserving any error from writing or syncing.
                try:
                    stream.close()
                except (OSError, UnicodeError) as close_error:
                    diagnostics.append(f"Could not close temporary file {temporary_path}: {close_error}")
                raise
            stream.close()
            os.replace(temporary_path, self.path)
        except BaseException as exc:
            # Discard an uncommitted file even when an unexpected error must propagate.
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    diagnostics.append(f"Could not remove temporary file {temporary_path}: {cleanup_error}")

            if isinstance(exc, (OSError, UnicodeError)):
                message = f"{error_message}: {exc}"
                raise WorkbenchSummaryStateError(". ".join([message, *diagnostics])) from exc
            for detail in diagnostics:
                exc.add_note(detail)
            raise
