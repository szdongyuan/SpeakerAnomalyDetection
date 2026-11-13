"""Validated, atomic persistence for the workbench's daily round number."""

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from consts.running_consts import DEFAULT_DIR


@dataclass(frozen=True)
class RoundSnapshot:
    day: date
    number: int


class WorkbenchRoundStateError(Exception):
    """A round-state operation failed without committing a new snapshot."""


def _validate_snapshot(snapshot):
    if not isinstance(snapshot, RoundSnapshot):
        raise ValueError("state must be a RoundSnapshot")
    if type(snapshot.day) is not date:
        raise ValueError("date must be a local calendar date")
    if type(snapshot.number) is not int or snapshot.number < 1:
        raise ValueError("round_number must be a positive integer")


class WorkbenchRoundStore:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else self.default_path()

    @staticmethod
    def default_path():
        return Path(DEFAULT_DIR) / "ui/ui_config/workbench_round_state.json"

    def load(self) -> RoundSnapshot | None:
        try:
            with self.path.open(encoding="utf-8") as stream:
                data = json.load(stream)
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise WorkbenchRoundStateError(
                f"Cannot load workbench round state at {self.path}: {exc}"
            ) from exc

        try:
            if not isinstance(data, dict):
                raise ValueError("state must be a JSON object")
            if type(data["schema_version"]) is not int or data["schema_version"] != 1:
                raise ValueError("unsupported schema_version")
            day = data["date"]
            if not isinstance(day, str):
                raise ValueError("date must be an ISO calendar date")
            parsed_day = date.fromisoformat(day)
            if parsed_day.isoformat() != day:
                raise ValueError("date must use YYYY-MM-DD format")
            snapshot = RoundSnapshot(parsed_day, data["round_number"])
            _validate_snapshot(snapshot)
        except (ValueError, KeyError) as exc:
            raise WorkbenchRoundStateError(
                f"Cannot load workbench round state at {self.path}: {exc}"
            ) from exc
        return snapshot

    def save(self, snapshot: RoundSnapshot):
        try:
            _validate_snapshot(snapshot)
        except ValueError as exc:
            raise WorkbenchRoundStateError(
                f"Cannot save workbench round state at {self.path}: {exc}"
            ) from exc

        temporary_path = None
        try:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=self.path.parent,
                    prefix=f".{self.path.name}.", suffix=".tmp", delete=False,
                ) as stream:
                    temporary_path = Path(stream.name)
                    json.dump(
                        {
                            "schema_version": 1,
                            "date": snapshot.day.isoformat(),
                            "round_number": snapshot.number,
                        },
                        stream, ensure_ascii=False, indent=2,
                    )
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_path, self.path)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
        except OSError as exc:
            raise WorkbenchRoundStateError(
                f"Cannot save workbench round state at {self.path}: {exc}"
            ) from exc
