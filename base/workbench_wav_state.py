"""Process-local ledger for saved WAV identities and first terminal results."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from base.audio_label_policy import aggregate_judgements


@dataclass
class _SavedWav:
    recorded_at: datetime
    station_number: int
    file_name: str
    condition: str
    result: str = "未标记"
    terminal_received: bool = False


class WorkbenchWavState:
    """Accept authenticated upstream identities without inspecting WAV files."""

    def __init__(self):
        self._records = {}

    def register_recording(self, request_id, recorded_at, condition, stations):
        """Register new (request, source station) identities in arrival order."""
        added = False
        for station_id, station_number, path in stations:
            identity = request_id, station_id
            if identity not in self._records:
                self._records[identity] = _SavedWav(
                    recorded_at, station_number, Path(path).name, condition,
                )
                added = True
        return added

    def update_result(self, request_id, station_id, items):
        """Accept a known WAV's first terminal result, including no judgement."""
        record = self._records.get((request_id, station_id))
        if record is None or record.terminal_received:
            return False
        result = aggregate_judgements(item.get("judgement") for item in items)
        record.result = "未标记" if result == "--" else result
        record.terminal_received = True
        return True

    def records_for(self, day):
        """Export independent display dictionaries for the WAV's frozen date."""
        return [
            dict(
                completed_at=record.recorded_at,
                station_id=record.station_number,
                file_name=record.file_name,
                condition=record.condition,
                result=record.result,
            )
            for record in self._records.values()
            if record.recorded_at.date() == day
        ]
