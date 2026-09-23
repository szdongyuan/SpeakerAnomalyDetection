"""Small immutable values shared by the CSV ledger, service and worker."""

from dataclasses import dataclass

from consts.raw_audio_csv_consts import RAW_AUDIO_CSV_PROTOCOL_VERSION


@dataclass(frozen=True)
class CsvReservation:
    token_id: str
    owner_id: str


@dataclass(frozen=True)
class CsvAdmission:
    status: str
    reservation: CsvReservation | None = None


@dataclass(frozen=True)
class CsvExportRequest:
    task_id: str
    recording_id: str
    wav_path: str
    csv_path: str
    raw_channels: tuple[int, ...]
    owner_group: str
    owner_record: str


@dataclass(frozen=True)
class CsvLedgerSnapshot:
    phase: str
    capacity: int
    reserved: int
    queued: int
    active: int

    @property
    def outstanding(self) -> int:
        return self.reserved + self.queued + self.active


@dataclass(frozen=True)
class CsvTaskSnapshot:
    request: CsvExportRequest
    state: str
    generation: int | None = None
    terminal_status: str | None = None
    cleanup_diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class CsvMutationPermit:
    owner_id: str
    paths: tuple[str, ...]


@dataclass(frozen=True)
class CsvExportCommand:
    request: CsvExportRequest
    generation: int
    temporary_path: str
    zip_temporary_path: str
    protocol_version: int = RAW_AUDIO_CSV_PROTOCOL_VERSION

    @property
    def task_id(self) -> str:
        return self.request.task_id


@dataclass(frozen=True)
class CsvResult:
    """CSV metrics describe the uncompressed export; archive_path is the saved ZIP."""
    task_id: str
    generation: int
    csv_path: str
    worker_pid: int
    elapsed_seconds: float
    frames: int
    bytes_written: int
    archive_path: str
    archive_bytes: int
    csv_retained: bool
    cleanup_diagnostics: tuple[str, ...]
    csv_export_seconds: float
    zip_write_seconds: float
    zip_verify_seconds: float
    zip_publish_seconds: float
    csv_cleanup_seconds: float
    export_begin_seconds: float | None = None
    export_end_seconds: float | None = None


@dataclass(frozen=True)
class CsvFailure:
    task_id: str
    generation: int
    stage: str
    exception_type: str
    message: str
    cleanup_diagnostics: tuple[str, ...] = ()
    worker_pid: int | None = None
    export_begin_seconds: float | None = None
    export_end_seconds: float | None = None


@dataclass(frozen=True)
class CsvTiming:
    """Parent perf_counter observations; child timestamps never enter this clock."""
    stage: str
    parent_seconds: float
    recording_id: str = ""
    task_id: str = ""
    token_id: str = ""
    generation: int | None = None
    worker_pid: int | None = None
    worker_ready_seconds: float | None = None


@dataclass(frozen=True)
class CsvServiceEvent:
    kind: str
    snapshot: CsvLedgerSnapshot
    task: CsvTaskSnapshot | None = None
    result: CsvResult | CsvFailure | None = None
    detail: str = ""
    timing: CsvTiming | None = None
    qt_delivery_seconds: float | None = None
