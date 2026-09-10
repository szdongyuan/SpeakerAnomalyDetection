"""Exact-session capture state retained until GUI publication enters the queue."""
from dataclasses import dataclass, field


@dataclass
class RecordingProcessContext:
    request: object
    direction: str
    tcp_completion: object
    preview_enabled: bool
    session: object = None
    session_binding_pending: bool = False
    processor: object = None
    final: bool = False
    failed: bool = False
    cancelled: bool = False
    validated_audio: object = None
    accepted_audio: object = None
    final_windows: object = None
    release_warned: bool = False
    sequence: int = 0
    cleanup_owned: bool = False
    release_deadline: float = 0.0
    recent_session_id: str = ""
    recorded_signal_info: dict = field(default_factory=dict)
    workflow_token: object = None
    publication_started: bool = False
    publication_delivered: bool = False
