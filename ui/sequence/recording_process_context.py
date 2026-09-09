"""Request-owned state for recording callbacks that may overlap in the UI."""
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
    product_condition_key: str = ""
    product_group_id: str = ""
    publication_group_id: str = ""
    product_condition_keys: tuple = ()
    manual_product_cycle_active: bool = False
    serial_product_condition_executing: bool = False
    directional_cycle_active: bool = False
    count_mode: str = ""
    barcode: str = ""
    analysis_result_dict: dict = field(default_factory=dict)
    recent_session_config_snapshot: dict = field(default_factory=dict)
    enabled_analysis_identifiers: tuple = ()
    analysis_required: bool = False
    analysis_executor: object = None
    analysis_label: str = ""
    analysis_diagnostics: tuple = ()
    analysis_items_data: dict = field(default_factory=dict)
    stimulus_info: dict = field(default_factory=dict)
    stimulus_signal: object = None
    product_report_config: dict = field(default_factory=dict)
    condition_record_cache: dict = field(default_factory=dict)
    awaiting_ok_ng: bool = False
    sn_clear_on_next_scan: bool = False
    pending_recent_session_append: bool = False
    product_completion_result: object = None
    cycle_completion_result: object = None
    business_completed: bool = False
    business_failure: str = ""
    business_effects_published: bool = False
    business_effects_result: dict = field(default_factory=dict)
    publication_started: bool = False
    publication_delivered: bool = False
    publication_attempts: int = 0
    publication_max_attempts: int = 3
    publication_retry_deadline: float = 0.0
    publication_retry_pending: bool = False
    publication_submission_failures: int = 0
    publication_max_submission_failures: int = 3
    publication_submission_retry_pending: bool = False
    publication_submit_status: str = ""
    publication_executor_generation: int = -1
    publication_reservation_executor: object = None
    publication_reservation_active: bool = False
    publication_delivery_finalizing: bool = False
    publication_recent_sessions_snapshot: dict = field(default_factory=dict)
    publication_owner_snapshot_frozen: bool = False
    active_transition_applied: bool = False
    count_visual_refresh: str = ""
    publication_audio: object = None
    publication_sample_rate: int = 0
    publication_final_windows: object = None
    business_effect_ledger: dict = field(default_factory=dict)
    business_effect_attempts: dict = field(default_factory=dict)
    pending_ui_session_record: object = None
