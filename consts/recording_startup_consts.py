"""Stable startup diagnostic vocabulary and per-attempt storage/output limits."""

PARENT_EVENT_BUDGET = 64
CHILD_EVENT_BUDGET = 32
MAX_FIELD_LENGTH = 128
MAX_MARK_FIELDS = 16
MAX_COUNTER = 2**31 - 1

EVENT_BEGIN = "begin"
EVENT_END = "end"
EVENT_SUMMARY = "summary"
EVENT_REQUEST_LINK = "request_link"
EVENT_REJECTED = "rejected"
FIRST_DELIVERED_BLOCK = "first_delivered_block"
FIRST_RETAINED_BLOCK = "first_retained_block"
FIRST_BLOCK_EVENTS = (FIRST_DELIVERED_BLOCK, FIRST_RETAINED_BLOCK)
CONTROL_EVENTS = frozenset((EVENT_BEGIN, EVENT_END, EVENT_SUMMARY, EVENT_REQUEST_LINK))

CONTEXT_FIELDS = (
    "backend", "sample_rate", "channel_count", "target_samples",
    "target_duration_seconds", "startup_trim_samples", "export_mode",
    "generation", "worker_pid", "resource_task_id", "reuse_result",
    "completion_boundary", "start_boundary", "rejection_reason",
)
RESERVED_FIELDS = frozenset(CONTEXT_FIELDS) | frozenset((
    "event", "trace_id", "request_id", "process", "pid", "timestamp_ns",
    "thread_id", "thread_name", "domain", "stage", "parent", "elapsed_ms",
    "total_ms", "stage_ms", "last_stage", "outcome", "error_type",
    "dropped_events", "dropped_fields", "delivery_failures",
    *FIRST_BLOCK_EVENTS,
))

# Shared spelling for the integration sites; only actually executed stages log.
STARTUP_STAGES = (
    "csv_admission", "prechecks", "workflow_prechecks", "entry_ui_cleanup",
    "ui_cleanup", "stream_cleanup", "ui_prepare",
    "process_events", "reset", "device_snapshot", "storage_context", "path",
    "mac_address", "directory", "parameters", "calibration", "request_build",
    "recent_session", "csv_path_permit", "service_start", "worker_ready",
    "command_send", "service_callback", "qt_callback", "worker_validate", "session_build",
    "capture_open", "open_wav", "adapter_create", "device_bind", "stream_start",
)

STARTUP_REJECTION_REASONS = (
    "csv_workflow_busy", "csv_capacity_full", "csv_service_unavailable",
    "csv_reservation_denied", "window_closing", "workflow_prepare_denied",
    "workflow_start_denied", "workflow_busy", "service_busy", "preflight_failed",
    "replay_unavailable", "metadata_preflight_denied", "ve_config_unavailable",
)

# Parent service / Qt event boundaries; shared with the GUI submission path.
EVENT_SERVICE_ACCEPT = "service_accept"
EVENT_SERVICE_SUBMIT = "service_submit"
EVENT_WORKER_SELECTED = "worker_selected"
EVENT_COMMAND_ENQUEUE = "command_enqueue"
EVENT_COMMAND_DEQUEUE = "command_dequeue"
EVENT_PARENT_STARTED = "parent_started"
EVENT_QT_POST = "qt_post"
EVENT_QT_DELIVERY = "qt_delivery"
EVENT_CALLBACK_ENTER = "callback_enter"
EVENT_CALLBACK_RETURN = "callback_return"

# Child-local boundaries; started_sent follows the actual connection.send.
EVENT_WORKER_COMMAND_RECEIVED = "worker_command_received"
EVENT_CAPTURE_THREAD_ENTER = "capture_thread_enter"
EVENT_CAPTURE_STARTED = "capture_started"
EVENT_STARTED_SENT = "started_sent"
