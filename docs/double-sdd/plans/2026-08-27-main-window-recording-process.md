# Main Window Recording Process Implementation Plan

> **For agentic workers:** REQUIRED: Use the `subagent-driven-development` skill to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move Launcher application capture, monitoring and recording WAV persistence into one independent process while keeping complete historical waveforms, existing workflows and analysis results.

**Architecture:** A persistent spawn worker owns sounddevice streams, bounded capture queues, waveform envelopes and WAV finalization. An instance-owned parent service provides asynchronous IPC, one-flight cumulative previews, background result reads, file leases and failure recovery. Qt adapters deliver session-scoped events to the existing sequence/calibration completion paths without writing audio in the GUI process.

**Tech Stack:** Python 3.12, multiprocessing spawn/Pipe/Event, threading, NumPy, existing sounddevice/soundfile/scipy WAV helpers, PyQt5, pytest.

---

## Authority and execution

Spec: `docs/double-sdd/specs/2026-08-27-main-window-recording-process-design.md` (all sections; A1–A13 are the acceptance matrix).

The user approved the written specification and authorized autonomous subsequent technical decisions, plan execution and safe local delivery while they are asleep. Do not pause at routine plan/execution confirmations. Do not publish remotely or create permanent final commits. Temporary commits support review and are removed/preserved as uncommitted contents during finishing.

Execute sequentially in the isolated worktree supplied by the orchestrator. The orchestrator passes the exact upstream runtime metadata path separately. Never recreate its base or scan for a different active task. One fresh implementer per task; after every implementation pass, fresh parallel spec-code and quality-code reviews of the same task range. `code-review` governs adjudication, and `verification-before-completion` governs evidence. Final aggregate review precedes `finishing-a-development-branch` and apply-back.

Python available on this host: `D:/Python/Python312/python.exe`; `python` currently resolves there. Run all commands from the isolated worktree, with `$env:QT_QPA_PLATFORM='offscreen'` for UI tests. Use repository modules from this worktree, not the original checkout through PYTHONPATH. Runtime configuration/assets may be copied by the orchestrator into the isolated worktree only for testing; never commit those ignored files.

Before implementation run the baseline set in §Verification. Classify existing failures separately; do not rewrite unrelated tests or production code to hide them. No hardware tests may open production devices without explicit selection; use injectable, importable fake backends for automatic process tests.

## File and interface map

| File | Responsibility / ownership |
| --- | --- |
| `base/recording_process_protocol.py` (new) | Picklable requests/events/results and shape/config validation. No Qt or device import. |
| `base/recording_capture.py` (new) | Child-local sounddevice capture, bounded owned blocks, monitor compatibility, file finalization. |
| `base/recording_worker.py` (new) | Top-level spawn target, command/result/preview routing, parent liveness and one-session lifecycle. |
| `base/recording_service.py` (new) | Parent worker lifecycle, session state, IPC threads, deadlines and result file leases. |
| `base/recording_result_reader.py` (new if needed) | Background cancellable result loading/lease cleanup; keep separate if it materially reduces service complexity. |
| `base/streaming_waveform_accumulator.py`, `base/multichannel_waveform_session.py` (new location) | Existing pure reducers relocated without algorithm change; old UI module paths re-export for compatibility. |
| `ui/recording_service_bridge.py` (new) | QObject queued delivery, one preview wakeup, per-session processor façade; no hardware/writer. |
| `main_window_Launcher.py`, `main_window.py` | Safe spawn imports, service ownership/injection, application shutdown. |
| `ui/sequence/sequence_widget.py`, `sequence_widget_analysis_ops.py`, `sequence_widget_streaming_ops.py`, `sequence_widget_ui_ops.py` | Asynchronous ordinary/streaming start, current-session preview/completion/failure, no UI recording writes. |
| `ui/sequence/sequence_widget_serial_trigger_ops.py` | Narrow lease-aware integration for aborted-round file cleanup; preserve serial business behavior. |
| `ui/calibration_window.py` | Same service for input calibration, unchanged calculation/persistence and cancel behavior. |
| `base/play_and_record.py`, `base/streaming_audio_processor.py` | Only narrowly needed compatibility/refactoring; legacy low-level callers may remain, main application must use new service. |
| `unit_test/base/recording_process_fakes.py` (new) | Importable deterministic backend/fake writer for real spawn tests. |
| `unit_test/base/test_recording_capture.py`, `unit_test/base/test_recording_service.py` (new) | Child-core unit and real-process integration tests. |
| `unit_test/ui/test_recording_process_integration.py` (new) | Actual Qt bridge, main/calibration adapter and lifecycle tests. |

Names may be refined only if responsibilities/interfaces remain clear. Do not create a giant copy of StreamingAudioProcessor or put Qt imports in worker dependencies. Existing accumulator code is moved/re-exported, not duplicated.

Suggested typed boundary (specific implementation: use decorated dataclass-based protocol types; payload data is simple protocol dataclass/dict values):

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class RecordingRequest:
    request_id: str
    purpose: str
    sample_rate: int
    target_samples: int
    channels: tuple[int, ...]
    device: dict
    path: str
    streaming: bool
    trim_samples: int
    monitor: dict
    calibration_metadata: dict | None
    validation_thresholds: dict

# Event envelope: protocol version + worker generation + request_id + kind + payload.
# Preview: sequence + sample_stop + physical channels + immutable time/amplitude arrays.
# Result: closed path + raw/final frame counts + sample_rate/channels + metadata status/warnings.
# Service methods: start(request, callbacks), cancel(request_id), accept/reject_result(request_id),
# release_preview(request_id, sequence), shutdown(callback). Every command is asynchronous.
```

Use actual valid dataclasses in production. No QObject, logger, stream, bound Qt method or local lambda enters a process payload. Callbacks live only on parent-side session objects. Test injection is an importable backend factory identifier, validated by the service; production uses the sounddevice backend by default without fallback.

## Task 1: Pure protocol, child capture and waveform/file ownership

**Files:** create protocol/capture and fake backend modules; relocate/re-export the two reducers; add `unit_test/base/test_recording_capture.py`; extend existing monitor/file/trim tests only when appropriate.

**Spec boundaries:** §3, §5.2–5.3, §6.1–6.3, §7.1, §10. No UI integration or parent service in this task.

- [x] **1. Write tests first.** Construct requests with selected channels `(0, 2)`, float32 chunks of unequal sizes and an overlong last block. Require exact ordered output, raw/final counts, trimmed WAV equality and sample-derived mono. Test borrowed driver buffer mutation after callback, queue-full failure, input overflow, monitor gain/mute/fade parity, metadata warning versus WAV failure, calibration no product trim/quality, and no Qt imports in the worker dependency graph.

```python
def test_capture_owns_input_and_trims_once(tmp_path):
    # Use the importable fake backend; known frames and physical columns.
    # Mutate its reusable input buffer immediately after invoking capture callback.
    # Supply startup trim 2 and raw target 9: final WAV is exactly selected[2:9].
    result = run_fake_capture(tmp_path, channels=(0, 2), target=9, trim=2)
    np.testing.assert_array_equal(result.read_audio(), result.expected[2:9])
    assert (result.raw_frames, result.final_frames) == (9, 7)
```

The fixture helper is test code to implement locally, not a production dependency. Use reusable deterministic chunks so future process tests share the same expected arrays.

- [x] **2. Red:** `python -m pytest unit_test/base/test_recording_capture.py -q`. Confirm missing new behavior, not a fixture/import typo.
- [x] **3. Implement protocol and pure reducers.** Use validated frozen request/event objects or equivalent explicit serializers. Move existing reducers to base and re-export names at old UI paths. Keep ≤4000 points, full accumulated time, and unchanged peak ordering/NaN behavior. All state is instance-owned.
- [x] **4. Implement capture.** Snapshot stream configuration, copy callback blocks, cap queue near two seconds by frame/byte accounting, handle overflow as failed. Use child consumer thread for writes and cumulative envelopes, never IPC inside audio callback. Reuse current monitor math rather than independently changing the fade algorithm. Distinguish effective streaming as OR of stream/monitor flags.
- [x] **5. Implement finalization.** Close stream, drain accepted blocks, close WAV, apply trim once, run existing quality validator, append metadata using existing warning contract, emit result only after handles close. Calibration uses parent-allocated temp file with no product quality/trim. Record exact path ownership; cancellation finalizes accepted main audio without successful completion. Errors carry session/stage/path and cannot become empty successful arrays.
- [x] **6. Green:** `python -m pytest unit_test/base/test_recording_capture.py unit_test/base/test_streaming_file_writer.py unit_test/base/test_recording_startup_trim.py unit_test/ui/test_streaming_waveform_accumulator.py unit_test/ui/test_multichannel_waveform_session.py -q`. Add monitor parity test command for whichever existing suite is extended.
- [x] **7. Temporary checkpoint:** `git add -- base unit_test ui/sequence/streaming_waveform_accumulator.py ui/sequence/multichannel_waveform_session.py`; inspect staged scope; `git commit -m "feat: isolate recording capture and file finalization core"`. Orchestrator records SHA and runs paired reviews before Task 2.

## Task 2: Real spawn service, bounded preview and file-lease lifecycle

**Files:** create `base/recording_worker.py`, `base/recording_service.py` and result-reader module if needed; extend protocol/fakes; add `unit_test/base/test_recording_service.py`.

**Spec boundaries:** §4–8, §9–10. Real spawn mandatory; do not postpone timeout/cancel/death behavior to UI integration.

- [x] **1. Tests first.** Create a real spawn service using an importable fake backend with deterministic audio and controlled events. Assert `parent_pid != capture_pid == writer_pid`. Healthy worker is reused. Pause preview acknowledgement while capture/file frame count continues. Verify one in-flight snapshot and next cumulative snapshot includes known paused-interval peaks; completed bypasses preview credit.
- [x] **2. Red:** `python -m pytest unit_test/base/test_recording_service.py -q`.
- [x] **3. Worker.** Top-level no-Qt target, separate control/preview channels and dedicated serialized sendwriters, explicit ready/started/completed/failed/cancelled; parent OS liveness monitor (not GUI heartbeat). Capture core runs independently of a blocked pipe. Preview credit prevents unbounded backlog; result_ack ends delivery busy. Parent-gone cleanup bounded and testable.
- [x] **4. Parent service.** Instance owns generation, one active request, async command writer and receiver. Register session before start, validate IDs/sequences, deduplicate terminal results and inspect process sentinel. Add injectable ready/start=10s, cancel/shutdown=5s and terminate=2s deadlines. On timeout retire worker and IPC; do not start another until confirmed old PID death. No sleeps/joins/I/O on GUI caller path.
- [x] **5. Result reader and leases.** Only read after child closes writers. Background chunked reader owns file lease, returns independent float32 arrays and validates expected frames/channels. `accept_result` follows parent semantic acceptance; reject/cancel defers ack/cleanup until reader closes. Cancellation timeout isolates its path and retires worker; new different path allowed after old PID exit. Old reader callbacks can only release old leases. Parent owns calibration temp directory and never deletes someone else's path.
- [x] **6. Add deterministic race tests.** Kill worker during recording/finalizing/pending delivery; inject duplicate/late messages; block native start/cancel; pause reader and cancel/close/restart same path; assert no early ack/delete/reuse. Verify no successful callback after failure/cancel, manual next recording uses new generation, calibration temp cleanup and parent process death. Use waitable events and deadlines rather than arbitrary long sleeps.
- [x] **7. Green:** `python -m pytest unit_test/base/test_recording_capture.py unit_test/base/test_recording_service.py -q`. All spawned helpers must be joined/closed in test finalizers. Check no process remains after suite.
- [x] **8. Checkpoint:** stage only Task 2 files and tests; `git commit -m "feat: add asynchronous recording subprocess service"`; paired reviews before Task 3.

## Task 3: Launcher, Qt bridge and main recording integration

**Files:** create `ui/recording_service_bridge.py`; modify Launcher, MainWindow, sequence widget and relevant mixins (including the serial-trigger cleanup path); adjust existing UI streaming/serial cleanup tests; create `unit_test/ui/test_recording_process_integration.py`. A dedicated recording-process mixin may keep the new adapter responsibility separate from the existing large streaming mixin.

**Integration clarification from call-path inspection:** `_abort_serial_product_round` currently calls `RecordingManager.delete_audio` after synchronous cleanup, and `_relabel_stored_audio_record` moves WAV files. With asynchronous recording these paths must honor service leases, including pending readers. Do not queue a bare deletion after releasing a path to reuse: retain the exact session's cleanup authority. The service's `result_ready` callback is provisional; only `accepted` permits authoritative state, database and analysis publication. `released` is distinct and authorizes path reuse after cleanup. This clarifies existing spec §§6.3–6.4 and §8; it does not change product behavior or add unrelated recording-manager refactoring.

**Spec boundaries:** main application portions of §3, §5.1, §6.3–6.4, §7–10. Input calibration remains Task 4, but inject service ownership now.

- [x] **1. Tests first.** Actual QApplication/bridge tests assert callbacks run on GUI thread while capture/writer run in child. Main route covers four streaming/monitor combinations, channel/device/config snapshots, rapid completion registration, stale preview, no double trim/WAV writes, no successful completion on failed/cancelled. Existing private tests expecting UI writer construction should be migrated to new boundary rather than preserving dead writing code.
- [x] **2. Red:** `python -m pytest unit_test/ui/test_recording_process_integration.py unit_test/ui/test_streaming_recording_config.py -q`.
- [x] **3. Safe entry and ownership.** `freeze_support()` before Qt startup, import UI only inside main startup path, create one RecordingService owner and bridge, pass explicitly to MainWindow/SequenceWindow. Keep optional construction support for existing isolated UI unit tests without opening a real device. No module-global service fallback. Imported Launcher in spawned child must not import/create application/splash.
- [x] **4. Asynchronous start.** `judge_play_and_record` keeps validation/locks/state, prepares request, begins active session and calls service. Ordinary mode disables preview but no longer blocks Qt. Flow remains busy until completion; started/failed route existing button/serial error state. Keep channel and calibration snapshots for entire round.
- [x] **5. Preview and completion.** Bridge coalesces a single pending preview wakeup and releases credit after consumed/ignored. Render received full envelopes using physical channel mapping; never append snapshot as raw audio. On completed read/validation, reuse existing business portion of `_on_streaming_complete` with finalized arrays: no UI WAV writer, no recrop, no metadata append. Child optional metadata warnings are logged once. Final workspace semantic validation precedes authoritative arrays; rendering failures remain presentation-only. Accept/reject result ownership exactly once.
- [x] **6. Cleanup.** Cancel active request via service; invalidate generation/queuedview, use async result-release continuation before same-path replay or file moves. Ordinary/streaming error paths restore existing barcode/channel/buttons and serial round state. Guard startup hidden SequenceWindow.close from destroying application service. Main actual close waits asynchronously for shutdown before final acceptance, retaining Excel/PDF cleanup behavior.
- [x] **7. Green:** `python -m pytest unit_test/ui/test_recording_process_integration.py unit_test/ui/test_streaming_recording_config.py unit_test/ui/test_streaming_event_dispatch.py unit_test/ui/test_streaming_waveform_runtime.py unit_test/ui/test_sequence_wav_calibration_metadata.py unit_test/ui/test_multichannel_recording_workspace.py unit_test/base/test_recording_storage_consistency.py unit_test/base/test_recording_paths.py -q`.

For this step use the actual existing paths in §Verification. Keep old direct helper behavior for non-main callers only where still in use; no frozen global Qt signals in the worker core. Do not migrate unrelated imports/workflows.
- [x] **8. Checkpoint:** `git commit -m "feat: connect main application recording to isolated worker"` after exact staging; paired reviews before Task 4.

## Task 4: Input calibration and complete shutdown integration

**Files:** `ui/calibration_window.py`, bridge/service integration refinements, `main_window.py` injection/exit refinements, existing calibration suites, process integration tests.

**Spec boundaries:** remaining §5.3, §6.3–6.4, §8 and A7/A10/A13.

- [x] **1. Tests first.** Input calibration records exactly 10 seconds single selected physical channel in actual fake child, unchanged SPL/v2pa/JSON and next-channel behavior. Close/reset/cancel releases session; delayed results cannot save JSON. Busy service never silently steals main recording. Calibration-only temp files never enter user recording tree or database.
- [x] **2. Red:** `python -m pytest unit_test/test_input_calibration_runtime.py unit_test/ui/test_recording_process_integration.py -q`.
- [x] **3. Wire capture only.** Replace active input calibration call to legacy `stream_record_without_play` with supplied bridge/service, preserving calculation and persistence. Migrate legacy queue-ready consumers to façade/event completion without real capture in GUI. Output calibration playback helpers and unused direct low-level recording calls remain out of scope unless reachable from active input UI.
- [x] **4. Lifecycle tests and fixes.** Cancel during starting/recording/delivering, calibrator close while reader paused, then new main session. Guard IDs and path leases. Actual QApplication aboutToQuit/MainWindow.close share idempotent service shutdown; native fake hangs require bounded terminate and no orphan. Child parent-liveness test launches an intermediate parent process, terminates it, and checks child exit.
- [x] **5. Green:** `python -m pytest unit_test/test_input_calibration_runtime.py unit_test/ui/test_input_calibration.py unit_test/base/test_streaming_calibration_queue.py unit_test/ui/test_recording_process_integration.py unit_test/base/test_recording_service.py -q`.
- [x] **6. Checkpoint:** `git commit -m "feat: isolate input calibration and application recording shutdown"`; paired reviews before Task 5.

## Task 5: Whole-workflow regression, resource checks and acceptance evidence

**Files:** existing main workflow/serial/recording tests as needed, new process test files, `docs/double-sdd/verification/2026-08-27-main-window-recording-process.md` (new). Targeted production corrections only for confirmed integration defects; no unrelated refactor.

- [x] **1. Add missing real-process acceptance coverage.** A1–A13 checklist must map to tests/commands, including 1s and 30s stopped preview consumption, 600s simulated accepted audio, multiple channels, ordinary and streaming, failed-write/noDB, stale completion, per-start/cancel deadlines, result-reader path race and healthy-worker reuse. The 600s input may be synthetic faster than wall clock, explicitly labeled; it is not a hardware throughput claim.
- [x] **2. Red for any missing behavior.** Extend natural existing fixtures, run the selected failing test, fix through implementation, rerun. Existing unrelated baseline failures are recorded, not hidden by broad test skips.
- [x] **3. Run focused and broad verification below.** Verify all relevant regressions, note exact counts and failures. Inspect helper process/reader thread cleanup, `git diff --check`, compile touched Python. Run full `python -m pytest unit_test -q` if baseline/environment allows. If it includes unrelated hardware/manual paths, document and run precise safe suites without declaring the full suite passed.
- [x] **4. Evidence document.** Record actual PID separation, file/sample/preview comparison results, timeout recovery and commands with exit status. Clearly distinguish fake-device process integration from actual device smoke. Record hardware availability without automatically recording from a real device. `record_vkinging_continuous.py` must be unchanged; no credentials/config/assets committed.
- [x] **5. Checkpoint:** `git commit -m "test: verify recording process isolation and workflow compatibility"`; paired reviews, then final whole-range quality review. Update checklist and evidence only to reflect actual results.

Task 5 execution evidence is recorded in `docs/double-sdd/verification/2026-08-27-main-window-recording-process.md`: focused gate 278 passed; supplemental safe suites 270 passed; business suite 112 passed plus one unchanged baseline key failure. Collection confirms two unrelated missing modules, so the full suite was not declared passing. The task also adds deterministic synthetic-writer pacing and the spec §6.4 main retained-resource notice. All five tasks passed paired spec/quality review, followed by a passing whole-implementation quality review. The orchestrator applied the result to the original checkout as uncommitted changes and reran the focused gate (278 passed, 139.67s) and supplemental suites (270 passed, 9.35s) on 2026-08-28. See the verification report for limitations and final delivery evidence.

## Verification commands

Baseline (before implementation, existing files only):

```powershell
$env:QT_QPA_PLATFORM='offscreen'
python -m pytest unit_test/base/test_streaming_file_writer.py unit_test/base/test_recording_startup_trim.py unit_test/ui/test_streaming_waveform_accumulator.py unit_test/ui/test_multichannel_waveform_session.py unit_test/ui/test_streaming_recording_config.py unit_test/ui/test_streaming_event_dispatch.py unit_test/test_input_calibration_runtime.py -q
```

Focused final:

```powershell
python -m pytest unit_test/base/test_recording_capture.py unit_test/base/test_recording_service.py unit_test/ui/test_recording_process_integration.py unit_test/ui/test_streaming_recording_config.py unit_test/ui/test_streaming_event_dispatch.py unit_test/ui/test_streaming_waveform_runtime.py unit_test/ui/test_sequence_wav_calibration_metadata.py unit_test/test_input_calibration_runtime.py -q
python -m pytest unit_test/test_multichannel_support.py unit_test/test_manual_product_condition_cycle.py unit_test/test_serial_product_condition_runtime.py unit_test/test_serial_product_round_cleanup.py unit_test/test_product_round_barcode_lock.py unit_test/base/test_recording_storage_consistency.py unit_test/base/test_recording_paths.py -q
python -m pytest unit_test -q
git diff --check
```

Expected for task-specific commands: all selected tests pass with no lingering helper process. Full suite must be reported honestly if baseline or environmental failures remain. New async tests should use event handshakes and injected short deadlines, bounded wait with diagnostic state, and `finally` cleanup. Test fake backends must not call sounddevice.query/open against live hardware.

## Review constraints and delivery

For every task reviewers must distinguish current-slice responsibilities from later tasks. Worker/service slices need real core behavior, not app wiring before Task 3. Later tasks must not use legacy fallback to make tests pass. Review exception boundaries (no duplicate broad catches, no success from lost audio) and instance/global-state rules from §10 of spec. New stable serialized fields may be protocol-local; shared constants belong in `consts` only when actually shared.

After final verification use `finishing-a-development-branch`. Apply final code back to the original checkout as uncommitted changes, preserving its ignored configurations and standalone recording scripts; reconcile already-identical spec/plan copies without overwriting user changes. Temporary checkpoints are not permanent final commits. Never push, open PR, delete unrelated worktrees or force cleanup with unverified paths. If runtime/hardware prevents a required real-device smoke, leave an explicit limitation rather than a false completion claim.
