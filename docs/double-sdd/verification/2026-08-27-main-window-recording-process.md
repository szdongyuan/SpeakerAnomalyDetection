# Recording process verification evidence

Date: 2026-08-28. Task 5 implementer, isolated worktree recording-process-20260827, starting checkpoint 6f070366f0a0a992ddaa2b460c8da18e3ea19220.

## Scope and environment

Windows source execution, Python 3.12 at D:/Python/Python312/python.exe, Qt offscreen. Capture tests use the importable process_dependencies fake backend and actual multiprocessing spawn, pipes, WAV writer, reader and Qt callbacks. No sound device or port was opened for verification. Synthetic duration is not a hardware throughput measurement. No packaged executable or actual sound-card smoke is claimed.

All pytest commands below ran from D:/dongyuan_ditting/SpeakerAnomalyDetection-MW/.worktrees/codex/recording-process-20260827 with QT_QPA_PLATFORM=offscreen. Each run used a fresh, nonexistent basetemp directly under the Windows system temporary directory; default pytest temporary-directory ACL failures were not worked around by modifying permissions. Reproduce a command with this prefix/suffix:

    $env:QT_QPA_PLATFORM='offscreen'
    & 'D:/Python/Python312/python.exe' -m pytest <arguments> --basetemp (Join-Path ([IO.Path]::GetTempPath()) ('recording-verify-' + [guid]::NewGuid()))

During Task 5 implementation, the original checkout, metadata JSON, runtime configurations, assets and standalone recording scripts were not edited. Test WAVs and calibration JSON use temporary paths. main_host now supplies an explicit temporary recording_root; the integration fixture isolates the DB and checks relabel destinations before moving. Fixture teardown rejects even a caught unsafe-move attempt. The pre-existing ignored audio_data/stored_data/OK/main.wav artifact was inspected, left untouched and excluded from delivery.

## Task 5 changes and RED/GREEN evidence

| Change | RED actually observed | GREEN actually observed |
| --- | --- | --- |
| Synthetic producer scheduling | Service filter synthetic_producer_writer_handshake: 1 passed / 1 failed, 3.43s. Held first write plus unpaced seven 16000-frame callbacks reproduced queue saturation; the paced case lacked the writer-consumption handshake. | Same service filter plus calibrator_close_with_paused_reader: 3 passed, 9.63s. Generated audio now waits for each write with a bounded, cancel-aware handshake. Production queue capacity and overflow handling are unchanged; the deliberately unpaced saturation case remains. |
| Main retained-resource notice | Integration filter cleanup_error_after_acceptance: 1 failed, 6.41s; actual denied service cleanup produced no QMessageBox warning. | 1 passed, 6.13s. The existing release_failed callback and bounded fallback now report once, retain the lease and preserve accepted samples/business publication. Notice/lifecycle filter: 6 passed, 8.96s, including duplicate, serial abort, close, new session and modal-close reentrancy. |
| 600-second acceptance synchronization | Service filter spawn_accepts_600: 2 failed, 21.04s, because the requested finalization release gate did not exist. An initial coverage attempt passed incidentally during finalization; it was replaced with an explicit gate so correctness does not depend on timing. | Service filter 'spawn_accepts_600 or paused_preview_credit': 4 passed, 52.33s. The fixture finalization gate is bounded and released in finally. No capture/service production change was necessary. |
| Test recording-root isolation | Integration filter main_host_relabel_destination: expected temporary destination instead resolved to the worktree's default audio_data/stored_data/OK/main.wav. Combined coverage run: 1 failed / 2 passed, 26.23s. | Path/isolation filter: 2 passed, 6.17s. Explicit temporary recording root and DB, plus a pre-move guard; production path rules were not changed. |
| Real child disk-write failure | Integration filter 'tcp_finish_is_not_sent and write_failed': 1 failed, 6.95s; absent fixture fault allowed success/DB publication. | Same filter: 1 passed, 6.43s. Injected write failure reaches the existing failure path exactly once, with no DB, analysis or TCP finish, and its failed file is removed. |

The existing 1s/30s pause test was strengthened without production changes: compare every final multi/mono sample and the real WAV, not only shape. This is added characterization coverage, not a claim of a newly fixed audio corruption defect. Local refactoring only simplified the diagnostic trace predicate.

## Commands and results

### Focused task-specific gate

    python -m pytest unit_test/base/test_recording_capture.py unit_test/base/test_recording_service.py unit_test/ui/test_recording_process_integration.py unit_test/ui/test_streaming_recording_config.py unit_test/ui/test_streaming_event_dispatch.py unit_test/ui/test_streaming_waveform_runtime.py unit_test/ui/test_sequence_wav_calibration_metadata.py unit_test/test_input_calibration_runtime.py -q

Initial run: 277 passed, 19 dependency deprecation warnings, 137.57s, exit 0. This preceded the added write_failed parameter. Final rerun: **278 passed**, 19 dependency deprecation warnings, **136.03s, exit 0**. This is the task-specific passing gate, separate from the known business/full-suite baseline failures.

After strengthening fixture teardown to reject even caught out-of-root move attempts, the complete affected integration suite was rerun:

    python -m pytest unit_test/ui/test_recording_process_integration.py -q

73 passed, 19 dependency deprecation warnings, 43.98s, exit 0. There were no unsafe relabel attempts and no test-owned service threads left alive at teardown.

### Existing business regressions, including the known baseline failure

    python -m pytest unit_test/test_multichannel_support.py unit_test/test_manual_product_condition_cycle.py unit_test/test_serial_product_condition_runtime.py unit_test/test_serial_product_round_cleanup.py unit_test/test_product_round_barcode_lock.py unit_test/base/test_recording_storage_consistency.py unit_test/base/test_recording_paths.py -q

112 passed / 1 failed, 18 dependency warnings, 6.80s, exit 1. The sole failure is TestManualProductConditionCycle.test_manual_product_condition_runtime_key_prefers_trigger_state: expected 01/02/03, actual uuid_6000/uuid_7000/uuid_8000. No code or expectation was changed to make this unrelated business rule pass. The orchestrator separately reproduced it in the ORIGINAL checkout at a2409cd (business code e8990f): 1 failed / 23 deselected, 6.22s, before apply-back. Its independent current business run also had 112 passed / 1 baseline failure, 6.50s. These are reported orchestrator results, not commands executed by this implementer.

### Supplemental safe regressions

    python -m pytest unit_test/base/test_streaming_file_writer.py unit_test/base/test_recording_startup_trim.py unit_test/base/test_streaming_calibration_queue.py unit_test/base/test_wav_calibration_metadata.py unit_test/base/test_recording_channel_selection.py unit_test/base/test_recording_calibration_snapshot.py unit_test/base/test_file_ops.py unit_test/ui/test_streaming_waveform_accumulator.py unit_test/ui/test_multichannel_waveform_session.py unit_test/ui/test_multichannel_recording_workspace.py unit_test/ui/test_input_calibration.py -q

270 passed, 18 dependency deprecation warnings, 9.00s, exit 0.

### Full-suite collection limitation

    python -m pytest unit_test/test_main.py unit_test/test_vkinging_continuous_recorder.py --collect-only -q

No tests collected, 2 collection errors, 0.62s, exit 1: missing main module and the intentionally un-copied ignored record_vkinging_continuous module. Consequently the full python -m pytest unit_test -q command was not run or claimed passing. No standalone recorder was copied/imported from the original checkout to evade this limitation.

Other baseline limitations supplied by the orchestrator: legacy THD tests mock removed calculate_thd/get_harmonic APIs (8 failed / 24 passed); broad UI execution has a pre-existing native abort at test_curve_color_config.py:222 and a stimulus point-count expectation of 102 versus actual 70. These unrelated failures were not edited, skipped inside tests or re-executed through unsafe broad UI/hardware paths.

### Compile, whitespace and resource checks

    python -m py_compile ui/sequence/sequence_widget_recording_process_ops.py unit_test/base/recording_process_fakes.py unit_test/base/test_recording_service.py unit_test/ui/test_recording_process_integration.py
    git diff --check
    Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Select-Object ProcessId,ParentProcessId,CommandLine

Compile and diff check exited 0 (Git emits only its existing LF-to-CRLF conversion notices). Both service fixtures verify shutdown and no live service threads; service tests also check worker_pid is None, and paused-reader tests explicitly assert reader closure. The six-round healthy/retired-worker test requires generations [1,1,2,2,3,3] and no accumulating helper threads. After the final tests and compiler process exited, the external process inventory contained only unrelated Python PIDs 19888 and 23412; no spawned recording/pytest helper remained. Both pre-existing processes were explicitly left alone.

## A1–A13 acceptance mapping

Test names below refer to the exact suites in the commands above. All are automated fake-device tests unless explicitly marked as a remaining deployment check.

| Acceptance | Actual tests and evidence |
| --- | --- |
| A1 — process isolation, ordinary/streaming | Service test_spawn_pid_complete_arrays_and_healthy_reuse (False/True); integration test_bridge_real_spawn_delivers_on_gui_thread_and_child_writes (four streaming/monitor combinations); test_launcher_import_in_fresh_process_has_no_qt_or_business_import. Capture/writer PIDs equal and differ from parent; Qt callbacks assert the GUI thread. |
| A2 — ordered exact audio, irregular/overlong chunks | Capture test_capture_owns_input_and_trims_once with single, noncontiguous and reversed channels, both modes; service PID/reuse and 600s tests compare every float32 multi/mono/WAV sample. Borrowed input is mutated immediately after each callback. |
| A3 — actual 1s/30s preview pause | test_paused_preview_credit_keeps_disk_progress_and_next_snapshot_cumulative: write trace advances from 3 to 7 frames while credit is held; no extra preview; resumed snapshot includes the .95 peak at frame 5 and starts at time zero. Final exact baseline is (known_audio(110) % .5).astype(float32), [5,0]=.95, first 100 frames, columns (0,2), including mono and WAV. Completion bypasses the second withheld credit. |
| A4 — 600 seconds, bounded previews | test_spawn_accepts_600_seconds_with_bounded_cumulative_previews_and_exact_wav for (2,) and (2,0). 1000Hz, target 600250, trim 250, final 600000 frames = 600 seconds; 301 generated chunks, overlong final input. Credit held while all raw frames are written, then cumulative preview sample_stop=600000 and last time=599.999. Every snapshot has at most 4000 points/channel; recursively inspected control IPC contains no arrays, preview IPC only bounded one-dimensional time/amplitude arrays. Qt test_paused_qt_preview_has_one_wakeup_while_child_file_continues proves one pending wakeup. Existing pure 600s reducer tests are supplemental, not substitutes for spawn. |
| A5 — completion, stale/duplicate events | Service test_duplicate_and_stale_events_cannot_redeliver_or_overwrite, test_acceptance_and_release_continuations_have_explicit_order, test_finalizing_state_is_observable_while_writer_is_closing; integration test_preview_replaces_cumulative_envelope_and_ignores_stale_final, test_cancel_view_after_service_acceptance_ignores_queued_success, main workflow DB-once/TCP completion tests. |
| A6 — modes, trim and monitoring | Capture test_effective_mode_and_monitor_gain_mute_fade, test_legacy_monitor_characterization; four-mode main/bridge tests; startup-trim, streaming-config and calibration-snapshot regression suites. Mid-recording device/trim changes cannot alter admitted request snapshots. |
| A7 — input calibration | test_calibration_child_captures_ten_seconds_and_saves_unchanged_json checks actual 441000 selected-channel samples, 94/114dB calculation, persisted factor and next channel; busy/cancel/reset/close tests, original calibration suites. Temporary WAV never enters user recording root/DB. |
| A8 — failure integrity | Capture queue saturation, input overflow, device identity, write/close faults and quality gate; service read failure; actual unpaced spawn queue saturation regression; integration unsuccessful_workflows[write_failed] asserts one invalid-recording notification, no saved DB/analysis/TCP finish, failed file cleanup. |
| A9 — child death and restart | test_worker_death_before_acceptance_fails_once_and_new_generation for recording/finalizing/delivering; test_dead_worker_between_offer_and_accept_command_cannot_publish_success; integration test_dead_worker_before_acceptance_never_publishes_success. Post-acceptance child death cannot revoke accepted results. |
| A10 — bounded lifecycle, parent death | test_startup_and_cancel_deadlines_retire_real_worker, test_healthy_ready_does_not_remove_next_round_start_deadline, native-close shutdown test, OS parent-death test (healthy and hung close), setup/thread failure ownership tests; actual MainWindow close/aboutToQuit and hidden SequenceWindow close tests. Only confirmed old PID death permits replacement. |
| A11 — presentation/metadata diagnostics | Capture metadata warning/close-ownership tests and preview exception test; service preview callback failure; integration test_final_plot_failure_is_warning_and_still_saves_once. Main cleanup_error_after_acceptance and main_resource_notice variants verify the resource warning is once, GUI-thread delivered, and cannot overwrite abort/close/new-session state. Calibration cleanup failure uses the same retained-lease notification contract. |
| A12 — existing business workflow | Exact seven-file business command above: multichannel, manual/serial conditions, round cleanup, barcode locks, storage and paths. 112 passing tests plus the explicitly separated baseline key failure; main integration final arrays, DB save once, direction/channel snapshots, no parent recording write/retrim/metadata, TCP recipient/completion snapshot, relabel leased source/destination protection. |
| A13 — readback/lease races | Service test_reader_lease_defers_ack_cleanup_and_reuse (cancel/reject/shutdown), reader timeout/new-generation isolation, retained-reader shutdown, exact metadata temporary ownership; integration calibrator_close_with_paused_reader_then_new_main, serial cleanup/reader lease, accepted cleanup denial. No early ack/delete/reuse; delayed readers release only old leases, no late success. |

## Recorded process/sample observations

The initial focused run used parent pytest PID 16264. Ordinary mode capture=writer=22188, streaming mode capture=writer=18552; each was reused for two rounds. The single-channel 600s child had capture=writer=11396, and reversed-multichannel child capture=writer=7752; both trace files reported written_frames=600250 and fed_chunks=301. These were real process trace observations from the temporary test directories, not mocked PID values. Per-test assertions independently require child != parent. Final arrays retain 600000 frames after the 250-frame trim; exact comparisons are implemented in tests, not inferred from the trace counts.

## Independent and earlier verification

The orchestrator independently ran the current service acceptance filter 'paused_preview_credit or synthetic_producer_writer_handshake or spawn_accepts_600_seconds': 6 passed / 53 deselected, 54.45s, exit 0. Its current integration filter 'cleanup_error_after_acceptance or main_resource_notice or relabel_destination or unsuccessful_workflows': 13 passed / 60 deselected, 11.60s, exit 0. Both used fresh temporary roots and no hardware.

Historical reports supplied by the orchestrator, not reruns by this implementer: Task 3 final 2401fda, exact eight-suite run 194 passed / 28.34s and core 106 passed / 67.99s. Task 4 initial 7c9b6c3 exact five-suite root run 202 passed / 111.89s; final 6f07036 root UI run 123 passed / 57.42s and quality reviewer exact five-suite run 208 passed / 119.19s. The Task 4 spec reviewer observed 207 passed / 1 scheduling failure / 131.49s; isolated retry passed in 7.71s. Task 5's writer-gate diagnostic addresses that fixture race without weakening overflow behavior.

## Remaining limits and ownership

Actual hardware throughput at deployment sample rate/channel count and frozen Windows executable startup remain manual deployment checks. No tracked packaging recipe (.spec, requirements, pyproject.toml or .bat) was found by the supplied git-ls-files query, so source spawn verification cannot be described as packaged validation. No sound device was selected or opened. Full-suite baseline failures are listed above.

The Task 3 FileOps.resolve_wav_label_target addition is a pure destination resolver used to deny moves into leased, not-yet-created paths before any file operation. It does not authorize changes to recording path rules, labels or business keys.

Read-only SHA256 checks of both original protected scripts match the supplied baseline:

- record_vkinging_continuous.py: 1F04219051E0D6652767D5AFB8672FAF081ED7B0527D1B7484BC74EC3B373054
- record_vkinging_continuous copy.py: ED09C95276A73D9C9C67F0B991FF141275E8C2E0733C6A360332D761CEE5A0E5

The following section records the subsequent orchestrator review and delivery steps.
## Final orchestrator verification and delivery — 2026-08-28

All five implementation tasks passed their paired specification and quality reviews. Task 5 reviewers each independently ran the 19-case acceptance filter: 65.23s and 64.36s, both exit 0. Final whole-implementation quality review passed; its independent full service and Qt integration run was **132 passed, 128.53s, exit 0**.

The result was applied to the original checkout at D:/dongyuan_ditting/SpeakerAnomalyDetection-MW. Before application, the original checkout was clean and only task-owned specification/plan checkpoints followed the recorded base. A recovery patch was captured; all **36 delivered files** were compared with their reviewed Git blobs and matched. A content-preserving mixed reset restored the original branch to the recorded base, leaving the task result as **uncommitted local changes with an empty staging area**. No push, PR or permanent task commit was created. The plan and this evidence document were then updated to record completed review and delivery.

Fresh verification from the original checkout (same exact commands above, fresh system-temporary directories):

- Focused eight-suite gate: **278 passed, 19 dependency warnings, 139.67s, exit 0**.
- Supplemental eleven-suite gate: **270 passed, 18 dependency warnings, 9.35s, exit 0**.
- All 33 changed Python files compiled; whitespace check passed. Module paths explicitly resolved to the original checkout, not the temporary worktree.
- Both standalone recording script SHA256 values still match the baselines above. Tests use temporary recording roots/DBs; copied runtime files, caches, logs and synthetic audio are excluded from delivery.

Restart main_window_Launcher.py to load the new implementation. During recording, GUI drawing/business processing run in the parent and capture/monitor/WAV writing run in the recording child. Existing running application instances were not restarted or modified. The known baseline test failures and unperformed real-device/packaged-executable checks remain as disclosed above.

After verification, normal guarded cleanup removed the temporary feature worktree and its exact recorded branch. Recovery patches remain in the ignored workflow directory. Final process inventory contained only the two untouched pre-existing Python processes; the original checkout retains the delivered files.
