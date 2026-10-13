# Retire the unreachable manual OK/NG entry

## Request and authorization
The user explicitly requests deletion of the full-audio-dependent manual OK/NG handler if it cannot be triggered from the current interface. Static tracing confirms its only production signal connections come from SequenceCountBoard.ok_btn/ng_btn, whose host is hidden in MotorDetectionLeftPanel. There are no production direct calls or shortcuts to clicked_ok_or_ng. Automatic labeling uses separate entry points.

This narrow cleanup is authorized now. The larger recording-envelope proposal remains pending its independent checksum policy decision. Do not implement that proposal or claim its specification is complete.

## Design
Remove clicked_ok_or_ng, its exclusive update_audio_label_info sender helper, the two signal connections, and the obsolete count-board OK/NG button construction and styling. Keep count-board mode/test availability callbacks, test/mark controls and restoration, shared recording-in-progress guard, update_recorded_signal_info_to_db, _persist_current_test_audio_label, _finalize_test_run, and the condition-path labeling handler. Do not change the separate old DirectionWaveformPanel or any visible result labels.

Preferred approach: remove the entire obsolete entry and its exclusive wiring. Keeping a no-op handler would retain a misleading compatibility dependency; removing all count-board or labeling infrastructure would affect live consumers and exceeds this request.

## Compatibility / Migration
- Backward compatibility: partial.
- Protected surfaces: current visible recording UI, automatic labels/database/file updates, barcode behavior, mode configuration, progress restore, WAV/CSV formats.
- Allowed breakage: direct invocation of the retired hidden clicked_ok_or_ng/update_audio_label_info methods and access to its obsolete count-board OK/NG buttons.
- Migration strategy: none for user files; update internal tests and fixtures of the retired entry.

## Verification
Verify no production references to the deleted entry remain. A real offscreen Qt signal-wiring test must succeed without obsolete buttons or callback stubs. Preserve behavior tests for active condition-label guards, automatic label persistence, mode callbacks/restoration and relevant current UI startup. Remove tests solely invoking the retired handler and replace their coverage with meaningful current behavior where applicable. Do not skip assertions or stub away shared persistence to make tests pass.

## Constraints
No new exception handling or globals are needed. Keep existing error boundaries; no broad catches, silent fallback, compatibility no-ops, or unrelated refactors. If a new exception is unavoidable it must have semantic/diagnostic/recovery value. Prefer instance/local state; no new single-use module-level constants except stable external/schema contracts. Work only in the requested isolated worktree; leave changes uncommitted for review.

## Result for the envelope proposal
The retired handler is no longer a compatibility requirement for removal of the full recording array. This does not prove that all other consumers are safe; the larger envelope design still requires its own dependency verification.
