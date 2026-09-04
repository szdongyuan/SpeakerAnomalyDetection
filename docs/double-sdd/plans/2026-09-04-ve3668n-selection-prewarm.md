# VE3668N Process-Lifetime Prewarm Implementation Plan

> **For agentic workers:** REQUIRED: Use the `subagent-driven-development` skill to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one best-effort, no-file VE3668N prewarm cycle per application process, with one automatic retry after recoverable native failure, busy-discard semantics, and signature-scoped blocking after an actual failed cycle.

**Architecture:** A bootstrap-owned `VePrewarmLifetime` is the process-lifetime source of truth and is injected into every reconstructed main window. A dedicated cross-process prewarm request uses the existing recording worker and its single `VeResourceController`; it reads and discards 0.5 seconds of per-channel frames without entering the WAV/result pipeline. `RecordingService` owns admission, incompatible-resource synchronization, generation retirement, one retry, timeouts, and structured first-cause diagnostics; the Qt bridge and main window only deliver state and enforce UI admission.

**Tech Stack:** Python 3, PyQt5, `multiprocessing`, `threading`, VkDaq ctypes SDK binding, pytest, existing fake VE SDK and Qt test fixtures.

---

## File Structure

- Create `base/ve3668n_prewarm_lifetime.py`: process-lifetime opportunity/outcome and signature-scoped admission state; no SDK or Qt calls.
- Modify `base/recording_process_protocol.py`: immutable, picklable prewarm request/result/fault contracts and validation.
- Modify `base/ve3668n_resource.py`: no-file prewarm adapter creation and structured native fault preservation while retaining the existing two-argument recording failure callback.
- Modify `base/recording_worker.py`: serialize prewarm commands through the existing single VE owner, emit prewarm-only events, and avoid `RecordingCapture`/WAV/result pipeline creation.
- Modify `base/recording_service.py`: atomic prewarm admission, incompatible VE release, deadlines, generation retirement, 0.75-second retry scheduling, and exactly-once completion.
- Modify `ui/recording_service_bridge.py`: GUI-thread prewarm API and hardware-busy projection.
- Modify `main_window_Launcher.py`: create and retain exactly one `VePrewarmLifetime` per application process.
- Modify `main_window.py`: require lifetime injection, trigger the one opportunity after valid discovery or the first valid hardware confirmation, render outcome, and gate calibration/hardware actions.
- Modify `ui/sequence/sequence_widget_recording_process_ops.py`: apply the same pending/failed-signature admission to every recording workflow.
- Modify `ui/sequence/sequence_widget_ui_ops.py`: keep recording controls synchronized with the unified admission result.
- Modify relevant barcode/serial entry points only if their tests prove they bypass `_can_start_recording_workflow`; prefer the existing shared guard instead of duplicating state checks.
- Modify `unit_test/base/test_ve3668n_protocol.py`, `unit_test/base/test_ve3668n_resource.py`, and `unit_test/base/test_ve3668n_service.py`: protocol, resource, worker/service, timeout, retry, and no-file coverage.
- Modify `unit_test/ui/test_ve3668n_persistent_hardware.py`, `unit_test/ui/test_ve3668n_hardware.py`, `unit_test/ui/test_ve3668n_recording.py`, and `unit_test/ui/test_ve3668n_workflow.py`: bridge, process-lifetime reconstruction, startup/switch trigger, busy discard, and all-entry admission coverage.

## Cross-Cutting Constraints

- Preserve existing public `RecordingService.start`, recording callback, WAV, metadata, file lease, and hardware-selection formats.
- All VkDaq calls remain on the existing worker-side VE owner thread. GUI, bridge, and parent service must never call the SDK directly.
- Do not use a mutable module global, a `QApplication` dynamic property, a config marker, or a filesystem marker for the one-per-process opportunity.
- `VePrewarmLifetime` is created only by application bootstrap and injected. Missing injection is a deterministic construction error, not a reason to create a fallback instance.
- Busy discard consumes the process opportunity, never queues or auto-runs later, emits no modal, and permits formal recording once the underlying hardware operation is safe.
- A prewarm that actually starts and fails twice records `failed_signature`; only that exact VE acquisition signature is blocked. Ordinary audio and different VE signatures remain usable without another prewarm.
- Preserve the first native operation/code/detail. Cleanup, worker fatal, and `owner exited before binding` are appended diagnostics and cannot overwrite it.
- Do not add broad exception catches, silent sound-card fallback, duplicate error normalization, or single-use module constants. Stable protocol/status values may be centralized at their protocol boundary.
- Every wait is deadline-driven and injectable in tests; do not block the Qt thread or add `sleep` to production paths.

### Task 1: Add Immutable Prewarm Protocol Contracts

**Files:**
- Modify: `base/recording_process_protocol.py:70-225`
- Test: `unit_test/base/test_ve3668n_protocol.py`

- [ ] **Step 1: Write failing request validation tests**

Add tests that construct a valid request and reject sound-device backends, unavailable devices, empty/duplicate channels, mismatched sample rates, invalid attempt numbers, and incorrect frame counts. The production request should freeze the device and ordered channels.

```python
def test_ve_prewarm_request_is_half_second_per_channel(device):
    request = VePrewarmRequest.create("warm-1", device, (7, 1), 51200, attempt=1)
    assert request.frames_per_channel == 25600
    assert request.channels == (7, 1)
    assert request.signature == ("vkinging", device["machine_id"], (7, 1), 51200)
```

- [ ] **Step 2: Run the focused protocol tests and verify failure**

Run: `python -m pytest unit_test/base/test_ve3668n_protocol.py -q`

Expected: FAIL because `VePrewarmRequest` and prewarm result contracts do not exist.

- [ ] **Step 3: Implement the minimal contracts**

Add immutable contracts with strict `__post_init__` validation:

```python
@dataclass(frozen=True)
class VePrewarmRequest:
    warmup_id: str
    device: Mapping
    channels: tuple[int, ...]
    sample_rate: int
    frames_per_channel: int
    attempt: int

    @classmethod
    def create(cls, warmup_id, device, channels, sample_rate, *, attempt):
        rate = validate_sample_rate(sample_rate)
        return cls(warmup_id, device, tuple(channels), rate,
                   max(1, math.ceil(rate * 0.5)), attempt)

    @property
    def target_samples(self):
        return self.frames_per_channel
```

Also add a picklable structured terminal payload carrying `warmup_id`, `generation`, `attempt`, `signature`, `success`, `stage`, nullable native `code`, `detail`, `frames_per_channel`, `handles_released`, diagnostics, and lifecycle counts. Reuse `_acquisition_signature`; do not parse an error string to recover a native code.

- [ ] **Step 4: Add pickle/immutability and malformed terminal tests**

Verify round-trip pickling, invalid generation/attempt/code/frame counts, signature mismatch, malformed diagnostics, and exact ordered-channel preservation.

- [ ] **Step 5: Run protocol tests**

Run: `python -m pytest unit_test/base/test_ve3668n_protocol.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add base/recording_process_protocol.py unit_test/base/test_ve3668n_protocol.py
git commit -m "feat: define VE prewarm protocol"
```

### Task 2: Add a No-File Resource Prewarm Adapter

**Files:**
- Modify: `base/ve3668n_resource.py:17-180, 300-485`
- Test: `unit_test/base/test_ve3668n_resource.py`
- Test helper: `unit_test/base/ve3668n_fakes.py`

- [ ] **Step 1: Write failing no-file prewarm tests**

Cover partial multichannel reads, exact per-channel accumulation, discarded blocks, successful detach to `IDLE`, reuse by a same-signature recording, zero WAV/result dependencies, and malformed native buffers. Explicitly inject a non-`ctypes.c_double` array, capacity different from `requested_frames * channel_count`, negative/oversized `returned_frames`, and a short scalar payload inconsistent with `returned_frames * channel_count`; every case must fail before delivery or progress advancement.

```python
def test_prewarm_discards_partial_multichannel_reads_and_retains_task(fake_sdk, device):
    request = VePrewarmRequest.create("warm-1", device, (7, 1), 8, attempt=1)
    adapter = controller.prewarm(request=request, fail=failures.append)
    assert adapter.start()
    assert adapter.completed.wait(1)
    assert adapter.progress_snapshot.frames == 4
    assert adapter.handles_released
    assert controller.signature == request.signature
    assert fake_sdk.calls("start_task") == 1
```

- [ ] **Step 2: Run the focused resource tests and verify failure**

Run: `python -m pytest unit_test/base/test_ve3668n_resource.py -q`

Expected: FAIL because the controller has no prewarm API or structured resource fault.

- [ ] **Step 3: Preserve structured native faults without breaking recording callbacks**

Add an instance-scoped immutable `VeResourceFault(stage, code, detail)` and an adapter `failure_snapshot` property. In `_initialize` and read failure boundaries, map `VkDaqError` directly from its fields; map other exceptions to `code=None` and `detail=str(exc)`. Keep existing `fail(stage, message)` invocation unchanged for recording compatibility.

- [ ] **Step 4: Implement `VeResourceController.prewarm` using the existing owner**

Create a short-lived adapter that uses the same binding, deadline, `_validate_read`, ordered-channel conversion, and detach confirmation as recording, but whose callback only advances progress and retains no audio. Expose one completion event; do not introduce a second owner thread or SDK client.

```python
def prewarm(self, *, request, fail):
    return VeRecordingAdapter(
        self, request,
        callback=lambda _physical, _frames, _a, _b: None,
        fail=fail,
        stop_event=threading.Event(),
        signature=ve_acquisition_signature(
            request.device, request.channels, request.sample_rate),
    )
```

Use the existing automatic target detach. If a small dedicated adapter wrapper is needed for a `completed` alias, keep it in this module and instance-scoped.

- [ ] **Step 5: Add native-failure precedence tests**

Inject `VkDaqStartTask(code=-12001)` followed by cleanup/bind diagnostics. Assert the adapter/controller preserve `stage="start_task"`, `code=-12001`, and original detail as first cause. Add read fault and detach timeout cases. Add explicit malformed-buffer type/capacity/returned-frame tests and assert no invalid scalar reaches the discard callback and per-channel progress remains at the last valid frame count.

- [ ] **Step 6: Run resource tests**

Run: `python -m pytest unit_test/base/test_ve3668n_resource.py -q`

Expected: PASS, including existing persistent-resource tests.

- [ ] **Step 7: Commit**

```bash
git add base/ve3668n_resource.py unit_test/base/test_ve3668n_resource.py unit_test/base/ve3668n_fakes.py
git commit -m "feat: add no-file VE resource prewarm"
```

### Task 3: Add Worker-Side Prewarm Commands and Events

**Files:**
- Modify: `base/recording_worker.py:160-300`
- Modify: `base/recording_process_protocol.py:405-440`
- Test: `unit_test/base/test_ve3668n_service.py`

- [ ] **Step 1: Write failing worker protocol tests**

Use the existing spawned-worker fake backend to prove that a `prewarm_ve` command starts the controller adapter, emits prewarm-only started/terminal events, reads the target frames, and creates no WAV, descriptor, result ack, preview, file lease, or recording session.

- [ ] **Step 2: Run the focused spawned-worker test**

Run: `python -m pytest unit_test/base/test_ve3668n_service.py -q -k "prewarm_worker"`

Expected: FAIL because the command is unknown.

- [ ] **Step 3: Implement one active worker prewarm state**

Add an internal worker state holding request, adapter, first structured fault, and exactly-once terminal flag. Accept `prewarm_ve` only when no `pipeline.active` capture and no existing prewarm are active. Start through `controller.prewarm`; poll completion in the existing loop; package lifecycle counts and structured fault into the protocol terminal.

Do not call `RecordingCapture`, create a writer, allocate a temporary directory, or add the warmup to `RecordingWorkerPipeline` finalizers.

- [ ] **Step 4: Handle cancellation, shutdown, and generic fatal ordering**

On shutdown/cancel, stop the adapter and preserve handles-released truth. If controller fatal and prewarm terminal race, emit only one prewarm terminal and one generation fatal diagnostic; ensure a later `owner exited before binding` cannot replace the first fault.

- [ ] **Step 5: Add malformed/overlap tests**

Assert recording/prewarm overlap is rejected, duplicate warmup IDs are protocol faults, old-generation commands are ignored, shutdown remains bounded, and no result pipeline capacity is consumed.

- [ ] **Step 6: Run worker and existing service tests**

Run: `python -m pytest unit_test/base/test_ve3668n_service.py unit_test/base/test_recording_worker_pipeline.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add base/recording_worker.py base/recording_process_protocol.py unit_test/base/test_ve3668n_service.py
git commit -m "feat: run VE prewarm in recording worker"
```

### Task 4: Implement Service Admission, Retry, and Generation Recovery

**Files:**
- Modify: `base/recording_service.py:125-330, 400-590, 650-840, 1025-1180`
- Test: `unit_test/base/test_ve3668n_service.py`

- [ ] **Step 1: Write failing atomic-admission tests**

Cover immediate `accepted`, `busy`, and `closing` statuses. Assert acceptance atomically makes `can_start_recording=false`, does not create a `RecordingSession`, and does not reserve a path or pipeline slot.

```python
status = service.prewarm_ve(request, completions.append)
assert status == "accepted"
assert not service.can_start_recording
assert not service.is_path_leased(any_path)
```

- [ ] **Step 2: Run focused admission tests**

Run: `python -m pytest unit_test/base/test_ve3668n_service.py -q -k "prewarm and admission"`

Expected: FAIL because `prewarm_ve` does not exist.

- [ ] **Step 3: Add `_PendingVePrewarm` and atomic service API**

Track frozen base request, current attempt, first fault, callback, worker generation, start/capture/detach deadlines, retry deadline, completion flag, and ownership safety. `prewarm_ve` must reject busy/closing before mutating state. An incompatible retained VE signature must enter the existing release preparation path under the same lock; only a successfully admitted release may later dispatch prewarm.

Represent an admitted prerequisite release as part of `_PendingVePrewarm`, not as an unrelated callback. If it later succeeds, dispatch attempt 1. If it fails/times out, never dispatch prewarm: finalize with the requested signature, release-stage first cause, and explicit `ownership_safe` derived only from confirmed worker death/handle closure. Preserve the failed signature for the GUI lifetime to gate; do not treat a release timeout as busy discard.

- [ ] **Step 4: Implement success and timeout handling**

Validate every worker event by generation, warmup ID, attempt, and signature. On success, store retained signature/lifecycle counts, clear pending state, recompute admission, and invoke exactly one success callback. Apply existing production defaults: worker ready 10 seconds, start 10 seconds, controller bind 3 seconds, `VeCaptureDeadline` 5.5 seconds for the half-second target with 5-second no-progress detection, detach 0.5 seconds, release 5 seconds, and shutdown 5 seconds. Expose/inject the existing monotonic clock in the prewarm state machine so tests can advance each boundary without wall-clock sleeps.

- [ ] **Step 5: Implement one generation-level retry**

For a retryable first failure, preserve the first structured fault, retire the worker, and wait for `_dead` to confirm process death and close IPC. Schedule attempt 2 at `now + 0.75` using `_tick`; do not sleep. Spawn a fresh worker and send a request with `attempt=2`. Attempt 2 terminal or a non-retryable fault completes failure exactly once.

- [ ] **Step 6: Implement kill/no-confirm safety**

Terminate immediately on retirement; at the existing 2-second terminate deadline issue kill. If OS death is still unconfirmed at that point, finish the prewarm callback as ownership-uncertain failure, keep service hardware admission closed, and never start attempt 2. A later `_dead` may clear ownership uncertainty but cannot rewrite the terminal.

- [ ] **Step 7: Add retry and first-cause tests**

Test:

- first `start_task -12001`, confirmed worker death, 0.75-second deadline, second success;
- two failures and no third worker;
- deterministic request validation with no retry;
- read timeout retry;
- detach failure retry only after death;
- kill without death confirmation gives one ownership-uncertain terminal;
- secondary worker fatal/bind errors do not overwrite first stage/code/detail;
- incompatible retained signature releases before dispatch;
- release admission busy returns busy and creates no pending release/prewarm;
- admitted incompatible release succeeds and dispatches exactly one attempt;
- admitted release later fails with confirmed worker death: no prewarm dispatch, one release failure terminal with requested `failed_signature` and `ownership_safe=true`;
- admitted release times out or worker death is unconfirmed: no prewarm dispatch, one release failure terminal with `ownership_safe=false`, service hardware admission remains closed;
- shutdown cancels without retry or GUI-oriented error.

Add deterministic injectable-clock boundary tests for every specified deadline, not only representative failures:

- ready at 10 seconds, including just-before/no-fire and at-deadline/fire;
- service start at 10 seconds versus controller bind at 3 seconds, proving the first observed boundary owns the first cause;
- total capture at native start + 5.5 seconds and independent 5-second no-progress timeout;
- detach at 0.5 seconds;
- prerequisite release at 5 seconds;
- terminate at 2 seconds followed by kill/no-death ownership uncertainty;
- retry dispatch just before and exactly at 0.75 seconds;
- shutdown at 5 seconds, with no retry or callback resurrection.

- [ ] **Step 8: Run service suites**

Run: `python -m pytest unit_test/base/test_ve3668n_service.py unit_test/base/test_recording_service_pipeline.py unit_test/base/test_recording_ve_release.py -q`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add base/recording_service.py unit_test/base/test_ve3668n_service.py unit_test/base/test_recording_service_pipeline.py unit_test/base/test_recording_ve_release.py
git commit -m "feat: orchestrate VE prewarm retry"
```

### Task 5: Expose Prewarm Through the Qt Bridge

**Files:**
- Modify: `ui/recording_service_bridge.py:15-115`
- Test: `unit_test/ui/test_ve3668n_persistent_hardware.py`

- [ ] **Step 1: Write failing bridge tests**

Assert GUI-thread-only invocation, immediate busy/closing status, queued exactly-once completion, pending hardware-busy projection, synchronous service completion safety, shutdown behavior, and callback delivery on the Qt thread.

- [ ] **Step 2: Run focused bridge tests**

Run: `python -m pytest unit_test/ui/test_ve3668n_persistent_hardware.py -q -k "prewarm"`

Expected: FAIL because the bridge API is absent.

- [ ] **Step 3: Implement `RecordingServiceBridge.prewarm_ve`**

Mirror the proven release bridge shape without merging their state:

```python
def prewarm_ve(self, request, callback):
    if QThread.currentThread() is not self.thread():
        raise RuntimeError("Recording bridge VE prewarm must start on its GUI thread")
    self._ve_prewarm_pending = True
    status = self.service.prewarm_ve(request, complete)
    if status != "accepted":
        self._ve_prewarm_pending = False
    return status
```

The exactly-once `complete` closure must clear pending before queueing the user callback. Include `_ve_prewarm_pending` in `hardware_busy`; do not conflate a busy-discard response with an accepted pending operation.

- [ ] **Step 4: Run bridge tests**

Run: `python -m pytest unit_test/ui/test_ve3668n_persistent_hardware.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ui/recording_service_bridge.py unit_test/ui/test_ve3668n_persistent_hardware.py
git commit -m "feat: bridge VE prewarm to Qt"
```

### Task 6: Add the Bootstrap-Owned Process Lifetime

**Files:**
- Create: `base/ve3668n_prewarm_lifetime.py`
- Modify: `main_window_Launcher.py:5-50`
- Modify: `main_window.py:24-45, 870-880`
- Test: `unit_test/ui/test_ve3668n_hardware.py`

- [ ] **Step 1: Write failing pure lifetime tests**

Test atomic `claim`, `succeed`, `skip_busy`, `fail`, stale token rejection, signature-scoped blocking, and immutable snapshots. Verify skipped busy allows every signature, failed blocks only exact signature, and no outcome restores `available`.

- [ ] **Step 2: Write failing reconstruction tests**

Create one lifetime, construct/destroy two main-window harnesses and two service/bridge harnesses with it, and assert the second sees the first outcome. Assert missing lifetime injection raises a clear `TypeError`/`ValueError` instead of constructing a fallback.

- [ ] **Step 3: Run focused tests**

Run: `python -m pytest unit_test/ui/test_ve3668n_hardware.py -q -k "prewarm_lifetime"`

Expected: FAIL because the type and injection do not exist.

- [ ] **Step 4: Implement the focused lifetime object**

Keep it independent of Qt and SDK. Use an instance lock only if callbacks can cross threads; expose snapshots rather than mutable fields.

```python
class VePrewarmLifetime:
    def claim(self, token, signature): ...
    def mark_succeeded(self, token, signature): ...
    def mark_skipped_busy(self, token, signature, detail): ...
    def mark_failed(self, token, signature, fault, *, ownership_safe): ...
    def admission_for(self, signature): ...  # pending / allowed / failed_signature
```

- [ ] **Step 5: Inject from every production bootstrap**

`MainWindowLauncher` creates one lifetime beside `QApplication`/recording service and passes it to `MainWindow`. The direct `main_window.py` development entry creates one explicitly before constructing the window. `MainWindow.__init__` requires `ve_prewarm_lifetime`; tests must inject it.

- [ ] **Step 6: Run lifetime and existing construction tests**

Run: `python -m pytest unit_test/ui/test_ve3668n_hardware.py unit_test/ui/test_ve3668n_workflow.py -q -k "lifetime or construction or launcher"`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add base/ve3668n_prewarm_lifetime.py main_window_Launcher.py main_window.py unit_test/ui/test_ve3668n_hardware.py unit_test/ui/test_ve3668n_workflow.py
git commit -m "feat: own VE prewarm for process lifetime"
```

### Task 7: Trigger the Single Opportunity at Startup and Hardware Confirmation

**Files:**
- Modify: `main_window.py:83-205, 540-610`
- Test: `unit_test/ui/test_ve3668n_hardware.py`
- Test: `unit_test/ui/test_ve3668n_workflow.py`

- [ ] **Step 1: Write failing startup trigger tests**

Cover sound-card restore (opportunity remains available), unavailable VE discovery (opportunity remains available), first valid VE discovery (one claim), duplicate/late discovery (no second claim), accepted prewarm success, and valid discovery while hardware busy (consume as skipped, no service call queued, no later auto-call, formal admission allowed after busy clears).

- [ ] **Step 2: Write failing hardware-confirmation tests**

Cover cancel/no consumption, first valid VE confirmation consumption, busy-discard, and all later machine/channel/rate/same-signature confirmations producing no prewarm after succeeded/failed/skipped consumption. Confirm switching ordinary audio never restores the opportunity.

- [ ] **Step 3: Run focused UI tests**

Run: `python -m pytest unit_test/ui/test_ve3668n_hardware.py unit_test/ui/test_ve3668n_workflow.py -q -k "prewarm"`

Expected: FAIL because MainWindow has no trigger coordinator.

- [ ] **Step 4: Implement one MainWindow coordinator path**

Add helpers such as `_try_start_ve_prewarm(device, channels, source)`, `_on_ve_prewarm_complete`, and `_ve_prewarm_signature`. Validate availability/channels/rate before `claim`. After a successful claim, freeze one protocol request and call the bridge.

If the bridge/service returns busy, call `mark_skipped_busy`, update status/log, and never retain a pending callback. If accepted, display initialization pending. If closing, consume silently. Match every completion by lifetime token, warmup ID, and signature before applying it.

For an admitted prerequisite release that later fails, call `mark_failed` with the originally requested signature and the service-provided `ownership_safe`. With `ownership_safe=true`, show one release-first-cause modal and reopen hardware selection; with `ownership_safe=false`, show one restart-required modal and keep hardware selection disabled. Neither branch may dispatch prewarm or convert to `skipped_busy`.

- [ ] **Step 5: Integrate existing VE release without double commands**

When the one opportunity is available and a confirmed VE selection needs an incompatible release, let `RecordingService.prewarm_ve` own release preparation; do not also call `release_ve`. When the opportunity is already consumed, retain the existing hardware-change `release_ve` behavior. Switching from VE to ordinary audio always retains existing release behavior.

- [ ] **Step 6: Render status and modal policy**

- pending: status “VE 设备正在初始化…”, no modal;
- succeeded: normal available status, no modal;
- skipped_busy: log/status only, no modal, no later prewarm;
- actual failed signature: one modal with first cause; ownership-safe suggests selecting another device, ownership-uncertain instructs restart;
- stale/closing callback: no UI mutation/modal.

- [ ] **Step 7: Run UI trigger tests**

Run: `python -m pytest unit_test/ui/test_ve3668n_hardware.py unit_test/ui/test_ve3668n_workflow.py -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add main_window.py unit_test/ui/test_ve3668n_hardware.py unit_test/ui/test_ve3668n_workflow.py
git commit -m "feat: trigger one VE prewarm per process"
```

### Task 8: Enforce Unified Recording and Calibration Admission

**Files:**
- Modify: `main_window.py:116-180, 608-630`
- Modify: `ui/sequence/sequence_widget_recording_process_ops.py:630-705`
- Modify: `ui/sequence/sequence_widget_ui_ops.py:550-590`
- Modify if required by failing tests: `ui/sequence/sequence_widget_barcode_ops.py`
- Modify if required by failing tests: `ui/sequence/sequence_widget_serial_trigger_ops.py`
- Test: `unit_test/ui/test_ve3668n_recording.py`
- Test: `unit_test/ui/test_ve3668n_workflow.py`
- Test: `unit_test/ui/test_recording_entry_overlap.py`

- [ ] **Step 1: Write failing admission matrix tests**

Parameterize button/manual, barcode, serial/hardware trigger, replay-triggered recording, and calibration entry points over:

- available ordinary audio: allowed by existing rules;
- VE pending: blocked;
- VE succeeded: allowed;
- VE skipped_busy after underlying hardware clears: allowed;
- VE exact failed signature: blocked with one diagnostic;
- different VE signature after failure: allowed with no second prewarm;
- ownership uncertain: all hardware operations blocked.

- [ ] **Step 2: Run focused admission tests**

Run: `python -m pytest unit_test/ui/test_ve3668n_recording.py unit_test/ui/test_ve3668n_workflow.py unit_test/ui/test_recording_entry_overlap.py -q -k "prewarm or admission"`

Expected: FAIL because current admission knows only recording service busy state.

- [ ] **Step 3: Add one shared admission query**

Inject the same lifetime into `SequenceWindow`. Extend `_can_start_recording_workflow` to compute the current VE acquisition signature and combine lifetime admission with existing service/capacity/analysis policy. Do not duplicate checks in each trigger method; update bypasses only when a failing test proves they do not call the shared guard.

- [ ] **Step 4: Gate calibration and hardware actions**

MainWindow calibration admission must reject pending, exact failed signature, and ownership uncertainty. Hardware selection is blocked only while pending/underlying ownership is unsafe; it remains available after an ownership-safe failed signature so the user can select ordinary audio or a different VE signature.

- [ ] **Step 5: Synchronize UI controls**

Update the existing UI refresh path so buttons reflect shared admission, while every action handler still rechecks admission to prevent stale enabled state races.

- [ ] **Step 6: Run admission and entry-point suites**

Run: `python -m pytest unit_test/ui/test_ve3668n_recording.py unit_test/ui/test_ve3668n_workflow.py unit_test/ui/test_recording_entry_overlap.py unit_test/ui/test_ve3668n_calibration.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add main_window.py ui/sequence/sequence_widget_recording_process_ops.py ui/sequence/sequence_widget_ui_ops.py ui/sequence/sequence_widget_barcode_ops.py ui/sequence/sequence_widget_serial_trigger_ops.py unit_test/ui/test_ve3668n_recording.py unit_test/ui/test_ve3668n_workflow.py unit_test/ui/test_recording_entry_overlap.py unit_test/ui/test_ve3668n_calibration.py
git commit -m "feat: gate capture on VE prewarm outcome"
```

### Task 9: Run Regression and Real-Hardware Verification

**Files:**
- Modify only if a test exposes an in-scope defect; do not broaden scope.
- Record manual evidence in the implementation handoff, not in a new persistent data format.

- [ ] **Step 1: Run focused base regression**

Run:

```bash
python -m pytest unit_test/base/test_vkinging_sdk.py unit_test/base/test_ve3668n_protocol.py unit_test/base/test_ve3668n_resource.py unit_test/base/test_ve3668n_service.py unit_test/base/test_recording_service_pipeline.py unit_test/base/test_recording_ve_release.py -q
```

Expected: PASS.

- [ ] **Step 2: Run focused UI regression**

Run:

```bash
python -m pytest unit_test/ui/test_ve3668n_persistent_hardware.py unit_test/ui/test_ve3668n_hardware.py unit_test/ui/test_ve3668n_recording.py unit_test/ui/test_ve3668n_workflow.py unit_test/ui/test_ve3668n_calibration.py unit_test/ui/test_recording_entry_overlap.py -q
```

Expected: PASS.

- [ ] **Step 3: Run the broader VE suite**

Run: `python -m pytest unit_test/base/test_ve3668n_*.py unit_test/ui/test_ve3668n_*.py -q`

Expected: PASS. If shell wildcard expansion is unavailable, enumerate with `Get-ChildItem` and pass the paths to pytest without changing files.

- [ ] **Step 4: Run static integrity checks**

Run:

```bash
python -m compileall -q base ui main_window.py main_window_Launcher.py
git diff --check
```

Expected: exit code 0 and no output from `git diff --check`.

- [ ] **Step 5: Perform Windows/VE3668N cold-start acceptance**

On the affected machine:

1. Restart Windows; do not manually restart VkDaqAssistant.
2. Start a fresh application process with saved Vkinging input.
3. Confirm exactly one prewarm cycle and then an immediate formal recording without `-12001`.
4. Repeat the fresh-process start/prewarm/immediate-recording sequence at least 10 consecutive times. Restart the application process for every cycle; include at least the first cycle after a full Windows restart. Every cycle must succeed without `-12001`, a secondary bind failure, or a user WAV from prewarm.
5. In one process switch ordinary audio -> Vkinging, change VE channels/rate, and re-confirm the same VE; confirm no second prewarm.
6. Exercise a busy-discard injection/harness; confirm no delayed prewarm and that formal recording becomes allowed when ownership is safe.
7. Inject first-attempt `-12001`; confirm one fresh-generation retry and success.
8. Inject two failures; confirm only the exact failed signature is blocked, ordinary audio/different VE signature remain usable without prewarm, and no failed WAV is created.

Record process instance, lifetime outcome, warmup ID, selection token, signature, generation, attempt durations, first cause, lifecycle counts, and whether any user file was produced.

- [ ] **Step 6: Commit any narrowly required verification fixes**

Only if Steps 1-5 exposed an in-scope defect:

```bash
git add <exact changed source and test files>
git commit -m "fix: complete VE prewarm verification"
```

Otherwise do not create an empty commit.
