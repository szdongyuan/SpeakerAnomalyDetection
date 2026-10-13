# Recording final envelope — Task 5 validation

Date: 2026-10-12. Worktree: `.worktrees/codex/recording-final-envelope`.
Starting checkpoint: `9a296d8b4746ce8341c91487a59c54ad316fbb8e`.
Spec: `docs/double-sdd/specs/2026-10-12-recording-final-envelope-design.md`.
Plan: `docs/double-sdd/plans/2026-10-12-recording-final-envelope.md`, Task 5.

## Result and scope

The planned UI regression passes (443 tests). The broad base run had 716 passes and two path-fixture failures; both fixtures were corrected, and the entire storage test file subsequently passed as part of the 74-test focused run. These are separate runs, not a claimed single 718-pass run. VE/queue/legacy streaming compatibility passed all 287 tests. No production source changes, commits, metadata edits, merges, or publication were made in Task 5.

The measured new 600-second completion contains no full multi/mono arrays, versus 635,040,000 bytes retained by the baseline. Old and new bounded waveform digests and per-channel point counts match exactly in both measured durations. Parent preparation is a few milliseconds, but **the final instrumented run does not demonstrate an end-to-end finalization speedup**: the new run's disk close dominated its tail. Memory reduction and removal of sample readback are supported independently of timing.

## Environment and measurement boundaries

- Windows 10 build 19045; Python 3.12.7, NumPy 1.26.4, soundfile 0.12.1.
- Local D: NTFS fixed volume, WDC WD5000AZLX-08K2TA0, SATA, 500,107,862,016-byte disk (Get-Partition/Get-Disk). Output beneath this worktree's `.pytest_cache`.
- `QT_QPA_PLATFORM=offscreen`; unique worktree-local pytest basetemps because the default system temp is inaccessible. Pytest processes ran serially; actual benchmark runs did not overlap pytest.
- Deterministic five-channel PCM24 data generated one 2,048-frame block at a time, 44,100 Hz, durations 2 and 600 seconds. No giant synthetic input array, realtime wait, device access, network share, or production recording. Temporary WAVs are removed after each run.
- Each baseline/new case starts a fresh parent interpreter and a separate capture child. The baseline child uses the real block writer and generated digest; its parent uses the retained ResultReader with a frozen main request. It is a benchmark-only old-route simulation, not a production fallback or exact replay of the old capture scheduler. The new case uses actual RecordingCapture/RecordingService and their accept/release lifecycle, with a paced synthetic input boundary.
- New main reader construction raises; child SoundFile sample reads also raise. Header/metadata inspection remains allowed. Independent CSV/analysis reads are not prohibited.
- CSV, analysis and Qt are absent from the process benchmark. DB/CSV/analysis business is mocked only in the separate Qt measurement below. Enabling real CSV/analysis can compete for disk and memory.
- Warm-cache synthetic file processing; no cache eviction or guaranteed disk isolation. Each route writes and immediately consumes its own file. Results are individual observations, not a hardware latency guarantee.

## TDD and regression evidence

| Step / log under `.pytest_cache` | Observed result | Interpretation |
|---|---:|---|
| `envelope_t5_compat_red.log` | 59 failed, 228 passed; exit 1 | VE fixture sent obsolete RecordingAudio; legacy runtime fixture lacked recovery helper and expected pre-PCM24 samples. |
| `envelope_t5_compat_green.log` | 287 passed; exit 0 | Prepared VE results, queue, voltage/calibration checks, and non-process array lifecycle retained. |
| Initial benchmark RED command | 3 failed; exit 1 | Script/module did not exist. |
| `envelope_benchmark_smoke.log` | 3 passed; exit 0 | Fresh real-process low-duration smoke and network-output rejection. |
| `envelope_base_regression_t5.log` | 716 passed, 2 failed; exit 1 | Two external-path fixtures placed tmp files inside DEFAULT_DIR by using the required worktree basetemp. |
| `envelope_t5_extra_red.log` | 16 failed, 29 passed; exit 1 | Missing benchmark digest output; 15 analysis-route fixtures still lacked prepared-result API. New Qt and rename characterization cases already passed. |
| `envelope_t5_extra_green.log` | 74 passed; exit 0 | Entire storage file, enhanced benchmark smoke, migrated original analysis route, and publication tests. |
| `envelope_ui_regression_t5.log` | 443 passed; exit 0 | Full planned UI suite after fixture migrations. |
| `envelope_benchmark_final.log` | exit 0 | Actual 2/600-second instrumented baseline and new runs. |

Storage fixtures now explicitly set DEFAULT_DIR to a sibling `application` directory, so their factory-audio path is truly external. Absolute-path persistence and rollback assertions remain intact. This fixes test isolation under the required basetemp, without changing storage behavior. No production RED/GREEN cycle was necessary because no production behavior changed; benchmark output additions did follow failing smoke assertions before implementation. No further refactor was needed.

Compatibility changes preserve failure and lifecycle assertions rather than dropping them: VE invalid descriptor tests now forge bounded/preparation objects instead of obsolete multi/mono fields; metadata-false injection targets actual parent preparation instead of a reader which main no longer starts. File voltage samples and frozen calibration are still independently inspected. Original analysis tests now use actual prepared results and assert raw fields stay None while the real spawned SPL worker reads the saved path. Legacy non-process runtime still retains raw arrays, compared with the existing PCM24 contract.

### Executed commands

All commands ran at the worktree root. Long commands used `python -u`, redirected complete output to the named log, and preserved the pytest exit code.

```powershell
$env:QT_QPA_PLATFORM='offscreen'
python -u -m pytest unit_test/ui/test_ve3668n_recording.py unit_test/ui/test_ve3668n_queue_parameters.py unit_test/ui/test_streaming_waveform_runtime.py -q --tb=short --basetemp=.pytest_cache/envelope_t5_compat_red
python -u -m pytest unit_test/ui/test_ve3668n_recording.py unit_test/ui/test_ve3668n_queue_parameters.py unit_test/ui/test_streaming_waveform_runtime.py -q --tb=short --basetemp=.pytest_cache/envelope_t5_compat_green
python -m pytest unit_test/base/test_recording_envelope_benchmark.py -q --basetemp=.pytest_cache/envelope_benchmark_red
python -u -m pytest unit_test/base/test_recording_envelope_benchmark.py -q --tb=short --basetemp=.pytest_cache/envelope_benchmark_smoke
python -u -m pytest unit_test/base/test_recording_final_envelope.py unit_test/base/test_recording_quality_accumulator.py unit_test/base/test_recording_final_result.py unit_test/base/test_recording_capture.py unit_test/base/test_recording_finalization.py unit_test/base/test_recording_metadata_append.py unit_test/base/test_recording_service.py unit_test/base/test_recording_service_pipeline.py unit_test/base/test_recording_worker_pipeline.py unit_test/base/test_recording_storage_consistency.py unit_test/base/test_recording_audio_lifetime.py -q --tb=short --basetemp=.pytest_cache/envelope_base_regression_t5
python -u -m pytest unit_test/base/test_recording_envelope_benchmark.py unit_test/ui/test_original_recording_analysis_route.py unit_test/ui/test_recording_envelope_publication.py -q --tb=short --basetemp=.pytest_cache/envelope_t5_extra_red
python -u -m pytest unit_test/base/test_recording_envelope_benchmark.py unit_test/base/test_recording_storage_consistency.py unit_test/ui/test_original_recording_analysis_route.py unit_test/ui/test_recording_envelope_publication.py -q --tb=short --basetemp=.pytest_cache/envelope_t5_extra_green
python -u -m pytest unit_test/ui/test_recording_envelope_publication.py unit_test/ui/test_recording_gui_finalization.py unit_test/ui/test_recording_process_integration.py unit_test/ui/test_raw_audio_csv_recording_integration.py unit_test/ui/test_original_recording_analysis_route.py unit_test/ui/test_multichannel_recording_workspace.py unit_test/ui/test_ve3668n_recording.py -q --tb=short --basetemp=.pytest_cache/envelope_ui_regression_t5
python -u tools/benchmark_recording_final_envelope.py --seconds 2 600 --channels 5 --sample-rate 44100 --block-frames 2048 --output .pytest_cache/recording-final-envelope-profile.json
git diff --check
```

### Coverage retained or extended

- Existing accumulator/quality/final-result suites cover irregular chunks, extrema ties, endpoints, capacity, non-finite/threshold evidence, ownership, and exact request binding.
- Capture/service suites cover preview off, relative/cumulative modes and preview failures while final envelopes match the WAV; cross-block/overlarge trim, metadata/header/close failures, quality failures, cancellation/rejection, late/generation events, and worker death. Other-purpose reader/digest behavior remains covered.
- Publication tests cover second-channel failure, explicit display error, preserved other-condition sources, cancelled new runs preserving saved sources, raw fields None and true frame count, and existing CSV/analysis enqueue behavior. Added real capture→relabel regression checks unchanged WAV bytes, updated current/source/cache paths, retained request identity, other-condition isolation, and later analysis path resolution.
- VE manual/automatic file analysis keeps file-local voltage/calibration, including absent/valid/invalidated calibration. Original analysis-route tests exercise the real subprocess queue, disabled analysis, cancellation/reentrancy and the real SPL worker with no raw fields.
- CSV's real background export remains in the full UI suite; it independently reads the saved file. Existing ownership tests prohibit a legacy path move while CSV owns it; changing a saved label does not authorize moving an in-flight export's source.

## Final instrumented benchmark

Raw JSON: `.pytest_cache/recording-final-envelope-profile.json`. GUI sample: `.pytest_cache/recording-final-envelope-gui-profile.json`. JSON/logs are local validation artifacts; no production audio is retained.

All wall/CPU values below are seconds. `total_parent` starts before capture/service launch and ends after result delivery; it is the whole accelerated synthetic operation, not finalization alone. Child capture spans its writer/setup/consumption/close/descriptor work, excluding interpreter import. Stage CPU uses process_time; concurrent threads overlap, so stage CPU values are not additive. Windows CPU clock granularity can report zero for short stages.

| Duration | Route | Parent total wall | Parent CPU | Child capture wall | Child capture CPU | Sample read calls |
|---:|---|---:|---:|---:|---:|---:|
| 2 | baseline | 1.695232 | 0.046875 | 0.767938 | 0.031250 | 2 |
| 2 | new | 0.933762 | 0.109375 | 0.225170 | 0.078125 | 0 |
| 600 | baseline | 22.952182 | 4.250000 | 17.978044 | 4.125000 | 404 |
| 600 | new | 24.132816 | 0.843750 | 23.737848 | 10.875000 | 0 |

The child tail starts at last synthetic input arrival before that block's write/digest/quality work and includes close/fsync and descriptor preparation. Baseline parent observation is immediately after descriptor recv returns; new observation is factory entry after descriptor validation. Both parent endpoints are result offer/reader completion on that same parent's perf_counter. IPC and dispatch before those parent observations are **not isolated**. The component sum adds durations measured locally in each process; it is not a subtraction of clocks or a complete end-to-end tail.

| Duration | Route | Child target→descriptor | Parent observation→offer | Tail component sum (excludes IPC/dispatch/GUI) |
|---:|---|---:|---:|---:|
| 2 | baseline | 0.610710 | 0.044751 | 0.655461 |
| 2 | new | 0.172000 | 0.006260 | 0.178260 |
| 600 | baseline | 2.141751 | 4.271471 | 6.413222 |
| 600 | new | 8.157000 | 0.002517 | 8.159517 |

The final 600-second new child tail is about 8.157 seconds, dominated by close at 8.120 seconds. The baseline's measured child-plus-parent components total about 6.413 seconds versus 8.160 seconds new. Therefore this run supports neither a guaranteed shorter tail nor a fixed completion deadline. The removed 4.271-second parent read/prepare stage is real, but it must not be described as saving that much total elapsed time when disk close varies. Across recording, the new route also incurs quality/envelope accumulation CPU; removing tail work does not make that work free.

### Stage wall and CPU

Cells are wall / process CPU seconds. “—” means that stage is absent in that route, not a zero-time measurement. Writer timings include PCM24 quantization and file I/O. Reader total includes read, digest, metadata, mono, quality and display, so its components must not be added again.

| Stage | 2 s baseline | 2 s new | 600 s baseline | 600 s new |
|---|---:|---:|---:|---:|
| write | 0.145567 / 0.000000 | 0.012325 / 0.000000 | 14.487611 / 2.765625 | 8.605176 / 2.765625 |
| close | 0.596568 / 0.000000 | 0.078650 / 0.000000 | 2.133169 / 0.000000 | 8.120048 / 0.109375 |
| capture_digest | 0.001360 / 0.000000 | — | 0.365280 / 0.390625 | — |
| quality_accumulate | — | 0.012745 / 0.000000 | — | 2.795372 / 4.062500 |
| envelope_accumulate | — | 0.016384 / 0.031250 | — | 3.391573 / 3.281250 |
| envelope_materialize | — | 0.060996 / 0.031250 | — | 0.008767 / 0.000000 |
| read | 0.002668 / 0.000000 | — | 0.563741 / 0.515625 | — |
| read_digest | 0.000954 / 0.000000 | — | 0.300847 / 0.250000 | — |
| display_prepare | 0.024163 / 0.015625 | — | 1.749950 / 1.421875 | — |
| reader_total | 0.042581 / 0.031250 | — | 4.270595 / 3.500000 | — |
| parent_prepare | — | 0.005871 / 0.000000 | — | 0.002282 / 0.000000 |

### Working set and retained arrays

GetProcessMemoryInfo counters include NumPy native allocations; tracemalloc is not used. Per-process columns are OS peak working sets observed during this interval. The combined column is the maximum periodically sampled parent+child current working sets, **not the sum of independent peaks**. Parent and child reads happen adjacent in time, not atomically. The poll requests 5 ms, but Windows scheduling/timer resolution can make actual intervals longer; brief peaks can be missed. Working-set sums can count shared pages twice and do not equal exclusive physical memory. Service shutdown is outside this measurement interval.

| Duration | Route | Parent peak MB | Child peak MB | Sampled combined peak MB | Samples | Full multi+mono bytes | Accumulator bytes | Bounded display bytes |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | baseline | 48.259 | 36.626 | 83.280 | 110 | 2,116,800 | 0 | 2,646,000 |
| 2 | new | 46.563 | 48.488 | 88.744 | 62 | 0 | 2,646,040 | 2,646,000 |
| 600 | baseline | 908.562 | 36.667 | 944.964 | 1452 | 635,040,000 | 0 | 2,878,908 |
| 600 | new | 71.021 | 50.369 | 113.275 | 1545 | 0 | 2,878,840 | 2,878,908 |

MB is decimal. At 600 seconds, the new accumulator owns 2,878,840 bytes and the prepared display 2,878,908 bytes. The result descriptor, immutable publication buffers, serialization copies, interpreter, libraries, queue and allocator retention add overhead; this is not a claim that total new memory is at most 10 MB. The baseline full multi+mono count is exactly 26,460,000 × (5 + 1) × 4 = 635,040,000 bytes, and its observed peak is larger because of reader/quality temporaries.

`test_large_recording_has_bounded_storage_from_creation` passed in the base regression. It constructs accumulators for 600 and 60,000 seconds, repeatedly appends blocks, verifies stable array identities/bytes and no history collection, and enforces a 2,879,920-byte bound for five channels at P=48,000. This structural evidence, not a single low RSS observation, supports bounded capacity.

Both 2-second routes produce 44,100 points/channel; both 600-second routes produce [47,981, 47,982, 47,982, 47,982, 47,982]. Independent BLAKE2b hashes of every bounded float64 time/float32 amplitude array match exactly between baseline and new for each duration. The smoke test asserts equality and the final JSON was independently checked with the same equality assertions.

### Separate Qt publication observation

`test_five_channel_long_payload_qt_publication_timing` measured 0.016132 s wall / 0.015625 s process CPU for five real offscreen plot windows with 48,000 points each, representing 600 seconds. It calls actual prepared-result publication and verifies every plotted value, true N, and empty raw fields. This is a bounded synthetic payload; it excludes QWidget painting/event-loop render completion and mocks DB/CSV/analysis business. It is not a measurement of the complete application or the benchmark's capture-to-screen latency.


## Limits and disposition

- The explicitly known toolbar baseline (one incomplete configuration fixture and three offscreen paint variants) was outside these requested regression commands and was not changed or newly reverified here. Earlier layout/current-product refresh investigations likewise do not establish a globally clean full-repository baseline. No blanket claim that unrelated failures are pre-existing is made.
- Hardware pacing, native device behavior, network storage, cold-cache performance and simultaneous production CSV/analysis load are not measured. No throughput or end-to-end latency guarantee follows from this local simulation.
- No main reader fallback was added. Hidden global manual OK/NG deletion and prior Task 1–4 checkpoints remain intact.
- `tools/` and `docs/` are ignored in this repository; only the intended benchmark file and this validation report should be explicitly staged by the orchestrator if a later checkpoint is authorized. The worktree and user changes remain uncommitted by this task.
