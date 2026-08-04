# Golden Plot Test Contract Synchronization Design

## Context

The current golden-sample dual-plot implementation intentionally leaves the two
X axes unlinked while plot ranges are prepared. After both plots have rendered,
`_finalize_plot_view_ranges_after_render()` computes the usable range and links
the deviation plot to the envelope plot. Two older layout/title tests still
expect the axes to be linked immediately after configuration, contradicting the
implemented lifecycle and the focused auto-range regression coverage.

The targeted test run also reports a separate SPEC-default assertion mismatch.
Per user direction, that mismatch is outside this correction and must remain
untouched.

## Goal

Synchronize the two stale golden-sample tests with the existing two-phase X-axis
link lifecycle without changing production behavior.

## Selected Approach

Update the affected tests so each one verifies both lifecycle phases:

1. Immediately after dual-plot configuration, the deviation plot has no X-axis
   link and neither plot has a Y-axis link.
2. After invoking `_finalize_plot_view_ranges_after_render()`, the deviation
   plot is X-linked to the envelope plot while Y axes remain independent.
3. Existing layout ordering, equal-height sizing, and title assertions remain
   intact.

This approach preserves the independent pre-link auto-range calculation that
prevents one plot from prematurely constraining the other.

## Alternatives Considered

- Link the plots during configuration: rejected because it contradicts the
  current range-finalization design and can hide valid X extents.
- Remove the link assertions entirely: rejected because the before/after
  lifecycle is an important regression contract.

## Files and Scope

- Modify `unit_test/ui/test_golden_sample_plot_layout.py`.
- Modify `unit_test/ui/test_golden_sample_plot_titles.py`.
- Do not modify production modules, configuration files, SPEC constants, or the
  separate SPEC-default test.
- Do not add ZIP or EXE artifacts.

## Verification

- Run the two affected tests and require both to pass.
- Run the changed-test-module suite used during preflight. The two stale
  golden-plot failures must be resolved; the separately accepted
  `test_spec_defaults_include_all_spectrum_modes` mismatch may remain and must be
  reported rather than changed.
- Run `git diff --check`.

## Compatibility / Migration

- Backward compatibility: required
- Protected surfaces: runtime plotting behavior, configuration defaults,
  fallback constants, serialized configuration, and user-visible plot layout
- Allowed breakage: none
- Migration strategy: none

## Exception and State Constraints

- No production exception behavior may change.
- Do not add broad catches, fallback behavior, or new throw sites.
- Do not add or alter module-level, class-level, or global state.

## Durable Workflow Metadata

- Recorded implementation base: `53a19ee56037e738c251bb82248e15a06f93eb6b`
- Tracking issue: `#797`
