# PRB Extension to 2-35 Harmonics Design

**Date:** 2025-12-10
**Status:** Approved for Implementation
**Author:** Claude Code (with user input)

## Overview

Extend Perceptual Rub & Buzz (PRB) analysis from the current 10-35 harmonic range to 2-35 harmonics, matching the full range of HD (Harmonic Distortion) analysis while maintaining the PRB name and implementing dynamic cumulative masking.

## Motivation

### Current State

| Analysis Type | Current Range | Should Be | Status |
|--------------|---------------|-----------|--------|
| HD (Harmonic Distortion) | 2-35 | 2-35 | ✅ Correct |
| RB (Rub & Buzz) | 10-35 | 10-35 | ✅ Correct (high-order only) |
| PRB (Perceptual Rub & Buzz) | 10-35 | **2-35** | ❌ Needs fix |

### Why Extend PRB to 2-35?

1. **Match HD's capability** - PRB should analyze the same harmonic range as HD for consistency
2. **Low-order harmonic perceptual analysis** - Harmonics 2-9 are often audible and contribute to overall perception
3. **Unified perceptual analysis** - Users can analyze perceptual loudness across all harmonics, not just high-order ones

### Why Keep RB at 10-35?

RB (Rub & Buzz) is specifically designed for **high-order harmonic distortion analysis** (rattling, buzzing sounds). The 10-35 range restriction is intentional and should remain.

## Design Decisions

### 1. UI Configuration Range: 2-35 Harmonics

**Change:** Update `PerceptualRbConfigWindow` to support harmonics 2-35.

**Implementation:**
```python
# perceptual_rb_config_dialog.py

def create_harmonic_slider_layout(self):
    # Change: 10-35 → 2-35
    for i in range(2, 36):  # Was: range(10, 36)
        # ... create checkbox for harmonic i

def __init__(self, config_manager, model_type):
    # Change: validation range 10-35 → 2-35
    loaded_labels = self.load_config.get("selected_labels", [])
    self.selected_labels = [h for h in loaded_labels if 2 <= h <= 35]  # Was: 10 <= h <= 35

def on_select_all_changed(self, state):
    if state == Qt.Checked:
        # Change: select all 2-35
        self.selected_labels = list(range(2, 36))  # Was: range(10, 36)
```

**Documentation updates:**
- Update all comments mentioning "10-35" → "2-35"
- Update class docstring to reflect new range
- Keep title as "感知 Rub & Buzz" (Perceptual Rub & Buzz)

### 2. Dynamic Cumulative Masking

**Problem:** Current cumulative masking uses fixed `masking_range = (1, 9)` for harmonics 10+. This doesn't work for harmonics 2-9.

**Solution:** Dynamically adjust `masking_range` based on the harmonic being analyzed.

**Psychoacoustic principle:**
- Any harmonic is masked by **all lower-order harmonics**
- Harmonic N should have `masking_range = (1, N-1)`

**Implementation approach:**

```python
# In PerceptualStepSignalHD.compute_distortion() or compute_perceptual_thd_batch()

for each harmonic_order in analyzed_harmonics:
    if masking_config and masking_config.get('enable_cumulative'):
        # Dynamic masking range: fundamental to (current - 1)
        dynamic_masking_range = (1, harmonic_order - 1)
        masking_config['masking_range'] = dynamic_masking_range

    # Compute perceptual loudness for this harmonic...
```

**Examples:**
- **2nd harmonic**: `masking_range = (1, 1)` → only fundamental masks it
- **5th harmonic**: `masking_range = (1, 4)` → fundamental + harmonics 2-4 mask it
- **15th harmonic**: `masking_range = (1, 14)` → fundamental + harmonics 2-14 mask it

**Backward compatibility:**
- If `masking_config = None`, use current behavior (fundamental-only masking)
- If `enable_cumulative = False`, no cumulative masking

### 3. Performance Impact

**Low-order harmonics are faster:**
- 2nd harmonic: 1 masking source (fundamental only)
- 5th harmonic: 4 masking sources (1-4)
- 10th harmonic: 9 masking sources (1-9)
- 35th harmonic: 34 masking sources (1-34)

**Expected performance:**
- 2-9 harmonics: **Faster** than current 10-35 (fewer masking sources)
- 10-35 harmonics: Same as current
- Overall: Negligible impact (~2% per cumulative masking design doc)

## Implementation Plan

### Step 1: Update UI Configuration Dialog

**File:** `ui/ui_analysis_config/perceptual_rb_config_dialog.py`

**Changes:**
1. Change loop range: `range(10, 36)` → `range(2, 36)`
2. Update validation: `10 <= h <= 35` → `2 <= h <= 35`
3. Update select-all: `list(range(10, 36))` → `list(range(2, 36))`
4. Update all comments: "10-35" → "2-35"
5. Update class docstring to reflect 2-35 range

**Test:**
- UI allows selecting any combination of 2-35 harmonics
- Select-all selects all 34 harmonics (2-35)
- Previously saved configs with 10-35 still load correctly

### Step 2: Implement Dynamic Masking Range

**File:** `base/pre_processing/perceptual_step_signal_hd.py` (or `harmonic_distortion_analyzer.py`)

**Changes:**
1. In `compute_distortion()` or `compute_perceptual_thd_batch()`:
   - For each analyzed harmonic order
   - Set `masking_config['masking_range'] = (1, harmonic_order - 1)`
   - Before calling masking computation

**Pseudo-code:**
```python
def compute_distortion(..., masking_config=None):
    for frame_idx in range(n_frames):
        # Extract harmonic orders for this frame
        for harmonic_order in selected_harmonic_orders:
            # Dynamic masking range
            if masking_config and masking_config.get('enable_cumulative'):
                masking_config['masking_range'] = (1, harmonic_order - 1)

            # Compute masking threshold and perceptual loudness
            ...
```

**Edge cases:**
- 2nd harmonic: `masking_range = (1, 1)` is valid (only fundamental)
- Fundamental (1st): No masking sources (should not be analyzed in PRB)

### Step 3: Update Documentation

**Files to update:**
1. `ui/signal_analysis_window.py` - PRB class docstring
2. `docs/features/perceptual-rub-and-buzz.md` - if exists
3. Any other documentation mentioning "PRB 10-35"

**Changes:**
- Update harmonic range descriptions from "10-35" to "2-35"
- Keep PRB name unchanged
- Clarify that RB remains 10-35 (high-order only)

### Step 4: Write Tests

**New test files:**

1. **`test_prb_low_order_harmonics.py`**
   - Test harmonics 2-9 can be analyzed
   - Verify perceptual loudness values are reasonable
   - Check low-order harmonics typically have higher loudness

2. **`test_dynamic_masking_range.py`**
   - Verify 2nd harmonic: `masking_range = (1, 1)`
   - Verify 5th harmonic: `masking_range = (1, 4)`
   - Verify 15th harmonic: `masking_range = (1, 14)`
   - Check masking threshold increases with harmonic order

**Regression tests:**
- Ensure 10-35 harmonic results unchanged
- Ensure RB (10-35 range) unaffected
- Ensure HD (2-35 range) unaffected

### Step 5: Validation and Commit

**Manual testing:**
1. Open PRB config dialog → verify 2-35 harmonics selectable
2. Select harmonics 2, 5, 10, 15 → run analysis
3. Verify results display correctly
4. Check low-order harmonics show higher loudness (if signal is typical)

**Automated testing:**
```bash
pytest unit_test/base/pre_processing/test_prb_low_order_harmonics.py -v
pytest unit_test/base/pre_processing/test_dynamic_masking_range.py -v
pytest unit_test/base/pre_processing/test_perceptual_*.py -v  # All perceptual tests
```

**Commit:**
- Clear commit message describing the extension
- Reference this design document

## Acceptance Criteria

✅ PRB UI can select 2-35 harmonics (any combination)
✅ Analyzing harmonics 2-9 completes successfully and displays results
✅ Low-order harmonic perceptual loudness values are reasonable
✅ Dynamic masking range correctly set:
   - 2nd harmonic → (1, 1)
   - 5th harmonic → (1, 4)
   - 15th harmonic → (1, 14)
✅ 10-35 harmonic analysis results unchanged (regression test)
✅ RB and HD functionality unaffected
✅ All unit tests pass

## Risk Assessment

**Low risk:**
- Core calculation engine already supports arbitrary harmonics
- UI change is straightforward (range update)
- Dynamic masking is additive (doesn't break existing behavior)

**Mitigation:**
- Comprehensive regression testing for 10-35 range
- Test low-order harmonics with known signals
- Validate masking range setting with unit tests

## Future Enhancements

**Potential improvements (not in scope):**
1. Visualize masking contributions in UI (which harmonics mask which)
2. Add presets for common harmonic selections (2-5, 10-15, etc.)
3. Export masking threshold data for analysis
4. Optimize masking computation for very low harmonics (2-3)

## References

- [Cumulative Harmonic Masking Design](2025-12-05-cumulative-harmonic-masking-design.md)
- [Perceptual Rub & Buzz Feature](../features/perceptual-rub-and-buzz.md)
- ISO 226:2003 - Equal-loudness contours
- Zwicker masking model
