# Harmonic Distortion Algorithm Report

## Executive Summary

This report provides a complete, reproducible specification of the Harmonic Distortion (HD) analysis system. The system calculates Total Harmonic Distortion (THD) for audio signals using a **two-phase architecture**: Configuration Phase (before recording) and Calculation Phase (after recording).

**Core Formula:**
```
THD = sqrt(sum(H_i²)) / sqrt(F² + sum(H_i²)) × 100%

Where:
- F = Fundamental frequency amplitude
- H_i = i-th harmonic amplitude (i = 2, 3, 4, ...)
```

---

## 1. Correct Algorithm Path

### 1.1 Three-Phase Architecture Overview

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                PHASE 1A: BUILD OVERALL INDEX (Before Recording)                ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─────────────────────┐                                                      ║
║  │ 1. Configure        │                                                      ║
║  │    Stimulus         │                                                      ║
║  │    Metadata         │                                                      ║
║  └──────────┬──────────┘                                                      ║
║             │                                                                 ║
║             │    KNOWN BEFORE RECORDING:                                      ║
║             │    • stimulus_method, stimulus_type                             ║
║             │    • start_freq, stop_freq, num_steps                           ║
║             │    • total_time, repeat_times, sample_rate                      ║
║             │                                                                 ║
║             ▼                                                                 ║
║  ┌────────────────────────────────────────────┐                               ║
║  │ 2. Build Overall Harmonic Index Matrix    │                               ║
║  │    • ALL harmonics (fundamental + 2-35)   │                               ║
║  │    • Column 0: sentinel (all zeros)       │                               ║
║  │    • Column 1: fundamental bins (+1)      │                               ║
║  │    • Column N: Nth harmonic bins (+1)     │                               ║
║  └──────────┬─────────────────────────────────┘                               ║
║             │                                                                 ║
║             ▼                                                                 ║
║  ┌────────────────────────────────┐                                           ║
║  │ OVERALL INDEX MATRIX           │                                           ║
║  │ Shape: (n_steps, max_order+1)  │                                           ║
║  │ Reusable for ALL configurations│                                           ║
║  └────────────────────────────────┘                                           ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                        │
                        │ Stored for reuse
                        ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║             PHASE 1B: SELECT USER CONFIGURATION (Before Recording)             ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─────────────────────┐                                                      ║
║  │ 3. User Selects     │                                                      ║
║  │    Harmonics        │                                                      ║
║  │    [2, 3, 4, 5]     │                                                      ║
║  └──────────┬──────────┘                                                      ║
║             │                                                                 ║
║             ▼                                                                 ║
║  ┌────────────────────────────────────┐                                       ║
║  │ 4. Extract Selected Columns        │                                       ║
║  │    columns = [1] + [2,3,4,5]       │                                       ║
║  │    = [1, 2, 3, 4, 5]               │                                       ║
║  │    (fund + 2nd + 3rd + 4th + 5th)  │                                       ║
║  └──────────┬─────────────────────────┘                                       ║
║             │                                                                 ║
║             ▼                                                                 ║
║  ┌────────────────────────────────────┐                                       ║
║  │ 5. Convert to Binary Mask Matrix   │                                       ║
║  │    Shape: (n_bins+1, n_steps)      │                                       ║
║  │    Row 0: dummy bin (zeros)        │                                       ║
║  └──────────┬─────────────────────────┘                                       ║
║             │                                                                 ║
║             ▼                                                                 ║
║  ┌────────────────────────────────────┐                                       ║
║  │ SELECTED HARMONIC MASK             │                                       ║
║  │ (Config-specific, ready to use)    │                                       ║
║  └────────────────────────────────────┘                                       ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                        │
                        │ Mask stored for calculation
                        ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          RECORDING PHASE                                       ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║                        ┌─────────────────────────────┐                        ║
║                        │    Record Audio Signal      │                        ║
║                        │    (Play stimulus,          │                        ║
║                        │     capture response)       │                        ║
║                        └─────────────────────────────┘                        ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                       │
                                       │ recorded_signal
                                       ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    PHASE 2: CALCULATION (After Recording)                      ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─────────────────────┐    ┌─────────────────────┐    ┌──────────────────┐  ║
║  │ 1. Frequency        │───▶│ 2. Apply Pre-built  │───▶│ 3. Compute THD   │  ║
║  │    Transform        │    │    Selected Mask    │    │                  │  ║
║  │    (FFT/STFT)       │    │                     │    │                  │  ║
║  └─────────────────────┘    └─────────────────────┘    └──────────────────┘  ║
║                                                                               ║
║           ONLY recorded_signal is needed in this phase                        ║
║           Mask was already built in Phase 1B                                  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### 1.2 Information Dependencies

| Information | Source | Known When | Phase |
|-------------|--------|------------|-------|
| `stimulus_method` | Config/Database | ✅ **Before** recording | Phase 1A |
| `stimulus_type` | Config/Database | ✅ **Before** recording | Phase 1A |
| `start_freq`, `stop_freq` | Config/Database | ✅ **Before** recording | Phase 1A |
| `num_steps` | Config/Database | ✅ **Before** recording | Phase 1A |
| `total_time` | Config/Database | ✅ **Before** recording | Phase 1A |
| `sample_rate` | Config/Database | ✅ **Before** recording | Phase 1A |
| `repeat_times` | Config/Database | ✅ **Before** recording | Phase 1A |
| `n_fft` | Calculated from above | ✅ **Before** recording | Phase 1A |
| `overall_index_matrix` | Built from config | ✅ **Before** recording | **Phase 1A** |
| `harmonic_orders` | **User Selection** | ✅ **Before** recording | **Phase 1B** |
| `selected_harmonic_mask` | Extract from overall index | ✅ **Before** recording | **Phase 1B** |
| `recorded_signal` | Recording | ❌ **After** recording | Recording |

**Key Insights:**
- **Phase 1A**: The overall index matrix (ALL harmonics) depends ONLY on stimulus configuration, computed once per config.
- **Phase 1B**: The selected harmonic mask depends on overall index + user harmonic selection, built instantly via column extraction.
- **Phase 2**: THD calculation depends ONLY on recorded signal + pre-built selected mask.

---

## 2. Phase 1A: Build Overall Index Matrix (Before User Selection)

### 2.1 Configure Stimulus Metadata

```python
stimulus_metadata = {
    # From database/config - known before recording
    'stimulus_method': 'steps',        # or 'chirps'
    'stimulus_type': 'linear',         # 'linear', 'log', 'mirror_linear', 'mirror_log'
    'start_freq': 500.0,               # Hz
    'stop_freq': 2000.0,               # Hz
    'num_steps': 16,                   # For step signals
    'total_time': 4.0,                 # seconds
    'repeat_times': 3,                 # repetitions
    'sample_rate': 44100               # Hz
}
```

### 2.2 Build Overall Index Matrix (All Harmonics)

In Phase 1A, we build the **overall index matrix** containing ALL harmonics (typically up to 35th order), regardless of what the user will later select. This matrix is computed once per stimulus configuration and can be reused for any harmonic selection.

**Key Properties:**
- **Column 0**: Sentinel column (all zeros, unused)
- **Column 1**: Fundamental frequency bins (+1 offset for dummy row)
- **Column N** (N ≥ 2): Nth harmonic bins (+1 offset for dummy row)
- **Max Order**: Typically 35 harmonics (adjustable based on Nyquist constraints)
- **Reusability**: Same overall matrix works for any user harmonic selection

### 2.3 Algorithm: Build Overall Index Matrix

#### 2.3.1 For Step Signals

**Algorithm: `build_step_signal_index_matrix()`**

```
INPUT:
  - stimulus_metadata (from config)
  - sample_rate
  - n_fft (FFT size after trimming)
  - max_harmonic_order (default 35)

DERIVED PARAMETERS (all computable before recording):
  single_rep_duration = total_time / repeat_times
  step_duration = single_rep_duration / num_steps
  step_samples = int(step_duration * sample_rate)
  n_fft = step_samples - 2 * trim_samples
  n_bins = n_fft // 2 + 1

ALGORITHM:
1. GENERATE fundamental frequencies for each step:
   IF stimulus_type == 'linear':
       fundamental_freqs = linspace(start_freq, stop_freq, num_steps)
   ELIF stimulus_type == 'log':
       fundamental_freqs = logspace(log10(start_freq), log10(stop_freq), num_steps)

2. CREATE FFT frequency bins:
   fft_freqs = rfftfreq(n_fft, d=1.0/sample_rate)

3. INITIALIZE overall index matrix:
   index_matrix = zeros((num_steps, max_harmonic_order + 1), dtype=int32)
   # Column 0: sentinel (stays zero)
   # Column 1: fundamental bins
   # Column N (N≥2): Nth harmonic bins

4. FOR each step_idx, fundamental_freq in enumerate(fundamental_freqs):

   a. FIND fundamental bin and store in Column 1:
      fundamental_bin = argmin(|fft_freqs - fundamental_freq|)
      index_matrix[step_idx, 1] = fundamental_bin + 1  # +1 for dummy row offset

   b. FOR harmonic_order in range(2, max_harmonic_order + 1):  # 2nd to 35th
      harmonic_freq = fundamental_freq * harmonic_order  # Direct formula

      IF harmonic_freq < nyquist (sample_rate / 2):
          harmonic_bin = argmin(|fft_freqs - harmonic_freq|)
          index_matrix[step_idx, harmonic_order] = harmonic_bin + 1  # +1 for dummy row
      ELSE:
          # Exceeds Nyquist - stays 0 (sentinel/dummy bin)
          index_matrix[step_idx, harmonic_order] = 0

OUTPUT:
  - index_matrix: shape (num_steps, max_harmonic_order+1) - contains ALL harmonics
  - fundamental_freqs: shape (num_steps,) - for result output
  - fft_freqs: FFT frequency bins (with dummy bin prepended)
```

**Key Points:**
- Column 0 remains all zeros (sentinel)
- Column 1 stores fundamental bins (NOT Column 0)
- Column N stores Nth harmonic (direct mapping: Column 2 = 2nd harmonic, Column 3 = 3rd harmonic)
- Formula is `fundamental_freq * harmonic_order` (NO +1 in frequency calculation)
- All meaningful indices have +1 offset to account for dummy row in spectrum

#### 2.3.2 For Chirp Signals

**Algorithm: `build_chirp_signal_index_matrix()`**

```
INPUT:
  - stimulus_metadata (from config)
  - sample_rate
  - stft_window_size (default 2048)
  - stft_hop_size (default 1024)
  - max_harmonic_order (default 35)

DERIVED PARAMETERS (all computable before recording):
  single_rep_duration = total_time / repeat_times
  num_samples = int(single_rep_duration * sample_rate)
  num_frames = 1 + (num_samples - stft_window_size) // stft_hop_size
  n_fft = stft_window_size
  n_bins = n_fft // 2 + 1

ALGORITHM:
1. CREATE time array (frame center times):
   time_array = arange(num_frames) * stft_hop_size / sample_rate + (n_fft / 2) / sample_rate

2. COMPUTE instantaneous frequency at each frame:
   SWITCH stimulus_type:
       'linear':
           f(t) = start_freq + (stop_freq - start_freq) * t / duration
       'log':
           f(t) = start_freq * exp(ln(stop_freq/start_freq) * t / duration)
       'mirror_linear':
           IF t < duration/2:
               f(t) = stop_freq - (stop_freq - start_freq) * t / (duration/2)
           ELSE:
               f(t) = start_freq + (stop_freq - start_freq) * (t - duration/2) / (duration/2)
       'mirror_log':
           IF t < duration/2:
               f(t) = stop_freq * exp(-ln(stop_freq/start_freq) * t / (duration/2))
           ELSE:
               f(t) = start_freq * exp(ln(stop_freq/start_freq) * (t - duration/2) / (duration/2))

3. CREATE FFT frequency bins:
   fft_freqs = rfftfreq(n_fft, d=1.0/sample_rate)

4. INITIALIZE overall index matrix:
   index_matrix = zeros((num_frames, max_harmonic_order + 1), dtype=int32)
   # Column 0: sentinel (stays zero)
   # Column 1: fundamental bins
   # Column N (N≥2): Nth harmonic bins

5. FOR each frame_idx, fundamental_freq in enumerate(instantaneous_freqs):

   a. FIND fundamental bin and store in Column 1:
      fundamental_bin = argmin(|fft_freqs - fundamental_freq|)
      index_matrix[frame_idx, 1] = fundamental_bin + 1  # +1 for dummy row offset

   b. FOR harmonic_order in range(2, max_harmonic_order + 1):  # 2nd to 35th
      harmonic_freq = fundamental_freq * harmonic_order  # Direct formula

      IF harmonic_freq < nyquist:
          harmonic_bin = argmin(|fft_freqs - harmonic_freq|)
          index_matrix[frame_idx, harmonic_order] = harmonic_bin + 1  # +1 for dummy row
      ELSE:
          # Exceeds Nyquist - stays 0 (sentinel/dummy bin)
          index_matrix[frame_idx, harmonic_order] = 0

OUTPUT:
  - index_matrix: shape (num_frames, max_harmonic_order+1) - contains ALL harmonics
  - fundamental_freqs: shape (num_frames,)
  - time_array: shape (num_frames,)
  - fft_freqs: FFT frequency bins (with dummy bin prepended)
```

**Key Points:**
- Same structure as step signals but time-varying
- Column 0 remains all zeros (sentinel)
- Column 1 stores fundamental bins
- Column N stores Nth harmonic (N=2,3,4...35)
- Formula is `fundamental_freq * harmonic_order` (NO +1 in frequency calculation)
- All meaningful indices have +1 offset for dummy row

---

## 3. Phase 1B: Select User Configuration (Before Recording)

### 3.1 User Selects Harmonics

```python
# User configures which harmonics to analyze via UI (hd_config_dialog.py)
harmonic_orders = [2, 3, 4, 5]  # User selects 2nd through 5th harmonics
```

### 3.2 Extract Selected Columns from Overall Index Matrix

Once the user selects specific harmonic orders, we extract the corresponding columns from the overall index matrix built in Phase 1A.

**Algorithm: Extract Selected Indices**

```
INPUT:
  - index_matrix: (num_steps_or_frames, max_harmonic_order+1) from Phase 1A
  - harmonic_orders: [2, 3, 4, 5] from user selection

ALGORITHM:
1. DETERMINE columns to extract:
   columns_to_extract = [1] + harmonic_orders  # Fundamental + selected harmonics
   # For [2,3,4,5]: columns = [1, 2, 3, 4, 5]
   # Column 1 = fundamental, Column 2 = 2nd harmonic, etc.

2. EXTRACT selected columns:
   selected_indices = index_matrix[:, columns_to_extract]
   # Shape: (num_steps_or_frames, len(columns_to_extract))

OUTPUT:
  - selected_indices: Index array for fundamental + selected harmonics
```

### 3.3 Convert Indices to Binary Mask Matrix

The selected indices are converted to a binary mask matrix for efficient THD computation.

**Algorithm: `create_mask_from_indices()`**

```
INPUT:
  - selected_indices: (num_steps_or_frames, num_selected_harmonics)
  - n_bins_with_dummy: Total number of FFT bins including dummy row 0

ALGORITHM:
1. INITIALIZE mask matrix:
   mask_matrix = zeros((n_bins_with_dummy, num_steps_or_frames), dtype=float32)
   # Row 0: dummy bin (stays zero)
   # Rows 1+: actual frequency bins

2. FOR each harmonic_idx in selected harmonics:
      bin_indices = selected_indices[:, harmonic_idx]  # Bin for each step/frame
      frame_indices = arange(num_steps_or_frames)
      mask_matrix[bin_indices, frame_indices] = 1.0  # Mark selected bins

OUTPUT:
  - mask_matrix: (n_bins_with_dummy, num_steps_or_frames)
    Binary mask where 1.0 indicates fundamental or selected harmonic
```

**Key Points:**
- Instant operation via NumPy advanced indexing (<0.1ms)
- Mask ready for element-wise multiplication with spectrum
- Row 0 (dummy bin) remains all zeros
- Any harmonic exceeding Nyquist has index 0, automatically maps to dummy bin

### 3.4 Dummy Bin Technique

**Problem:** Harmonics exceeding Nyquist frequency cannot be measured but should not cause errors.

**Solution:** Add a dummy bin at index 0:
1. Add +1 offset to all bin indices
2. Prepend a dummy bin (index 0) to spectrum in calculation phase
3. Harmonics exceeding Nyquist simply don't get mask entries
4. Dummy bin always has value 0, contributing nothing to THD

**Benefit:** No special-case logic needed anywhere.

---

## 4. Phase 2: Calculation (After Recording)

### 4.1 Overview

In this phase, the mask is **already built**. Only the recorded signal processing occurs.

```python
def calculate_thd(
    recorded_signal: np.ndarray,
    pre_built_mask: MaskData  # From Phase 1
) -> Dict[str, np.ndarray]:
```

### 4.2.1 Step Signals: FFT vs STFT Approaches

**FFT Approach (with boundary trimming):**
```
1. Split repetition into steps
2. Trim boundaries: step_signal[trim_samples:-trim_samples]
3. Batch FFT on all trimmed segments
4. FFT size: step_samples - 2*trim_samples
```

**STFT Approach (no trimming, with windowing):**
```
1. Compute STFT on entire repetition signal
2. Window size: step_samples (full step duration)
3. Hop size: step_samples (no overlap → exactly one frame per step)
4. Window function: Hann (default) or other scipy.signal.stft windows
5. STFT returns exactly num_steps frames
```

Both approaches produce identical output structure: (n_bins, num_steps) spectrum matrix.

**Trade-offs:**
- FFT + trimming: Removes boundary artifacts, rectangular window (spectral leakage)
- STFT + Hann: No trimming needed, better spectral characteristics, unified with chirp processing

### 4.2 For Step Signals

**Algorithm: Calculate THD with Pre-built Mask**

```
INPUT:
  - recorded_signal: ndarray
  - pre_built_mask_data:
      - mask_matrix: (n_bins+1, num_steps)
      - fundamental_freqs: (num_steps,)
      - fundamental_bins: (num_steps,)
  - stimulus_metadata (for repetition info)

ALGORITHM:
1. SPLIT into repetitions:
   IF repeat_times > 1:
       repetition_length = len(recorded_signal) // repeat_times
       repetitions = [signal[i*rep_len:(i+1)*rep_len] for i in range(repeat_times)]
   ELSE:
       repetitions = [recorded_signal]

2. FOR each repetition_signal:

   a. SPLIT into steps:
      step_samples = len(repetition_signal) // num_steps
      FOR step_idx = 0 to num_steps - 1:
          start = step_idx * step_samples
          step_signal = repetition_signal[start : start + step_samples]
          trimmed = step_signal[trim_samples : -trim_samples]
          step_segments.append(trimmed)

   b. BATCH FFT (vectorized):
      step_matrix = zeros((max_segment_len, num_steps))
      FOR i, seg in enumerate(step_segments):
          step_matrix[:len(seg), i] = seg

      spectrum_matrix = abs(rfft(step_matrix, axis=0))  # Shape: (n_bins, num_steps)

   c. ADD dummy bin:
      spectrum_with_dummy = insert(spectrum_matrix, 0, 0.0, axis=0)  # Shape: (n_bins+1, num_steps)

   d. COMPUTE THD using pre-built mask:
      thd = compute_thd_batch(spectrum_with_dummy, mask_matrix, fundamental_bins)

   e. STORE results for this repetition

3. AVERAGE across repetitions:
   averaged_thd = average_by_frequency(thd_per_rep, freq_per_rep, tolerance=0.5Hz)

OUTPUT:
  {
    'frequencies': fundamental_freqs,
    'thd': averaged_thd,
    'num_repetitions': repeat_times
  }
```

### 4.3 For Chirp Signals

**Algorithm: Calculate THD with Pre-built Mask**

```
INPUT:
  - recorded_signal: ndarray
  - pre_built_mask_data:
      - mask_matrix: (n_bins+1, num_frames)
      - fundamental_freqs: (num_frames,)
      - time_array: (num_frames,)
      - fundamental_bins: (num_frames,)
  - STFT parameters

ALGORITHM:
1. SPLIT into repetitions (same as step signals)

2. FOR each repetition_signal:

   a. COMPUTE STFT:
      freqs, times, Zxx = scipy.signal.stft(
          repetition_signal,
          fs=sample_rate,
          window=stft_window_type,
          nperseg=stft_window_size,
          noverlap=stft_window_size - stft_hop_size,
          nfft=stft_window_size,
          return_onesided=True,
          boundary=None,
          padded=False
      )
      stft_magnitude = abs(Zxx)  # Shape: (n_bins, n_frames)

   b. ADD dummy bin:
      stft_with_dummy = insert(stft_magnitude, 0, 0.0, axis=0)

   c. ALIGN frame counts (if needed):
      num_frames = min(stft_with_dummy.shape[1], mask_matrix.shape[1])
      stft_with_dummy = stft_with_dummy[:, :num_frames]
      mask_trimmed = mask_matrix[:, :num_frames]
      fundamental_bins_trimmed = fundamental_bins[:num_frames]

   d. COMPUTE THD using pre-built mask:
      thd = compute_thd_batch(stft_with_dummy, mask_trimmed, fundamental_bins_trimmed)

   e. STORE results

3. AVERAGE across repetitions

OUTPUT:
  {
    'frequencies': fundamental_freqs,
    'thd': averaged_thd,
    'times': time_array,
    'num_repetitions': repeat_times
  }
```

### 4.4 THD Batch Computation

**Algorithm: `compute_thd_batch()`**

```
INPUT:
  - spectrum_matrix: (n_bins+1, n_steps_or_frames)  # With dummy bin
  - mask_matrix: (n_bins+1, n_steps_or_frames)       # Pre-built in Phase 1
  - fundamental_bins: (n_steps_or_frames,)           # Indices into spectrum

ALGORITHM:
1. EXTRACT fundamental amplitudes (vectorized):
   n_cols = spectrum_matrix.shape[1]
   row_indices = fundamental_bins.astype(int)
   col_indices = arange(n_cols)
   fundamental_amplitudes = spectrum_matrix[row_indices, col_indices]

2. CREATE harmonic-only mask (exclude fundamental):
   harmonic_mask = mask_matrix.copy()
   harmonic_mask[row_indices, col_indices] = 0.0

3. COMPUTE harmonic power (vectorized):
   harmonic_amplitudes_squared = (spectrum_matrix ** 2) * harmonic_mask
   harmonic_power = sum(harmonic_amplitudes_squared, axis=0)

4. COMPUTE THD (vectorized):
   fundamental_power = fundamental_amplitudes ** 2
   total_power = fundamental_power + harmonic_power
   safe_total_power = maximum(total_power, 1e-10)  # Avoid division by zero

   thd_ratio = sqrt(harmonic_power / safe_total_power)
   thd_percentage = thd_ratio * 100.0

OUTPUT: thd_percentage  # Shape: (n_steps_or_frames,)
```

---

## 5. System Architecture

### 5.1 Correct Class Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: Configuration Layer                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  HarmonicMaskConfigurator                                           │
│  ├── configure_stimulus(stimulus_metadata)                          │
│  ├── set_harmonic_selection(harmonic_orders)                        │
│  └── build_mask() -> MaskData                                       │
│                                                                     │
│  MaskData (Data Container)                                          │
│  ├── mask_matrix: ndarray                                           │
│  ├── fundamental_freqs: ndarray                                     │
│  ├── fundamental_bins: ndarray                                      │
│  ├── time_array: ndarray (chirps only)                              │
│  └── metadata: dict                                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ MaskData (pre-built)
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: Calculation Layer                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  THDCalculator                                                      │
│  ├── calculate(recorded_signal, mask_data) -> ResultDict            │
│  └── Internal methods:                                              │
│      ├── _split_repetitions()                                       │
│      ├── _compute_spectrum() (FFT or STFT)                          │
│      ├── _compute_thd_batch()                                       │
│      └── _average_repetitions()                                     │
│                                                                     │
│  ResultDict                                                         │
│  ├── frequencies: ndarray                                           │
│  ├── thd: ndarray                                                   │
│  ├── times: ndarray (chirps only)                                   │
│  └── num_repetitions: int                                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 File Locations (Current Implementation)

| Component | File Path |
|-----------|-----------|
| Entry Point | `base/pre_processing/audio_thd_frequency_response_analysis.py` |
| Base Class | `base/pre_processing/harmonic_distortion_analyzer.py` |
| Step Analyzer | `base/pre_processing/step_signal_hd.py` |
| Chirp Analyzer | `base/pre_processing/chirp_signal_hd.py` |
| Index Builder | `base/pre_processing/harmonic_index_builder.py` |
| Mask Builder | `base/pre_processing/harmonic_mask_builder.py` |

---

## 6. Data Structures

### 6.1 Stimulus Metadata (Phase 1A Input)

```python
stimulus_metadata = {
    'stimulus_method': str,      # 'steps' or 'chirps'
    'stimulus_type': str,        # 'linear', 'log', 'mirror_linear', 'mirror_log'
    'start_freq': float,         # Starting frequency in Hz
    'stop_freq': float,          # Stopping frequency in Hz
    'total_time': float,         # Total duration in seconds
    'repeat_times': int,         # Number of signal repetitions
    'sample_rate': int,          # Sample rate in Hz
    'num_steps': int,            # Number of steps (step signals only)
}
```

### 6.2 Overall Index Matrix (Phase 1A Output)

```python
# For Step Signals
overall_index_steps = {
    'index_matrix': np.ndarray,      # Shape: (num_steps, max_harmonic_order+1), int32
                                     # Column 0: sentinel (all zeros)
                                     # Column 1: fundamental bins (+1 offset)
                                     # Column N (N≥2): Nth harmonic bins (+1 offset)
    'fundamental_freqs': np.ndarray, # Shape: (num_steps,), float64
    'fft_freqs': np.ndarray,         # FFT frequency bins with dummy bin prepended
    'n_fft': int,
    'max_harmonic_order': int,       # Typically 35
}

# For Chirp Signals
overall_index_chirp = {
    'index_matrix': np.ndarray,      # Shape: (num_frames, max_harmonic_order+1), int32
                                     # Column 0: sentinel (all zeros)
                                     # Column 1: fundamental bins (+1 offset)
                                     # Column N (N≥2): Nth harmonic bins (+1 offset)
    'fundamental_freqs': np.ndarray, # Shape: (num_frames,), float64
    'time_array': np.ndarray,        # Shape: (num_frames,), float64
    'fft_freqs': np.ndarray,         # FFT frequency bins with dummy bin prepended
    'n_fft': int,
    'max_harmonic_order': int,       # Typically 35
}
```

### 6.3 Selected Harmonic Mask (Phase 1B Output / Phase 2 Input)

```python
# For Step Signals
mask_data_steps = {
    'mask_matrix': np.ndarray,      # Shape: (n_bins+1, num_steps), float32
    'fundamental_freqs': np.ndarray, # Shape: (num_steps,), float64
    'fundamental_bins': np.ndarray,  # Shape: (num_steps,), int32
    'n_fft': int,
    'stimulus_type': str,
}

# For Chirp Signals
mask_data_chirp = {
    'mask_matrix': np.ndarray,      # Shape: (n_bins+1, num_frames), float32
    'fundamental_freqs': np.ndarray, # Shape: (num_frames,), float64
    'fundamental_bins': np.ndarray,  # Shape: (num_frames,), int32
    'time_array': np.ndarray,        # Shape: (num_frames,), float64
    'n_fft': int,
    'stft_hop_size': int,
    'stimulus_type': str,
}
```

### 6.4 Result Dictionary (Phase 2 Output)

```python
result = {
    'frequencies': np.ndarray,   # Fundamental frequencies in Hz
    'thd': np.ndarray,           # THD percentages (0-100)
    'times': np.ndarray,         # Time array (chirps only)
    'num_repetitions': int,      # Number of repetitions averaged
}
```

---

## 7. Instantaneous Frequency Formulas

### 7.1 Linear Chirp

```
f(t) = f_start + (f_stop - f_start) * t / T

Where:
  - f_start = starting frequency
  - f_stop = stopping frequency
  - T = duration
  - t = time point
```

### 7.2 Logarithmic Chirp

```
f(t) = f_start * exp(ln(f_stop / f_start) * t / T)

Equivalent to:
f(t) = f_start * (f_stop / f_start)^(t/T)
```

### 7.3 Mirror Linear Chirp

```
First half (0 ≤ t < T/2):   f(t) = f_stop - (f_stop - f_start) * t / (T/2)
Second half (T/2 ≤ t ≤ T): f(t) = f_start + (f_stop - f_start) * (t - T/2) / (T/2)

Pattern: high → low → high
```

### 7.4 Mirror Logarithmic Chirp

```
First half (0 ≤ t < T/2):   f(t) = f_stop * exp(-ln(f_stop/f_start) * t / (T/2))
Second half (T/2 ≤ t ≤ T): f(t) = f_start * exp(ln(f_stop/f_start) * (t - T/2) / (T/2))

Pattern: high → low → high (logarithmic scale)
```

---

## 8. Configuration Parameters

### 8.1 Step Signal Parameters

| Parameter | Default | Description | Phase |
|-----------|---------|-------------|-------|
| `trim_samples` | 2205 | Samples to remove from step boundaries (~0.05s at 44.1kHz) | Phase 1A |
| `harmonic_orders` | User selection | Which harmonics to include [2, 3, 4, ...] | Phase 1B |

### 8.2 Chirp Signal Parameters

| Parameter | Default | Description | Phase |
|-----------|---------|-------------|-------|
| `stft_window_size` | 2048 | STFT window size in samples | Phase 1A |
| `stft_hop_size` | 1024 | STFT hop size in samples | Phase 1A |
| `stft_window_type` | 'hann' | Window function for STFT | Phase 2 |
| `harmonic_orders` | User selection | Which harmonics to include | Phase 1B |

### 8.3 Frequency Constraints

```
Nyquist frequency = sample_rate / 2 = 22050 Hz (at 44.1kHz)

Maximum fundamental frequency for each harmonic:
  - 2nd harmonic (2×): fundamental ≤ 11025 Hz
  - 3rd harmonic (3×): fundamental ≤ 7350 Hz
  - 4th harmonic (4×): fundamental ≤ 5512 Hz
  - 5th harmonic (5×): fundamental ≤ 4410 Hz
  - 10th harmonic (10×): fundamental ≤ 2205 Hz
```

---

## 9. Error Handling

### 9.1 Phase 1 Validation Errors

| Error | Condition | Message |
|-------|-----------|---------|
| `ValueError` | Invalid stimulus_method | "stimulus_method must be 'steps' or 'chirps'" |
| `ValueError` | Invalid stimulus_type | "stimulus_type must be 'linear', 'log', 'mirror_linear', or 'mirror_log'" |
| `ValueError` | Harmonic order < 2 | "Harmonic orders must be >= 2" |
| `ValueError` | num_steps <= 0 | "num_steps must be positive" |
| `ValueError` | Frequencies <= 0 | "Frequencies must be positive" |

### 9.2 Phase 2 Validation Errors

| Error | Condition | Message |
|-------|-----------|---------|
| `ValueError` | Signal length mismatch | "Signal length doesn't match expected duration" |
| `ValueError` | Step too short for trim | "Step duration too short for trimming" |

### 9.3 Numerical Safety

- Division by zero: `safe_total_power = maximum(total_power, 1e-10)`
- Empty arrays: Checked before operations
- Frame alignment: Trim to minimum of mask and spectrum frame counts

---

## 10. Performance Characteristics

### 10.1 Phase 1A (Build Overall Index) Timing

| Operation | Time |
|-----------|------|
| Build overall index matrix (16 steps, 35 harmonics) | <3ms |
| Build overall index matrix (430 frames, 35 harmonics) | <8ms |
| Find nearest bin (single) | <0.01ms |

### 10.2 Phase 1B (Select Harmonics) Timing

| Operation | Time |
|-----------|------|
| Extract selected columns from index matrix | <0.1ms |
| Convert indices to binary mask | <0.2ms |

### 10.3 Phase 2 (Calculation) Timing

| Operation | Time |
|-----------|------|
| FFT batch (16 steps) | 1-2ms |
| STFT (430 frames) | 10-20ms |
| THD batch computation | 1-3ms |
| Repetition averaging | <1ms |
| **Total (step, 3 reps)** | **15-25ms** |
| **Total (chirp, 2 reps)** | **65-135ms** |

### 10.4 Memory Usage

| Component | Size |
|-----------|------|
| Step mask (2049 bins, 16 steps) | 131KB |
| Chirp mask (1025 bins, 430 frames) | 1.7MB |
| Step spectrum matrix | 262KB |
| Chirp STFT matrix | 3.5MB |

---

## 11. Usage Examples

### 11.1 Correct Three-Phase Usage

```python
from base.pre_processing.harmonic_index_builder import HarmonicIndexBuilder
from base.pre_processing.step_signal_hd import StepSignalHD

# ═══════════════════════════════════════════════════════════════════
# PHASE 1A: Build Overall Index Matrix (BEFORE user selection)
# ═══════════════════════════════════════════════════════════════════

# Step 1: Define stimulus configuration (from database/config)
stimulus_metadata = {
    'stimulus_method': 'steps',
    'stimulus_type': 'linear',
    'start_freq': 500,
    'stop_freq': 2000,
    'num_steps': 16,
    'total_time': 4.0,
    'repeat_times': 3,
    'sample_rate': 44100
}

# Step 2: Build overall index matrix with ALL harmonics (1-35)
index_builder = HarmonicIndexBuilder()
trim_samples = 2205
single_rep_duration = stimulus_metadata['total_time'] / stimulus_metadata['repeat_times']
step_duration = single_rep_duration / stimulus_metadata['num_steps']
step_samples = int(step_duration * stimulus_metadata['sample_rate'])
n_fft = step_samples - 2 * trim_samples

# Build complete index matrix (ALL harmonics, reusable)
index_matrix, fundamental_freqs, fft_freqs = index_builder.build_step_signal_index_matrix(
    stimulus_metadata,
    sr=44100,
    n_fft=n_fft,
    max_harmonic_order=35  # Build ALL harmonics
)

# Overall index matrix is now ready and can be reused for any harmonic selection

# ═══════════════════════════════════════════════════════════════════
# PHASE 1B: Select User Configuration (BEFORE recording)
# ═══════════════════════════════════════════════════════════════════

# Step 3: User selects specific harmonics (from UI)
harmonic_orders = [2, 3, 4, 5]  # User wants 2nd, 3rd, 4th, 5th harmonics

# Step 4: Extract selected columns and convert to mask (instant operation)
columns_to_extract = [1] + harmonic_orders  # [1, 2, 3, 4, 5]
selected_indices = index_matrix[:, columns_to_extract]

# Convert to binary mask
n_bins_with_dummy = fft_freqs.shape[0]
mask_matrix = index_builder.create_mask_from_indices(
    index_matrix,
    harmonic_orders,
    n_bins_with_dummy
)

# Extract fundamental bins from column 1
fundamental_bins = index_matrix[:, 1]

# Selected mask is now ready for calculation

# ═══════════════════════════════════════════════════════════════════
# RECORDING PHASE
# ═══════════════════════════════════════════════════════════════════

# ... Play stimulus and record response ...
# recorded_signal = record_audio()

# ═══════════════════════════════════════════════════════════════════
# PHASE 2: Calculation (AFTER recording)
# ═══════════════════════════════════════════════════════════════════

analyzer = StepSignalHD(sample_rate=44100)

# Pass pre-built mask data to analyzer
result = analyzer.compute_distortion(
    recorded_signal,
    stimulus_metadata,
    harmonic_orders,
    harmonic_mask=(mask_matrix, fundamental_freqs, fundamental_bins),
    trim_samples=trim_samples
)

print(f"Frequencies: {result['frequencies']}")
print(f"THD values: {result['thd']}")
```

### 11.2 Chirp Signal Example

```python
from base.pre_processing.harmonic_index_builder import HarmonicIndexBuilder
from base.pre_processing.chirp_signal_hd import ChirpSignalHD

# ═══════════════════════════════════════════════════════════════════
# PHASE 1A: Build Overall Index Matrix
# ═══════════════════════════════════════════════════════════════════

stimulus_metadata = {
    'stimulus_method': 'chirps',
    'stimulus_type': 'log',
    'start_freq': 80,
    'stop_freq': 8000,
    'total_time': 4.0,
    'repeat_times': 2,
    'sample_rate': 44100
}

index_builder = HarmonicIndexBuilder()
stft_window_size = 2048
stft_hop_size = 1024

# Build complete index matrix
index_matrix, freqs, times, fft_freqs = index_builder.build_chirp_signal_index_matrix(
    stimulus_metadata,
    sr=44100,
    n_fft=stft_window_size,
    hop_length=stft_hop_size,
    max_harmonic_order=35
)

# ═══════════════════════════════════════════════════════════════════
# PHASE 1B: Select User Configuration
# ═══════════════════════════════════════════════════════════════════

harmonic_orders = [2, 3]

# Extract selected columns and convert to mask
columns_to_extract = [1] + harmonic_orders
selected_indices = index_matrix[:, columns_to_extract]

n_bins_with_dummy = fft_freqs.shape[0]
mask_matrix = index_builder.create_mask_from_indices(
    index_matrix,
    harmonic_orders,
    n_bins_with_dummy
)

fundamental_bins = index_matrix[:, 1]

# ═══════════════════════════════════════════════════════════════════
# RECORDING PHASE → PHASE 2: Calculation
# ═══════════════════════════════════════════════════════════════════

analyzer = ChirpSignalHD(sample_rate=44100)
result = analyzer.compute_distortion(
    recorded_signal,
    stimulus_metadata,
    harmonic_orders,
    harmonic_mask=(mask_matrix, freqs, times, fundamental_bins),
    stft_window_size=stft_window_size,
    stft_hop_size=stft_hop_size
)
```

---

## 12. Complete Data Flow Diagram

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                           COMPLETE DATA FLOW                                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝

                    ┌─────────────────────────────────────────┐
                    │         DATABASE / CONFIG                │
                    │  • stimulus_method: 'steps'              │
                    │  • stimulus_type: 'linear'               │
                    │  • start_freq: 500 Hz                    │
                    │  • stop_freq: 2000 Hz                    │
                    │  • num_steps: 16                         │
                    │  • total_time: 4.0s                      │
                    │  • repeat_times: 3                       │
                    │  • sample_rate: 44100 Hz                 │
                    └───────────────┬─────────────────────────┘
                                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌───────────────────┐     ┌─────────────────────┐
│  Derived Params    │     │                     │
│  • n_fft = 35280   │     │                     │
│  • n_bins = 17641  │     │                     │
└──────────┬─────────┘     │                     │
           │               │                     │
═══════════│═══════════════│═════════════════════│══════════════════════════
           │               │                     │
           ▼               │                     │
        ┌──────────────────────────────────────┐│
        │  PHASE 1A: BUILD OVERALL INDEX       ││
        │  (HarmonicIndexBuilder)              ││
        │                                      ││
        │  Initialize: (num_steps, 36)         ││
        │  Column 0: sentinel (all zeros)      ││
        │                                      ││
        │  FOR step = 0 to 15:                 ││
        │    fund_freq = linspace(500,2000,16) ││
        │    fund_bin = find_bin(fund_freq)    ││
        │    index[step, 1] = fund_bin + 1     ││
        │                                      ││
        │    FOR harmonic_order = 2 to 35:     ││
        │      harm_freq = fund_freq * order   ││
        │      IF harm_freq < nyquist:         ││
        │        harm_bin = find_bin(harm_freq)││
        │        index[step, order] = bin + 1  ││
        │      ELSE:                           ││
        │        index[step, order] = 0        ││
        │                                      ││
        └──────────────────┬───────────────────┘│
                           │                    │
                           ▼                    │
        ┌──────────────────────────────────────┐│
        │     OVERALL INDEX MATRIX             ││
        │                                      ││
        │  Shape: (16, 36)                     ││
        │  Column 0: sentinel (zeros)          ││
        │  Column 1: fundamental bins          ││
        │  Column N: Nth harmonic bins         ││
        │  Reusable for ANY harmonic selection ││
        └──────────────────┬───────────────────┘│
                           │                    │
                           │ Stored, reusable   │
                           │                    │
═══════════════════════════│════════════════════│══════════════════════════
                           │                    │
                           │                    ▼
                           │            ┌───────────────────┐
                           │            │   UI: HD Config    │
                           │            │   Dialog           │
                           │            │                    │
                           │            │  User selects:     │
                           │            │  [2] [3] [4] [5]   │
                           │            └────────┬──────────┘
                           │                     │
                           ▼                     ▼
        ┌──────────────────────────────────────────┐
        │  PHASE 1B: SELECT & EXTRACT              │
        │                                          │
        │  1. Extract columns: [1, 2, 3, 4, 5]     │
        │     (fundamental + selected harmonics)   │
        │                                          │
        │  2. Convert to binary mask:              │
        │     mask_matrix = zeros(17642, 16)       │
        │     FOR each selected column:            │
        │       bin_indices = index[:, col]        │
        │       mask[bin_indices, steps] = 1.0     │
        │                                          │
        │  3. Extract fundamental bins:            │
        │     fund_bins = index[:, 1]              │
        │                                          │
        └──────────────────┬───────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │      SELECTED HARMONIC MASK          │
        │                                      │
        │  mask_matrix: (17642, 16)            │
        │  fundamental_freqs: [500...2000]     │
        │  fundamental_bins: [bin indices]     │
        └──────────────────┬───────────────────┘
                           │
                           │ STORED (ready for recording)
                           │
═══════════════════════════│════════════════════════════════════════════════
                           │
                           │                    ┌───────────────────────────────┐
                           │                    │     RECORDING PHASE           │
                           │                    │                               │
                           │                    │  Play stimulus signal         │
                           │                    │  Record speaker response      │
                           │                    │                               │
                           │                    │  recorded_signal: (529200,)   │
                           │                    │  (4.0s × 3 reps × 44100 Hz)   │
                           │                    └───────────────┬───────────────┘
                           │                                    │
                           │                                    │
═══════════════════════════│════════════════════════════════════│═══════════════
                           │                                    │
                           ▼                                    ▼
        ┌──────────────────────────────────────────────────────────────────────┐
        │                   PHASE 2: CALCULATION                                │
        │                                                                      │
        │  ┌────────────────────┐   ┌────────────────────┐                     │
        │  │ Pre-built Mask     │   │ Recorded Signal    │                     │
        │  │ (from Phase 1B)    │   │ (from Recording)   │                     │
        │  └─────────┬──────────┘   └─────────┬──────────┘                     │
        │            │                        │                                │
        │            │              ┌─────────▼──────────┐                     │
        │            │              │ 1. Split Reps      │                     │
        │            │              │    3 repetitions   │                     │
        │            │              └─────────┬──────────┘                     │
        │            │                        │                                │
        │            │              ┌─────────▼──────────┐                     │
        │            │              │ 2. Split Steps     │                     │
        │            │              │    16 steps each   │                     │
        │            │              └─────────┬──────────┘                     │
        │            │                        │                                │
        │            │              ┌─────────▼──────────┐                     │
        │            │              │ 3. Trim + FFT      │                     │
        │            │              │    Batch operation │                     │
        │            │              └─────────┬──────────┘                     │
        │            │                        │                                │
        │            │              ┌─────────▼──────────┐                     │
        │            │              │ 4. Add Dummy Bin   │                     │
        │            │              │    spectrum[0] = 0 │                     │
        │            │              └─────────┬──────────┘                     │
        │            │                        │                                │
        │            └────────────────────────┼────────────────┐               │
        │                                     │                │               │
        │                           ┌─────────▼──────────┐     │               │
        │                           │ 5. Apply Mask      │◀────┘               │
        │                           │    mask * spectrum │                     │
        │                           └─────────┬──────────┘                     │
        │                                     │                                │
        │                           ┌─────────▼──────────┐                     │
        │                           │ 6. Compute THD     │                     │
        │                           │    Vectorized      │                     │
        │                           └─────────┬──────────┘                     │
        │                                     │                                │
        │                           ┌─────────▼──────────┐                     │
        │                           │ 7. Average Reps    │                     │
        │                           │                    │                     │
        │                           └─────────┬──────────┘                     │
        │                                     │                                │
        └─────────────────────────────────────┼────────────────────────────────┘
                                              │
                                              ▼
                            ┌─────────────────────────────────────┐
                            │            RESULT                    │
                            │                                     │
                            │  frequencies: [500, 600, ..., 2000] │
                            │  thd: [1.2%, 1.5%, ..., 2.1%]       │
                            │  num_repetitions: 3                 │
                            └─────────────────────────────────────┘
```

---

## 12. Summary

### 12.1 Correct Three-Phase Path

```
PHASE 1A: Build Overall Index Matrix
1. Configure Stimulus (from database/config)
       ↓
2. Build Overall Harmonic Index Matrix (ALL harmonics 1-35)
       ↓
   [Stored - reusable for any harmonic selection]

PHASE 1B: Select User Configuration
3. User Selects Harmonics (from UI, e.g., [2, 3, 4, 5])
       ↓
4. Extract Selected Columns from Overall Index (instant)
       ↓
5. Convert to Binary Mask Matrix
       ↓
   [Selected mask ready - BEFORE recording]

RECORDING PHASE
6. Record Audio Signal

PHASE 2: Calculation
7. Apply Pre-built Mask to Recorded Signal
       ↓
8. Compute THD
```

### 12.2 Key Principles

**Three-Phase Separation:**
1. **Phase 1A (Build Overall)**: Index matrix with ALL harmonics depends ONLY on stimulus configuration
2. **Phase 1B (Select)**: Mask extraction depends on overall index + user harmonic selection
3. **Phase 2 (Calculate)**: THD computation depends on recorded signal + pre-built mask

**Benefits:**
- **Reusability**: Overall index matrix built once, reused for any harmonic selection
- **Instant Selection**: Mask generation via column extraction (<0.1ms) vs full rebuild (20-40ms)
- **Pre-computation**: All configuration work done before recording
- **Clean Architecture**: Clear separation between configuration, selection, and calculation phases
- **No Caching Needed**: Index-based approach eliminates cache management overhead

---

**Document Version:** 3.0
**Generated:** 2025-11-24
**Architecture:** Three-Phase (Build Overall → Select → Calculate)
