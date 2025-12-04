# Multi-Channel Recording Implementation Plan

## Executive Summary

This plan details the implementation of multi-channel audio recording (1-4 channels) while preserving AI analysis, Spectrogram, and SPL analysis. All other analysis modules (FR, HD, LP, PD, PM, ED) will be removed. Storage locations remain unchanged, and visualizations will be adapted for multi-channel display.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Task Breakdown](#task-breakdown)
3. [Phase 1: Core Data Structure Changes](#phase-1-core-data-structure-changes)
4. [Phase 2: Recording Layer Changes](#phase-2-recording-layer-changes)
5. [Phase 3: Storage and File Management](#phase-3-storage-and-file-management)
6. [Phase 4: Analysis Module Updates](#phase-4-analysis-module-updates)
7. [Phase 5: UI and Visualization](#phase-5-ui-and-visualization)
8. [Phase 6: Configuration and Cleanup](#phase-6-configuration-and-cleanup)
9. [Testing Strategy](#testing-strategy)
10. [Risk Mitigation](#risk-mitigation)

---

## Architecture Overview

### Current State (Single Channel)
- Hardcoded `channels=1` throughout codebase
- Data structure stores 1D numpy arrays
- Analysis functions expect 1D input
- Visualization displays single plots

### Target State (Multi-Channel)
- Configurable channels (1-4)
- Data structure stores 2D numpy arrays: `shape=(samples, channels)`
- Analysis functions process per-channel
- Visualization displays multi-channel subplots/lines

### Data Flow Changes

```
BEFORE:
Audio Input → Single Channel Recording → 1D Array (samples,) → Single Analysis → Single Plot

AFTER:
Audio Input → Multi-Channel Recording → 2D Array (samples, channels) → Per-Channel Analysis → Multi-Plot Display
```

---

## Task Breakdown

### Critical Files to Modify

| Priority | File Path | Purpose | Complexity |
|----------|-----------|---------|------------|
| 🔴 High | `base/data_struct/data_deal_struct.py` | Core data storage | Low |
| 🔴 High | `base/soundcard_audio_processor.py` | Recording functions | Medium |
| 🔴 High | `base/streaming_audio_processor.py` | Streaming recording | Medium |
| 🔴 High | `base/load_config.py` | Configuration generation | Low |
| 🟡 Medium | `ui/signal_analysis_window.py` | Spectrogram & SPL classes | High |
| 🟡 Medium | `base/recording_management.py` | WAV save/load | Low |
| 🟡 Medium | `base/save_data.py` | File I/O functions | Low |
| 🟢 Low | `ui/sequence/sequence_widget.py` | Recording workflow | Medium |
| 🟢 Low | `base/play_and_record.py` | Entry point functions | Medium |

### Analysis Modules to Keep/Remove

| Module | Status | Location in Code |
|--------|--------|------------------|
| **AI** | ✅ Keep (no changes needed) | `ui/signal_analysis_window.py:60-145` |
| **Spectrogram (Spec)** | ✅ Keep & Modify | `ui/signal_analysis_window.py:557-720` |
| **SPL** | ✅ Keep & Modify | `ui/signal_analysis_window.py:147-259` |
| Frequency Response (FR) | ❌ Remove | `ui/signal_analysis_window.py:261-371` |
| Harmonic Distortion (HD) | ❌ Remove | `ui/signal_analysis_window.py:373-494` |
| Loose Particle (LP) | ❌ Remove | `ui/signal_analysis_window.py:722-808` |
| Peak Detection (PD) | ❌ Remove | `ui/signal_analysis_window.py:810-1014` |
| Pattern Match (PM) | ❌ Remove | `ui/signal_analysis_window.py:1016-1215` |
| Pipeline (ED) | ❌ Remove | `ui/signal_analysis_window.py:1217-1450` |

---

## Phase 1: Core Data Structure Changes

### File: `base/data_struct/data_deal_struct.py`

**Objective:** Update singleton to handle multi-channel arrays

**Current Implementation:**
```python
class DataDealStruct(object):
    def __init__(self):
        self.store_wave_data = None  # 1D array: (samples,)
        self.split_repeat_data = None
        self.stimulus_data = None
        self.sample_rate = None
```

**Target Implementation:**
```python
class DataDealStruct(object):
    def __init__(self):
        self.store_wave_data = None  # 2D array: (samples, channels) or 1D: (samples,) for backward compat
        self.split_repeat_data = None  # 2D array: (samples, channels) if split
        self.stimulus_data = None  # Keep 1D - stimulus is always single channel
        self.sample_rate = None
        self.num_channels = 1  # NEW: Track channel count
```

**Implementation Steps:**

1. **Add channel tracking attribute** (Line 19, after `self.sample_rate = None`)
   ```python
   self.num_channels = 1  # Default to single channel
   ```

2. **Update `clear_data()` method** (Line 29-34)
   ```python
   def clear_data(self):
       self.store_wave_data = None
       self.split_repeat_data = None
       self.num_channels = 1  # Reset to default
       self.fft_result = None
       self.stft_result = None
   ```

3. **Add helper method to detect channel count** (After line 54)
   ```python
   def update_channel_count(self):
       """Automatically update num_channels based on store_wave_data shape."""
       if self.store_wave_data is not None:
           if self.store_wave_data.ndim == 1:
               self.num_channels = 1
           elif self.store_wave_data.ndim == 2:
               self.num_channels = self.store_wave_data.shape[1]
   ```

**Verification:**
- No change to existing 1D array behavior
- 2D arrays: `shape[1]` gives channel count
- Backward compatible with existing code

---

## Phase 2: Recording Layer Changes

### 2.1 File: `base/soundcard_audio_processor.py`

**Objective:** Support multi-channel recording in blocking mode

#### Task 2.1.1: Update `sd_rec()` method (Lines 42-52)

**Current Code:**
```python
@staticmethod
def sd_rec(recorded_dict):
    num_frames = recorded_dict.get("num_frames", 441000)
    sample_rate = recorded_dict.get("sample_rate", 44100)
    channels = recorded_dict.get("channels", 1)  # Always 1
    blocking = recorded_dict.get("blocking", True)
    prolong_frames = recorded_dict.get("prolong_frames", 0)
    recorded_data = sd.rec(frames=num_frames, samplerate=sample_rate, channels=channels, blocking=blocking).T[0]  # [0] forces mono
    if prolong_frames > 0:
        recorded_data = recorded_data[prolong_frames:]
    return error_code.OK, recorded_data
```

**Target Code:**
```python
@staticmethod
def sd_rec(recorded_dict):
    num_frames = recorded_dict.get("num_frames", 441000)
    sample_rate = recorded_dict.get("sample_rate", 44100)
    channels = recorded_dict.get("channels", 1)  # Now supports 1-4
    blocking = recorded_dict.get("blocking", True)
    prolong_frames = recorded_dict.get("prolong_frames", 0)

    # Record with specified channels (sd.rec returns shape: (frames, channels))
    recorded_data = sd.rec(frames=num_frames, samplerate=sample_rate, channels=channels, blocking=blocking)

    # Handle shape based on channel count
    if channels == 1:
        recorded_data = recorded_data[:, 0]  # Convert (frames, 1) → (frames,) for backward compatibility
    # else: keep (frames, channels) shape

    # Remove prolong frames if needed
    if prolong_frames > 0:
        recorded_data = recorded_data[prolong_frames:]

    return error_code.OK, recorded_data
```

**Key Changes:**
- Line 48: Remove `.T[0]` which forced mono
- Add conditional reshaping: single channel → 1D, multi-channel → 2D

#### Task 2.1.2: Update `sd_play_rec()` method (Lines 14-24)

**Current Code:**
```python
def sd_play_rec(self, record_dict, stimulus_dict, recording_path):
    data = stimulus_dict.get("data") * stimulus_dict.get("amplitude")
    prepare_frames = record_dict.get("prepare_frames", 1000)
    prolong_frames = record_dict.get("prolong_frames", 10000)
    prolong_data = [0] * prepare_frames + list(data) + [0] * prolong_frames
    sr = stimulus_dict.get("sr")
    rec_data = sd.playrec(prolong_data, samplerate=sr, channels=1, blocking=True).T[0]  # Hardcoded channels=1
    align_frames = self.calculate_alignment(data, rec_data)
    aligned_data = rec_data[align_frames: align_frames + len(data)]
    save_audio_simple(recording_path, aligned_data, sr)
    return error_code.OK, aligned_data
```

**Target Code:**
```python
def sd_play_rec(self, record_dict, stimulus_dict, recording_path):
    data = stimulus_dict.get("data") * stimulus_dict.get("amplitude")
    prepare_frames = record_dict.get("prepare_frames", 1000)
    prolong_frames = record_dict.get("prolong_frames", 10000)
    channels = record_dict.get("channels", 1)  # NEW: Get channel count from config
    prolong_data = [0] * prepare_frames + list(data) + [0] * prolong_frames
    sr = stimulus_dict.get("sr")

    # Record multi-channel (shape: (frames, channels))
    rec_data = sd.playrec(prolong_data, samplerate=sr, channels=channels, blocking=True)

    # Alignment: use channel 0 as reference (most reliable for GCC-PHAT)
    if channels == 1:
        rec_data_mono = rec_data[:, 0]
    else:
        rec_data_mono = rec_data[:, 0]  # Use first channel for alignment

    align_frames = self.calculate_alignment(data, rec_data_mono)

    # Align all channels together
    if channels == 1:
        aligned_data = rec_data_mono[align_frames: align_frames + len(data)]
    else:
        aligned_data = rec_data[align_frames: align_frames + len(data), :]  # Keep all channels

    save_audio_simple(recording_path, aligned_data, sr)
    return error_code.OK, aligned_data
```

**Key Changes:**
- Line 20: Use configurable `channels` from `record_dict`
- Alignment: Always use channel 0 as reference (stimulus is mono)
- Preserve all channels in aligned output

---

### 2.2 File: `base/streaming_audio_processor.py`

**Objective:** Support multi-channel streaming recording

#### Task 2.2.1: Update `_audio_callback()` method (Lines 38-78)

**Current Code:**
```python
def _audio_callback(self, indata, frames, time_info, status):
    if status:
        self.logger.warning(f"Audio callback status: {status}")

    # Copy data to avoid issues with buffer reuse
    chunk = indata.copy().flatten()  # PROBLEM: flatten() destroys channel info

    # ... rest of logic ...
```

**Target Code:**
```python
def _audio_callback(self, indata, frames, time_info, status):
    if status:
        self.logger.warning(f"Audio callback status: {status}")

    # Copy data preserving channel structure
    # indata shape: (frames, channels)
    if self.num_channels == 1:
        chunk = indata.copy().flatten()  # (frames,) for single channel
    else:
        chunk = indata.copy()  # (frames, channels) for multi-channel

    # Track samples captured (always count samples, not samples*channels)
    samples_before = self.samples_captured
    chunk_length = len(chunk) if self.num_channels == 1 else chunk.shape[0]
    self.samples_captured += chunk_length

    # ... rest of alignment logic remains same ...
```

**Key Changes:**
- Remove `.flatten()` for multi-channel
- Adjust sample counting: use `shape[0]` not `len()` for 2D arrays

#### Task 2.2.2: Add channel tracking attribute (Line 34, after `self.sample_rate`)

```python
def __init__(self):
    self.logger = LogManager.set_log_handler("streaming_core")
    self.stream = None
    self.audio_queue = queue.Queue()
    self.accumulated_chunks = []
    self.is_recording = False
    self.target_samples = 0
    self.samples_captured = 0
    self.sample_rate = 44100
    self.num_channels = 1  # NEW: Track channel count
    self.error_occurred = False
    self.error_message = ""
```

#### Task 2.2.3: Update `start_streaming_rec()` method (Lines 99-150)

**Before Line 118, add:**
```python
# NEW: Extract channel count from recording config
channels = kwargs.get('channels', 1)
self.num_channels = channels
```

**Line 131-136 change to:**
```python
# Create input stream with multi-channel support
self.stream = sd.InputStream(
    samplerate=sample_rate,
    channels=self.num_channels,  # CHANGED: use instance variable
    callback=self._audio_callback,
    blocksize=2048
)
```

#### Task 2.2.4: Update `start_streaming_playrec()` method (Lines 152-248)

**Before Line 180, add:**
```python
# NEW: Extract channel count
channels = kwargs.get('channels', 1)
self.num_channels = channels
```

**Lines 225-232 change to:**
```python
# Create input stream (recording) - multi-channel
self.stream = sd.InputStream(
    samplerate=sample_rate,
    channels=self.num_channels,  # CHANGED
    callback=self._audio_callback,
    blocksize=2048,
    device=input_device['index'] if input_device else None
)
```

#### Task 2.2.5: Update `get_recorded_data()` method (Lines 274-284)

**Target Code:**
```python
def get_recorded_data(self):
    """
    Get the complete recorded audio data.

    Returns:
        np.ndarray: Complete recorded audio
            - Shape (samples,) for single channel
            - Shape (samples, channels) for multi-channel
    """
    if not self.accumulated_chunks:
        return np.array([], dtype=np.float32)

    # Concatenate along time axis (axis=0)
    if self.num_channels == 1:
        return np.concatenate(self.accumulated_chunks).astype(np.float32)
    else:
        return np.concatenate(self.accumulated_chunks, axis=0).astype(np.float32)
```

---

### 2.3 File: `base/load_config.py`

**Objective:** Generate multi-channel configuration

#### Task 2.3.1: Update `get_rec_and_play_dict_base_sequence_dict()` method (Lines 161-182)

**Current Code (Lines 176-181):**
```python
recorded_dict = {
    "channels": 1,  # HARDCODED
    "sr": data_struct.sample_rate,
    "num_frames": num_frames,
    "prolong_frames": prolong_frames,
}
```

**Target Code:**
```python
# NEW: Get channel count from sequence config or default to 1
channel_count = sequence_dict.get("channel_count", 1)

recorded_dict = {
    "channels": channel_count,  # CHANGED: dynamic channel count
    "sr": data_struct.sample_rate,
    "num_frames": num_frames,
    "prolong_frames": prolong_frames,
}
```

**Key Changes:**
- Read `channel_count` from sequence configuration
- Default to 1 for backward compatibility
- This allows per-sequence channel configuration

---

## Phase 3: Storage and File Management

### 3.1 File: `base/save_data.py`

**Objective:** Save/load multi-channel WAV files correctly

#### Task 3.1.1: Update `save_audio_simple()` function (Line 13)

**Current Implementation (assuming):**
```python
def save_audio_simple(save_path, audio, sr):
    """Save audio to WAV file."""
    audio_float = audio.astype("float32")
    wavfile.write(save_path, sr, audio_float)
```

**Target Implementation:**
```python
def save_audio_simple(save_path, audio, sr):
    """
    Save audio to WAV file with multi-channel support.

    Args:
        save_path (str): Output WAV file path
        audio (np.ndarray): Audio data
            - Shape (samples,) for mono
            - Shape (samples, channels) for multi-channel
        sr (int): Sample rate
    """
    audio_float = audio.astype("float32")

    # scipy.io.wavfile expects:
    # - 1D array (samples,) for mono
    # - 2D array (samples, channels) for multi-channel
    # Our data is already in this format, so no change needed

    wavfile.write(save_path, sr, audio_float)
```

**Verification:**
- `scipy.io.wavfile.write()` natively supports multi-channel
- No code change needed, just verification of format

---

### 3.2 File: `base/load_audio.py`

**Objective:** Load multi-channel WAV files

#### Task 3.2.1: Update loading functions (Line 124 area)

**Current Code:**
```python
# Comment: "we assume audio is mono channel"
y, sr = librosa.load(audio_path, sr=sr)
```

**Target Code:**
```python
# Load audio with multi-channel support
y, sr = librosa.load(audio_path, sr=sr, mono=False)  # mono=False preserves channels

# If y is 1D, it's mono (shape: (samples,))
# If y is 2D, it's multi-channel (shape: (channels, samples))
# Need to transpose to match our convention: (samples, channels)
if y.ndim == 2:
    y = y.T  # (channels, samples) → (samples, channels)
```

**Key Changes:**
- Set `mono=False` in `librosa.load()`
- Transpose 2D arrays to match convention

---

### 3.3 File: `base/recording_management.py`

**Objective:** Update database schema for multi-channel metadata

#### Task 3.3.1: Update `save_signal_info_to_db()` function (Lines 41-59)

**Add channel count to database entry:**

**Current Implementation (approximate):**
```python
def save_signal_info_to_db(self, file_path, product_model, sample_rate, record_date, labels, barcode, stimulus_id):
    cursor.execute("""
        INSERT INTO audio_data_table
        (file_path, product_model, sample_rate, record_date, labels, barcode, stimulus_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (file_path, product_model, sample_rate, record_date, labels, barcode, stimulus_id))
```

**Target Implementation:**
```python
def save_signal_info_to_db(self, file_path, product_model, sample_rate, record_date, labels, barcode, stimulus_id, num_channels=1):
    cursor.execute("""
        INSERT INTO audio_data_table
        (file_path, product_model, sample_rate, record_date, labels, barcode, stimulus_id, num_channels)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (file_path, product_model, sample_rate, record_date, labels, barcode, stimulus_id, num_channels))
```

**Database Migration:**
Add column to existing table:
```sql
ALTER TABLE audio_data_table ADD COLUMN num_channels INTEGER DEFAULT 1;
```

**Note:** This requires database migration script or manual update

---

## Phase 4: Analysis Module Updates

### 4.1 File: `ui/signal_analysis_window.py` - Spectrogram Class

**Objective:** Display 4-subplot spectrogram for multi-channel

#### Task 4.1.1: Update `calculate_spec()` method (Lines 613-719)

**Current Structure:**
```python
def calculate_spec(self):
    recorded_signal = self.data_struct.store_wave_data  # 1D array
    sample_rate = self.data_struct.sample_rate

    # ... calculate single spectrogram ...

    # Display single plot
    self.plot_container_layout.addWidget(cqt_plot_widget)
```

**Target Structure:**
```python
def calculate_spec(self):
    recorded_signal = self.data_struct.store_wave_data
    sample_rate = self.data_struct.sample_rate
    num_channels = self.data_struct.num_channels

    # Clear existing plots
    self._clear_plot_container()

    # Handle single vs multi-channel
    if num_channels == 1:
        # Existing single-channel logic (backward compatible)
        signal_1d = recorded_signal if recorded_signal.ndim == 1 else recorded_signal[:, 0]
        self._plot_single_spectrogram(signal_1d, sample_rate, channel_index=0)
    else:
        # Multi-channel: create subplot grid
        self._plot_multi_channel_spectrogram(recorded_signal, sample_rate, num_channels)

    self.set_color_font_size()
```

#### Task 4.1.2: Add helper method `_clear_plot_container()` (After line 580)

```python
def _clear_plot_container(self):
    """Remove all widgets from plot container."""
    while self.plot_container_layout.count():
        item = self.plot_container_layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
```

#### Task 4.1.3: Refactor single-channel plotting (New method after line 719)

```python
def _plot_single_spectrogram(self, signal, sample_rate, channel_index):
    """
    Plot spectrogram for a single channel.

    Args:
        signal (np.ndarray): 1D audio signal
        sample_rate (int): Sample rate
        channel_index (int): Channel number (0-based)
    """
    # Extract config
    n_fft = self.analysis_config.get("n_fft", 2048)
    hop_length = self.analysis_config.get("hop_length", 256)
    color_map = self.analysis_config.get("color_map", "viridis")
    window_func = self.analysis_config.get("window_func", "hann")
    freq_scale_type = self.analysis_config.get("freq_scale_type", "linear")
    top_limit = self.analysis_config.get("top_limit", 70)
    bottom_limit = self.analysis_config.get("bottom_limit", 50)
    custom_limit_flag = self.analysis_config.get("custom_limit", False)

    mid_value = (top_limit - bottom_limit) / 2
    max_value = top_limit + mid_value
    min_value = bottom_limit - mid_value

    # Calculate spectrogram
    if freq_scale_type == "log":
        # CQT (log scale) - COPY existing code from lines 630-666
        fmin_cqt = librosa.note_to_hz("C1")
        CQT_complex, freqs, times = AudioThdFrequencyResponseAnalysis().compute_cqt(
            y=signal, sr=sample_rate, hop_length=hop_length, n_fft=n_fft, fmin=fmin_cqt
        )
        CQT_mag = np.abs(CQT_complex)
        CQT_db = librosa.amplitude_to_db(CQT_mag, ref=20e-6)
        Z = CQT_db.T

        # ... tick setup code ...

        plot_widget, colorbar = plot_2d_image(
            x=times, y=freqs, z=Z,
            title=f"Spectrogram (Log Scale) - Channel {channel_index + 1}",
            xlabel="Time (s)", ylabel="Frequency (Hz)",
            colormap=color_map,
            x_range=(times.min(), times.max()),
            y_range=(freqs.min(), freqs.max()),
            background_color="white"
        )
    else:
        # STFT (linear scale) - COPY existing code from lines 668-714
        spec = np.abs(librosa.stft(y=signal, n_fft=n_fft, hop_length=hop_length, window=window_func))
        spec_dB = librosa.amplitude_to_db(spec, ref=20e-6)
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
        times = librosa.times_like(spec_dB, sr=sample_rate, hop_length=hop_length)

        # ... existing plotting code ...
        plot_widget = self.stft_plot_widget
        plot_widget.setTitle(f"Spectrogram (Linear Scale) - Channel {channel_index + 1}")

    # Add to layout
    self.plot_container_layout.addWidget(plot_widget)
    self.current_plot_widget = plot_widget

    # Apply custom limits
    if custom_limit_flag and hasattr(self, 'stft_colorbar') and self.stft_colorbar:
        self.stft_colorbar.setLevels((min_value, max_value))
```

#### Task 4.1.4: Add multi-channel plotting method (New method after previous)

```python
def _plot_multi_channel_spectrogram(self, multi_signal, sample_rate, num_channels):
    """
    Plot spectrograms for multiple channels in a grid layout.

    Args:
        multi_signal (np.ndarray): 2D array (samples, channels)
        sample_rate (int): Sample rate
        num_channels (int): Number of channels (1-4)
    """
    # Create matplotlib figure with subplots
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    # Extract config
    n_fft = self.analysis_config.get("n_fft", 2048)
    hop_length = self.analysis_config.get("hop_length", 256)
    color_map = self.analysis_config.get("color_map", "viridis")
    window_func = self.analysis_config.get("window_func", "hann")
    freq_scale_type = self.analysis_config.get("freq_scale_type", "linear")

    # Determine subplot layout: 2x2 for 3-4 channels, 2x1 for 2 channels, 1x1 for 1 channel
    if num_channels <= 2:
        nrows, ncols = num_channels, 1
    else:
        nrows, ncols = 2, 2

    fig = Figure(figsize=(12, 8), facecolor='white')
    canvas = FigureCanvas(fig)

    for ch_idx in range(num_channels):
        ax = fig.add_subplot(nrows, ncols, ch_idx + 1)

        # Extract channel signal
        signal_ch = multi_signal[:, ch_idx]

        # Calculate spectrogram
        if freq_scale_type == "log":
            fmin_cqt = librosa.note_to_hz("C1")
            CQT_complex, freqs, times = AudioThdFrequencyResponseAnalysis().compute_cqt(
                y=signal_ch, sr=sample_rate, hop_length=hop_length, n_fft=n_fft, fmin=fmin_cqt
            )
            CQT_db = librosa.amplitude_to_db(np.abs(CQT_complex), ref=20e-6)

            # Plot with log scale
            im = ax.pcolormesh(times, freqs, CQT_db, shading='auto', cmap=color_map)
            ax.set_yscale('log')
            ax.set_ylabel('Frequency (Hz)')
        else:
            spec = np.abs(librosa.stft(y=signal_ch, n_fft=n_fft, hop_length=hop_length, window=window_func))
            spec_dB = librosa.amplitude_to_db(spec, ref=20e-6)
            freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
            times = librosa.times_like(spec_dB, sr=sample_rate, hop_length=hop_length)

            # Plot with linear scale
            im = ax.pcolormesh(times, freqs, spec_dB, shading='auto', cmap=color_map)
            ax.set_ylabel('Frequency (Hz)')

        ax.set_xlabel('Time (s)')
        ax.set_title(f'Channel {ch_idx + 1}')
        fig.colorbar(im, ax=ax, label='Magnitude (dB)')

    fig.tight_layout()

    # Add canvas to layout
    self.plot_container_layout.addWidget(canvas)
    self.current_plot_widget = canvas
```

**Key Design Decisions:**
- Use matplotlib for multi-channel (easier subplot management)
- Keep PyQtGraph for single-channel (existing functionality)
- 2x2 grid layout for 3-4 channels
- Each subplot shows one channel's spectrogram

---

### 4.2 File: `ui/signal_analysis_window.py` - SPL Class

**Objective:** Display single plot with 4 colored lines for multi-channel SPL

#### Task 4.2.1: Update `calculate_spl()` method (Lines 156-188)

**Current Structure:**
```python
def calculate_spl(self):
    recorded_signal = self.data_struct.store_wave_data  # 1D
    sample_rate = self.data_struct.sample_rate
    signal_duration = np.linspace(0, len(recorded_signal) / sample_rate, len(recorded_signal))

    # Calculate SPL for single channel
    signal_spl = AudioThdFrequencyResponseAnalysis().spl_calculation(
        recorded_signal, reference_pressure, deviation=self.deviation_value
    )

    # ... smoothing and plotting ...
```

**Target Structure:**
```python
def calculate_spl(self):
    recorded_signal = self.data_struct.store_wave_data
    sample_rate = self.data_struct.sample_rate
    num_channels = self.data_struct.num_channels
    reference_pressure = 20e-6

    # Handle single vs multi-channel
    if num_channels == 1:
        # Existing single-channel logic (backward compatible)
        signal_1d = recorded_signal if recorded_signal.ndim == 1 else recorded_signal[:, 0]
        signal_duration = np.linspace(0, len(signal_1d) / sample_rate, len(signal_1d))

        signal_spl = AudioThdFrequencyResponseAnalysis().spl_calculation(
            signal_1d, reference_pressure, deviation=self.deviation_value
        )

        if self.analysis_config["smooth_checked"]:
            signal_spl = smooth(signal_spl, window_size=1102, method="rms")

        # Existing plotting logic
        self._plot_single_spl(signal_duration, signal_spl)

        self.result = {
            "signal_duration": signal_duration.tolist(),
            "recorded_signal": signal_1d.tolist(),
            "signal_spl": signal_spl.tolist(),
        }
    else:
        # Multi-channel: calculate SPL for each channel
        signal_duration = np.linspace(0, recorded_signal.shape[0] / sample_rate, recorded_signal.shape[0])

        spl_channels = []
        for ch_idx in range(num_channels):
            signal_ch = recorded_signal[:, ch_idx]
            spl_ch = AudioThdFrequencyResponseAnalysis().spl_calculation(
                signal_ch, reference_pressure, deviation=self.deviation_value
            )
            if self.analysis_config["smooth_checked"]:
                spl_ch = smooth(spl_ch, window_size=1102, method="rms")
            spl_channels.append(spl_ch)

        # Plot all channels on same axes
        self._plot_multi_channel_spl(signal_duration, spl_channels, num_channels)

        self.result = {
            "signal_duration": signal_duration.tolist(),
            "recorded_signal": recorded_signal.tolist(),
            "signal_spl_channels": [spl.tolist() for spl in spl_channels],
        }

    return self.result
```

#### Task 4.2.2: Refactor single-channel SPL plotting (New method after line 258)

```python
def _plot_single_spl(self, signal_duration, signal_spl):
    """
    Plot SPL for a single channel (existing logic).

    Args:
        signal_duration (np.ndarray): Time array
        signal_spl (np.ndarray): SPL values
    """
    limit_checked = self.analysis_config.get("limit_checked")
    self_defined = self.analysis_config.get("self_defined")

    if limit_checked:
        if self_defined:
            upper_limit = self.analysis_config.get("upper_limit")
            lower_limit = self.analysis_config.get("lower_limit")
            self.plot_spl(signal_duration, signal_spl, upper_limit=upper_limit, lower_limit=lower_limit)
        else:
            excel_path = self.analysis_config.get("config_dir")
            result = Frequency.load_excel_limit(excel_path)
            if not result:
                return False
            csv_time_list, csv_upper_list, csv_lower_list = result
            self.plot_spl_with_limits(signal_duration, signal_spl, csv_time_list, csv_upper_list, csv_lower_list)
    else:
        self.plot_spl(signal_duration, signal_spl)
```

#### Task 4.2.3: Add multi-channel SPL plotting method (New method)

```python
def _plot_multi_channel_spl(self, signal_duration, spl_channels, num_channels):
    """
    Plot SPL for multiple channels on the same axes with different colors.

    Args:
        signal_duration (np.ndarray): Time array (same for all channels)
        spl_channels (list of np.ndarray): SPL values for each channel
        num_channels (int): Number of channels
    """
    self.analysis_plot.clear()

    # Define colors for up to 4 channels
    channel_colors = [
        (51, 196, 77),   # Green - Channel 1
        (77, 144, 255),  # Blue - Channel 2
        (255, 153, 51),  # Orange - Channel 3
        (204, 51, 255),  # Purple - Channel 4
    ]

    # Plot each channel with different color
    for ch_idx in range(num_channels):
        color = channel_colors[ch_idx]
        pen = mkPen(color=color, width=2)
        self.analysis_plot.plot(
            signal_duration,
            spl_channels[ch_idx],
            pen=pen,
            name=f'Channel {ch_idx + 1}'  # For legend
        )

    # Add legend
    self.analysis_plot.addLegend()

    # Labels and grid
    self.analysis_plot.setLabel("left", "SPL (dB)")
    self.analysis_plot.setLabel("bottom", "Time (s)")
    self.analysis_plot.showGrid(x=True, y=True)

    # Optional: Add limit lines (use first channel's limits as reference)
    limit_checked = self.analysis_config.get("limit_checked")
    if limit_checked:
        self_defined = self.analysis_config.get("self_defined")
        if self_defined:
            upper_limit = float(self.analysis_config.get("upper_limit"))
            lower_limit = float(self.analysis_config.get("lower_limit"))

            dashed_pen = mkPen(color=(128, 0, 128), width=1, style=Qt.DashLine)
            lower_limit_line = pg.InfiniteLine(angle=0, pos=lower_limit, pen=dashed_pen)
            upper_limit_line = pg.InfiniteLine(angle=0, pos=upper_limit, pen=dashed_pen)
            self.analysis_plot.addItem(lower_limit_line)
            self.analysis_plot.addItem(upper_limit_line)
```

**Key Design Decisions:**
- All channels on same plot (easier comparison)
- Distinct colors per channel (up to 4)
- Shared time axis
- Optional: Limit lines apply to all channels (or remove for multi-channel)
- Legend shows channel labels

---

### 4.3 Remove Unused Analysis Modules

**Objective:** Clean up code by removing FR, HD, LP, PD, PM, ED classes

#### Task 4.3.1: Delete class definitions from `ui/signal_analysis_window.py`

**Lines to delete:**
- `class Frequency` (lines 261-371) - Frequency Response
- `class Distortion` (lines 373-494) - Harmonic Distortion
- `class LooseParticle` (lines 722-808) - Loose Particle Detection
- `class PeakDetection` (lines 810-1014) - Peak Detection
- `class PatternMatch` (lines 1016-1215) - Pattern Matching
- `class PipelinePdPm` (lines 1217-1450) - Pipeline (PD+PM)

#### Task 4.3.2: Update analysis type mapping (Lines 34-55)

**Current Code:**
```python
analysis_class_map = {
    "SPL": Spl,
    "FR": Frequency,     # DELETE
    "HD": Distortion,    # DELETE
    "AI": AI,
    "Spec": Spectrogram,
    "LP": LooseParticle, # DELETE
    "PD": PeakDetection, # DELETE
    "PM": PatternMatch,  # DELETE
    "ED": PipelinePdPm,  # DELETE
}
```

**Target Code:**
```python
analysis_class_map = {
    "SPL": Spl,
    "AI": AI,
    "Spec": Spectrogram,
}
```

#### Task 4.3.3: Remove related UI configuration dialogs

**Files to check and clean:**
- `ui/ui_analysis_config/fr_config_dialog.py` - DELETE if exists
- `ui/ui_analysis_config/hd_config_dialog.py` - DELETE if exists
- `ui/ui_analysis_config/lp_config_dialog.py` - DELETE if exists
- `ui/ui_analysis_config/pd_config_dialog.py` - DELETE if exists
- `ui/ui_analysis_config/pm_config_dialog.py` - DELETE if exists

**Note:** Only delete if these files are exclusively for removed modules

---

## Phase 5: UI and Visualization

### 5.1 File: `ui/sequence/sequence_widget.py`

**Objective:** Update recording workflow for multi-channel

#### Task 5.1.1: Update recording data storage (Line 906 area, in `_on_streaming_complete()`)

**Current Code (approximate):**
```python
def _on_streaming_complete(self):
    recorded_data = self.streaming_processor.get_recorded_data()
    self.data_struct.store_wave_data = recorded_data  # Store 1D array
    self.data_struct.sample_rate = self.sample_rate
    # ... rest of processing ...
```

**Target Code:**
```python
def _on_streaming_complete(self):
    recorded_data = self.streaming_processor.get_recorded_data()
    self.data_struct.store_wave_data = recorded_data  # Now stores 1D or 2D array
    self.data_struct.sample_rate = self.sample_rate
    self.data_struct.update_channel_count()  # NEW: Update channel count from data shape
    # ... rest of processing ...
```

#### Task 5.1.2: Update waveform plotting for multi-channel (Line 888 area, in `_poll_streaming_queue()`)

**Current Logic:**
- Real-time waveform updates during streaming
- Expects 1D signal

**Target Logic:**
- For single channel: existing behavior
- For multi-channel: plot all channels or first channel only (performance)

**Implementation:**
```python
def _update_waveform_plot(self, chunk):
    """Update waveform plot with new audio chunk."""
    # Handle multi-channel chunks
    if chunk.ndim == 1:
        # Single channel - existing logic
        self._append_to_waveform_plot(chunk)
    else:
        # Multi-channel - plot only channel 0 for performance
        self._append_to_waveform_plot(chunk[:, 0])
```

**Design Decision:** Only plot channel 0 during streaming for performance. Full multi-channel visualization happens after recording completes.

---

### 5.2 File: `ui/graph_widget.py`

**Objective:** Verify 2D image plotting supports multi-channel

#### Task 5.2.1: Review `plot_2d_image()` function

**Expected:** Function should already support any 2D array input (time x frequency)

**Verification needed:**
- Confirm colormap rendering works for multi-channel subplots
- Ensure axis scaling is correct

**No code changes expected** - just verification pass

---

## Phase 6: Configuration and Cleanup

### 6.1 Add Channel Configuration to UI

**Objective:** Allow users to select 1-4 channels in recording settings

#### Task 6.1.1: Add channel selector to recording configuration UI

**File:** `ui/recording_config_dialog.py` (or equivalent settings UI)

**Add UI Element:**
```python
# Channel count selector (QSpinBox or QComboBox)
self.channel_count_spinbox = QSpinBox()
self.channel_count_spinbox.setRange(1, 4)  # 1 to 4 channels
self.channel_count_spinbox.setValue(1)  # Default to 1
self.channel_count_spinbox.setSuffix(" channels")

# Add to layout
form_layout.addRow("Recording Channels:", self.channel_count_spinbox)
```

**Save to Configuration:**
```python
config["channel_count"] = self.channel_count_spinbox.value()
```

#### Task 6.1.2: Update sequence configuration schema

**File:** `ui/sequence/sequence_config.json` (or equivalent)

**Add field:**
```json
{
  "sequence_name": "Multi-Channel Test",
  "channel_count": 4,
  "stimulus_id": "...",
  "analysis_sequence": ["AI", "Spec", "SPL"]
}
```

---

### 6.2 Database Migration

**Objective:** Add `num_channels` column to existing database

#### Task 6.2.1: Create migration script

**File:** `database/migrate_add_channels.py` (NEW FILE)

```python
import sqlite3
from consts.model_consts import DEFAULT_DIR

def migrate_database():
    """Add num_channels column to audio_data_table."""
    db_path = DEFAULT_DIR + "database/audio_data.db"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if column already exists
        cursor.execute("PRAGMA table_info(audio_data_table)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'num_channels' not in columns:
            print("Adding num_channels column...")
            cursor.execute("""
                ALTER TABLE audio_data_table
                ADD COLUMN num_channels INTEGER DEFAULT 1
            """)
            conn.commit()
            print("Migration successful!")
        else:
            print("Column num_channels already exists.")

    except Exception as e:
        print(f"Migration failed: {e}")
        conn.rollback()

    finally:
        conn.close()

if __name__ == "__main__":
    migrate_database()
```

**Execution:** Run once before deploying multi-channel code

---

### 6.3 Update Analysis Configuration

**Objective:** Ensure Spec and SPL config dialogs work with multi-channel

#### Task 6.3.1: Verify `ui/ui_analysis_config/spec_config_dialog.py`

**No changes expected** - spectrogram settings are per-channel, same config applies to all

#### Task 6.3.2: Verify `ui/ui_analysis_config/spl_config_dialog.py`

**Consideration:** Should SPL limits apply to all channels or per-channel?

**Decision:** Apply same limits to all channels (simpler UX)

**No code changes** - existing limit logic works

---

## Phase 7: Testing Strategy

### 7.1 Unit Tests

#### Test 7.1.1: Data Structure Tests
```python
def test_data_struct_single_channel():
    data_struct = DataDealStruct()
    data_struct.store_wave_data = np.random.randn(44100)  # 1D
    data_struct.update_channel_count()
    assert data_struct.num_channels == 1

def test_data_struct_multi_channel():
    data_struct = DataDealStruct()
    data_struct.store_wave_data = np.random.randn(44100, 4)  # 2D, 4 channels
    data_struct.update_channel_count()
    assert data_struct.num_channels == 4
```

#### Test 7.1.2: Recording Tests
```python
def test_sd_rec_single_channel():
    recorded_dict = {"channels": 1, "num_frames": 44100, "sample_rate": 44100}
    status, data = SoundcardAudioProcessor.sd_rec(recorded_dict)
    assert data.ndim == 1  # Should be 1D
    assert len(data) == 44100

def test_sd_rec_multi_channel():
    recorded_dict = {"channels": 4, "num_frames": 44100, "sample_rate": 44100}
    status, data = SoundcardAudioProcessor.sd_rec(recorded_dict)
    assert data.ndim == 2  # Should be 2D
    assert data.shape == (44100, 4)
```

#### Test 7.1.3: File I/O Tests
```python
def test_save_load_multi_channel_wav():
    # Generate test data
    test_data = np.random.randn(44100, 4).astype('float32')
    save_path = "/tmp/test_4ch.wav"

    # Save
    save_audio_simple(save_path, test_data, 44100)

    # Load
    loaded_data, sr = load_audio(save_path)

    # Verify
    assert loaded_data.shape == (44100, 4)
    np.testing.assert_array_almost_equal(test_data, loaded_data, decimal=5)
```

---

### 7.2 Integration Tests

#### Test 7.2.1: End-to-End Recording Flow
1. Set channel count to 4 in UI
2. Start streaming recording
3. Verify `store_wave_data` has shape `(samples, 4)`
4. Trigger Spec analysis
5. Verify 4 subplots are displayed
6. Trigger SPL analysis
7. Verify 4 colored lines are displayed

#### Test 7.2.2: Backward Compatibility
1. Set channel count to 1
2. Perform recording
3. Verify `store_wave_data` is 1D
4. Verify Spec shows single plot (not subplot)
5. Verify SPL shows single line

---

### 7.3 Manual Test Cases

| Test Case | Steps | Expected Result |
|-----------|-------|-----------------|
| **TC-01: Single Channel** | 1. Set channels=1<br>2. Record<br>3. Run Spec | Single spectrogram plot displayed |
| **TC-02: Dual Channel** | 1. Set channels=2<br>2. Record<br>3. Run Spec | 2x1 subplot layout with 2 spectrograms |
| **TC-03: Quad Channel** | 1. Set channels=4<br>2. Record<br>3. Run Spec | 2x2 subplot layout with 4 spectrograms |
| **TC-04: SPL Multi-Channel** | 1. Set channels=4<br>2. Record<br>3. Run SPL | Single plot with 4 colored lines |
| **TC-05: File Storage** | 1. Record 4-channel<br>2. Check WAV file | WAV file has 4 channels, plays correctly |
| **TC-06: AI Analysis** | 1. Record 4-channel<br>2. Run AI | AI analysis runs (no errors) |
| **TC-07: Database** | 1. Record 4-channel<br>2. Check DB | `num_channels=4` stored in database |

---

## Risk Mitigation

### Risk 1: Breaking Existing Single-Channel Recordings
**Mitigation:**
- Keep 1D array support throughout codebase
- Default `channels=1` in all configs
- Extensive backward compatibility testing

### Risk 2: Performance Degradation with Multi-Channel
**Mitigation:**
- Use numpy vectorized operations (no loops)
- Only plot channel 0 during streaming
- Full multi-channel visualization only after recording completes

### Risk 3: Alignment Issues in Multi-Channel Play+Record
**Mitigation:**
- Use channel 0 as alignment reference (most reliable)
- Apply same alignment offset to all channels
- Test with known delays to verify

### Risk 4: Storage Space for Multi-Channel Files
**Impact:** 4-channel files are 4x larger
**Mitigation:**
- Monitor disk usage
- Consider compression for archival (not implemented in this phase)
- Document storage requirements for users

---

## Implementation Order

### Week 1: Core Infrastructure
1. ✅ Phase 1: Data Structure Changes (1 day)
2. ✅ Phase 2: Recording Layer Changes (2 days)
3. ✅ Phase 3: Storage and File Management (1 day)
4. ✅ Testing: Unit tests for core components (1 day)

### Week 2: Analysis and Visualization
1. ✅ Phase 4.1: Spectrogram Multi-Channel (2 days)
2. ✅ Phase 4.2: SPL Multi-Channel (1 day)
3. ✅ Phase 4.3: Remove Unused Modules (1 day)
4. ✅ Testing: Integration tests (1 day)

### Week 3: UI and Polish
1. ✅ Phase 5: UI Updates (2 days)
2. ✅ Phase 6.1: Configuration UI (1 day)
3. ✅ Phase 6.2: Database Migration (0.5 day)
4. ✅ Testing: Manual test cases (1.5 days)

---

## Success Criteria

### Functional Requirements
- ✅ Support 1-4 channel recording
- ✅ Spectrogram displays 4 subplots (2x2 grid)
- ✅ SPL displays 4 lines on single plot
- ✅ AI analysis continues to work (no changes)
- ✅ Storage locations unchanged
- ✅ Backward compatible with existing single-channel recordings

### Non-Functional Requirements
- ✅ No performance degradation vs single-channel
- ✅ UI remains responsive during multi-channel recording
- ✅ File sizes are reasonable (< 50MB for 10s @ 4ch)
- ✅ Database migration completes without data loss

---

## Appendix A: File Modification Summary

| File | Lines Changed | Complexity | Time Estimate |
|------|---------------|------------|---------------|
| `base/data_struct/data_deal_struct.py` | +10 | Low | 30 min |
| `base/soundcard_audio_processor.py` | +25 | Medium | 2 hours |
| `base/streaming_audio_processor.py` | +30 | Medium | 2 hours |
| `base/load_config.py` | +3 | Low | 15 min |
| `base/save_data.py` | +5 | Low | 30 min |
| `base/load_audio.py` | +5 | Low | 30 min |
| `base/recording_management.py` | +10 | Low | 1 hour |
| `ui/signal_analysis_window.py` (Spec) | +150 | High | 6 hours |
| `ui/signal_analysis_window.py` (SPL) | +80 | Medium | 3 hours |
| `ui/signal_analysis_window.py` (Delete) | -600 | Low | 1 hour |
| `ui/sequence/sequence_widget.py` | +15 | Medium | 2 hours |
| `database/migrate_add_channels.py` | +30 | Low | 1 hour |

**Total Estimated Time:** ~20 hours (2.5 days of focused development)

---

## Appendix B: Configuration Examples

### Example 1: Single-Channel Configuration (Backward Compatible)
```json
{
  "sequence_name": "Standard Test",
  "channel_count": 1,
  "stimulus_id": "sweep_100_10k",
  "analysis_sequence": ["AI", "Spec", "SPL"]
}
```

### Example 2: Quad-Channel Configuration
```json
{
  "sequence_name": "Multi-Channel Test",
  "channel_count": 4,
  "stimulus_id": "sweep_100_10k",
  "analysis_sequence": ["Spec", "SPL", "AI"]
}
```

---

## Appendix C: Database Schema Update

### Before Migration:
```sql
CREATE TABLE audio_data_table (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT,
    product_model TEXT,
    sample_rate INTEGER,
    record_date TEXT,
    labels TEXT,
    barcode TEXT,
    stimulus_id TEXT
);
```

### After Migration:
```sql
CREATE TABLE audio_data_table (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT,
    product_model TEXT,
    sample_rate INTEGER,
    record_date TEXT,
    labels TEXT,
    barcode TEXT,
    stimulus_id TEXT,
    num_channels INTEGER DEFAULT 1  -- NEW COLUMN
);
```

---

## Appendix D: Data Shape Reference

| Context | Single Channel | Multi-Channel (4) |
|---------|----------------|-------------------|
| **Recording Output** | `(samples,)` | `(samples, 4)` |
| **DataDealStruct.store_wave_data** | `(samples,)` | `(samples, 4)` |
| **WAV File** | Mono | 4-channel |
| **Spectrogram Input** | `(samples,)` | `(samples, 4)` → split to 4x `(samples,)` |
| **SPL Input** | `(samples,)` | `(samples, 4)` → split to 4x `(samples,)` |
| **Librosa Load Output** | `(samples,)` | `(samples, 4)` after transpose |

---

## Contact and Questions

For implementation questions or clarifications, refer to:
- **Current Recording Architecture**: See "Architecture Overview" section above
- **Data Flow**: See diagrams in Phase 1-2
- **Visualization Details**: See Phase 4 (Spectrogram/SPL)

---

**END OF IMPLEMENTATION PLAN**
