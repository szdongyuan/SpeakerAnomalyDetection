# Perceptual Rub & Buzz Analysis

## Overview

Perceptual Rub & Buzz (PRB) is an advanced psychoacoustic analysis that measures the perceived loudness of harmonics (2nd-35th) in phons, taking into account the human ear's frequency response and masking effects. Unlike standard Rub & Buzz (RB) which measures only high-order harmonic distortion (10th-35th) as a percentage, PRB analyzes a broader range and provides a measure of how loud these artifacts actually sound to the human ear.

## When to Use

Use Perceptual Rub & Buzz when:
- Evaluating subjective audibility of mechanical defects
- Correlating objective measurements with listening tests
- Prioritizing quality issues by perceived loudness
- Comparing units based on audible differences rather than technical measurements

## Psychoacoustic Models

### ISO 226 Equal-Loudness Contours

PRB uses ISO 226:2003 equal-loudness contours to convert sound pressure level (SPL) to phons. A phon is a unit of loudness level where:
- At 1 kHz, 1 phon = 1 dB SPL
- At other frequencies, the phon value represents the equivalent loudness at 1 kHz
- Accounts for human ear's reduced sensitivity at low and high frequencies

### Simultaneous Masking

PRB applies simultaneous masking models based on critical band theory:
- Strong fundamental tones mask nearby weaker harmonics
- Masking spreads upward in frequency more than downward
- Uses Bark scale for perceptual frequency spacing
- Harmonics below masking threshold don't contribute to perceived loudness

## Configuration

### Harmonic Selection
- **Range:** 2nd to 35th harmonics (34 harmonics total)
- **Selection:** Individual harmonics or "Select All"
- **Minimum:** At least one harmonic must be selected
- **Note:** Standard RB (Rub & Buzz) remains 10th-35th for high-order distortion analysis

### Calculation Pipeline

1. **Phase 1A:** Build harmonic index matrix (reuses standard THD architecture)
2. **Phase 1B:** Create mask for selected harmonics
3. **Phase 2:** Perceptual calculation:
   - Extract harmonic amplitudes from STFT spectrum
   - Convert amplitudes to SPL (dB)
   - Apply simultaneous masking from fundamental
   - Convert unmasked SPL to phons using ISO 226
   - Sum phons across audible harmonics

## Output

The analysis displays:
- **X-axis:** Fundamental frequency (logarithmic scale)
- **Y-axis:** Perceived loudness in phons
- **Curve:** Total perceived loudness of selected harmonics (2nd-35th range available)

## Differences from Standard Rub & Buzz

| Feature | RB (Rub & Buzz) | PRB (Perceptual Rub & Buzz) |
|---------|-----------------|----------------------------|
| Harmonic Range | 10th-35th only | 2nd-35th (full range) |
| Metric | THD percentage | Perceived loudness (phons) |
| Frequency Weighting | None | ISO 226 equal-loudness |
| Masking | None | Simultaneous masking |
| Units | % | phons |
| Typical Range | 0-10% | 0-100 phons |
| Interpretation | Technical distortion | Subjective loudness |

## Implementation Details

### Architecture

- **Psychoacoustic Utils:** `psychoacoustic_utils.py` (ISO 226 + masking models)
- **Perceptual Analyzers:** `PerceptualStepSignalHD`, `PerceptualChirpSignalHD`
- **Config Dialog:** `PerceptualRbConfigWindow`
- **Analysis Widget:** `PerceptualRubAndBuzz`
- **Backend:** `AudioThdFrequencyResponseAnalysis._calculate_perceptual_thd_three_phase()`

### Class Mapping

```python
"PRB": PerceptualRubAndBuzz
```

### Configuration Structure

```json
{
  "selected_labels": [10, 15, 20, 25, 30],
  "all_checked": false,
  "type": "PRB"
}
```

## Interpretation Guide

### Phon Ranges

- **0-10 phons:** Barely audible, likely not a quality issue
- **10-30 phons:** Audible in quiet environments, minor concern
- **30-60 phons:** Clearly audible, moderate quality issue
- **60+ phons:** Very audible, significant quality issue

### Comparison with Standard RB

A high RB value (high THD %) doesn't always mean audible distortion because:
- Low frequencies may have high technical distortion but low perceived loudness
- Harmonics may be masked by the fundamental
- Ear sensitivity varies with frequency

PRB accounts for these factors, providing a more accurate measure of subjective annoyance.

## Troubleshooting

### PRB shows low values but RB shows high values
- This is expected when harmonics are masked or at frequencies with low ear sensitivity
- The distortion exists technically but isn't audible

### PRB shows high values at specific frequencies
- These frequencies have audible high-order harmonics
- Investigate mechanical resonances or driver non-linearities at these frequencies

### Negative or zero phon values
- Harmonics are fully masked or below threshold of hearing
- This is normal and indicates no audible distortion

## References

- ISO 226:2003 - Normal equal-loudness-level contours
- Zwicker, E., & Fastl, H. (1999). Psychoacoustics: Facts and Models
- Moore, B. C. (2012). An Introduction to the Psychology of Hearing

## See Also

- [Rub & Buzz (RB)](./rub-and-buzz.md)
- Harmonic Distortion (HD)
- Three-Phase THD Architecture
