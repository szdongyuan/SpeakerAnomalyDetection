# Speaker Anomaly Detection

A comprehensive audio analysis system for speaker testing and quality control.

## Analysis Features

- **SPL**: Sound Pressure Level analysis
- **FR**: Frequency Response analysis
- **HD**: Harmonic Distortion (2nd-35th harmonics)
- **RB**: Rub & Buzz (high-order harmonic distortion, 10th-35th harmonics)
- **PRB**: Perceptual Rub & Buzz (psychoacoustic loudness in phons, 2nd-35th harmonics)
- **AI**: Machine learning-based anomaly detection
- **Spec**: Spectrogram visualization
- **LP**: Loose Particle detection
- **PD**: Peak Detection
- **PM**: Pattern Matching
- **ED**: Pipeline (Peak Detection + Pattern Matching)

## Getting Started

### Requirements
- Python 3.x
- PyQt5
- NumPy
- SciPy

### Installation
```bash
pip install -r requirements.txt
```

### Running Tests
```bash
PYTHONPATH=/Users/fh/Code/Work/thdrbprb/SpeakerAnomalyDetection pytest unit_test/ -v
```

## Documentation

- [Rub & Buzz Analysis](docs/features/rub-and-buzz.md)
- [Perceptual Rub & Buzz Analysis](docs/features/perceptual-rub-and-buzz.md)

## License

Proprietary
