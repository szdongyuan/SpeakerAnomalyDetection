# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyQt5-based speaker anomaly detection system (谛听异音检测) with machine learning capabilities. The application provides a desktop GUI for audio signal processing, analysis, and AI-powered anomaly detection in speakers.

## Architecture

### Core Components

- **Main Application**: `main_window_Launcher.py` is the entry point, loads splash screen and initializes `main_window.py`
- **UI Layer**: PyQt5-based interface in `ui/` directory with modular windows for different functions
- **Audio Processing**: `base/pre_processing/` contains signal processing modules (FFT, THD, peak detection, etc.)
- **Machine Learning**: `machine_learning/` contains ML models (CNN, RNN, SVC, Transformer) and model management
- **Database**: SQLite-based data management in `base/db_manager.py` for audio data, models, and users
- **Hardware Interface**: Sound device management and calibration systems

### Key Directory Structure

- `base/` - Core functionality (database, audio processing, logging, configuration)
- `ui/` - All PyQt5 GUI components and windows
- `machine_learning/` - ML models and training infrastructure
- `consts/` - Constants and configuration parameters
- `unit_test/` - Test files using pytest framework

### Data Flow

1. Audio acquisition through sound devices (`base/sound_device_manager.py`)
2. Signal preprocessing pipeline (`base/pre_processing/`)
3. Feature extraction and analysis
4. ML model inference for anomaly detection
5. Results storage in SQLite database

## Development Commands

### Running the Application
```bash
python main_window_Launcher.py
```

### Testing
```bash
python -m pytest unit_test/
```
Test individual modules:
```bash
python -m pytest unit_test/base/test_[module_name].py
```

### Database Operations
The system uses SQLite with automatic table creation. Database path is configured in `consts/model_consts.py`.

## Technical Specifications

### Dependencies
- PyQt5 for GUI framework
- TensorFlow/Keras for ML models
- scikit-learn for traditional ML algorithms
- NumPy/SciPy for signal processing
- SQLite3 for data persistence
- concurrent-log-handler for logging

### User Management
Three-tier access system: Admin, Engineer, Operator with different permission levels configured in the main window.

### Audio Processing Pipeline
- Sample rate: Configured in `consts/model_consts.py`
- Preprocessing: Emphasis, noise handling, data alignment
- Feature extraction: FFT, THD, frequency response analysis
- Peak detection and signal analysis tools

### ML Model Types
- CNN: Convolutional neural networks for pattern recognition
- RNN: Recurrent networks for temporal analysis
- SVC: Support Vector Classifier for traditional ML approach
- Transformer: Attention-based models for advanced analysis

## Configuration

System configuration is managed through JSON files in `ui/ui_config/` and Python constants in `consts/` directory. The application supports stimulus signal configuration, analysis parameters, and sequence management.