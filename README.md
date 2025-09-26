# Realtime Gesture Recognition

A real-time gesture recognition system that processes multi-channel EMG signals and performs machine learning-based classification.

## Features

- Real-time EMG signal processing and feature extraction
- Multi-channel signal support (configurable number of channels)
- Machine learning pipeline with XGBoost classifier
- Real-time training and prediction capabilities
- UDP socket communication for data streaming
- Modular architecture with pluggable processors and feature extractors
- Configurable parameters for signal processing and model training

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bruno-rda/realtime-gesture-recognition.git
cd realtime-gesture-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main application:
```bash
python main.py
```

The system will:
- Bind to a UDP socket (default: 0.0.0.0:8000)
- Process incoming EMG data packets
- Perform real-time gesture recognition
- Display predictions in the console

## Configuration

Modify `config.py` to adjust system parameters:

- **Data Processing**: Number of channels, window size, step size, sampling rate
- **Model Parameters**: Learning rate, max depth, number of estimators
- **Network Settings**: UDP IP and port configuration
- **Training Options**: Cross-validation, model saving, experiment directory

## Architecture

The project follows a modular design with the following components:

- **EMG Processing**: Signal cleaning and preprocessing modules
- **Feature Extraction**: Multiple feature extraction strategies (manual, TSFEL, TSFRESH)
- **Real-time Components**: Training and prediction pipelines
- **Configuration**: Centralized settings management
- **Utilities**: Helper functions for data handling

## Data Format

The system expects EMG data packets containing floating-point values representing multi-channel EMG signals. Each packet should contain data for all configured channels.

## License

This project is licensed under the terms specified in the LICENSE file.
