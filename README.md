# Real-time Gesture Recognition

A machine learning system for real-time gesture recognition using multi-channel biosignals.

## Overview

This system processes EMG and EEG signals in real-time to classify gestures using machine learning. It supports both training and prediction modes with configurable signal processing pipelines.

## Key Features

- **Real-time Processing**: Live EMG signal analysis and gesture classification
- **Multi-channel Support**: Configurable number of input channels
- **Machine Learning Pipeline**: Sklearn pipeline with feature selection and classification model
- **Modular Architecture**: Pluggable signal processors and feature extractors
- **Multiple Interfaces**: CLI and GUI frontends available
- **Flexible Configuration**: Adjustable parameters for signal processing and model training

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python main.py
   ```

3. **Configure settings in `config.py`** for your specific setup (channels, sampling rate, model parameters)

## System Components

- **Signal Processing**: Multi-channel EMG/EEG signal cleaning and feature extraction
- **Machine Learning**: Training pipeline with cross-validation and real-time prediction
- **Communication**: UDP socket and serial communication interfaces
- **Frontend**: CLI and GUI controllers for user interaction

## Data Format

The system expects multi-channel signal data as floating-point arrays, with configurable channel count and sampling rates.