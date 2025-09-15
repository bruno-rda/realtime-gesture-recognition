import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from .base import EMGProcessor
from feature_extraction import FeatureExtractor

class ManualProcessor(EMGProcessor):
    def __init__(self, feature_extractor: FeatureExtractor):
        super().__init__(feature_extractor)

    def clean_emg_data(
        self, 
        signal: np.ndarray, 
        sampling_rate: int,
        **kwargs
    ) -> np.ndarray:
        # Helper functions
        def bandpass_filter(signal, fs, low=20, high=450, order=2):
            b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
            return filtfilt(b, a, signal)

        def notch_filter(signal, fs, freq=50.0, Q=30):
            b, a = iirnotch(freq, Q, fs)
            return filtfilt(b, a, signal)
        
        filtered = bandpass_filter(signal, sampling_rate)
        filtered = notch_filter(filtered, sampling_rate)
        return filtered