import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from ..base import SignalCleaner

class BandpassNotchFilter(SignalCleaner):
    def __init__(
        self,
        low: int = 20,
        high: int = 450,
        order: int = 2,
        freq: float = 50.0,
        Q: int = 30
    ):
        super().__init__()
        self.low = low
        self.high = high
        self.order = order
        self.freq = freq
        self.Q = Q

    def clean_signal(
        self, 
        signal: np.ndarray, 
        sampling_rate: int
    ) -> np.ndarray:
        # Helper functions
        def bandpass_filter(signal, fs):
            b, a = butter(
                self.order, 
                [self.low/(fs/2), self.high/(fs/2)], 
                btype='band'
            )
            return filtfilt(b, a, signal)

        def notch_filter(signal, fs):
            b, a = iirnotch(self.freq, self.Q, fs)
            return filtfilt(b, a, signal)
        
        filtered = bandpass_filter(signal, sampling_rate)
        filtered = notch_filter(filtered, sampling_rate)
        return filtered