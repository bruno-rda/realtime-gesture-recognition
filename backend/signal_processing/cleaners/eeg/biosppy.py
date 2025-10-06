import numpy as np
from biosppy.signals import eeg
from ..base import SignalCleaner

class EEGBiosppy(SignalCleaner):
    def __init__(self, show: bool = False):
        super().__init__()
        self.show = show

    def clean_signal(
        self, 
        signal: np.ndarray, 
        sampling_rate: int
    ) -> np.ndarray:
        return eeg.eeg(
            signal=signal, 
            sampling_rate=sampling_rate, 
            show=self.show
        )['filtered']
    