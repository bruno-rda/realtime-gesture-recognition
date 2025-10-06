import numpy as np
from biosppy.signals import emg
from ..base import SignalCleaner

class EMGBiosppy(SignalCleaner):
    def __init__(self, show: bool = False):
        super().__init__()
        self.show = show

    def clean_signal(
        self, 
        signal: np.ndarray, 
        sampling_rate: int
    ) -> np.ndarray:
        return emg.emg(
            signal=signal, 
            sampling_rate=sampling_rate, 
            show=self.show
        )['filtered']
    