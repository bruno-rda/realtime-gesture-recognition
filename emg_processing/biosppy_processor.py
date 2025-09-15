import numpy as np
from biosppy.signals import emg
from .base import EMGProcessor
from feature_extraction import FeatureExtractor

class BiosppyProcessor(EMGProcessor):
    def __init__(self, feature_extractor: FeatureExtractor):
        super().__init__(feature_extractor)

    def clean_emg_data(
        self, 
        signal: np.ndarray, 
        sampling_rate: int,
        **kwargs
    ) -> np.ndarray:
        return emg.emg(
            signal=signal, 
            sampling_rate=sampling_rate, 
            show=kwargs.get('show', False)
        )['filtered']
    