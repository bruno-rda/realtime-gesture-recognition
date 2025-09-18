import numpy as np
import tsfel
import pandas as pd
from tqdm import tqdm
from .base import FeatureExtractor
from .utils import sliding_windows

class TsfelFeatureExtractor(FeatureExtractor):
    def __init__(self, verbose: bool = False):
        super().__init__()
        self.verbose = verbose

    def extract_features(
        self, 
        signal: np.ndarray, 
        window_size: float, 
        step_size: float, 
        sampling_rate: int
    ) -> np.ndarray:
        features = []
        pbar = tqdm(total=len(signal), desc='Extracting features', disable=not self.verbose)
        step = int(window_size * sampling_rate)

        for window in sliding_windows(signal, window_size, step_size, sampling_rate):
            cfg = tsfel.get_features_by_domain()
            window_features = tsfel.time_series_features_extractor(cfg, pd.DataFrame(window), verbose=0, fs=sampling_rate)
            window_features = window_features.values[0]
            features.append(window_features)
            pbar.update(step)

        pbar.close()

        return np.array(features)