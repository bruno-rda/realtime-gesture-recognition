import numpy as np
import pandas as pd
import tsfresh
from ..base import FeatureExtractor
from ..windowing import sliding_windows

class TsfreshFeatures(FeatureExtractor):
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
        
        segments = []
        segment_id = []
        segment_time = []

        for i, window in enumerate(sliding_windows(signal, window_size, step_size, sampling_rate)):
            segments.extend(window)
            segment_id.extend([i] * len(window))
            segment_time.extend(list(range(len(window))))

        df = pd.DataFrame({
            'id': segment_id,
            'time': segment_time,
            'signal': segments
        })

        features_df = tsfresh.extract_features(
            df,
            column_id='id',
            column_sort='time',
            disable_progressbar=not self.verbose
        )

        return features_df.values