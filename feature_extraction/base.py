import numpy as np
from .utils import sliding_windows_indices

class FeatureExtractor:
    def __init__(self):
        pass

    def extract_features(
        self, 
        signal: np.ndarray, 
        window_size: float, 
        step_size: float, 
        sampling_rate: int
    ) -> np.ndarray:
        """
        Extract features from a signal.

        Args:
            signal: The signal to extract features from.
            window_size: The size of the window to extract features from.
            step_size: The step size to extract features from.
            sampling_rate: The sampling rate of the signal.

        Returns:
            A numpy array of features.
        """
        raise NotImplementedError
    
    def align_labels_and_groups(
        self, 
        labels: np.ndarray,
        groups: np.ndarray,
        window_size: float, 
        step_size: float, 
        sampling_rate: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Align the labels and groups to the center of each sliding window over the DataFrame.
        """
        indices = list(sliding_windows_indices(labels, window_size, step_size, sampling_rate))
        center_indices = [start + (end - start) // 2 for start, end in indices]

        return labels[center_indices], groups[center_indices]