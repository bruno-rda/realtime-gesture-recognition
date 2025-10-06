import numpy as np

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
    
    def __str__(self):
        return f'{self.__class__.__name__}({self.__dict__})'

    def __repr__(self):
        return self.__str__()