import numpy as np

class SignalCleaner:
    def __init__(self):
        pass

    def clean_signal(
        self, 
        signal: np.ndarray, 
        sampling_rate: int
    ) -> np.ndarray:
        """
        Clean the signal data.

        Args:
            signal: The signal to clean.
            sampling_rate: The sampling rate of the signal.
            
        Returns:
            A numpy array of the cleaned signal.
        """
        raise NotImplementedError
    
    def __str__(self):
        return f'{self.__class__.__name__}({self.__dict__})'

    def __repr__(self):
        return self.__str__()