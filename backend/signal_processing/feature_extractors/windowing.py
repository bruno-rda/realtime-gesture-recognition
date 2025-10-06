import numpy as np
from typing import Generator

def sliding_windows_indices(
    signal: np.ndarray, 
    window_size: float, 
    step_size: float, 
    sampling_rate: int
) -> Generator[tuple[int, int], None, None]:
    window_samples = int(window_size * sampling_rate)
    step_samples = int(step_size * sampling_rate)

    for start in range(0, len(signal) - window_samples + 1, step_samples):
        yield start, start + window_samples

def sliding_windows(
    signal: np.ndarray, 
    window_size: float, 
    step_size: float, 
    sampling_rate: int
) -> Generator[np.ndarray, None, None]:
    for start, end in sliding_windows_indices(signal, window_size, step_size, sampling_rate):
        yield signal[start:end]

def sliding_window_center(
    array: np.ndarray, 
    window_size: float, 
    step_size: float, 
    sampling_rate: int
) -> np.ndarray:
    """
    Get the center of each sliding window.
    """
    indices = list(sliding_windows_indices(array, window_size, step_size, sampling_rate))
    center_indices = [start + (end - start) // 2 for start, end in indices]

    return array[center_indices]