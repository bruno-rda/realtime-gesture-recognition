# Utils for feature extraction

def sliding_windows_indices(signal, window_size, step_size, sampling_rate):
    window_samples = int(window_size * sampling_rate)
    step_samples = int(step_size * sampling_rate)
    for start in range(0, len(signal) - window_samples + 1, step_samples):
        yield start, start + window_samples

def sliding_windows(signal, window_size, step_size, sampling_rate):
    for start, end in sliding_windows_indices(signal, window_size, step_size, sampling_rate):
        yield signal[start:end]
