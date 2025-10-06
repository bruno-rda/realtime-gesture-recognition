import numpy as np
from ..windowing import sliding_windows
from ..base import FeatureExtractor
import scipy

def get_simple_features(window, sampling_rate):
    rms = np.sqrt(np.mean(window**2))
    mav = np.mean(np.abs(window))
    wl = np.sum(np.abs(np.diff(window)))
    zc = np.sum(np.diff(np.sign(window)) != 0)
    ssc = np.sum(np.diff(np.sign(np.diff(window))) != 0)

    return [rms, mav, wl, zc, ssc]

def get_advanced_features(window, sampling_rate):
    rms = np.sqrt(np.mean(window ** 2))
    mav = np.mean(np.abs(window))
    wl = np.sum(np.abs(np.diff(window)))
    zc = np.sum(np.diff(np.sign(window)) != 0)
    ssc = np.sum(np.diff(np.sign(np.diff(window))) != 0)
    var = np.var(window)
    iemg = np.sum(np.abs(window))
    std = np.std(window)
    skewness = scipy.stats.skew(window)
    kurtosis = scipy.stats.kurtosis(window)
    wamp = np.sum(np.abs(np.diff(window)) > 0.05)

    # Hjorth Parameters
    activity = np.var(window)
    mobility = np.sqrt(np.var(np.diff(window)) / activity)
    complexity = np.sqrt(np.var(np.diff(np.diff(window))) / np.var(np.diff(window)))

    # Frequency Domain
    fft_vals = np.abs(np.fft.rfft(window))
    fft_freqs = np.fft.rfftfreq(len(window), d=1/sampling_rate)
    power = np.sum(fft_vals**2)
    mnf = np.sum(fft_freqs * fft_vals) / np.sum(fft_vals)
    mdf = fft_freqs[np.where(np.cumsum(fft_vals) >= np.sum(fft_vals)/2)[0][0]]
    spectral_entropy = -np.sum((fft_vals / np.sum(fft_vals)) * np.log2(fft_vals / np.sum(fft_vals) + 1e-12))

    return [
        rms, mav, wl, zc, ssc,
        var, iemg, std, skewness, kurtosis, wamp, 
        activity, mobility, complexity, 
        mnf, mdf, spectral_entropy
    ]


class CustomFeatures(FeatureExtractor):
    def __init__(self, simple: bool = True):
        super().__init__()
        self.simple = simple
        self.get_features = get_simple_features if simple else get_advanced_features

    def extract_features(
        self,
        signal: np.ndarray, 
        window_size: float, 
        step_size: float, 
        sampling_rate: int
    ) -> np.ndarray:
        
        return np.array([
            self.get_features(window, sampling_rate)
            for window in sliding_windows(signal, window_size, step_size, sampling_rate)
        ])