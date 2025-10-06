import pandas as pd
from typing import NamedTuple, Optional
from dataclasses import dataclass
from .feature_extractors import FeatureExtractor
from .cleaners import SignalCleaner
import logging

logger = logging.getLogger(__name__)

class Dataset(NamedTuple):
    """Container for ML-ready dataset."""
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    label_mapping: dict[int, str]

@dataclass
class ChannelConfig:
    """Configuration for a single channel (EMG or EEG)."""
    signal_cleaner: SignalCleaner
    feature_extractor: FeatureExtractor