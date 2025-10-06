import numpy as np
import pandas as pd
from typing import Optional
from .feature_extractors import sliding_window_center
from .config import ChannelConfig, Dataset
import logging

logger = logging.getLogger(__name__)

class SignalProcessor:
    """End-to-end pipeline for signal processing and dataset creation."""
    
    def __init__(
        self, 
        emg_config: Optional[ChannelConfig] = None,
        eeg_config: Optional[ChannelConfig] = None,
        emg_column_indices: Optional[list[int]] = None,
    ):
        self.emg_config = emg_config
        self.eeg_config = eeg_config
        self.emg_column_indices = emg_column_indices
        self._validate()

    def _validate(self):
        assert self.emg_config or self.eeg_config, 'At least one config required'

        if self.emg_config and self.eeg_config:
            assert self.emg_column_indices is not None, \
                'emg_column_indices required when using both EMG and EEG'

    def _process_emg(
        self,
        signal: np.ndarray,
        window_size: float,
        step_size: float,
        sampling_rate: int
    ) -> np.ndarray:
        emg_features = []
        for signal in signal.T:
            clean = self.emg_config.signal_cleaner.clean_signal(
                signal, sampling_rate
            )
            features = self.emg_config.feature_extractor.extract_features(
                clean, window_size, step_size, sampling_rate
            )
            emg_features.append(features)
            
        return emg_features

    def _process_eeg(
        self,
        signal: np.ndarray,
        window_size: float,
        step_size: float,
        sampling_rate: int
    ) -> np.ndarray:
        eeg_features = []
        for signal in signal.T:
            clean = self.eeg_config.signal_cleaner.clean_signal(
                signal, sampling_rate
            )
            features = self.eeg_config.feature_extractor.extract_features(
                clean, window_size, step_size, sampling_rate
            )
            eeg_features.append(features)

        return eeg_features

    def process_signals(
        self, 
        signals: np.ndarray,
        window_size: float,
        step_size: float,
        sampling_rate: int
    ) -> np.ndarray:
        """
        Process multi-channel signals through cleaning and feature extraction.
        
        Args:
            signals: (n_samples, n_channels) array of raw signals
            window_size: Window duration in seconds
            step_size: Step duration in seconds
            sampling_rate: Sampling frequency in Hz
            
        Returns:
            (n_windows, n_features) array of extracted features
        """
        all_features = []

        if self.emg_config:
            emg_signals = (
                signals[:, self.emg_column_indices]
                if self.emg_column_indices
                else signals
            )
            
            features = self._process_emg(
                emg_signals, 
                window_size, 
                step_size, 
                sampling_rate
            )
            all_features.extend(features)

        if self.eeg_config:
            if self.emg_column_indices:
                all_cols = set(range(signals.shape[1]))
                eeg_cols = sorted(all_cols - set(self.emg_column_indices))
                eeg_signals = signals[:, eeg_cols]
            else:
                eeg_signals = signals

            features = self._process_eeg(
                eeg_signals, 
                window_size, 
                step_size, 
                sampling_rate
            )

            all_features.extend(features)
        
        return np.hstack(all_features)
    
    def build_dataset(
        self,
        df: pd.DataFrame,
        sampling_rate: int,
        window_size: float,
        step_size: float,
        ignore_labels: Optional[list] = None
    ) -> Dataset:
        """
        Build a dataset from a labeled dataframe.
        
        Args:
            df: DataFrame with signal columns, 'label', 'group', and optionally 'TIMESTAMP'
            sampling_rate: Signal sampling rate in Hz
            window_size: Feature window duration in seconds
            step_size: Window step duration in seconds
            ignore_labels: Labels to exclude from dataset
            
        Returns:
            Dataset with features (X), encoded labels (y), groups, and label mapping
        """
        logger.info(f'Building dataset from shape {df.shape}')
        
        # Extract signals
        signal_cols = df.columns.difference(['TIMESTAMP', 'label', 'group'])
        signals = df[signal_cols].values
        
        # Process signals
        X = self.process_signals(
            signals, window_size, step_size, sampling_rate
        )
        
        # Apply windowing to labels and groups
        y_windowed = sliding_window_center(
            df['label'].values, window_size, step_size, sampling_rate
        )
        groups_windowed = sliding_window_center(
            df['group'].values, window_size, step_size, sampling_rate
        )
        
        # Create dataframe and clean
        dataset_df = pd.DataFrame(X)
        dataset_df.dropna(axis=1, inplace=True)
        dataset_df['label'] = y_windowed
        dataset_df['group'] = groups_windowed
        
        # Filter ignored labels
        if ignore_labels:
            dataset_df = dataset_df[~dataset_df['label'].isin(ignore_labels)]
        
        # Encode labels
        X_final = dataset_df.drop(columns=['label', 'group'])
        label_cat = dataset_df['label'].astype('category')
        y_encoded = label_cat.cat.codes
        groups_final = dataset_df['group']
        label_mapping = dict(enumerate(label_cat.cat.categories))
        
        logger.info(f'Final dataset shape: {dataset_df.shape}')
        logger.info(f'Label mapping: {label_mapping}')
        
        return Dataset(X_final, y_encoded, groups_final, label_mapping)

    def __str__(self):
        return f'{self.__class__.__name__}({self.__dict__})'

    def __repr__(self):
        return self.__str__()