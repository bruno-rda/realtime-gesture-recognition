import numpy as np
import pandas as pd
from typing import Optional, NamedTuple, Any
from feature_extraction import FeatureExtractor
import logging

logger = logging.getLogger(__name__)

class DataSplit(NamedTuple):
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    label_mapping: dict[int, Any]

class EMGProcessor:
    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor

    def clean_emg_data(
        self, 
        signal: np.ndarray, 
        sampling_rate: int,
        **kwargs
    ) -> np.ndarray:
        """
        Clean the EMG data.

        Args:
            signal: The signal to clean.
            sampling_rate: The sampling rate of the signal.
        
        Returns:
            A numpy array of features.
        """
        raise NotImplementedError
    
    def process(
        self, 
        signals: np.ndarray, 
        window_size: float, 
        step_size: float, 
        sampling_rate: int,
        **kwargs
    ) -> np.ndarray:
        """
        Clean the EMG data and extract features.

        Args:
            signals: An np array where each column is a signal.
            window_size: The size of the window to process.
            step_size: The step size to process.
            sampling_rate: The sampling rate of the signal.
        
        Returns:
            A numpy array of features.
        """
        all_channel_features = []
        
        for signal in signals.T:
            # Clean the signal
            clean_signal = self.clean_emg_data(
                signal=signal, 
                sampling_rate=sampling_rate, 
                **kwargs
            )

            # Extract features
            features = self.feature_extractor.extract_features(
                signal=clean_signal, 
                window_size=window_size, 
                step_size=step_size, 
                sampling_rate=sampling_rate
            )

            all_channel_features.append(features)
        
        # Stack features horizontally
        X = np.hstack(all_channel_features)
        return X
        
    def get_X_y_groups(
        self, 
        df: pd.DataFrame, 
        sampling_rate: int, 
        window_size: float, 
        step_size: float,
        ignore_labels: Optional[list[Any]] = None
    ) -> DataSplit:
        """
        Get the X, y, groups, and label mapping from the dataframe.
        Rows that contain ignored labels are filtered out.
        """
        logger.info(f'Initial DF shape: {df.shape}')

        # Select signal channels
        signal_columns = df.columns.difference(['TIMESTAMP', 'label', 'group'])
        signal_data = df[signal_columns].values

        # Process the signals
        X = self.process(
            signals=signal_data,
            window_size=window_size, 
            step_size=step_size, 
            sampling_rate=sampling_rate
        )
        y, groups = self.feature_extractor.align_labels_and_groups(
            labels=df['label'].values,
            groups=df['group'].values,
            window_size=window_size, 
            step_size=step_size, 
            sampling_rate=sampling_rate
        )

        # Add labels and groups to the dataframe
        df_out = pd.DataFrame(X)
        df_out.dropna(inplace=True, axis=1)
        df_out['label'] = y
        df_out['group'] = groups

        if ignore_labels is not None:
            # Filter out ignored labels
            df_out = df_out[~df_out['label'].isin(ignore_labels)]

        # Get the X, y, groups, and label mapping
        X_final = df_out.drop(columns=['label', 'group'])
        label_cat = df_out['label'].astype('category')
        y_encoded = label_cat.cat.codes
        groups_final = df_out['group']

        # Map the encoded labels to the original labels
        label_mapping = dict(enumerate(label_cat.cat.categories))
        
        logger.info(f'Final DF shape: {df_out.shape}')
        logger.info(f'Label mapping: {label_mapping}')
        
        return DataSplit(
            X=X_final,
            y=y_encoded,
            groups=groups_final,
            label_mapping=label_mapping
        )