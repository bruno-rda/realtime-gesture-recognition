import numpy as np
import pandas as pd
from feature_extraction import FeatureExtractor
import logging

logger = logging.getLogger(__name__)

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
        step_size: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the X, y, and groups from the dataframe.
        """
        logger.info(f'Initial DF shape: {df.shape}')
        ebr_channels = df.columns.difference(['TIMESTAMP', 'label', 'group'])

        X = self.process(
            signals=df[ebr_channels].values,
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

        df_out = pd.DataFrame(X)
        df_out.dropna(inplace=True, axis=1)
        logger.info(f'Processed DF shape: {df_out.shape}')

        # Add labels and groups to the dataframe
        df_out['label'] = y
        df_out['group'] = groups

        # Keep only valid labels
        df_out = df_out[~df_out['label'].isin([0, 100, 101])]
        X = df_out.drop(columns=['label', 'group'])
        y = df_out['label'] - 1
        logger.info(f'Final DF shape: {df_out.shape}')
        
        return X, y, df_out['group']