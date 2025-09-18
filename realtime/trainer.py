import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Any
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold
from emg_processing import EMGProcessor
import logging

logger = logging.getLogger(__name__)

class RealTimeTrainer:
    def __init__(
        self, 
        pipeline: Pipeline, 
        processor: EMGProcessor,
        window_size: float,
        step_size: float,
        sampling_rate: int,
        cross_validate: bool = False,
        should_save: bool = True,
        base_dir: str = './realtime/experiments'
    ):
        self.df = pd.DataFrame()
        self.processor = processor
        self.window_size = window_size
        self.step_size = step_size
        self.sampling_rate = sampling_rate

        self.pipeline = pipeline
        self.training = True
        self.curr_group = 0
        self.curr_steps = 0

        # Training parameters
        self.cross_validate = cross_validate
        self.should_save = should_save
        self.base_dir = base_dir
    
    @property
    def metadata(self) -> dict[str, Any]:
        def get_class_name(obj: Any) -> str:
            return obj.__class__.__name__
        
        return {
            # Data information
            'shape': self.df.shape,
            'labels': self.df['label'].unique().tolist(),
            'n_groups': len(self.df['group'].unique()),

            # Processing information
            'sampling_rate': self.sampling_rate,
            'window_size': self.window_size,
            'step_size': self.step_size,
            'processor_class': get_class_name(self.processor),
            'feature_extractor_class': get_class_name(self.processor.feature_extractor),
            'feature_extractor_params': self.processor.feature_extractor.__dict__,

            # Model information
            'pipeline_params': self.pipeline.get_params(deep=True),
        }
    
    def update(self, rows: np.ndarray, label: Any) -> None:
        # If not training, do nothing
        if not self.training:
            return
        
        assert label is not None, 'Label cannot be None'

        if rows.ndim == 1:
            rows = np.expand_dims(rows, axis=0)

        self.curr_steps += rows.shape[0]
        columns = [f'EBR_{i + 1}' for i in range(rows.shape[-1] - 1)] + ['TIMESTAMP']

        df_row = pd.DataFrame(rows, columns=columns)
        df_row['label'] = label
        df_row['group'] = self.curr_group

        self.df = pd.concat([self.df, df_row], ignore_index=True)
    
    def switch_group(self) -> None:
        self.curr_group += 1
        self.curr_steps = 0

    def train(self) -> None:
        assert self.training, 'Cannot train if not in training mode'

        if self.df.empty:
            raise ValueError('Cannot train if no data has been collected')

        X, y, groups, _ = self.processor.get_X_y_groups(
            df=self.df, 
            sampling_rate=self.sampling_rate,
            window_size=self.window_size, 
            step_size=self.step_size, 
        )

        if self.cross_validate:
            logger.info('Performing cross-validation...')
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True)

            scores = cross_val_score(
                self.pipeline, X, y, 
                cv=cv, 
                groups=groups,
                scoring='accuracy',
                n_jobs=-1
            )

            logger.info(f'Cross-validation completed - Mean: {scores.mean():.5f} (±{scores.std():.5f})')
            logger.debug(f'All CV scores: {np.array2string(scores, precision=5)}')
        
        self.pipeline.fit(X, y)
        self.training = False
        logger.info('Training completed.')
        
        if self.should_save:
            self.save()

    def save(self) -> Optional[str]:
        if self.df.empty:
            logger.warning('Model was not saved because no data has been collected')
            return
        
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            
            existing = os.listdir(self.base_dir)
            next_num = max(int(x) for x in existing if x.isdigit()) + 1 if existing else 0
            new_exp_dir = f'{self.base_dir}/{next_num}'
            os.makedirs(new_exp_dir)

            # Save recovered data
            self.df.to_csv(f'{new_exp_dir}/emg_data.csv')

            # Save the model if has been trained
            if not self.training:
                joblib.dump(self.pipeline, f'{new_exp_dir}/pipeline.joblib')
            
            # Metadata about the experiment and the model
            with open(f'{new_exp_dir}/metadata.json', 'w') as f:
                json.dump(self.metadata, f, indent=4, default=str)

            logger.info(f'Model saved to: {new_exp_dir}')
            return new_exp_dir

        except Exception as e:
            raise RuntimeError(f'Error saving: {e}')
    
    def reset_model(self) -> None:
        """ 
        Resets the model to its initial training state. 
        This method is useful if you want to further train the model
        without losing the data collected so far.
        """
        self.training = True
        self.pipeline = clone(self.pipeline)

    def reset(self) -> None:
        """
        Clears all collected data and resets the model's state. 
        This allows starting a new training session from scratch.
        """
        self.df = pd.DataFrame()
        self.curr_group = 0
        self.curr_steps = 0
        self.training = True
        self.pipeline = clone(self.pipeline)