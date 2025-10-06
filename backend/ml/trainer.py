import os
import json
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Any
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold
from backend.signal_processing import SignalProcessor
import logging

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self, 
        pipeline: Pipeline, 
        processor: SignalProcessor,
        window_size: float,
        step_size: float,
        sampling_rate: int,
        cross_validate: bool = False,
        should_save: bool = True,
        base_dir: str = './backend/ml/experiments'
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

        self.label_mapping: Optional[dict[int, Any]] = None
    
    @classmethod
    def from_path(cls, path: str) -> 'Trainer':
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    @property
    def metadata(self) -> dict[str, Any]:
        metadata = {
            # Processing information
            'sampling_rate': self.sampling_rate,
            'window_size': self.window_size,
            'step_size': self.step_size,
            'processor': self.processor,

            # Model information
            'label_mapping': self.label_mapping,
        }

        if not self.df.empty:
            labels = self.df['label'].unique().tolist()
            groups_by_label = (
                self.df.groupby('label')['group'].nunique().astype(str).to_dict()
            )
            
            metadata.update({
                'shape': self.df.shape,
                'labels': labels,
                'n_groups': len(self.df['group'].unique()),
                'n_groups_by_label': groups_by_label,
            })

        if not self.training:
            metadata['pipeline_params'] = self.pipeline.get_params(deep=True)

        return metadata
    
    def update(self, rows: np.ndarray, label: Any) -> None:
        assert self.training, 'Cannot update if not in training mode'
        assert label, 'Label cannot be empty at update'

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

        X, y, groups, label_mapping = self.processor.build_dataset(
            df=self.df, 
            sampling_rate=self.sampling_rate,
            window_size=self.window_size, 
            step_size=self.step_size, 
        )

        self.label_mapping = label_mapping

        if self.cross_validate:
            if groups.groupby(y).nunique().min() == 1:
                logger.warning(
                    'At least one label one unique group — this may cause uneven class '
                    'distribution in cross-validation folds and reduce reliability of results.'
                )
            
            logger.info('Performing cross-validation...')
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1)

            scores = cross_val_score(
                self.pipeline, X, y, 
                cv=cv, 
                groups=groups,
                scoring='accuracy',
                n_jobs=-1
            )

            logger.info(f'Cross-validation completed - Mean: {scores.mean():.5f} (±{scores.std():.5f})')
            logger.info(f'All CV scores: {np.array2string(scores, precision=5)}')
        
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
            self.df.to_csv(f'{new_exp_dir}/signal_data.csv')

            # Save the model if has been trained
            if not self.training:
                joblib.dump(self.pipeline, f'{new_exp_dir}/pipeline.joblib')
            
            # Metadata about the experiment and the model
            with open(f'{new_exp_dir}/metadata.json', 'w') as f:
                json.dump(self.metadata, f, indent=4, default=str)

            with open(f'{new_exp_dir}/trainer.pkl', 'wb') as f:
                pickle.dump(self, f)

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
        self.label_mapping = None
        self.curr_group = 0
        self.curr_steps = 0
        self.training = True
        self.pipeline = clone(self.pipeline)