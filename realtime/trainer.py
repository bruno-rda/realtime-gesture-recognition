import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional
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
        trial_ms: int = 10000,
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
        self.curr_label = None
        self.curr_group = 0
        self.curr_steps = 0
        self.max_steps = sampling_rate * trial_ms / 1000
        self.cross_validate = cross_validate
        self.should_save = should_save
        self.base_dir = base_dir
    
    def update(self, rows: np.ndarray) -> None:
        # If not training, do nothing
        if not self.training:
            return

        if self.curr_label is None:
            while not isinstance(self.curr_label, int):
                try:
                    self.curr_label = int(input("Enter the label: "))
                except ValueError:
                    print("Invalid label. Please enter an integer.")
            
            if self.curr_label == -1:
                self.train()

            return

        # If max steps is reached, reset label and start a new group
        if self.curr_steps >= self.max_steps:
            self.curr_steps = 0
            self.curr_group += 1
            self.curr_label = None
            return
        
        if rows.ndim == 1:
            rows = np.expand_dims(rows, axis=0)

        self.curr_steps += rows.shape[0]
        columns = [f'EBR_{i + 1}' for i in range(rows.shape[-1] - 1)] + ['TIMESTAMP']

        df_row = pd.DataFrame(rows, columns=columns)
        df_row['label'] = self.curr_label + 1
        df_row['group'] = self.curr_group

        self.df = pd.concat([self.df, df_row], ignore_index=True)

    def train(self) -> None:
        logger.info(f'Starting data processing')
        self.training = False

        X, y, groups = self.processor.get_X_y_groups(
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

            logger.info('=== Cross-validation scores ===')
            logger.info(f'Mean accuracy: {scores.mean():.5f}')
            logger.info(f'Standard deviation: {scores.std():.5f}')
            logger.info(f'All scores: {np.array2string(scores, precision=5)}')
        
        self.pipeline.fit(X, y)
        logger.info('Training completed.')
        
        if self.should_save:
            self.save()

    def save(self) -> Optional[str]:
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            
            existing = os.listdir(self.base_dir)
            next_num = max(int(x) for x in existing if x.isdigit()) + 1 if existing else 0

            new_exp_dir = f'{self.base_dir}/{next_num}'
            os.makedirs(new_exp_dir)

            # Save files in the new directory
            self.df.to_csv(f'{new_exp_dir}/emg_data.csv')
            joblib.dump(self.pipeline, f'{new_exp_dir}/pipeline.joblib')
            logger.info(f'Model saved to {new_exp_dir}...')
            return new_exp_dir

        except Exception as e:
            logger.error(f'Error saving: {e}')
    
    def switch_to_training(self) -> None:
        logger.info('Switching to training mode...')
        self.training = True
        self.curr_label = None
        self.pipeline = clone(self.pipeline)

    def reset(self) -> None:
        logger.info('Resetting trainer...')
        self.df = pd.DataFrame()
        self.curr_label = None
        self.curr_group = 0
        self.curr_steps = 0
        self.training = True
        self.pipeline = clone(self.pipeline)