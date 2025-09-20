from typing import Optional
import numpy as np
from collections import deque
from sklearn.pipeline import Pipeline
from emg_processing import EMGProcessor

class RealTimePredictor:
    def __init__(
        self, 
        pipeline: Pipeline,
        processor: EMGProcessor,
        window_size: float,
        step_size: float,
        sampling_rate: int,
    ):
        self.pipeline = pipeline
        self.processor = processor
        self.window_size = window_size
        self.step_size = step_size
        self.sampling_rate = sampling_rate
        self.window_samples = int(window_size * self.sampling_rate)
        self.step_samples = int(step_size * self.sampling_rate)

        self.readings = deque(maxlen=self.window_samples)
        self.remaining_steps = self.step_samples

    def predict(self) -> tuple[int, np.ndarray]:
        # Clean the signals and extract features
        X = self.processor.process(
            signals=np.array(self.readings),
            window_size=self.window_size,
            step_size=self.step_size,
            sampling_rate=self.sampling_rate
        )

        probs = self.pipeline.predict_proba(X)[0]
        return np.argmax(probs), probs

    def update(self, row: np.ndarray) -> Optional[int]:
        """
        Update the predictor with a new reading.

        Args:
            row: The new reading including the timestamp.

        Returns:
            The predicted label if the window is full, 
            and enough steps have passed, otherwise None.
        """
        if row.ndim != 1:
            raise ValueError('Row must be a 1D array')

        # Remove the timestamp from the row
        self.readings.append(row[:-1])

        if len(self.readings) < self.window_samples:
            return None
        
        self.remaining_steps -= 1
        if self.remaining_steps > 0:
            return None

        self.remaining_steps = self.step_samples
        
        return self.predict()