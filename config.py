from typing import Optional
from pydantic_settings import BaseSettings
import logging

class Settings(BaseSettings):
    # Socket
    udp_ip: str = '0.0.0.0'
    udp_port: int = 8000

    # Data processing
    n_channels: int = 5
    window_size: float = 1
    step_size: float = 0.05
    sampling_rate: int = 1200

    # Model training
    cross_validate: bool = True
    should_save: bool = True
    base_dir: str = './realtime/experiments'

    # Model hyperparameters
    feature_selector_percentile: int = 90
    model_seed: Optional[int] = 1
    model_learning_rate: float = 0.3
    model_max_depth: int = 40
    model_n_estimators: int = 300

    # Logging
    log_level: str = 'INFO'

settings = Settings()

logging.basicConfig(
    level=settings.log_level,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def get_settings() -> Settings:
    return settings