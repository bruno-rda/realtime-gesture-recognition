from typing import Optional, Any
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

    # Serial
    serial_port: str = 'COM3'
    serial_baudrate: int = 9600
    serial_timeout: float = 1
    message_mapping: Optional[dict[Any, Any]] = None

    # Model training
    cross_validate: bool = True
    should_save: bool = True
    experiments_base_dir: str = './realtime/experiments'
    trainer_path: Optional[str] = None

    # Model hyperparameters
    feature_selector_percentile: int = 90
    model_seed: Optional[int] = 1
    model_learning_rate: float = 0.3
    model_max_depth: int = 40
    model_n_estimators: int = 300

    # Interface
    show_probs: bool = True
    
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