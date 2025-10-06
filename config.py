from typing import Optional, Any
from pydantic_settings import BaseSettings

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
    serial_port: str = '/dev/cu.usbserial-210'
    serial_baudrate: int = 9600
    serial_timeout: float = 1
    serial_chunk_size: int = 1
    message_mapping: Optional[dict[Any, Any]] = None

    # Model training
    cross_validate: bool = True
    should_save: bool = True
    experiments_base_dir: str = './backend/ml/experiments'
    trainer_path: Optional[str] = './backend/ml/experiments/2/trainer.pkl'

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