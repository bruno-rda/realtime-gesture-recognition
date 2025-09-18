from .base import EMGProcessor
from .manual_processor import ManualProcessor
from .biosppy_processor import BiosppyProcessor

__all__ = [
    'EMGProcessor',
    'ManualProcessor',
    'BiosppyProcessor'
]