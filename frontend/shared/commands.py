import os
import json
from dataclasses import dataclass
from typing import Callable, Optional
from .modes import Modes

@dataclass
class Command:
    key: str
    description: str
    action: Callable[[], None]
    short_description: Optional[str] = None
    next_state: Optional[Modes] = None

    def __post_init__(self):
        if self.short_description is None:
            self.short_description = self.description


class CommandHandler:
    def __init__(self, controller):
        self.controller = controller
    
    def train_model(self):
        self.controller.trainer.train()
    
    def save_trainer(self):
        self.controller.trainer.save()
    
    def quit_data_collection(self):
        self.controller.trainer.switch_group()
        self.controller.current_label = None
    
    def reset_all(self):
        self.confirm('Are you sure you want to reset all data and the model?')
        self.controller.trainer.reset()
        self.controller.predictor.reset()
    
    def reset_model(self):
        self.confirm('Are you sure you want to reset the model?')
        self.controller.trainer.reset_model()
        self.controller.predictor.reset()
    
    def toggle_serial_connection(self):
        if self.controller.communicator is None:
            raise ValueError('Communicator was not provided')

        if self.controller.communicator.is_active:
            self.controller.communicator.close()
        else:
            self.controller.communicator.open()
    
    def set_label(self):
        label = self.request_label()
        
        if not label:
            raise ValueError('Label cannot be empty')

        self.controller.current_label = label
    
    def confirm(self, message: str):
        confirmation = self.request_confirmation(message)
        
        if not confirmation:
            raise RuntimeError('Action cancelled')

    def request_label(self) -> str:
        raise NotImplementedError
    
    def request_confirmation(self, message: str) -> bool:
        raise NotImplementedError

    def show_trainer_metadata(self):
        raise NotImplementedError

def get_command_mapping(
    handler: CommandHandler
) -> dict[Modes, dict[str, Command]]:
    return {
        Modes.MAIN: {
            'a': Command(
                key='a',
                description='Request label and start data collection     → switches to Data Collection Mode',
                short_description='Request label and start data collection',
                action=handler.set_label,
                next_state=Modes.DATA_COLLECTION
            ),
            't': Command(
                key='t',
                description='Train the model                             → switches to Prediction Mode',
                short_description='Train the model',
                action=handler.train_model,
                next_state=Modes.PREDICTION
            ),
            'm': Command(
                key='m',
                description='Show trainer metadata',
                action=handler.show_trainer_metadata
            ),
            's': Command(
                key='s',
                description='Save trainer information',
                action=handler.save_trainer
            ),
            'r': Command(
                key='r',
                description='Reset data and trainer to initial state',
                action=handler.reset_all
            )
        },
        Modes.DATA_COLLECTION: {
            'q': Command(
                key='q',
                description='Quit sample collection                     → returns to Main Mode',
                short_description='Quit sample collection',
                action=handler.quit_data_collection,
                next_state=Modes.MAIN
            )
        },
        Modes.PREDICTION: {
            's': Command(
                key='s',
                description='Save trainer information',
                action=handler.save_trainer
            ),
            'r': Command(
                key='r',
                description='Reset all data and the model               → returns to Main Mode',
                short_description='Reset all data and the model',
                action=handler.reset_all,
                next_state=Modes.MAIN
            ),
            'm': Command(
                key='m',
                description='Reset the model only                       → returns to Main Mode',
                short_description='Reset the model only',
                action=handler.reset_model,
                next_state=Modes.MAIN
            ),
            'x': Command(
                key='x',
                description='Toggle serial connection',
                action=handler.toggle_serial_connection
            )
        }
    }