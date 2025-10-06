import numpy as np
import threading
from typing import Optional, Any
from .modes import Modes
from .commands import Command, CommandHandler, get_command_mapping
from backend.ml import Trainer, Predictor
from backend.io import SerialCommunicator


class Controller:
    def __init__(
        self,
        trainer: Trainer, 
        predictor: Predictor,
        command_handler: CommandHandler,
        commands: dict[Modes, dict[str, Command]],
        communicator: Optional[SerialCommunicator] = None,
        show_probs: bool = True,
    ):
        self.trainer = trainer
        self.predictor = predictor
        self.communicator = communicator
        self.show_probs = show_probs

        self.current_label = None
        self.current_mode = (
            Modes.MAIN if trainer.training else Modes.PREDICTION
        )
        self._mode_lock = threading.Lock()

        self.command_handler = command_handler
        self.commands: dict[Modes, dict[str, Command]] = commands
        self.running = False
    
    
    def switch_mode(self, mode: Modes):
        with self._mode_lock:
            self.current_mode = mode
            self.handle_switch_mode()

    def execute_command(self, command_key: str):
        command = self.commands[self.current_mode].get(command_key)

        if not command:
            self.handle_command_not_found(command_key)
            return
        
        self.handle_command_exists(command)
        
        try:
            command.action()
            
            if command.next_state is not None:
                self.switch_mode(command.next_state)
                
        except Exception as e:
            self.handle_command_error(command, e)

    def update(self, data: np.ndarray) -> Optional[Any]:
        # Use a lock to ensure that the mode is not changed while updating
        with self._mode_lock:
            match self.current_mode:
                case Modes.DATA_COLLECTION:
                    return self.update_data_collection(data)
            
                case Modes.PREDICTION:
                    return self.update_prediction(data)

    def update_data_collection(self, data: np.ndarray):
        self.trainer.update(data, self.current_label)
        self.update_data_collection_status()

    def update_prediction(self, data: np.ndarray):
        mapped_pred = None

        for row in data:
            result = self.predictor.update(row)

            if result is None:
                continue
            
            pred, probs = result
            
            # Map prediction to label
            mapped_pred = self.trainer.label_mapping.get(pred, pred)

            # Format probabilities if needed
            if self.show_probs:
                f_pred = {
                    label: f'{probs[i]:.5f}'
                    for i, label in self.trainer.label_mapping.items()
                }
            else:
                f_pred = mapped_pred

            self.update_prediction_status(f_pred)

            if self.communicator and self.communicator.is_active:
                self.communicator.send(mapped_pred)

        return mapped_pred
    
    def start(self):
        self.running = True
        self.handle_start()

    def stop(self):
        if self.running:
            self.running = False
            self.handle_stop()

    def handle_switch_mode(self):
        raise NotImplementedError
    
    def handle_command_exists(self, command: Command):
        raise NotImplementedError

    def handle_command_not_found(self, command_key: str):
        raise NotImplementedError

    def handle_command_error(self, command: Command, error: Exception):
        raise NotImplementedError

    def update_data_collection_status(self):
        raise NotImplementedError

    def update_prediction_status(self, f_pred: str):
        raise NotImplementedError

    def handle_start(self):
        raise NotImplementedError

    def handle_stop(self):
        raise NotImplementedError