import numpy as np
import logging
from pynput import keyboard
from typing import Optional, Any
from realtime.interface.menus import MenuState, MENU_INFO
from realtime.interface.commands import (
    CommandHandler, 
    Command, 
    get_command_mapping
)
from realtime.trainer import RealTimeTrainer
from realtime.predictor import RealTimePredictor
from realtime.communicator import SerialCommunicator

logger = logging.getLogger(__name__)


class InterfaceHandler:
    def __init__(
        self, 
        trainer: RealTimeTrainer, 
        predictor: RealTimePredictor,
        communicator: Optional[SerialCommunicator] = None,
        show_probs: bool = True
    ):
        self.trainer = trainer
        self.predictor = predictor
        self.communicator = communicator
        self.show_probs = show_probs

        self.current_label = None
        self.current_mode = (
            MenuState.MAIN if trainer.training else MenuState.PREDICTION
        )

        self.command_handler = CommandHandler(self)
        self.commands: dict[MenuState, dict[str, Command]] = (
            get_command_mapping(self.command_handler)
        )
        
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.print_menu()
    
    def print_menu(self):
        mode_info = MENU_INFO[self.current_mode]
        print(f"\n* {mode_info['name']}:")
        print(mode_info['description'])

        print('Commands:')

        current_commands = self.commands[self.current_mode]
        for key, command in current_commands.items():
            print(f'    - {key}: {command.description}')
        
        print()
    
    def switch_mode(self, mode: MenuState):
        logger.info(f'Switched to {mode.value!r} mode')
        self.current_mode = mode
        self.print_menu()
    
    def refresh_line(self, text: str = ''):
        print(f'\r\033[K{text}', end='', flush=True)

    def on_press(self, key):
        if not hasattr(key, 'char'):
            return

        key_char = key.char.lower()
        current_commands = self.commands[self.current_mode]
        command = current_commands.get(key_char)

        if command:
            self.refresh_line()
            logger.info(f'Pressed {key_char!r}: {command.short_description}')
            
            try:
                command.action()
                
                # Handle state transition if specified
                if command.next_state is not None:
                    self.switch_mode(command.next_state)
                    
            except Exception as e:
                logger.warning(f'Command execution failed: {e}')
    
    def update(self, data: np.ndarray) -> Optional[Any]:
        match self.current_mode:
            case MenuState.DATA_COLLECTION:
                return self._update_data_collection(data)
        
            case MenuState.PREDICTION:
                return self._update_prediction(data)

    def _update_data_collection(self, data: np.ndarray):
        self.trainer.update(data, self.current_label)
        self.refresh_line(
            f'[Collecting: {self.current_label!r}] Steps: {self.trainer.curr_steps}'
        )

    def _update_prediction(self, data: np.ndarray):
        result = None

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

            self.refresh_line(f'[Prediction Mode] Prediction {self.predictor.n_predictions}: {f_pred}')

            if self.communicator and self.communicator.is_active:
                self.communicator.send(mapped_pred)

        return result

    def start(self):
        self.listener.start()
        logger.info('Interface started - listening for keyboard input')

    def stop(self):
        self.listener.stop()
        logger.info('Interface stopped')