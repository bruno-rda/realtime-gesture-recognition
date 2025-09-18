import numpy as np
import logging
from pynput import keyboard
from typing import Dict
from realtime.interface.menus import MenuState, MENU_INFO
from realtime.interface.commands import CommandHandler, Command, get_command_mapping

logger = logging.getLogger(__name__)


class InterfaceHandler:
    def __init__(self, trainer, predictor):
        self.trainer = trainer
        self.predictor = predictor
        self.current_mode = MenuState.MAIN
        self.current_label = None
        
        self.command_handler = CommandHandler(self)
        self.commands: Dict[MenuState, Dict[str, Command]] = get_command_mapping(self.command_handler)
        
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

    def on_press(self, key):
        if not hasattr(key, 'char'):
            return

        key_char = key.char.lower()
        current_commands = self.commands[self.current_mode]
        command = current_commands.get(key_char)

        if command:
            logger.info(f'Pressed {key_char!r}: {command.short_description}')
            
            try:
                command.action()
                
                # Handle state transition if specified
                if command.next_state is not None:
                    self.switch_mode(command.next_state)
                    
            except Exception as e:
                logger.warning(f'Command execution failed: {e}')
    
    def update(self, data: np.ndarray) -> None:
        match self.current_mode:
            case MenuState.DATA_COLLECTION:
                self.trainer.update(data, self.current_label)
                
                print(
                    f'\r[Collecting: {self.current_label!r}] Steps: {self.trainer.curr_steps}',
                    end='', flush=True
                )
        
            case MenuState.PREDICTION:
                for row in data:
                    pred = self.predictor.update(row)

                    if pred is not None:
                        print(f'\r[Prediction Mode] Prediction: {pred}', end='', flush=True)
    
    def start(self):
        self.listener.start()
        logger.info('Controller started - listening for keyboard input')

    def stop(self):
        self.listener.stop()
        logger.info('Controller stopped')