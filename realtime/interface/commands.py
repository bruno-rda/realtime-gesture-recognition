import os
from dataclasses import dataclass
from typing import Callable, Optional
from realtime.interface.menus import MenuState

@dataclass
class Command:
    key: str
    description: str
    action: Callable[[], None]
    short_description: Optional[str] = None
    next_state: Optional[MenuState] = None

    def __post_init__(self):
        if self.short_description is None:
            self.short_description = self.description

class CommandHandler:
    def __init__(self, controller):
        self.controller = controller
    
    def prompt_for_label(self):
        self.controller.current_label = input('Enter label: ')
        
        if not self.controller.current_label:
            raise ValueError('Label cannot be empty')
    
    def train_model(self):
        self.controller.trainer.train()
    
    def save_trainer(self):
        self.controller.trainer.save()
    
    def quit_data_collection(self):
        self.controller.trainer.switch_group()
    
    def _confirm(self, message: str):
        if input(f'{message} (y/n): ').lower() != 'y':
            raise RuntimeError('Action cancelled')
    
    def reset_all(self):
        self._confirm('Are you sure you want to reset all data and the model?')
        self.controller.trainer.reset()
    
    def reset_model(self):
        self._confirm('Are you sure you want to reset the model?')
        self.controller.trainer.reset_model()
    
    def print_menu(self):
        self.controller.print_menu()
    
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def temporary_disable_listener(self):
        input('Press enter to enable listener: ')
        self.print_menu()
    
    def toggle_serial_connection(self):
        if self.controller.communicator is None:
            raise ValueError('Communicator was not provided')

        if self.controller.communicator.is_active:
            self.controller.communicator.close()
        else:
            self.controller.communicator.open()
        

def get_command_mapping(
    handler: CommandHandler
) -> dict[MenuState, dict[str, Command]]:
    
    # Commands that are shared between all modes
    shared_commands = {
        '&': Command(
            key='&',
            description='Disable listener until you press enter',
            action=handler.temporary_disable_listener
        ),
        'd': Command(
            key='d',
            description='Clear screen',
            action=handler.clear_screen
        ),
        'h': Command(
            key='h',
            description='Show this menu help',
            action=handler.print_menu
        )
    }

    commands = {
        MenuState.MAIN: {
            'a': Command(
                key='a',
                description='Prompt for label and start data collection  → switches to Data Collection Mode',
                short_description='Prompt for label and start data collection',
                action=handler.prompt_for_label,
                next_state=MenuState.DATA_COLLECTION
            ),
            't': Command(
                key='t',
                description='Train the model                             → switches to Prediction Mode',
                short_description='Train the model',
                action=handler.train_model,
                next_state=MenuState.PREDICTION
            ),
            's': Command(
                key='s',
                description='Save trainer information',
                action=handler.save_trainer
            )
        },
        MenuState.DATA_COLLECTION: {
            'q': Command(
                key='q',
                description='Quit sample collection                     → returns to Main Mode',
                short_description='Quit sample collection',
                action=handler.quit_data_collection,
                next_state=MenuState.MAIN
            )
        },
        MenuState.PREDICTION: {
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
                next_state=MenuState.MAIN
            ),
            'm': Command(
                key='m',
                description='Reset the model only                       → returns to Main Mode',
                short_description='Reset the model only',
                action=handler.reset_model,
                next_state=MenuState.MAIN
            ),
            'x': Command(
                key='x',
                description='Toggle serial connection',
                action=handler.toggle_serial_connection
            )
        }
    }

    return {
        menu_name: {**menu_commands, **shared_commands}
        for menu_name, menu_commands in commands.items()
    }