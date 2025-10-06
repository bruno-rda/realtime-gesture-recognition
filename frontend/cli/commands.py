import os
import json
from ..shared.commands import Command, CommandHandler, get_command_mapping
from .modes import Modes


class CLICommandHandler(CommandHandler):
    def __init__(self, controller):
        super().__init__(controller)
    
    # CLI specific methods

    def print_menu(self):
        self.controller.print_menu()
    
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    # Abstract methods from CommandHandler base class

    def request_label(self):
        return input('Enter label: ')
    
    def request_confirmation(self, message: str):
        return input(f'{message} (y/n): ').lower() == 'y'
    
    def show_trainer_metadata(self):
        print(
            json.dumps(
                self.controller.trainer.metadata, 
                indent=4,
                default=str
            )
        )
    

def get_cli_command_mapping(
    handler: CLICommandHandler
) -> dict[Modes, dict[str, Command]]:
    # Commands that are shared between all modes
    shared_commands = {
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
    
    commands = get_command_mapping(handler)

    return {
        menu_name: {**menu_commands, **shared_commands}
        for menu_name, menu_commands in commands.items()
    }