from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.patch_stdout import patch_stdout
import threading
import time
import logging
from typing import Optional, Any
from ..shared.controller import Controller
from .modes import MODES_INFO
from .commands import Command, CLICommandHandler, get_cli_command_mapping
from backend.ml import Trainer, Predictor
from backend.io import SerialCommunicator

logger = logging.getLogger(__name__)


class CLIController(Controller):
    def __init__(
        self,
        trainer: Trainer,
        predictor: Predictor,
        communicator: Optional[SerialCommunicator] = None,
        show_probs: bool = True
    ):
        command_handler = CLICommandHandler(self)
        commands = get_cli_command_mapping(command_handler)
        super().__init__(
            trainer,
            predictor,
            command_handler,
            commands,
            communicator,
            show_probs
        )
        
        # Status management
        self._status_text = ''
        self._last_render_time = 0
        self._render_interval = 0.033  # ~30 FPS
        self._app = None
        
        self.session = PromptSession(
            bottom_toolbar=self._get_toolbar,
            style=Style.from_dict({'bottom-toolbar': 'bg:default noreverse'}),
        )
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self._print_lock = threading.Lock()
        
        self.print_menu()

    def _get_toolbar(self):
        return self._status_text

    def print_menu(self):
        mode_info = MODES_INFO[self.current_mode]
        with self._print_lock:
            print(f"\n* {mode_info['name']}:")
            print(mode_info['description'])
            print('Commands:')

            current_commands = self.commands[self.current_mode]
            for key, command in current_commands.items():
                print(f'  - {key}: {command.description}')
            print()

    def _update_status(self, text: str):
        current_time = time.time()

        self._status_text = text
        
        # Rate limit the actual rendering
        if current_time - self._last_render_time >= self._render_interval:
            self._last_render_time = current_time
            
            # Force prompt_toolkit to re-render
            if self._app:
                self._app.invalidate()


    def _input_loop(self):
        with patch_stdout(raw=True):
            while self.running:
                try:
                    self._app = self.session.app
                    
                    user_input = self.session.prompt('> ')
                    user_input = user_input.strip().lower()
                    
                    if user_input:
                        self.execute_command(user_input)
                        
                except (EOFError, KeyboardInterrupt):
                    self.stop()
                    break
                except Exception as e:
                    logger.error(f'Input error: {e}')
                    self.handle_command_error(None, e)

    # Abstract methods from Controller base class

    def handle_switch_mode(self):
        logger.info(f'Switched to {self.current_mode.name!r} mode')
        self.print_menu()
        self._status_text = ''

    def handle_command_exists(self, command: Command):
        logger.info(f'Executed {command.key!r}: {command.short_description}')

    def handle_command_not_found(self, command_key: str):
        with self._print_lock:
            print(f"\n[Unknown Command] '{command_key}'")

    def handle_command_error(self, command: Command, error: Exception):
        logger.warning(f'Command execution failed: {error}')

    def update_data_collection_status(self):
        status = f' Collecting: {self.current_label!r} | Steps: {self.trainer.curr_steps} '
        self._update_status(status)

    def update_prediction_status(self, f_pred: str):
        status = f' Prediction Mode | Prediction {self.predictor.n_preds}: {f_pred} '
        self._update_status(status)

    def handle_start(self):
        self.input_thread.start()
        logger.info('CLIController started - ready for command input')

    def handle_stop(self):
        self._status_text = ''  # Clear status bar
        logger.info('CLIController stopped')