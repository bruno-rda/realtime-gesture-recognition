import numpy as np
from pynput import keyboard
from realtime.trainer import RealTimeTrainer
from realtime.predictor import RealTimePredictor
import logging

logger = logging.getLogger(__name__)

MENUS = {
   'main':  '''
    * Main Mode:
        - This is the default starting mode.
        - Does not perform actions by default.
        - You can return here from:
            - Data Collection Mode (after quitting)
            - Prediction Mode (after resetting the model or data)

        Commands:
            - a: Prompt for label and start data collection  → switches to Data Collection Mode
            - t: Train the model                            → switches to Prediction Mode
            - s: Save trainer information
            - h: Show this menu help
    ''',

    'data_collection': '''
    * Data Collection Mode:
        - Collects training samples with the current label.
        - Entered by pressing 'a' in Main Mode.

        Commands:
            - q: Quit sample collection                     → returns to Main Mode
            - h: Show this menu help
    ''',

    'prediction': '''
    * Prediction Mode:
        - Predicts the label of the current sample by default.
        - Entered by pressing 't' in Main Mode.

        Commands:
            - s: Save trainer information
            - r: Reset all data and the model               → returns to Main Mode
            - m: Reset the model only                       → returns to Main Mode
            - h: Show this menu help
    '''
}


class Controller:
    def __init__(
        self, 
        trainer: RealTimeTrainer, 
        predictor: RealTimePredictor
    ):
        self.trainer = trainer
        self.predictor = predictor
        self.current_mode = 'main'
        self.current_label = None
        self.listener = keyboard.Listener(on_press=self.on_press)
        print(MENUS['main'])
    
    def switch_mode(self, mode: str):
        self.current_mode = mode
        logger.info(f'Switched to {mode!r} mode')
        print(MENUS[mode])

    def on_press(self, key):
        if not hasattr(key, 'char'):
            return
        
        current_mode = self.current_mode
        key = key.char

        match current_mode:
            case 'main':
                match key:
                    case 'a':
                        self.current_label = input('Enter label: ')
                        self.switch_mode('data_collection')
                    case 't':
                        self.trainer.train()
                        self.switch_mode('prediction')
                    case 's':
                        self.trainer.save()
                    case 'h':
                        print(MENUS['main'])
            case 'data_collection':
                match key:
                    case 'q':
                        self.current_label = None
                        # Advance group to separate this trial batch
                        self.trainer.switch_group()
                        self.switch_mode('main')
                    case 'h':
                        print(MENUS['data_collection'])
            case 'prediction':
                match key:
                    case 's':
                        self.trainer.save()
                    case 'r':
                        self.trainer.reset()
                        self.switch_mode('main')
                    case 'm':
                        self.trainer.reset_model()
                        self.switch_mode('main')
                    case 'h':
                        print(MENUS['prediction'])
    
    def update(self, data: np.ndarray) -> None:
        match self.current_mode:
            case 'data_collection':
                self.trainer.update(data, self.current_label)
                print(
                    f'\r[Collecting: {self.current_label!r}] Steps: {self.trainer.curr_steps} ',
                    end='', flush=True
                )
        
            case 'prediction':
                for row in data:
                    pred = self.predictor.update(row)

                    if pred is not None:
                        print(f'\r[Prediction Mode] Prediction: {pred}', end='', flush=True)
    
    def start(self):
        self.listener.start()

    def stop(self):
        self.listener.stop()