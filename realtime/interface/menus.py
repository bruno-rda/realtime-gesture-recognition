from enum import Enum

class MenuState(Enum):
    MAIN = 'main'
    DATA_COLLECTION = 'data_collection'
    PREDICTION = 'prediction'


MENU_INFO = {
    MenuState.MAIN: {
        'name': 'Main Mode',
        'description': '''
- This is the default starting mode.
- Does not perform actions by default.
- You can return here from:
    - Data Collection Mode (after quitting)
    - Prediction Mode (after resetting the model or data)
        '''
    },
    MenuState.DATA_COLLECTION: {
        'name': 'Data Collection Mode',
        'description': '''
- Collects training samples with the current label.
- Entered by pressing 'a' in Main Mode.
        '''
    },
    MenuState.PREDICTION: {
        'name': 'Prediction Mode',
        'description': '''
- Predicts the label of the current sample by default.
- Entered by pressing 't' in Main Mode.
        '''
    }
}