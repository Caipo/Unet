import platform
from pathlib import Path

class Machine():
    def __init__(self):
        if platform.system() == 'Darwin':
            
            base = Path(r'/Users/work/Data/damage')
            self.image_path = base / 'Numpy' 
            self.label_path = base / 'Mask'

        elif platform.system() == 'Linux':
            self.image_path = Path(r'/home/nick/Data/Numpy')
            self.label_path = Path(r'/home/nick/Data/Mask')







