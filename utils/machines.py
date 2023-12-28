import platform
from pathlib import Path

class Machine():
    def __init__(self):
        if platform.system() == 'Darwin':
            self.image_path = Path(r'/Users/work/Data/damage/Numpy')
            self.label_path = Path(r'/Users/work/Data/damage/Mask')


        elif platform.system() == 'Linux':
            self.image_path = Path(r'/home/nick/Data/Numpy')
            self.label_path = Path(r'/home/nick/Data/Mask')







