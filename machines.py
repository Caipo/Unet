import platform
from pathlib import Path

class Machine():
    def __init__(self):
        if platform.system() == 'Darwin':
            self.image_path = Path(r'/home/user/damage/data/Numpy')
            self.lable_path = Path(r'/home/user/damage/Mask')

        elif name == 'Linux'
            self.image_path = Path(r'/home/nick/Data/Numpy')
            self.lable_path = Path(r'/home/nick/Data/Mask')







