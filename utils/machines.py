import platform
from pathlib import Path
import os 

class Machine():
    def __init__(self):
        if platform.system() == 'Darwin':
            self.image_path = Path(r'/Users/work/Data/damage/Numpy')
            self.label_path = Path(r'/Users/work/Data/damage/Mask')

            self.test_image_path = Path(r'/Users/work/Data/damage/Numpy')
            self.test_label_path = Path(r'/Users/work/Data/damage/Mask')

            self.predic_path  = Path(r'/Users/work/Unet/Save/')

        elif platform.system() == 'Linux':
            self.image_path = Path(r'/home/jin/Data/Numpy')
            self.label_path = Path(r'/home/jin/Data/Mask')

            self.test_image_path = Path(r'/Users/work/Data/damage/Numpy')
            self.test_label_path = Path(r'/Users/work/Data/damage/Mask')

            self.predic_path  = Path(r'/home/jin/Unet/Save/Predic')
        


        self.save_path = Path(os.getcwd(),  'Save')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        

        losses_path = save_path / 'losses' 
        if not os.path.exists(losses_path):
            os.makedirs(losses_path)

        if not os.path.exists(predic_path):
            os.makedirs(predic_path)
