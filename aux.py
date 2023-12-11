from pathlib import Path
from datetime import date
from glob import glob
from random import sample 
import numpy as np
import os
import csv

def save_model(model, epochs, batch_size):
    if os.name == 'nt':
        save_path = Path(r'C:\Users\employee\Desktop\damage\Save')
        index = str(len(glob(str(save_path) + '/*.keras')))
    else:
        save_path = Path('/home/user/save')
        index = str(len(glob(str(save_path) + '/*.keras')))

    model.save(str(save_path) + '/' + index + '.keras')
    
    meta = {'Index': index,
    'Type' : 'Unet', 
    'Optimizer' : 'Adam',
    'Batch Size' : batch_size,
    'Loss': 'Binary Cross Entropy', 
    'Epocs': epochs,
    'Date': date.today()
    }
    
    with open(save_path / 'meta.csv', mode='a') as file:
        writer = csv.writer(file)
        writer.writerow(list(meta.values()))


def load_data(batch_size):
    if os.name == 'nt':
        image_path = Path(r'C:\Users\employee\Desktop\damage\data\Numpy')
        lable_path = Path(r'C:\Users\employee\Desktop\damage\Mask')
        
    else:
        image_path = Path(r'/home/user/damage/data/Numpy')
        lable_path = Path(r'/home/user/damage/Mask')
    

    images = list() 
    masks = list()

    files =  sample(list(glob( str(image_path) + '/*.npy')), batch_size)


    size = 572
    for pth in files:

        pth = Path(pth)
        arry = np.load(pth)

        height, width, _ = arry.shape
        arry = arry.astype('float32') / 255
        mask = np.load(str(lable_path) + '/' + pth.name)

        for x in range(height // size):
            for y in range(width // size):
                s = size 
                images.append(arry[s * x: s * (x+1), s * y : s * (y+1),:]) 
                m = mask[s * x: s * (x+1), s * y : s * (y+1)]
                m = np.rot90(m)
                m = np.flipud(m)
                masks.append(m)
          
      

    return np.array(images, dtype='float32'), np.array(masks, dtype='float32')
