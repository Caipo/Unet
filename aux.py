import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date
from glob import glob
from patchify import patchify
from random import sample 
import numpy as np
import math
import os
import csv

def save_model(model, epochs, batch_size, losses):
    save_path =  Path(os.getcwd()) / 'Save'

    index = str(len(glob(str(save_path) + '/*.keras')))

    model.save(str(save_path) + '/' + index +'.keras') 
    xs = [x for x in range(len(losses))]
    plt.plot(xs, losses)
    plt.savefig(f'Save/Losses/{index}_losses.png')

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

def load_train():
    return

def load_data(batch_size = 6):
    if os.name == 'nt':
        image_path = Path(r'/home/user/damage/data/Numpy')
        lable_path = Path(r'/home/user/damage/Mask')

    elif os.name == 'posix':
        image_path = Path(r'/home/nick/Data/Numpy')
        lable_path = Path(r'/home/nick/Data/Mask')

    else:
        image_path = Path(r'C:\Users\employee\Desktop\damage\data\Numpy')
        lable_path = Path(r'C:\Users\employee\Desktop\damage\Mask')
    

    images = list() 
    masks = list()
    data = list()

    files =  list(glob( str(image_path) + '/*.npy'))


    size = 572
    for pth in files:

        pth = Path(pth)
        arry = np.load(pth)

        height, width, _ = arry.shape
        arry = arry.astype('float32') / 255
        mask = np.load(str(lable_path) + '/' + pth.name)

        #step_size = lambda x : int(( size * math.ceil(x / size) - x) / (x // size))
        #step_ = (size - step_size(height) + 1, size - step_size(width))
        step_ = 572
        
        img  = np.squeeze(patchify(arry, (572, 572, 3), step=step_))
        msk  = np.squeeze(patchify(mask, (572, 572, 2), step=step_))

        for i in range(len(msk)):
            for j in range(i):
                yield img[i][j], msk[i][j]
