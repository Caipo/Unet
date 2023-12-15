import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date
from glob import glob
from patchify import patchify
from random import sample 
from machines import Machine
import numpy as np
import math
import os
import csv

def save_model(model, epochs, batch_size, losses):
    save_path =  Path(os.getcwd()) / 'Save'

    index = str(len(glob(str(save_path) + '/*.keras'))) # For refrence 
    model.save(str(save_path) + '/' + index +'.keras') 

    # Loss Plot
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

def load_data():
    machine = Machine() 
    image_path = machine.image_path
    lable_path = machine.label_path 

    images = list() 
    masks = list()
    data = list()

    files =  list(glob( str(image_path) + '/*.npy'))


    size = 572
    for path in files:

        path = np.load(Path(path))
        mask = np.load(str(lable_path) + '/' + Path(path).name)

        height, width, _ = arry.shape
        image = image.astype('float32') / 255

        # Over lap code not ready yet
        #step_size = lambda x : int(( size * math.ceil(x / size) - x) / (x // size))
        #step_ = (size - step_size(height) + 1, size - step_size(width))
        step_size = 572
        
        image_patched = np.squeeze(patchify(arry, (572, 572, 3), step=step_size))
        mask_patched = np.squeeze(patchify(mask, (572, 572, 2), step=step_size))


        for row in range(len(msk)):
            for col in range(i):
                yield img[row][col], msk[row][col]
