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

def load_data():
    machine = Machine() 
    image_path = machine.image_path
    lable_path = machine.label_path 

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
