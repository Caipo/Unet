import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date
from glob import glob
from patchify import patchify
from random import sample 
from utils.machines import Machine
import numpy as np
import math
import os
import csv

def save_model(model, epochs, batch_size, losses, train_time):
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
    'Train_Time' : round(train_time),
    'Epocs': epochs,
    'Date': date.today()
    }
    
    with open(save_path / 'meta.csv', mode='a') as file:
        writer = csv.writer(file)
        writer.writerow(list(meta.values()))

def load_data(patch = True, test = False):
    machine = Machine() 
    
    if not test:
        image_path = machine.image_path
        label_path = machine.label_path 

    else:
        image_path = machine.test_image_path
        label_path = machine.test_label_path

    images = list() 
    masks = list()
    data = list()
    files =  list(glob( str(image_path) + '/*.npy'))


    size = 572
    for path in files:
        # Loading
        image = np.load(Path(path))
        mask = np.load(str(label_path) + '/' + str(Path(path).name))

        height, width, _ = image.shape

        # Normalizing 
        image = image.astype('float32') / 255

        if not patch:
            yield image, mask

        # Padding 
        pad_height =  math.ceil(height / size) * size - height 
        pad_width =  math.ceil(width / size) * size - width 

        image = np.pad(image, [(0,pad_height), (0, pad_width), (0,0)])
        mask = np.pad(mask, [(0,pad_height), (0, pad_width), (0,0)])

        step_size = 572
        
        # Patching
        image_patched = np.squeeze(patchify(image, (572, 572, 3), step=step_size))
        mask_patched = np.squeeze(patchify(mask, (572, 572, 2), step=step_size))

        
    
        for row in range(len(mask_patched)):
            for col in range(row):
                yield image_patched[row][col], mask_patched[row][col]

def load_test():
    machine = Machine() 
    
    image_path = machine.test_image_path
    label_path = machine.test_label_path

    images = list() 
    masks = list()
    data = list()
    files =  list(glob( str(image_path) + '/*.npy'))


    size = 572
    for path in files:
        # Loading
        image = np.load(Path(path))
        mask = np.load(str(label_path) + '/' + str(Path(path).name))

        height, width, _ = image.shape

        # Normalizing 
        image = image.astype('float32') / 255

        yield image, mask

