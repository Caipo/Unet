if __name__ == '__main__':
    import sys
    model_id = input('Pick Model: ')
    sys.path.append('..')

print('Starting Imports')
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path
from glob import glob
from model import unet 
from PIL import Image
from datetime import date
from utils.aux import load_data
from utils.machines import Machine
import os
import csv
from random import randint



images = list() 
masks = list()

print('Loading Data Set')
m = Machine()
image_path = m.image_path
label_path = m.label_path 
predic_path = m.predic_path
save_path = m.save_path

def make_picture(model, image, mask, i):
    output_path = Path(os.getcwd(),  'Output')
    image = np.array(image)
    mask = np.array(mask)

    for idx, pred in enumerate(model.predict(image)):

        # Prediction Image
        img1 = image[idx]
        img1 = (img1 * 255).astype(np.uint8) 
        raw_img = np.copy(img1) 
        img1[pred[:,:,0] >= 0.05] = (255, 0, 0)
       
        m = mask[idx]
        img2 = image[idx]
        img2 = (img2 * 255).astype(np.uint8) 
        img2[m[:,:,0] != 0] = (0, 255, 0)

        img = np.concatenate((img1, raw_img ), axis=1)
        img = np.concatenate((img, img2), axis = 1)

        Image.fromarray(img).save(f'{str(predic_path)}/{str(idx)}-{str(i)}.jpg' )

def eval():
    index = str(len(glob("save_path" + "/*.keras")))
    

    model.save(str(save_path) + '/' + index + '.keras')
    
    #test_loss, test_acc = model.evaluate(test_images,  test_masks, verbose=2)
    print('Test Loss: ', test_loss, ' Test acc: ', test_acc)

    meta = {'Index': index,
    'Type' : 'Unet', 
    #'Test Loss': test_loss,
    #'Test_acc': test_acc,
    'Date': date.today()
    }
    
    with open(save_path / 'eval.csv', mode='a') as file:
        writer = csv.writer(file)
        writer.writerow(list(meta.values()))

if __name__ == '__main__':
    print('Loading Model')
    save_path = Path('/home/jin/Unet/Save')
    model = tf.keras.models.load_model(str(save_path) +  f'/{model_id}.keras')
    
    dataset = tf.data.Dataset.from_generator(load_data, output_types=(tf.float32, tf.float32))
    i = 0
    for images, masks in dataset.batch(10).take(5):
        make_picture(model, images, masks, i)
        i += 1
