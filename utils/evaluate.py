if __name__ == '__main__':
    model_id = input('Pick Model: ')

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
from utils.machines
import os
import csv




images = list() 
masks = list()

print('Loading Data Set')
m = machine()
image_path = m.image_path
label_path = m.label_path 
predic_path = m.predic_path
save_path = m.save_path

def make_picture(model, image, mask):
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
        Image.fromarray(img).save( str(predic_path) + '/' + str(idx) + '.jpg' )

print("Testing")
def save_eval(model):
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
    model = tf.keras.models.load_model(str(save_path) +  f'/{model_id}.keras')
    
    dataset = tf.data.Dataset.from_generator(load_data, output_types=(tf.float32, tf.float32))
    for images, masks in dataset.batch(50).take(1):
        make_picture(model, images, masks)

