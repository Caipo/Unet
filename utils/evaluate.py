if __name__ == '__main__':
    import sys
    
    model_id = input('Pick Model: ')
    sys.path.append('/home/jin/Unet')

print('Starting Imports')
import numpy as np
import tensorflow as tf
import math
from tqdm import tqdm
from pathlib import Path
from glob import glob
from model import unet 
from patchify import patchify, unpatchify
from PIL import Image
from datetime import date
from utils.aux import load_data, load_test
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

        Image.fromarray(img).save(f'{str(predic_path)}/{str(idx) + 10 *  str(i)}.jpg')

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

def label_test_images(model, confidence = 0.1):
  test_dataset = tf.data.Dataset.from_generator(load_test,
  output_types=(tf.float32, tf.float32)
  )
  count = 0 
  for image, mask in test_dataset.batch(1).take(5):
      # Batch, Height, Width, Channels 
      image = image[0]
      mask = mask[0]
      height, width, _ = image.shape
        
      # padding calulations
      size = 572
      pad_height =  math.ceil(height / size) * size - height
      pad_width =  math.ceil(width / size) * size - width
      image = np.pad(image, [(0,pad_height), (0, pad_width), (0,0)])
      mask = np.pad(mask, [(0,pad_height), (0, pad_width), (0,0)])
      
      image_patched = np.squeeze(patchify(image, (572, 572, 3), step=size))
      mask_patched = np.squeeze(patchify(mask, (572, 572, 2), step=size))

      rows = image_patched.shape[0] 
      cols = image_patched.shape[1] 
      
      # flattening
      image_patched = image_patched.reshape( (rows * cols, 572, 572, 3))
      new_img = np.copy(image_patched) * 255
    
     
      for i in range(rows * cols):
          for idx, pred in enumerate(model.predict( image_patched[i].reshape(( 1, size, size, 3)))):
              img1 = new_img[i] 
              img1[pred[:,:,0] >= confidence] = (255, 0, 0)

      new_img = new_img.reshape( (rows, cols, 1, size, size, 3))
      new_height, new_width = new_img.shape[0], new_img.shape[1]
      stiched = unpatchify(new_img, (new_height * size , new_width * size, 3))
      stiched = stiched.astype(np.uint8)
      Image.fromarray(stiched).save(f'{str(predic_path)}/{str(count)}.jpg')
      count += 1
       

if __name__ == '__main__':
    print('Loading Model')
    save_path = Path('/home/jin/Unet/Save')
    model = tf.keras.models.load_model(str(save_path) +  f'/{model_id}.keras')
    
    label_test_images(model)
