model_id = input('Pick Model: ')

print('Starting Imports')
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
from glob import glob
from model import unet 
from PIL import Image
from datetime import date
import csv




image_path = Path(r'/home/user/damage/Numpy')
lable_path = Path(r'/home/user/damage/Mask')

images = list() 
masks = list()
print('Loading Data Set')
for idx, pth in enumerate(tqdm(glob( str(image_path) + '/*.npy'))):
    
    if idx == 10:
        break

    pth = Path(pth)
    arry = np.load(pth)
    arry = arry.astype('float64') / 255

    height, width, _ = arry.shape
    size = 572

    mask = np.load(str(lable_path) + '/' + pth.name)


    for x in range(height // size):
        for y in range(width // size):
            s = size 
            images.append(arry[s * x: s * (x+1), s * y : s * (y+1),:]) 
            m = mask[s * x: s * (x+1), s * y : s * (y+1)]
            m = np.rot90(m)
            m = np.flipud(m)
            masks.append(m)


train_images, test_images, train_masks, test_masks = train_test_split(images, 
masks, test_size = 0.2, random_state=42)


def make_picture(mask, image):
    image = (image * 255).astype(np.uint8) 
    mask = mask.astype('int32')

    image[mask[:,:,0] == 1] = (255, 0, 0)
    Image.fromarray(image).show()


print('Loading Model')
save_path = Path(r'/home/users/models')
model = tf.keras.models.load_model(str(save_path) +  f'/{model_id}.keras')
breakpoint()

print("Testing")
def save_eval(model):
    global test_images, train_images, epochs, optimizer

    save_path = Path(r'/home/users/models')
    index = str(len(glob("save_path" + "/*.keras")))
    

    model.save(str(save_path) + '/' + index + '.keras')
    
    test_loss, test_acc = model.evaluate(test_images,  test_masks, verbose=2)
    print('Test Loss: ', test_loss, ' Test acc: ', test_acc)

    meta = {'Index': index,
    'Type' : 'Unet', 
    'Train Size': len(train_images),
    'Test Size': len(test_images),
    'Test Loss': test_loss,
    'Test_acc': test_acc,
    'Date': date.today()
    }
    
    with open(save_path / 'eval.csv', mode='a') as file:
        writer = csv.writer(file)
        writer.writerow(list(meta.values()))

