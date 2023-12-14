print('Imports')
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from model import unet 
from aux import save_model, load_data
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from evaluate import make_picture
from tqdm import tqdm


# Nobs and dials
epochs = 50 
batch_size = 6 
optimizer = 'adam'


model = unet()
print('Compling Model')
model.compile(optimizer='adam',
              loss = keras.losses.BinaryCrossentropy(),
              metrics = [tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0]), tf.keras.metrics.Recall()] )

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')
losses = list()

print('Fitting Model')
dataset = tf.data.Dataset.from_generator(load_data, output_types=(tf.float32, tf.float32))
i = 0
temp = list()
for images, masks in dataset.batch(1).take(1):
    breakpoint()
    model.fit(images,
              masks[:,:,:,0],
              epochs=100, 
              callbacks=[tensorboard_callback]
              )
    i += 1
    temp.append(float(model.history.history['loss'][0]))
    if i % 100 == 0:
        losses.append( np.mean(np.array(temp)))
        temp = list()

save_model(model, epochs, batch_size, losses)

for images, masks in dataset.batch(50).take(1):
    make_picture(model, images, masks)

