print('Imports')
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from model2 import unet
from aux import save_model, load_data
from evaluate import make_picture
from tqdm import tqdm
from keras.losses import BinaryCrossentropy
from keras.metrics import Recall, IoU
from keras.optimizers import Adam

# Nobs and dials
epochs = 50 
batch_size = 6 
optimizer = 'adam'
learning_rate = 0.001


model = unet()
print('Compling Model')
model.compile(optimizer=Adam(learning_rate=0.001),
              loss = BinaryCrossentropy(),
              metrics = [IoU(num_classes=2, target_class_ids=[0]),
                         Recall()
                        ])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')

print('Fitting Model')
#Data gen
dataset = tf.data.Dataset.from_generator(load_data, output_types=(tf.float32, tf.float32))

# For loss Graph
i = 0
losses = list()
temp = list()
for images, masks in dataset.batch(16).take(1).repeat(500):
    model.fit(images,
              masks,
              epochs=1, 
              callbacks=[tensorboard_callback]
              )
    i += 1
    temp.append(float(model.history.history['loss'][0]))

    if i % 10 == 0:
        losses.append( np.mean(np.array(temp)))
        temp = list()

save_model(model, epochs, batch_size, losses)

for images, masks in dataset.batch(50).take(1):
    make_picture(model, images, masks)

