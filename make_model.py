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
from tqdm import tqdm




# Nobs and dials
limit_size = 999999 
epochs = 100 
batch_size = 1
optimizer = 'adam'


model = unet()
print('Compling Model')
model.compile(optimizer='adam',
              loss = keras.losses.BinaryCrossentropy(),
              metrics = [tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])] )

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)

print('Fitting Model')
for i in range(epochs):
    images, masks = load_data(batch_size)
    model.fit(images,
              masks, 
              epochs=1, 
              callbacks=[tensorboard_callback]
              )

save_model(model, epochs, batch_size)

