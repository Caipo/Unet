print('Imports')
import os
import numpy as np
import tensorflow as tf
from time import time
from model import unet
from utils.aux import save_model, load_data
from pathlib import Path
from glob import glob
from utils.evaluate import make_picture, label_test_images
from keras.losses import BinaryCrossentropy
from keras.metrics import Recall, IoU, Accuracy
from keras.optimizers.legacy import Adam


# Nobs and dials
epochs = 200 
batch_size = 4
optimizer = 'adam'
learning_rate = 0.001

print('Compling Model')
model = unet()
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss = BinaryCrossentropy(),
              metrics = [IoU(num_classes=2, target_class_ids=[0]), 
                         Recall(),
                         Accuracy()
                        ]
              )

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')

print('Generating data set')
dataset = tf.data.Dataset.from_generator(load_data,  
                                         output_types=(tf.float32, tf.float32)
                                         )

start_time = time()
print('Fitting Model')

# For loss Graph
losses = []
for images, masks in dataset.batch(batch_size).take(1).repeat(epochs):
    model.fit(images,
              masks,
              epochs=1, 
              callbacks=[tensorboard_callback]
              )
    losses.append(float(model.history.history['loss'][0]))


losses = np.array(losses)
losses = losses.reshape(-1, len(losses) // epochs).mean(axis=1) 
train_time = (start_time - time()) / 3600 # In hours
print('Saving Model')
# Saves the model + meta data and loss plot 
save_model(model, epochs, batch_size, losses, train_time)

print('Making Predictions')
# Sample predictions from the train set
i = 0 

label_test_images(model)
