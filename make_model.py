print('Imports')
import os
import numpy as np
import tensorflow as tf
from model import unet
from utils.aux import save_model, load_data
from utils.evaluate import make_picture
from keras.losses import BinaryCrossentropy
from keras.metrics import Recall, IoU, Accuracy
from keras.optimizers.legacy import Adam

# Nobs and dials
epochs = 1000 
batch_size = 1 
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
                                         output_types=(tf.float32, tf.float32))

print('Fitting Model')
losses = []
# For loss Graph
for images, masks in dataset.batch(batch_size).take(99999).repeat(epochs):
    model.fit(images,
              masks,
              epochs=1, 
              callbacks=[tensorboard_callback]
              )

    # Loss plot data 
    losses.append(float(model.history.history['loss'][0]))

print('Saving Model')
# Saves the model + meta data and loss plot 
save_model(model, epochs, batch_size, losses)

print('Making Predictions')
# Sample predictions from the train set
for images, masks in dataset.batch(20).take(1):
    make_picture(model, images, masks)

