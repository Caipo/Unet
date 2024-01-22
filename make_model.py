print('Imports')
import os
import numpy as np
import tensorflow as tf
from time import time
from model import unet
from utils.aux import save_model, load_data
from utils.evaluate import make_picture
from keras.losses import BinaryCrossentropy
from keras.metrics import Recall, IoU, Accuracy
from keras.optimizers.legacy import Adam


# Nobs and dials
epochs = 300 
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
losses = []
temp = []
save_path =  Path(os.getcwd()) / 'Save'
index = str(len(glob(str(save_path) + '/*.keras'))) # For refrence   

# For loss Graph
for images, masks in dataset.batch(batch_size).take(99999).repeat(epochs):
    model.fit(images,
              masks,
              epochs=1, 
              callbacks=[tensorboard_callback]
              )
    
    # Code for the loss plot
    temp.append(float(model.history.history['loss'][0])) 
    if len(temp) >= len(dataset):
        # Loss plot data 
        losses.append(np.mean(np.array(temp)))

        xs = [x for x in range(len(losses))]
        plt.plot(xs, losses)
        plt.savefig(f'Save/Losses/{index}_losses.png')


        temp = []

train_time = (start_time - time()) / 3600 # In hours
print('Saving Model')
# Saves the model + meta data and loss plot 
save_model(model, epochs, batch_size, losses, train_time)

print('Making Predictions')
# Sample predictions from the train set
i = 0 
for images, masks in dataset.batch(10).take(5):
    make_picture(model, images, masks, i)
    i += 1
