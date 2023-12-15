import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, Reshape, Conv2DTranspose, UpSampling2D 
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def down(input_net, layer):
    pad = 'same'
    act = 'relu'

    height = input_net.shape[1]
    width = input_net.shape[2]

    filt = pow(2, layer + 5)

    con1 = Conv2D(filt, (3,3), strides=(1,1), activation=act, padding=pad)(input_net)
#    con1 = CenterCrop(height - 2, height - 2)(con1)

    con2 = Conv2D(filt, (3,3), strides=(1,1), activation=act, padding=pad)(con1)
#    con2 = CenterCrop(height -4, height - 4)(con2)
    
    return con2

def up(input_net, con, layer):
    pad = 'same'
    act = 'relu'

    height = input_net.shape[1]
    width = input_net.shape[2]

    cat1 = Concatenate()([CenterCrop(height, width)(con), input_net ])

    con1 = Conv2D(1024 // pow(2, layer), (3,3), strides=(1,1), activation=act, padding=pad)(cat1)
#    con1 = CenterCrop(height - 2, height - 2)(con1)

    con2 = Conv2D(1024 // pow(2, layer), (3,3), strides=(1,1), activation=act, padding=pad)(con1)
#    con2 = CenterCrop(height - 4, height - 4)(con2)
    return con2

def unet():
    input_net = Input((572,572,3), dtype = 'float32')
    
    pool_pad = 'same'
    act = 'relu'

    d1 = down(input_net, 1)
    max1 = MaxPooling2D((2,2), (2,2))(d1)

    d2 = down(max1, 2)
    max2 = MaxPooling2D((2,2), (2,2))(d2)

    d3 = down(max2, 3)
    max3 = MaxPooling2D((2,2), (2,2))(d3)

    d4 = down(max3, 4)
    max4 = MaxPooling2D((2,2), (2,2))(d4)

    d5 = down(max4, 5)
    con1 = Conv2DTranspose(512, 3, 2, padding=pool_pad)(d5)

    u1 = up(con1, d4, 1)
    con2 = Conv2DTranspose(256, 3, 2, padding=pool_pad)(u1)

    u2 = up(con2, d3, 2)
    con3 = Conv2DTranspose(128, 3, 2, padding=pool_pad)(u2)

    u3 = up(con3, d2, 3)
    con4 = Conv2DTranspose(64, 3, 2, padding=pool_pad)(u3)
    
    u4 = up(con4, d1, 4)
    last_con = Conv2D(2, (1,1), strides=(1,1), activation='softmax', padding='same')(u4)
    
    breakpoint()
    output = tf.image.resize(last_con, [572, 572]) 
    model = Model(inputs = input_net, outputs = output)
    return model
