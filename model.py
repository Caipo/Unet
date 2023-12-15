import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, \
Concatenate, Input, Conv2DTranspose, Dropout, BatchNormalization

from tensorflow.keras.layers.experimental.preprocessing import CenterCrop
from tensorflow.keras.models import Model


def double_conv(input_layer, filt ):
    conv_a = Conv2D(filt, (3,3), activation='relu', padding='same')(input_layer)
    conv_a = BatchNormalization()(conv_a)
    conv_b = Conv2D(filt, (3,3), activation='relu', padding='same')(conv_a)
    conv_b = BatchNormalization()(conv_b)
    return conv_b

def down(input_layer, filt):
    return double_conv(input_layer, filt)

def up(input_layer, skip, filt):
    skip = CenterCrop(input_layer.shape[1], input_layer.shape[2])(skip)
    cat = Concatenate()([input_layer, skip])
    cat = Dropout(0.5)(cat)
    return double_conv(cat, filt)

def unet():
    image = Input((572,572,3), dtype = 'float32')

    # ENCODING 
    encoding = list()
    for i in range(4):
        filt = pow(2, i + 6)
        if i == 0:
            down_block = down(image, filt)
            drop = 0.25

        else:
            down_block = down(pool, filt)
            drop = 0.5

        encoding.append(down_block)
        pool = MaxPooling2D((2,2))(down_block)
        pool = Dropout(drop)(pool)

    
    bottom = double_conv(pool, filt = 1024)
   
    # DECODING 
    decoding = list()
    for i in range(4):
        filt = 1024 // pow(2, i + 1)

        if i == 0:
            layer = bottom
        else:
            layer = up_block

        conv = Conv2DTranspose(filt, (3,3), strides=(2,2), padding='same')(layer) 

        up_block = up(conv, encoding[-1 * (i + 1)], filt)
        decoding.append(up_block) 


    last_conv = Conv2D(2, (1,1), padding='same', activation='sigmoid')(decoding[-1])
    mask = tf.image.resize(last_conv, [572, 572])
    return Model(inputs = image, outputs=mask)

if __name__ == '__main__':
    unet()
