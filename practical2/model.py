from keras.models import Model
from keras.layers import Input, Dense, Flatten, Convolution2D, MaxPooling2D, Dropout


def get_model():

    # Model parameters
    rows, cols = 28, 28
    input_shape = (rows, cols, 1)

    nb_classes = 10

    hidden_size = 512

    inp = Input(shape=input_shape)
    
    conv_depth_1 = 32
    conv_depth_2 = 64
    kernel_size = 3
    pool_size = 2
    drop_prob = 0.25
    # Addition: CNN

    conv_1 = Convolution2D(
        conv_depth_1, (kernel_size, kernel_size),
        padding='same',
        activation='relu')(inp)
    conv_2 = Convolution2D(
        conv_depth_1, (kernel_size, kernel_size),
        padding='same',
        activation='relu')(conv_1)
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
    drop_1 = Dropout(drop_prob)(pool_1)
    conv_3 = Convolution2D(
        conv_depth_2, (kernel_size, kernel_size),
        padding='same',
        activation='relu')(drop_1)
    conv_4 = Convolution2D(
        conv_depth_2, (kernel_size, kernel_size),
        padding='same',
        activation='relu')(conv_3)
    pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
    drop_2 = Dropout(drop_prob)(pool_2)

    # conv2 = Convolution2D(hidden_size, 3, 3)(pool1)
    # pool2 = MaxPooling2D()(conv2)

    flat = Flatten()(drop_2)
    hidden_1 = Dense(hidden_size, activation='relu')(flat)
    drop_3 = Dropout(0.1)(hidden_1)
    hidden_2 = Dense(hidden_size, activation='relu')(drop_3)
    out = Dense(nb_classes, activation='softmax')(hidden_2)

    model = Model(inputs=inp, outputs=out)

    print(model.summary())

    return model


if __name__ == '__main__':

    model = get_model()
"""
    Avenues:
    - Make a CNN before flattening and running standard NN.
    - Regularise
    - remove vanishing gradient problem
"""