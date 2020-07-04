# -*- coding: utf-8 -*-
'''
Convolutional Neural Network
'''

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Dropout
from keras import backend as K


class CNNNet:
    @staticmethod
    def build(input_shape_width, input_shape_height, classes,
              weight_path='', input_shape_depth=3):
        '''
        weight_path: a .hdf5 file. If exists, we can load model.
        '''

        # initialize the model
        model = Sequential()

        input_shape = (input_shape_height, input_shape_width,
                       input_shape_depth)
        # if we are using "channels first", update the input shape
        if K.image_data_format() == 'channels_first':
            input_shape = (input_shape_depth, input_shape_height,
                           input_shape_width)

        # first Convolution + relu + pooling layer
        model.add(Conv2D(filters=80, kernel_size=(3, 3),
                         padding='same', input_shape=input_shape))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        # second convolutional layer
        model.add(Conv2D(filters=80, kernel_size=(3, 3),
                         padding='same'))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        # third convolutional layer
        model.add(Conv2D(filters=64, kernel_size=(3, 3),
                         padding='same'))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        # # fourth convolutional layer
        # model.add(Conv2D(filters=64, kernel_size=(3, 3),
        #                  padding='same'))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


        # Flattening
        model.add(Flatten())

        # Full connection
        model.add(Dense(units=512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # # Full connection
        # model.add(Dense(units=256))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.5))

        # output layer
        model.add(Dense(units=classes))
        model.add(Activation('softmax'))

        if weight_path:
            model.load_weights(weight_path)

        # return the constructed network architecture
        return model
