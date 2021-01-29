
import os
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential

import time


basedir = os.path.abspath(os.path.dirname(__file__))


def train(input_shape, train_x, train_y, save=False, save_time = False):
    model = Sequential()
    # block1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=input_shape[1:]))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(Dense(4096, activation='relu', name='fc2'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax', name='predictions'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    start_time = time.asctime(time.localtime(time.time()))
    model.fit(x=train_x.reshape(input_shape), y=train_y, batch_size=256, epochs=20)
    end_time = time.asctime(time.localtime(time.time()))
    if save_time:
        f = open(os.path.join(basedir, 'network', 'models', 'test', 'train_time_tf.txt'), 'w')
        f.write('start_time: ' + start_time + '\n')
        f.write('end_time: ' + end_time + '\n')
        f.close()
    if save:
        model.save(os.path.join(basedir, 'network', 'models', 'test', 'fashion_mnist_tf.h5'))

    return model
