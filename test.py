
import os
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
import tensorflow as tf

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
    model.fit(x=train_x.reshape(input_shape), y=train_y, batch_size=128, epochs=100)
    end_time = time.asctime(time.localtime(time.time()))
    if save_time:
        f = open(os.path.join(basedir, 'network', 'models', 'test', 'tf', 'train_time.txt'), 'w')
        f.write('start_time: ' + start_time + '\n')
        f.write('end_time' + end_time + '\n')
        f.close()
    if save:
        model.save(os.path.join(basedir, 'network', 'models', 'test', 'tf', 'fashion_mnist_1.h5'))

    return model


def test1():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    shape = (-1, 28, 28, 1)
    # print(shape[1:])
    train(shape, x_train, y_train, True, True)


def test2():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


if __name__ == "__main__":
    # test2()
    test1()
