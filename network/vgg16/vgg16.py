import os
import time
from keras.applications import vgg16
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from keras.optimizers import SGD
import keras


basedir = os.path.abspath(os.path.dirname(__file__))


def get_model_from_lib():
    vgg16.VGG16(include_top=False, weights=None, input_shape=(32, 32, 3), classes=10)


def get_model_from_self():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(32, 32, 3), padding='same', activation='relu'
                     , name='conv1_1'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool1'))

    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv2_1'))
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool2'))

    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv3_1'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv3_2'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool3'))

    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv4_1'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv4_2'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool4'))

    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv5_1'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv5_2'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool5'))

    model.add(Flatten(name='flatten'))
    model.add(Dense(512, activation='relu', name='dense1'))
    model.add(Dense(512, activation='relu', name='dense2'))
    model.add(Dense(10, activation='softmax', name='dense3'))

    model.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(-1, 32, 32, 3)
    x_test = x_test.reshape(-1, 32, 32, 3)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    # y_train = y_train.reshape(-1, 1, 10)
    # y_test = y_test.reshape(-1, 1, 10)
    return (x_train, y_train), (x_test, y_test)


def train_model():
    model = get_model_from_self()
    model_file_name = 'vgg16_cifar10_' + keras.backend.backend() + '.h5'
    model_path_name = os.path.join(basedir.split('vgg16')[0], 'models', 'vgg16', model_file_name)
    model.save(model_path_name)
    (x_train, y_train), (x_test, y_test) = load_data()
    print('backend: ' + keras.backend.backend())
    print('start_time: ' + time.asctime(time.localtime(time.time())))
    model.fit(x_train, y_train, batch_size=32, epochs=100, validation_split=0.1, verbose=1)
    print('end_time: ' + time.asctime(time.localtime(time.time())))
    model.save(model_path_name)
    print(model.evaluate())
    print('\n\n\n')


if __name__ == '__main__':
    train_model()
