import os
import time
from keras.applications import vgg16
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras import regularizers
import keras


basedir = os.path.abspath(os.path.dirname(__file__))


def get_model_from_lib():
    return vgg16.VGG16(include_top=False, weights=None, input_shape=(32, 32, 3), classes=10)


def get_model_from_self():
    weight_decay = 0.0005

    model = Sequential()

    # 1
    model.add(Conv2D(64, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu'
                     , kernel_regularizer=regularizers.l2(weight_decay), name='conv1_1'))
    model.add(BatchNormalization(name='batch1_1'))
    model.add(Dropout(0.3))
    # 2
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'
                     , kernel_regularizer=regularizers.l2(weight_decay), name='conv1_2'))
    model.add(BatchNormalization(name='batch1_2'))
    model.add(MaxPooling2D((2, 2), name='pool1'))
    # 3
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'
                     , kernel_regularizer=regularizers.l2(weight_decay), name='conv2_1'))
    model.add(BatchNormalization(name='batch2_1'))
    model.add(Dropout(0.4))
    # 4
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'
                     , kernel_regularizer=regularizers.l2(weight_decay), name='conv2_2'))
    model.add(BatchNormalization(name='batch2_2'))
    model.add(MaxPooling2D((2, 2), name='pool2'))
    # 5
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'
                     , kernel_regularizer=regularizers.l2(weight_decay), name='conv3_1'))
    model.add(BatchNormalization(name='batch3_1'))
    model.add(Dropout(0.4))
    # 6
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'
                     , kernel_regularizer=regularizers.l2(weight_decay), name='conv3_2'))
    model.add(BatchNormalization(name='batch3_2'))
    model.add(Dropout(0.4))
    # 7
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'
                     , kernel_regularizer=regularizers.l2(weight_decay), name='conv3_3'))
    model.add(BatchNormalization(name='batch3_3'))
    model.add(MaxPooling2D((2, 2), name='pool3'))
    # 8
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'
                     , kernel_regularizer=regularizers.l2(weight_decay), name='conv4_1'))
    model.add(BatchNormalization(name='batch4_1'))
    model.add(Dropout(0.4))
    # 9
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'
                     , kernel_regularizer=regularizers.l2(weight_decay), name='conv4_2'))
    model.add(BatchNormalization(name='batch4_2'))
    model.add(Dropout(0.4))
    # 10
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'
                     , kernel_regularizer=regularizers.l2(weight_decay), name='conv4_3'))
    model.add(BatchNormalization(name='batch4_3'))
    model.add(MaxPooling2D((2, 2), name='pool4'))
    # 11
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'
                     , kernel_regularizer=regularizers.l2(weight_decay), name='conv5_1'))
    model.add(BatchNormalization(name='batch5_1'))
    model.add(Dropout(0.4))
    # 12
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'
                     , kernel_regularizer=regularizers.l2(weight_decay), name='conv5_2'))
    model.add(BatchNormalization(name='batch5_2'))
    model.add(Dropout(0.4))
    # 13
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'
                     , kernel_regularizer=regularizers.l2(weight_decay), name='conv5_3'))
    model.add(BatchNormalization(name='batch5_3'))
    model.add(MaxPooling2D((2, 2), name='pool5'))
    # 14
    model.add(Flatten(name='flatten'))
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(weight_decay), name='dense1'))
    model.add(BatchNormalization(name='batch6_1'))
    # 15
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(weight_decay), name='dense2'))
    model.add(BatchNormalization(name='batch6_2'))
    # 16
    model.add(Dropout(0.5))
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
