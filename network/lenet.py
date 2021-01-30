# -*- coding: utf-8 -*-

from keras import optimizers
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.regularizers import l2
from keras.datasets import mnist
from keras import backend

from network.net import Net


class LeNet(Net):

    def __init__(self, model_filename='lenet.h5', input_shape=(28, 28, 1), num_classes=10, epochs=200, batch_size=128,
                 weight_decay=1e-3, train=True):
        super(LeNet, self).__init__('lenet', model_filename, batch_size, epochs, num_classes, train)
        self.input_shape = input_shape
        self.weight_decay = weight_decay

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(6, (5, 5), padding='valid', activation='relu',
                         kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay),
                         input_shape=self.input_shape))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Conv2D(16, (5, 5), padding='valid', activation='relu',
                         kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay)))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(120, activation='relu', kernel_initializer='he_normal',
                        kernel_regularizer=l2(self.weight_decay)))
        model.add(Dense(84, activation='relu', kernel_initializer='he_normal',
                        kernel_regularizer=l2(self.weight_decay)))
        model.add(Dense(self.num_classes, activation='softmax', kernel_initializer='he_normal',
                        kernel_regularizer=l2(self.weight_decay)))
        sgd = optimizers.SGD(lr=self.weight_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        model.summary()
        return model


def train_lenet():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    print(x_train[0] / 255)
    lenet = LeNet(model_filename='lenet_mnist_'+backend.backend()+'.h5', epochs=10, input_shape=(28, 28, 1), weight_decay=1e-3)
    lenet.train(x_train / 255, y_train)


if __name__ == '__main__':
    train_lenet()
