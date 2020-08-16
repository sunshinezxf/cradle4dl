# -*- coding: utf-8 -*-

from tensorflow.keras.datasets import mnist

from network.lenet import LeNet


def train_lenet():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    print(x_train[0] / 255)
    lenet = LeNet(model_filename='lenet_mnist_1.h5', epochs=10, input_shape=(28, 28, 1), weight_decay=1e-3)
    lenet.train(x_train / 255, y_train)


def top_equal(k, o_1, o_2):
    print("top %s", k)
    print(o_1 == o_2)
    return
