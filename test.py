
import os
from keras.datasets import fashion_mnist
from keras.datasets import cifar10
from keras.models import load_model
import tensorflow as tf
from utils import nnutil

import time

from network.test_model import test_model


basedir = os.path.abspath(os.path.dirname(__file__))


def test1():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    shape = (-1, 28, 28, 1)
    # print(shape[1:])
    test_model.train(shape, x_train, y_train, True, True)


def test2():
    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print(tf.keras.backend.backend())


def test3():
    model = load_model(basedir + "/network/models/test/tf/" + "fashion_mnist_1.h5")
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_test = x_test.reshape(-1, 28, 28, 1)
    outputs = model.predict(x_test)
    print(outputs)
    # layers_output1 = nnutil.layers_output(model, x_test)
    # print(layers_output1.shape)
    # layers_output2 = nnutil.layers_output_new(model, x_test)


def test4():
    model = load_model(basedir + "/network/models/vgg16/" + "vgg16_cifar10_tensorflow.h5")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_test = x_test.reshape(-1, 32, 32, 3)
    # outputs = model.predict(x_test)
    # print(outputs)
    layers_output1 = nnutil.layers_output(model, x_test[0:1])
    print(layers_output1.shape)
    # layers_output2 = nnutil.layers_output_new(model, x_test)


def test5():
    model = load_model(basedir + "/network/models/vgg16/" + "vgg16_cifar10_tensorflow.h5")
    for layer in model.layers:
        print(layer)
    layers = nnutil.extract_model_layer(model)
    for layer in layers:
        print(layer)
    print(len(model.layers))
    print(len(layers))
    # print(layers)
    # layers_output2 = nnutil.layers_output_new(model, x_test)


def test6():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = tf.image.resize_images(x_train, [64, 64], 0)
    print(x_train.shape)


if __name__ == "__main__":
    test6()
