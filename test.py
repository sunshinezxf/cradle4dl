
import os
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import load_model
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


if __name__ == "__main__":
    test3()
