# -*- coding: utf-8 -*-

from tensorflow.keras.datasets import mnist

from network.lenet import LeNet

import numpy as np


def train_lenet():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    print(x_train[0] / 255)
    lenet = LeNet(model_filename='lenet_mnist_1.h5', epochs=10, input_shape=(28, 28, 1), weight_decay=1e-3)
    lenet.train(x_train / 255, y_train)


def top_equal(o_1, o_2):
    o_1 = np.asarray(o_1)
    o_2 = np.asarray(o_2)
    o_1_index = o_1.argmax()
    o_2_index = o_2.argmax()
    if o_1_index == o_2_index:
        return True
    return False


# 当第k位有两个相同value的index时，可能出现问题
def top_k(output, label, k):
    if k >= len(output):
        return True
    for i in range(1, k + 1):
        index = np.argmax(output)
        if index == label:
            return True
        else:
            output[index] = np.min(output)
    return False


# i的改变并没有改变np.argmax(output)的值
# 用于计算单个输出的类间距离
def _class_distance(output, label, k):
    for i in range(1, k + 1):
        index = np.argmax(output)
        if index == label:
            return np.power(2, k - 1)
    return 0


# 用于计算两个输出之间的类间距离
def class_based_distance(o_1, o_2, oracle, k):
    diff_1 = _class_distance(o_1, oracle, k)
    diff_2 = _class_distance(o_2, oracle, k)
    return np.abs(diff_1 - diff_2)


# 用于计算两个输出之间的平均绝对偏差(MAD)
def mean_absolute_distance(o_1, o_2, oracle):
    diff_1 = np.abs(o_1 - oracle)
    diff_2 = np.abs(o_2 - oracle)
    if diff_1 == diff_2 and diff_1 == 0:
        return 0
    else:
        return np.abs(diff_1 - diff_2) / (diff_1 + diff_2)
