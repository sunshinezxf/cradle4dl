
import os
from keras.models import load_model
from keras.datasets import mnist
import cradle

import time


basedir = os.path.abspath(os.path.dirname(__file__)).split('test')[0]


def test4():
    # 前置参数
    m_type = "Classification"
    dis_threshold = 16
    p = 0.001
    k = 5

    # 加载模型
    model_dir = basedir + "/network/models/lenet/"
    model_path = [model_dir + "lenet_mnist_tensorflow.h5", model_dir + "lenet_mnist_theano.h5"]
    print("load model")
    mo = load_model(model_path[0])
    mc = load_model(model_path[1])

    # 数据获取
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(-1, 28, 28, 1)

    # detect
    print('start time: ' + time.asctime(time.localtime(time.time())))
    (inconsistency, dis_list, inconsistency_count) = cradle.detect(mo, mc, x_test, y_test, dis_threshold, p, m_type, k)
    # localize
    if inconsistency:
        print("high rate of inconsistency")
        print(inconsistency_count, "/", len(x_test))
        in_layers = cradle.localize(mo, mc, x_test, dis_list)
        for in_layer in in_layers:
            print(in_layer)
        print('end time: ' + time.asctime(time.localtime(time.time())))
    else:
        print("low rate of inconsistency")
        print(inconsistency_count, "/", len(x_test))
        print('end time: ' + time.asctime(time.localtime(time.time())))


if __name__ == '__main__':
    test4()
