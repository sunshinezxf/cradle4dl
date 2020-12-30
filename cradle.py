# -*- coding: utf-8 -*-

import os

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

import keras

import numpy as np

from utils import nnutil

basedir = os.path.abspath(os.path.dirname(__file__))


# 输入两个模型实例及验证数据，判断结果是否正确
def detect(model_o, model_c, input_list):
    for i in range(len(input_list)):
        _detect_single(model_o, model_c, input_list[i])
    return


# 输入两个模型实例及验证数据，判断结果是否正确
def _detect_single(model_o, model_c, ins_input):
    ins_input = ins_input.reshape(-1, 28, 28, 1)
    try:
        output_o = model_o.predict(ins_input)[0]
        output_c = model_c.predict(ins_input)[0]
        is_equal = _output_compare(output_o, output_c)
        if is_equal is not True:
            # 需要进行hidden state的分析
            print("further analysis")
    except RuntimeError:
        print("input prediction trigger a bug which interrupt the prediction")
    return


# 输出结果比对
def _output_compare(o_1, o_2):
    is_equal = nnutil.top_equal(o_1, o_2)
    if is_equal is not True:
        print("input_o: %s" % str(o_1))
        print("input_c: %s" % str(o_2))
    return is_equal


# 定位问题发生的位置
def localize(model_o, model_c):
    return


# 数据预处理
def pre_process():
    return


if __name__ == "__main__":
    # 训练模型用
    nnutil.train_lenet()
    # model_dir = basedir + "/network/models/lenet/"
    # model_path = [model_dir + "lenet_mnist_1.h5", model_dir + "lenet_mnist_2.h5"]
    # print("load model")
    # mo = load_model(model_path[0])
    # mc = load_model(model_path[1])
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_test = x_test.reshape(-1, 28, 28, 1)
    # y_test = keras.utils.to_categorical(y_test, 10)
    # print(mo.evaluate(x_test, y_test))
    # print(mc.evaluate(x_test, y_test))
    # detect(model_o=mo, model_c=mc, input_list=x_test)

