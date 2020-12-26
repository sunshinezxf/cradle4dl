# -*- coding: utf-8 -*-

import os

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

import numpy as np

from utils import nnutil

basedir = os.path.abspath(os.path.dirname(__file__))


def detect(model_o, model_c, input_list, label_list, distance_threshold, percent, model_type="Regression", k=5):
    input_list = input_list.reshape(-1, 28, 28, 1)
    try:
        output_list_o = model_o.predict(input_list)
        output_list_c = model_c.predict(input_list)
    except RuntimeError:
        print("input prediction trigger a bug which interrupt the prediction")
    distance_list = []
    in_single_count = 0
    for i in range(len(input_list)):
        distance = distance_calculate(output_list_o[i], output_list_c[i], label_list[i], model_type, k)
        if distance >= distance_threshold:
            in_single_count += 1
        distance_list.append(distance)

    if in_single_count / len(input_list) >= percent:
        return True, distance_list
    else:
        return False, distance_list


# 计算两个单个输出值的distance
def distance_calculate(output_o, output_c, label, model_type, k):
    if model_type == "Classification":
        distance = nnutil.class_based_distance(output_o, output_c, label, k)
        return distance
    else:
        distance = nnutil.mad_based_distance(output_o, output_c, label)
        return distance


# 定位问题发生的位置
def localize(model_o, model_c, input_list, distance_list):
    distance_list = np.asarray(distance_list)
    largest_index = np.argmax(distance_list)
    single_input = [input_list[largest_index]]
    layers_output_o = nnutil.layers_output(model_o, single_input)
    layers_output_c = nnutil.layers_output(model_c, single_input)
    layers = nnutil.extract_model_layer(model_o)

    pre_max_distance = 0
    layer_distance_list = []
    rate_of_change_list = []
    for i in range(len(layers_output_o)):
        layer_distance = nnutil.mean_absolute_deviation(layers_output_o[i][0], layers_output_c[i][0])
        layer_distance_list.append(layer_distance)
        rate_of_change_list.append(nnutil.rate_of_change(layer_distance, pre_max_distance))
        pre_max_distance = np.max(pre_max_distance, layer_distance)

    # 计算rate_of_change_list的三分位数，大于此数的layer需要高亮
    rate_of_change_list = np.asarray(rate_of_change_list)
    in_layers = []
    for i in range(len(rate_of_change_list) / 3):
        index = np.argmax(rate_of_change_list)
        print(layers[index])
        in_layers.append({
            "layer": layers[index],
            "distance": layer_distance_list[index],
            "rate_of_change": rate_of_change_list[i]
        })
    return in_layers


# 数据预处理
def pre_process():
    return


if __name__ == "__main__":
    # 前置参数
    m_type = "Classification"
    dis_threshold = 0
    p = 0.1

    # 训练模型用
    # nnutil.train_lenet()
    model_dir = basedir + "/network/models/lenet/"
    model_path = [model_dir + "lenet_mnist_1.h5", model_dir + "lenet_mnist_2.h5"]
    print("load model")
    mo = load_model(model_path[0])
    mc = load_model(model_path[1])

    # 数据获取
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(-1, 28, 28, 1)

    (inconsistency, dis_list) = detect(mo, mc, x_test, y_test, dis_threshold, p, m_type)
    if inconsistency:
        print("认为结果不符合预期")
        localize(mo, mc, x_test, dis_list)
    else:
        print("认为结果符合预期")

