# -*- coding: utf-8 -*-

import os

import numpy as np
from keras.models import load_model
from keras.datasets import cifar10

from utils import nnutil

import time


basedir = os.path.abspath(os.path.dirname(__file__))


def detect(model_o, model_c, input_list, label_list, distance_threshold, percent, model_type="Regression", k=5):
    """
    :param model_o: 第一个模型
    :param model_c: 第二个模型
    :param input_list: 输入列表
    :param label_list: 标签列表
    :param distance_threshold: 距离阈值，超过此阈值表示某两个结果“不一致”
    :param percent: 百分比阈值，当输入列表中导致了“不一致”的输入个数超过此百分比时，表示两个对比模型在此距离阈值下的“不一致”
    :param model_type: Classification或者Regression，字符串，表示模型的类型
    :param k: 当model_type为Classification时，用于top_k计算
    :return: (boolean, distance_list, in_single_count)
            boolean: 表示是否判定两个对比模型在此阈值下不一致
            distance_list: 对于input_list的每个输入在两个对比模型上的结果，计算distance获得的结果距离列表
            in_single_count: input_list造成“不一致“的输入个数
    """
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

    print(in_single_count / len(input_list))
    if in_single_count / len(input_list) >= percent:
        return True, distance_list, in_single_count
    else:
        return False, distance_list, in_single_count


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
    single_input = input_list[largest_index:largest_index+1]
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
        pre_max_distance = np.max([pre_max_distance, layer_distance])

    # 计算rate_of_change_list的三分位数，大于此数的layer需要高亮
    rate_of_change_list = np.asarray(rate_of_change_list)
    in_layers = []
    for i in range(int(len(rate_of_change_list) / 3)):
        index = np.argmax(rate_of_change_list)
        # print(layers[index])
        in_layers.append({
            "distance": layer_distance_list[index],
            "rate_of_change": rate_of_change_list[i],
            "layer": layers[index]
        })
        rate_of_change_list[index] = np.min(rate_of_change_list)
    return in_layers


# 数据预处理
def pre_process():
    return


if __name__ == '__main__':
    # 前置参数
    m_type = "Classification"
    dis_threshold = 16
    p = 0.01
    k = 5

    # 训练模型用
    # nnutil.train_lenet()

    # 加载模型
    model_dir = basedir + "/network/models/vgg16/"
    model_path = [model_dir + "vgg16_cifar10_tensorflow.h5", model_dir + "vgg16_cifar10_theano.h5"]
    print("load model")
    mo = load_model(model_path[0])
    mc = load_model(model_path[1])

    # 数据获取
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # x_test = x_test.reshape(-1, 32, 32, 1)

    # detect
    print('start time: ' + time.asctime(time.localtime(time.time())))
    (inconsistency, dis_list, inconsistency_count) = detect(mo, mc, x_test, y_test, dis_threshold, p, m_type, k)
    # localize
    if inconsistency:
        print("high rate of inconsistency")
        print(inconsistency_count, "/", len(x_test))
        in_layers = localize(mo, mc, x_test, dis_list)
        for in_layer in in_layers:
            print(in_layer)
        print('end time: ' + time.asctime(time.localtime(time.time())))
    else:
        print("low rate of inconsistency")
        print(inconsistency_count, "/", len(x_test))
        print('end time: ' + time.asctime(time.localtime(time.time())))
