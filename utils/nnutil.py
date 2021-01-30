# -*- coding: utf-8 -*-
import math

from keras import backend

import numpy as np

import json


def top_equal(o_1, o_2):
    o_1 = np.asarray(o_1)
    o_2 = np.asarray(o_2)
    o_1_index = o_1.argmax()
    o_2_index = o_2.argmax()
    if o_1_index == o_2_index:
        return True
    return False


# 当第k位有两个相同value的index时，可能出现问题
# 由于碰撞概率极小，此问题可忽略
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


# i的改变并没有改变np.argmax(output)的值，添加else后正常
# 用于计算单个输出的类间距离
def _class_distance(output, label, k):
    for i in range(1, k + 1):
        index = np.argmax(output)
        if index == label:
            return np.power(2, k - i)
        else:
            output[index] = np.min(output)
    return 0


# 用于计算两个输出之间的类间距离
def class_based_distance(o_1, o_2, oracle, k):
    diff_1 = _class_distance(o_1, oracle, k)
    diff_2 = _class_distance(o_2, oracle, k)
    return np.abs(diff_1 - diff_2)


# MAD
def mean_absolute_deviation(y, o):
    return np.mean(np.abs(y-o))


# 用于计算两个输出之间的平均绝对偏差(MAD)
def mad_based_distance(o_1, o_2, oracle):
    diff_1 = mean_absolute_deviation(o_1, oracle)
    diff_2 = mean_absolute_deviation(o_2, oracle)
    if diff_1 == diff_2 and diff_1 == 0:
        return 0
    else:
        return np.abs(diff_1 - diff_2) / (diff_1 + diff_2)


# 用于分离一个model的layers
# 返回一个字典list，每个字典中存储一个layer的数据
def extract_model_layer(model):
    layers_json = json.loads(model.to_json())
    if type(layers_json['config']).__name__ == 'list':
        return layers_json['config']
    else:
        return layers_json['config']['layers']


# 计算model每一层的输出，并返回list
def layers_output(model, first_layer_input):
    output_list = []
    for layer in model.layers:
        target_func = backend.function([model.layers[0].input], [layer.output])
        # print("layer input: ", first_layer_input.shape)
        output_list.append(target_func([first_layer_input]))
        # print("layer output: ", output_list[-1].shape)

    return output_list


def layers_output_new(model, first_layer_input):
    output_list = []
    target_func = backend.function([model.layers[1].input, model.layers[1].output])
    # output_list.append(target_func(first_layer_input))
    # for i in range(1, len(model.layers)):
    #     target_func = backend.function([model.layers[i].input, model.layers[i].output])
    #     print("layer input: ", output_list[i-1].shape)
    #     output_list.append(target_func(output_list[i-1]))
    #     print("layer output: ", output_list[-1].shape)

    return output_list


def rate_of_change(current_distance, pre):
    return (current_distance - pre) / (pre + math.pow(10, -7))
