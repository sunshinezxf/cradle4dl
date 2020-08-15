# -*- coding: utf-8 -*-

import os
import tensorflow as tf

from keras.models import load_model
from tensorflow import keras

from utils import nnutil

basedir = os.path.abspath(os.path.dirname(__file__))


# 输入两个模型实例及验证数据，判断结果是否正确
def detect(model_o, model_c):
    return


# 输入两个模型实例及验证数据，判断结果是否正确
def detect(model_o, model_c, ins_input):
    output_o = model_o.predict(ins_input)
    output_c = model_c.predict(ins_input)
    return


# 定位问题发生的位置
def localize(model_o, model_c):
    return


# 数据预处理
def pre_process():
    return


if __name__ == "__main__":
    nnutil.train_lenet()
    # print(basedir)
    # model_path = basedir + "/models/"
    # model_path = [model_path + "lenet.h5", model_path + "lenet.h5"]
    # print("load model")
    # mo = load_model(model_path[0])
    # mc = load_model(model_path[1])
    # val = ""
    # detect(mo, mc, val)
