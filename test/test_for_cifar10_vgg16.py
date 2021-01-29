
import os
from keras.models import load_model
from keras.datasets import cifar10
import cradle


basedir = os.path.abspath(os.path.dirname(__file__)).split('test')[0]


def test4():
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
    (inconsistency, dis_list, inconsistency_count) = cradle.detect(mo, mc, x_test, y_test, dis_threshold, p, m_type, k)
    # localize
    if inconsistency:
        print("认为结果不符合预期")
        print(inconsistency_count, "/", len(x_test))
        in_layers = cradle.localize(mo, mc, x_test, dis_list)
        for in_layer in in_layers:
            print(in_layers)
    else:
        print("认为结果符合预期")
        print(inconsistency_count, "/", len(x_test))


if __name__ == '__main__':
    test4()
    # print(basedir)
