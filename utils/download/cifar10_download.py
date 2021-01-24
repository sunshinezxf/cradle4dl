
from keras.datasets import cifar10


def download_cifar10():
    print("开始下载数据集cifar10")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print("完成下载数据集cifar10")
    print(x_train.shape)


if __name__ == "__main__":
    download_cifar10()
