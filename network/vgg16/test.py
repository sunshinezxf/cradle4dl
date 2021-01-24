
from keras.datasets import cifar10


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(x_train.shape)
