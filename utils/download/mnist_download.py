from keras.datasets import mnist


def mnist_download():
    print("开始下载mnist")
    mnist.load_data()
    print("完成下载mnist")


if __name__ == "__main__":
    mnist_download()
