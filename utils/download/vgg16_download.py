from tensorflow.keras.applications import vgg16


def vgg16_download():
    print("开始下载vgg16")
    vgg16.VGG16()
    print("完成下载vgg16")


if __name__ == "__main__":
    vgg16_download()
