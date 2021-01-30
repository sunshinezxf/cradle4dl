from keras.applications import nasnet


def nasnet_download():
    print("开始下载NASNetLarge")
    nasnet.NASNetLarge()
    print("完成下载NASNetLarge")


if __name__ == "__main__":
    nasnet_download()
