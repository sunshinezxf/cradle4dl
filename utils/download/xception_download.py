from tensorflow.keras.applications import xception


def xception_download():
    print("开始下载Xception")
    xception.Xception()
    print("完成下载Xception")


if __name__ == "__main__":
    xception_download()
