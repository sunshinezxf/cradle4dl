# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

basedir = os.path.abspath(os.path.dirname(__file__))


class Net(object):

    def __init__(self, name="net", file_name="net.hdf5", batch_size=128, epochs=200, num_classes=10, train=True):
        self.name = name
        self.file_name = file_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = num_classes
        file_path = os.path.join(basedir, "models", self.name)
        if not train:
            try:
                self._model = load_model(os.path.join(basedir, "models", self.name, self.file_name))
            except (ValueError, IOError):
                print("Fail to load trained model: ", name, " from ", file_path)
        else:
            if not os.path.exists(os.path.join(basedir, "models")):
                os.mkdir(os.path.join(basedir, "models"))
            if not os.path.exists(os.path.join(basedir, "models",  self.name)):
                os.mkdir(os.path.join(basedir, "models", self.name))

    def summary(self):
        if self._model:
            self._model.summary()
        else:
            print("Model not loaded")

    def build_model(self):
        pass

    def get_model(self):
        return self._model

    def train(self, x_train, y_train):
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        print("train input shape: {}".format(x_train.shape))
        print("train label shape: {}".format(y_train.shape))
        model = self.build_model()
        model.summary()
        model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1, shuffle=True)
        model.save(os.path.join(basedir, 'models', self.name, self.file_name))
        self._model = model

    def predict(self, input_list):
        return self._model.predict(input_list)

    def predict_one(self, input):
        return self._model.predict([input])[0]

    def accuracy(self, x_test, y_test):
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        return self._model.evaluate(x_test, y_test, verbose=0)[1]
