# -*- coding: utf-8 -*-

import os

from ModelV3 import ModelV3
from ModelV2 import ModelV2

import pandas as pd
import numpy as np
import sklearn.model_selection

import keras
import tensorflow as tf
from keras import backend as K
from keras import losses
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input,Dense,Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Concatenate, Dot
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session

########################################################################################################################

option = 2
config = {"min_revs":5,
          "emb_size":32,
          "learning_rate":0.0001,
          "lr_decay":0.0,
          "batch_size":1024,
          "epochs":50,
          "c_loss":.5}
seed = 100

params = {
    "learning_rate": [0.000001, 0.00001, 0.0001, 0.001],
    "emb_size": [32, 64, 128],
    "batch_size": [512, 1024, 2048],
}

city = "Barcelona"

'''
modelv2 = ModelV2(city=city, option=option, config=config, seed=seed)
modelv2.gridSearchV1(params)
'''
modelv3 = ModelV3(city=city, option=option, config=config, seed=seed)
modelv3.gridSearchV1(params)


#ToDo: Preguntar por el número mínimo de valoraciones
#ToDo: Grid-Search
#ToDo: Test emb imagen con hilos == sin hilos


