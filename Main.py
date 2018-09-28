# -*- coding: utf-8 -*-

import os
import argparse

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

parser = argparse.ArgumentParser()
parser.add_argument('-m', type=int)
args = parser.parse_args()

model= 3 if args.m == None else args.m

option = 2
config = {"min_revs":5,
          "emb_size":32,
          "learning_rate":0.0001,
          "lr_decay":0.0,
          "batch_size":1024,
          "epochs":50,
          "c_loss":.5,
          "gs_max_slope":-1e-8}
seed = 100

city = "Gijon"

if (model == 2):

    params = {
        "learning_rate": [1e-07,1e-05,1e-03, 1e-01],
        "batch_size": [512],
    }

    modelv2 = ModelV2(city=city, option=option, config=config, seed=seed)
    modelv2.gridSearchV1(params, max_epochs=5000)

if (model == 3):

    params = {
        "learning_rate": [1e-07,1e-05,1e-03, 1e-01],
        "emb_size": [512,1024,2048],
        "batch_size": [512],
    }

    modelv3 = ModelV3(city=city, option=option, config=config, seed=seed)
    print(modelv3.N_USR,modelv3.N_RST)
    modelv3.gridSearchV1(params, max_epochs=5000)

#ToDo: Revisar creación de datos
#ToDo: Grid-Search
#ToDo: Test emb imagen con hilos == sin hilos


