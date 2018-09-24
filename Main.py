# -*- coding: utf-8 -*-

import os

from MainModel import MainModel

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
config = {"min_revs":10,
          "emb_size":32,
          "learning_rate":0.001,
          "lr_decay":0.0,
          "batch_size":1024,
          "epochs":100,
          "c_loss":.5}
seed = 100

city = "Gijon"
modelGijon = MainModel(city=city, option=option, config=config, seed=seed)
modelGijon.train_step1(save=True, show_epoch_info=True)
#modelGijon.dev()
#modelGijon.test()

exit()

#ToDo: Cambio a nuevo modelo /generación de datos
#ToDo: Grid-Search
#ToDo: Test emb imagen con hilos == sin hilos
