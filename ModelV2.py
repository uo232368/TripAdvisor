# -*- coding: utf-8 -*-
from ModelClass import ModelClass
from ModelClass import LossHistory

import os

import keras
import pandas as pd
import numpy as np
import sklearn.model_selection
import tensorflow as tf
import random

from scipy.stats import linregress

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

class ModelV2(ModelClass):

    def __init__(self,city,option,config,seed = 2):
        modelName = "modelv2"
        ModelClass.__init__(self,city,option,config,seed = seed, name=modelName)

    def getModel(self):

        #Eliminar info de TF
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        #Utilizar solo memoria GPU necesaria
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        num_users = self.N_USR
        emb_users = self.CONFIG["emb_size"]

        img_size = self.V_IMG

        num_restaurants = self.N_RST
        emb_restaurants = self.CONFIG["emb_size"]

        in_size = num_users+img_size+num_restaurants

        #print(num_users,img_size,num_restaurants)

        bin_out = 1

        learning_rate = self.CONFIG["learning_rate"]
        lr_decay = self.CONFIG["lr_decay"]
        batch_size = self.CONFIG["batch_size"]
        epochs = self.CONFIG["epochs"]
        c_loss = self.CONFIG["c_loss"]

        # Model
        # -----------------------------------------------------------------------------------------------------------------------

        first_hidden_layer = 4096

        # first input model
        in_user = Input(shape=(num_users,), name="input_user")

        # second input model
        in_image = Input(shape=(img_size,), name="input_image")

        # second input model
        in_rest = Input(shape=(num_restaurants,), name="input_restaurant")

        # merge input models
        concat = Concatenate()([in_user, in_image,in_rest])
        d1 = Dense(first_hidden_layer,activation='relu')(concat)
        d2 = Dense(first_hidden_layer//2,activation='relu')(d1)
        d3 = Dense(first_hidden_layer//4,activation='relu')(d2)
        d4 = Dense(first_hidden_layer//8,activation='relu')(d3)
        d5 = Dense(first_hidden_layer//16,activation='relu')(d4)
        out = Dense(1,activation='sigmoid', name="out_layer")(d5)

        model = Model(inputs=[in_user, in_image,in_rest], outputs=[out])

        # decay: float >= 0. Learning rate decay over each update.
        adam = keras.optimizers.Adam(lr=learning_rate, decay=lr_decay)

        if (os.path.exists(self.MODEL_PATH)):
            self.printW("Cargando pesos de un modelo anterior...")
            model.load_weights(self.MODEL_PATH)

        # model.compile(optimizer=adam, loss=custom_loss_fn, loss_weights=[(1 - c_loss), c_loss])
        model.compile(optimizer=adam, loss=["binary_crossentropy"])

        # summarize layers
        #print(model.summary())
        plot_model(model, to_file=self.MODEL_NAME+'_net.png')

        return model

    def train_v1(self,save=True, show_epoch_info=True):

        verbosity = 2

        # Transformar los datos de TRAIN al formato adecuado
        oh_users = to_categorical(self.TRAIN_V1.id_user, num_classes=self.N_USR)
        y_image = np.zeros((len(self.TRAIN_V1),self.V_IMG))
        oh_rests = to_categorical(self.TRAIN_V1.id_restaurant, num_classes=self.N_RST)

        y_likes = self.TRAIN_V1.like.values

        # y_image = np.row_stack(self.TRAIN_V2.vector.values)

        # Definir un checkpoint para ir almacendando el modelo
        callbacks_list = []

        if (save):
            checkpoint = ModelCheckpoint(self.MODEL_PATH, verbose=0,period=5)
            callbacks_list.append(checkpoint)

        if (show_epoch_info):
            history = LossHistory(self.CONFIG)
            callbacks_list.append(history)
            verbosity = 0

        self.MODEL.fit([oh_users, y_image, oh_rests], [y_likes], epochs=self.CONFIG['epochs'], batch_size=self.CONFIG['batch_size'], callbacks=callbacks_list, verbose=verbosity)

        bin_pred = self.MODEL.predict([oh_users, y_image, oh_rests], verbose=0)

        self.getF1(bin_pred,y_likes,title="TRAIN", verbose=True)

    def gridSearchV1(self, params):

        def gsStep(lr, bs):
            K.set_value(self.MODEL.optimizer.lr, lr)
            self.MODEL.fit([usr_train, img_train, res_train], [out_train], epochs=1, batch_size=bs, verbose=0)
            loss = self.MODEL.evaluate([usr_dev, img_dev, res_dev], [out_dev],verbose=0)
            return loss

        #---------------------------------------------------------------------------------------------------------------
        usr_train = to_categorical(self.TRAIN_V1.id_user, num_classes=self.N_USR)
        img_train = np.zeros((len(self.TRAIN_V1), self.V_IMG))
        res_train = to_categorical(self.TRAIN_V1.id_restaurant, num_classes=self.N_RST)
        out_train = self.TRAIN_V1.like.values
        #---------------------------------------------------------------------------------------------------------------
        usr_dev = to_categorical(self.DEV.id_user, num_classes=self.N_USR)
        img_dev = np.zeros((len(self.DEV), self.V_IMG))
        res_dev = to_categorical(self.DEV.id_restaurant, num_classes=self.N_RST)
        out_dev = self.DEV.like.values
        #---------------------------------------------------------------------------------------------------------------

        combs = []
        max_epochs = 1000
        last_n_epochs = 10
        dev_hist = []

        for lr in params['learning_rate']:
            for bs in params['batch_size']:
                combs.append([lr,bs])

        for c in combs:
            lr = c[0]; bs = c[1]
            ep = 0

            for e in range(max_epochs):
                ep +=1

                loss = gsStep(lr,bs)
                dev_hist.append(loss)

                print(ep,lr,bs,loss)

                if(len(dev_hist)==last_n_epochs):
                    slope = self.getSlope(dev_hist);
                    dev_hist.pop(0)

                    if(slope> -1e-5 ):
                        break
                        self.printW("STOP:"+str(slope))

            print("-"*40)

            #Reiniciar modelo e historial
            self.MODEL = self.getModel()
            dev_hist.clear()

        return False

    def dev(self):

        # Transformar los datos de DEV al formato adecuado
        oh_users = to_categorical(self.DEV.id_user, num_classes=self.N_USR)
        y_image = np.zeros((len(self.DEV), self.V_IMG))
        oh_rests = to_categorical(self.DEV.id_restaurant, num_classes=self.N_RST)

        y_likes = self.DEV.like.values


        bin_pred = self.MODEL.predict([oh_users,y_image, oh_rests], verbose=0)

        self.getF1(bin_pred, y_likes,title="DEV")



