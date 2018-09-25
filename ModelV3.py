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

class ModelV3(ModelClass):

    def __init__(self,city,option,config,seed = 2):

        modelName= "modelv3"
        ModelClass.__init__(self,city,option,config,seed = seed, name = modelName)

    def getModel(self):

        def normalized_mse_loss(y_pred, y_real):
            minMSE = self.MSE_DATA[0]
            maxMSE = self.MSE_DATA[1]
            avgMSE = self.MSE_DATA[2]

            return tf.losses.mean_squared_error(y_pred, y_real) / (avgMSE)

        # -------------------------------------------------------------------------------------------------------------------

        #Eliminar info de TF
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        #Utilizar solo memoria GPU necesaria
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        num_users = self.N_USR
        emb_users = self.CONFIG["emb_size"]

        num_restaurants = self.N_RST
        emb_restaurants = self.CONFIG["emb_size"]

        bin_out = 1
        img_out = self.V_IMG

        learning_rate = self.CONFIG["learning_rate"]
        lr_decay = self.CONFIG["lr_decay"]
        batch_size = self.CONFIG["batch_size"]
        epochs = self.CONFIG["epochs"]
        c_loss = self.CONFIG["c_loss"]

        # Model
        # -----------------------------------------------------------------------------------------------------------------------

        # first input model
        visible1 = Input(shape=(num_users,), name="input_user")
        d1 = Dense(emb_users, name="emb_user")(visible1)

        # second input model
        visible2 = Input(shape=(num_restaurants,), name="input_restaurant")
        d2 = Dense(emb_restaurants, name="emb_restaurant")(visible2)

        # merge input models
        concat = Concatenate()([d1, d2])
        dotprod = Dot(axes=1)([d1, d2])
        dotprod = Activation("sigmoid", name="dotprod")(dotprod)

        output_img = Dense(img_out, name="output_image", activation="relu")(concat)

        model = Model(inputs=[visible1, visible2], outputs=[dotprod, output_img])

        # decay: float >= 0. Learning rate decay over each update.
        adam = keras.optimizers.Adam(lr=learning_rate, decay=lr_decay)

        if (os.path.exists(self.MODEL_PATH)):
            self.printW("Cargando pesos de un modelo anterior...")
            model.load_weights(self.MODEL_PATH)

        # model.compile(optimizer=adam, loss=custom_loss_fn, loss_weights=[(1 - c_loss), c_loss])
        model.compile(optimizer=adam, loss=["binary_crossentropy", normalized_mse_loss],loss_weights=[(1 - c_loss), c_loss])

        # summarize layers
        #print(model.summary())
        plot_model(model, to_file=self.MODEL_NAME+'_net.png')

        return model

    def gridSearchV1(self, params):

        def gsStep(bs):
            self.MODEL.fit([usr_train, res_train], [bin_train, img_train], epochs=1, batch_size=bs, verbose=0)
            loss = self.MODEL.evaluate([usr_dev, res_dev], [bin_dev, img_dev], verbose=0)
            return loss

        # ---------------------------------------------------------------------------------------------------------------
        usr_train = to_categorical(self.TRAIN_V1.id_user, num_classes=self.N_USR)
        res_train = to_categorical(self.TRAIN_V1.id_restaurant, num_classes=self.N_RST)
        img_train = np.zeros((len(self.TRAIN_V1), self.V_IMG))
        bin_train = self.TRAIN_V1.like.values
        # ---------------------------------------------------------------------------------------------------------------
        usr_dev = to_categorical(self.DEV.id_user, num_classes=self.N_USR)
        res_dev = to_categorical(self.DEV.id_restaurant, num_classes=self.N_RST)
        img_dev = np.zeros((len(self.DEV), self.V_IMG))
        bin_dev = self.DEV.like.values
        # ---------------------------------------------------------------------------------------------------------------

        combs = []
        max_epochs = 1000
        last_n_epochs = 10
        dev_hist = []

        for lr in params['learning_rate']:
            for bs in params['batch_size']:
                for em in params['emb_size']:
                    combs.append([lr, bs, em])

        for c in combs:
            lr = c[0];
            bs = c[1]
            em = c[2]
            ep = 0

            # Reiniciar modelo e historial
            self.CONFIG['learning_rate'] = lr
            self.CONFIG['emb_size'] = em
            self.MODEL = self.getModel()

            dev_hist.clear()

            for e in range(max_epochs):
                ep += 1

                loss = gsStep(bs)
                dev_hist.append(loss[0])
                print(ep, lr, bs, em, loss[0],loss[1],loss[2])

                if (len(dev_hist) == last_n_epochs):
                    slope = self.getSlope(dev_hist);
                    dev_hist.pop(0)

                    if (slope > -1e-5):
                        print("[STOPPED AT "+str(slope)+"]")
                        break

            print("-" * 40)


    def train_step1(self, save=True, show_epoch_info=True):

        #self.MODEL_PATH = self.MODEL_PATH+"_step1"

        # Transformar los datos de TRAIN al formato adecuado
        oh_users = to_categorical(self.TRAIN_V1.id_user, num_classes=self.N_USR)
        oh_rests = to_categorical(self.TRAIN_V1.id_restaurant, num_classes=self.N_RST)

        y_likes = self.TRAIN_V1.like.values
        y_image = np.zeros((len(self.TRAIN_V1),self.V_IMG))

        # Definir un checkpoint para ir almacendando el modelo
        callbacks_list = []

        if(save):
            checkpoint = ModelCheckpoint(self.MODEL_PATH, verbose=0)
            callbacks_list.append(checkpoint)

        if(show_epoch_info):
            history = LossHistory(self.CONFIG)
            callbacks_list.append(history)

        self.MODEL.fit([oh_users, oh_rests], [y_likes, y_image], epochs=self.CONFIG['epochs'], batch_size=self.CONFIG['batch_size'],callbacks=callbacks_list, verbose=0)

        bin_pred, img_pred = self.MODEL.predict([oh_users, oh_rests], verbose=0)


        self.getF1(bin_pred,y_likes,title="TRAIN")

    def dev(self):

        # Transformar los datos de TRAIN al formato adecuado
        oh_users = to_categorical(self.DEV.id_user, num_classes=self.N_USR)
        oh_rests = to_categorical(self.DEV.id_restaurant, num_classes=self.N_RST)

        y_likes = self.DEV.like.values
        y_image = np.row_stack(self.DEV.vector.values)

        bin_pred, img_pred = self.MODEL.predict([oh_users, oh_rests], verbose=0)

        self.getF1(bin_pred, y_likes,title="DEV")

    def test(self):

        # Transformar los datos de TRAIN al formato adecuado
        oh_users = to_categorical(self.TEST.id_user, num_classes=self.N_USR)
        oh_rests = to_categorical(self.TEST.id_restaurant, num_classes=self.N_RST)

        y_likes = self.TEST.like.values
        y_image = np.row_stack(self.TEST.vector.values)

        bin_pred, img_pred = self.MODEL.predict([oh_users, oh_rests], verbose=0)

        self.getF1(bin_pred, y_likes,title="TEST")
