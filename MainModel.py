# -*- coding: utf-8 -*-
import os

import keras
import pandas as pd
import numpy as np
import sklearn.model_selection
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

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

########################################################################################################################

class LossHistory(keras.callbacks.Callback):

    FIRST_TIME = True

    def __init__(self,config):
        self.FIRST_TIME = True
        self.CONFIG = config

        self.HEADER = list(config.keys())
        self.VALUES = list((str(x).replace(".",",") for x in list(config.values())))


    def on_epoch_end(self, epoch, logs=None):

        if(self.FIRST_TIME):
            self.HEADER.extend(list(logs.keys()))
            self.FIRST_TIME=False
            print("\t".join(self.HEADER))

        line=[]
        line.extend(self.VALUES)
        line.extend((str(x).replace(".",",") for x in list(logs.values())))

        print("\t".join(line))

########################################################################################################################

class MainModel():

    def __init__(self,city,option,config,seed = 2):
        self.CITY = city
        self.OPTION = option
        self.PATH = "/mnt/hdd/pperez/TripAdvisor/" + self.CITY.lower() + "_data/"
        self.IMG_PATH = self.PATH + "images/"
        self.SEED = seed

        self.printB("Obteniendo datos...")

        train, dev, test, n_rest, n_usr, v_img, mse_data = self.__getData()

        self.TRAIN = train
        self.DEV = dev
        self.TEST = test

        self.N_RST = n_rest
        self.N_USR = n_usr
        self.V_IMG = v_img
        self.MSE_DATA = mse_data

        self.printB("Creando modelo...")

        self.CONFIG = config
        self.MODEL = self.__getModel()

    def __getModel(self):

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

        model_path = "models/model_"+self.CITY.lower()+"_option"+str(self.OPTION)
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

        if (os.path.exists(model_path)):
            self.printW("Cargando pesos de un modelo anterior...")
            model.load_weights(model_path)

        # model.compile(optimizer=adam, loss=custom_loss_fn, loss_weights=[(1 - c_loss), c_loss])
        model.compile(optimizer=adam, loss=["binary_crossentropy", normalized_mse_loss],
                      loss_weights=[(1 - c_loss), c_loss])

        # summarize layers
        # print(model.summary())
        #plot_model(model, to_file='the_net.png')

        return model

    def __getData(self):

        USR_TMP = pd.DataFrame(columns=["real_id","id_user"])
        REST_TMP = pd.DataFrame(columns=["real_id","id_restaurant"])

        IMG = pd.read_pickle(self.PATH + "img-option"+str(self.OPTION)+".pkl")

        RVW = pd.read_pickle(self.PATH + "reviews.pkl")
        RVW["num_images"] = RVW.images.apply(lambda x: len(x))
        RVW["like"] = RVW.rating.apply(lambda x: 1 if x > 30 else 0)

        RVW = RVW.loc[RVW.num_images > 0]

        #Obtener tabla real_id -> id para usuarios
        USR_TMP.real_id = RVW.sort_values("userId").userId.unique()
        USR_TMP.id_user = range(0,len(USR_TMP))

        #Obtener tabla real_id -> id para restaurantes
        REST_TMP.real_id = RVW.sort_values("restaurantId").restaurantId.unique()
        REST_TMP.id_restaurant = range(0,len(REST_TMP))

        #Mezclar datos
        RET = RVW.merge(USR_TMP, left_on='userId', right_on='real_id', how='inner')
        RET = RET.merge(REST_TMP, left_on='restaurantId', right_on='real_id', how='inner')

        RET = RET[['date', 'images', 'index', 'language', 'rating', 'restaurantId','reviewId', 'text', 'title', 'url', 'userId', 'num_images', 'real_id_x', 'id_user', 'real_id_y', 'id_restaurant', 'like']]

        # -------------------------------------------------------------------------------------------------------------------
        # Forzar usuarios de una review con una imagen a TRAIN

        ids = []

        for i, g in RET.groupby("id_user"):
            if(len(g)==1 and g.num_images.values[0]==1):
                ids.append(i)

        self.printG("Moviendo a TRAIN "+str(len(ids))+" usuarios con una review de una imagen...")

        RET_ONEONE = RET.loc[RET.id_user.isin(ids)]
        RET = RET.loc[~RET.id_user.isin(ids)]

        # -------------------------------------------------------------------------------------------------------------------
        # TRAIN/DEV/TEST

        TRAIN_PROP = 0.6 - (len(RET_ONEONE)/len(RVW))
        DEV_PROP = (1-TRAIN_PROP)/2
        TEST_PROP = (1-TRAIN_PROP)/2

        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(RET.iloc[:, :-1], RET.iloc[:, -1],test_size=TEST_PROP, random_state=self.SEED)
        X_train, X_dev, Y_train, Y_dev = sklearn.model_selection.train_test_split(X_train, Y_train, test_size=DEV_PROP,random_state=self.SEED)

        TRAIN = pd.DataFrame(X_train)
        TRAIN['like'] = Y_train
        TRAIN = TRAIN.append(RET_ONEONE,ignore_index=True).reset_index()

        DEV = pd.DataFrame(X_dev)
        DEV['like'] = Y_dev

        TEST = pd.DataFrame(X_test)
        TEST['like'] = Y_test

        # -------------------------------------------------------------------------------------------------------------------
        # OVERSAMPLING (SOLO EN TRAIN)

        TRAIN_ONE = TRAIN.loc[TRAIN.like==1]
        TRAIN_ZRO = TRAIN.loc[TRAIN.like==0]

        TRAIN = TRAIN_ONE.append(TRAIN_ZRO,ignore_index=True)
        TRAIN_ZRO_SMPLE = TRAIN_ZRO.sample(len(TRAIN_ONE)-len(TRAIN_ZRO), replace=True, random_state=self.SEED)
        TRAIN = TRAIN.append(TRAIN_ZRO_SMPLE,ignore_index=True)

        self.printG("\t· TRAIN: "+str(len(TRAIN)))
        self.printG("\t· DEV: "+str(len(DEV)))
        self.printG("\t· TEST: "+str(len(TEST)))

        #-------------------------------------------------------------------------------------------------------------------

        TRAIN = IMG.merge(TRAIN, left_on='review', right_on='reviewId', how='inner')
        DEV = IMG.merge(DEV, left_on='review', right_on='reviewId', how='inner')
        TEST = IMG.merge(TEST, left_on='review', right_on='reviewId', how='inner')

        TRAIN = TRAIN.drop(columns=['restaurantId','userId','url','text','title','date','real_id_x','real_id_y','images','num_images','index','rating','language','review'])
        DEV = DEV.drop(columns=['restaurantId','userId','url','text','title','date','real_id_x','real_id_y','images','num_images','index','rating','language','review'])
        TEST = TEST.drop(columns=['restaurantId','userId','url','text','title','date','real_id_x','real_id_y','images','num_images','index','rating','language','review'])

        self.printW("Las reviews salen repetidas en función del número de imagenes")

        IMG_2 = np.row_stack(IMG.vector.values)
        M_IMG = np.mean(IMG_2,axis=0)

        IMG_2 = np.apply_along_axis(lambda x:np.power(x-M_IMG,2),1,IMG_2)
        IMG_2 = np.apply_along_axis(lambda x:np.mean(x),1,IMG_2)
        MeanMSE = np.apply_along_axis(lambda x:np.mean(x),0,IMG_2)
        MaxMSE = np.apply_along_axis(lambda x:np.max(x),0,IMG_2)
        MinMSE = np.apply_along_axis(lambda x:np.min(x),0,IMG_2)

        return(TRAIN,DEV,TEST,len(REST_TMP),len(USR_TMP),len(IMG.iloc[0].vector), [MinMSE,MaxMSE,MeanMSE])

    def __getF1(self,pred,real):
        return 0

    def train(self, save=True, show_epoch_info=True):

        # Transformar los datos de TRAIN al formato adecuado
        oh_users = to_categorical(self.TRAIN.id_user, num_classes=self.N_USR)
        oh_rests = to_categorical(self.TRAIN.id_restaurant, num_classes=self.N_RST)

        y_likes = self.TRAIN.like.values
        y_image = np.row_stack(self.TRAIN.vector.values)

        # Definir un checkpoint para ir almacendando el modelo
        callbacks_list = []

        if(save):
            checkpoint = ModelCheckpoint(config['model_path'], verbose=0)
            callbacks_list.append(checkpoint)

        if(show_epoch_info):
            history = LossHistory(self.CONFIG)
            callbacks_list.append(history)

        self.MODEL.fit([oh_users, oh_rests], [y_likes, y_image], epochs=self.CONFIG['epochs'], batch_size=self.CONFIG['batch_size'],callbacks=callbacks_list, verbose=0)

        bin_pred, img_pred = model.predict([oh_users, oh_rests], verbose=0)

        self.__getF1(pred,y_likes)

    def printW(self,text):
        print(bcolors.WARNING+str("[AVISO] ")+str(text)+bcolors.ENDC)

    def printG(self,text):
        print(bcolors.OKGREEN+str("[INFO] ")+str(text)+bcolors.ENDC)

    def printB(self,text):
        print(bcolors.OKBLUE+str(text)+bcolors.ENDC)