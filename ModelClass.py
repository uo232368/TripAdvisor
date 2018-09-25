# -*- coding: utf-8 -*-
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
from scipy.stats import linregress

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

class ModelClass():

    def __init__(self,city,option,config,name,seed = 2 ):

        self.CITY = city
        self.OPTION = option
        self.PATH = "/mnt/hdd/pperez/TripAdvisor/" + self.CITY.lower() + "_data/"
        self.IMG_PATH = self.PATH + "images/"
        self.SEED = seed
        self.CONFIG = config
        self.MODEL_NAME = name

        self.printB("Obteniendo datos...")

        train1, train2, dev, test, n_rest, n_usr, v_img, mse_data = self.getData()

        self.TRAIN_V1 = train1
        self.TRAIN_V2 = train2
        self.DEV = dev
        self.TEST = test

        self.N_RST = n_rest
        self.N_USR = n_usr
        self.V_IMG = v_img
        self.MSE_DATA = mse_data

        self.printB("Creando modelo...")

        self.MODEL_PATH = "models/"+self.MODEL_NAME+"_" + self.CITY.lower() + "_option" + str(self.OPTION)
        self.MODEL = self.getModel()

    def getModel(self):
        self.printW("FN SIN IMPLEMENTAR")
        exit()

    def getData(self):

        USR_TMP = pd.DataFrame(columns=["real_id", "id_user"])
        REST_TMP = pd.DataFrame(columns=["real_id", "id_restaurant"])

        IMG = pd.read_pickle(self.PATH + "img-option" + str(self.OPTION) + ".pkl")
        RVW = pd.read_pickle(self.PATH + "reviews.pkl")

        IMG['review'] = IMG.review.astype(int)
        RVW["reviewId"] = RVW.reviewId.astype(int)

        RVW["num_images"] = RVW.images.apply(lambda x: len(x))
        RVW["like"] = RVW.rating.apply(lambda x: 1 if x > 30 else 0)
        RVW = RVW.loc[(RVW.userId != "")]

        # Eliminar usuarios con menos de min_revs
        # ---------------------------------------------------------------------------------------------------------------
        old_len = len(RVW)

        USR_LST = RVW.groupby("userId", as_index=False).count()
        USR_LST = USR_LST.loc[(USR_LST.like >= self.CONFIG['min_revs']), "userId"].values
        RVW = RVW.loc[RVW.userId.isin(USR_LST)]

        self.printW(
            "Eliminado usuarios con menos de " + str(self.CONFIG['min_revs']) + " valoraciones quedan un " + str(
                (len(RVW) / old_len) * 100) + " % del total de reviews.")

        # Obtener ID para ONE-HOT de usuarios y restaurantes
        # ---------------------------------------------------------------------------------------------------------------

        # Obtener tabla real_id -> id para usuarios
        USR_TMP.real_id = RVW.sort_values("userId").userId.unique()
        USR_TMP.id_user = range(0, len(USR_TMP))

        # Obtener tabla real_id -> id para restaurantes
        REST_TMP.real_id = RVW.sort_values("restaurantId").restaurantId.unique()
        REST_TMP.id_restaurant = range(0, len(REST_TMP))

        # Mezclar datos
        RET = RVW.merge(USR_TMP, left_on='userId', right_on='real_id', how='inner')
        RET = RET.merge(REST_TMP, left_on='restaurantId', right_on='real_id', how='inner')

        RVW = RET[['date', 'images', 'index', 'language', 'rating', 'restaurantId', 'reviewId', 'text', 'title', 'url',
                   'userId', 'num_images', 'real_id_x', 'id_user', 'real_id_y', 'id_restaurant', 'like']]

        # Separar en reviews con y sin imágen
        # ---------------------------------------------------------------------------------------------------------------
        RVW_IM = RVW.loc[RVW.num_images > 0]
        RVW_NIM = RVW.loc[RVW.num_images == 0]

        self.printG("Reviews con imágen: " + str(len(RVW_IM)))
        self.printG("Reviews sin imágen: " + str(len(RVW_NIM)))

        # Agrupar por usuario-review
        # ---------------------------------------------------------------------------------------------------------------
        GRP_RVW_IM = RVW_IM.groupby(["userId", "restaurantId"])
        GRP_RVW_NIM = RVW_NIM.groupby(["userId", "restaurantId"])

        random.seed(self.SEED)

        TRAIN = []
        DEV = []
        TEST = []

        self.printG("Separando en TRAIN/DEV/TEST valoraciones con imagen...")

        for i, g in GRP_RVW_IM:
            rnd = random.random()
            if (rnd < 0.05):
                TEST.extend(g.reviewId.values)
            elif (rnd >= 0.05 and rnd < 0.1):
                DEV.extend(g.reviewId.values)
            else:
                TRAIN.extend(g.reviewId.values)

        self.printG("Separando en TRAIN/DEV/TEST valoraciones sin imagen...")
        for i, g in GRP_RVW_NIM:
            rnd = random.random()
            if (rnd < 0.05):
                TEST.extend(g.reviewId.values)
            elif (rnd >= 0.05 and rnd < 0.1):
                DEV.extend(g.reviewId.values)
            else:
                TRAIN.extend(g.reviewId.values)

        # Obtener conjuntos de TRAIN/DEV/TEST
        # ---------------------------------------------------------------------------------------------------------------

        self.printG("Generando conjuntos finales...")

        TRAIN_v1 = RVW.loc[RVW.reviewId.isin(TRAIN)]
        TRAIN_v2 = TRAIN_v1.loc[TRAIN_v1.num_images > 0]
        DEV = RVW.loc[RVW.reviewId.isin(DEV)]
        TEST = RVW.loc[RVW.reviewId.isin(TEST)]

        # -------------------------------------------------------------------------------------------------------------------
        # OVERSAMPLING (SOLO EN TRAIN)

        self.printG("Oversampling en TRAIN_V1...")

        TRAIN_ONE = TRAIN_v1.loc[TRAIN_v1.like == 1]
        TRAIN_ZRO = TRAIN_v1.loc[TRAIN_v1.like == 0]

        TRAIN_v1 = TRAIN_ONE.append(TRAIN_ZRO, ignore_index=True)
        TRAIN_ZRO_SMPLE = TRAIN_ZRO.sample(len(TRAIN_ONE) - len(TRAIN_ZRO), replace=True, random_state=self.SEED)
        TRAIN_v1 = TRAIN_v1.append(TRAIN_ZRO_SMPLE, ignore_index=True)

        self.printG("Oversampling en TRAIN_V2...")

        TRAIN_ONE = TRAIN_v2.loc[TRAIN_v2.like == 1]
        TRAIN_ZRO = TRAIN_v2.loc[TRAIN_v2.like == 0]

        TRAIN_v2 = TRAIN_ONE.append(TRAIN_ZRO, ignore_index=True)
        TRAIN_ZRO_SMPLE = TRAIN_ZRO.sample(len(TRAIN_ONE) - len(TRAIN_ZRO), replace=True, random_state=self.SEED)
        TRAIN_v2 = TRAIN_v2.append(TRAIN_ZRO_SMPLE, ignore_index=True)

        # -------------------------------------------------------------------------------------------------------------------
        # Añadir vectores de imágenes

        TRAIN_v1['vector'] = 0

        TRAIN_v2 = IMG.merge(TRAIN_v2, left_on='review', right_on='reviewId', how='inner')

        DEV = IMG.merge(DEV, left_on='review', right_on='reviewId', how='inner')
        TEST = IMG.merge(TEST, left_on='review', right_on='reviewId', how='inner')

        TRAIN_v1 = TRAIN_v1.drop(
            columns=['restaurantId', 'userId', 'url', 'text', 'real_id_x', 'real_id_y', 'title', 'date', 'images',
                     'num_images', 'index', 'rating', 'language'])
        TRAIN_v2 = TRAIN_v2.drop(
            columns=['restaurantId', 'userId', 'url', 'text', 'real_id_x', 'real_id_y', 'title', 'date', 'images',
                     'num_images', 'index', 'rating', 'language', 'review'])

        DEV = DEV.drop(
            columns=['restaurantId', 'userId', 'url', 'text', 'title', 'date', 'real_id_x', 'real_id_y', 'images',
                     'num_images', 'index', 'rating', 'language', 'review'])
        TEST = TEST.drop(
            columns=['restaurantId', 'userId', 'url', 'text', 'title', 'date', 'real_id_x', 'real_id_y', 'images',
                     'num_images', 'index', 'rating', 'language', 'review'])

        self.printW("Las reviews salen repetidas en función del número de imagenes")

        IMG_2 = np.row_stack(IMG.vector.values)
        M_IMG = np.mean(IMG_2, axis=0)

        IMG_2 = np.apply_along_axis(lambda x: np.power(x - M_IMG, 2), 1, IMG_2)
        IMG_2 = np.apply_along_axis(lambda x: np.mean(x), 1, IMG_2)
        MeanMSE = np.apply_along_axis(lambda x: np.mean(x), 0, IMG_2)
        MaxMSE = np.apply_along_axis(lambda x: np.max(x), 0, IMG_2)
        MinMSE = np.apply_along_axis(lambda x: np.min(x), 0, IMG_2)

        return (TRAIN_v1, TRAIN_v2, DEV, TEST, len(REST_TMP), len(USR_TMP), len(IMG.iloc[0].vector), [MinMSE, MaxMSE, MeanMSE])

    def train(self):
        self.printW("FN SIN IMPLEMENTAR")
        exit()

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

    def getF1(self,pred,real, title = "", verbose=True):
        TP=0
        FN=0
        TN=0
        FP=0

        for i in range(len(pred)):
            p = 1 if  pred[i][0] > .5  else 0
            r = real[i]

            if(p==1 and r==1):TP+=1
            if(p==1 and r==0):FP+=1
            if(p==0 and r==1):FN+=1
            if(p==0 and r==0):TN+=1

        pre = TP/(TP+FP)
        rec = TP/(TP+FN)
        f1 = 2*((pre*rec)/(pre+rec))

        if(verbose):
            line_lng = 45

            if(title!=""):
                self.printB("‒" * line_lng)
                self.printB(" "+str(title),bold=True)

            self.printB("‒"*line_lng)
            self.printB("TP: "+str(TP)+"\tFP: "+str(FP)+"\tFN: "+str(FN)+"\tTN: "+str(TN))
            self.printB("‒"*line_lng)
            self.printB("Precision: \t"+str(pre))
            self.printB("Recall: \t"+str(rec))
            self.printB("F1: \t"+str(f1), bold=True)
            self.printB("‒"*line_lng)

        return(TP,FP,FN,TN,pre,rec,f1)

    def getSlope(self,data):
            r = linregress(range(len(data)), data)
            return r.slope

    def printW(self,text):
        print(bcolors.WARNING+str("[AVISO] ")+str(text)+bcolors.ENDC)

    def printG(self,text):
        print(bcolors.OKGREEN+str("[INFO] ")+str(text)+bcolors.ENDC)

    def printB(self,text, bold=False):
        if(bold):
            print(bcolors.BOLD+bcolors.OKBLUE+str(text)+bcolors.ENDC+bcolors.ENDC)
        else:
            print(bcolors.OKBLUE + str(text) + bcolors.ENDC)