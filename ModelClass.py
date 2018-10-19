# -*- coding: utf-8 -*-
import os, time
import warnings

import numpy as np
import pandas as pd
import pickle

import random as rn

import keras
import tensorflow as tf
from keras import backend as K
from keras import losses
from keras.utils import *
from keras.models import Model
from keras.layers import Input,Dense,Activation,Concatenate, Dot,Conv2D,MaxPooling2D, Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from scipy.stats import linregress
from sklearn import metrics
from sklearn import utils


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

    def __init__(self,model_class):
        self.FIRST_TIME = True
        self.MODEL_CLASS = model_class

        self.usr_dev = to_categorical(model_class.DEV.id_user, num_classes=model_class.N_USR)
        self.img_dev = np.zeros((len(model_class.DEV), model_class.V_IMG))
        self.res_dev = to_categorical(model_class.DEV.id_restaurant, num_classes=model_class.N_RST)
        self.out_dev = model_class.DEV.like.values

        #self.HEADER = list(config.keys())
        #self.VALUES = list((str(x).replace(".",",") for x in list(config.values())))

    def on_batch_end(self, batch, logs=None):
        dev_loss = self.MODEL_CLASS.MODEL.evaluate([self.usr_dev, self.res_dev], [self.out_dev, self.img_dev],verbose=0)

        pred_dev, _ = self.MODEL_CLASS.MODEL.predict([self.usr_dev, self.res_dev], verbose=0)
        TP, FP, FN, TN = self.MODEL_CLASS.getConfMatrix(pred_dev, self.out_dev, verbose=False)
        dev_accuracy = (TP + TN) / sum([TP, FP, FN, TN])

        print(str(logs["batch"])+"\t"+str(logs["dotprod_loss"]).replace(".",",")+"\t"+str(dev_loss[1]).replace(".",",")+"\t"+str(dev_accuracy))

########################################################################################################################

class ModelClass():

    def __init__(self,city,option,config,name,seed = 2 ):

        self.CITY = city
        self.OPTION = option
        self.PATH = "/media/HDD/pperez/TripAdvisor/" + self.CITY.lower() + "_data/"
        self.IMG_PATH = self.PATH + "images/"
        self.SEED = seed
        self.CONFIG = config
        self.MODEL_NAME = name

        #Eliminar info de TF
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['PYTHONHASHSEED'] = '0'
        warnings.filterwarnings('always')

        #Fijar las semillas de numpy y TF
        np.random.seed(self.SEED)
        rn.seed(self.SEED)
        tf.set_random_seed(self.SEED)

        #Utilizar solo memoria GPU necesaria
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        #config.intra_op_parallelism_threads = 1
        #config.inter_op_parallelism_threads = 1

        sess = tf.Session(graph=tf.get_default_graph(), config=config)
        K.set_session(sess)

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

        # Mirar si ya existen los datos
        # ---------------------------------------------------------------------------------------------------------------
        file_path = self.PATH+"model_data_"+str(self.CONFIG['oversampling']).lower()+"/"

        if(os.path.exists(file_path)):
            self.printW("Cargando datos generados previamente...")

            TRAIN_v1 =  self.getPickle(file_path, "TRAIN_v1")
            TRAIN_v2 =  self.getPickle(file_path, "TRAIN_v2")
            DEV =       self.getPickle(file_path, "DEV")
            TEST =      self.getPickle(file_path, "TEST")
            REST_TMP =  self.getPickle(file_path, "REST_TMP")
            USR_TMP =   self.getPickle(file_path, "USR_TMP")
            IMG =       self.getPickle(file_path, "IMG")
            MSE =       self.getPickle(file_path, "MSE")

            self.getCardinality(TRAIN_v1.like, title="TRAIN_v1" ,verbose=True)
            self.getCardinality(TRAIN_v2.like, title="TRAIN_v2" ,verbose=True)
            self.getCardinality(DEV.like, title="DEV" ,verbose=True)
            self.getCardinality(TEST.like, title="TEST" ,verbose=True)

            return(TRAIN_v1,TRAIN_v2,DEV,TEST,REST_TMP,USR_TMP,IMG,MSE)

        else:
            os.makedirs(file_path)

        # ---------------------------------------------------------------------------------------------------------------

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

        TRAIN = []
        DEV = []
        TEST = []

        self.printG("Separando en TRAIN/DEV/TEST valoraciones con imagen...")

        for i, g in GRP_RVW_IM:
            rnd = rn.random()
            if (rnd < 0.05):
                TEST.extend(g.reviewId.values)
            elif (rnd >= 0.05 and rnd < 0.1):
                DEV.extend(g.reviewId.values)
            else:
                TRAIN.extend(g.reviewId.values)

        self.printG("Separando en TRAIN/DEV/TEST valoraciones sin imagen...")
        for i, g in GRP_RVW_NIM:
            rnd = rn.random()
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

        if(self.CONFIG['oversampling']!='none'):

            self.printG("Oversampling en TRAIN_V1...")

            TRAIN_ONE = TRAIN_v1.loc[TRAIN_v1.like == 1]
            TRAIN_ZRO = TRAIN_v1.loc[TRAIN_v1.like == 0]

            self.printG("\tZeros:"+str(len(TRAIN_ZRO))+"\tOnes:"+str(len(TRAIN_ONE)))

            TRAIN_v1 = TRAIN_ONE.append(TRAIN_ZRO, ignore_index=True)

            if(self.CONFIG['oversampling']=="auto"): SMPLE_ITEMS = len(TRAIN_ONE)-len(TRAIN_ZRO)
            else: SMPLE_ITEMS = len(TRAIN_ZRO)*(int(self.CONFIG['oversampling'])-1)

            TRAIN_ZRO_SMPLE = TRAIN_ZRO.sample(SMPLE_ITEMS, replace=True, random_state=self.SEED)

            TRAIN_v1 = TRAIN_v1.append(TRAIN_ZRO_SMPLE, ignore_index=True)

            self.printG("\tZeros:"+str(len(TRAIN_v1)-sum(TRAIN_v1.like))+"\tOnes:"+str(sum(TRAIN_v1.like)))

            self.printG("Oversampling en TRAIN_V2...")

            TRAIN_ONE = TRAIN_v2.loc[TRAIN_v2.like == 1]
            TRAIN_ZRO = TRAIN_v2.loc[TRAIN_v2.like == 0]

            self.printG("\tZeros:"+str(len(TRAIN_ZRO))+"\tOnes:"+str(len(TRAIN_ONE)))

            TRAIN_v2 = TRAIN_ONE.append(TRAIN_ZRO, ignore_index=True)

            if(self.CONFIG['oversampling']=="auto"): SMPLE_ITEMS = len(TRAIN_ONE)-len(TRAIN_ZRO)
            else: SMPLE_ITEMS = len(TRAIN_ZRO)*(int(self.CONFIG['oversampling'])-1)

            TRAIN_ZRO_SMPLE = TRAIN_ZRO.sample(SMPLE_ITEMS, replace=True, random_state=self.SEED)
            TRAIN_v2 = TRAIN_v2.append(TRAIN_ZRO_SMPLE, ignore_index=True)

            self.printG("\tZeros:"+str(len(TRAIN_v2)-sum(TRAIN_v2.like))+"\tOnes:"+str(sum(TRAIN_v2.like)))

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
                     'num_images', 'index', 'rating', 'language', 'review', 'image'])

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

        # MEZCLAR DATOS ------------------------------------------------------------------------------------------------

        TRAIN_v1 = utils.shuffle(TRAIN_v1, random_state = self.SEED)
        TRAIN_v2 = utils.shuffle(TRAIN_v2, random_state = self.SEED)
        DEV = utils.shuffle(DEV, random_state = self.SEED)
        TEST = utils.shuffle(TEST, random_state = self.SEED)

        # ALMACENAR PICKLE ------------------------------------------------------------------------------------------------

        self.toPickle(file_path,"TRAIN_v1",TRAIN_v1)
        self.toPickle(file_path,"TRAIN_v2",TRAIN_v2)
        self.toPickle(file_path,"DEV",DEV)
        self.toPickle(file_path,"TEST",TEST)
        self.toPickle(file_path,"REST_TMP",len(REST_TMP))
        self.toPickle(file_path,"USR_TMP",len(USR_TMP))
        self.toPickle(file_path,"IMG",len(IMG.iloc[0].vector))
        self.toPickle(file_path,"MSE",[MinMSE, MaxMSE, MeanMSE])

        return (TRAIN_v1, TRAIN_v2, DEV, TEST, len(REST_TMP), len(USR_TMP), len(IMG.iloc[0].vector), [MinMSE, MaxMSE, MeanMSE])

    def newgetData(self):

        # Mirar si ya existen los datos
        # ---------------------------------------------------------------------------------------------------------------
        '''
        file_path = self.PATH + "model_data_" + str(self.CONFIG['oversampling']).lower() + "/"

        if (os.path.exists(file_path)):
            self.printW("Cargando datos generados previamente...")

            TRAIN_v1 = self.getPickle(file_path, "TRAIN_v1")
            TRAIN_v2 = self.getPickle(file_path, "TRAIN_v2")
            DEV = self.getPickle(file_path, "DEV")
            TEST = self.getPickle(file_path, "TEST")
            REST_TMP = self.getPickle(file_path, "REST_TMP")
            USR_TMP = self.getPickle(file_path, "USR_TMP")
            IMG = self.getPickle(file_path, "IMG")
            MSE = self.getPickle(file_path, "MSE")

            self.getCardinality(TRAIN_v1.like, title="TRAIN_v1", verbose=True)
            self.getCardinality(TRAIN_v2.like, title="TRAIN_v2", verbose=True)
            self.getCardinality(DEV.like, title="DEV", verbose=True)
            self.getCardinality(TEST.like, title="TEST", verbose=True)

            return (TRAIN_v1, TRAIN_v2, DEV, TEST, REST_TMP, USR_TMP, IMG, MSE)

        else:
            os.makedirs(file_path)
        '''

        # ---------------------------------------------------------------------------------------------------------------

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

        # Eliminar usuarios con menos de min_pos_revs positivos
        # ---------------------------------------------------------------------------------------------------------------

        USR_LST = RVW.groupby("userId", as_index=False).sum()
        USR_LST = USR_LST.loc[(USR_LST.like >= self.CONFIG['min_pos_revs']), "userId"].values

        print(USR_LST)


        exit()



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
        TRAIN = []
        DEV = []
        TEST = []



        exit()

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

        TRAIN = []
        DEV = []
        TEST = []

        self.printG("Separando en TRAIN/DEV/TEST valoraciones con imagen...")

        for i, g in GRP_RVW_IM:
            rnd = rn.random()
            if (rnd < 0.05):
                TEST.extend(g.reviewId.values)
            elif (rnd >= 0.05 and rnd < 0.1):
                DEV.extend(g.reviewId.values)
            else:
                TRAIN.extend(g.reviewId.values)

        self.printG("Separando en TRAIN/DEV/TEST valoraciones sin imagen...")
        for i, g in GRP_RVW_NIM:
            rnd = rn.random()
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

        if (self.CONFIG['oversampling'] != 'none'):

            self.printG("Oversampling en TRAIN_V1...")

            TRAIN_ONE = TRAIN_v1.loc[TRAIN_v1.like == 1]
            TRAIN_ZRO = TRAIN_v1.loc[TRAIN_v1.like == 0]

            self.printG("\tZeros:" + str(len(TRAIN_ZRO)) + "\tOnes:" + str(len(TRAIN_ONE)))

            TRAIN_v1 = TRAIN_ONE.append(TRAIN_ZRO, ignore_index=True)

            if (self.CONFIG['oversampling'] == "auto"):
                SMPLE_ITEMS = len(TRAIN_ONE) - len(TRAIN_ZRO)
            else:
                SMPLE_ITEMS = len(TRAIN_ZRO) * (int(self.CONFIG['oversampling']) - 1)

            TRAIN_ZRO_SMPLE = TRAIN_ZRO.sample(SMPLE_ITEMS, replace=True, random_state=self.SEED)

            TRAIN_v1 = TRAIN_v1.append(TRAIN_ZRO_SMPLE, ignore_index=True)

            self.printG("\tZeros:" + str(len(TRAIN_v1) - sum(TRAIN_v1.like)) + "\tOnes:" + str(sum(TRAIN_v1.like)))

            self.printG("Oversampling en TRAIN_V2...")

            TRAIN_ONE = TRAIN_v2.loc[TRAIN_v2.like == 1]
            TRAIN_ZRO = TRAIN_v2.loc[TRAIN_v2.like == 0]

            self.printG("\tZeros:" + str(len(TRAIN_ZRO)) + "\tOnes:" + str(len(TRAIN_ONE)))

            TRAIN_v2 = TRAIN_ONE.append(TRAIN_ZRO, ignore_index=True)

            if (self.CONFIG['oversampling'] == "auto"):
                SMPLE_ITEMS = len(TRAIN_ONE) - len(TRAIN_ZRO)
            else:
                SMPLE_ITEMS = len(TRAIN_ZRO) * (int(self.CONFIG['oversampling']) - 1)

            TRAIN_ZRO_SMPLE = TRAIN_ZRO.sample(SMPLE_ITEMS, replace=True, random_state=self.SEED)
            TRAIN_v2 = TRAIN_v2.append(TRAIN_ZRO_SMPLE, ignore_index=True)

            self.printG("\tZeros:" + str(len(TRAIN_v2) - sum(TRAIN_v2.like)) + "\tOnes:" + str(sum(TRAIN_v2.like)))

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
                     'num_images', 'index', 'rating', 'language', 'review', 'image'])

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

        # MEZCLAR DATOS ------------------------------------------------------------------------------------------------

        TRAIN_v1 = utils.shuffle(TRAIN_v1, random_state=self.SEED)
        TRAIN_v2 = utils.shuffle(TRAIN_v2, random_state=self.SEED)
        DEV = utils.shuffle(DEV, random_state=self.SEED)
        TEST = utils.shuffle(TEST, random_state=self.SEED)

        # ALMACENAR PICKLE ------------------------------------------------------------------------------------------------

        self.toPickle(file_path, "TRAIN_v1", TRAIN_v1)
        self.toPickle(file_path, "TRAIN_v2", TRAIN_v2)
        self.toPickle(file_path, "DEV", DEV)
        self.toPickle(file_path, "TEST", TEST)
        self.toPickle(file_path, "REST_TMP", len(REST_TMP))
        self.toPickle(file_path, "USR_TMP", len(USR_TMP))
        self.toPickle(file_path, "IMG", len(IMG.iloc[0].vector))
        self.toPickle(file_path, "MSE", [MinMSE, MaxMSE, MeanMSE])

        return (
        TRAIN_v1, TRAIN_v2, DEV, TEST, len(REST_TMP), len(USR_TMP), len(IMG.iloc[0].vector), [MinMSE, MaxMSE, MeanMSE])

    def train(self):
        self.printW("FN SIN IMPLEMENTAR")
        exit()

    #- STATS -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

    def getDataStats(self):

        file = open("out.tsv", "w")

        RVW = pd.read_pickle(self.PATH + "reviews.pkl")
        RVW["like"] = RVW.rating.apply(lambda x: 1 if x > 30 else 0)
        RVW["num_images"] = RVW.images.apply(lambda x: len(x))

        RVW = RVW.loc[(RVW.userId != "")]

        # Todos los usuarios
        ALL_USR_LST = RVW.groupby("userId", as_index=False).count()

        # Usuarios con X reviews mínimo
        USR_LST = ALL_USR_LST.loc[(ALL_USR_LST.like >= 5), "userId"].values
        RVW = RVW.loc[RVW.userId.isin(USR_LST)]

        RVW_USR = RVW.groupby('userId')

        print(len(ALL_USR_LST),len(USR_LST))


        for i,r in RVW_USR:
            usr = i
            total = len(r)
            pos = sum(r.like.values)
            neg = total - pos

            imgs = r.loc[(r.num_images > 0)]

            total_imgs = len(imgs)
            pos_imgs = sum(imgs.like.values)
            neg_imgs = total_imgs - pos_imgs

            log = "\t".join(map(lambda x: str(x), [usr, pos, neg, pos_imgs, neg_imgs])) + "\n"

            file.write(log)

        file.close()

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

    def getConfMatrix(self, pred, real, title ="", verbose=True):

        f = np.vectorize(lambda x: 1 if(x<.5) else 0)
        pred_tmp = f(np.array(pred[:,0]))

        TN, FP, FN, TP = metrics.confusion_matrix(real,pred_tmp).ravel()

        return(TP,FP,FN,TN)

    def getAUC(self,pred,real):
        auc = metrics.roc_auc_score(np.array(real),np.array(pred[:,0]))
        return auc

    def getF1(self,pred,real, invert=False):

        if(invert):
            TN, FN, FP, TP = self.getConfMatrix(pred, real)
        else:
            TP, FP, FN, TN = self.getConfMatrix(pred, real)

        PR = TP/(TP+FP) if (TP+FP)>0 else 0
        RC = TP/(TP+FN) if (TP+FN)>0 else 0
        F1 = 2*((PR*RC)/(PR+RC)) if (PR+RC)>0 else 0

        return F1

    def getBIN_AUC(self,pred,real):
        f = np.vectorize(lambda x: 1 if (x < .5) else 0)
        pred = f(np.array(pred[:, 0]))
        auc = metrics.roc_auc_score(np.array(real),pred)
        return auc

    def getCardinality(self,data,verbose=False, title = ""):

        total = len(data)
        ones = sum(data)
        zeros = total-ones

        if(verbose):
            self.printB("\t"+title+"\t\tzeros:"+str(zeros)+" ones:"+str(ones))

        return total, zeros, ones

    def getSlope(self,data):
            r = linregress(range(len(data)), data)
            return r.slope

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

    def toPickle(self,path,name,data):
        with open(path+name, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def getPickle(self,path,name):
        with open(path+name, 'rb') as handle:
            data = pickle.load(handle)
        return data

    def printW(self,text):
        print(bcolors.WARNING+str("[AVISO] ")+str(text)+bcolors.ENDC)

    def printG(self,text):
        print(bcolors.OKGREEN+str("[INFO] ")+str(text)+bcolors.ENDC)

    def printB(self,text, bold=False):
        if(bold):
            print(bcolors.BOLD+bcolors.OKBLUE+str(text)+bcolors.ENDC+bcolors.ENDC)
        else:
            print(bcolors.OKBLUE + str(text) + bcolors.ENDC)