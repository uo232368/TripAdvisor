# -*- coding: utf-8 -*-
import os, time
import warnings
import re

import numpy as np
import pandas as pd
import pickle
import math
import signal, sys
import hashlib
import itertools as it

import random as rn
from PIL import Image

import keras
import tensorflow as tf
from keras import backend as K
from keras import losses
from keras.utils import *
from keras.models import Model
from keras.layers import Input,Dense,Activation,Concatenate, Dot,Conv2D,MaxPooling2D, Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

import scipy
from scipy.spatial import distance_matrix
from scipy.stats import linregress
from sklearn import metrics
from sklearn import utils
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split


import ssl
import urllib
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

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

        signal.signal(signal.SIGINT, self.signal_handler)

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
        #os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

        train1, train2, train3,train3_1,train4, dev, dev2,dev3,dev3_1, test,test2,test3, n_rest, n_usr, v_img, mse_data = self.getData()

        self.TRAIN_V1 = train1
        self.TRAIN_V2 = train2
        self.TRAIN_V3 = train3
        self.TRAIN_V3_1 = train3_1
        self.TRAIN_V4 = train4

        self.DEV = dev
        self.DEV_V2 = dev2
        self.DEV_V3 = dev3
        self.DEV_V3_1 = dev3_1

        self.TEST = test
        self.TEST_V2 = test2
        self.TEST_V3 = test3

        self.N_RST = n_rest
        self.N_USR = n_usr
        self.V_IMG = v_img
        self.MSE_DATA = mse_data

        self.printB("Creando modelo...")

        self.MODEL_PATH = "models/"+self.MODEL_NAME+"_" + self.CITY.lower() + "_option" + str(self.OPTION)
        self.MODEL = self.getModel()
        self.SESSION = None

        print("\n")
        print("#"*50)
        print(' '+self.MODEL_NAME.upper())
        print("#"*50)

    def signal_handler(self,signal, frame):
        self.stop()
        sys.exit(0)

    def getModel(self):
        self.printW("FN SIN IMPLEMENTAR")
        exit()

    def getFilteredData(self,verbose=True):

        IMG = pd.read_pickle(self.PATH + "img-option" + str(self.OPTION) + ".pkl")
        RVW = pd.read_pickle(self.PATH + "reviews.pkl")

        IMG['review'] = IMG.review.astype(int)
        RVW["reviewId"] = RVW.reviewId.astype(int)

        RVW["num_images"] = RVW.images.apply(lambda x: len(x))
        RVW["like"] = RVW.rating.apply(lambda x: 1 if x > 30 else 0)
        RVW = RVW.loc[(RVW.userId != "")]

        #USR5 = RVW.loc[RVW.reviewId.isin([270316041,344306570])]
        #print(USR5[['url']].values)
        #exit()

        # Eliminar RESTAURANTES con menos de 5 revs
        # ---------------------------------------------------------------------------------------------------------------
        RST_LST = RVW.groupby("restaurantId", as_index=False).count()

        old_rest_len = len(RST_LST)

        RST_LST = RST_LST.loc[(RST_LST.like >= self.CONFIG['min_rest_revs']), "restaurantId"].values

        if(verbose):self.printW("Eliminado "+str(old_rest_len-len(RST_LST))+" restaurantes (con menos de " + str(self.CONFIG['min_rest_revs']) + " reviews) de un total de "+str(old_rest_len)+" ("+str(1-(len(RST_LST)/old_rest_len))+" %)")

        old_rvw_len = len(RVW)
        RVW = RVW.loc[RVW.restaurantId.isin(RST_LST)]

        if (verbose):self.printW("\t- Se pasa de "+str(old_rvw_len)+" reviews a "+str(len(RVW))+" reviews.")


        # Eliminar usuarios con menos de min_revs
        # ---------------------------------------------------------------------------------------------------------------
        '''
        USR_LST = RVW.groupby("userId", as_index=False).count()

        old_usr_len = len(USR_LST)
        USR_LST = USR_LST.loc[(USR_LST.like >= self.CONFIG['min_revs']), "userId"].values

        self.printW("Eliminado "+str(old_usr_len-len(USR_LST))+" usuarios (con menos de " + str(self.CONFIG['min_revs']) + " reviews) de un total de "+str(old_usr_len)+" ("+str(1-(len(USR_LST)/old_usr_len))+" %)")

        old_rvw_len = len(RVW)
        RVW = RVW.loc[RVW.userId.isin(USR_LST)]

        self.printW("\t- Se pasa de "+str(old_rvw_len)+" reviews a "+str(len(RVW))+" reviews.")
        '''
        # Eliminar usuarios con menos de min_pos_revs positivos
        # ---------------------------------------------------------------------------------------------------------------

        USR_LST = RVW.groupby("userId", as_index=False).sum()
        old_usr_len = len(USR_LST)

        USR_LST = USR_LST.loc[(USR_LST.like >= self.CONFIG['min_usr_revs']), "userId"].values
        #USR_LST = USR_LST.loc[(USR_LST.like == self.CONFIG['min_usr_revs']), "userId"].values;self.printE("SOLO 5 REVIEWS") # <===============

        if (verbose):self.printW("Eliminado "+str(old_usr_len-len(USR_LST))+" usuarios (con menos de " + str(self.CONFIG['min_usr_revs']) + " reviews positivas) de un total de "+str(old_usr_len)+" ("+str(1-(len(USR_LST)/old_usr_len))+" %)")

        old_rvw_len = len(RVW)

        RVW = RVW.loc[RVW.userId.isin(USR_LST)]

        if (verbose):self.printW("\t- Se pasa de "+str(old_rvw_len)+" reviews a "+str(len(RVW))+" reviews.")

        # Obtener ID para ONE-HOT de usuarios y restaurantes
        # ---------------------------------------------------------------------------------------------------------------

        USR_TMP = pd.DataFrame(columns=["real_id", "id_user"])
        REST_TMP = pd.DataFrame(columns=["real_id", "id_restaurant"])

        # Obtener tabla real_id -> id para usuarios
        USR_TMP.real_id = RVW.sort_values("userId").userId.unique()
        USR_TMP.id_user = range(0, len(USR_TMP))

        # Obtener tabla real_id -> id para restaurantes
        REST_TMP.real_id = RVW.sort_values("restaurantId").restaurantId.unique()
        REST_TMP.id_restaurant = range(0, len(REST_TMP))

        # Mezclar datos
        RET = RVW.merge(USR_TMP, left_on='userId', right_on='real_id', how='inner')
        RET = RET.merge(REST_TMP, left_on='restaurantId', right_on='real_id', how='inner')

        RVW = RET[['date', 'images', 'language', 'rating', 'restaurantId', 'reviewId', 'text', 'title', 'url', 'userId', 'num_images', 'id_user', 'id_restaurant', 'like']]

        if (verbose):
            print("USUARIOS:"+str(len(RVW.userId.unique())))
            print("RESTAURANTES:"+str(len(RVW.restaurantId.unique())))


        return RVW, IMG, USR_TMP,REST_TMP

    def getData(self):

        # Mirar si ya existen los datos
        # ---------------------------------------------------------------------------------------------------------------

        file_path = self.PATH +"model_data"
        file_path += "_"+str(self.CONFIG['min_usr_revs'])
        file_path += "_"+str(self.CONFIG['min_rest_revs'])
        file_path += "_"+str(self.CONFIG['top_n_new_items'])
        file_path += "_"+str(self.CONFIG['train_pos_rate'])
        file_path += "_"+ ("PRB" if(self.CONFIG['use_rest_provs']) else "RNDM")

        if (self.CONFIG['new_train_examples'] == -1): file_path += "_AUTO"
        elif (self.CONFIG['new_train_examples'] == -2): file_path += "_AUTO2"
        else: file_path += "_"+str(self.CONFIG['new_train_examples'])

        file_path += "/"



        if (os.path.exists(file_path)):
            self.printW("Cargando datos generados previamente...")

            TRAIN_v1 = self.getPickle(file_path, "TRAIN_v1")
            TRAIN_v2 = self.getPickle(file_path, "TRAIN_v2")
            TRAIN_v3 = self.getPickle(file_path, "TRAIN_v3")
            TRAIN_v3_1 = self.getPickle(file_path, "TRAIN_v3_1")
            TRAIN_v4 = self.getPickle(file_path, "TRAIN_v4")

            DEV = self.getPickle(file_path, "DEV")
            DEV_v2 = self.getPickle(file_path, "DEV_v2")
            DEV_v3 = self.getPickle(file_path, "DEV_v3")
            DEV_v3_1 = self.getPickle(file_path, "DEV_v3_1")

            TEST = self.getPickle(file_path, "TEST")
            TEST_v2 = self.getPickle(file_path, "TEST_v2")
            TEST_v3 = self.getPickle(file_path, "TEST_v3")

            REST_TMP = self.getPickle(file_path, "REST_TMP")
            USR_TMP = self.getPickle(file_path, "USR_TMP")
            IMG = self.getPickle(file_path, "IMG")
            MSE = self.getPickle(file_path, "MSE")


            self.getCardinality(TRAIN_v1.like, title="TRAIN_v1", verbose=True)
            self.getCardinality(TRAIN_v2.like, title="TRAIN_v2", verbose=True)
            self.getCardinality(TRAIN_v3.like, title="TRAIN_v3", verbose=True)
            self.getCardinality(TRAIN_v3_1.like, title="TRAIN_v3_1", verbose=True)

            self.getCardinality(DEV.like, title="DEV", verbose=True)
            self.getCardinality(DEV_v2.like, title="DEV_v2", verbose=True)
            self.getCardinality(DEV_v3.like, title="DEV_v3", verbose=True)
            self.getCardinality(DEV_v3_1.like, title="DEV_v3_1", verbose=True)

            self.getCardinality(TEST.like, title="TEST", verbose=True)
            self.getCardinality(TEST_v2.like, title="TEST_v2", verbose=True)
            self.getCardinality(TEST_v3.like, title="TEST_v3", verbose=True)

            return (TRAIN_v1, TRAIN_v2,TRAIN_v3,TRAIN_v3_1,TRAIN_v4, DEV,DEV_v2,DEV_v3,DEV_v3_1, TEST,TEST_v2,TEST_v3, REST_TMP, USR_TMP, IMG, MSE)



        # ---------------------------------------------------------------------------------------------------------------

        RVW, IMG, USR_TMP, REST_TMP = self.getFilteredData();

        # Mover ejemplos positivos a donde corresponde
        # ---------------------------------------------------------------------------------------------------------------

        TRAIN = pd.DataFrame()
        DEV = pd.DataFrame()
        TEST = pd.DataFrame()

        #GRP_USR = RVW.groupby(["userId"])

        TRAIN_POS_RATE = self.CONFIG['train_pos_rate']
        DEV_TEST_RATE = self.CONFIG['dev_test_pos_rate']

        assert TRAIN_POS_RATE+DEV_TEST_RATE+DEV_TEST_RATE == 1.0, "Error en porcentajes"

        if(TRAIN_POS_RATE==1):assert self.CONFIG['min_usr_revs'] >= 3, "Mínimo 3 ejemplos positivos (TRAIN/DEV/TEST) "

        POS_REVS = RVW.loc[(RVW.like==1)].reset_index(drop=True)
        POS_REVS = POS_REVS.sort_values(by=['id_user','reviewId'], ascending=[True,True]); print("SORTED")

        def split_fn(d):

            items = len(d)

            if(TRAIN_POS_RATE!=1.0): dev_test_items = int(items * DEV_TEST_RATE)
            else:dev_test_items = 1

            train_items = items - (2 * dev_test_items)
            assert dev_test_items >= 1, "No puede haber menos de 1 item en dev y test"
            d["TO_TRAIN"] = 0; d["TO_DEV"] = 0; d["TO_TEST"] = 0

            upper = False

            if(upper):
                d.iloc[:dev_test_items , -1] = 1
                d.iloc[dev_test_items:-train_items , -2] = 1
                d.iloc[-train_items:, -3] = 1

            else:
                d.iloc[:train_items,-3] = 1
                d.iloc[train_items:train_items+dev_test_items,-2] = 1
                d.iloc[train_items+dev_test_items:,-1] = 1

            return d

        POS_REVS = POS_REVS.groupby('id_user').apply(split_fn)

        TRAIN = POS_REVS.loc[POS_REVS.TO_TRAIN==1]
        DEV = POS_REVS.loc[POS_REVS.TO_DEV==1]
        TEST = POS_REVS.loc[POS_REVS.TO_TEST==1]

        TRAIN = TRAIN.drop(columns=["TO_TRAIN","TO_DEV","TO_TEST"])
        DEV = DEV.drop(columns=["TO_TRAIN","TO_DEV","TO_TEST"])
        TEST = TEST.drop(columns=["TO_TRAIN","TO_DEV","TO_TEST"])

        # Mover ejemplos negativos a donde corresponde
        # ---------------------------------------------------------------------------------------------------------------

        NEG_REVS = RVW.loc[(RVW.like!=1)]
        TRAIN = TRAIN.append(NEG_REVS, ignore_index=True,sort=False) #Todos a TRAIN

        # Crear ejemplos nuevos para compensar distribución de clases
        # ---------------------------------------------------------------------------------------------------------------

        N_NEW_ITEMS = len(TRAIN.loc[TRAIN.like==1])-len(NEG_REVS)

        assert len(TRAIN.loc[TRAIN.like==1])> len(NEG_REVS), "Existen más negativos que positivos..."

        N_USERS = len(USR_TMP)

        if (self.CONFIG['new_train_examples'] == -1):
            ITEMS_PER_USR = (N_NEW_ITEMS // N_USERS)
            ITEMS_LEFT = N_NEW_ITEMS - (ITEMS_PER_USR * N_USERS)
            if (ITEMS_LEFT > 0): ITEMS_PER_USR += 1

        elif(self.CONFIG['new_train_examples'] == -2):
            ITEMS_PER_USR = int((len(REST_TMP) / len(USR_TMP)) * 100)

        else:
            ITEMS_PER_USR = self.CONFIG['new_train_examples'];


        self.printW("Se añaden " + str(ITEMS_PER_USR) + " items nuevos por usuario. ("+str((ITEMS_PER_USR*N_USERS))+" en total)")


        #Obtener la lista de restaurantes, el número de reviews y la probabilidad de aparecer
        rst_ids_prob = RVW.groupby("id_restaurant", as_index=False).apply(lambda x : pd.Series({"n_rvs":len(x)})).reset_index(drop=True)
        #rst_ids_prob["prob"] = rst_ids_prob["n_rvs"] / sum(rst_ids_prob["n_rvs"].values)


        rest_ids = set(rst_ids_prob.index)

        used_restaurants = {} #Contiene todos los pares usr restaurante ya ulitizados hasta el momento

        def append_no_reviewed_restaurants(data):
            no_reviewed_rests = list(rest_ids.difference(set(data.id_restaurant.values)))

            if(self.CONFIG['use_rest_provs']):
                no_reviewed_probs = rst_ids_prob.loc[no_reviewed_rests,"n_rvs"].values
                no_reviewed_probs = no_reviewed_probs / sum(no_reviewed_probs)
            else:
                no_reviewed_probs = [1 / len(no_reviewed_rests)] * len(no_reviewed_rests)

            #Selecionar items en función del número de reviews (más probables si más reviews)
            no_reviewed_items = np.random.choice(no_reviewed_rests, ITEMS_PER_USR, replace=False,p=no_reviewed_probs)

            used_restaurants[data.id_user.values[0]] = np.append(data.id_restaurant.values,no_reviewed_items)

            ret = pd.DataFrame(-1, index=np.arange(ITEMS_PER_USR), columns=data.columns)
            ret["id_user"]=data.id_user.values[0]
            ret["like"]=0
            ret["id_restaurant"]=no_reviewed_items
            ret = ret.drop(columns="userId")

            return ret

        NEW_REVS = RVW.groupby(['userId']).apply(append_no_reviewed_restaurants).reset_index()
        NEW_REVS = NEW_REVS.drop(columns="level_1")

        NEW_TRAIN = TRAIN.append(NEW_REVS, ignore_index=True, sort=True) #Todos a TRAIN


        # Añadir al conjunto de DEV los 100 restaurantes no vistos
        # ---------------------------------------------------------------------------------------------------------------

        TOPN_NEW_ITEMS = self.CONFIG['top_n_new_items'];

        dev_used_restaurants = {} #Contiene todos los pares usr restaurante ya ulitizados hasta el momento

        def append_topn_items_dev(data):

            idUser = data.id_user.values[0]
            used_rests = used_restaurants[idUser]

            no_reviewed_rests = list(rest_ids.difference(set(used_rests)))
            
            if(self.CONFIG['use_rest_provs']):
                no_reviewed_probs = rst_ids_prob.loc[no_reviewed_rests, "n_rvs"].values
                no_reviewed_probs = no_reviewed_probs / sum(no_reviewed_probs)
            else:
                no_reviewed_probs = [1/len(no_reviewed_rests)]*len(no_reviewed_rests)

            # Selecionar items en función del número de reviews (más probables si más reviews)
            no_reviewed_items = np.random.choice(no_reviewed_rests, TOPN_NEW_ITEMS, replace=False, p=no_reviewed_probs)

            dev_used_restaurants[idUser] = no_reviewed_items

            ret = pd.DataFrame(-1, index=np.arange(TOPN_NEW_ITEMS), columns=data.columns)

            ret["id_user"] = data.id_user.values[0]
            ret["like"] = 0
            ret["id_restaurant"] = no_reviewed_items
            ret = ret.drop(columns="userId")

            ret = ret.append(data.drop(columns=["userId"]),ignore_index=True)

            return ret

        NEW_DEV = DEV.groupby(['userId']).apply(append_topn_items_dev).reset_index(drop=True)
        #NEW_DEV = NEW_DEV.drop(columns="level_1")

        #Añadir los restaurantes usados en dev a la lista total
        for i in used_restaurants:used_restaurants[i] = np.append(used_restaurants[i],dev_used_restaurants[i])

        # Añadir al conjunto de TEST los 1000 restaurantes no vistos
        # ---------------------------------------------------------------------------------------------------------------
        TOPN_NEW_ITEMS = self.CONFIG['top_n_new_items'];

        def append_topn_items_test(data):

            idUser = data.id_user.values[0]
            used_rests = used_restaurants[idUser]

            no_reviewed_rests = list(rest_ids.difference(set(used_rests)))
            
            if(self.CONFIG['use_rest_provs']):
                no_reviewed_probs = rst_ids_prob.loc[no_reviewed_rests, "n_rvs"].values
                no_reviewed_probs = no_reviewed_probs / sum(no_reviewed_probs)
            else:
                no_reviewed_probs = [1/len(no_reviewed_rests)]*len(no_reviewed_rests)

            # Selecionar items en función del número de reviews (más probables si más reviews)
            no_reviewed_items = np.random.choice(no_reviewed_rests, TOPN_NEW_ITEMS, replace=False, p=no_reviewed_probs)

            ret = pd.DataFrame(-1, index=np.arange(TOPN_NEW_ITEMS), columns=data.columns)
            ret["id_user"] = data.id_user.values[0]
            ret["like"] = 0
            ret["id_restaurant"] = no_reviewed_items
            ret = ret.drop(columns="userId")

            ret = ret.append(data.drop(columns=["userId"]),ignore_index=True)

            return ret

        NEW_TEST = TEST.groupby(['userId']).apply(append_topn_items_test).reset_index(drop=True)
        #NEW_TEST = NEW_TEST.drop(columns="level_1")


        # Obtener conjuntos de TRAIN/DEV/TEST
        # ---------------------------------------------------------------------------------------------------------------

        self.printG("Generando conjuntos finales...")

        TRAIN_v1 = NEW_TRAIN
        TRAIN_v2 = TRAIN.loc[(TRAIN.num_images>0)] #Tiene que ser TRAIN, dado que no queremos los items de relleno o nuevos

        DEV = NEW_DEV
        DEV_v2 = NEW_DEV

        TEST = NEW_TEST
        TEST_v2 = NEW_TEST

        # -------------------------------------------------------------------------------------------------------------------
        # Añadir vectores de imágenes

        TRAIN_v1['vector'] = 0
        TRAIN_v2 = IMG.merge(TRAIN_v2, left_on='review', right_on='reviewId', how='inner')

        DEV_v2 = IMG.merge(DEV_v2, left_on='review', right_on='reviewId', how='inner')
        TEST_v2 = IMG.merge(TEST_v2, left_on='review', right_on='reviewId', how='inner')

        DEV_v2 = DEV_v2.drop(columns=['restaurantId', 'url', 'text',  'title', 'date', 'images', 'num_images', 'rating', 'language', 'review', 'image'])
        TEST_v2 = TEST_v2.drop(columns=['restaurantId', 'url', 'text', 'title', 'date', 'images', 'num_images', 'rating', 'language',  'review', 'image'])

        TRAIN_v1 = TRAIN_v1.drop(columns=['restaurantId', 'userId', 'url', 'text',  'title', 'date', 'images', 'num_images', 'rating', 'language'])
        TRAIN_v2 = TRAIN_v2.drop(columns=['restaurantId', 'userId', 'url', 'text',  'title', 'date', 'images', 'num_images', 'rating', 'language', 'review', 'image'])

        self.printW("Las reviews v2 salen repetidas en función del número de imagenes")


        IMG_2 = np.row_stack(IMG.vector.values)
        M_IMG = np.mean(IMG_2, axis=0)

        IMG_2 = np.apply_along_axis(lambda x: np.power(x - M_IMG, 2), 1, IMG_2)
        IMG_2 = np.apply_along_axis(lambda x: np.mean(x), 1, IMG_2)
        MeanMSE = np.apply_along_axis(lambda x: np.mean(x), 0, IMG_2)
        MaxMSE = np.apply_along_axis(lambda x: np.max(x), 0, IMG_2)
        MinMSE = np.apply_along_axis(lambda x: np.min(x), 0, IMG_2)

        # Igualar TrainV2 a TrainV1
        # ---------------------------------------------------------------------------------------------------------------
        '''
        def myfn(data): return (pd.Series({"len": len(data)}))
        TST = TRAIN_v2.groupby(['id_user', 'id_restaurant']).apply(myfn).reset_index()
        print(np.mean(TST.len))
        '''

        left_items = len(TRAIN_v1)-len(TRAIN_v2)
        sampled_items = TRAIN_v2.sample(left_items, replace=True,random_state=self.SEED)
        TRAIN_v2 = TRAIN_v2.append(sampled_items, ignore_index=True)

        # Crear TRAINv3
        # ---------------------------------------------------------------------------------------------------------------

        #restaurante -> Review
        INXS = RVW[["id_restaurant","reviewId"]].drop_duplicates()
        IMGS = IMG.merge(INXS, left_on='review', right_on='reviewId', how='inner')

        # Coger los reales del TRAINv1 y añadirles imagen
        TRAIN_v3 = TRAIN_v1.loc[TRAIN_v1.reviewId>=0]
        TRAIN_v3 = TRAIN_v3.drop(columns=['vector'])
        TRAIN_v3 = IMG.merge(TRAIN_v3, left_on='review', right_on='reviewId', how='right')
        TRAIN_v3.vector = TRAIN_v3.vector.fillna(0)

        def add_img(data):
            id = data.id_restaurant.values[0]
            rst_imgs = IMGS.loc[IMGS.id_restaurant == id]

            if (len(rst_imgs) == 0):
                data['vector'] = 0
            else:
                imgs = rst_imgs.sample(len(data), replace=True)
                data['vector'] = imgs['vector'].values

            return data

        #[DESCARTADO] Añadir imagen a los falsos del v1
        #TV3_tmp = TRAIN_v1.loc[TRAIN_v1.reviewId < 0].groupby("id_restaurant").apply(add_img)

        #Añadir los falsos del v1
        TV3_tmp = TRAIN_v1.loc[TRAIN_v1.reviewId < 0]
        TRAIN_v3 = TRAIN_v3.append(TV3_tmp, ignore_index = True).reset_index()

        TRAIN_v3["image"] = ~TRAIN_v3.review.isnull()
        TRAIN_v3 = TRAIN_v3.drop(columns=['review'])

        #Cambiar los 0's por vectores
        def fill_zeros(data):
            if(type(data) is int):return np.zeros(len(IMG.iloc[0].vector))
            else: return data

        TRAIN_v3['vector'] = TRAIN_v3.vector.apply(fill_zeros)

        # Hacer DEV_v3 y TEST_v3
        # --------------------------------------------------------------------------------------------------------------
        # restaurante -> Review
        INXS = RVW[["id_restaurant","id_user", "reviewId","like"]].drop_duplicates()
        IMGS = IMG.merge(INXS, left_on='review', right_on='reviewId', how='inner')

        #Obtener los restaurantes con 5 o más imágenes en total.
        RST_DEV = RVW.groupby("id_restaurant").apply(lambda x: pd.Series({"id_restaurant": x.id_restaurant.values[0], "imgs": sum(x.num_images.values)}))
        RST_DEV = RST_DEV.loc[RST_DEV.imgs >= 5, "id_restaurant"].values

        #De los que tienen imágen del DEV, obtener los que están el la lista anterior
        DEV_v3 = DEV_v2.loc[DEV_v2.id_restaurant.isin(RST_DEV)]

        #De los que tienen imágen del TEST, obtener los que están el la lista anterior
        TEST_v3 = TEST_v2.loc[TEST_v2.id_restaurant.isin(RST_DEV)]

        # Añadir el resto de imágenes del restaurante
        def addImgRest(data):
            id_r = data.id_restaurant.values[0]
            id_u = data.id_user.values[0]
            id_rev = data.reviewId.values[0]

            tmp = IMGS.loc[IMGS.id_restaurant == id_r]
            tmp['is_dev'] = 0
            tmp['like'] = 0
            tmp['id_user'] = id_u

            tmp.loc[tmp.reviewId.isin(data.reviewId.values), "is_dev"] = 1

            tmp['reviewId'] = id_rev

            tmp = tmp.drop(['review'], axis=1)

            return tmp

        #Finalmente, añadir al conjunto todas las imágenes del restaurante, indicando cuales son las de DEV
        DEV_v3 = DEV_v3.groupby("reviewId").apply(addImgRest).reset_index(drop=True)
        DEV_v3 = DEV_v3[['id_user', 'id_restaurant', 'reviewId', 'is_dev', 'vector','like']]

        #Finalmente, añadir al conjunto todas las imágenes del restaurante, indicando cuales son las de DEV
        TEST_v3 = TEST_v3.groupby("reviewId").apply(addImgRest).reset_index(drop=True)
        TEST_v3 = TEST_v3[['id_user', 'id_restaurant', 'reviewId', 'is_dev', 'vector','like']]

        # Crear TRAIN_V3_1 y DEV_V3_1
        # --------------------------------------------------------------------------------------------------------------

        #Utilizar los ejemplos de TRAIN que posean imágenes
        TRAIN_v3_1 = TRAIN_v3.loc[TRAIN_v3.image==True]

        #Añadir los positivos de DEV
        DVP = DEV_v3.loc[DEV_v3.is_dev == 1, ['index', 'id_restaurant', 'like','id_user', 'reviewId', 'vector']]
        DVP['like'] =1
        TRAIN_v3_1 = TRAIN_v3_1.append(DVP)

        #Separar en TRAIN y DEV
        T, D = train_test_split(TRAIN_v3_1, test_size=0.3, random_state=self.SEED)
        TRAIN_v3_1 = T
        DEV_v3_1 = D

        #Triplicar los negativos de TRAIN
        TRAIN_v3_1 = TRAIN_v3_1.append(TRAIN_v3_1.loc[TRAIN_v3_1.like==0])
        TRAIN_v3_1 = TRAIN_v3_1.append(TRAIN_v3_1.loc[TRAIN_v3_1.like==0])

        TRAIN_v3_1 = TRAIN_v3_1.sample(frac=1)
        DEV_v3_1 = DEV_v3_1.sample(frac=1)

        # Crear TRAIN_V4 DEV_V4 y TEST_V4
        # --------------------------------------------------------------------------------------------------------------

        IMGS = IMG.merge(RVW[["id_restaurant", "id_user", "reviewId"]], left_on='review', right_on='reviewId',
                         how='inner')
        TRAIN_v4 = modelv4.TRAIN_V3.loc[modelv4.TRAIN_V3.image == True]

        def myfn(data):

            neg = 5

            id_r = data.id_restaurant.values[0]
            id_u = data.id_user.values[0]
            id_rv = data.reviewId.values[0]

            img_best = data.vector.values[0]

            images = IMGS.loc[(IMGS.id_restaurant == id_r) & (IMGS.id_user != id_u), "vector"]

            if (len(images) == 0): return

            images = np.row_stack(images.values)

            dists = scipy.spatial.distance.cdist([img_best], images, "euclidean")
            indx = np.argsort(dists)[0][-neg:]
            img_worst = images[indx, :]

            ret = pd.DataFrame([data.squeeze()] * len(img_worst))
            ret["worst"] = img_worst.tolist()
            ret = ret.drop(columns=["item", "image", "like"])

            return (ret)

        TRAIN_v4 = TRAIN_v4 = TRAIN_v4.drop(columns=["index"]).reset_index(drop=True)
        TRAIN_v4["item"] = TRAIN_v4.index
        TRAIN_v4 = TRAIN_v4.groupby("item").apply(myfn).reset_index(drop=True)

        # MEZCLAR DATOS ------------------------------------------------------------------------------------------------

        TRAIN_v1 = utils.shuffle(TRAIN_v1, random_state=self.SEED).reset_index(drop=True)
        TRAIN_v2 = utils.shuffle(TRAIN_v2, random_state=self.SEED).reset_index(drop=True)
        TRAIN_v3 = utils.shuffle(TRAIN_v3, random_state=self.SEED).reset_index(drop=True)
        TRAIN_v3_1 = utils.shuffle(TRAIN_v3_1, random_state=self.SEED).reset_index(drop=True)
        TRAIN_v4 = utils.shuffle(TRAIN_v4, random_state=self.SEED).reset_index(drop=True)

        # ALMACENAR PICKLE ------------------------------------------------------------------------------------------------

        os.makedirs(file_path)

        self.toPickle(file_path, "TRAIN_v1", TRAIN_v1)
        self.toPickle(file_path, "TRAIN_v2", TRAIN_v2)
        self.toPickle(file_path, "TRAIN_v3", TRAIN_v3)
        self.toPickle(file_path, "TRAIN_v3_1", TRAIN_v3_1)
        self.toPickle(file_path, "TRAIN_v4", TRAIN_v4)

        self.toPickle(file_path, "DEV", DEV)
        self.toPickle(file_path, "DEV_v2", DEV_v2)
        self.toPickle(file_path, "DEV_v3", DEV_v3)
        self.toPickle(file_path, "DEV_v3_1", DEV_v3_1)

        self.toPickle(file_path, "TEST", TEST)
        self.toPickle(file_path, "TEST_v2", TEST_v2)
        self.toPickle(file_path, "TEST_v3", TEST_v3)

        self.toPickle(file_path, "REST_TMP", len(REST_TMP))
        self.toPickle(file_path, "USR_TMP", len(USR_TMP))
        self.toPickle(file_path, "IMG", len(IMG.iloc[0].vector))
        self.toPickle(file_path, "MSE", [MinMSE, MaxMSE, MeanMSE])

        return (TRAIN_v1, TRAIN_v2,TRAIN_v3,TRAIN_v3_1,TRAIN_v4, DEV,DEV_v2,DEV_v3,DEV_v3_1, TEST,TEST_v2,TEST_v3 ,len(REST_TMP), len(USR_TMP), len(IMG.iloc[0].vector), [MinMSE, MaxMSE, MeanMSE])

    def train(self):
        raise NotImplementedError

    def dev(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def gridSearchPrint(self,epoch,train,dev):
        raise NotImplementedError

    def gridSearch(self, params, max_epochs = 50, start_n_epochs = 5, last_n_epochs = 5):

        def createCombs():

            def flatten(lst):
                return sum(([x] if not isinstance(x, list) else flatten(x)
                            for x in lst), [])

            combs = []
            level = 0
            for v in params.values():
                if (len(combs)==0):
                    combs = v
                else:
                    combs = list(it.product(combs, v))
                level+=1

                if(level>1):
                    for i in range(len(combs)):
                        combs[i] = flatten(combs[i])

            return pd.DataFrame(combs, columns=params.keys())

        def configNet(comb):
            comb.pop("Index")

            for k in comb.keys():
                assert (k in self.CONFIG.keys())
                self.CONFIG[k]=comb[k]

        #-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#

        combs = createCombs()
        self.printW("Existen "+str(len(combs))+" combinaciones posibles")

        for c in combs.itertuples():

            stop_param = []

            c = dict(c._asdict())

            #Configurar la red
            configNet(c)

            #Crear el modelo
            self.MODEL = self.getModel()

            #Imprimir la configuración
            self.printConfig(filter=c.keys())

            #Configurar y crear sesion
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            with tf.Session(graph=self.MODEL, config=config) as session:

                #Almacenar session
                self.SESSION = session

                #Inicializar variables
                init = tf.global_variables_initializer()
                init.run(session=self.SESSION)

                #Para cada epoch...
                for e in range(max_epochs):

                    #Almacenar parámetros del GS
                    self.CURRENT_EPOCH = e

                    #TRAIN
                    train_ret = self.train()

                    #DEV
                    dev_ret, stop = self.dev()

                    stop_param.append(stop)

                    #Imprimir linea
                    self.gridSearchPrint(e,train_ret,dev_ret)

                    # Si en las n epochs anteriores la pendiente es menor que valor, parar
                    if (len(stop_param) >= start_n_epochs + last_n_epochs):
                        slope = self.getSlope(stop_param[-last_n_epochs:]);
                        if (slope > self.CONFIG['gs_max_slope']): break

    def finalTrain(self, epochs = 1):

        # Unir Train y DEV (SOLO LOS REALES, TODOS CAMBIARÍAN MUCHO EL TRAIN)

        self.TRAIN_V1 = self.TRAIN_V1.append(self.DEV.loc[self.DEV.like==1], sort=False)
        self.TRAIN_V2 = self.TRAIN_V2.append(self.DEV_V2.loc[self.DEV_V2.like==1], sort=False)
        self.DEV = None
        self.DEV_V2 = None

        # Configurar y crear sesion
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(graph=self.MODEL, config=config) as session:
            # Almacenar session
            self.SESSION = session

            # Inicializar variables
            init = tf.global_variables_initializer()
            init.run(session=self.SESSION)

            # Para cada epoch...
            for e in range(epochs):
                print(str(e)+"/"+str(epochs))

                # TRAIN
                train_ret = self.train()


            #Test final
            test_ret = self.test()

            # Imprimir linea
            self.gridSearchPrint(epochs,train_ret, test_ret)

    #- STATS -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

    def getDataStats(self):

        print("NO UTILIZAR; DESACTUALIZADO")

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

    def getTailGraph(self):

        file = open("tail_graph_"+self.CITY.lower()+".tsv", "w")

        RVW, IMG, USR_TMP, REST_TMP = self.getFilteredData();

        def count(data): return(pd.Series({'reviews':len(data)}))
        RES = RVW.groupby('restaurantId').apply(count)
        RES = RES.sort_values("reviews", ascending=False).reset_index()

        tot_rest = len(RES)
        tot_revs = sum(RES.reviews.values)

        items = 0
        revs = 0

        file.write("% REVIEWS\t% REST\tREVIEWS\tREST\n")

        for i, r in RES.iterrows():
            items+=1
            revs+=r.reviews

            file.write(str(revs/tot_revs)+"\t"+str(items/tot_rest)+"\t"+str(revs)+"\t"+str(items)+"\n")

        file.close()

        return None

    def getUsersGraph(self):

        file = open("users_graph_"+self.CITY.lower()+".tsv", "w")

        RVW, IMG, USR_TMP, REST_TMP = self.getFilteredData();

        def myfn(r):
            pos = sum(r.like)
            neg = len(r) - pos

            return pd.Series({"pos": pos, "neg": neg, "total": pos + neg})

        RET = RVW.groupby("userId").apply(myfn)
        RET = RET.sort_values("total",ascending=False).reset_index()

        file.write("ID\tPOS\tNEG\tTOTAL\n")

        id = 0
        for i,d in RET.iterrows():
            file.write(str(id)+"\t"+str(d.pos)+"\t"+str(d.neg)+"\t"+str(d.total)+"\n")
            id+=1

        file.close()

    def intraRestImage(self):
        '''
        Calcula la imágen centroide de cada restaurante (los de más de 10 fotos).
        Calcula las distancias de todos con todos y retorna los restaurantes con los centroides más parecidos (menos distantes)
        :return:
        '''

        def myfn2(data):
            rst_imgs = np.unique(np.row_stack(data.vector), axis=0)

            if (len(rst_imgs) >= 10):
                cnt = np.mean(rst_imgs, axis=0)
                return pd.Series({"vector": cnt})

        RVW, _,_,_ = self.getFilteredData()

        RVW['url_rst'] = RVW.url.apply(lambda x: re.sub(r"\-r\d+", "", x))

        rt = self.TRAIN_V2.groupby(['id_restaurant']).apply(myfn2).reset_index()
        rt = rt.loc[~rt.vector.isna()]

        matrix = np.row_stack(rt.vector)
        dists_v = scipy.spatial.distance.pdist(matrix)

        min_d = np.sort(dists_v)[0]

        dists = scipy.spatial.distance.squareform(dists_v)
        i, j = np.where(dists == min_d)

        rids = rt.iloc[i, :].id_restaurant.values

        urls = np.unique(RVW.loc[RVW.id_restaurant.isin(rids)].url_rst.values)

        for u in urls: print(u)

    def imageDistance(self):

        def saveImage(path, img_src):
            # si está descargada, skip
            if (os.path.isfile(path)): return True

            gcontext = ssl.SSLContext(ssl.PROTOCOL_TLSv1)  # Only for gangstars

            try:
                a = urllib.request.urlopen(img_src, context=gcontext)
            except:
                return False

            if (a.getcode() != 200):
                return False

            try:
                f = open(path, 'wb')
                f.write(a.read())
                f.close()
                return path
            except Exception as e:
                print(e)
                return False

        RVW, IMG,_,_ = self.getFilteredData()
        RVW = RVW.loc[RVW.num_images > 0]
        RVW = IMG.merge(RVW, left_on='review', right_on='reviewId', how='inner')

        RVW = RVW.sample(len(RVW)).reset_index()
        matrix = np.row_stack(RVW.vector)

        X_embedded = TSNE(n_components=2,verbose=1).fit_transform(matrix)

        Clusters = 200

        kmeans = MiniBatchKMeans(n_clusters=Clusters,random_state = self.SEED, verbose=1,batch_size=512)
        model = kmeans.fit(X_embedded)
        RVW['cluster'] = model.labels_

        fig = plt.figure()
        #ax = Axes3D(fig)

        mx_x, mx_y = np.max(X_embedded,0)
        mn_x, mn_y = np.min(X_embedded,0)

        s1 = (mx_x - mn_x) / (np.sqrt(Clusters) * 2)
        s2 = (mx_y - mn_y) / (np.sqrt(Clusters) * 2)

        plt.xlim(mn_x, mx_x+s1)
        plt.ylim(mn_y-s2, mx_y)

        #ax.scatter3D(X_embedded[:,0],X_embedded[:,1],X_embedded[:,2], c=model.labels_)
        #img = mpimg.imread('stinkbug.png')

        ret = pd.DataFrame(columns = ["pos","url"])

        for i,g in RVW.groupby('cluster'):
            indxs = g.index.values

            tsne = X_embedded[indxs,:]
            tsne = np.mean(tsne, axis=0)

            img = g.sample(1)
            url = img.images.values[0][img.image.values[0]-1]['image_url_lowres']

            img_name = hashlib.md5(str(url).encode('utf-8')).hexdigest()
            path = "tmp_img/" + str(img_name) + ".jpg"
            saveImage(path, url)

            ret = ret.append({"pos":tsne,"url":path},ignore_index=True)

            img = matplotlib.pyplot.imread(path)
            #plt.imshow(img,extent=(tsne[0]+2,tsne[0],tsne[1],-tsne[1]+2),origin="lower")
            # (left, right, bottom, top)


            plt.imshow(img,extent=(tsne[0],tsne[0]+s1,tsne[1]-s2,tsne[1]),origin="upper")

        fig1 = plt.gcf()
        #plt.show()
        #plt.draw()
        fig1.savefig('T-SNE.pdf', dpi=500)

        exit()

        matplotlib.pyplot.imread()


        plt.scatter(X_embedded[:,0],X_embedded[:,1], c=model.labels_)
        plt.show()

        exit()

        print()



        cluster_data = RVW.loc[RVW.cluster == 0]

        for i,d in cluster_data.iterrows():
            url = d.images[d.image-1]['image_url_lowres']

            img_name = hashlib.md5(str(url).encode('utf-8')).hexdigest()
            saveImage("tmp_img/"+str(img_name)+".jpg",url)

        print(kmeans)

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

    def getTopN(self, results, data):

        users = data.id_user.values
        likes = data.like.values

        results = utils.shuffle(results) #Para evitar que el 1 sea el primero cuando todos son 0.5

        results = results.sort_values(['id_user', 'prediction'], ascending=[True, True]).reset_index(drop=True)

        def getMultipleHits(data):

            data = data.reset_index()

            ones = data.loc[(data.like == 1)]
            zeros = data.loc[(data.like == 0)]

            columns = list(map(lambda x: "hit_"+str(x),self.CONFIG['top']))
            columns.append("real_pos")
            ret = pd.DataFrame( columns=columns)


            for i,p in ones.iterrows():
                #tmp = zeros.prediction.values
                #tmp = -np.sort(-np.append(tmp,p.prediction))
                #real_pos, = np.where(tmp == p.prediction)


                real_pos = np.searchsorted(zeros.prediction.values,p.prediction)
                real_pos= self.CONFIG['top_n_new_items']-(real_pos-1)

                tmp = {}

                for t in self.CONFIG['top']:
                    hit = 1 if (real_pos <= t) else 0
                    tmp['hit_' + str(t)] = hit

                tmp['real_pos'] = real_pos

                ret = ret.append(tmp, ignore_index=True)

                #print(data.id_user.values[0],real_pos)



            return ret

        #Si solo hay un 1 positivo por usuario en DEV/TEST
        if(self.CONFIG['train_pos_rate']==1.0):
            results['local_indx'] = list(range(1, 101)) * sum(likes)
            like_pos = results.loc[results.like == 1].copy()
            like_pos['real_pos'] = 101 - like_pos['local_indx']

            results = like_pos[['real_pos']].copy()

            for t in self.CONFIG['top']:
                results['hit_' + str(t)] = results['real_pos'] <= t;
                results['hit_' + str(t)] = results['hit_' + str(t)].astype(int)

            res = (results.iloc[:, 1:].sum() / sum(likes)) * 100.0

        #Si solo hay más de 1 positivo por usuario en DEV/TEST
        else:
            #Mejorable??
            results = results.groupby('id_user').apply(getMultipleHits).reset_index(drop=True)
            res= (results.iloc[:,0:-1].sum()/sum(likes))*100.0

        #recall = hits / sum(likes)
        #precision = recall / self.CONFIG['top']

        avg_pos = np.mean(results.real_pos.values)
        median_pos = np.median(results.real_pos.values)

        return res.to_dict(),avg_pos,median_pos

    def getConfMatrix(self, pred, real, title ="", verbose=True):

        f = np.vectorize(lambda x: 1 if(x<.5) else 0)
        pred_tmp = f(np.array(pred[:,0]))

        self.printE("REVISAR")
        exit()

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

    def matrixToImage(self, matrix, path = "matrix"):

        w, h = matrix.shape

        matrix = np.abs(matrix)
        max_v = np.max(matrix)
        min_v = np.min(matrix)

        matrix = (matrix-min_v) / (max_v-min_v)
        matrix = matrix * 255

        img = Image.fromarray(matrix)
        img = img.convert('L')
        img.save(path+".png")

    def toPickle(self,path,name,data):
        with open(path+name, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def getPickle(self,path,name):
        with open(path+name, 'rb') as handle:
            data = pickle.load(handle)
        return data

    def printConfig(self, filter=[]):

        tmp = self.CONFIG
        tmp['seed']=self.SEED
        tmp['city']=self.CITY

        print("-" * 50)

        print(hashlib.md5(str(tmp).encode('utf-8')).hexdigest())

        print("-" * 50)

        for key, value in tmp.items():

            line = bcolors.BOLD + key + ": " + bcolors.ENDC + str(value)

            if(len(filter)>0):
                if(key in filter):
                    print(line)

            else:
                print(line)


        print("-"*50)

    def printE(self,text):
        print(bcolors.FAIL+str("[ERROR] ")+str(text)+bcolors.ENDC)

    def printW(self,text):
        print(bcolors.WARNING+str("[AVISO] ")+str(text)+bcolors.ENDC)

    def printG(self,text):
        print(bcolors.OKGREEN+str("[INFO] ")+str(text)+bcolors.ENDC)

    def printB(self,text, bold=False):
        if(bold):
            print(bcolors.BOLD+bcolors.OKBLUE+str(text)+bcolors.ENDC+bcolors.ENDC)
        else:
            print(bcolors.OKBLUE + str(text) + bcolors.ENDC)