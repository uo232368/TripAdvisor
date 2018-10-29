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

        file_path = self.PATH + "model_data"
        file_path += "_"+str(self.CONFIG['min_revs'])+"_"+str(self.CONFIG['min_pos_revs'])+"/"

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

        # Eliminar usuarios sin imágenes
        # ---------------------------------------------------------------------------------------------------------------

        RVW = RVW.loc[RVW.num_images>0]

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
        old_len = len(RVW)

        USR_LST = RVW.groupby("userId", as_index=False).sum()
        USR_LST = USR_LST.loc[(USR_LST.like >= self.CONFIG['min_pos_revs']), "userId"].values
        RVW = RVW.loc[RVW.userId.isin(USR_LST)]

        self.printW("Eliminado usuarios con menos de " + str(self.CONFIG['min_pos_revs']) + " valoraciones positivas.")


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

        # Mover ejemplos positivos a donde corresponde
        # ---------------------------------------------------------------------------------------------------------------
        TRAIN = pd.DataFrame()
        DEV = pd.DataFrame()
        TEST = pd.DataFrame()

        #GRP_USR = RVW.groupby(["userId"])

        POS_REVS = RVW.loc[(RVW.like==1)]
        POS_REVS = POS_REVS.sort_values(by=['userId'])

        POS_REVS['id_user_test'] = POS_REVS.id_user.shift(-1)
        POS_REVS.loc[POS_REVS.id_user_test.isnull(),"id_user_test"] = -1

        POS_REVS_TEST = POS_REVS.loc[(POS_REVS.id_user!=POS_REVS.id_user_test)].drop(columns=['id_user_test'])
        POS_REVS = POS_REVS.loc[(POS_REVS.id_user==POS_REVS.id_user_test)].drop(columns=['id_user_test'])

        POS_REVS['id_user_dev'] = POS_REVS['id_user'].shift(-1)
        POS_REVS.loc[POS_REVS['id_user_dev'].isnull(),"id_user_dev"] = -1

        POS_REVS_DEV = POS_REVS.loc[(POS_REVS['id_user']!=POS_REVS['id_user_dev'])].drop(columns=['id_user_dev'])
        POS_REVS = POS_REVS.loc[(POS_REVS['id_user']==POS_REVS['id_user_dev'])].drop(columns=['id_user_dev'])

        TRAIN = TRAIN.append(POS_REVS, ignore_index=True)
        DEV = DEV.append(POS_REVS_DEV, ignore_index=True)
        TEST = TEST.append(POS_REVS_TEST, ignore_index=True)

        # Mover ejemplos negativos a donde corresponde
        # ---------------------------------------------------------------------------------------------------------------

        NEG_REVS = RVW.loc[(RVW.like!=1)]

        TRAIN = TRAIN.append(NEG_REVS, ignore_index=True) #Todos a TRAIN

        # Crear ejemplos nuevos para compensar distribución de clases
        # ---------------------------------------------------------------------------------------------------------------

        N_NEW_ITEMS = len(POS_REVS)-len(NEG_REVS)
        N_USERS = len(USR_TMP)

        ITEMS_PER_USR = (N_NEW_ITEMS//N_USERS)
        ITEMS_LEFT = N_NEW_ITEMS-(ITEMS_PER_USR*N_USERS)

        if(ITEMS_LEFT>0):ITEMS_PER_USR+=1

        self.printW("Se añaden " + str(ITEMS_PER_USR) + " items nuevos por usuario. ("+str((ITEMS_PER_USR*N_USERS))+" en total)")
        self.printE("Sobran " + str(ITEMS_LEFT) + " items que no se añaden.")

        rest_ids = set(REST_TMP.id_restaurant)

        used_restaurants = {} #Contiene todos los pares usr restaurante ya ulitizados hasta el momento

        def append_no_reviewed_restaurants(data):
            no_reviewed = list(rest_ids.difference(set(data.id_restaurant.values)))
            no_reviewed = rn.sample(no_reviewed,ITEMS_PER_USR)
            
            used_restaurants[data.id_user.values[0]] = np.append(data.id_restaurant.values,no_reviewed)

            ret = pd.DataFrame(-1, index=np.arange(ITEMS_PER_USR), columns=data.columns)
            ret["id_user"]=data.id_user.values[0]
            ret["like"]=0
            ret["id_restaurant"]=no_reviewed
            ret = ret.drop(columns="userId")

            return ret

        NEW_REVS = RVW.groupby(['userId']).apply(append_no_reviewed_restaurants).reset_index()
        NEW_REVS = NEW_REVS.drop(columns="level_1")

        TRAIN = TRAIN.append(NEW_REVS, ignore_index=True, sort=True) #Todos a TRAIN

        # Añadir al conjunto de DEV los 1000 restaurantes no vistos
        # ---------------------------------------------------------------------------------------------------------------

        TOPN_NEW_ITEMS = 100;

        dev_used_restaurants = {} #Contiene todos los pares usr restaurante ya ulitizados hasta el momento

        def append_topn_items_dev(data):

            idUser = data.id_user.values[0]
            used_rests = used_restaurants[idUser]

            no_reviewed = list(rest_ids.difference(set(used_rests)))
            no_reviewed = rn.sample(no_reviewed, TOPN_NEW_ITEMS)

            dev_used_restaurants[idUser] = no_reviewed

            ret = pd.DataFrame(-1, index=np.arange(TOPN_NEW_ITEMS), columns=data.columns)
            ret["id_user"] = data.id_user.values[0]
            ret["like"] = 0
            ret["id_restaurant"] = no_reviewed
            ret = ret.drop(columns="userId")

            ret = ret.append(data.drop(columns=["userId"]),ignore_index=True)

            return ret

        NEW_DEV = DEV.groupby(['userId']).apply(append_topn_items_dev).reset_index()
        NEW_DEV = NEW_DEV.drop(columns="level_1")


        #Añadir los restaurantes usados en dev a la lista total
        for i in used_restaurants:used_restaurants[i] = np.append(used_restaurants[i],dev_used_restaurants[i])


        # Añadir al conjunto de TEST los 1000 restaurantes no vistos
        # ---------------------------------------------------------------------------------------------------------------


        def append_topn_items_test(data):

            idUser = data.id_user.values[0]
            used_rests = used_restaurants[idUser]

            no_reviewed = list(rest_ids.difference(set(used_rests)))
            no_reviewed = rn.sample(no_reviewed, TOPN_NEW_ITEMS)

            ret = pd.DataFrame(-1, index=np.arange(TOPN_NEW_ITEMS), columns=data.columns)
            ret["id_user"] = data.id_user.values[0]
            ret["like"] = 0
            ret["id_restaurant"] = no_reviewed
            ret = ret.drop(columns="userId")

            ret = ret.append(data.drop(columns=["userId"]),ignore_index=True)

            return ret


        NEW_TEST = TEST.groupby(['userId']).apply(append_topn_items_test).reset_index()
        NEW_TEST = NEW_TEST.drop(columns="level_1")


        # Obtener conjuntos de TRAIN/DEV/TEST
        # ---------------------------------------------------------------------------------------------------------------

        self.printG("Generando conjuntos finales...")

        TRAIN_v1 = TRAIN
        TRAIN_v2 = TRAIN.loc[(TRAIN.num_images>0)]
        DEV = NEW_DEV
        TEST = NEW_TEST

        # -------------------------------------------------------------------------------------------------------------------
        # Añadir vectores de imágenes

        TRAIN_v1['vector'] = 0

        TRAIN_v2 = IMG.merge(TRAIN_v2, left_on='review', right_on='reviewId', how='inner')


        ##DEV = IMG.merge(DEV, left_on='review', right_on='reviewId', how='inner')
        ##TEST = IMG.merge(TEST, left_on='review', right_on='reviewId', how='inner')

        TRAIN_v1 = TRAIN_v1.drop(
            columns=['restaurantId', 'userId', 'url', 'text', 'real_id_x', 'real_id_y', 'title', 'date', 'images',
                     'num_images', 'index', 'rating', 'language'])
        TRAIN_v2 = TRAIN_v2.drop(
            columns=['restaurantId', 'userId', 'url', 'text', 'real_id_x', 'real_id_y', 'title', 'date', 'images',
                     'num_images', 'index', 'rating', 'language', 'review', 'image'])

        #DEV = DEV.drop(columns=['restaurantId', 'userId', 'url', 'text', 'title', 'date', 'real_id_x', 'real_id_y', 'images','num_images', 'index', 'rating', 'language', 'review'])
        #TEST = TEST.drop(columns=['restaurantId', 'userId', 'url', 'text', 'title', 'date', 'real_id_x', 'real_id_y', 'images','num_images', 'index', 'rating', 'language', 'review'])

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
        #DEV = utils.shuffle(DEV, random_state=self.SEED)
        #TEST = utils.shuffle(TEST, random_state=self.SEED)

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

    def getTopN(self, model, top=10):

        pred_dev = self.dev(model)

        users = self.DEV.id_user.values
        likes = self.DEV.like.values

        results = pd.DataFrame(index=np.arange(len(pred_dev)),columns=["id_user","prediction","like"],)
        results["id_user"]= users
        results["prediction"]= pred_dev
        results["like"]= likes

        hits=0

        for us,gr in results.groupby('id_user'):
            sorted = gr.sort_values('prediction',ascending=False).reset_index(drop=True)

            if (len(sorted.loc[(sorted.index<top)&(sorted.like==1)])==1):
                hits+=1

        recall = hits / sum(likes)
        precision = recall / top

        return hits, precision, recall

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