# -*- coding: utf-8 -*-
import os, time
import warnings
import re
import sys
from tqdm import tqdm

import numpy as np
import pandas as pd

import pickle
import math
import signal, sys
import hashlib
import itertools as it
import urllib
import random

import random as rn
from PIL import Image

import keras
import tensorflow as tf
from keras import backend as K
from keras import losses
from keras.optimizers import Adam
from keras.utils import *
from keras.models import Model, Sequential
from keras.layers import Input,Dense,Activation,Concatenate, Dot,Conv2D,MaxPooling2D, Dropout
from keras.layers import Embedding, Flatten, GlobalAveragePooling2D,BatchNormalization,GaussianNoise
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard

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

    def __init__(self,city,option,config,name,date,seed = 2 ):

        signal.signal(signal.SIGINT, self.signal_handler)

        self.CITY = city
        self.OPTION = option
        self.PATH = "/media/HDD/pperez/TripAdvisor/" + self.CITY.lower() + "_data/"
        self.DATA_PATH = "/media/HDD/pperez/TripAdvisor/" + self.CITY.lower() + "_data/"+name.upper()+"/"

        self.IMG_PATH = self.PATH + "images_lowres/"
        self.DATE = date
        self.SEED = seed
        self.CONFIG = config
        self.MODEL_NAME = name

        #Eliminar info de TF
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['PYTHONHASHSEED'] = '0'
        #os.environ["CUDA_VISIBLE_DEVICES"] = ""

        warnings.filterwarnings('always')
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        #Fijar las semillas de numpy y TF
        np.random.seed(self.SEED)
        rn.seed(self.SEED)
        tf.set_random_seed(self.SEED)

        '''
        #Utilizar solo memoria GPU necesaria
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        #config.intra_op_parallelism_threads = 1
        #config.inter_op_parallelism_threads = 1

        sess = tf.Session(graph=tf.get_default_graph(), config=config)
        K.set_session(sess)
        '''

        self.printB("Obteniendo datos...")

        self.DATA = self.getData()

        self.printB("Creando modelo...")

        self.MODEL_PATH = "models/"+self.MODEL_NAME+"_" + self.CITY.lower() + "_option" + str(self.OPTION)
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
        self.printW("FN SIN IMPLEMENTAR")
        exit()

    def getData(self):
        self.printW("FN SIN IMPLEMENTAR")
        exit()

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
                        if (slope > self.CONFIG['gs_max_slope']):
                            print("-"*50)
                            print("MIN: "+str(np.min(stop_param))+"\t POS: "+str(np.argmin(stop_param)+1))
                            print("MAX: "+str(np.max(stop_param))+"\t POS: "+str(np.argmax(stop_param)+1))
                            print("-"*50)
                            break

    def finalTrain(self, epochs = 1):

        self.TRAIN = self.TRAIN_DEV
        self.DEV = self.TEST

        if(os.path.exists(self.MODEL_PATH)):
            self.printE("Ya existe el modelo. Utilizar predict")
            exit()

        self.MODEL = self.getModel()

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
                self.CURRENT_EPOCH = e
                print(str(e)+"/"+str(epochs))

                # TRAIN
                train_ret = self.train()

            #Guardar modelo
            #saver = tf.train.Saver(max_to_keep=1)
            #saver.save(session, save_path=self.MODEL_PATH)

            #Test final
            test_ret = self.dev()

            # Imprimir linea
            self.gridSearchPrint(epochs-1,train_ret, test_ret[0])

    def predict(self):

        if(not os.path.exists(self.MODEL_PATH)):
            self.printW("No existe un modelo guardado. Se necesita un 'finalTrain()' previo.")
            exit()

        self.DEV = self.TEST

        # Configurar y crear sesion
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.MODEL = self.getModel()

        with tf.Session(graph=self.MODEL, config=config) as session:
            # Restore variables from disk.
            saver = tf.train.Saver()
            saver.restore(session, self.MODEL_PATH)

            self.CURRENT_EPOCH=-1

            #Almacenar session
            self.SESSION = session

            # Test final
            test_ret = self.dev()

            # Imprimir linea
            self.gridSearchPrint(-1, -1, test_ret[0])

    #- STATS -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

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

        #RVW = RVW.sample(len(RVW)).reset_index()
        RVW = RVW.sample(20000).reset_index()

        matrix = np.row_stack(RVW.vector)

        X_embedded = TSNE(n_components=2,verbose=1).fit_transform(matrix)

        Clusters = 300

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

        #Si hay más de 1 positivo por usuario en DEV/TEST
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

    def plothist(self, data, column,title="Histogram",titleX="X Axis", titleY="Y Axis", bins=10, save=None):

        plt.ioff()

        items = bins

        plt.hist(data[str(column)], bins=range(1, items + 2), edgecolor='black',align="left")  # arguments are passed to np.histogram
        labels = list(map(lambda x: str(x),range(1, items + 1)))
        labels[-1] = "≥"+labels[-1]
        plt.xticks(range(1, items + 1),labels)
        plt.title(str(title))

        plt.xlabel(titleX)
        plt.ylabel(titleY)

        if(save is None):plt.show()
        else: plt.savefig(str(save) )

        plt.close()

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