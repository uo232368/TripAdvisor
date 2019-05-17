# -*- coding: utf-8 -*-

import os

import threading

import numpy as np
import pandas as pd
import time
import math

import keras
import tensorflow as tf
from keras.models import Model
from keras.preprocessing import image
from keras.backend.tensorflow_backend import set_session
from keras.applications.imagenet_utils import decode_predictions

#-----------------------------------------------------------------------------------------------------------------------

class Option2Helper():

    def joinImages(self,path, name= "img.pkl"):

        ret = pd.DataFrame(columns=Option2.img_cols)

        for f in os.listdir(Option2.TMP_FOLDER):

            if(("images-" in f) and (".pkl" in f)):
                tmp_ret = pd.read_pickle(Option2.TMP_FOLDER+"/"+f)
                ret = ret.append(tmp_ret,ignore_index=True)

        if(len(ret)>0):pd.to_pickle(ret,path+name)

        #Eliminar los ficheros de la carpeta.
        for f in os.listdir(Option2.TMP_FOLDER):
            if(("images-" in f)  and (".pkl" in f)):
                os.remove(Option2.TMP_FOLDER+"/"+f)

class Option2(threading.Thread):

    CITY = ""
    NET = ""
    PATH = ""
    IMG_PATH = ""
    TMP_FOLDER = ".temp"
    DATA = None

    model = None

    img_cols = ["review","image","vector"]

    def __init__(self, threadID, name, counter,model, path, image_path,data,net="RESV2", city="Barcelona"):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

        self.NET = net
        self.CITY = city
        self.PATH = path
        self.IMG_PATH = image_path

        self.DATA = data

        self.model = model

    def loadImage(self, PATH):

        if (self.NET == "RES"):
            img = image.load_img(PATH, target_size=(224, 224))
            img = image.img_to_array(img)
            x = keras.applications.resnet50.preprocess_input(np.expand_dims(img.copy(), axis=0))

        if (self.NET == "RESV2" or self.NET=="FOOD"):
            img = image.load_img(PATH, target_size=(299, 299))
            img = image.img_to_array(img)
            x = keras.applications.inception_resnet_v2.preprocess_input(np.expand_dims(img.copy(), axis=0))

        return x

    def run(self):

        ret = pd.DataFrame(columns=self.img_cols)

        for j,r in self.DATA.iterrows():
            print("Thread "+str(self.threadID)+": "+str(j)+" de "+str(len(self.DATA)))
            id = r.reviewId
            imgs = r.images

            img_indx = 0
            for im in imgs:
                img_name = str(img_indx)+".jpg"
                img_path = self.IMG_PATH+id+"/"+img_name
                img_indx+=1

                img = self.loadImage(img_path)
                preds = self.model.predict(img)

                ret = ret.append({"review":id,"image":img_indx,"vector":preds[0]},ignore_index=True)

        pd.to_pickle(ret, self.TMP_FOLDER + "/images-" + str(self.threadID) + ".pkl")


########################################################################################################################


