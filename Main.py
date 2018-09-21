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

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.dotLosses = []
        self.imgLosses = []

    def on_batch_end(self, batch, logs={}):
        self.dotLosses.append(logs.get('dotprod_loss'))
        self.imgLosses.append(logs.get('output_image_loss'))

def printW(text):
    print(bcolors.WARNING+str(text)+bcolors.ENDC)

def getData(CITY,PATH,IMG_PATH, option,seed):

    USR_TMP = pd.DataFrame(columns=["real_id","id_user"])
    REST_TMP = pd.DataFrame(columns=["real_id","id_restaurant"])

    IMG = pd.read_pickle(PATH + "img-option"+str(option)+".pkl")

    RVW = pd.read_pickle(PATH + "reviews.pkl")
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

    printW(str(len(ids))+" usuarios con una review de una imagen.")

    RET_ONEONE = RET.loc[RET.id_user.isin(ids)]
    RET = RET.loc[~RET.id_user.isin(ids)]

    # -------------------------------------------------------------------------------------------------------------------
    # TRAIN/DEV/TEST

    TRAIN_PROP = 0.6 - (len(RET_ONEONE)/len(RVW))
    DEV_PROP = (1-TRAIN_PROP)/2
    TEST_PROP = (1-TRAIN_PROP)/2

    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(RET.iloc[:, :-1], RET.iloc[:, -1],test_size=TEST_PROP, random_state=seed)
    X_train, X_dev, Y_train, Y_dev = sklearn.model_selection.train_test_split(X_train, Y_train, test_size=DEV_PROP,random_state=seed)

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
    TRAIN_ZRO_SMPLE = TRAIN_ZRO.sample(len(TRAIN_ONE)-len(TRAIN_ZRO), replace=True, random_state=seed)
    TRAIN = TRAIN.append(TRAIN_ZRO_SMPLE,ignore_index=True)

    printW("-"*80)
    printW("TRAIN: "+str(len(TRAIN)))
    printW("DEV: "+str(len(DEV)))
    printW("TEST: "+str(len(TEST)))
    printW("-"*80)

    #-------------------------------------------------------------------------------------------------------------------

    TRAIN = IMG.merge(TRAIN, left_on='review', right_on='reviewId', how='inner')
    DEV = IMG.merge(DEV, left_on='review', right_on='reviewId', how='inner')
    TEST = IMG.merge(TEST, left_on='review', right_on='reviewId', how='inner')

    TRAIN = TRAIN.drop(columns=['restaurantId','userId','url','text','title','date','real_id_x','real_id_y','images','num_images','index','rating','language','review'])
    DEV = DEV.drop(columns=['restaurantId','userId','url','text','title','date','real_id_x','real_id_y','images','num_images','index','rating','language','review'])
    TEST = TEST.drop(columns=['restaurantId','userId','url','text','title','date','real_id_x','real_id_y','images','num_images','index','rating','language','review'])

    printW("AVISO: las reviews salen repetidas en función del número de imagenes")
    printW("-"*80)

    IMG_2 = np.row_stack(IMG.vector.values)
    M_IMG = np.mean(IMG_2,axis=0)

    IMG_2 = np.apply_along_axis(lambda x:np.power(x-M_IMG,2),1,IMG_2)
    IMG_2 = np.apply_along_axis(lambda x:np.mean(x),1,IMG_2)
    MeanMSE = np.apply_along_axis(lambda x:np.mean(x),0,IMG_2)
    MaxMSE = np.apply_along_axis(lambda x:np.max(x),0,IMG_2)
    MinMSE = np.apply_along_axis(lambda x:np.min(x),0,IMG_2)

    return(TRAIN,DEV,TEST,len(REST_TMP),len(USR_TMP),len(IMG.iloc[0].vector), [MinMSE,MaxMSE,MeanMSE])

def getModel(N_RST,N_USR, V_IMG, config,MSE_DATA):

    def normalized_mse_loss(y_pred,y_real):
        minMSE = MSE_DATA[0]
        maxMSE = MSE_DATA[1]
        avgMSE = MSE_DATA[2]

        return tf.losses.mean_squared_error(y_pred,y_real)/(avgMSE)

    #-------------------------------------------------------------------------------------------------------------------

    num_users = N_USR
    emb_users = config["emb_size"]

    num_restaurants = N_RST
    emb_restaurants = config["emb_size"]

    bin_out = 1
    img_out = V_IMG

    model_path = config["model_path"]
    learning_rate = config["learning_rate"]
    lr_decay = config["lr_decay"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    c_loss = config["c_loss"]

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

    if (os.path.exists(model_path)): model.load_weights(model_path)

    #model.compile(optimizer=adam, loss=custom_loss_fn, loss_weights=[(1 - c_loss), c_loss])
    model.compile(optimizer=adam, loss=["binary_crossentropy",normalized_mse_loss], loss_weights=[(1 - c_loss), c_loss])

    # summarize layers
    # print(model.summary())
    plot_model(model, to_file='the_net.png')

    return model

########################################################################################################################

option = 2
config = {"emb_size":20,
          "learning_rate":0.00001,
          "lr_decay":0.0,
          "batch_size":1024,
          "epochs":100,
          "c_loss":.5}
seed = 100

city = "Gijon"
modelGijon = MainModel(city=city, option=option, config=config, seed=seed)
modelGijon.train(save=False, show_epoch_info=True)

exit()

seed = 2
np.random.seed(seed)
#tf.set_random_seed(seed)

# Utilizar la memoria necesaria
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

#-----------------------------------------------------------------------------------------------------------------------
CITY = "Gijon"
#CITY = "Barcelona"
#CITY = "Madrid"

OPTION = 2
PATH = "/mnt/hdd/pperez/TripAdvisor/" + CITY.lower() + "_data/"
IMG_PATH = PATH + "images/"

TRAIN,DEV,TEST, N_RST,N_USR, V_IMG, MSE_DATA = getData(CITY,PATH,IMG_PATH, option=OPTION,seed=seed)

config = {"model_path":"models/model_"+CITY.lower()+"_option"+str(OPTION),
          "emb_size":20,
          "learning_rate":0.00001,
          "lr_decay":0.0,
          "batch_size":1024,
          "epochs":100000,
          "c_loss":.5}

model = getModel(N_RST,N_USR, V_IMG, config,MSE_DATA)

#ToDo: LOS USUARIO DE 1 REVIEW CON 1 IMAGEN ESTÁN EN TRAIN
#ToDo: PRECISION/RECALL
#ToDo: Grid-Search
#ToDo: Test emb imagen con hilos == sin hilos

# Train
#-----------------------------------------------------------------------------------------------------------------------

oh_users = to_categorical(TRAIN.id_user,num_classes=N_USR)
oh_rests = to_categorical(TRAIN.id_restaurant,num_classes=N_RST)

y_likes = TRAIN.like.values
y_image = np.row_stack(TRAIN.vector.values)

# checkpoint
checkpoint = ModelCheckpoint(config['model_path'], verbose=0)
history = LossHistory()
callbacks_list = [checkpoint,history]

model.fit([oh_users,oh_rests], [y_likes,y_image], epochs=config['epochs'], batch_size=config['batch_size'],callbacks=callbacks_list,verbose=2)

loss = model.evaluate([oh_users,oh_rests],[y_likes,y_image], verbose=0)
print(loss)

# Dev
#-----------------------------------------------------------------------------------------------------------------------
DATA = DEV
oh_users = to_categorical(DATA.id_user,num_classes=N_USR)
oh_rests = to_categorical(DATA.id_restaurant,num_classes=N_RST)

y_likes = DATA.like.values
y_image = np.row_stack(DATA.vector.values)

loss = model.evaluate([oh_users,oh_rests],[y_likes,y_image], verbose=0)
print(loss)

y_likes_pred,y_image_pred = model.predict([oh_users,oh_rests],verbose=0)

aciertos = 0

for i in range(len(y_likes)):
    real = y_likes[i]
    pred = 1 if  y_likes_pred[i][0] > .5  else 0

    if(real == pred): aciertos+=1


print(aciertos/len(y_likes))



exit()
for i in range(2):
    real = y_image[i]
    pred = y_image_pred[i]
    print(real)
    print(pred)
    print()


