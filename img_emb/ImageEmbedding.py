# -*- coding: utf-8 -*-

from Option2 import Option2
from Option2 import Option2Helper

import time
import pandas as pd
import keras
import tensorflow as tf
from keras.models import Model
from keras.preprocessing import image
from keras.backend.tensorflow_backend import set_session
from keras.applications.imagenet_utils import decode_predictions
#-----------------------------------------------------------------------------------------------------------------------

def waitForEnd(threads):

    stop=False
    while not stop:
        stop = True
        for i in threads:
            if(i.isAlive()):stop=False
        time.sleep(5)
    print("END")
    print("-"*50)

def loadNet(NET):
    # Utilizar la memoria necesaria
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    if (NET == "RES"):
        net = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None,
                                                   input_shape=None, pooling=None, classes=1000)
        inP = net.input
        ouT = net.get_layer('flatten_1').output
        net = Model(inputs=inP, outputs=ouT)

    if (NET == "RESV2"):
        net = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet',
                                                                       input_tensor=None, input_shape=None,
                                                                       pooling=None, classes=1000)
        inP = net.input
        ouT = net.get_layer('avg_pool').output
        net = Model(inputs=inP, outputs=ouT)

    net._make_predict_function()

    return net

def init():

    NET = "RESV2"
    CITY = "Barcelona"
    LOWRES= True

    #PATH = "../data/" + CITY.lower() + "_data/"
    PATH = "/media/HDD/pperez/TripAdvisor/" + CITY.lower() + "_data/"
    IMG_PATH = PATH + "images/"
    if(LOWRES):IMG_PATH = PATH+"images_lowres/"


    tst = pd.read_pickle(PATH+"img-option2-new.pkl")
    print(len(tst))
    exit()

    data = pd.read_pickle(PATH + "reviews.pkl")
    data["num_images"] = data.images.apply(lambda x: len(x))
    data = data.loc[data.num_images > 0]

    data = tst.merge(data, right_on="reviewId",left_on="review")
    data["img_url"] = data.apply(lambda x: x.images[x.image - 1]['image_url_lowres'], axis=1)

    print(tst)
    exit()
    

    '''

    n_threads = 25 #25
    threads = []

    data = pd.read_pickle(PATH+"reviews.pkl")
    data["num_images"] = data.images.apply(lambda x: len(x))
    data = data.loc[data.num_images > 0]

    #data=data.loc[data.restaurantId=="12874563"]

    len_data = len(data)
    len_data_thread = len_data // n_threads

    model = loadNet(NET)


    for i in range(n_threads):

        data_from = i * len_data_thread
        data_to = (i + 1) * len_data_thread
        if (i == n_threads - 1): data_to = len_data
        data_thread = data.iloc[data_from:data_to, :].reset_index(drop=True)

        temp_thread = Option2(i,"Name",i,model,path=PATH,image_path = IMG_PATH, city=CITY, net=NET, data=data_thread)
        threads.append(temp_thread)
        threads[i].start()

    waitForEnd(threads)

    OPH = Option2Helper()
    OPH.joinImages(PATH);

    pkl = pd.read_pickle(PATH+"img-option2-new.pkl")
    print(len(pkl))
    '''


if __name__ == "__main__":

    init()
