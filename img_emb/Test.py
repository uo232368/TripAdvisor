import os

import numpy as np
import pandas as pd
import time
import math

import keras
import tensorflow as tf
import scipy
from keras.models import Model
from keras.preprocessing import image
from keras.backend.tensorflow_backend import set_session
from keras.applications.imagenet_utils import decode_predictions


def loadNet(NET):
    # Utilizar la memoria necesaria

    config = tf.ConfigProto()

    #config.intra_op_parallelism_threads = 1
    #config.inter_op_parallelism_threads = 1

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

directory = "test_images"
NET = "RESV2"
CITY = "Barcelona"

directory = "/media/HDD-Nserver1/pperez/TripAdvisor/" + CITY.lower() + "_data/"
img_directory = directory +"images/"


for f in os.listdir(img_directory):
    print(f)



exit()

#os.environ["CUDA_VISIBLE_DEVICES"]=""

model = loadNet(NET)

compare = "lj-9b910215727cb989c1dabe6556400317"

names = []
res = []
best = []

for f in os.listdir(directory):

    img = image.load_img(directory+"/"+f, target_size=(299, 299))
    img = image.img_to_array(img)
    img = keras.applications.inception_resnet_v2.preprocess_input(np.expand_dims(img.copy(), axis=0))

    preds = model.predict(img)

    if(compare in f):
        best = preds[0]
    else:
        res.append(preds[0])
        names.append(f)

    print(f, "\t",preds[0])
    #if("91047416dd9de61f309e079a945203fe" in f):
    #print( len(preds[0]))


res = np.row_stack(res)
d = scipy.spatial.distance.cdist(res,[best]).reshape(-1,)
d_s = np.sort(d)
min = np.argsort(d)
names = np.array(names)[min]

for m in range(len(min)):
    print(compare,"\t",names[m],"\t",d_s[m])
