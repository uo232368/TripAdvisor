# -*- coding: utf-8 -*-

import argparse

from ModelV3 import *
from ModelV2 import *


########################################################################################################################

# Obtener argumentos por linea de comandos
parser = argparse.ArgumentParser()
parser.add_argument('-m', type=int,help="Modelo a utilizar")
parser.add_argument('-d', type=float,help="DropOut")
parser.add_argument('-i', type=int,help="Codificación utilizada para las imágenes")
parser.add_argument('-s', type=int,help="Semilla")
parser.add_argument('-c', type=str,help="Ciudad", )
parser.add_argument('-gpu', type=str,help="Gpu")
parser.add_argument('-over', type=str,help="Oversampling clase 0 ['none' (desactivar), 'auto' (clase 0 y 1 equilibradas) , '2' (duplicar clase 0), '3' (triplicar clase 0) ...]")

args = parser.parse_args()

model= 2 if args.m == None else args.m
option = 2 if args.i == None else args.i
seed = 100 if args.s == None else args.s
city = "Barcelona" if args.c == None else args.c
gpu = 0 if args.gpu == None else args.gpu
dpout = 0.5 if args.d == None else args.d
over = 'none' if args.over == None else args.over

os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)

config = {"min_revs":5,
          "emb_size":32,
          "learning_rate":1e-3,
          "lr_decay":0.0,
          "c_loss": .5,
          "batch_size":128,
          "epochs":1,
          "oversampling":over, #None (desactivar), "auto"(clase 0 y 1 equilibradas) , 2 (duplicar clase 0), 3 (triplicar clase 0) ...
          "dropout":dpout, # Prob de eliminar no de mantener
          "gs_max_slope":1e-8}

########################################################################################################################


if (model == 2):

    params = {
        #"learning_rate": [1e-05, 1e-06, 1e-07, 1e-08],
        "learning_rate": [1e-05],
    }

    modelv2 = ModelV2(city=city, option=option, config=config, seed=seed)
    modelv2.gridSearchV1(params, max_epochs=500)

if (model == 3):

    params = {
        #"learning_rate": [1e-03, 1e-04, 1e-05, 1e-06],
        #"emb_size": [128,256,512,1024],
        "learning_rate": [1e-05],
        "emb_size": [1024]
    }

    modelv3 = ModelV3(city=city, option=option, config=config, seed=seed)
    modelv3.gridSearchV1(params, max_epochs=500)

#ToDo: Test emb imagen con hilos == sin hilos
#ToDo: BatchNormalization Layer
#ToDo: AUC binario en TRAIN Y DEV

