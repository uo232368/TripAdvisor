# -*- coding: utf-8 -*-

import argparse

from ModelV3 import *
from ModelV2 import *


########################################################################################################################

# Obtener argumentos por linea de comandos
parser = argparse.ArgumentParser()
parser.add_argument('-m', type=int,help="Modelo a utilizar")
parser.add_argument('-i', type=int,help="Codificación utilizada para las imágenes")
parser.add_argument('-s', type=int,help="Semilla")
parser.add_argument('-c', type=str,help="Ciudad", )
parser.add_argument('-gpu', type=str,help="Gpu")

args = parser.parse_args()

model= 2 if args.m == None else args.m
option = 2 if args.i == None else args.i
seed = 100 if args.s == None else args.s
city = "Barcelona" if args.c == None else args.c
gpu = 0 if args.gpu == None else args.gpu

os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)

config = {"min_revs":5,
          "emb_size":32,
          "learning_rate":0.0001,
          "lr_decay":0.0,
          "batch_size":512,
          "epochs":10,
          "c_loss":.5,
          "gs_max_slope":-1e-8}

########################################################################################################################

if (model == 2):

    params = {
        "learning_rate": [1e-01, 1e-03, 1e-05, 1e-07,1e-09],
    }

    modelv2 = ModelV2(city=city, option=option, config=config, seed=seed)
    modelv2.gridSearchV1(params, max_epochs=500)

if (model == 3):

    params = {
        "learning_rate": [1e-01, 1e-03, 1e-05, 1e-07,1e-09],
        "emb_size": [128,256,512,1024],
        "tests":5
    }

    modelv3 = ModelV3(city=city, option=option, config=config, seed=seed)
    modelv3.randomSearchV1(params, max_epochs=500)

#ToDo: Test emb imagen con hilos == sin hilos


