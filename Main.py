# -*- coding: utf-8 -*-

import argparse

from ModelV1 import *
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
parser.add_argument('-lr', nargs='+', type=float, help='Lista de learning-rates a probar')
parser.add_argument('-emb', nargs='+', type=int, help='Lista de embeddings a probar')
parser.add_argument('-hidd', nargs='+', type=int, help='Lista de hidden a probar')
parser.add_argument('-top', nargs='+', type=int, help='Lista de tops a calcular')

args = parser.parse_args()

model= 2 if args.m == None else args.m
option = 2 if args.i == None else args.i
seed = 100 if args.s == None else args.s
city = "Barcelona" if args.c == None else args.c
gpu = 1 if args.gpu == None else args.gpu
top = [1,5,10,15,20] if args.top == None else args.top
dpout = 1.0 if args.d == None else args.d
lrates = [1e-3] if args.lr == None else args.lr
embsize = [512] if args.emb == None else args.emb
hidden_size = [128] if args.hidd == None else args.hidd

os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)

config = {"min_rest_revs":0,
          "min_usr_revs":5,
          "top_n_new_items": 99,
          "train_pos_rate":1.0,
          "dev_test_pos_rate":0.0,
          "use_rest_provs": False,

          "top": top,
          "emb_size":embsize[0],
          "hidden_size": hidden_size[0],
          "learning_rate": lrates[0],
          "dropout": dpout,  # Prob de mantener

          #Para train
          "epochs":5,
          "batch_size": 512,
          "gs_max_slope":-1e-8}

########################################################################################################################


#DEEP
if (model == 1):

    params = {
        "learning_rate": lrates,
        "emb_size": embsize,
    }
    modelv1 = ModelV1(city=city, option=option, config=config, seed=seed)
    modelv1.gridSearchV1(params, max_epochs=500)
    #modelv1.gridSearchV2(params, max_epochs=500)

#DOTPROD
if (model == 2):

    params = {
        "learning_rate": lrates,
        "emb_size": embsize,
    }

    modelv2 = ModelV2(city=city, option=option, config=config, seed=seed)
    modelv2.gridSearchV1(params, max_epochs=500)


#ToDo: Normalizar capa de intermedia (BatchNormalization)
#ToDo: Probar con un ejemplo único en DOTPROD para verificar el buen funcionamiento

#ToDo: Test emb imagen con hilos == sin hilos
#ToDo: BatchNormalization Layer
#ToDo: AUC binario en TRAIN Y DEV

