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
parser.add_argument('-lr', nargs='+', type=float, help='Lista de learning-rates a probar')
parser.add_argument('-emb', nargs='+', type=int, help='Lista de embeddings a probar')

args = parser.parse_args()

model= 2 if args.m == None else args.m
option = 2 if args.i == None else args.i
seed = 100 if args.s == None else args.s
city = "Barcelona" if args.c == None else args.c
gpu = 1 if args.gpu == None else args.gpu
dpout = 0.0 if args.d == None else args.d
over = '2' if args.over == None else args.over
lrates = [1e-3] if args.lr == None else args.lr
embsize = [128] if args.emb == None else args.emb

os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)

config = {"min_revs":5,
          "min_pos_revs":3,
          "emb_size":32,
          "learning_rate":1e-3,
          "lr_decay":0.0,
          "c_loss": 0,
          "batch_size":512,
          "epochs":5,
          "oversampling":over, #None (desactivar), "auto"(clase 0 y 1 equilibradas) , 2 (duplicar clase 0), 3 (triplicar clase 0) ...
          "dropout":dpout, # Prob de eliminar no de mantener
          "gs_max_slope":-1e-8}

########################################################################################################################

if (model == 2):

    params = {
        "learning_rate": lrates,
    }

    modelv2 = ModelV2(city=city, option=option, config=config, seed=seed)
    modelv2.getDataStats()
    exit()

    modelv2.gridSearchV1(params, max_epochs=500)

if (model == 3):

    params = {
        "learning_rate": lrates,
        "emb_size": embsize,
    }

    modelv3 = ModelV3(city=city, option=option, config=config, seed=seed)
    modelv3.newgetData()
    exit()
    modelv3.gridSearchV1(params, max_epochs=500)


#ToDo: PARA CADA USUARIO DISTRIBUCION EN TRAIN DEV Y TEST
#ToDo: MIRAR QUE PASA EN CADA BATCH
#ToDo: DATOS: MÁS DE 10 y MÁS DE 3 POSITIVAS (SALEN 7067 USUARIOS)

#ToDo: ADAPTAR A TOP-N

#ToDo: Test emb imagen con hilos == sin hilos
#ToDo: BatchNormalization Layer
#ToDo: AUC binario en TRAIN Y DEV

