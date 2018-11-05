# -*- coding: utf-8 -*-

import argparse

from ModelV1 import *


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
parser.add_argument('-top', nargs='+', type=int, help='Lista de tops a calcular')

args = parser.parse_args()

model= 1 if args.m == None else args.m
option = 2 if args.i == None else args.i
seed = 125 if args.s == None else args.s
city = "Barcelona" if args.c == None else args.c
gpu = 1 if args.gpu == None else args.gpu
top = [5,10] if args.top == None else args.top
dpout = 1.0 if args.d == None else args.d
lrates = [1e-6] if args.lr == None else args.lr
embsize = [128] if args.emb == None else args.emb

os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)

config = {"min_revs":5,
          "min_pos_revs":5,

          "top": top,
          "emb_size":embsize[0],
          "learning_rate": lrates[0],
          "dropout": dpout,  # Prob de mantener

          #Para train
          "epochs":5,
          "batch_size": 512,
          "gs_max_slope":1e-8}

########################################################################################################################

if (model == 1):

    params = {
        "learning_rate": lrates,
        "emb_size": embsize,
    }

    modelv1 = ModelV1(city=city, option=option, config=config, seed=seed)
    modelv1.gridSearchV1(params, max_epochs=500)


#ToDo: Normalizar capa de intermedia (BatchNormalization)
#ToDo: Utilizar hits como criterio de parada?
#ToDo: Sigmoide o ReLu?
#ToDo: Entrenar más parecido a DEV y TEST (proporcion 1 vs 100)?

#ToDo: Test emb imagen con hilos == sin hilos
#ToDo: BatchNormalization Layer
#ToDo: AUC binario en TRAIN Y DEV

