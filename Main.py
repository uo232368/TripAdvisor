# -*- coding: utf-8 -*-

import argparse

from ModelV1 import *
from ModelV2 import *
from GridSearch import *


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
parser.add_argument('-hidd2', nargs='+', type=int, help='Lista de hidden2 a probar')
parser.add_argument('-top', nargs='+', type=int, help='Lista de tops a calcular')

args = parser.parse_args()

model= 2 if args.m == None else args.m
option = 2 if args.i == None else args.i
seed = 100 if args.s == None else args.s
city = "Barcelona" if args.c == None else args.c
gpu = 0 if args.gpu == None else args.gpu
dpout = 1.0 if args.d == None else args.d
lrates = [1e-3] if args.lr == None else args.lr
embsize = [512] if args.emb == None else args.emb
hidden_size = [128] if args.hidd == None else args.hidd
hidden2_size = [256] if args.hidd2 == None else args.hidd2

os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)

config = {"min_rest_revs":0,
          "min_usr_revs":3,
          "new_train_examples":-2, # -1 para auto, -2 para (rest/user)*100 y 0 para ninguno

          "regularization":0, # 0 para desactivar
          "regularization_beta": 2.5e-6,
          "use_rest_provs": False,
          "top_n_new_items": 99,
          "train_pos_rate":1.0,
          "dev_test_pos_rate":0.0,
          "top": [1,5,10,15,20],

          "emb_size":embsize[0],
          "hidden_size": hidden_size[0],# 0 para eliminar
          "hidden2_size": hidden2_size[0],  # 0 para eliminar

          "learning_rate": lrates[0],
          "learning_rate_img": lrates[0],
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
    modelv2.basicModel(test=False)

    #modelv2.gridSearchV1(params, max_epochs=500)
    #modelv2.gridSearchV2(params, max_epochs=500)

    #TODO: VER ERROR UTILIZANDO SOLO RESTAURANTES DE 0 FOTOS; 1 FOTO... y comparar con los mismo en el modelo básico
    #TODO: Ejecutar prueba que falta

    '''
    rsts = modelv2.DEV_V2.id_restaurant.unique()
    lens = []

    for rst in rsts:
        dta = modelv2.TRAIN_V2.loc[(modelv2.TRAIN_V2.id_restaurant==rst)]
        if(len(dta)>0): dta = np.unique(np.row_stack(dta.vector.values),axis=0)
        lens.append(len(dta))

    lens = np.array(lens)

    for i in range(0,200):
        print(str(i)+"\t"+str(len(np.where(lens>=i)[0]))+"\t"+str(len(np.where(lens==i)[0])))
    
    '''


#ToDo: Test emb imagen con hilos == sin hilos

