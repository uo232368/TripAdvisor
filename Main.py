# -*- coding: utf-8 -*-

import argparse

from ModelV1 import *
from ModelV2 import *
from ModelV3 import *


########################################################################################################################

# Obtener argumentos por linea de comandos
parser = argparse.ArgumentParser()
parser.add_argument('-m', type=int,help="Modelo a utilizar")
parser.add_argument('-d', type=float,help="DropOut")
parser.add_argument('-i', type=int,help="Codificación utilizada para las imágenes")
parser.add_argument('-s', type=int,help="Semilla")
parser.add_argument('-e', type=int,help="Epochs")
parser.add_argument('-c', type=str,help="Ciudad", )
parser.add_argument('-gpu', type=str,help="Gpu")
parser.add_argument('-lr', nargs='+', type=float, help='Lista de learning-rates a probar')
parser.add_argument('-emb', nargs='+', type=int, help='Lista de embeddings a probar')
parser.add_argument('-hidd', nargs='+', type=int, help='Lista de hidden a probar')
parser.add_argument('-hidd2', nargs='+', type=int, help='Lista de hidden2 a probar')
parser.add_argument('-top', nargs='+', type=int, help='Lista de tops a calcular')

parser.add_argument('-imgs', type=int,help="Usar imágenes", )

parser.add_argument('-rst', type=int,help="min_rest_revs")
parser.add_argument('-usr', type=int,help="min_usr_revs")

args = parser.parse_args()

model= 3 if args.m == None else args.m
option = 2 if args.i == None else args.i
epochs= 13 if args.e == None else args.e
seed = 100 if args.s == None else args.s
city = "Barcelona" if args.c == None else args.c
gpu = 1 if args.gpu == None else args.gpu
dpout = .5 if args.d == None else args.d
lrates = [1e-3] if args.lr == None else args.lr
embsize = [512] if args.emb == None else args.emb
hidden_size = [128] if args.hidd == None else args.hidd
hidden2_size = [0] if args.hidd2 == None else args.hidd2

use_images = 0 if args.imgs == None else args.imgs

min_rest_revs = 0 if args.rst == None else args.rst
min_usr_revs = 3 if args.usr == None else args.usr

os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)

config = {"min_rest_revs":min_rest_revs,
          "min_usr_revs":min_usr_revs,
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

          "use_images":use_images,
          "epochs":epochs,
          "batch_size": 512,
          "gs_max_slope":-1e-8}

########################################################################################################################

#DOTPROD
if (model == 3):

    use_images=1
    lrates = [1e-06]
    dpout = [0.5]
    config["hidden_size"] = 1024
    config["hidden2_size"] = 128

    params = {
        "dropout":dpout,
        "use_images": [use_images],
        "learning_rate": lrates
    }

    modelv3 = ModelV3(city=city, option=option, config=config, seed=seed)
    modelv3.gridSearch(params,max_epochs=1000)

if (model == 2):

    params = {
        "dropout":[dpout],
        "use_images": [use_images]
    }

    modelv2 = ModelV2(city=city, option=option, config=config, seed=seed)

    #modelv2.basicModel(test=False, mode="random")
    #modelv2.basicModel(test=False, mode="centroid")

    #modelv2.imageDistance()
    #modelv2.intraRestImage()

    modelv2.gridSearch(params, max_epochs=100)

    #modelv2.finalTrain(epochs = epochs)

    exit()


    '''
    #OBTENER, PARA CADA RESTAURANTE DEL TRAIN, LA MEDIA/MODA/MEDIANA/MAX/MIN DE LA DISTANCIA ENTRE SUS FOTOS
    #NOTA: No se cuentan los que poseen una sola foto (all 0's)

    def myfn1(data):
        rst_imgs = np.unique(np.row_stack(data.vector), axis=0)
        dists = scipy.spatial.distance.pdist(rst_imgs)

        if(len(dists)>0):return pd.Series({'min':np.min(dists),'max':np.max(dists),'mean':np.mean(dists),'median':np.median(dists)})
        #else:return pd.Series({'min':0,'max':0,'mean':0,'median':0,'mode':0})

    rt = modelv2.TRAIN_V2.groupby(['id_restaurant']).apply(myfn1).reset_index()
    rt = rt.dropna(axis=0)
    rt.to_csv("IMGbyRestaurant.csv")
    print(np.mean(rt['min'].values),np.mean(rt['max'].values),np.mean(rt['mean'].values),np.mean(rt['median'].values))
    exit()
    '''



    #OBTENER, DE TODAS LAS FOTOS, LA MEDIA/MODA/MEDIANA/MAX/MIN DE LA DISTANCIA ENTRE ELLAS
    #train_imgs = np.unique(np.row_stack(modelv2.TRAIN_V2.vector), axis=0)
    #np.savetxt("unique_train_imgs.csv", train_imgs, delimiter=";")

    #dists = scipy.spatial.distance.pdist(train_imgs)
    #np.savetxt("unique_train_imgs_dists.csv", dists, delimiter=";")

    dists = np.loadtxt("unique_train_imgs_dists.csv",delimiter=";")

    print({'min':np.min(dists),'max':np.max(dists),'mean':np.mean(dists),'median':np.median(dists),'mode':scipy.stats.mode(dists).mode[0]})

    exit()


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

