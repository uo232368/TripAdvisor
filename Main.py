# -*- coding: utf-8 -*-

import argparse

from ModelV1 import *
from ModelV2 import *
from ModelV3 import *
from ModelV3_1 import *
from ModelV4 import *
from ModelV4_1 import *
from ModelV4_DEEP import *
from ModelV4_1_DEEP import *

from ModelV4D import *
from ModelV4D2 import *

from ModelV5 import *

########################################################################################################################

# Obtener argumentos por linea de comandos
parser = argparse.ArgumentParser()
parser.add_argument('-m', type=int,help="Modelo a utilizar")
parser.add_argument('-stage', type=str,help="grid o test")
parser.add_argument('-d', nargs='+',type=float,help="DropOut")
parser.add_argument('-i', type=int,help="Codificación utilizada para las imágenes")
parser.add_argument('-ims', type=int,help="Tamaño de la imágen")
parser.add_argument('-s', type=int,help="Semilla")
parser.add_argument('-e', type=int,help="Epochs")
parser.add_argument('-c', type=str,help="Ciudad", )
parser.add_argument('-gpu', type=str,help="Gpu")
parser.add_argument('-lr', nargs='+', type=float, help='Lista de learning-rates a probar')
parser.add_argument('-lrdcay', type=str,help="Learning rate decay (None or linear_cosine)")

parser.add_argument('-emb', nargs='+', type=int, help='Lista de embeddings a probar')
parser.add_argument('-pref', type=str,help="Preferencias (5,5+5...[dentro del rest y fuera])", )
parser.add_argument('-hidd', nargs='+', type=int, help='Lista de hidden a probar')
parser.add_argument('-hidd2', nargs='+', type=int, help='Lista de hidden2 a probar')
parser.add_argument('-top', nargs='+', type=int, help='Lista de tops a calcular')

parser.add_argument('-imgs', type=int,help="Usar imágenes", )

parser.add_argument('-rst', type=int,help="min_rest_revs")
parser.add_argument('-usr', type=int,help="min_usr_revs")

args = parser.parse_args()

model= 5 if args.m == None else args.m
stage= "grid" if args.stage == None else args.stage

embsize = [512] if args.emb == None else args.emb
gpu = 1 if args.gpu == None else args.gpu
dpout = [0.8] if args.d == None else args.d
lrates = [1e-5] if args.lr == None else args.lr
lrDecay = None if args.lrdcay == None else args.lrdcay
pref = "10+10" if args.pref == None else args.pref

option = 2 if args.i == None else args.i
ims = 32 if args.ims == None else args.ims

epochs= 200 if args.e == None else args.e
seed = 100 if args.s == None else args.s
city = "Gijon" if args.c == None else args.c
hidden_size = [128] if args.hidd == None else args.hidd
hidden2_size = [0] if args.hidd2 == None else args.hidd2
use_images = 1 if args.imgs == None else args.imgs

os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)

config = {"top": [1,5,10,15,20],
          "neg_examples":pref,

          "emb_size":embsize[0],
          "hidden_size": hidden_size[0],# 0 para eliminar
          "hidden2_size": hidden2_size[0],  # 0 para eliminar

          "learning_rate": lrates[0],
          "lr_decay": lrDecay,
          "dropout": dpout[0],  # Prob de mantener

          "use_images":use_images,
          "img_size": ims,
          "epochs":epochs,
          "batch_size": 512,
          "gs_max_slope":-1e-8}

date = "20_02_2019"

########################################################################################################################

'''
--------------------------------------------------------
EN VALIDACIÓN
--------------------------------------------------------
BARCELONA
PCNT_CNT:	 0.4381273562431728
PCNT-1_CNT:	 0.3906386550350211
PCNT_RND:	 0.3555655000336792
PCNT-1_RND:	 0.30807679882552746

GIJON
PCNT_CNT:	 0.39692190181396025
PCNT-1_CNT:	 0.35734737387132054
PCNT_RND:	 0.32222577581785583
PCNT-1_RND:	 0.28265124787521606

MADRID
PCNT_CNT:	 0.42082363682102336
PCNT-1_CNT:	 0.3824912936740395
PCNT_RND:	 0.3494287533154011
PCNT-1_RND:	 0.31109641016841716
--------------------------------------------------------
EN TEST
--------------------------------------------------------
BARCELONA
PCNT_CNT:	 0.4891844333397853
PCNT-1_CNT:	 0.41992049331855974
PCNT_RND:	 0.37976842960432555
PCNT-1_RND:	 0.3105044895830999

GIJON
PCNT_CNT:	 0.44549465406942046
PCNT-1_CNT:	 0.3921319817197127
PCNT_RND:	 0.34898941105912096
PCNT-1_RND:	 0.29562673870941325

MADRID
PCNT_CNT:	 0.4590914132794897
PCNT-1_CNT:	 0.40182278008324473
PCNT_RND:	 0.37093191476729886
PCNT-1_RND:	 0.3136632815710538

'''

if (model == 5):

    params = {"learning_rate": lrates, "dropout":dpout}

    config["batch_size"] = 256

    modelv5 = ModelV5(city=city, option=option, config=config, date=date, seed=seed)
    modelv5.gridSearch(params, max_epochs=epochs, start_n_epochs=epochs, last_n_epochs=epochs)#15/10 o 30/20


if (model == 45):
    # ToDo: Documentar

    modelv4d = ModelV4D2(city=city, option=option, config=config, date=date, seed=seed)

    if("grid" in stage):

        params = {"learning_rate": lrates, "dropout": dpout}

        if (config["lr_decay"] is None):
            modelv4d.gridSearch(params, max_epochs=3000, start_n_epochs=30, last_n_epochs=20)  # 15/10 o 30/20
        else:
            modelv4d.gridSearch(params, max_epochs=epochs, start_n_epochs=epochs)  # CON DECAY Y 100 EPOCHS

        # modelv4d.gridSearch(params, max_epochs=3000, start_n_epochs=30, last_n_epochs=20)#15/10 o 30/20

    if("test" in stage):
        modelv4d.finalTrain(epochs=epochs)

    #modelv4d.predict()

    #modelv4d.getBaselines(test=False) # EN VALIDACION
    #modelv4d.getBaselines(test=True)  # EN TEST

    exit()

if (model == 44):

    params = {"learning_rate": lrates, "dropout":dpout}

    modelv4d = ModelV4D(city=city, option=option, config=config, date=date, seed=seed)
    print(" Negativos: " + config["neg_examples"])
    print("#" * 50)

    modelv4d.gridSearch(params, max_epochs=3000, start_n_epochs=100, last_n_epochs=20)#15/10 o 30/20

if (model == 43):

    params = {"learning_rate": lrates}

    modelv43 = ModelV4_1_DEEP(city=city, option=option, config=config, date=date, seed=seed)
    print(" Negativos: " + config["neg_examples"])
    print("#" * 50)

    modelv43.gridSearch(params, max_epochs=3000, start_n_epochs=50)

if (model == 42):

    params = {"learning_rate": lrates}

    modelv42 = ModelV4_DEEP(city=city, option=option, config=config, date=date, seed=seed)
    print(" Negativos: " + config["neg_examples"])
    print("#" * 50)

    modelv42.gridSearch(params, max_epochs=3000, start_n_epochs=50)

if (model == 41):

    params = {"learning_rate": lrates}

    modelv41 = ModelV4_1(city=city, option=option, config=config, date=date, seed=seed)
    print(" Negativos: " + config["neg_examples"])
    print("#" * 50)

    modelv41.gridSearch(params, max_epochs=3000, start_n_epochs=50)

if (model == 4):

    params = { "learning_rate": lrates }

    modelv4 = ModelV4(city=city, option=option, config=config, date=date,seed=seed)
    print(" Negativos: "+config["neg_examples"])
    print("#" * 50)


    #modelv4.getBaselines()

    modelv4.gridSearch(params,max_epochs=3000, start_n_epochs=50)


