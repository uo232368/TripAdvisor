# -*- coding: utf-8 -*-

import argparse

from ModelV1 import *
from ModelV2 import *
from ModelV3 import *
from ModelV3_1 import *
from ModelV4 import *


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

model= 4 if args.m == None else args.m
option = 2 if args.i == None else args.i
epochs= 13 if args.e == None else args.e
seed = 100 if args.s == None else args.s
city = "Barcelona" if args.c == None else args.c
gpu = 1 if args.gpu == None else args.gpu
dpout = 0.5 if args.d == None else args.d
lrates = [1e-4] if args.lr == None else args.lr
embsize = [512] if args.emb == None else args.emb
hidden_size = [128] if args.hidd == None else args.hidd
hidden2_size = [0] if args.hidd2 == None else args.hidd2

use_images = 1 if args.imgs == None else args.imgs

min_rest_revs = 0 if args.rst == None else args.rst
min_usr_revs = 3 if args.usr == None else args.usr

os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)

config = {"top": [1,5,10,15,20],
          "min_usr_rvws":3,
          "neg_examples":5,

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
if (model == 4):

    #ToDo: Revisar la loss y el producto escalar y tal
    #ToDo: Crear conjunto
    #ToDo: Documentar

    params = { "learning_rate": lrates }

    modelv4 = ModelV4(city=city, option=option, config=config, seed=seed)

    def trainFn(data):
        return(pd.Series({"id_restaurant":data.id_restaurant.values[0],"#reviews":len(data.reviewId.unique()),"#comp":len(data)}))
    train = modelv4.TRAIN.groupby("id_restaurant").apply(trainFn).reset_index(drop=True)

    def devFn(data):
        return(pd.Series({"id_restaurant":data.id_restaurant.values[0],"#reviews":len(data.reviewId.unique()),"#comp":len(data)}))

    dev = modelv4.DEV.groupby("id_restaurant").apply(devFn).reset_index(drop=True)

    exit()


    '''

    RVW, IMG, USR_TMP, REST_TMP = modelv4.getFilteredData()


    rst = 3481 #1174
    usr = 1981
    rvw = [126730978]

    RST = RVW.loc[RVW.id_restaurant==rst]
    RST_RVWS = RST.reviewId.values
    RST_IMGS = IMG.loc[IMG.review.isin(RST_RVWS)]
    RST_IMGS = IMG.merge(RST[["reviewId", "images","id_user"]], right_on="reviewId", left_on="review")
    RST_IMGS["images"] = RST_IMGS.apply(lambda x: x.images[x.image - 1]['image_url_lowres'], axis=1)
    RST_IMGS["name"] = RST_IMGS.images.apply(lambda x: hashlib.md5(str(x).encode('utf-8')).hexdigest())
    IMG["first"] = IMG.vector.apply(lambda x: x[0])

    dir = "tmp_img/" + str(rst)
    os.makedirs(dir, exist_ok=True)

    print(RST_IMGS.loc[RST_IMGS.reviewId==rvw,"name"].values)

    for i, d in RST_IMGS.iterrows():
        print(d.images, d['name'], d.vector)
        urllib.request.urlretrieve(d.images, dir + "/" + d['name'] + ".jpg")

    exit()

    '''

    modelv4.gridSearch(params,max_epochs=3000, start_n_epochs=100)


if (model == 31):

    config["hidden_size"] = 1024
    config["hidden2_size"] = 128

    params = {
        "dropout":[dpout],
        "use_images": [use_images],
        "learning_rate": lrates
    }

    modelv31 = ModelV3_1(city=city, option=option, config=config, seed=seed)

    print("-" * 50)
    print(len(modelv31.TRAIN_V3_1), len(modelv31.TRAIN_V3_1.loc[modelv31.TRAIN_V3_1.like == 1]) / len(modelv31.TRAIN_V3_1),len(modelv31.TRAIN_V3_1.loc[modelv31.TRAIN_V3_1.like == 0]) / len(modelv31.TRAIN_V3_1))
    print(len(modelv31.DEV_V3_1), len(modelv31.DEV_V3_1.loc[modelv31.DEV_V3_1.like == 1]) / len(modelv31.DEV_V3_1),len(modelv31.DEV_V3_1.loc[modelv31.DEV_V3_1.like == 0]) / len(modelv31.DEV_V3_1))
    print("-" * 50)

    params = {"dropout": [.5],"use_images": [1],"learning_rate": [1e-3]}
    modelv31.gridSearch(params, max_epochs=3000, last_n_epochs=10)

    config["dropout"] = params["dropout"]
    config["use_images"] = params["use_images"]
    config["learning_rate"] = params["learning_rate"]

    # modelv3.finalTrain_PRUEBA(epochs = 56)

    exit()

    # TRAIN3 + DEV3 ==> 71309 + y 17489 - ===> 80.3% / 19.7%

if (model == 3):

    #use_images=1
    #lrates = [1e-06]
    #dpout = [0.5]

    config["hidden_size"] = 1024
    config["hidden2_size"] = 128

    params = {
        "dropout":[dpout],
        "use_images": [use_images],
        "learning_rate": lrates
    }

    modelv3 = ModelV3(city=city, option=option, config=config, seed=seed)


    '''
    T = modelv3.TRAIN_V3[["vector","like"]]
    T['sum']= T.vector.apply(lambda x : sum(x))
    T_img = T.loc[T['sum'] > 0]
    print(len(T_img),len(T_img.loc[T_img.like==1]),len(T_img.loc[T_img.like==0])) # 76163 58674 17489
    T_noi = T.loc[T['sum']==0]
    print(len(T_noi),len(T_noi.loc[T_noi.ike==1]),len(T_noi.loc[T_noi.like==0])) # 859522 109265 750257
    '''
    '''
    El 8,13% de las reviews tiene imagen (77% + / 22% -)
    El 91,8% de las reviews no tiene imagen (12% + / 87% -)
    '''

    '''
    def maxdist(data):
        dev = np.row_stack(data.vector.values)
        dsts = scipy.spatial.distance.pdist(dev, 'euclidean')
        return pd.Series({"max":np.max(dsts)})

    EMDUYSU = modelv3.DEV_V3.groupby('reviewId').apply(maxdist).reset_index(drop=True)
    EMDUYSU = np.mean(EMDUYSU['max']) #21.317396829939234
    '''

    modelv3.gridSearch(params,max_epochs=3000)

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

