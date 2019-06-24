# -*- coding: utf-8 -*-

import argparse

from src.ModelV1 import *
from src.ModelV2 import *
from src.ModelV3 import *
from src.ModelV3_1 import *
from src.ModelV4 import *
from src.ModelV4_1 import *
from src.ModelV4_DEEP import *
from src.ModelV4_1_DEEP import *

from src.ModelV4D import *
from src.ModelV4D2 import *

from src.ModelV5 import *
from src.ModelV5K import *

from src.ModelV6 import *
from src.ModelV60 import *


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

parser.add_argument('-nimg', type=str,help="Negativos para las imágenes (5,5+5...[dentro del rest y fuera])", )
parser.add_argument('-nlike', type=str ,help="Negativos para los likes (n para (rst/usr*100))", )

parser.add_argument('-hidd', nargs='+', type=int, help='Lista de hidden a probar')
parser.add_argument('-hidd2', nargs='+', type=int, help='Lista de hidden2 a probar')
parser.add_argument('-top', nargs='+', type=int, help='Lista de tops a calcular')

parser.add_argument('-use_imgs', type=int,help="Usar imágenes", )
parser.add_argument('-use_like', type=int,help="Usar Like", )
parser.add_argument('-cnn', type=int,help="Usar cnn", )

parser.add_argument('-rst', type=int,help="min_rest_revs")
parser.add_argument('-usr', type=int,help="min_usr_revs")

args = parser.parse_args()

model= 6 if args.m == None else args.m
stage= "grid" if args.stage == None else args.stage

gpu = 1  if args.gpu == None else args.gpu

dpout = [1.0] if args.d == None else args.d
lrates = [1e-2] if args.lr == None else args.lr

lrDecay = "linear_cosine" if args.lrdcay == None else args.lrdcay

nimg = "10+10" if args.nimg == None else args.nimg
nlike = "0" if args.nlike == None else args.nlike

min_revs_usr = 10 # Número mínimo de reviews para el like
min_revs_rst = 5 # Número mínimo de reviews para el like

epochs= 100 if args.e == None else args.e
seed = 100 if args.s == None else args.s
city = "Gijon" if args.c == None else args.c

use_images = 0  if args.use_imgs == None else args.use_imgs
use_like =  1 if args.use_like == None else args.use_like
use_cnn = 0 if args.cnn == None else args.cnn

os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)

config = {"top": [1,2,3,4,5,10,15,20],
          "neg_images":nimg,
          "neg_likes": nlike,

          "min_revs_usr": min_revs_usr,
          "min_revs_rst": min_revs_rst,

          "learning_rate": lrates[0],
          "lr_decay": lrDecay,
          "dropout": dpout[0],  # Prob de mantener

          "use_images":use_images,
          "use_like": use_like,
          "use_cnn":use_cnn,

          "epochs":epochs,
          "batch_size": 512,
          "gs_max_slope":-1e-8,
          "gpu":gpu}

date = "24_04_2019"

########################################################################################################################

if (model == 6):

    #Cargar el modelo adecuado
    if(not config["use_cnn"]):
        modelv6 = ModelV60(city=city, config=config, date=date, seed=seed)
    else:
        exit()

    #modelv6.getDataStats()

    #modelv6.getBaselines()
    #modelv6.getBaselines(test=True)

    #modelv6.getDetailedResults()
    #exit()

    #modelv6.likeBaseline(test=True)

    # Ejecutar la fase pertinente
    if("grid" in stage):

        params = {"learning_rate": lrates, "dropout": dpout}

        if (config["lr_decay"] is None):
            modelv6.gridSearch(params, max_epochs=3000, start_n_epochs=30, last_n_epochs=20)  # 15/10 o 30/20
        else:
            modelv6.gridSearch(params, max_epochs=epochs, start_n_epochs=epochs)  # CON DECAY Y 100 EPOCHS

    if("test" in stage):
        modelv6.finalTrain(epochs=epochs, save=True)


elif (model == 45):

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

    config = {"neg_examples": "10+10",
              "emb_size": 512,
              "learning_rate": 1e-4,
              "lr_decay": "lrDecay",
              "dropout": .8,
              "epochs": 100,
              "batch_size": 512,
              "gs_max_slope": -1e-8}


    modelv4d = ModelV4D2(city=city, option=option, config=config, date=date, seed=seed)
    #modelv4d.getDataStats()

    '''
    import sklearn
    PCA_space = 50
    TSNE_space = 2
    clusters= 100

    IMG = pd.read_pickle(modelv4d.PATH + "img-option" + str(modelv4d.OPTION) + "-new.pkl")
    IMG['review'] = IMG.review.astype(int)
    IMG["id_img"] = IMG.index
    IMG = IMG.sample(1000)
    IMGM = np.row_stack(IMG.vector.values)
    URLS =IMG[["id_img"]].merge(modelv4d.URLS)

    #IMGM = sklearn.decomposition.PCA(n_components=PCA_space).fit_transform(IMGM)
    #IMGM = sklearn.manifold.TSNE(n_components=TSNE_space, verbose=1).fit_transform(IMGM)

    kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=clusters,random_state = 0, verbose=1,batch_size=512, max_no_improvement=200 )
    kmeans = kmeans.fit(IMGM)

    print("-"*50)
    print(clusters, PCA_space,sklearn.metrics.silhouette_score(IMGM, kmeans.labels_, sample_size=len(IMGM)))

    URLS["cluster"] = kmeans.labels_

    path = "tmp_img/clusters/"
    os.makedirs(path,exist_ok=True)

    for idx,c in URLS.groupby("cluster"):
        #if(idx!=2):continue

        path_c = path+str(idx)+"/"
        os.makedirs(path_c, exist_ok=True)

        for _, data in c.iterrows():
            urllib.request.urlretrieve(data.url, path_c + str(data.name)+".jpg")

    exit()
    
    '''

    if("grid" in stage):

        params = {"learning_rate": lrates, "dropout": dpout}

        if (config["lr_decay"] is None):
            modelv4d.gridSearch(params, max_epochs=3000, start_n_epochs=30, last_n_epochs=20)  # 15/10 o 30/20
        else:
            modelv4d.gridSearch(params, max_epochs=epochs, start_n_epochs=epochs)  # CON DECAY Y 100 EPOCHS

        # modelv4d.gridSearch(params, max_epochs=3000, start_n_epochs=30, last_n_epochs=20)#15/10 o 30/20

    if("test" in stage):
        modelv4d.finalTrain(epochs=epochs)

    modelv4d.predict()

    #modelv4d.getBaselines(test=False) # EN VALIDACION
    #modelv4d.getBaselines(test=True)  # EN TEST

    exit()
