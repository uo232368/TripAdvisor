# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import requests
from io import BytesIO
from scipy.spatial import distance
import math

def getFilteredData(PATH):

    IMG_CNN  = pd.read_pickle(PATH + "img-option2-new.pkl")
    IMG_FOOD = pd.read_pickle(PATH + "img-food.pkl")

    RVW = pd.read_pickle(PATH + "reviews.pkl")

    IMG_CNN['review'] = IMG_CNN.review.astype(int)
    IMG_FOOD['review'] = IMG_FOOD.review.astype(int)
    RVW["reviewId"] = RVW.reviewId.astype(int)

    RVW["num_images"] = RVW.images.apply(lambda x: len(x))
    RVW = RVW.loc[(RVW.num_images > 0)]  # Eliminar reviews sin imagen
    RVW["like"] = RVW.rating.apply(lambda x: 1 if x > 30 else 0)
    RVW = RVW.loc[(RVW.userId != "")]

    # Obtener ID para ONE-HOT de usuarios y restaurantes
    # ---------------------------------------------------------------------------------------------------------------

    USR_TMP = pd.DataFrame(columns=["real_id", "id_user"])
    REST_TMP = pd.DataFrame(columns=["real_id", "id_restaurant"])

    # Obtener tabla real_id -> id para usuarios
    USR_TMP.real_id = RVW.sort_values("userId").userId.unique()
    USR_TMP.id_user = range(0, len(USR_TMP))

    # Obtener tabla real_id -> id para restaurantes
    REST_TMP.real_id = RVW.sort_values("restaurantId").restaurantId.unique()
    REST_TMP.id_restaurant = range(0, len(REST_TMP))

    # Mezclar datos
    RET = RVW.merge(USR_TMP, left_on='userId', right_on='real_id', how='inner')
    RET = RET.merge(REST_TMP, left_on='restaurantId', right_on='real_id', how='inner')

    RVW = RET[['date', 'images', 'language', 'rating', 'restaurantId', 'reviewId', 'text', 'title', 'url', 'userId',
               'num_images', 'id_user', 'id_restaurant', 'like']]

    IMG_CNN["id_img"]  = IMG_CNN.index
    IMG_FOOD["id_img"] = IMG_FOOD.index

    URLS = RVW[["reviewId", "images"]].merge(IMG_CNN, left_on="reviewId", right_on="review")
    URLS["url"] = URLS.apply(lambda x: x.images[x.image - 1]['image_url_lowres'], axis=1)
    URLS = URLS[["id_img","url"]]

    RVW = RVW.drop(columns=['restaurantId', 'userId', 'url', 'text', 'title', 'date', 'rating', 'language', 'like'])


    RVW = RVW.merge(IMG_CNN[["review", "id_img"]], left_on="reviewId", right_on="review")

    # RVW["url"] = RVW.apply(lambda x: x.images[x.image - 1]['image_url_lowres'], axis=1)
    # RVW["name"] = RVW.url.apply(lambda x: hashlib.md5(str(x).encode('utf-8')).hexdigest())
    RVW = RVW.drop(columns=['images', 'review', 'reviewId', 'num_images'])

    IMG_CNN  = np.row_stack(IMG_CNN.vector.values)
    IMG_FOOD = np.row_stack(IMG_FOOD.vector.values)

    return RVW, IMG_CNN, IMG_FOOD ,URLS, USR_TMP, REST_TMP

def readURL(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def getClosest(IMG,DATA,RST_IMGS,URLS, max_it=5):

    if(len(DATA)<=1): return

    ITEM    = DATA[IMG]
    ITEM_ID = RST_IMGS[IMG]

    DATA = np.delete(DATA, (IMG), axis=0)
    RST_IMGS = np.delete(RST_IMGS, (IMG), axis=0)

    DSTS = distance.cdist(DATA,[ITEM]).T
    DST_OR = np.sort(DSTS)[0]
    ORD = RST_IMGS[np.argsort(DSTS)][0]

    cols = min(max_it+1, len(ORD))
    for i in range(cols):

        if(i==0):
            url = URLS.loc[URLS.id_img == ITEM_ID].url.values[0]
            img = readURL(url)

            plt.subplot(1, cols, 1)
            plt.imshow(img)
            plt.axis('off')

        else:

            url = URLS.loc[URLS.id_img == ORD[i-1]].url.values[0]
            img = readURL(url)

            plt.subplot(1,cols,i+1)
            plt.imshow(img)
            plt.title("d="+str(np.round(DST_OR[i-1],decimals=2)))
            plt.axis('off')

    # Show the graph
    plt.show()

    return


########################################################################################################################

PATH = "/media/HDD/pperez/TripAdvisor/gijon_data/"

RVW, IMG_CNN, IMG_FOOD, URLS, USR_TMP, REST_TMP  = getFilteredData(PATH)

RST_IMGS = RVW.loc[RVW.id_restaurant==51].id_img.values

print("Restaurante con "+str(len(RST_IMGS))+" imágenes")

CNN_VEC  = IMG_CNN[RST_IMGS]
FOOD_VEC = IMG_FOOD[RST_IMGS]

CONCAT_VEC = np.concatenate((CNN_VEC, FOOD_VEC), axis = 1)

image = np.random.randint(0, len(CNN_VEC))  # >Seleccionar una imagen

getClosest(image,CNN_VEC,RST_IMGS,URLS)
getClosest(image,FOOD_VEC,RST_IMGS,URLS)

getClosest(image,CONCAT_VEC,RST_IMGS,URLS)


exit()

