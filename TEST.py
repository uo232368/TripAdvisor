import numpy as np
import pandas as pd
from ModelV2 import *

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

          "emb_size":512,
          "hidden_size": 128,# 0 para eliminar
          "hidden2_size": 0,  # 0 para eliminar

          "learning_rate":  1e-3,
          "learning_rate_img": 1e-3,
          "dropout": 1,  # Prob de mantener

          "use_images": 0,
          "epochs":1,
          "batch_size": 512,
          "gs_max_slope":-1e-8}

params = {
    "learning_rate": [1e-3],
    "emb_size": [512],
    "hidden_size": [128],
    "use_images":[0,1]
}

modelv2 = ModelV2(city="Gijon", option=2, config=config, seed=100)
modelv2.gridSearch(params, max_epochs=50)
modelv2.finalTrain(epochs)

exit()

'''
def sigmoid(x, derivative=False):
  return x*(1-x) if derivative else 1/(1+np.exp(-x))

b1 = np.genfromtxt("b1.csv",  delimiter="\t")
E1 = np.genfromtxt("E1.csv",  delimiter="\t")
E2 = np.genfromtxt("E2.csv",  delimiter="\t")
T1 = np.genfromtxt("T1.csv",  delimiter="\t")
B1_1 = np.genfromtxt("B1_1.csv",  delimiter="\t")


USR = E1[0,:]

RST_1 = E2[866,:]
RST_2 = E2[1689,:]

USR_RST_1 = np.concatenate((USR, RST_1), axis=0)
USR_RST_2 = np.concatenate((USR, RST_2), axis=0)

HIDDEN_1 = np.matmul(USR_RST_1,T1)
HIDDEN_2 = np.matmul(USR_RST_2,T1)

HIDDEN_1 = HIDDEN_1 + b1
HIDDEN_2 = HIDDEN_2 + b1

HIDDEN_1[HIDDEN_1<0]=0
HIDDEN_2[HIDDEN_2<0]=0

OUT_1 = np.matmul(HIDDEN_1,B1_1)
OUT_2 = np.matmul(HIDDEN_2,B1_1)


print(OUT_1, sigmoid(OUT_1))
#print(OUT_2)
'''


pred_img = pd.read_csv("pred_img.csv", index_col=0).as_matrix()
real_img = pd.read_csv("real_img.csv", index_col=0).as_matrix()

pred_mean = np.mean(pred_img,0)
count = 0
for i in range(512):

    dist_mean = np.linalg.norm(real_img[i, :] - pred_mean)
    dist_pred = np.linalg.norm(real_img[i, :] - pred_img[i, :])

    print(dist_pred,dist_mean)

    if(dist_pred>dist_mean):count+=1

print(count)