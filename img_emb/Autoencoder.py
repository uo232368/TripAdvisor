# -*- coding: utf-8 -*-

import keras
import tensorflow as tf
from keras.models import load_model
from keras.datasets import cifar10
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
import matplotlib.pyplot as plt

import os
import pickle
import numpy as np
import pandas as pd
import scipy
import PIL

########################################################################################################################

def getNet(model_file,img_width,img_height, gpu=0,lr = 1e-3):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))


    if (os.path.exists(model_file)):
        print("Loading old model...")
        model = load_model(model_file)
        encoder_model = load_model(saveDir + "e_" + name)
        decoder_model = load_model(saveDir + "d_" + name)
        K.set_value(model.optimizer.lr, lr)

        return encoder_model, decoder_model, model
    
    input_img = Input(shape=(img_width,img_height, 3))
    x = Conv2D(8, (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    encoded = MaxPooling2D((2, 2), padding='same', name="embeddigs")(x)

    input_decoder = Input(shape=encoded.get_shape().as_list()[1:])
    x = Conv2D(64, (3, 3), padding='same')(input_decoder)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Activation('sigmoid')(x)

    encoder_model = Model(input_img, encoded)
    decoder_model = Model(input_decoder, decoded)

    model = Model(encoder_model.input, decoder_model(encoded))
    model.compile(optimizer=Adam(lr=lr, decay=1e-6), loss='binary_crossentropy')


    #model.summary()
    #exit()

    return encoder_model, decoder_model,model

def getNet2(model_file, img_width, img_height,gpu=0, lr=1e-3):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    if (os.path.exists(model_file)):
        print("Loading old model...")
        model = load_model(model_file)
        K.set_value(model.optimizer.lr, lr)

        return model

    input_img = Input(shape=(img_width, img_height, 3))
    x = Conv2D(64, (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    encoded = MaxPooling2D((2, 2), padding='same', name="embeddigs")(x)

    x = Conv2D(8, (3, 3), padding='same')(encoded)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Activation('sigmoid')(x)

    model = Model(input_img, decoded)
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy')

    return model

def generateData():

    ret  = pd.DataFrame(columns=["path"])

    for root, dirs, files in os.walk(img_path):
        if(len(files)>0):
            tmp = list(map(lambda x : root+"/"+x,files))
            ret = ret.append(pd.DataFrame({"path":tmp}).reset_index(drop=True), ignore_index=True)

    ret.to_pickle("img_list")

def loadData():

    train_data = pd.read_pickle("img_list")

    datagen = ImageDataGenerator(rescale=1. / 255.,
                                 validation_split=0.05)
                                 #rotation_range=90,
                                 #width_shift_range=0.1,
                                 #height_shift_range=0.1,
                                 #zoom_range=0.2)

    #datagen = ImageDataGenerator(rescale=1. / 255., validation_split=0.25)

    train_generator = datagen.flow_from_dataframe(
        dataframe=train_data,
        directory=None,
        x_col="path",
        subset="training",
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
        class_mode="input",
        target_size=(img_width, img_height))

    valid_generator = datagen.flow_from_dataframe(
        dataframe=train_data,
        directory=None,
        x_col="path",
        subset="validation",
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
        class_mode="input",
        target_size=(img_width, img_height))

    return train_generator, valid_generator, None

def showOrigDec(orig, dec, num=10, padding=0):
    n = num
    plt.figure(figsize=(16, 4))

    orig.reset()
    orig.batch_size = num+padding
    orig_items = orig.next()[0]

    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(orig_items[padding+i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i +1 + n)
        plt.imshow(dec[padding+i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

def plotDist(ret,show = 5):

    # Obtener distancias
    c = ret.loc[ret.chosen == 1, "vector"].values[0]
    v = np.row_stack(ret.vector.values)
    d = scipy.spatial.distance.cdist(v, [c]).reshape(-1, )
    ret["dist"] = d

    # Ordenar e imprimir
    ret = ret.sort_values("dist")

    plt.figure(figsize=(14, 1))
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=.99, wspace=0, hspace=0)


    ret["show"] = 0
    ret.iloc[:show, ret.columns.get_loc("show")] = 1
    ret.iloc[-show:, ret.columns.get_loc("show")] = 1

    fg_n = 0
    for i, r in ret.iterrows():
        print(i, r["name"], r["dist"])

        if (r["show"]):
            ax = plt.subplot( 1,show * 2, fg_n + 1)
            plt.imshow(np.array(PIL.Image.open(dir + "/" + r["name"])))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            fg_n += 1

    plt.show()

def embeddingTests():

    inp_imgs = train_generator.next()[0]
    enc_imgs = encoder_model.predict(inp_imgs)

    #enc_imgs += np.random.random_sample(enc_imgs.shape)*0.5
    # noise = np.flip(enc_imgs)


    #mn = np.mean(enc_imgs, axis=0)*2.5

    '''
    mn = np.zeros((16, 16, 64))
    mn[:, [7,8], :] = 1
    mn[[7,8], :, :] = 1

    for s in range(len(enc_imgs)):
        enc_imgs[s] += mn

    '''
    dec_imgs = decoder_model.predict(enc_imgs)

    plt.figure(figsize=(16, 4))

    n = 10
    padding= int(np.random.random(1)*(len(inp_imgs)-n))

    for i in range(n):

        # display original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(inp_imgs[i+padding])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display emb
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(np.mean(enc_imgs[i+padding],axis=2))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + (2 * n))
        plt.imshow(dec_imgs[i+padding])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

    return 0

########################################################################################################################

#https://github.com/shibuiwilliam/Keras_Autoencoder/blob/master/Cifar_Conv_AutoEncoder.ipynb

seed = 45
batch_size = 128
learning_rate = 1e-3

img_width = 256
img_height = 256

epochs = 0
saveDir = ".temp/models/"

model_n = 1
step="train"
gpu=1

name = "AutoEncoder.hdf5"
if(model_n==2):name = "AutoEncoder2.hdf5"

chkpt = saveDir + name

img_path = "/media/HDD/pperez/TripAdvisor/gijon_data/images_lowres"
if not os.path.isdir(saveDir): os.makedirs(saveDir)

#generateData()
train_generator,valid_generator,_ = loadData()

if("AutoEncoder2" in chkpt): model = getNet2(chkpt,img_width,img_height, gpu=gpu,lr=learning_rate)
else: encoder_model, decoder_model, model = getNet(chkpt,img_width,img_height,gpu=gpu, lr=learning_rate)

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size


if("train" in step):

    #es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
    #cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    if(epochs>0):
        history = model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        #callbacks=[es_cb, cp_cb],
                        verbose=1,
                        epochs=epochs)

        encoder_model.save(saveDir+"e_"+name)
        decoder_model.save(saveDir+"d_"+name)
        model.save(saveDir+name)

    embeddingTests()

    exit()

elif("test" in step):
    STEP_SIZE_VALID=valid_generator.n//512

    dev = model.predict_generator(valid_generator, steps=STEP_SIZE_VALID, verbose=1)

    items = 10
    max_pad = len(dev)-items

    showOrigDec(valid_generator, dev, num=items, padding=np.random.randint(0,max_pad))

    #showOrigDec(valid_generator, dev_emb, num=items, padding=np.random.randint(0,max_pad))

else:

    if (model_n==1):model_emb = Model(inputs=model.input, outputs=model.get_layer("max_pooling2d_3").output)
    elif(model_n==2): model_emb = Model(inputs=model.input, outputs=model.get_layer("embeddigs").output)

    datagen = ImageDataGenerator(rescale=1. / 255.)

    dir = "test_images"
    rest = "goiko"
    compare = "66b8585fd266ab1d65ce6fd7389784ff"

    rest_generator = datagen.flow_from_directory(
        directory=dir,
        batch_size=len(os.listdir(dir+"/"+rest)),
        class_mode=None,
        seed=seed,#NO QUITAR
        shuffle=False,#NO QUITAR
        target_size=(img_width, img_height))

    dev_emb = model_emb.predict_generator(rest_generator, steps=1, verbose=1)

    ret = pd.DataFrame(columns=["name","vector"])

    #Obtener vectores
    for i,d in enumerate(rest_generator.filenames):
        v = dev_emb[i].reshape(-1,)
        name = d #d.split("/")[1]
        chosen = 0

        if(compare in name): chosen=1

        ret = ret.append({"name":name,"vector":v,"chosen":chosen}, ignore_index=True)

    #ret.to_csv("v2")
    #exit()

    plotDist(ret, show=5)



