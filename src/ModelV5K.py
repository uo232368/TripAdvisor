# -*- coding: utf-8 -*-
from src.ModelV5 import *

from keras.preprocessing.image import ImageDataGenerator, Iterator
from keras.models import Sequential, Model
from keras.callbacks import TensorBoard
from keras.optimizers import Adam, RMSprop
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Lambda, GlobalAveragePooling2D
from keras.backend.tensorflow_backend import set_session
from keras.layers.advanced_activations import LeakyReLU

from keras.callbacks import Callback
import threading

import cv2


########################################################################################################################

class ModelV5K(ModelV5):

    class DevCallback(Callback):

        def __init__(self, parent, batches):
            self.PARENT = parent
            self.BATCHES = batches


        def on_epoch_end(self, epoch, logs=None):
            self.PARENT.CURRENT_EPOCH = epoch

            dev_ret, _ = self.PARENT.dev(self.BATCHES)

            # Imprimir linea
            self.PARENT.gridSearchPrint(epoch, logs['loss'], dev_ret)

    class TrainSequence(Sequence):

        def __init__(self, data, batch_size, load_img_method):
            self.BATCHES = np.array_split(data, len(data) // batch_size)
            self.BATCH_SIZE = batch_size
            self.LOAD_IMG = load_img_method

        def __len__(self):
            return len(self.BATCHES)

        def __getitem__(self, idx):
            data_ids = self.BATCHES[idx]

            best_images = self.LOAD_IMG(data_ids.best_path.values)
            worst_images = self.LOAD_IMG(data_ids.worst_path.values)

            return ([np.array(data_ids.id_user.values),
                    np.array(data_ids.id_restaurant.values),
                    np.asarray(best_images),
                    np.array(data_ids.id_rest_worst.values),
                    np.asarray(worst_images)],
                   [np.zeros(len(best_images), dtype=np.int),np.zeros(len(best_images),dtype=np.int)])

    class DevSequence(Sequence):

        def __init__(self, data, batch_size, load_img_method):
            self.BATCHES = np.array_split(data, len(data) // batch_size)
            self.BATCH_SIZE = batch_size
            self.LOAD_IMG = load_img_method

        def __len__(self):
            return len(self.BATCHES)

        def __getitem__(self, idx):
            data_ids = self.BATCHES[idx]

            batch_size = len(data_ids)
            best_images = self.LOAD_IMG(data_ids.best_path.values)

            return ([np.array(data_ids.id_user.values),
                    np.array(data_ids.id_restaurant.values),
                    np.asarray(best_images)])

    def __init__(self,city,option,config,date,seed = 2,modelName= "modelv5"):
        ModelV5.__init__(self,city,option,config,date,seed = seed)

        self.MODEL_NAME = "ModelV5+KERAS"
        print(" "+self.MODEL_NAME)
        print("#"*50)

        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

    def getModel(self):

        def getSiameseModel(input_user, input_rest_best, input_imag_best,input_rest_worst,input_imag_worst, usr_emb_size,rst_emb_size,img_emb_size):

            def custom_loss(best, worst):
                def loss(y_true, y_pred):
                    return tf.reduce_mean(tf.maximum(0.0, (1-(best-worst))))
                return loss

            model_u = Sequential()
            model_u.add(Embedding(self.N_USR, usr_emb_size, input_shape=(1,)))
            model_u.add(Flatten())

            model_r = Sequential()
            model_r.add(Embedding(self.N_USR, rst_emb_size, input_shape=(1,)))
            model_r.add(Flatten())

            model_i = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet',input_tensor=None, input_shape=None, pooling=None, classes=1000)
            for layer in model_i.layers:layer.trainable = False

            x = Dense(img_emb_size * 2)(model_i.get_layer('avg_pool').output)
            x = LeakyReLU(alpha=0.3)(x) #Activation("relu")(x)
            x = Dropout(1 - self.CONFIG["dropout"])(x)
            x = Dense(img_emb_size)(x)
            x = LeakyReLU(alpha=0.3)(x) #Activation("relu")(x)
            model_i = Model(inputs=model_i.input, outputs=x)

            '''
            model_i = keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights="imagenet", input_shape=(img_width, img_height, 3))
            for layer in model_i.layers:layer.trainable = False

            x = GlobalAveragePooling2D()(model_i.output)
            #x = Flatten()(x)
            x = Dense(img_emb_size * 2, activation="relu")(x)
            x = Dropout(1 - self.CONFIG["dropout"])(x)
            x = Dense(img_emb_size, activation="relu")(x)
            model_i = Model(inputs=model_i.input, outputs=x)
            '''
            '''
            # Creamos un modelo secuencial, compuesto por una secuencia lineal de capas
            model_i = Sequential()
            model_i.add(Conv2D(64, (3, 3), padding='same',input_shape=(img_width, img_height, 3)))
            #model_i.add(BatchNormalization())
            model_i.add(Activation('relu'))
            model_i.add(MaxPooling2D((2, 2), padding='same'))
            model_i.add(Conv2D(32, (3, 3), padding='same', input_shape=(img_width, img_height, 3)))
            #model_i.add(BatchNormalization())
            model_i.add(Activation('relu'))
            model_i.add(MaxPooling2D((2, 2), padding='same'))
            model_i.add(Conv2D(16, (3, 3), padding='same', input_shape=(img_width, img_height, 3)))
            #model_i.add(BatchNormalization())
            model_i.add(Activation('relu'))
            model_i.add(MaxPooling2D((2, 2), padding='same'))
            model_i.add(Flatten())
            model_i.add(Dense(img_emb_size))
            #model_i.add(BatchNormalization())
            model_i.add(Activation('relu'))
            '''

            model_main = keras.layers.concatenate([model_u.output, model_r.output,model_i.output])
            model_main = Dropout(1 - self.CONFIG["learning_rate"])(model_main)

            model_main = Dense(emb_size)(model_main)
            #model_main = BatchNormalization()(model_main)
            model_main = Activation('relu')(model_main)
            model_main = Dropout(1 - self.CONFIG["dropout"])(model_main)

            model_main = Dense(emb_size // 2)(model_main)
            #model_main = BatchNormalization()(model_main)
            model_main = Activation('relu')(model_main)
            model_main = Dropout(1 - self.CONFIG["dropout"])(model_main)

            model_main = Dense(emb_size // 4)(model_main)
            #model_main = BatchNormalization()(model_main)
            model_main = Activation('relu')(model_main)
            model_main = Dropout(1 - self.CONFIG["dropout"])(model_main)

            model_main_out = Dense(1)(model_main)

            model = Model(inputs=[model_u.input, model_r.input, model_i.input], outputs=model_main_out)

            best_out = model([input_user,input_rest_best,input_imag_best])
            worst_out = model([input_user,input_rest_worst,input_imag_worst])


            #out_lmbd = Lambda(lambda x: (2 - (x[0] - x[1])))
            #out = out_lmbd([best_out, worst_out])

            model_final = Model(inputs=[input_user,input_rest_best,input_imag_best,input_rest_worst,input_imag_worst], outputs=[best_out, worst_out])

            opt = Adam(lr=self.CONFIG["learning_rate"])
            #opt = RMSprop(lr=self.CONFIG["learning_rate"], epsilon=1.0 )
            model_final.compile(loss=custom_loss(best_out, worst_out), optimizer=opt)

            return model_final

        img_width = self.CONFIG["img_size"]
        img_height = img_width

        emb_size = self.CONFIG['emb_size']
        usr_emb_size = emb_size  # 512
        rst_emb_size = emb_size // 2  # 256
        img_emb_size = emb_size // 2 # 256

        input_user       = Input(shape=(1,), name="input_user")
        input_rest_best  = Input(shape=(1,), name="input_rest_best")
        input_imag_best  = Input(shape=(img_width, img_height, 3,), name="input_imag_best")
        input_rest_worst = Input(shape=(1,), name="input_rest_worst")
        input_imag_worst = Input(shape=(img_width, img_height, 3,), name="input_imag_worst")

        return getSiameseModel(input_user, input_rest_best, input_imag_best,input_rest_worst,input_imag_worst, usr_emb_size,rst_emb_size,img_emb_size)

    def dev(self,dev_sequence):

        def getPos(r):

            r = r.reset_index(drop=True)
            id_r = r.id_restaurant.unique()[0]
            #pos = len(r)-max(r.loc[r.is_dev == 1].index.values)
            pos = min(r.loc[r.is_dev == 1].index.values)+1

            return pos

        dev_pred = []

        pos_model = []
        pcnt_model = []
        pcnt1_model = []

        partial_model = Model(inputs= self.MODEL.get_layer("model_2").get_input_at(0), outputs= self.MODEL.get_layer("model_2").get_output_at(0))
        dev_pred = partial_model.predict_generator(dev_sequence,steps=dev_sequence.__len__(),workers=10)

        dev_img_res = self.DEV.copy()
        dev_img_res["prediction"] = dev_pred

        RET = pd.DataFrame(columns=["id_user","id_restaurant","n_photos","n_photos_dev","model_pos","pcnt_model","pcnt1_model"])

        for i,r in dev_img_res.groupby(["id_user","id_restaurant"]):
            r = r.sort_values("prediction", ascending=False)
            dev = r.loc[r.is_dev==1]

            item_pos = getPos(r)

            pos_model.append(item_pos)
            pcnt_model.append(item_pos/len(r))
            pcnt1_model.append((item_pos-1)/len(r))

            RET = RET.append({"id_user":i[0],"id_restaurant":i[1],"n_photos":len(r),"n_photos_dev":len(dev),"model_pos":pos_model[-1],"pcnt_model":pcnt_model[-1],"pcnt1_model":pcnt1_model[-1]},ignore_index=True)

        #if(self.CURRENT_EPOCH==10):RET.to_excel("docs/"+self.DATE+"/"+self.MODEL_NAME+"_Results("+self.CONFIG["neg_examples"]+").xls")

        return ((pos_model, pcnt_model, pcnt1_model), np.mean(pcnt1_model))

    ####################################################################################################################
    def loadImages(self,batch):
            ret_images = []

            # loop over the input house paths
            for path in batch:
                # load the input image, resize it to be 32 32, and then
                # update the list of input images
                img = cv2.imread(self.IMG_PATH + path)
                r, g, b = cv2.split(img)  # get b,g,r
                img = cv2.merge([b, g, r])

                img = cv2.resize(img, (self.CONFIG["img_size"], self.CONFIG["img_size"]))
                img = img * 1.0
                img /= 255.0
                ret_images.append(img)

            return ret_images

    def gridSearch(self, params, max_epochs = 50, start_n_epochs = 5, last_n_epochs = 5):

        def createCombs():

            def flatten(lst):
                return sum(([x] if not isinstance(x, list) else flatten(x)
                            for x in lst), [])

            combs = []
            level = 0
            for v in params.values():
                if (len(combs)==0):
                    combs = v
                else:
                    combs = list(it.product(combs, v))
                level+=1

                if(level>1):
                    for i in range(len(combs)):
                        combs[i] = flatten(combs[i])

            return pd.DataFrame(combs, columns=params.keys())

        def configNet(comb):
            comb.pop("Index")

            for k in comb.keys():
                assert (k in self.CONFIG.keys())
                self.CONFIG[k]=comb[k]

        #-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#

        combs = createCombs()
        self.printW("Existen "+str(len(combs))+" combinaciones posibles")

        for c in combs.itertuples():

            stop_param = []

            c = dict(c._asdict())

            #Configurar la red
            configNet(c)

            #Crear el modelo
            self.MODEL = self.getModel()

            #Imprimir la configuración
            self.printConfig(filter=c.keys())

            #Configurar y crear sesion
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            keras.backend.set_session(sess)

            self.SESSION = sess

            #Conjuntos de entrenamiento
            #train_batches = np.array_split(self.TRAIN, len(self.TRAIN) // self.CONFIG['batch_size'])
            #dev_batches = np.array_split(self.DEV, len(self.DEV)  // (self.CONFIG['batch_size']*2))

            keras.backend.get_session().run(tf.global_variables_initializer())

            #mycallback = self.DevCallback(self,dev_batches)
            tb_call = TensorBoard(log_dir='logs', batch_size= self.CONFIG['batch_size'], write_graph=True, write_images=True,update_freq='batch')

            train_sequence = self.TrainSequence(self.TRAIN,self.CONFIG['batch_size'], self.loadImages)
            dev_sequence = self.DevSequence(self.DEV, self.CONFIG['batch_size'], self.loadImages)

            for e in range(max_epochs):
                self.CURRENT_EPOCH = e


                train_ret = self.MODEL.fit_generator(train_sequence,
                                                 steps_per_epoch=200,#train_sequence.__len__(),
                                                 epochs=1,
                                                 verbose=1,
                                                 shuffle=False,
                                                 use_multiprocessing=False,
                                                 workers=10,
                                                 max_queue_size=40,
                                                 callbacks=[tb_call])


                data = train_sequence.__getitem__(0)[0]

                img_model = Model(inputs=self.MODEL.get_layer("model_2").get_input_at(0)[-1],outputs=self.MODEL.get_layer("model_2").get_layer("dense_2").output)
                sms_model = Model(inputs=self.MODEL.get_layer("model_2").get_input_at(0),outputs=self.MODEL.get_layer("model_2").get_output_at(0))
                sms_model_concat = Model(inputs=sms_model.get_input_at(0),outputs=sms_model.get_layer("concatenate_1").output)

                pred_c_b = sms_model_concat.predict([data[0], data[1], data[2]])
                pred_c_w = sms_model_concat.predict([data[0], data[3], data[4]])

                '''
                batch_data = train_sequence.__getitem__(0)[0]
                pdctn_b = img_model.predict(batch_data[2])
                pdctn_w = img_model.predict(batch_data[4])
                print(np.linalg.norm(pdctn_b),np.linalg.norm(pdctn_w))
                '''

                dev_ret, _ = self.dev(dev_sequence)
                self.gridSearchPrint(e, train_ret.history['model_2_loss'][0], dev_ret)



            exit()




