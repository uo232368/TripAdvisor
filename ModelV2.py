# -*- coding: utf-8 -*-
from ModelClass import *

########################################################################################################################

class ModelV2(ModelClass):

    def __init__(self,city,option,config,seed = 2):
        modelName = "modelv2"
        ModelClass.__init__(self,city,option,config,seed = seed, name=modelName)

    def getModel(self):

        num_users = self.N_USR
        emb_users = self.CONFIG["emb_size"]

        img_size = self.V_IMG

        num_restaurants = self.N_RST
        emb_restaurants = self.CONFIG["emb_size"]

        in_size = num_users+img_size+num_restaurants

        bin_out = 1

        learning_rate = self.CONFIG["learning_rate"]
        lr_decay = self.CONFIG["lr_decay"]
        batch_size = self.CONFIG["batch_size"]
        epochs = self.CONFIG["epochs"]
        c_loss = self.CONFIG["c_loss"]

        # Model
        # -----------------------------------------------------------------------------------------------------------------------

        first_hidden_layer = 1024

        # first input model
        in_user = Input(shape=(num_users,), name="input_user")

        # second input model
        in_image = Input(shape=(img_size,), name="input_image")

        # second input model
        in_rest = Input(shape=(num_restaurants,), name="input_restaurant")

        # merge input models
        concat = Concatenate()([in_user, in_image,in_rest])
        d1 = Dense(first_hidden_layer,activation='relu')(concat)
        d2 = Dense(first_hidden_layer//2,activation='relu')(d1)
        d3 = Dense(first_hidden_layer//4,activation='relu')(d2)
        d4 = Dense(first_hidden_layer//8,activation='relu')(d3)
        d5 = Dense(first_hidden_layer//16,activation='relu')(d4)
        out = Dense(1,activation='sigmoid', name="out_layer")(d5)

        model = Model(inputs=[in_user, in_image,in_rest], outputs=[out])

        adam = keras.optimizers.Adam(lr=learning_rate, decay=lr_decay)
        #adam = tf.train.AdamOptimizer(learning_rate = learning_rate)

        if (os.path.exists(self.MODEL_PATH)):
            self.printW("Cargando pesos de un modelo anterior...")
            model.load_weights(self.MODEL_PATH)

        #Modelo paralelo
        #model = multi_gpu_model(model, gpus=2)

        # model.compile(optimizer=adam, loss=custom_loss_fn, loss_weights=[(1 - c_loss), c_loss])
        model.compile(optimizer=adam, loss=["binary_crossentropy"])
        #model.compile(optimizer=adam, loss=[f1_loss])

        # summarize layers
        #print(model.summary())
        plot_model(model, to_file=self.MODEL_NAME+'_net.png')

        return model

    def train_v1(self,save=True, show_epoch_info=True):

        verbosity = 2

        # Transformar los datos de TRAIN al formato adecuado
        oh_users = to_categorical(self.TRAIN_V1.id_user, num_classes=self.N_USR)
        y_image = np.zeros((len(self.TRAIN_V1),self.V_IMG))
        oh_rests = to_categorical(self.TRAIN_V1.id_restaurant, num_classes=self.N_RST)

        y_likes = self.TRAIN_V1.like.values

        # y_image = np.row_stack(self.TRAIN_V2.vector.values)

        # Definir un checkpoint para ir almacendando el modelo
        callbacks_list = []

        if (save):
            checkpoint = ModelCheckpoint(self.MODEL_PATH, verbose=0,period=5)
            callbacks_list.append(checkpoint)

        if (show_epoch_info):
            history = LossHistory(self.CONFIG)
            callbacks_list.append(history)
            verbosity = 0

        self.MODEL.fit([oh_users, y_image, oh_rests], [y_likes], epochs=self.CONFIG['epochs'], batch_size=self.CONFIG['batch_size'], callbacks=callbacks_list, verbose=verbosity)

        bin_pred = self.MODEL.predict([oh_users, y_image, oh_rests], verbose=0)

        self.getF1(bin_pred,y_likes,title="TRAIN", verbose=True)

    def gridSearchV1(self, params,max_epochs = 500):

        def fs(val):
            return(str(val).replace(".",","))

        def gsStep(model):
            tss = time.time()
            model.fit([usr_train, img_train, res_train], [out_train], epochs=1, batch_size=self.CONFIG["batch_size"],verbose=0,shuffle=False)

            pred_dev = model.predict([usr_dev, img_dev, res_dev],verbose=0)
            auc = self.getAUC(pred_dev,out_dev)

            return auc, time.time()-tss

        #---------------------------------------------------------------------------------------------------------------
        usr_train = to_categorical(self.TRAIN_V1.id_user, num_classes=self.N_USR)
        img_train = np.zeros((len(self.TRAIN_V1), self.V_IMG))
        res_train = to_categorical(self.TRAIN_V1.id_restaurant, num_classes=self.N_RST)
        out_train = self.TRAIN_V1.like.values
        #---------------------------------------------------------------------------------------------------------------
        usr_dev = to_categorical(self.DEV.id_user, num_classes=self.N_USR)
        img_dev = np.zeros((len(self.DEV), self.V_IMG))
        res_dev = to_categorical(self.DEV.id_restaurant, num_classes=self.N_RST)
        out_dev = self.DEV.like.values

        #---------------------------------------------------------------------------------------------------------------

        start_n_epochs = 5
        last_n_epochs = 5

        combs = []
        dev_hist = []

        for lr in params['learning_rate']:
            combs.append([lr])

        for c in combs:
            lr = c[0];
            ep = 0

            #Reiniciar modelo e historial
            dev_hist.clear()
            self.CONFIG['learning_rate'] = lr
            model = self.getModel()

            for e in range(max_epochs):
                ep +=1

                auc,time_e = gsStep(model)
                dev_hist.append(auc)

                print(fs(ep)+"\t"+fs(lr)+"\t"+fs(auc))

                #Si no se mejora nada de nada en una epoch, fuera.
                if(len(dev_hist)>1 and np.std(dev_hist)==0):break

                #Si en las n epochs anteriores la pendiente supera un minimo, parar
                if(len(dev_hist)==start_n_epochs+last_n_epochs):
                    slope = self.getSlope(dev_hist[-last_n_epochs:]);
                    dev_hist.pop(0)

                    if (slope < self.CONFIG['gs_max_slope']):
                        break

            print("-"*50)

    def dev(self):

        # Transformar los datos de DEV al formato adecuado
        oh_users = to_categorical(self.DEV.id_user, num_classes=self.N_USR)
        y_image = np.zeros((len(self.DEV), self.V_IMG))
        oh_rests = to_categorical(self.DEV.id_restaurant, num_classes=self.N_RST)

        y_likes = self.DEV.like.values


        bin_pred = self.MODEL.predict([oh_users,y_image, oh_rests], verbose=0)

        self.getF1(bin_pred, y_likes,title="DEV")



