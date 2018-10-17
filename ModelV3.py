# -*- coding: utf-8 -*-
from ModelClass import *

########################################################################################################################

class ModelV3(ModelClass):

    def __init__(self,city,option,config,seed = 2):

        modelName= "modelv3"
        ModelClass.__init__(self,city,option,config,seed = seed, name = modelName)

    def getModel(self):

        def normalized_mse_loss(y_pred, y_real):
            minMSE = self.MSE_DATA[0]
            maxMSE = self.MSE_DATA[1]
            avgMSE = self.MSE_DATA[2]

            return tf.losses.mean_squared_error(y_pred, y_real) / (avgMSE)

        # -------------------------------------------------------------------------------------------------------------------

        num_users = self.N_USR
        emb_users = self.CONFIG["emb_size"]

        num_restaurants = self.N_RST
        emb_restaurants = self.CONFIG["emb_size"]

        bin_out = 1
        img_out = self.V_IMG

        learning_rate = self.CONFIG["learning_rate"]
        lr_decay = self.CONFIG["lr_decay"]
        batch_size = self.CONFIG["batch_size"]
        epochs = self.CONFIG["epochs"]
        c_loss = self.CONFIG["c_loss"]

        # Model
        # -----------------------------------------------------------------------------------------------------------------------

        # first input model
        visible1 = Input(shape=(num_users,), name="input_user")
        d1 = Dense(emb_users, name="emb_user")(visible1)
        d1 = Dropout(self.CONFIG['dropout'])(d1)

        # second input model
        visible2 = Input(shape=(num_restaurants,), name="input_restaurant")
        d2 = Dense(emb_restaurants, name="emb_restaurant")(visible2)
        d2 = Dropout(self.CONFIG['dropout'])(d2)

        # merge input models
        concat = Concatenate()([d1, d2])
        dotprod = Dot(axes=1)([d1, d2])
        dotprod = Activation("sigmoid", name="dotprod")(dotprod)

        output_img = Dense(img_out, name="output_image", activation="relu")(concat)

        model = Model(inputs=[visible1, visible2], outputs=[dotprod, output_img])

        adam = keras.optimizers.Adam(lr=learning_rate, decay=lr_decay)

        if (os.path.exists(self.MODEL_PATH)):
            self.printW("Cargando pesos de un modelo anterior...")
            model.load_weights(self.MODEL_PATH)

        # model.compile(optimizer=adam, loss=custom_loss_fn, loss_weights=[(1 - c_loss), c_loss])
        model.compile(optimizer=adam, loss=["binary_crossentropy", normalized_mse_loss],loss_weights=[(1 - c_loss), c_loss])

        # summarize layers
        #print(model.summary())
        #plot_model(model, to_file=self.MODEL_NAME+'_net.png')

        return model

    def gridSearchV1(self, params,max_epochs = 500):

        def fs(val):
            return(str(val).replace(".",","))

        def gsStep(model):
            tss = time.time()

            model.fit([usr_train, res_train], [out_train, img_train], epochs=1, batch_size=self.CONFIG["batch_size"],verbose=0,shuffle=False)

            pred_train , _ = model.predict([usr_train, res_train],verbose=0)
            train_auc = self.getAUC(pred_train,out_train)
            train_bin_auc = self.getBIN_AUC(pred_train,out_train)

            pred_dev , _ = model.predict([usr_dev, res_dev],verbose=0)
            dev_auc = self.getAUC(pred_dev,out_dev)
            dev_bin_auc = self.getBIN_AUC(pred_dev,out_dev)
            dev_loss = model.evaluate([usr_dev, res_dev], [out_dev, img_dev],verbose=0)
            TP, FP, FN, TN = self.getConfMatrix(pred_dev, out_dev, verbose=False)

            dev_f1 = self.getF1(pred_dev, out_dev, invert=True)
            dev_accuracy = (TP + TN) / sum([TP, FP, FN, TN])

            return train_auc,dev_auc,train_bin_auc,dev_bin_auc,dev_loss[1],dev_f1,dev_accuracy,time.time()-tss,TP, FP, FN, TN

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

        # Generar combinaciones y seleccionar aleatoriamente X
        #---------------------------------------------------------------------------------------------------------------

        start_n_epochs = 5
        last_n_epochs = 5

        combs = []
        dev_hist = []

        for lr in params['learning_rate']:
            for emb in params['emb_size']:
                combs.append([lr,emb])

        #combs = rn.sample(combs,params["tests"])
        #combs.sort(reverse=True)

        #---------------------------------------------------------------------------------------------------------------

        del self.MODEL



        for c in combs:
            lr = c[0];
            emb = c[1]
            ep = 0

            #Reiniciar modelo e historial
            dev_hist.clear()
            self.CONFIG['learning_rate'] = lr
            self.CONFIG['emb_size'] = emb
            model = self.getModel()
            self.MODEL = model

            for e in range(max_epochs):
                ep +=1

                train_auc, dev_auc, train_bin_auc, dev_bin_auc,  dev_loss, dev_f1, dev_accuracy, time_e, TP, FP, FN, TN = gsStep(model)

                dev_hist.append(dev_loss)

                logLn = fs(ep)+"\t"+fs(lr)+"\t"+fs(emb)+"\t"+fs(train_auc)+"\t"+fs(dev_auc)+"\t"+fs(train_bin_auc)+"\t"+fs(dev_bin_auc)
                logLn += "\t"+fs(dev_loss)+"\t"+fs(dev_f1)+"\t"+fs(dev_accuracy)
                logLn += "\t"+fs(TP)+"\t"+fs(FP)+"\t"+fs(FN)+"\t"+fs(TN)

                print(logLn)

                #Si no se mejora nada de nada en una epoch, fuera.
                if(len(dev_hist)>1 and np.std(dev_hist)==0):break

                #Si en las n epochs anteriores la pendiente supera un minimo, parar
                if(len(dev_hist)==start_n_epochs+last_n_epochs):
                    slope = self.getSlope(dev_hist[-last_n_epochs:]);
                    dev_hist.pop(0)

                    if (slope > self.CONFIG['gs_max_slope']):
                        break

            print("-"*50)
            del model

    def train_step1(self, save=True, show_epoch_info=True):

        #self.MODEL_PATH = self.MODEL_PATH+"_step1"

        # Transformar los datos de TRAIN al formato adecuado
        oh_users = to_categorical(self.TRAIN_V1.id_user, num_classes=self.N_USR)
        oh_rests = to_categorical(self.TRAIN_V1.id_restaurant, num_classes=self.N_RST)

        y_likes = self.TRAIN_V1.like.values
        y_image = np.zeros((len(self.TRAIN_V1),self.V_IMG))

        # Definir un checkpoint para ir almacendando el modelo
        callbacks_list = []

        if(save):
            checkpoint = ModelCheckpoint(self.MODEL_PATH, verbose=0)
            callbacks_list.append(checkpoint)

        if(show_epoch_info):
            history = LossHistory(self.CONFIG)
            callbacks_list.append(history)

        self.MODEL.fit([oh_users, oh_rests], [y_likes, y_image], epochs=self.CONFIG['epochs'], batch_size=self.CONFIG['batch_size'],callbacks=callbacks_list, verbose=0)

        bin_pred, img_pred = self.MODEL.predict([oh_users, oh_rests], verbose=0)

        self.getConfMatrix(bin_pred, y_likes)
        auc = self.getAUC(bin_pred,y_likes)
        print("TRAIN AUC-ROC: "+str(auc))

    def dev(self):

        # Transformar los datos de TRAIN al formato adecuado
        oh_users = to_categorical(self.DEV.id_user, num_classes=self.N_USR)
        oh_rests = to_categorical(self.DEV.id_restaurant, num_classes=self.N_RST)

        y_likes = self.DEV.like.values
        y_image = np.row_stack(self.DEV.vector.values)

        bin_pred, img_pred = self.MODEL.predict([oh_users, oh_rests], verbose=0)

        self.getConfMatrix(bin_pred, y_likes, title="DEV")

    def test(self):

        # Transformar los datos de TRAIN al formato adecuado
        oh_users = to_categorical(self.TEST.id_user, num_classes=self.N_USR)
        oh_rests = to_categorical(self.TEST.id_restaurant, num_classes=self.N_RST)

        y_likes = self.TEST.like.values
        y_image = np.row_stack(self.TEST.vector.values)

        bin_pred, img_pred = self.MODEL.predict([oh_users, oh_rests], verbose=0)

        self.getConfMatrix(bin_pred, y_likes, title="TEST")
