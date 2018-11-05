# -*- coding: utf-8 -*-
from ModelClass import *

########################################################################################################################

class ModelV1(ModelClass):

    def __init__(self,city,option,config,seed = 2):

        modelName= "modelv1"
        ModelClass.__init__(self,city,option,config,seed = seed, name = modelName)

    def getModel(self):

        # Creación del grafo de TF.
        graph = tf.Graph()

        with graph.as_default():

            tf.set_random_seed(self.SEED)

            concat_size = self.N_USR + self.N_RST
            hidden_size = self.CONFIG['emb_size']

            # Número global de iteraciones
            global_step_bin = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step_bin')
            global_step_img = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step_img')

            dpout = tf.Variable(1.0, dtype=tf.float32, trainable=False, name='dpout')

            # Datos de entrada -----------------------------------------------------------------------------------------------------------
            # Array del tamaño del batch con las X
            user_rest_input = tf.placeholder(tf.int32, shape=[None, 2], name="user_rest_input")

            # Capas de salida -----------------------------------------------------------------------------------------------------------

            bin_labels = tf.placeholder(tf.float32, shape=[None, 1], name="bin_labels")
            img_labels = tf.placeholder(tf.float32, shape=[None, self.V_IMG], name="img_labels")

            # Embeddings -----------------------------------------------------------------------------------------------------------------

            # Matriz T1 que transforma la concatenación de la entrada a un espacio de menor dimension
            T1 = tf.Variable(tf.truncated_normal([concat_size, hidden_size], mean=0.0, stddev=1.0 / math.sqrt(hidden_size)),name="T1")

            B1 = tf.Variable(tf.truncated_normal([hidden_size, 1], mean=0.0, stddev=1.0 / math.sqrt(1)),name="B1")
            R1 = tf.Variable(tf.truncated_normal([hidden_size, self.V_IMG], mean=0.0, stddev=1.0 / math.sqrt(self.V_IMG)),name="R1")

            # Operaciones -----------------------------------------------------------------------------------------------------------------

            #Obtener las columnas correspondientes a cada usuario y restaurante para poseteriormente
            user_h1 = tf.nn.embedding_lookup(T1, user_rest_input[:,0])
            rest_h1 = tf.nn.embedding_lookup(T1, user_rest_input[:,1]+self.N_USR) # <------ Muy importante

            h1 = tf.add(user_h1,rest_h1) # Se suman ambos para simular una multiplicación tradicional

            #h1_mean, h1_var = tf.nn.moments(h1, axes=[0,1]) # axes=[0] para hacerlo por columnas
            #h1 = tf.nn.batch_normalization(h1, mean=h1_mean, variance=h1_var, offset=None, scale=None, variance_epsilon=1e-9)

            h1 = tf.nn.sigmoid(h1)
            h1 = tf.nn.dropout(h1, keep_prob=dpout)

            # Transformar a 32 documento
            out_bin = tf.matmul(h1, B1)

            # Transformar a 32 cancion
            out_img = tf.matmul(h1, R1)

            # Cálculo de LOSS y optimizador- ---------------------------------------------------------------------------------------------

            # Obtener las losses

            batch_bin_prob = tf.nn.sigmoid(out_bin, name='batch_bin_prob')

            batch_softplus = tf.nn.softplus((1 - 2 * bin_labels) * out_bin, name='batch_softplus')
            loss_softplus = tf.reduce_mean(batch_softplus, name='loss_softplus')

            loss_rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(img_labels, out_img))), name='loss_rmse')

            # Minimizar la loss
            train_step_bin = tf.train.AdamOptimizer(name='train_step_bin',learning_rate=self.CONFIG['learning_rate']).minimize(loss=loss_softplus, global_step=global_step_bin)
            train_step_img = tf.train.AdamOptimizer(name='train_step_img',learning_rate=self.CONFIG['learning_rate']).minimize(loss=loss_rmse, global_step=global_step_img)

            # Crear objeto encargado de almacenar la red
            saver = tf.train.Saver(max_to_keep=1)

        return graph

    def gridSearchV1(self, params,max_epochs = 500):

        # Generar combinaciones y seleccionar aleatoriamente X
        # ---------------------------------------------------------------------------------------------------------------

        start_n_epochs = 5
        last_n_epochs = 5

        combs = []

        for lr in params['learning_rate']:
            for emb in params['emb_size']:
                combs.append({'learning_rate':lr, 'emb_size':emb})


        # Para cada combinación...
        # ---------------------------------------------------------------------------------------------------------------

        for c in combs:

            #Variables para almacenar los resultados de la ejecución
            train_loss_comb = []
            dev_loss_comb = []
            stop_param_comb = []

            start_n_epochs = 5
            last_n_epochs = 5

            #Cambiar los parámetros en la configuración
            self.CONFIG['learning_rate']= c['learning_rate']
            self.CONFIG['emb_size']= c['emb_size']

            #Obtener el modelo
            self.MODEL = self.getModel()

            print("-"*70)
            print(self.CONFIG)

            #Configurar y crear sesion
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            with tf.Session(graph=self.MODEL, config=config) as session:

                self.SESSION = session

                # Inicializar variables
                init = tf.global_variables_initializer()
                init.run(session=self.SESSION)

                for e in range(max_epochs):

                    # Entrenar el modelo ###################################################################################
                    train_loss = []

                    for batch in np.array_split(self.TRAIN_V1, len(self.TRAIN_V1)//self.CONFIG['batch_size']):

                        batch_tm = batch[['id_user','id_restaurant','like']].values

                        feed_dict = {"user_rest_input:0": batch_tm[:,[0,1]], "bin_labels:0": batch_tm[:,[2]],'dpout:0':self.CONFIG['dropout']}

                        _, batch_softplus = self.SESSION.run(['train_step_bin:0','batch_softplus:0'],feed_dict=feed_dict)

                        train_loss.extend(batch_softplus[:,0])

                    # Probar en DEV ########################################################################################
                    dev_res = pd.DataFrame()
                    dev_loss = []

                    for batch_d in np.array_split(self.DEV, 2):

                        batch_dtfm = batch_d.copy()
                        batch_dtfm = batch_dtfm[['id_user','id_restaurant','like']]

                        batch_dm = batch_dtfm.values
                        feed_dict = {"user_rest_input:0": batch_dm[:,[0,1]], "bin_labels:0": batch_dm[:,[2]],'dpout:0':1.0}
                        batch_bin_prob,batch_softplus = self.SESSION.run(['batch_bin_prob:0','batch_softplus:0'],feed_dict=feed_dict)

                        batch_dtfm['prediction'] = batch_bin_prob[:,0]
                        dev_loss.extend(batch_softplus[:,0])

                        dev_res = dev_res.append(batch_dtfm,ignore_index=True)

                    hits = self.getTopN(dev_res)
                    hits = list(hits.values())

                    train_loss_comb.append(np.average(train_loss))
                    dev_loss_comb.append(np.average(dev_loss))
                    stop_param_comb.append(hits[0])

                    log_items = [e,self.CONFIG['emb_size'],self.CONFIG['learning_rate'],train_loss_comb[-1],dev_loss_comb[-1]]
                    log_items.extend(hits)
                    log_line = "\t".join(map(lambda x:str(x),log_items))
                    print(log_line)

                    # Si en las n epochs anteriores la pendiente es menor que valor, parar
                    if (len(stop_param_comb) >= start_n_epochs + last_n_epochs):
                        slope = self.getSlope(stop_param_comb[-last_n_epochs:]);
                        if (slope < self.CONFIG['gs_max_slope']):
                            break



    def stop(self):

        if(self.SESSION!=None):
            self.printW("Cerrando sesión de tensorflow...")
            self.SESSION.close()

        if(self.MODEL!=None):
            tf.reset_default_graph()

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

        self.MODEL.fit([oh_users, oh_rests], [y_likes, y_image], epochs=self.CONFIG['epochs'], batch_size=self.CONFIG['batch_size'],callbacks=callbacks_list, verbose=2)

        bin_pred, img_pred = self.MODEL.predict([oh_users, oh_rests], verbose=0)

        TP, FP, FN, TN = self.getConfMatrix(bin_pred, y_likes)
        print(TP, FP, FN, TN)

    def dev(self, model):

        if(model==None):model = self.MODEL

        usr_dev = to_categorical(self.DEV.id_user, num_classes=self.N_USR)
        res_dev = to_categorical(self.DEV.id_restaurant, num_classes=self.N_RST)

        pred_dev, _ = model.predict([usr_dev, res_dev], verbose=0, batch_size=128)

        return pred_dev
