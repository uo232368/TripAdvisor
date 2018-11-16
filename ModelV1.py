# -*- coding: utf-8 -*-
from ModelClass import *

########################################################################################################################

class ModelV1(ModelClass):

    def __init__(self,city,option,config,seed = 2):

        modelName= "modelv1 (deep)"
        ModelClass.__init__(self,city,option,config,seed = seed, name = modelName)

    def getModel(self):

        # Creación del grafo de TF.
        graph = tf.Graph()

        with graph.as_default():

            tf.set_random_seed(self.SEED)
            tf.set_random_seed(self.SEED)

            emb_size = self.CONFIG['emb_size']
            concat_size = emb_size*2

            #hidden0_size = emb_size
            hidden_size = self.CONFIG['hidden_size']

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

            E1 = tf.Variable(tf.truncated_normal([self.N_USR, emb_size], mean=0.0, stddev=1.0 / math.sqrt(emb_size)),name="E1")
            E2 = tf.Variable(tf.truncated_normal([self.N_RST, emb_size], mean=0.0, stddev=1.0 / math.sqrt(emb_size)),name="E2")

            # HIDDEN ---------------------------------------------------------------------------------------------------------------------
            T1 = tf.Variable(tf.truncated_normal([concat_size, hidden_size], mean=0.0, stddev=1.0 / math.sqrt(hidden_size)),name="T1")
            b1 = tf.Variable(tf.zeros([hidden_size]), name="b1")

            #T2 = tf.Variable(tf.truncated_normal([hidden0_size, hidden_size], mean=0.0, stddev=1.0 / math.sqrt(hidden_size)),name="T2")
            #b2 = tf.Variable(tf.zeros([hidden_size]), name="b2")

            # Salida ---------------------------------------------------------------------------------------------------------------------

            B1 = tf.Variable(tf.truncated_normal([hidden_size, 1], mean=0.0, stddev=1.0 / math.sqrt(1)),name="B1")
            R1 = tf.Variable(tf.truncated_normal([hidden_size, self.V_IMG], mean=0.0, stddev=1.0 / math.sqrt(self.V_IMG)),name="R1")

            # Operaciones -----------------------------------------------------------------------------------------------------------------

            user_emb = tf.nn.embedding_lookup(E1, user_rest_input[:,0])
            user_emb = tf.nn.dropout(user_emb, keep_prob=dpout)

            rest_emb = tf.nn.embedding_lookup(E1, user_rest_input[:,1])
            rest_emb = tf.nn.dropout(rest_emb, keep_prob=dpout)

            h0 = tf.concat([user_emb,rest_emb],axis=1, name="concat_h0")
            h0 = tf.matmul(h0,T1, name="matmul_h0") +  b1
            h0 = tf.nn.dropout(h0, keep_prob=dpout)
            h0 = tf.nn.relu(h0)

            #h1 = tf.matmul(h0,T2, name="matmul_h1") +  b2

            #h1_mean, h1_var = tf.nn.moments(h1, axes=[0])
            #h1_scale = tf.Variable(tf.ones([hidden_size]))
            #h1_beta = tf.Variable(tf.zeros([hidden_size]))
            #h1 = tf.nn.batch_normalization(h1, mean=h1_mean, variance=h1_var, offset=h1_beta, scale=h1_scale, variance_epsilon=1e-5)

            #h1 = tf.nn.dropout(h1, keep_prob=dpout)
            #h1 = tf.nn.relu(h1)

            out_bin = tf.matmul(h0, B1)

            # multiregresión
            out_img = tf.matmul(h0, R1)

            # Cálculo de LOSS y optimizador- ---------------------------------------------------------------------------------------------

            # Obtener las losses

            batch_bin_prob = tf.nn.sigmoid(out_bin, name='batch_bin_prob')

            batch_softplus = tf.nn.softplus((1 - 2 * bin_labels) * out_bin, name='batch_softplus')
            loss_softplus = tf.reduce_mean(batch_softplus, name='loss_softplus')

            batch_rmse =tf.square(tf.subtract(img_labels, out_img), name='batch_rmse')
            loss_rmse = tf.sqrt(tf.reduce_mean(batch_rmse), name='loss_rmse')

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

            #Imprimir la configuración actual
            self.printConfig(filter=['min_usr_revs','min_rest_revs','train_pos_rate','emb_size','hidden_size','learning_rate','dropout'])

            #Configurar y crear sesion
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            with tf.Session(graph=self.MODEL, config=config) as session:

                self.SESSION = session

                tf.set_random_seed(self.SEED)

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


                    for batch_d in np.array_split(self.DEV, 40):

                        batch_dtfm = batch_d.copy()
                        batch_dtfm = batch_dtfm[['id_user','id_restaurant','like']]

                        batch_dm = batch_dtfm.values
                        feed_dict = {"user_rest_input:0": batch_dm[:,[0,1]], "bin_labels:0": batch_dm[:,[2]],'dpout:0':1.0}
                        batch_bin_prob,batch_softplus = self.SESSION.run(['batch_bin_prob:0','batch_softplus:0'],feed_dict=feed_dict)

                        batch_dtfm['prediction'] = batch_bin_prob[:,0]
                        dev_loss.extend(batch_softplus[:,0])

                        dev_res = dev_res.append(batch_dtfm,ignore_index=True)

                    hits,avg_pos,median_pos = self.getTopN(dev_res)
                    hits = list(hits.values())

                    train_loss_comb.append(np.average(train_loss))
                    dev_loss_comb.append(np.average(dev_loss))
                    stop_param_comb.append(avg_pos)

                    log_items = [e,self.CONFIG['emb_size'],self.CONFIG['learning_rate']]
                    log_items.extend(np.round([train_loss_comb[-1],dev_loss_comb[-1]],decimals=4))
                    log_items.append(bcolors.OKBLUE)
                    log_items.extend(np.round(hits,decimals=4))
                    log_items.append(bcolors.ENDC)
                    log_items.extend(np.round([avg_pos,median_pos],decimals=4))
                    log_line = "\t".join(map(lambda x:str(x),log_items))
                    print(log_line)


                    # Si en las n epochs anteriores la pendiente es menor que valor, parar
                    if (len(stop_param_comb) >= start_n_epochs + last_n_epochs):
                        slope = self.getSlope(stop_param_comb[-last_n_epochs:]);
                        if (slope > self.CONFIG['gs_max_slope']):
                            break

    def gridSearchV2(self, params, max_epochs=500):

        # Generar combinaciones y seleccionar aleatoriamente X
        # ---------------------------------------------------------------------------------------------------------------

        start_n_epochs = 5
        last_n_epochs = 5

        combs = []

        for lr in params['learning_rate']:
            for emb in params['emb_size']:
                combs.append({'learning_rate': lr, 'emb_size': emb})

        # Para cada combinación...
        # ---------------------------------------------------------------------------------------------------------------

        for c in combs:

            # Variables para almacenar los resultados de la ejecución
            train_loss_comb = []
            dev_loss_comb = []
            stop_param_comb = []

            start_n_epochs = 5
            last_n_epochs = 5

            # Cambiar los parámetros en la configuración
            self.CONFIG['learning_rate'] = c['learning_rate']
            self.CONFIG['emb_size'] = c['emb_size']

            # Obtener el modelo
            self.MODEL = self.getModel()

            print("-" * 70)
            print(self.CONFIG)

            # Configurar y crear sesion
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            with tf.Session(graph=self.MODEL, config=config) as session:

                self.SESSION = session

                # Inicializar variables
                init = tf.global_variables_initializer()
                init.run(session=self.SESSION)

                for e in range(max_epochs):

                    # Entrenar el modelo ###################################################################################
                    train_bin_loss = []
                    train_img_loss = []

                    train_bin_batches = np.array_split(self.TRAIN_V1, len(self.TRAIN_V1) // self.CONFIG['batch_size'])
                    train_img_batches = np.array_split(self.TRAIN_V2, len(train_bin_batches))

                    for bn in range(len(train_bin_batches)):

                        batch_train_bin = train_bin_batches[bn][['id_user', 'id_restaurant', 'like']].values
                        batch_train_img = train_img_batches[bn][['id_user', 'id_restaurant','vector']].values

                        feed_dict_bin = {"user_rest_input:0": batch_train_bin[:, [0, 1]], "bin_labels:0": batch_train_bin[:, [2]],'dpout:0': self.CONFIG['dropout']}
                        feed_dict_img = {"user_rest_input:0": batch_train_img[:, [0, 1]], "img_labels:0": np.row_stack(batch_train_img[:, [2]][:,0]),'dpout:0': self.CONFIG['dropout']}

                        _, batch_softplus = self.SESSION.run(['train_step_bin:0', 'batch_softplus:0'],feed_dict=feed_dict_bin)
                        _, batch_rmse, loss_rmse = self.SESSION.run(['train_step_img:0', 'batch_rmse:0','loss_rmse:0'],feed_dict=feed_dict_img)

                        train_bin_loss.extend(batch_softplus[:, 0])
                        train_img_loss.extend(np.concatenate( batch_rmse, axis=0 ).tolist())


                    # Probar en DEV ########################################################################################
                    dev_res = pd.DataFrame()
                    dev_loss = []

                    for batch_d in np.array_split(self.DEV, 40):
                        batch_dtfm = batch_d.copy()
                        batch_dtfm = batch_dtfm[['id_user', 'id_restaurant', 'like']]

                        batch_dm = batch_dtfm.values
                        feed_dict = {"user_rest_input:0": batch_dm[:, [0, 1]], "bin_labels:0": batch_dm[:, [2]], 'dpout:0': 1.0}
                        batch_bin_prob, batch_softplus = self.SESSION.run(['batch_bin_prob:0', 'batch_softplus:0'], feed_dict=feed_dict)

                        batch_dtfm['prediction'] = batch_bin_prob[:, 0]
                        dev_loss.extend(batch_softplus[:, 0])

                        dev_res = dev_res.append(batch_dtfm, ignore_index=True)

                    hits, avg_pos, median_pos = self.getTopN(dev_res)
                    hits = list(hits.values())

                    train_loss_comb.append(np.average(train_bin_loss))
                    dev_loss_comb.append(np.average(dev_loss))
                    stop_param_comb.append(avg_pos)

                    log_items = [e, self.CONFIG['emb_size'], self.CONFIG['learning_rate'], train_loss_comb[-1],np.sqrt(np.mean(train_img_loss)), dev_loss_comb[-1]]
                    log_items.extend(hits)
                    log_items.extend([avg_pos, median_pos])
                    log_line = "\t".join(map(lambda x: str(x), log_items))
                    print(log_line)

                    # Si en las n epochs anteriores la pendiente es menor que valor, parar
                    if (len(stop_param_comb) >= start_n_epochs + last_n_epochs):
                        slope = self.getSlope(stop_param_comb[-last_n_epochs:]);
                        if (slope > self.CONFIG['gs_max_slope']):
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
