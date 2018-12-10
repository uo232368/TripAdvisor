# -*- coding: utf-8 -*-
from ModelClass import *

########################################################################################################################

class ModelV2(ModelClass):

    def __init__(self,city,option,config,seed = 2):

        modelName= "modelv2  (dotprod)"
        ModelClass.__init__(self,city,option,config,seed = seed, name = modelName)

    def getModel(self):

        # Creación del grafo de TF.
        graph = tf.Graph()

        with graph.as_default():

            tf.set_random_seed(self.SEED)

            emb_size = self.CONFIG['emb_size']
            hidden_size = self.CONFIG['hidden_size']
            hidden2_size = self.CONFIG['hidden2_size']

            concat_size = emb_size*2

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

            # Salida ---------------------------------------------------------------------------------------------------------------------


            if(hidden_size==0):
                R0 = tf.Variable(tf.truncated_normal([concat_size, self.V_IMG], mean=0.0, stddev=1.0 / math.sqrt(self.V_IMG)),name="R0")
            elif(hidden2_size==0):
                R0 = tf.Variable(tf.truncated_normal([concat_size, hidden_size], mean=0.0, stddev=1.0 / math.sqrt(hidden_size)),name="R0")
                R1 = tf.Variable(tf.truncated_normal([hidden_size, self.V_IMG], mean=0.0, stddev=1.0 / math.sqrt(self.V_IMG)),name="R1")
            else:
                R0 = tf.Variable(tf.truncated_normal([concat_size, hidden_size], mean=0.0, stddev=1.0 / math.sqrt(hidden_size)), name="R0")
                R1 = tf.Variable(tf.truncated_normal([hidden_size, hidden2_size], mean=0.0, stddev=1.0 / math.sqrt(hidden2_size)),name="R1")
                R2 = tf.Variable(tf.truncated_normal([hidden2_size, self.V_IMG], mean=0.0, stddev=1.0 / math.sqrt(self.V_IMG)),name="R2")


            # Operaciones -----------------------------------------------------------------------------------------------------------------

            user_emb = tf.nn.embedding_lookup(E1, user_rest_input[:,0])
            rest_emb = tf.nn.embedding_lookup(E2, user_rest_input[:,1])

            user_emb = tf.nn.dropout(user_emb,keep_prob=dpout);
            rest_emb = tf.nn.dropout(rest_emb,keep_prob=dpout);

            c1 = tf.concat([user_emb,rest_emb],axis=1, name="concat_r1")

            # Producto escalar
            out_bin = tf.reduce_sum(tf.multiply(user_emb, rest_emb), 1, keepdims=True)

            # multiregresión
            if(hidden_size==0):
                out_img = tf.matmul(c1, R0, name='out_img')
            elif (hidden2_size == 0):
                h_img = tf.matmul(c1, R0, name='h_img')
                h_img = tf.nn.dropout(h_img,keep_prob=dpout);
                h_img = tf.nn.relu(h_img);
                out_img = tf.matmul(h_img, R1, name='out_img')
            else:
                h_img = tf.matmul(c1, R0, name='h_img')
                h_img = tf.nn.dropout(h_img, keep_prob=dpout);
                h_img = tf.nn.relu(h_img);

                h2_img = tf.matmul(h_img, R1, name='h2_img')
                h2_img = tf.nn.dropout(h2_img, keep_prob=dpout);
                h2_img = tf.nn.relu(h2_img);

                out_img = tf.matmul(h2_img, R2, name='out_img')


            # Cálculo de LOSS y optimizador ----------------------------------------------------------------------------------------------

            batch_bin_prob = tf.nn.sigmoid(out_bin, name='batch_bin_prob')
            batch_softplus = tf.nn.softplus((1 - 2 * bin_labels) * out_bin, name='batch_softplus')
            loss_softplus = tf.reduce_mean(batch_softplus, name='loss_softplus')

            batch_rmse =tf.square(tf.subtract(img_labels, out_img), name='batch_rmse')
            loss_rmse = tf.sqrt(tf.reduce_mean(batch_rmse), name='loss_rmse')

            #Regularizar las losses ------------------------------------------------------------------------------------------------------

            if(self.CONFIG['regularization']==2):
                rgl_e1 = tf.nn.l2_loss(E1)
                rgl_e2 = tf.nn.l2_loss(E2)
                rgl_r1 = tf.nn.l2_loss(R1)

                beta = self.CONFIG['regularization_beta']

                loss_softplus = loss_softplus +(beta*rgl_e1)+(beta*rgl_e2)
                loss_rmse = loss_rmse +(beta*rgl_e1) +(beta*rgl_e2) +(beta*rgl_r1)

            elif(self.CONFIG['regularization']!=0):
                self.printE("Regularización no implementada")
                exit()

            # Minimizar la loss
            train_step_bin = tf.train.AdamOptimizer(name='train_step_bin',learning_rate=self.CONFIG['learning_rate']).minimize(loss=loss_softplus, global_step=global_step_bin)
            train_step_img = tf.train.AdamOptimizer(name='train_step_img',learning_rate=self.CONFIG['learning_rate_img']).minimize(loss=loss_rmse, global_step=global_step_img)

            # Crear objeto encargado de almacenar la red
            saver = tf.train.Saver(max_to_keep=1)

        return graph

    def basicModel(self, test=False):

        TRAIN_DATA = self.TRAIN_V2
        filtern = 0

        if(filtern!=0):
            self.printE("UTILIZANDO RESTAURANTES EN TRAIN CON "+str(filtern)+" FOTO")
            def myfn(data): return(pd.Series({"imgs":len(np.unique(np.row_stack(data.vector.values),axis=0))}))
            tmp_data = self.TRAIN_V2.groupby('id_restaurant').apply(myfn).reset_index()
            rst_list = tmp_data.loc[tmp_data.imgs==filtern].id_restaurant.values
            TRAIN_DATA = self.TRAIN_V2.loc[self.TRAIN_V2.id_restaurant.isin(rst_list)]

            print(len(TRAIN_DATA))

        #Para cada restaurante del TRAIN, seleccionar una foto aleatoria como predicción
        def random_prediction(data):
            train_imgs = np.unique(np.row_stack(data.vector), axis=0)
            smple = data.sample(1)
            smple['train_items'] = len(train_imgs)
            return smple

        pred_train = TRAIN_DATA.groupby("id_restaurant").apply(random_prediction)

        if(test==True):eval_name='TEST'; eval_data = self.TEST_V2
        else:eval_name='DEV'; eval_data = self.DEV_V2

        #Obtener la loss con los datos de evaluación
        dev_img_dists = []
        ret = pd.DataFrame(columns=['items_train', 'min_dist'])

        for i, dta in eval_data.groupby(['id_user', 'id_restaurant']):
            usr, rst = i

            pdct = pred_train.loc[pred_train.id_restaurant==rst]

            if(len(pdct)>0):
                n_train_items = pdct.train_items.values[0]
                pdct = np.row_stack(pdct.vector.values)
                other = np.row_stack(dta.vector.values)

                min_d = min(distance_matrix(pdct, other)[0])

                ret = ret.append({'items_train': n_train_items, 'min_dist': min_d}, ignore_index=True)

                dev_img_dists.append(min_d)

        '''
        for t in range(0, 400):
            print(str(t) + "\t" + str(np.mean(ret.loc[ret.items_train >= t].min_dist.values)))
        '''

        self.printG("Modelo básico en "+eval_name+": "+str(np.average(dev_img_dists)))

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

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

            # Imprimir la configuración actual
            #self.printConfig(filter=['min_usr_revs', 'min_rest_revs', 'train_pos_rate', 'emb_size', 'hidden_size', 'learning_rate', 'dropout', 'new_train_examples','use_rest_provs'])
            self.printConfig()

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

                    for batch_d in np.array_split(self.DEV, 20):

                        batch_dtfm = batch_d.copy()
                        batch_dtfm = batch_dtfm[['id_user','id_restaurant','like']]

                        batch_dm = batch_dtfm.values
                        feed_dict = {"user_rest_input:0": batch_dm[:,[0,1]], "bin_labels:0": batch_dm[:,[2]],'dpout:0':1.0}
                        batch_bin_prob,batch_softplus = self.SESSION.run(['batch_bin_prob:0','batch_softplus:0'],feed_dict=feed_dict)

                        batch_dtfm['prediction'] = batch_bin_prob[:,0]
                        dev_loss.extend(batch_softplus[:,0])

                        dev_res = dev_res.append(batch_dtfm,ignore_index=True)

                    hits, avg_pos,median_pos = self.getTopN(dev_res)
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

    def gridSearchV2(self, params,max_epochs = 500):

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
            train_bin_loss_comb = []
            train_img_loss_comb = []

            dev_loss_comb = []
            dev_img_comb = []

            stop_param_comb = []

            start_n_epochs = 5
            last_n_epochs = 5

            #Cambiar los parámetros en la configuración
            self.CONFIG['learning_rate']= c['learning_rate']
            self.CONFIG['emb_size']= c['emb_size']

            #Obtener el modelo
            self.MODEL = self.getModel()

            # Imprimir la configuración actual
            self.printConfig()

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
                    #train_res = pd.DataFrame()

                    train_bin_loss = []
                    train_img_loss = []

                    filtern=0

                    if (filtern != 0):
                        self.printE("UTILIZANDO RESTAURANTES EN TRAIN CON "+str(filtern)+" FOTOS")

                        def myfn(data): return (pd.Series({"imgs": len(np.unique(np.row_stack(data.vector.values), axis=0))}))

                        tmp_data = self.TRAIN_V2.groupby('id_restaurant').apply(myfn).reset_index()
                        rst_list = tmp_data.loc[tmp_data.imgs == filtern].id_restaurant.values
                        self.TRAIN_V2 = self.TRAIN_V2.loc[self.TRAIN_V2.id_restaurant.isin(rst_list)]

                        print(len(self.TRAIN_V2))


                    train_bin_batches = np.array_split(self.TRAIN_V1, len(self.TRAIN_V1) // self.CONFIG['batch_size'])
                    train_img_batches = np.array_split(self.TRAIN_V2, len(train_bin_batches))

                    for bn in range(len(train_bin_batches)):

                        batch_train_bin = train_bin_batches[bn][['id_user', 'id_restaurant', 'like']].values
                        batch_train_img = train_img_batches[bn][['id_user', 'id_restaurant','vector']].values

                        feed_dict_bin = {"user_rest_input:0": batch_train_bin[:, [0, 1]], "bin_labels:0": batch_train_bin[:, [2]],'dpout:0': self.CONFIG['dropout']}
                        feed_dict_img = {"user_rest_input:0": batch_train_img[:, [0, 1]], "img_labels:0": np.row_stack(batch_train_img[:, [2]][:,0]),'dpout:0': self.CONFIG['dropout']}

                        _, batch_softplus = self.SESSION.run(['train_step_bin:0', 'batch_softplus:0'],feed_dict=feed_dict_bin)
                        _, batch_rmse, loss_rmse,out_img = self.SESSION.run(['train_step_img:0', 'batch_rmse:0','loss_rmse:0','out_img:0'],feed_dict=feed_dict_img)

                        train_bin_loss.extend(batch_softplus[:, 0])
                        train_img_loss.extend(np.concatenate( batch_rmse, axis=0 ).tolist())

                        pred_mean_image = np.mean(out_img, 0)
                        dt = np.row_stack(batch_train_img[:, [2]][:,0])

                        if(bn%100==0):
                            dm = distance_matrix(dt, out_img)

                            #print(np.mean(dm))

                            pos = []

                            for r in range(len(dm)):
                                np.argsort(dm[r,:])
                                real_pos = np.where(np.argsort(dm[r, :]) == r)[0][0] + 1
                                pos.append(real_pos)

                            print(np.mean(pos), np.median(pos))

                    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

                    # Probar en DEV (TOP-N) ############################################################################
                    dev_bin_res = pd.DataFrame()
                    dev_loss = []

                    for batch_d in np.array_split(self.DEV, 10):

                        batch_dtfm = batch_d.copy()
                        batch_dtfm = batch_dtfm[['id_user', 'id_restaurant', 'like']]

                        batch_dm = batch_dtfm.values
                        feed_dict = {"user_rest_input:0": batch_dm[:, [0, 1]], "bin_labels:0": batch_dm[:, [2]],'dpout:0': 1.0}
                        batch_bin_prob, batch_softplus = self.SESSION.run(['batch_bin_prob:0', 'batch_softplus:0'], feed_dict=feed_dict)

                        batch_dtfm['prediction'] = batch_bin_prob[:, 0]
                        dev_loss.extend(batch_softplus[:, 0])

                        dev_bin_res = dev_bin_res.append(batch_dtfm, ignore_index=True)

                    hits, avg_pos,median_pos = self.getTopN(dev_bin_res)
                    hits = list(hits.values())

                    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

                    # Probar en DEV (IMG) ##############################################################################

                    dev_img_res = self.DEV_V2
                    dev_img_dists = []
                    dev_img_data = dev_img_res[['id_user', 'id_restaurant','vector']].values

                    real_image_data = np.row_stack(dev_img_data[:, [2]][:, 0])

                    feed_dict_img = {"user_rest_input:0": dev_img_data[:, [0, 1]],"img_labels:0": real_image_data, 'dpout:0': 1.0}
                    predicted_image_data = self.SESSION.run('out_img:0',feed_dict=feed_dict_img)
                    dev_img_res["predicted"] = predicted_image_data.tolist()

                    user_rest_dev_img = dev_img_res.groupby(['id_user', 'id_restaurant'])

                    #ToDo: Probar los que tengan más de x en TRAIN

                    ret = pd.DataFrame(columns=['items_train','min_dist'])

                    for ix, dta in user_rest_dev_img:

                        usr,rst = ix
                        train_dta = self.TRAIN_V2.loc[self.TRAIN_V2.id_restaurant == rst]

                        if(len(train_dta)>0):

                            pdct = np.array(dta.predicted.values[0])
                            train_imgs = np.unique(np.row_stack(train_dta.vector), axis = 0)
                            dis_train_imgs = distance_matrix([pdct], train_imgs)[0]
                            pos = np.argsort(dis_train_imgs)[0]
                            pdct = train_imgs[pos,:]

                            other = np.row_stack(dta.vector.values)
                            min_d = min(distance_matrix([pdct], other)[0])

                            ret = ret.append({'items_train':len(train_imgs),'min_dist':min_d},ignore_index=True)

                            dev_img_dists.append(min_d)

                    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

                    for t in range(0,400):
                        print(str(t) + "\t" + str(np.mean(ret.loc[ret.items_train >= t].min_dist.values)))

                    train_bin_loss_comb.append(np.average(train_bin_loss))
                    train_img_loss_comb.append(np.sqrt(np.average(train_img_loss)))

                    dev_loss_comb.append(np.average(dev_loss))
                    dev_img_comb.append(np.average(dev_img_dists))
                    stop_param_comb.append(avg_pos)

                    log_items = [e,self.CONFIG['emb_size'],self.CONFIG['learning_rate']]
                    log_items.extend(np.round([train_bin_loss_comb[-1],train_img_loss_comb[-1],dev_loss_comb[-1]],decimals=4))

                    log_items.append(bcolors.OKBLUE)
                    log_items.extend(np.round(hits,decimals=4))
                    log_items.append(bcolors.ENDC)

                    log_items.append(bcolors.OKGREEN)
                    log_items.append(np.round(dev_img_comb[-1], decimals=4))
                    log_items.append(bcolors.ENDC)

                    log_items.extend(np.round([avg_pos,median_pos],decimals=4))
                    log_line = "\t".join(map(lambda x:str(x),log_items))
                    log_line = log_line.replace("\t\t","\t")
                    print(log_line)

                    # Si en las n epochs anteriores la pendiente es menor que valor, parar
                    if (len(stop_param_comb) >= start_n_epochs + last_n_epochs):
                        slope = self.getSlope(stop_param_comb[-last_n_epochs:]);
                        if (slope > self.CONFIG['gs_max_slope']):
                            break

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

    def stop(self):

        if(self.SESSION!=None):
            self.printW("Cerrando sesión de tensorflow...")
            self.SESSION.close()

        if(self.MODEL!=None):
            tf.reset_default_graph()
